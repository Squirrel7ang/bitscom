import os
import argparse
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import bitscom
from psgd.models.llama.llama_nn import LlamaConfig, MyLlamaForCausalLM


class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length=2048, text_field="text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_field]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length + 1,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        if len(tokens) < 2:
            tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]

        if len(tokens) > self.seq_length + 1:
            tokens = tokens[: self.seq_length + 1]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (
                self.seq_length + 1 - len(tokens)
            )

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def get_dataloader(
    *,
    dp_size: int,
    dp_rank: int,
    dataset_name: str,
    dataset_config: str,
    tokenizer_name: str,
    seq_length: int,
    batch_size: int,
    num_workers: int,
    split: str,
    use_auth_token: bool,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=False,
            use_auth_token=use_auth_token,
        )
    except OSError:
        print("[warn] tokenizer unavailable; using llama-tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            use_fast=False,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "c4":
        dataset_config = dataset_config or "en"
        dataset = load_dataset("allenai/c4", dataset_config, split=split, streaming=False)
        text_field = "text"
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        text_field = "text"

    tokenized_dataset = TokenizedDataset(
        dataset,
        tokenizer,
        seq_length=seq_length,
        text_field=text_field,
    )

    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, tokenizer


def partition_llama_model(config: LlamaConfig, stage_idx: int, num_stages: int):
    with torch.device("meta"):
        model = MyLlamaForCausalLM(config)
        if dist.is_initialized() and dist.get_rank() == 0:
            total_params = sum(p.numel() for p in model.parameters()) / 1e9
            print(f"[rank 0] model params: {total_params:.2f}B")

    num_layers = config.num_hidden_layers
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    start_layer = stage_idx * layers_per_stage + min(stage_idx, remainder)
    end_layer = start_layer + layers_per_stage + (1 if stage_idx < remainder else 0)

    for i in list(model.model.layers.keys()):
        if not (start_layer <= int(i) < end_layer):
            del model.model.layers[i]

    if len(model.model.layers) == 0:
        import torch.nn as nn

        model.model.layers = nn.ModuleDict({"dummy": nn.Identity()})

    if stage_idx == 0:
        model.lm_head = None
        model.model.final_norm = None
    elif stage_idx == num_stages - 1:
        model.model.embed_tokens = None
    else:
        model.model.embed_tokens = None
        model.model.final_norm = None
        model.lm_head = None

    assigned_layers = [int(i) for i in model.model.layers.keys()]
    print(f"[partition] stage {stage_idx}: layers {assigned_layers}")
    return model


def _loss_fn(output, target):
    shift_logits = output[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=0,
    )


def _sync_grads_lowbit(
    *,
    lowbit_group: bitscom.LowBitGroup,
    dp_size: int,
    module: torch.nn.Module,
):
    for param in module.parameters():
        if param.grad is None:
            continue
        lowbit_group.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
        param.grad.div_(dp_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--use_auth_token", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./llama7b_checkpoints")
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--micro_batches", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--simulate_quantization", action="store_true")
    parser.add_argument("--stochastic_rounding", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    if args.backend == "lowbit":
        bitscom.init(bitwidth=args.bitwidth)

    dist.init_process_group(backend=args.backend, init_method="env://")
    world_size = dist.get_world_size()
    pp_size = args.pp_size

    if world_size % pp_size != 0:
        raise RuntimeError(
            f"world_size {world_size} must be divisible by pp_size {pp_size}"
        )

    dp_size = world_size // pp_size
    device_mesh = init_device_mesh("cuda", (dp_size, pp_size), mesh_dim_names=("dp", "pp"))
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        rope_theta=10000.0,
        pad_token_id=0,
        tie_word_embeddings=True,
    )

    stage_idx = pp_mesh.get_local_rank()
    stage_model = partition_llama_model(config, stage_idx, pp_size)
    stage_model.to_empty(device=device, recurse=True)
    stage_model.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )

    stage = PipelineStage(
        stage_model,
        stage_index=stage_idx,
        num_stages=pp_size,
        device=device,
        group=pp_mesh.get_group(),
    )

    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=args.lr)

    dp_rank = dist.get_rank() // pp_size
    dataloader, _ = get_dataloader(
        dp_size=dp_size,
        dp_rank=dp_rank,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=2,
        split="train",
        use_auth_token=args.use_auth_token,
    )

    schedule = ScheduleGPipe(stage, n_microbatches=args.micro_batches, loss_fn=_loss_fn)

    lowbit_group = bitscom.LowBitGroup(
        bitwidth=args.bitwidth,
        process_group=dp_mesh.get_group(),
        simulate_quantization=args.simulate_quantization,
        stochastic_rounding=args.stochastic_rounding,
    )

    global_step = 0
    epoch_iter = range(args.epochs)
    for epoch in epoch_iter:
        if stage.is_last:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        else:
            pbar = dataloader

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device) if stage.is_last else None
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if stage.is_first:
                schedule.step(input_ids, attention_mask=attention_mask)
            elif stage.is_last:
                losses = []
                schedule.step(target=labels, losses=losses, attention_mask=attention_mask)
                loss = torch.stack(losses).mean()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            else:
                schedule.step(attention_mask=attention_mask)

            _sync_grads_lowbit(
                lowbit_group=lowbit_group,
                dp_size=dp_size,
                module=stage.submod,
            )
            optimizer.step()

            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
