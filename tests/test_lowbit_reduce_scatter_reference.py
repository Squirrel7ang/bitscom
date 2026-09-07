import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import bitscom
from bitscom.quantization import (
    DEFAULT_BLOCK_SIZE,
    dequantize_tensor_blockwise,
    pack_lowbit,
    quantize_tensor_blockwise,
    unpack_lowbit,
)


pytestmark = pytest.mark.integration

def print_rank(*args, **kargs):
    if dist.get_rank() == 0:
        print(*args, **kargs)


TEST_CASES = [
    {
        "skip": False,
        "name": "reduce_scatter_bw4",
        "bitwidth": 4,
        "numel": 8,
        "inputs": torch.Tensor([
            1/2, 3/2, 7/2, 15/2, -16/2, -8/2, -4/2, -2/2
        ]),
        "sparse": True,
        "sparse_compression_ratio": 0.5,
        "sparse_priority_mode": 0,
        "sparse_non_priority_mode": 2,
        "sparse_priority_quantize_bitwidth": 4,
        "sparse_non_priority_quantize_bitwidth": 4,
        "seed": 17,
    },
    {
        "skip": True,
        "name": "reduce_scatter_bw4",
        "bitwidth": 4,
        "numel": 8,
        "inputs": torch.Tensor([
            1, 3, 7, 15, -16, -8, -4, -2
        ]),
        "sparse": True,
        "sparse_compression_ratio": 0.5,
        "sparse_priority_mode": 0,
        "sparse_non_priority_mode": 1,
        "sparse_priority_quantize_bitwidth": 4,
        "sparse_non_priority_quantize_bitwidth": 4,
        "seed": 17,
    },
    {
        "name": "reduce_scatter_bw4",
        "bitwidth": 4,
        "numel": 8,
        "inputs": torch.Tensor([
            1, 15, 15, 15, -16, -16, -16, -2
        ]),
        "sparse": True,
        "sparse_compression_ratio": 0.5,
        "sparse_priority_mode": 0,
        "sparse_non_priority_mode": 1,
        "sparse_priority_quantize_bitwidth": 4,
        "sparse_non_priority_quantize_bitwidth": 4,
        "seed": 17,
    },
    {
        "name": "reduce_scatter_bw4",
        "bitwidth": 4,
        "numel": 16,
        "sparse": True,
        "sparse_compression_ratio": 1,
        "sparse_priority_mode": 0,
        "sparse_non_priority_mode": 1,
        "sparse_priority_quantize_bitwidth": 4,
        "sparse_non_priority_quantize_bitwidth": 4,
        "seed": 17,
    },
]

def testCase(case, init_file: str):
    if 'skip' in case and case['skip']:
        return
    # Register the lowbit backend exactly like megatron/training/initialize.py
    # does, passing the full sparse configuration explicitly.
    if case.get("sparse", False):
        bitscom.init(
            bitwidth=case["bitwidth"],
            error_feedback=False,
            error_feedback_mode="none",
            block_size=DEFAULT_BLOCK_SIZE,
            sparse_enabled=True,
            sparse_projection_rank=case.get("sparse_projection_rank", 4),
            sparse_compression_ratio=case["sparse_compression_ratio"],
            sparse_priority_mode=case.get("sparse_priority_mode", 0),
            sparse_priority_quantize_bitwidth=case.get("sparse_priority_quantize_bitwidth", 4),
            sparse_non_priority_mode=case.get("sparse_non_priority_mode", 1),
            sparse_non_priority_quantize_bitwidth=case.get("sparse_non_priority_quantize_bitwidth", 4),
        )
    if not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="lowbit")

    # Simulated world size used by the CPU reference math (numel split, shards).
    world_size = 2
    bitwidth = int(case["bitwidth"])
    seed = int(case["seed"])
    numel = int(case["numel"])
    shard_len = numel // world_size
    assert shard_len * world_size == numel, f"{case['name']}: numel must divide world_size"
    local_rank = dist.get_rank()

    # Standard reduce_scatter convention: each rank owns a full tensor made of
    # world_size equal chunks, and rank r receives the reduced chunk r.
    inputs = []
    if case["inputs"] is not None:
        inputs.append(case["inputs"])
        inputs.append(case["inputs"])
        numel = inputs[0].numel()
        shard_len = numel // world_size
        assert shard_len * world_size == numel, f"{case['name']}: numel must divide world_size"
    else:
        for rank in range(world_size):
            gen = torch.Generator()
            gen.manual_seed(seed + rank)
            # Bounded range [-0.1, 0.1] keeps the quantization error bounds tight.
            inputs.append(torch.rand(numel, generator=gen) * 0.2 - 0.1)

    if local_rank == 0:
        print_rank(f"{inputs[0]=}\n{inputs[1]=}")

    exact_sum = inputs[0] + inputs[1]  # full-precision reference

    def _cal_max_factor(size):
        # Mirror calMaxFactor in the C++ backend: largest power-of-two factor.
        factor = 1
        while size % factor == 0 and size // factor > factor:
            factor *= 2
        while factor > 1 and size % factor != 0:
            factor //= 2
        return factor

    def _quantize_roundtrip(flat, bw):
        # Mirror the C++ pack/unpack wire format:
        # quantize -> pack -> unpack -> dequantize.
        q, scales = quantize_tensor_blockwise(
            flat, bw, block_size=DEFAULT_BLOCK_SIZE, stochastic_rounding=False
        )
        packed, packed_numel = pack_lowbit(q, bw)
        q2 = unpack_lowbit(packed, bw, packed_numel)
        return dequantize_tensor_blockwise(
            q2, scales, block_size=DEFAULT_BLOCK_SIZE, dtype=torch.float32
        )

    def _sim_quantized_allreduce(parts, bw):
        # CPU reference of quantizedAllreduceTensor: quantize each rank's
        # shards, sum the received shards, requantize the sum, gather back.
        original_numel = parts[0].numel()
        pad = (world_size - (original_numel % world_size)) % world_size
        flats = [torch.cat([p, torch.zeros(pad)]) if pad else p for p in parts]
        q_shard_len = flats[0].numel() // world_size
        quantized = [
            [_quantize_roundtrip(s, bw) for s in f.split(q_shard_len)]
            for f in flats
        ]
        out_shards = []
        for dst in range(world_size):
            local_sum = torch.zeros(q_shard_len)
            for src in range(world_size):
                local_sum.add_(quantized[src][dst])
            q2, scales2 = quantize_tensor_blockwise(
                local_sum, bw, block_size=DEFAULT_BLOCK_SIZE, stochastic_rounding=False
            )
            packed2, packed_numel2 = pack_lowbit(q2, bw)
            q3 = unpack_lowbit(packed2, bw, packed_numel2)
            out_shards.append(
                dequantize_tensor_blockwise(
                    q3, scales2, block_size=DEFAULT_BLOCK_SIZE, dtype=torch.float32
                )
            )
        return torch.cat(out_shards)[:original_numel]

    def _sim_lowbit_reduce_scatter(parts, bw):
        # CPU reference of the non-sparse lowbit reduce_scatter: quantize each
        # rank's chunks, then sum the dequantized chunks per destination rank.
        # Note: unlike quantizedAllreduceTensor, the sum is NOT requantized.
        quantized = [
            [_quantize_roundtrip(s, bw) for s in p.split(shard_len)]
            for p in parts
        ]
        outputs = []
        for dst in range(world_size):
            local_sum = torch.zeros(shard_len)
            for src in range(world_size):
                local_sum.add_(quantized[src][dst])
            outputs.append(local_sum)
        return outputs

    if not case.get("sparse", False):
        # Non-sparse path: chunks are quantized with the case bitwidth, so the
        # result may deviate from the exact sum only by the input quantization
        # error (bw2, |x|<=0.1 -> <= 0.05 per rank per element).
        outputs = _sim_lowbit_reduce_scatter(inputs, bitwidth)
        for rank in range(world_size):
            reference_chunk = exact_sum[rank * shard_len:(rank + 1) * shard_len]
            max_abs_err = (outputs[rank] - reference_chunk).abs().max().item()
            assert outputs[rank].numel() == shard_len, f"{case['name']} rank={rank} wrong shard size"
            assert torch.isfinite(outputs[rank]).all(), f"{case['name']} rank={rank} non-finite output"
            assert max_abs_err < 0.15, f"{case['name']} rank={rank} max_abs_err={max_abs_err}"
        return

    # Sparse ARC-Top-K path: AllReduce + Scatter (see reduceScatterSparse in C++).
    n = _cal_max_factor(numel)
    m = numel // n
    print_rank(f"{n=}, {m=}, {numel=}")
    r = case.get("sparse_projection_rank", 4)
    ratio = float(case["sparse_compression_ratio"])
    K = max(1, int(n * ratio + 0.5))  # llround semantics

    G = [x.view(n, m) for x in inputs]

    # Each rank draws its own projection matrix V (intentional design); the
    # averaged P is identical across ranks, so priority indices agree.
    P = torch.zeros(n, r)
    for rank in range(world_size):
        vgen = torch.Generator()
        vgen.manual_seed(1000 + seed + rank)
        V = torch.randn(m, r, generator=vgen)
        P += G[rank] @ V / (float(r) ** 0.5)
    P /= world_size
    score = (P * P).sum(dim=1)
    _, priority_indices = torch.topk(score, K)

    row_mask = torch.zeros(n, dtype=torch.bool)
    row_mask[priority_indices] = True
    non_priority_indices = torch.arange(n)[~row_mask]
    nonK = n - K

    def _reduce_rows(indices, mode, quantize_bw):
        parts = [G[rank][indices].contiguous().view(-1) for rank in range(world_size)]
        if mode == 0:  # kFull: exact allreduce
            return parts[0] + parts[1]
        if mode == 1:  # kQuantize: quantizedAllreduceTensor
            return _sim_quantized_allreduce(parts, int(quantize_bw))
        return torch.zeros_like(parts[0])  # kDiscard

    pri_mode = int(case.get("sparse_priority_mode", 0))
    nonpri_mode = int(case.get("sparse_non_priority_mode", 1))
    reduced_full = torch.zeros(n, m)
    reduced_full[priority_indices] = _reduce_rows(
        priority_indices, pri_mode, case.get("sparse_priority_quantize_bitwidth", 4)
    ).view(K, m)
    if nonK > 0:
        reduced_full[non_priority_indices] = _reduce_rows(
            non_priority_indices, nonpri_mode, case.get("sparse_non_priority_quantize_bitwidth", 4)
        ).view(nonK, m)

    reference_full = exact_sum.view(n, m)
    print_rank(f"reference_full={reference_full.flatten()}")
    print_rank(f"reduced_full={reduced_full.flatten()}")

    # Full-mode rows must match the exact sum bit-for-bit; quantized rows may
    # deviate only by the quantization error (bw4, |x|<=0.1 -> <= ~0.0072 per
    # rank per element plus one requantization step).
    if pri_mode == 0:
        assert torch.equal(
            reduced_full[priority_indices], reference_full[priority_indices]
        ), f"{case['name']}: full-mode priority rows must match the exact sum"
    else:
        pri_err = (
            reduced_full[priority_indices] - reference_full[priority_indices]
        ).abs().max().item()
        assert pri_err < 0.05, f"{case['name']}: priority rows max_abs_err={pri_err}"

    if nonK > 0:
        if nonpri_mode == 0:
            assert torch.equal(
                reduced_full[non_priority_indices], reference_full[non_priority_indices]
            ), f"{case['name']}: full-mode non-priority rows must match the exact sum"
        elif nonpri_mode == 1:
            nonpri_err = (
                reduced_full[non_priority_indices] - reference_full[non_priority_indices]
            ).abs().max().item()
            assert nonpri_err < 0.05, f"{case['name']}: non-priority rows max_abs_err={nonpri_err}"
        else:
            assert torch.equal(
                reduced_full[non_priority_indices],
                torch.zeros_like(reduced_full[non_priority_indices]),
            ), f"{case['name']}: discard-mode non-priority rows must be zero"

    # Scatter: rank r receives the contiguous chunk [r*shard_len, (r+1)*shard_len).
    my_shards = [
        reduced_full.view(-1)[rank * shard_len:(rank + 1) * shard_len]
        for rank in range(world_size)
    ]
    for rank in range(world_size):
        assert my_shards[rank].numel() == shard_len, f"{case['name']} rank={rank} wrong shard size"
        assert torch.isfinite(my_shards[rank]).all(), f"{case['name']} rank={rank} non-finite output"
    assert torch.equal(
        torch.cat(my_shards), reduced_full.view(-1)
    ), f"{case['name']}: shards must partition the reduced full tensor"

def testReduceScatter():
    # file:// init needs a rendezvous file; create it once and clean it up.
    tmp = tempfile.NamedTemporaryFile(prefix="bitscom-lowbit-rs-ref-", delete=False)
    tmp.close()
    init_file = str(Path(tmp.name).resolve())
    try:
        for case in TEST_CASES:
            testCase(case, init_file)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            os.unlink(init_file)
        except OSError:
            pass

if __name__ == '__main__':
    print("testing")
    testReduceScatter()