"""最小 NCCL 连通性测试，无 bitscom 依赖。"""
import os
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group(backend="nccl")

print(f"[rank {rank}] world_size={dist.get_world_size()} init OK", flush=True)

# 测试通信
t = torch.ones(16, device=f"cuda:{rank}") * (rank + 1)
print(f"[rank {rank}] before all_reduce: {t[0].item()}", flush=True)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
print(f"[rank {rank}] after all_reduce: {t[0].item()}", flush=True)

dist.destroy_process_group()
print(f"[rank {rank}] done", flush=True)
