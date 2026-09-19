"""双机 32 卡（每机 16 卡），跨节点两两一组共 16 组，组内 reduce_scatter 性能对比。

组 g 由 u62 与 u210 的第 g 张卡组成（local_rank 相同 → 全局 rank g 与 g+16）。
每组用纯稀疏(10%) bitscom 与纯 NCCL 各跑 ITERS 次 reduce_scatter，输出两者平均耗时与加速比。
"""

import os
import time

import torch
import torch.distributed as dist

import bitscom

N = 1024 * 1024   # 每个分片元素数
ITERS = 4         # 每组迭代次数


def log(*a):
    print(*a, flush=True)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 跨节点两两一组：组 g = {u62 local g, u210 local g} = {rank g, rank g + world_size/2}
    gid = local_rank
    group_ranks = [gid, gid + world_size // 2]

    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    # 纯稀疏 10%：priority 行全精度同步，非 priority 行丢弃
    bitscom.init(sparse_enabled=True, sparse_compression_ratio=0.1,
                 sparse_non_priority_mode=2)

    # use_local_synchronization=True：让每个不相交组的 group_name 按 ranks hash 生成
    # （否则 torch 用本地递增计数器，不同组会撞同一 group_name，store 里 NCCL id 冲突）。
    nccl_g = dist.new_group(ranks=group_ranks, backend="nccl",
                            use_local_synchronization=True,
                            device_id=torch.device("cuda", local_rank))
    bits_g = dist.new_group(ranks=group_ranks, backend="lowbit",
                            use_local_synchronization=True)

    def bench(group):
        inp = [torch.full((N,), float(rank), device="cuda") for _ in range(2)]
        out = torch.zeros(N, device="cuda")
        dist.reduce_scatter(output=out, input_list=inp, group=group)  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.reduce_scatter(output=out, input_list=inp, group=group)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ITERS * 1000.0  # ms/iter

    ms_nccl = bench(nccl_g)
    ms_bits = bench(bits_g)

    dist.barrier()
    if rank == group_ranks[0]:  # 组内 rank0（u62 侧）
        log(f"[group {gid}] nccl {ms_nccl:.2f}ms/iter  bitscom {ms_bits:.2f}ms/iter  "
            f"speedup {ms_nccl / ms_bits:.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
