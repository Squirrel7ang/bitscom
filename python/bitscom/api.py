"""
High-level API for low-bit distributed communication.
"""

import torch
import torch.distributed as dist
from typing import Optional, List

from .quantization import (
    DEFAULT_BLOCK_SIZE,
    SUPPORTED_BITWIDTHS,
    CompressedTensor,
    compress_tensor,
    dequantize_tensor_blockwise,
    decompress_tensor,
    pack_lowbit,
    quantize_tensor_blockwise,
    roundtrip_tensor,
    unpack_lowbit,
)


class _ImmediateWork:
    """Simple Work-like object for sync-completed Python collectives."""

    def wait(self):
        return True


class LowBitGroup:
    """
    对 torch.distributed process_group 的封装，
    提供低比特通信原语。

    使用方式:
        # 方式1: 使用 lowbit backend
        bitscom.init()
        dist.init_process_group(backend="lowbit")
        group = LowBitGroup(bitwidth=4)
        group.all_reduce(tensor)

        # 方式2: 使用已有 process group
        dist.init_process_group(backend="nccl")
        group = LowBitGroup(bitwidth=4, process_group=dist.group.WORLD)
        group.all_reduce(tensor)
    """

    def __init__(
        self,
        bitwidth: int = 4,
        process_group: Optional[dist.ProcessGroup] = None,
        simulate_quantization: bool = False,
        stochastic_rounding: bool = False,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        """
        Args:
            bitwidth: 量化比特宽度 (1, 2, 4, 8, 12, 16)
            process_group: 使用的 process group，None 表示使用默认 group
            simulate_quantization: 使用非 lowbit backend 时，
                在通信前做一次量化-反量化模拟
            stochastic_rounding: 量化时使用随机舍入（默认关闭）
            block_size: block quantization size
        """
        if bitwidth not in SUPPORTED_BITWIDTHS:
            raise ValueError(
                f"bitwidth must be one of {SUPPORTED_BITWIDTHS}, got {bitwidth}"
            )
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.bitwidth = bitwidth
        self.pg = process_group or dist.distributed_c10d._get_default_group()
        self.simulate_quantization = simulate_quantization
        self.stochastic_rounding = stochastic_rounding
        self.block_size = int(block_size)

    @property
    def rank(self) -> int:
        return dist.get_rank(self.pg)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.pg)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
        local_group: Optional[dist.ProcessGroup] = None,
        inter_group: Optional[dist.ProcessGroup] = None,
        chunk_size: Optional[int] = None,
        local_quantize: bool = False,
    ):
        """
        低比特 all_reduce。

        当使用 lowbit backend 时，底层 C++ 会自动进行
        pack -> NCCL allreduce -> unpack 的流程。
        """
        if (
            local_group is not None
            and inter_group is None
            and self.bitwidth < 8
            and op == dist.ReduceOp.SUM
        ):
            # Single-node topology: keep local communication dense and avoid
            # introducing packing/unpacking overhead where bandwidth is not the bottleneck.
            if local_quantize:
                flat = tensor.contiguous().view(-1)
                reduced = self._lowbit_allreduce_via_alltoall_group(flat, local_group).view_as(tensor)
                tensor.copy_(reduced.to(dtype=tensor.dtype))
            else:
                # Single-node topology: local collective does not need compression.
                dist.all_reduce(tensor, op=op, group=local_group)
            if async_op:
                return _ImmediateWork()
            return None

        if self._should_use_pipeline_a(op, local_group, inter_group):

            self._hierarchical_lowbit_allreduce_pipeline_a(
                tensor,
                local_group=local_group,
                inter_group=inter_group,
                chunk_size=chunk_size,
                local_quantize=local_quantize,
            )
            if async_op:
                return _ImmediateWork()
            return None

        if self._should_use_lowbit_path(op):
            self._lowbit_allreduce_via_alltoall(tensor)
            if async_op:
                return _ImmediateWork()
            return None

        if self.simulate_quantization:
            tensor.copy_(roundtrip_tensor(tensor, self.bitwidth))
        work = dist.all_reduce(tensor, op=op, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work

    def _should_use_lowbit_path(self, op: dist.ReduceOp) -> bool:
        return (
            self.bitwidth < 8
            and self.world_size > 1
            and op == dist.ReduceOp.SUM
        )

    def _should_use_pipeline_a(
        self,
        op: dist.ReduceOp,
        local_group: Optional[dist.ProcessGroup],
        inter_group: Optional[dist.ProcessGroup],
    ) -> bool:
        return (
            local_group is not None
            and inter_group is not None
            and self.bitwidth < 8
            and op == dist.ReduceOp.SUM
        )

    def _split_flat_chunks(self, flat: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        chunks = []
        for start in range(0, flat.numel(), chunk_size):
            chunks.append(flat[start : start + chunk_size])
        return chunks

    def _should_use_dual_stream_pipeline(
        self,
        *,
        is_cuda: bool,
        local_size: int,
        global_size: int,
    ) -> bool:
        # Only use dual-stream overlap when inter-node communication exists.
        return is_cuda and local_size < global_size

    def _lowbit_allreduce_via_alltoall_group(
        self,
        flat: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        original_numel = int(flat.numel())
        if original_numel == 0:
            return flat

        pad = (world_size - (original_numel % world_size)) % world_size
        if pad:
            flat = torch.cat(
                [
                    flat,
                    torch.zeros(pad, dtype=flat.dtype, device=flat.device),
                ],
                dim=0,
            )

        shard_len = flat.numel() // world_size
        shards = list(flat.split(shard_len))
        send_packed = []
        send_scales = []
        for shard in shards:
            q_shard, scales = quantize_tensor_blockwise(
                shard,
                self.bitwidth,
                block_size=self.block_size,
                stochastic_rounding=self.stochastic_rounding,
            )
            send_packed.append(pack_lowbit(q_shard, self.bitwidth)[0])
            send_scales.append(scales)

        recv_packed = [torch.empty_like(send_packed[0]) for _ in range(world_size)]
        dist.all_to_all(recv_packed, send_packed, group=group)

        recv_scales = [torch.empty_like(send_scales[0]) for _ in range(world_size)]
        dist.all_to_all(recv_scales, send_scales, group=group)

        local_sum = torch.zeros(shard_len, dtype=torch.float32, device=flat.device)
        for src_rank in range(world_size):
            q_part = unpack_lowbit(recv_packed[src_rank], self.bitwidth, shard_len)
            fp_part = dequantize_tensor_blockwise(
                q_part,
                recv_scales[src_rank],
                block_size=self.block_size,
                dtype=torch.float32,
                device=flat.device,
            )
            local_sum.add_(fp_part)

        q_reduced, reduced_scales = quantize_tensor_blockwise(
            local_sum,
            self.bitwidth,
            block_size=self.block_size,
            stochastic_rounding=self.stochastic_rounding,
        )
        packed_reduced, _ = pack_lowbit(q_reduced, self.bitwidth)

        gathered_packed = [torch.empty_like(packed_reduced) for _ in range(world_size)]
        dist.all_gather(gathered_packed, packed_reduced, group=group)

        gathered_scales = [
            torch.empty_like(reduced_scales) for _ in range(world_size)
        ]
        dist.all_gather(gathered_scales, reduced_scales, group=group)

        out_shards = []
        for rank_idx in range(world_size):
            q_shard = unpack_lowbit(gathered_packed[rank_idx], self.bitwidth, shard_len)
            fp_shard = dequantize_tensor_blockwise(
                q_shard,
                gathered_scales[rank_idx],
                block_size=self.block_size,
                dtype=torch.float32,
                device=flat.device,
            )
            out_shards.append(fp_shard)

        restored = torch.cat(out_shards, dim=0)
        return restored[:original_numel]

    def _hierarchical_lowbit_allreduce_pipeline_a(
        self,
        tensor: torch.Tensor,
        *,
        local_group: dist.ProcessGroup,
        inter_group: dist.ProcessGroup,
        chunk_size: Optional[int],
        local_quantize: bool,
    ) -> None:
        flat = tensor.contiguous().view(-1)
        if flat.numel() == 0:
            return

        chunk_elems = int(chunk_size) if chunk_size is not None else max(1, flat.numel() // 4)
        chunks = self._split_flat_chunks(flat, chunk_elems)
        num_chunks = len(chunks)

        global_rank = dist.get_rank(self.pg)
        local_rank = dist.get_rank(local_group)
        local_size = dist.get_world_size(local_group)
        global_size = dist.get_world_size(self.pg)
        is_local_leader = global_rank % local_size == 0

        rank_tensor = torch.tensor(
            [global_rank],
            dtype=torch.int64,
            device=flat.device,
        )
        # gathered_local_ranks = [torch.empty_like(rank_tensor) for _ in range(local_size)]
        # print(f"[Global Rank {global_rank}] Local rank: {local_rank}, Local size: {local_size}, running on group: {local_group}")
        # dist.all_gather(gathered_local_ranks, rank_tensor, group=local_group)
        # print(f"[Global Rank {global_rank}] Gathered local ranks: {[t.item() for t in gathered_local_ranks]}")
        # local_leader_global = int(gathered_local_ranks[0].item())

        numels = [0] * num_chunks
        packed_templates = [None] * num_chunks
        scale_templates = [None] * num_chunks
        inter_results = [None] * num_chunks
        bcast_buffers = [None] * num_chunks
        packed_bcasts = [None] * num_chunks
        bcast_scale_tensors = [None] * num_chunks

        def _local_phase(idx: int) -> None:
            chunk = chunks[idx]
            # print(f"[Global Rank {global_rank}] Starting local phase for chunk {idx} with numel {chunk.numel()}")
            if not local_quantize:
                # In the default mode, local collectives stay full precision and
                # only the inter-node stage is compressed.
                # Local communication is high-bandwidth: keep it full precision.
                # print(f"[Global Rank {global_rank}] Starting local all-reduce for chunk {idx} with numel {chunk.numel()}")
                dist.reduce(chunk, dst=dist.get_rank() - dist.get_rank() % local_size, group=local_group, op=dist.ReduceOp.SUM)
                # dist.reduce(chunk, dst=0, group=local_group, op=dist.ReduceOp.SUM)
                
                # dist.all_reduce(chunk, group=local_group, op=dist.ReduceOp.SUM)
                # print(f"[Global Rank {global_rank}] Finished local all-reduce for chunk {idx}")
                return

            # Compatibility mode: quantize the local stage as well, which
            # preserves the original all-lowbit pipeline behavior.
            q_local, local_scales = quantize_tensor_blockwise(
                chunk,
                self.bitwidth,
                block_size=self.block_size,
                stochastic_rounding=self.stochastic_rounding,
            )
            packed_local, numel = pack_lowbit(q_local, self.bitwidth)
            numels[idx] = numel
            packed_templates[idx] = packed_local
            scale_templates[idx] = local_scales

            gathered_packed = [torch.empty_like(packed_local) for _ in range(local_size)]
            dist.all_gather(gathered_packed, packed_local, group=local_group)

            gathered_scales = [torch.empty_like(local_scales) for _ in range(local_size)]
            dist.all_gather(gathered_scales, local_scales, group=local_group)

            if is_local_leader:
                local_sum = torch.zeros(numel, dtype=torch.float32, device=chunk.device)
                for gather_idx in range(local_size):
                    q_part = unpack_lowbit(gathered_packed[gather_idx], self.bitwidth, numel)
                    fp_part = dequantize_tensor_blockwise(
                        q_part,
                        gathered_scales[gather_idx],
                        block_size=self.block_size,
                        dtype=torch.float32,
                        device=chunk.device,
                    )
                    local_sum.add_(fp_part)
                inter_results[idx] = local_sum

        def _inter_phase(idx: int) -> None:
            chunk = chunks[idx]
            if is_local_leader:
                # The inter stage always uses the low-bit all-reduce path because
                # this is where bandwidth pressure is highest in multi-node runs.
                # print(f"[Global Rank {global_rank}] Starting inter-node phase for chunk {idx} with numel {chunk.numel()}")
                inter_in = inter_results[idx] if local_quantize else chunk.to(dtype=torch.float32)
                inter_results[idx] = self._lowbit_allreduce_via_alltoall_group(inter_in, inter_group)
            else:
                inter_results[idx] = None
            barrier_device_ids = [torch.cuda.current_device()] if chunk.is_cuda else None
            dist.barrier(group=local_group, device_ids=barrier_device_ids)

        def _finalize_phase(idx: int) -> None:
            chunk = chunks[idx]
            if not local_quantize:
                if is_local_leader:
                    bcast_buffers[idx] = inter_results[idx].to(dtype=chunk.dtype)
                else:
                    bcast_buffers[idx] = torch.empty_like(chunk)

                # print(f"[Global Rank {global_rank}] Broadcasting inter-node result for chunk {idx} with numel {chunk.numel()}")
                # dist.broadcast(bcast_buffers[idx], src=dist.get_rank() - dist.get_rank() % local_size, group=local_group)
                dist.broadcast(bcast_buffers[idx], src=0, group=local_group)
                chunk.copy_(bcast_buffers[idx])
                return

            if is_local_leader:
                q_bcast, bcast_scales = quantize_tensor_blockwise(
                    inter_results[idx],
                    self.bitwidth,
                    block_size=self.block_size,
                    stochastic_rounding=self.stochastic_rounding,
                )
                packed_bcast, _ = pack_lowbit(q_bcast, self.bitwidth)
                packed_bcasts[idx] = packed_bcast
                bcast_scale_tensors[idx] = bcast_scales
            else:
                packed_bcasts[idx] = torch.empty_like(packed_templates[idx])
                bcast_scale_tensors[idx] = torch.empty_like(scale_templates[idx])

            # dist.broadcast(packed_bcasts[idx], src=dist.get_rank() - dist.get_rank() % local_size, group=local_group)
            # dist.broadcast(bcast_scale_tensors[idx], src=dist.get_rank() - dist.get_rank() % local_size, group=local_group)
            dist.broadcast(packed_bcasts[idx], src=0, group=local_group)
            dist.broadcast(bcast_scale_tensors[idx], src=0, group=local_group)

            q_recv = unpack_lowbit(packed_bcasts[idx], self.bitwidth, numels[idx])
            fp_recv = dequantize_tensor_blockwise(
                q_recv,
                bcast_scale_tensors[idx],
                block_size=self.block_size,
                dtype=torch.float32,
                device=chunk.device,
            )
            chunk.copy_(fp_recv.to(dtype=chunk.dtype))

        if not self._should_use_dual_stream_pipeline(
            is_cuda=flat.is_cuda,
            local_size=local_size,
            global_size=global_size,
        ):
            for idx in range(num_chunks):
                _local_phase(idx)
                _inter_phase(idx)
                _finalize_phase(idx)
            return

        intra_stream = torch.cuda.Stream(device=flat.device)
        inter_stream = torch.cuda.Stream(device=flat.device)
        event_list_intra = [torch.cuda.Event() for _ in range(num_chunks)]
        event_list_inter = [torch.cuda.Event() for _ in range(num_chunks)]

        # Warmup
        with torch.cuda.stream(intra_stream):
            _local_phase(0)
            event_list_intra[0].record(intra_stream)

        if num_chunks == 1:
            with torch.cuda.stream(inter_stream):
                inter_stream.wait_event(event_list_intra[0])
                _inter_phase(0)
                event_list_inter[0].record(inter_stream)
            with torch.cuda.stream(intra_stream):
                intra_stream.wait_event(event_list_inter[0])
                _finalize_phase(0)
            intra_stream.synchronize()
            inter_stream.synchronize()
            return

        with torch.cuda.stream(intra_stream):
            _local_phase(1)
            event_list_intra[1].record(intra_stream)

        with torch.cuda.stream(inter_stream):
            inter_stream.wait_event(event_list_intra[0])
            _inter_phase(0)
            event_list_inter[0].record(inter_stream)

        # Steady
        for idx in range(2, num_chunks):
            with torch.cuda.stream(intra_stream):
                intra_stream.wait_event(event_list_inter[idx - 2])
                _finalize_phase(idx - 2)
                _local_phase(idx)
                event_list_intra[idx].record(intra_stream)

            with torch.cuda.stream(inter_stream):
                inter_stream.wait_event(event_list_intra[idx - 1])
                _inter_phase(idx - 1)
                event_list_inter[idx - 1].record(inter_stream)

        # Cooldown
        with torch.cuda.stream(intra_stream):
            intra_stream.wait_event(event_list_inter[num_chunks - 2])
            _finalize_phase(num_chunks - 2)

        with torch.cuda.stream(inter_stream):
            inter_stream.wait_event(event_list_intra[num_chunks - 1])
            _inter_phase(num_chunks - 1)
            event_list_inter[num_chunks - 1].record(inter_stream)

        with torch.cuda.stream(intra_stream):
            intra_stream.wait_event(event_list_inter[num_chunks - 1])
            _finalize_phase(num_chunks - 1)

        intra_stream.synchronize()
        inter_stream.synchronize()

    def _lowbit_allreduce_via_alltoall(self, tensor: torch.Tensor) -> None:
        flat = tensor.contiguous().view(-1)
        restored = self._lowbit_allreduce_via_alltoall_group(flat, self.pg).view_as(tensor)
        tensor.copy_(restored.to(dtype=tensor.dtype))

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool = False,
    ):
        """低比特 all_gather。"""
        if self.simulate_quantization:
            tensor.copy_(roundtrip_tensor(tensor, self.bitwidth))
        work = dist.all_gather(tensor_list, tensor, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
    ):
        """低比特 reduce_scatter。"""
        if self.simulate_quantization:
            for t in input_list:
                t.copy_(roundtrip_tensor(t, self.bitwidth))
        work = dist.reduce_scatter(
            output, input_list, op=op, group=self.pg, async_op=True
        )
        if not async_op:
            work.wait()
            return None
        return work

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        async_op: bool = False,
    ):
        """broadcast（通常不需要压缩）。"""
        work = dist.broadcast(tensor, src=src, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        """将浮点 tensor 压缩为低比特打包表示。"""
        return compress_tensor(tensor, self.bitwidth)

    def decompress(
        self,
        compressed: CompressedTensor,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """将打包表示解压回浮点 tensor。"""
        return decompress_tensor(compressed, dtype=dtype, device=device)
