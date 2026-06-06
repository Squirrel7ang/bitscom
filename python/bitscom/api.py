"""
High-level API for low-bit distributed communication.
"""

import os
import time
from typing import Optional, List

import torch
import torch.distributed as dist

from .quantization import (
    DEFAULT_BLOCK_SIZE,
    SUPPORTED_BITWIDTHS,
    _HAS_CUDA_KERNELS,
    CompressedTensor,
    compress_tensor,
    decompress_tensor,
    quantize_pack_tensor_blockwise,
    roundtrip_tensor,
    unpack_dequantize_tensor_blockwise,
)


def _bitscom_timing_enabled() -> bool:
    return True


def _bitscom_timing_log_dir() -> str:
    directory = os.path.dirname(os.path.abspath(__file__))
    while True:
        if (
            os.path.isdir(os.path.join(directory, "polar-sgd"))
            and os.path.isdir(os.path.join(directory, "bitscom"))
        ):
            root = directory
            break
        parent = os.path.dirname(directory)
        if parent == directory:
            root = os.getcwd()
            break
        directory = parent
    log_dir = os.path.join(root, "debug_logs", "timing")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _bitscom_timing(message: str) -> None:
    if not _bitscom_timing_enabled():
        return
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1
    line = f"[bitscom-timing rank={rank} t={time.time():.6f}] {message}"
    log_file = os.path.join(_bitscom_timing_log_dir(), f"bitscom_rank{rank}.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class _ImmediateWork:
    """Simple Work-like object for sync-completed Python collectives."""

    def wait(self):
        return True


class _CudaEventWork:
    """Work-like handle backed by a CUDA event recorded on a comm stream."""

    def __init__(
        self,
        event: torch.cuda.Event,
        keepalive: Optional[List[torch.Tensor]] = None,
    ):
        self.event = event
        self.keepalive = keepalive or []

    def wait(self):
        self.event.synchronize()
        return True

    def block_current_stream(self):
        torch.cuda.current_stream().wait_event(self.event)
        return True

    def is_completed(self) -> bool:
        return bool(self.event.query())


class _NativeLowBitAllReduceWork:
    """Work-like low-bit all-reduce built from native async collectives."""

    def __init__(
        self,
        owner,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ):
        self.owner = owner
        self.tensor = tensor
        self.group = group
        self.debug = owner._comm_debug_enabled()
        self.t0 = time.perf_counter()
        self.completed = False
        self.final_future = torch.futures.Future()
        self._fallback_wait_driven = False

        self.flat = tensor.contiguous().view(-1)
        self.world_size = dist.get_world_size(group)
        self.original_numel = int(self.flat.numel())
        self.empty = self.original_numel == 0
        if self.empty:
            self.packed_work = None
            self.scales_work = None
            self.completed = True
            self.final_future.set_result(True)
            return

        pad = (self.world_size - (self.original_numel % self.world_size)) % self.world_size
        if pad:
            self.flat = torch.cat(
                [
                    self.flat,
                    torch.zeros(pad, dtype=self.flat.dtype, device=self.flat.device),
                ],
                dim=0,
            )

        self.shard_len = self.flat.numel() // self.world_size
        shards = list(self.flat.split(self.shard_len))
        if self.debug:
            owner._comm_debug(
                "lowbit async allreduce start "
                f"numel={self.original_numel} padded_numel={self.flat.numel()} "
                f"world_size={self.world_size} shard_len={self.shard_len} "
                f"dtype={self.flat.dtype} device={self.flat.device} "
                f"block_size={owner.block_size} cuda_kernels={_HAS_CUDA_KERNELS}"
            )
        owner._comm_evidence(
            "lowbit_allreduce_async_plan",
            "meaning='bitscom will launch native async collectives for the "
            "packed low-bit exchange so network communication can overlap "
            "with caller-side compute before wait()' "
            f"original_numel={self.original_numel} padded_numel={int(self.flat.numel())} "
            f"world_size={self.world_size} shard_len={self.shard_len}",
        )

        self.send_packed = []
        self.send_scales = []
        for shard_idx, shard in enumerate(shards):
            packed, scales, _ = quantize_pack_tensor_blockwise(
                shard,
                owner.bitwidth,
                block_size=owner.block_size,
                stochastic_rounding=owner.stochastic_rounding,
            )
            self.send_packed.append(packed)
            self.send_scales.append(scales)
            if self.debug:
                owner._comm_debug(
                    f"lowbit async quantize shard={shard_idx} "
                    f"packed_numel={packed.numel()} scales_numel={scales.numel()}"
                )

        self.recv_packed = [
            torch.empty_like(self.send_packed[0]) for _ in range(self.world_size)
        ]
        self.recv_scales = [
            torch.empty_like(self.send_scales[0]) for _ in range(self.world_size)
        ]
        self.packed_work = dist.all_to_all(
            self.recv_packed,
            self.send_packed,
            group=group,
            async_op=True,
        )
        self.scales_work = dist.all_to_all(
            self.recv_scales,
            self.send_scales,
            group=group,
            async_op=True,
        )
        self._chain_after_first_collectives()

    @staticmethod
    def _work_future(work):
        get_future = getattr(work, "get_future", None)
        if get_future is None:
            return None
        return get_future()

    def _chain_after_first_collectives(self) -> None:
        if not self.owner._full_async_chain_enabled():
            self._fallback_wait_driven = True
            return

        packed_future = self._work_future(self.packed_work)
        scales_future = self._work_future(self.scales_work)
        if packed_future is None or scales_future is None:
            self._fallback_wait_driven = True
            return

        torch.futures.collect_all([packed_future, scales_future]).then(
            self._after_first_collectives
        )

    def _after_first_collectives(self, _future) -> None:
        try:
            _future.wait()
            self._launch_second_collectives()
        except BaseException as exc:
            self.final_future.set_exception(exc)

    def _launch_second_collectives(self) -> None:
        owner = self.owner
        local_sum = torch.zeros(
            self.shard_len,
            dtype=torch.float32,
            device=self.flat.device,
        )
        for src_rank in range(self.world_size):
            fp_part = unpack_dequantize_tensor_blockwise(
                self.recv_packed[src_rank],
                self.recv_scales[src_rank],
                owner.bitwidth,
                self.shard_len,
                block_size=owner.block_size,
                dtype=torch.float32,
                device=self.flat.device,
            )
            local_sum.add_(fp_part)

        self.packed_reduced, self.reduced_scales, _ = quantize_pack_tensor_blockwise(
            local_sum,
            owner.bitwidth,
            block_size=owner.block_size,
            stochastic_rounding=owner.stochastic_rounding,
        )
        self.gathered_packed = [
            torch.empty_like(self.packed_reduced) for _ in range(self.world_size)
        ]
        self.gathered_scales = [
            torch.empty_like(self.reduced_scales) for _ in range(self.world_size)
        ]
        self.gather_packed_work = dist.all_gather(
            self.gathered_packed,
            self.packed_reduced,
            group=self.group,
            async_op=True,
        )
        self.gather_scales_work = dist.all_gather(
            self.gathered_scales,
            self.reduced_scales,
            group=self.group,
            async_op=True,
        )
        packed_future = self._work_future(self.gather_packed_work)
        scales_future = self._work_future(self.gather_scales_work)
        if packed_future is None or scales_future is None:
            self.gather_packed_work.wait()
            self.gather_scales_work.wait()
            self._finalize()
            return

        torch.futures.collect_all([packed_future, scales_future]).then(
            self._after_second_collectives
        )

    def _after_second_collectives(self, _future) -> None:
        try:
            _future.wait()
            self._finalize()
        except BaseException as exc:
            self.final_future.set_exception(exc)

    def _finalize(self) -> None:
        owner = self.owner
        out_shards = []
        for rank_idx in range(self.world_size):
            fp_shard = unpack_dequantize_tensor_blockwise(
                self.gathered_packed[rank_idx],
                self.gathered_scales[rank_idx],
                owner.bitwidth,
                self.shard_len,
                block_size=owner.block_size,
                dtype=torch.float32,
                device=self.flat.device,
            )
            out_shards.append(fp_shard)

        restored = torch.cat(out_shards, dim=0)[: self.original_numel]
        self.tensor.copy_(restored.view_as(self.tensor).to(dtype=self.tensor.dtype))
        self.completed = True
        if self.debug:
            owner._comm_debug(
                f"lowbit async allreduce done elapsed_s={time.perf_counter() - self.t0:.3f}"
            )
        self.final_future.set_result(True)

    def wait(self):
        if self._fallback_wait_driven and not self.completed:
            self.packed_work.wait()
            self.scales_work.wait()
            self._launch_second_collectives()
        self.final_future.wait()
        return True

    def is_completed(self) -> bool:
        return self.completed


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
        backend_allreduce: bool = False,
    ):
        """
        Args:
            bitwidth: 量化比特宽度 (1, 2, 4, 8, 12, 16)
            process_group: 使用的 process group，None 表示使用默认 group
            simulate_quantization: 使用非 lowbit backend 时，
                在通信前做一次量化-反量化模拟
            stochastic_rounding: 量化时使用随机舍入（默认关闭）
            block_size: block quantization size
            backend_allreduce: delegate all_reduce to the process-group backend
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
        self.backend_allreduce = bool(backend_allreduce)

    def _comm_debug_enabled(self) -> bool:
        return os.environ.get("BITSCOM_COMM_DEBUG", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _comm_debug(self, message: str) -> None:
        if not self._comm_debug_enabled():
            return
        try:
            rank = dist.get_rank(self.pg)
        except Exception:
            rank = -1
        print(
            f"[bitscom-debug rank={rank} bitwidth={self.bitwidth}] {message}",
            flush=True,
        )

    def _full_async_chain_enabled(self) -> bool:
        return os.environ.get("BITSCOM_UNSAFE_FULL_ASYNC_CHAIN", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _trace_explain_enabled(self) -> bool:
        return os.environ.get("TRACE_EXPLAIN", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _comm_evidence(self, action: str, message: str) -> None:
        if not self._trace_explain_enabled():
            return
        try:
            rank = dist.get_rank(self.pg)
        except Exception:
            rank = -1
        print(
            f"[trace-evidence rank={rank}] component=bitscom "
            f"action={action} bitwidth={self.bitwidth} {message}",
            flush=True,
        )

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
        if self._comm_debug_enabled():
            self._comm_debug(
                "all_reduce entry "
                f"numel={int(tensor.numel())} dtype={tensor.dtype} "
                f"device={tensor.device} op={op} async_op={async_op} "
                f"local_group={local_group is not None} "
                f"inter_group={inter_group is not None} "
                f"local_quantize={local_quantize} "
                f"chunk_size={chunk_size}"
            )
        self._comm_evidence(
            "all_reduce_entry",
            "meaning='bitscom LowBitGroup received a DP gradient buffer from "
            "the training code and will choose the communication implementation' "
            f"numel={int(tensor.numel())} dtype={tensor.dtype} "
            f"device={tensor.device} async_op={async_op} "
            f"local_group={local_group is not None} "
                f"inter_group={inter_group is not None}",
        )
        if self.backend_allreduce:
            self._comm_debug("path=backend_lowbit_allreduce")
            work = dist.all_reduce(tensor, op=op, group=self.pg, async_op=True)
            if async_op:
                return work
            work.wait()
            return None

        if (
            local_group is not None
            and inter_group is None
            and self.bitwidth < 8
            and op == dist.ReduceOp.SUM
        ):
            # Single-node topology: keep local communication dense and avoid
            # introducing packing/unpacking overhead where bandwidth is not the bottleneck.
            if local_quantize:
                self._comm_debug("path=local_group_lowbit")
                if async_op:
                    return self._lowbit_allreduce_via_alltoall_async(
                        tensor,
                        local_group,
                    )
                self._lowbit_allreduce_via_alltoall_into_(tensor, local_group)
            else:
                # Single-node topology: local collective does not need compression.
                self._comm_debug("path=local_group_dense")
                work = dist.all_reduce(
                    tensor,
                    op=op,
                    group=local_group,
                    async_op=True,
                )
                if async_op:
                    return work
                work.wait()
            return None

        if self._should_use_pipeline_a(op, local_group, inter_group):
            self._comm_debug("path=hierarchical_lowbit_pipeline_a")
            self._comm_evidence(
                "path_selected",
                "meaning='bitscom selected hierarchical low-bit all-reduce: "
                "dense local aggregation, low-bit inter-node communication, "
                "then local broadcast back to all local ranks' "
                f"local_quantize={local_quantize} chunk_size={chunk_size}",
            )
            if async_op:
                # The flat low-bit path below has native async collectives.
                # pipeline_a still contains ordered local reduce/barrier/broadcast
                # phases, so keep its semantics synchronous until it gets a
                # dedicated native Work implementation.
                self._hierarchical_lowbit_allreduce_pipeline_a(
                    tensor,
                    local_group=local_group,
                    inter_group=inter_group,
                    chunk_size=chunk_size,
                    local_quantize=local_quantize,
                )
                return _ImmediateWork()
            self._hierarchical_lowbit_allreduce_pipeline_a(
                tensor,
                local_group=local_group,
                inter_group=inter_group,
                chunk_size=chunk_size,
                local_quantize=local_quantize,
            )
            return None

        if self._should_use_lowbit_path(op):
            self._comm_debug("path=flat_lowbit_alltoall")
            self._comm_evidence(
                "path_selected",
                "meaning='bitscom selected flat low-bit all-reduce over the "
                "whole process group' implementation=alltoall_quantized",
            )
            if async_op:
                return self._lowbit_allreduce_via_alltoall_async(tensor, self.pg)
            self._lowbit_allreduce_via_alltoall(tensor)
            return None

        if self.simulate_quantization:
            self._comm_debug("path=dense_with_quantization_simulation")
            tensor.copy_(roundtrip_tensor(tensor, self.bitwidth))
        else:
            self._comm_debug("path=dense_all_reduce")
        work = dist.all_reduce(tensor, op=op, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work

    def all_reduce_stream(
        self,
        tensor: torch.Tensor,
        *,
        stream: torch.cuda.Stream,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: Optional[dist.ProcessGroup] = None,
        post_scale: float = 1.0,
    ):
        """Launch a low-bit all-reduce as a stream-ordered CUDA workload.

        This path keeps the whole low-bit sequence on one caller-provided
        stream: pack, all-to-all, unpack/reduce, repack, all-gather, final
        unpack/copy. It returns an event-backed Work; callers can delay
        waiting until the reduced tensor is actually needed.
        """
        process_group = group or self.pg
        if op != dist.ReduceOp.SUM or not self._should_use_lowbit_path_for_group(
            op,
            process_group,
        ):
            if not tensor.is_cuda:
                dist.all_reduce(tensor, op=op, group=process_group)
                if post_scale != 1.0:
                    tensor.mul_(post_scale)
                return _ImmediateWork()
            stream.wait_stream(torch.cuda.current_stream(tensor.device))
            with torch.cuda.stream(stream):
                work = dist.all_reduce(
                    tensor,
                    op=op,
                    group=process_group,
                    async_op=True,
                )
                self._block_work_on_current_stream(work)
                if post_scale != 1.0:
                    tensor.mul_(post_scale)
                done = torch.cuda.Event()
                done.record(stream)
            return _CudaEventWork(done)

        if not tensor.is_cuda:
            self._lowbit_allreduce_via_alltoall_into_(tensor, process_group)
            if post_scale != 1.0:
                tensor.mul_(post_scale)
            return _ImmediateWork()

        return self._lowbit_allreduce_via_alltoall_stream_(
            tensor,
            process_group,
            stream=stream,
            post_scale=post_scale,
        )

    def _should_use_lowbit_path(self, op: dist.ReduceOp) -> bool:
        return (
            self.bitwidth < 8
            and self.world_size > 1
            and op == dist.ReduceOp.SUM
        )

    def _should_use_lowbit_path_for_group(
        self,
        op: dist.ReduceOp,
        group: dist.ProcessGroup,
    ) -> bool:
        return (
            self.bitwidth < 8
            and dist.get_world_size(group) > 1
            and op == dist.ReduceOp.SUM
        )

    @staticmethod
    def _block_work_on_current_stream(work) -> None:
        if hasattr(work, "block_current_stream"):
            t0 = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_block_work_on_current_stream:block_current_stream"
            ):
                work.block_current_stream()
            _bitscom_timing(
                "block_work_on_current_stream block_current_stream returned "
                f"elapsed_ms={(time.perf_counter() - t0) * 1000.0:.3f}"
            )
        else:
            t0 = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_block_work_on_current_stream:fallback_wait"
            ):
                work.wait()
            _bitscom_timing(
                "block_work_on_current_stream fallback_wait returned "
                f"elapsed_ms={(time.perf_counter() - t0) * 1000.0:.3f}"
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
        debug = self._comm_debug_enabled()
        t0 = time.perf_counter()
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
        if debug:
            self._comm_debug(
                "lowbit allreduce start "
                f"numel={original_numel} padded_numel={flat.numel()} "
                f"world_size={world_size} shard_len={shard_len} "
                f"dtype={flat.dtype} device={flat.device} "
                f"block_size={self.block_size} "
                f"cuda_kernels={_HAS_CUDA_KERNELS}"
            )
        self._comm_evidence(
            "lowbit_allreduce_plan",
            "meaning='bitscom will split the flat gradient, quantize/pack each "
            "shard, exchange packed payloads, unpack/dequantize, sum, then "
            "gather the reduced low-bit shards back' "
            f"original_numel={original_numel} padded_numel={int(flat.numel())} "
            f"world_size={world_size} shard_len={shard_len} "
            f"block_size={self.block_size} cuda_kernels={_HAS_CUDA_KERNELS}",
        )

        send_packed = []
        send_scales = []
        for shard_idx, shard in enumerate(shards):
            if debug:
                self._comm_debug(f"quantize shard {shard_idx} start")
            if shard_idx == 0:
                self._comm_evidence(
                    "quantize_pack",
                    "meaning='floating-point gradient shard is converted into "
                    "low-bit packed values plus per-block scales before network "
                    "communication' "
                    f"shard_idx={shard_idx} shard_numel={int(shard.numel())}",
                )
            packed, scales, _ = quantize_pack_tensor_blockwise(
                shard,
                self.bitwidth,
                block_size=self.block_size,
                stochastic_rounding=self.stochastic_rounding,
            )
            send_packed.append(packed)
            send_scales.append(scales)
            if debug:
                self._comm_debug(
                    f"quantize shard {shard_idx} done "
                    f"packed_numel={packed.numel()} scales_numel={scales.numel()}"
                )

        recv_packed = [torch.empty_like(send_packed[0]) for _ in range(world_size)]
        if debug:
            self._comm_debug("all_to_all packed start")
        self._comm_evidence(
            "collective",
            "meaning='packed low-bit payloads are exchanged across ranks; this "
            "is the network communication over compressed data' collective=all_to_all "
            f"world_size={world_size}",
        )
        dist.all_to_all(recv_packed, send_packed, group=group)
        if debug:
            self._comm_debug("all_to_all packed done")

        recv_scales = [torch.empty_like(send_scales[0]) for _ in range(world_size)]
        if debug:
            self._comm_debug("all_to_all scales start")
        dist.all_to_all(recv_scales, send_scales, group=group)
        if debug:
            self._comm_debug("all_to_all scales done")

        local_sum = torch.zeros(shard_len, dtype=torch.float32, device=flat.device)
        for src_rank in range(world_size):
            if debug:
                self._comm_debug(f"unpack received shard from src={src_rank} start")
            fp_part = unpack_dequantize_tensor_blockwise(
                recv_packed[src_rank],
                recv_scales[src_rank],
                self.bitwidth,
                shard_len,
                block_size=self.block_size,
                dtype=torch.float32,
                device=flat.device,
            )
            local_sum.add_(fp_part)
            if debug:
                self._comm_debug(f"unpack received shard from src={src_rank} done")

        if debug:
            self._comm_debug("quantize reduced shard start")
        packed_reduced, reduced_scales, _ = quantize_pack_tensor_blockwise(
            local_sum,
            self.bitwidth,
            block_size=self.block_size,
            stochastic_rounding=self.stochastic_rounding,
        )
        if debug:
            self._comm_debug(
                "quantize reduced shard done "
                f"packed_numel={packed_reduced.numel()} "
                f"scales_numel={reduced_scales.numel()}"
            )

        gathered_packed = [torch.empty_like(packed_reduced) for _ in range(world_size)]
        if debug:
            self._comm_debug("all_gather packed start")
        self._comm_evidence(
            "collective",
            "meaning='each rank gathers the reduced packed shards so every rank "
            "can reconstruct the full reduced gradient buffer' collective=all_gather "
            f"world_size={world_size}",
        )
        dist.all_gather(gathered_packed, packed_reduced, group=group)
        if debug:
            self._comm_debug("all_gather packed done")

        gathered_scales = [
            torch.empty_like(reduced_scales) for _ in range(world_size)
        ]
        if debug:
            self._comm_debug("all_gather scales start")
        dist.all_gather(gathered_scales, reduced_scales, group=group)
        if debug:
            self._comm_debug("all_gather scales done")

        out_shards = []
        for rank_idx in range(world_size):
            if debug:
                self._comm_debug(f"unpack gathered shard {rank_idx} start")
            fp_shard = unpack_dequantize_tensor_blockwise(
                gathered_packed[rank_idx],
                gathered_scales[rank_idx],
                self.bitwidth,
                shard_len,
                block_size=self.block_size,
                dtype=torch.float32,
                device=flat.device,
            )
            out_shards.append(fp_shard)
            if debug:
                self._comm_debug(f"unpack gathered shard {rank_idx} done")

        restored = torch.cat(out_shards, dim=0)
        if debug:
            self._comm_debug(
                f"lowbit allreduce done elapsed_s={time.perf_counter() - t0:.3f}"
            )
        self._comm_evidence(
            "lowbit_allreduce_done",
            "meaning='bitscom has reconstructed the reduced gradient buffer and "
            "returns it to the POLAR hook' "
            f"elapsed_s={time.perf_counter() - t0:.3f}",
        )
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
        local_leader_global = global_rank - local_rank
        debug = self._comm_debug_enabled()
        if debug:
            self._comm_debug(
                "pipeline_a start "
                f"numel={int(flat.numel())} chunks={num_chunks} "
                f"chunk_elems={chunk_elems} local_rank={local_rank} "
                f"local_size={local_size} global_rank={global_rank} "
                f"global_size={global_size} is_local_leader={is_local_leader} "
                f"local_leader_global={local_leader_global}"
            )
        self._comm_evidence(
            "pipeline_a_plan",
            "meaning='hierarchical bitscom path: local ranks first aggregate "
            "inside a node, local leader performs low-bit inter-node all-reduce, "
            "then the result is broadcast inside the node' "
            f"chunks={num_chunks} chunk_elems={chunk_elems} "
            f"local_rank={local_rank} local_size={local_size} "
            f"is_local_leader={is_local_leader}",
        )

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
            if debug:
                self._comm_debug(
                    f"pipeline_a local_phase chunk={idx} start "
                    f"numel={int(chunk.numel())} local_quantize={local_quantize}"
                )
            if idx == 0:
                self._comm_evidence(
                    "pipeline_local_phase",
                    "meaning='within-node aggregation starts; by default this "
                    "phase is dense because intra-node bandwidth is not the "
                    "bottleneck' "
                    f"chunk={idx} numel={int(chunk.numel())} "
                    f"local_quantize={local_quantize}",
                )
            # print(f"[Global Rank {global_rank}] Starting local phase for chunk {idx} with numel {chunk.numel()}")
            if not local_quantize:
                # In the default mode, local collectives stay full precision and
                # only the inter-node stage is compressed.
                # Local communication is high-bandwidth: keep it full precision.
                # print(f"[Global Rank {global_rank}] Starting local all-reduce for chunk {idx} with numel {chunk.numel()}")
                dist.reduce(chunk, dst=local_leader_global, group=local_group, op=dist.ReduceOp.SUM)
                
                # dist.all_reduce(chunk, group=local_group, op=dist.ReduceOp.SUM)
                # print(f"[Global Rank {global_rank}] Finished local all-reduce for chunk {idx}")
                if debug:
                    self._comm_debug(f"pipeline_a local_phase chunk={idx} dense_reduce done")
                return

            # Compatibility mode: quantize the local stage as well, which
            # preserves the original all-lowbit pipeline behavior.
            packed_local, local_scales, numel = quantize_pack_tensor_blockwise(
                chunk,
                self.bitwidth,
                block_size=self.block_size,
                stochastic_rounding=self.stochastic_rounding,
            )
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
                    fp_part = unpack_dequantize_tensor_blockwise(
                        gathered_packed[gather_idx],
                        gathered_scales[gather_idx],
                        self.bitwidth,
                        numel,
                        block_size=self.block_size,
                        dtype=torch.float32,
                        device=chunk.device,
                    )
                    local_sum.add_(fp_part)
                inter_results[idx] = local_sum
            if debug:
                self._comm_debug(f"pipeline_a local_phase chunk={idx} lowbit done")

        def _inter_phase(idx: int) -> None:
            chunk = chunks[idx]
            if debug:
                self._comm_debug(
                    f"pipeline_a inter_phase chunk={idx} start "
                    f"is_local_leader={is_local_leader}"
                )
            if idx == 0:
                self._comm_evidence(
                    "pipeline_inter_node_phase",
                    "meaning='the local leader now performs low-bit inter-node "
                    "all-reduce; this is the bandwidth-sensitive part accelerated "
                    "by bitscom quantization' "
                    f"chunk={idx} is_local_leader={is_local_leader}",
                )
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
            if debug:
                self._comm_debug(f"pipeline_a inter_phase chunk={idx} done")

        def _finalize_phase(idx: int) -> None:
            chunk = chunks[idx]
            if debug:
                self._comm_debug(
                    f"pipeline_a finalize_phase chunk={idx} start "
                    f"is_local_leader={is_local_leader}"
                )
            if idx == 0:
                self._comm_evidence(
                    "pipeline_finalize_phase",
                    "meaning='the reduced inter-node result is distributed back "
                    "to local ranks so every GPU receives the averaged DP "
                    "gradient buffer' "
                    f"chunk={idx} is_local_leader={is_local_leader}",
                )
            if not local_quantize:
                if is_local_leader:
                    bcast_buffers[idx] = inter_results[idx].to(dtype=chunk.dtype)
                else:
                    bcast_buffers[idx] = torch.empty_like(chunk)

                # print(f"[Global Rank {global_rank}] Broadcasting inter-node result for chunk {idx} with numel {chunk.numel()}")
                dist.broadcast(bcast_buffers[idx], src=local_leader_global, group=local_group)
                chunk.copy_(bcast_buffers[idx])
                if debug:
                    self._comm_debug(f"pipeline_a finalize_phase chunk={idx} dense_broadcast done")
                return

            if is_local_leader:
                packed_bcast, bcast_scales, _ = quantize_pack_tensor_blockwise(
                    inter_results[idx],
                    self.bitwidth,
                    block_size=self.block_size,
                    stochastic_rounding=self.stochastic_rounding,
                )
                packed_bcasts[idx] = packed_bcast
                bcast_scale_tensors[idx] = bcast_scales
            else:
                packed_bcasts[idx] = torch.empty_like(packed_templates[idx])
                bcast_scale_tensors[idx] = torch.empty_like(scale_templates[idx])

            dist.broadcast(packed_bcasts[idx], src=local_leader_global, group=local_group)
            dist.broadcast(bcast_scale_tensors[idx], src=local_leader_global, group=local_group)

            fp_recv = unpack_dequantize_tensor_blockwise(
                packed_bcasts[idx],
                bcast_scale_tensors[idx],
                self.bitwidth,
                numels[idx],
                block_size=self.block_size,
                dtype=torch.float32,
                device=chunk.device,
            )
            chunk.copy_(fp_recv.to(dtype=chunk.dtype))
            if debug:
                self._comm_debug(f"pipeline_a finalize_phase chunk={idx} lowbit_broadcast done")

        if not self._should_use_dual_stream_pipeline(
            is_cuda=flat.is_cuda,
            local_size=local_size,
            global_size=global_size,
        ):
            for idx in range(num_chunks):
                _local_phase(idx)
                _inter_phase(idx)
                _finalize_phase(idx)
            if debug:
                self._comm_debug("pipeline_a done")
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
        if debug:
            self._comm_debug("pipeline_a done")

    def _lowbit_allreduce_via_alltoall(self, tensor: torch.Tensor) -> None:
        self._lowbit_allreduce_via_alltoall_into_(tensor, self.pg)

    def _lowbit_allreduce_via_alltoall_async(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ):
        return _NativeLowBitAllReduceWork(self, tensor, group)

    def _lowbit_allreduce_via_alltoall_stream_(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        *,
        stream: torch.cuda.Stream,
        post_scale: float,
    ):
        func_t0 = time.perf_counter()
        debug = self._comm_debug_enabled()
        world_size = dist.get_world_size(group)
        original_numel = int(tensor.numel())
        done = torch.cuda.Event()
        _bitscom_timing(
            "lowbit_stream enter "
            f"numel={original_numel} world_size={world_size} "
            f"dtype={tensor.dtype} device={tensor.device}"
        )
        if original_numel == 0:
            stream.wait_stream(torch.cuda.current_stream(tensor.device))
            with torch.cuda.stream(stream):
                done.record(stream)
            _bitscom_timing(
                "SUMMARY lowbit_stream empty exit "
                f"elapsed_ms={(time.perf_counter() - func_t0) * 1000.0:.3f}"
            )
            return _CudaEventWork(done)

        t_wait_stream = time.perf_counter()
        stream.wait_stream(torch.cuda.current_stream(tensor.device))
        _bitscom_timing(
            "lowbit_stream wait_stream returned "
            f"elapsed_ms={(time.perf_counter() - t_wait_stream) * 1000.0:.3f}"
        )
        with torch.cuda.stream(stream):
            stream_body_t0 = time.perf_counter()
            t_layout = time.perf_counter()
            flat = tensor.contiguous().view(-1)
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
            _bitscom_timing(
                "lowbit_stream stage=layout returned "
                f"padded_numel={int(flat.numel())} shard_len={int(shard_len)} "
                f"elapsed_ms={(time.perf_counter() - t_layout) * 1000.0:.3f}"
            )
            if debug:
                self._comm_debug(
                    "lowbit stream allreduce enqueue "
                    f"numel={original_numel} padded_numel={flat.numel()} "
                    f"world_size={world_size} shard_len={shard_len} "
                    f"dtype={flat.dtype} device={flat.device} "
                    f"block_size={self.block_size} "
                    f"cuda_kernels={_HAS_CUDA_KERNELS}"
                )

            send_packed = []
            send_scales = []
            t_quantize_send = time.perf_counter()
            for shard in shards:
                packed, scales, _ = quantize_pack_tensor_blockwise(
                    shard,
                    self.bitwidth,
                    block_size=self.block_size,
                    stochastic_rounding=self.stochastic_rounding,
                )
                send_packed.append(packed)
                send_scales.append(scales)
            _bitscom_timing(
                "lowbit_stream stage=quantize_send returned "
                f"chunks={len(send_packed)} "
                f"elapsed_ms={(time.perf_counter() - t_quantize_send) * 1000.0:.3f}"
            )

            recv_packed = [torch.empty_like(send_packed[0]) for _ in range(world_size)]
            t_launch = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:launch_all_to_all_packed"
            ):
                packed_work = dist.all_to_all(
                    recv_packed,
                    send_packed,
                    group=group,
                    async_op=True,
                )
            _bitscom_timing(
                "lowbit_stream stage=launch_all_to_all_packed returned "
                f"elapsed_ms={(time.perf_counter() - t_launch) * 1000.0:.3f}"
            )
            t_block = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:block_all_to_all_packed"
            ):
                self._block_work_on_current_stream(packed_work)
            _bitscom_timing(
                "lowbit_stream stage=block_all_to_all_packed returned "
                f"elapsed_ms={(time.perf_counter() - t_block) * 1000.0:.3f}"
            )

            recv_scales = [torch.empty_like(send_scales[0]) for _ in range(world_size)]
            t_launch = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:launch_all_to_all_scales"
            ):
                scales_work = dist.all_to_all(
                    recv_scales,
                    send_scales,
                    group=group,
                    async_op=True,
                )
            _bitscom_timing(
                "lowbit_stream stage=launch_all_to_all_scales returned "
                f"elapsed_ms={(time.perf_counter() - t_launch) * 1000.0:.3f}"
            )
            t_block = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:block_all_to_all_scales"
            ):
                self._block_work_on_current_stream(scales_work)
            _bitscom_timing(
                "lowbit_stream stage=block_all_to_all_scales returned "
                f"elapsed_ms={(time.perf_counter() - t_block) * 1000.0:.3f}"
            )

            t_reduce = time.perf_counter()
            local_sum = torch.zeros(shard_len, dtype=torch.float32, device=flat.device)
            for src_rank in range(world_size):
                fp_part = unpack_dequantize_tensor_blockwise(
                    recv_packed[src_rank],
                    recv_scales[src_rank],
                    self.bitwidth,
                    shard_len,
                    block_size=self.block_size,
                    dtype=torch.float32,
                    device=flat.device,
                )
                local_sum.add_(fp_part)
            _bitscom_timing(
                "lowbit_stream stage=unpack_reduce returned "
                f"elapsed_ms={(time.perf_counter() - t_reduce) * 1000.0:.3f}"
            )

            t_requantize = time.perf_counter()
            packed_reduced, reduced_scales, _ = quantize_pack_tensor_blockwise(
                local_sum,
                self.bitwidth,
                block_size=self.block_size,
                stochastic_rounding=self.stochastic_rounding,
            )
            _bitscom_timing(
                "lowbit_stream stage=quantize_reduced returned "
                f"elapsed_ms={(time.perf_counter() - t_requantize) * 1000.0:.3f}"
            )

            gathered_packed = [
                torch.empty_like(packed_reduced) for _ in range(world_size)
            ]
            t_launch = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:launch_all_gather_packed"
            ):
                gather_packed_work = dist.all_gather(
                    gathered_packed,
                    packed_reduced,
                    group=group,
                    async_op=True,
                )
            _bitscom_timing(
                "lowbit_stream stage=launch_all_gather_packed returned "
                f"elapsed_ms={(time.perf_counter() - t_launch) * 1000.0:.3f}"
            )
            t_block = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:block_all_gather_packed"
            ):
                self._block_work_on_current_stream(gather_packed_work)
            _bitscom_timing(
                "lowbit_stream stage=block_all_gather_packed returned "
                f"elapsed_ms={(time.perf_counter() - t_block) * 1000.0:.3f}"
            )

            gathered_scales = [
                torch.empty_like(reduced_scales) for _ in range(world_size)
            ]
            t_launch = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:launch_all_gather_scales"
            ):
                gather_scales_work = dist.all_gather(
                    gathered_scales,
                    reduced_scales,
                    group=group,
                    async_op=True,
                )
            _bitscom_timing(
                "lowbit_stream stage=launch_all_gather_scales returned "
                f"elapsed_ms={(time.perf_counter() - t_launch) * 1000.0:.3f}"
            )
            t_block = time.perf_counter()
            with torch.profiler.record_function(
                "bitscom:_lowbit_stream:block_all_gather_scales"
            ):
                self._block_work_on_current_stream(gather_scales_work)
            _bitscom_timing(
                "lowbit_stream stage=block_all_gather_scales returned "
                f"elapsed_ms={(time.perf_counter() - t_block) * 1000.0:.3f}"
            )

            out_shards = []
            t_unpack_gathered = time.perf_counter()
            for rank_idx in range(world_size):
                fp_shard = unpack_dequantize_tensor_blockwise(
                    gathered_packed[rank_idx],
                    gathered_scales[rank_idx],
                    self.bitwidth,
                    shard_len,
                    block_size=self.block_size,
                    dtype=torch.float32,
                    device=flat.device,
                )
                out_shards.append(fp_shard)
            _bitscom_timing(
                "lowbit_stream stage=unpack_gathered returned "
                f"elapsed_ms={(time.perf_counter() - t_unpack_gathered) * 1000.0:.3f}"
            )

            t_restore = time.perf_counter()
            restored = torch.cat(out_shards, dim=0)[:original_numel]
            tensor.copy_(restored.view_as(tensor).to(dtype=tensor.dtype))
            if post_scale != 1.0:
                tensor.mul_(post_scale)
            _bitscom_timing(
                "lowbit_stream stage=restore_tensor returned "
                f"elapsed_ms={(time.perf_counter() - t_restore) * 1000.0:.3f}"
            )
            t_done_record = time.perf_counter()
            done.record(stream)
            _bitscom_timing(
                "lowbit_stream stage=done_record returned "
                f"elapsed_ms={(time.perf_counter() - t_done_record) * 1000.0:.3f}"
            )
            _bitscom_timing(
                "lowbit_stream stream-body exit "
                f"elapsed_ms={(time.perf_counter() - stream_body_t0) * 1000.0:.3f}"
            )

        t_keepalive = time.perf_counter()
        keepalive = (
            [flat, local_sum, packed_reduced, reduced_scales, restored]
            + send_packed
            + send_scales
            + recv_packed
            + recv_scales
            + gathered_packed
            + gathered_scales
            + out_shards
        )
        _bitscom_timing(
            "lowbit_stream keepalive built "
            f"elapsed_ms={(time.perf_counter() - t_keepalive) * 1000.0:.3f}"
        )
        _bitscom_timing(
            "SUMMARY lowbit_stream exit "
            f"elapsed_ms={(time.perf_counter() - func_t0) * 1000.0:.3f}"
        )
        return _CudaEventWork(done, keepalive=keepalive)

    def _lowbit_allreduce_via_alltoall_into_(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> None:
        flat = tensor.contiguous().view(-1)
        restored = self._lowbit_allreduce_via_alltoall_group(flat, group).view_as(tensor)
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
