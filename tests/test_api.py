from types import SimpleNamespace
import time

import pytest
import torch

from bitscom.api import LowBitGroup


class DummyWork:
    def __init__(self):
        self.wait_called = 0

    def wait(self):
        self.wait_called += 1
        return True


class FutureBackedWork:
    def __init__(self, future):
        self.future = future
        self.wait_called = 0

    def get_future(self):
        return self.future

    def wait(self):
        self.wait_called += 1
        return self.future.wait()


@pytest.fixture
def fake_dist(monkeypatch):
    calls = {}
    default_group = object()
    shared = {}

    def fake_get_default_group():
        return default_group

    def fake_get_rank(group):
        calls["rank_group"] = group
        return 1

    def fake_get_world_size(group):
        calls["world_group"] = group
        return 8

    def fake_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        calls.setdefault("all_to_all", []).append(
            {
                "group": group,
                "async_op": async_op,
                "input_len": len(input_tensor_list),
                "output_len": len(output_tensor_list),
            }
        )
        for out, inp in zip(output_tensor_list, input_tensor_list):
            out.copy_(inp)
        return DummyWork()

    def make_collective(name):
        def _fn(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return DummyWork()

        return _fn

    patch_target = "bitscom.api.dist"
    monkeypatch.setattr(f"{patch_target}.distributed_c10d._get_default_group", fake_get_default_group)
    monkeypatch.setattr(f"{patch_target}.get_rank", fake_get_rank)
    monkeypatch.setattr(f"{patch_target}.get_world_size", fake_get_world_size)
    monkeypatch.setattr(f"{patch_target}.all_reduce", make_collective("all_reduce"))
    monkeypatch.setattr(f"{patch_target}.all_gather", make_collective("all_gather"))
    monkeypatch.setattr(f"{patch_target}.reduce_scatter", make_collective("reduce_scatter"))
    monkeypatch.setattr(f"{patch_target}.broadcast", make_collective("broadcast"))
    monkeypatch.setattr(f"{patch_target}.all_to_all", fake_all_to_all)

    return SimpleNamespace(calls=calls, default_group=default_group, shared=shared)


def test_lowbit_group_uses_default_group(fake_dist):
    group = LowBitGroup(bitwidth=4)
    assert group.pg is fake_dist.default_group
    assert group.rank == 1
    assert group.world_size == 8


@pytest.mark.parametrize("bad_bitwidth", [0, 3, 15, 17])
def test_lowbit_group_rejects_unsupported_bitwidth(bad_bitwidth):
    with pytest.raises(ValueError):
        LowBitGroup(bitwidth=bad_bitwidth, process_group=object())


def test_all_reduce_sync_waits(fake_dist):
    group = LowBitGroup(bitwidth=8, process_group=object())
    t = torch.ones(16, dtype=torch.float32)

    ret = group.all_reduce(t, async_op=False)

    assert ret is None
    assert fake_dist.calls["all_reduce"]["kwargs"]["async_op"] is True


def test_all_reduce_async_returns_work(fake_dist):
    group = LowBitGroup(bitwidth=8, process_group=object())
    t = torch.ones(16, dtype=torch.float32)

    work = group.all_reduce(t, async_op=True)

    assert isinstance(work, DummyWork)


def test_simulate_quantization_changes_values(fake_dist):
    group = LowBitGroup(bitwidth=2, process_group=object(), simulate_quantization=True)
    t = torch.linspace(-1.0, 1.0, 128)
    before = t.clone()

    group.all_reduce(t, async_op=False)

    assert not torch.equal(before, t)


def test_compress_decompress_methods(fake_dist):
    group = LowBitGroup(bitwidth=8, process_group=object())
    x = torch.randn(32)

    packed = group.compress(x)
    y = group.decompress(packed)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_lowbit_allreduce_path_uses_alltoall_and_allgather(fake_dist, monkeypatch):
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)
    monkeypatch.setattr("bitscom.api.dist.get_rank", lambda group: 0)

    gather_calls = {"count": 0}

    def fake_all_gather(output_tensor_list, input_tensor, group=None, async_op=False):
        gather_calls["count"] += 1
        for out in output_tensor_list:
            out.copy_(input_tensor)
        return DummyWork()

    monkeypatch.setattr("bitscom.api.dist.all_gather", fake_all_gather)

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)

    group.all_reduce(t, async_op=False)

    assert "all_reduce" not in fake_dist.calls
    assert "all_to_all" in fake_dist.calls
    assert len(fake_dist.calls["all_to_all"]) == 2
    assert gather_calls["count"] == 2


def test_lowbit_allreduce_async_launches_native_collectives(fake_dist, monkeypatch):
    monkeypatch.delenv("BITSCOM_UNSAFE_FULL_ASYNC_CHAIN", raising=False)
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)
    monkeypatch.setattr("bitscom.api.dist.get_rank", lambda group: 0)

    gather_calls = {"count": 0}

    def fake_all_gather(output_tensor_list, input_tensor, group=None, async_op=False):
        gather_calls["count"] += 1
        for out in output_tensor_list:
            out.copy_(input_tensor)
        return DummyWork()

    monkeypatch.setattr("bitscom.api.dist.all_gather", fake_all_gather)

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)

    work = group.all_reduce(t, async_op=True)

    assert "all_reduce" not in fake_dist.calls
    assert "all_to_all" in fake_dist.calls
    assert len(fake_dist.calls["all_to_all"]) == 2
    assert all(call["async_op"] is True for call in fake_dist.calls["all_to_all"])
    assert gather_calls["count"] == 0

    assert work.wait() is True
    assert gather_calls["count"] == 2
    assert work.is_completed() is True


def test_lowbit_allreduce_async_future_chain_advances_before_wait(fake_dist, monkeypatch):
    monkeypatch.setenv("BITSCOM_UNSAFE_FULL_ASYNC_CHAIN", "1")
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)
    monkeypatch.setattr("bitscom.api.dist.get_rank", lambda group: 0)

    first_stage_futures = []
    gather_calls = {"count": 0}

    def fake_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        for out, inp in zip(output_tensor_list, input_tensor_list):
            out.copy_(inp)
        fut = torch.futures.Future()
        first_stage_futures.append(fut)
        return FutureBackedWork(fut)

    def fake_all_gather(output_tensor_list, input_tensor, group=None, async_op=False):
        gather_calls["count"] += 1
        for out in output_tensor_list:
            out.copy_(input_tensor)
        fut = torch.futures.Future()
        fut.set_result(True)
        return FutureBackedWork(fut)

    monkeypatch.setattr("bitscom.api.dist.all_to_all", fake_all_to_all)
    monkeypatch.setattr("bitscom.api.dist.all_gather", fake_all_gather)

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)

    work = group.all_reduce(t, async_op=True)

    assert len(first_stage_futures) == 2
    assert gather_calls["count"] == 0

    for fut in first_stage_futures:
        fut.set_result(True)

    deadline = time.time() + 2.0
    while gather_calls["count"] < 2 and time.time() < deadline:
        time.sleep(0.01)

    assert gather_calls["count"] == 2
    assert work.wait() is True
    assert work.is_completed() is True


def test_lowbit_allreduce_non_sum_falls_back_to_regular_allreduce(fake_dist, monkeypatch):
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)

    group.all_reduce(t, op=torch.distributed.ReduceOp.MAX, async_op=False)

    assert "all_reduce" in fake_dist.calls


def test_lowbit_allreduce_requires_divisible_numel(fake_dist, monkeypatch):
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

    group.all_reduce(t, async_op=False)
    assert t.shape == (3,)


def test_all_reduce_uses_pipeline_a_when_local_and_inter_groups_passed(fake_dist, monkeypatch):
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)

    called = {"count": 0, "chunk_size": None, "local_quantize": None}

    def fake_pipeline(self, tensor, *, local_group, inter_group, chunk_size, local_quantize):
        called["count"] += 1
        called["chunk_size"] = chunk_size
        called["local_quantize"] = local_quantize
        tensor.add_(1.0)

    monkeypatch.setattr(
        "bitscom.api.LowBitGroup._hierarchical_lowbit_allreduce_pipeline_a",
        fake_pipeline,
    )

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)
    local_group = object()
    inter_group = object()

    group.all_reduce(
        t,
        async_op=False,
        local_group=local_group,
        inter_group=inter_group,
        chunk_size=32,
    )

    assert called["count"] == 1
    assert called["chunk_size"] == 32
    assert called["local_quantize"] is False
    assert torch.allclose(t, torch.tensor([1.0, 2.0]))


def test_all_reduce_local_group_only_uses_full_precision_allreduce(fake_dist, monkeypatch):
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)

    called = {"local_allreduce": 0, "pipeline": 0}

    def fake_local_allreduce(self, flat, group):
        called["local_allreduce"] += 1
        return flat + 2.0

    def fake_pipeline(self, tensor, *, local_group, inter_group, chunk_size, local_quantize):
        called["pipeline"] += 1
        tensor.add_(999.0)

    monkeypatch.setattr(
        "bitscom.api.LowBitGroup._lowbit_allreduce_via_alltoall_group",
        fake_local_allreduce,
    )

    monkeypatch.setattr(
        "bitscom.api.LowBitGroup._hierarchical_lowbit_allreduce_pipeline_a",
        fake_pipeline,
    )

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)
    local_group = object()

    group.all_reduce(
        t,
        async_op=False,
        local_group=local_group,
        inter_group=None,
        chunk_size=16,
    )

    assert called["local_allreduce"] == 0
    assert called["pipeline"] == 0
    assert "all_reduce" in fake_dist.calls
    assert fake_dist.calls["all_reduce"]["kwargs"]["group"] is local_group


def test_all_reduce_local_group_only_quantized_path(fake_dist, monkeypatch):
    monkeypatch.setattr("bitscom.api.dist.get_world_size", lambda group: 2)

    called = {"local_allreduce": 0, "pipeline": 0}

    def fake_local_allreduce(self, flat, group):
        called["local_allreduce"] += 1
        return flat + 2.0

    def fake_pipeline(self, tensor, *, local_group, inter_group, chunk_size, local_quantize):
        called["pipeline"] += 1

    monkeypatch.setattr(
        "bitscom.api.LowBitGroup._lowbit_allreduce_via_alltoall_group",
        fake_local_allreduce,
    )
    monkeypatch.setattr(
        "bitscom.api.LowBitGroup._hierarchical_lowbit_allreduce_pipeline_a",
        fake_pipeline,
    )

    group = LowBitGroup(bitwidth=4, process_group=object())
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)
    local_group = object()

    group.all_reduce(
        t,
        async_op=False,
        local_group=local_group,
        inter_group=None,
        chunk_size=16,
        local_quantize=True,
    )

    assert called["local_allreduce"] == 1
    assert called["pipeline"] == 0
    assert torch.allclose(t, torch.tensor([2.0, 3.0]))


def test_dual_stream_guard_single_node_uses_single_stream(fake_dist):
    group = LowBitGroup(bitwidth=4, process_group=object())
    assert group._should_use_dual_stream_pipeline(
        is_cuda=True,
        local_size=8,
        global_size=8,
    ) is False


def test_dual_stream_guard_multi_node_uses_dual_stream(fake_dist):
    group = LowBitGroup(bitwidth=4, process_group=object())
    assert group._should_use_dual_stream_pipeline(
        is_cuda=True,
        local_size=4,
        global_size=8,
    ) is True
