import os
import time

import pytest
import torch
import torch.distributed as dist

import bitscom
from bitscom.quantization import (
    compress_tensor,
    decompress_tensor,
    roundtrip_tensor,
    DEFAULT_BLOCK_SIZE
)

from torch.profiler import ProfilerActivity, profile, record_function

COUNT=4

bitscomPG: dist.ProcessGroup
standardPG: dist.ProcessGroup


def initialize():
    global bitscomPG
    global standardPG

    local_rank = int(os.environ["LOCAL_RANK"])

    bitscom.init(
        bitwidth=4,
        error_feedback=False,
        error_feedback_mode="none",
        block_size=DEFAULT_BLOCK_SIZE,
        sparse_enabled=True,
        sparse_projection_rank=4,
        sparse_compression_ratio=0.1,
        sparse_priority_mode=0,
        sparse_priority_quantize_bitwidth=4,
        sparse_non_priority_mode=2,
        sparse_non_priority_quantize_bitwidth=4,
    )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size=dist.get_world_size()
    bitscomPG = dist.new_group(ranks=list(range(world_size)), backend='lowbit')
    standardPG = dist.new_group(ranks=list(range(world_size)), backend='nccl')

    torch.set_default_device(f"cuda:{torch.cuda.current_device()}")


def testBitscom():
    global bitscomPG
    global standardPG
    # 对 FP32 对 4MB 数据通信，
    # 4MB = 4B * 1024 * 1024
    input_list=[
        torch.randn(1024, 1024),
        torch.randn(1024, 1024),
    ]
    output = torch.zeros(1024, 1024)
    start = time.time()
    for i in range(COUNT):
        dist.reduce_scatter(output=output, input_list=input_list, group=bitscomPG)
    end = time.time()
    print(f"[BITSCOM] time: {start-end:.2f}")


def testStandard():
    global bitscomPG
    global standardPG
    # 对 FP32 对 4MB 数据通信，
    # 4MB = 4B * 1024 * 1024
    input_list=[
        torch.randn(1024, 1024),
        torch.randn(1024, 1024),
    ]
    output = torch.zeros(1024, 1024)
    start = time.time()
    for i in range(COUNT):
        dist.reduce_scatter(output=output, input_list=input_list, group=standardPG)
    end = time.time()
    print(f"[STANDARD] time: {start-end:.2f}")


def testMain():
    initialize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function('bitscom'):
            testBitscom()
        with record_function('standard'):
            testStandard()

    if dist.get_rank() == 0:
        prof.export_chrome_trace("./trace.json")
        

if __name__ == '__main__':
    testMain()