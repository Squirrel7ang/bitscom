#!/usr/bin/env bash
# 双机 32 卡（每机 16 卡）跨节点两两一组共 16 组 reduce_scatter 性能对比。
#
# 用法（在各自节点上运行，先 master 后 worker）：
#   bash tests/run_sparse_rs_16group.sh 0    # master 节点 10.31.10.62 上执行
#   bash tests/run_sparse_rs_16group.sh 1    # worker 节点 10.31.10.210 上执行
#
# 参数：节点编号 0 / 1（不做校验，直接透传给 torchrun 的 --node_rank）。
#   每组由 u62 与 u210 同 local_rank 的两张卡组成，各做 4 次 reduce_scatter，
#   对比纯稀疏(10%) bitscom 与纯 NCCL，master 上打印每组耗时与加速比。
#
# 可选环境变量：
#   NPROC_PER_NODE      默认 16（每机卡数，凑成 16 组）
#   MASTER_ADDR         默认 10.31.10.62
#   MASTER_PORT         默认 29502
#   NCCL_SOCKET_IFNAME  直连网卡名（默认 ens1f0，即 10.31.10.x 的 TCP 网卡）
set -euo pipefail

NODE_RANK="${1:-}"

MASTER_ADDR="${MASTER_ADDR:-10.31.10.62}"
MASTER_PORT="${MASTER_PORT:-29502}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# corex 环境兜底（非交互 ssh 不加载 ~/.bashrc，LD_LIBRARY_PATH 为空时 NCCL 段错误）
COREX_PATH="${COREX_PATH:-/usr/local/corex-4.4.0}"
if [[ -d "$COREX_PATH/lib64" ]]; then
    export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/corex/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_HOME="${CUDA_HOME:-$COREX_PATH}"
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 16-group sparse rs: node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT nproc=$NPROC_PER_NODE"

exec torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/test_sparse_rs_16group.py"
