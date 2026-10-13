#!/usr/bin/env bash
# 跨机量化 reduce_scatter 基准启动脚本（双机双卡，先 master 后 worker）。
#
# 用法：
#   bash tests/run_bench_quant.sh 0     # master（10.31.10.62）
#   bash tests/run_bench_quant.sh 1     # worker（10.31.10.210）
#
# 可覆盖环境变量：MASTER_ADDR / MASTER_PORT / BENCH_ELEMS / BENCH_ITERS /
#   NCCL_SOCKET_IFNAME / NCCL_IB_DISABLE
set -euo pipefail

NODE_RANK="${1:-}"
MASTER_ADDR="${MASTER_ADDR:-10.31.10.62}"
MASTER_PORT="${MASTER_PORT:-29510}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

COREX_PATH="${COREX_PATH:-/usr/local/corex-4.4.0}"
if [[ -d "$COREX_PATH/lib64" ]]; then
    export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/corex/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_HOME="${CUDA_HOME:-$COREX_PATH}"
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_SOCKET_IFNAME NCCL_IB_DISABLE
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> bench_quant: node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT elems=${BENCH_ELEMS:-4194304} iters=${BENCH_ITERS:-32}"

exec torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/bench_quant_comm.py"
