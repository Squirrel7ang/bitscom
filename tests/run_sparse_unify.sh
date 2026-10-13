#!/usr/bin/env bash
# 稀疏化 reduce_scatter「精度统一」规则的跨机测试启动脚本（双机双卡）。
#
# 用法（在各自节点上运行，先 master 后 worker，两边参数必须一致）：
#   bash tests/run_sparse_unify.sh 0 'PRI_MODE=0 PRI_BW=4 NONPRI_MODE=1 NONPRI_BW=2'
#   bash tests/run_sparse_unify.sh 1 'PRI_MODE=0 PRI_BW=4 NONPRI_MODE=1 NONPRI_BW=2'
#
# 参数：
#   $1  node_rank（0 master / 1 worker）
#   $2  配置串，形如 'PRI_MODE=.. PRI_BW=.. NONPRI_MODE=.. NONPRI_BW=..'
#       mode: 0=Full 1=Quant 2=Discard
set -euo pipefail

NODE_RANK="${1:-}"
CFG="${2:-}"

MASTER_ADDR="${MASTER_ADDR:-10.31.10.62}"
MASTER_PORT="${MASTER_PORT:-29505}"
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
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> sparse unify rs: node_rank=$NODE_RANK cfg='$CFG' master=$MASTER_ADDR:$MASTER_PORT"

exec env $CFG torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/test_sparse_unify_reduce_scatter.py"
