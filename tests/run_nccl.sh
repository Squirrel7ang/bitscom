#!/usr/bin/env bash
# 纯 NCCL reduce_scatter 跨机测试启动脚本（双机双卡）。
#
# 用法（在各自节点上运行，先 master 后 worker）：
#   bash tests/run_nccl.sh 0    # master 节点 10.31.10.62 上执行
#   bash tests/run_nccl.sh 1    # worker 节点 10.31.10.210 上执行
#
# 参数：节点编号 0 / 1（不做校验，直接透传给 torchrun 的 --node_rank）。
#
# 可选环境变量：
#   MASTER_ADDR         默认 10.31.10.62
#   MASTER_PORT         默认 29500
#   NPROC_PER_NODE      默认 2（双卡）
#   NCCL_SOCKET_IFNAME  直连网卡名（直连网卡非默认路由网卡时必填，如 eth1）
set -euo pipefail

# ---- 参数：节点编号 0 / 1 ----
NODE_RANK="${1:-}"

# ---- 可覆盖配置 ----
MASTER_ADDR="${MASTER_ADDR:-10.31.10.62}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# ---- corex 运行环境兜底 ----
# 非交互式 ssh / 调度器不会加载 ~/.bashrc，LD_LIBRARY_PATH 为空时 NCCL 会在
# 第一次通信直接段错误（exitcode -11/139），且不打印任何 NCCL INFO 日志。
# 这里按登录 shell 的配置补齐，保证手工跑和 ssh 跑行为一致。
COREX_PATH="${COREX_PATH:-/usr/local/corex-4.4.0}"
if [[ -d "$COREX_PATH/lib64" ]]; then
    export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/corex/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CPATH="$COREX_PATH/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$COREX_PATH/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CUDA_HOME="${CUDA_HOME:-$COREX_PATH}"
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE

# ---- 脚本目录定位到 tests，保证从任意 cwd 都能跑 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 启动纯 NCCL reduce_scatter 测试: node_rank=$NODE_RANK  master=$MASTER_ADDR:$MASTER_PORT  nproc_per_node=$NPROC_PER_NODE"

exec torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/test_nccl_reduce_scatter.py"
