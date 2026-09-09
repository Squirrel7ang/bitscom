#!/usr/bin/env bash
# 跨机 bitscom 性能测试启动脚本。
#
# 用法（在各自节点上运行）：
#   bash tests/run_cross_node.sh 0    # master 节点 10.31.10.62 上执行
#   bash tests/run_cross_node.sh 1    # worker 节点 10.31.10.210 上执行
#
# 先跑 node 0（master 起 rendezvous 服务端），再跑 node 1。
#
# 可选环境变量：
#   MASTER_ADDR         默认 10.31.10.62
#   MASTER_PORT         默认 29500
#   NPROC_PER_NODE      默认 8
#   NCCL_SOCKET_IFNAME  直连网卡名（直连网卡非默认路由网卡时必填，如 eth1）
set -euo pipefail

# ---- 参数：节点编号 0 / 1 ----
NODE_RANK="${1:-}"
if [[ -z "$NODE_RANK" ]]; then
    echo "用法: bash $0 <0|1>" >&2
    exit 2
fi
if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
    echo "错误: 节点编号只能是 0 或 1，收到 '$NODE_RANK'" >&2
    exit 2
fi

# ---- 可覆盖配置 ----
MASTER_ADDR="${MASTER_ADDR:-10.31.10.62}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# ---- 脚本目录定位到 tests，保证从任意 cwd 都能跑 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 启动跨机测试: node_rank=$NODE_RANK  master=$MASTER_ADDR:$MASTER_PORT  nproc_per_node=$NPROC_PER_NODE"

exec torchrun \
    --nnodes=2 \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/test_perf_cross_node.py"
