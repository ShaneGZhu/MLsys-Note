#!/bin/bash
# 多节点聚合部署：后 4 机 × 每机 2 worker（TP4）= 8 worker + 1 frontend
# 用法：bash agg_router_glm_multinode.sh [--approx] [--router-mode <mode>]
#
# --approx       : 近似 KV 路由（不发布 KV events），用于消融实验
# --router-mode  : 路由模式，默认 kv，可选 round-robin / least-loaded / random

set -e

# -----------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------
VENV=/root/paddlejob/workspace/env_run/output/zhushengguang/kvc/dynamo/.venv
DYNAMO_EXAMPLES=/root/paddlejob/workspace/env_run/output/zhushengguang/kvc/dynamo/examples
HOSTFILE=/root/paddlejob/workspace/hostfile
MODEL_NAME="zai-org/GLM-4.5-Air"
MODEL_PATH=/root/paddlejob/workspace/env_run/output/zhushengguang/models/GLM-4.5-Air
CHAT_TEMPLATE=$MODEL_PATH/chat_template.jinja
HTTP_PORT=${DYN_HTTP_PORT:-8000}

# file discovery 共享目录：所有节点通过 GPFS 共享，frontend 和 worker 注册到同一目录
export DYN_FILE_KV=/root/paddlejob/gpfsspace/dynamo_store_kv

# -----------------------------------------------------------------------
# launch_utils 提供 wait_any_exit（监控所有后台进程，任一退出则清理全部）
# -----------------------------------------------------------------------
source "$DYNAMO_EXAMPLES/common/launch_utils.sh"

# -----------------------------------------------------------------------
# 命令行参数解析
# --approx       : 近似路由模式，worker 不发 KV events，frontend 加 --no-kv-events
# --router-mode  : 指定路由模式（默认 kv）
# -----------------------------------------------------------------------
APPROX_MODE=false
ROUTER_MODE=kv
while [[ $# -gt 0 ]]; do
    case $1 in
        --approx)
            APPROX_MODE=true
            shift
            ;;
        --router-mode)
            ROUTER_MODE=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--approx] [--router-mode {kv|round-robin|least-loaded|random}]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "多节点 Dynamo 启动"
echo "  Router mode : $ROUTER_MODE"
echo "  Approx mode : $APPROX_MODE"
echo "  HTTP port   : $HTTP_PORT"
echo "  DYN_FILE_KV : $DYN_FILE_KV"
echo "=========================================="

# -----------------------------------------------------------------------
# 清理旧的 file discovery 注册信息，防止旧 worker checksum 冲突
# -----------------------------------------------------------------------
rm -rf "$DYN_FILE_KV"
mkdir -p "$DYN_FILE_KV"

# -----------------------------------------------------------------------
# 激活 venv（frontend 在本机运行，直接 source）
# -----------------------------------------------------------------------
source "$VENV/bin/activate"

# -----------------------------------------------------------------------
# 启动 frontend（本机，rank-0）
# --discovery-backend file : 通过 GPFS 共享目录做服务发现
# --request-plane tcp      : 请求直连 worker，无需 NATS
# --router-mode            : 由参数控制，切对照组只需改此参数重启 frontend
# --model-path             : 提供本地路径，避免 frontend 尝试从 HF 拉 model config
# -----------------------------------------------------------------------
FRONTEND_ARGS=(
    --discovery-backend file
    --request-plane tcp
    --router-mode "$ROUTER_MODE"
    --http-port "$HTTP_PORT"
    --model-name "$MODEL_NAME"
    --model-path "$MODEL_PATH"
)
if [ "$APPROX_MODE" = true ]; then
    FRONTEND_ARGS+=(--no-kv-events)
fi

echo ">>> 启动 frontend (router-mode=$ROUTER_MODE)..."
python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}" > /tmp/frontend.log 2>&1 &

# launch_worker.sh 路径（与本脚本同目录）
LAUNCH_WORKER="$(dirname "$(readlink -f "$0")")/launch_worker.sh"

# -----------------------------------------------------------------------
# 遍历 hostfile，对每台机器用 mpirun 调用 launch_worker.sh
# 端口分配规则（NODE_IDX 从 0 开始）：
#   DYN_SYSTEM_PORT : 8081 + NODE_IDX*2  / 8082 + NODE_IDX*2
#   ZMQ port        : 5557 + NODE_IDX*2  / 5558 + NODE_IDX*2
# 所有配置通过 mpirun -x 传入环境变量，launch_worker.sh 直接读取
# -----------------------------------------------------------------------
NODE_IDX=0
while IFS=' ' read -r HOST _; do
    SYS_PORT0=$((8081 + NODE_IDX * 2))
    SYS_PORT1=$((8082 + NODE_IDX * 2))
    ZMQ_PORT0=$((5557 + NODE_IDX * 2))
    ZMQ_PORT1=$((5558 + NODE_IDX * 2))

    echo ">>> 启动 $HOST: sys_port=$SYS_PORT0/$SYS_PORT1, zmq=$ZMQ_PORT0/$ZMQ_PORT1"

    mpirun -n 1 --allow-run-as-root -H "$HOST" \
        -x VENV="$VENV" \
        -x MODEL_PATH="$MODEL_PATH" \
        -x MODEL_NAME="$MODEL_NAME" \
        -x CHAT_TEMPLATE="$CHAT_TEMPLATE" \
        -x DYN_FILE_KV="$DYN_FILE_KV" \
        -x APPROX_MODE="$APPROX_MODE" \
        -x SYS_PORT0="$SYS_PORT0" \
        -x SYS_PORT1="$SYS_PORT1" \
        -x ZMQ_PORT0="$ZMQ_PORT0" \
        -x ZMQ_PORT1="$ZMQ_PORT1" \
        bash "$LAUNCH_WORKER" &

    NODE_IDX=$((NODE_IDX + 1))
done < "$HOSTFILE"

echo ""
echo "所有 worker 已在后台启动（共 $((NODE_IDX * 2)) 个）"
echo "等待约 2 分钟加载完成后检查注册数量："
echo "  ls $DYN_FILE_KV | wc -l   # 应为 $((NODE_IDX * 2))"
echo ""
echo "查看 worker 日志示例："
echo "  tail -f /tmp/worker_<HOST>_0.log"
echo "查看 frontend 日志："
echo "  tail -f /tmp/frontend.log"
echo ""

# 等待任意子进程退出（worker 崩溃或 Ctrl+C），EXIT trap 会清理所有子进程
wait_any_exit
