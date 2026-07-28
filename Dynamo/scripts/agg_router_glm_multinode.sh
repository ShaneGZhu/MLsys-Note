#!/bin/bash
# 多节点聚合部署：hostfile 中每台机器上按 GPUS_PER_WORKER 切卡组起 worker + node0 起 1 个 frontend
# 用法：bash agg_router_glm_multinode.sh [--profile <name>] [--approx] [--router-mode <mode>] [--no-clean]
#
# --profile      : configs/<name>.env，模型与并行配置（默认 glm45air_tp4）
#                  例：--profile glm45air_dp8ep8 → 整机 8 卡一个 DP-attention + EP worker
# --approx       : 近似 KV 路由（worker 不发布 KV events，frontend 加 --no-kv-events），消融用
# --router-mode  : 路由模式，默认 kv，可选 round-robin / least-loaded / random
# --no-clean     : 启动前不清理残留进程（默认会清理，防止端口/显存冲突）
#
# 环境变量：
#   DYN_BENCH_ROOT: 共享根目录，默认 $W/scripts/dynamo_bench（hostfile / 日志 / discovery / 实验产物都在这里）
#   HOSTFILE      : worker 节点列表，默认 $BENCH_ROOT/hostfile_4node（本 job 只用后 4 台机器）
#   LOG_DIR       : 日志目录，默认 $BENCH_ROOT/logs
#   DYN_FILE_KV   : file discovery 目录，默认 $BENCH_ROOT/store_kv
#   DYN_HTTP_PORT : frontend HTTP 端口，默认 8000

set -u

# -----------------------------------------------------------------------
# 路径配置
# 注意：/tmp 是**节点本地**目录，mpirun 拉起的远端进程读不到，
#       所有需要跨节点共享的文件（脚本 / profile / hostfile / discovery / 日志）必须放共享路径
#       BENCH_ROOT 默认落在 /root/paddlejob/inference-public/zhushengguang（软链接为 workspace/env_run/output/...），
#       它跨节点共享且跨 job 持久；/root/paddlejob/gpfsspace 虽然也共享，但是 job 级空间，job 结束即失效
# -----------------------------------------------------------------------
DYNAMO_ROOT=/root/paddlejob/workspace/env_run/output/zhushengguang/kvc/dynamo
VENV=$DYNAMO_ROOT/.venv
BENCH_ROOT=${DYN_BENCH_ROOT:-/root/paddlejob/workspace/env_run/output/zhushengguang/scripts/dynamo_bench}
HOSTFILE=${HOSTFILE:-$BENCH_ROOT/hostfile_4node}
HTTP_PORT=${DYN_HTTP_PORT:-8000}
LOG_DIR=${LOG_DIR:-$BENCH_ROOT/logs}
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

export PATH="$VENV/bin:$PATH"
# file discovery 共享目录：所有节点通过共享存储访问，frontend 和 worker 注册到同一目录
export DYN_FILE_KV=${DYN_FILE_KV:-$BENCH_ROOT/store_kv}

mkdir -p "$BENCH_ROOT" "$LOG_DIR"

# hostfile 缺省生成：取 job hostfile 的后 4 台（node0 只跑 frontend）
if [ ! -f "$HOSTFILE" ]; then
    tail -4 /root/paddlejob/workspace/hostfile > "$HOSTFILE"
fi

# 参数初始化
APPROX_MODE=false
ROUTER_MODE=kv
DO_CLEAN=true
WORKERS_ONLY=false
PROFILE=${PROFILE:-glm45air_tp4}

# bash 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)      PROFILE=$2; shift 2 ;;
        --approx)       APPROX_MODE=true; shift ;;
        --router-mode)  ROUTER_MODE=$2; shift 2 ;;
        --no-clean)     DO_CLEAN=false; shift ;;
        --workers-only) WORKERS_ONLY=true; DO_CLEAN=false; shift ;;
        -h|--help)
            echo "Usage: $0 [--profile <name>] [--approx] [--router-mode {kv|round-robin|least-loaded|random}] [--no-clean] [--workers-only]"
            echo "  --workers-only : 不重启 frontend、不清理已有 worker，只补起 worker"
            echo "  可用 profile   : $(cd "$SCRIPT_DIR/configs" && ls *.env | sed 's/\.env//' | tr '\n' ' ')"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 输入sglang backend 需要的参数，包括模型名，数据路径等，最后传入launch_worker.sh
PROFILE_FILE="$SCRIPT_DIR/configs/$PROFILE.env"
[ -f "$PROFILE_FILE" ] || { echo "profile 不存在: $PROFILE_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$PROFILE_FILE"

echo "=========================================="
echo "多节点 Dynamo 启动"
echo "  Profile     : $PROFILE ($MODEL_NAME, ${GPUS_PER_WORKER} GPU/worker)"
echo "  Engine args : ${ENGINE_ARGS[*]}"
echo "  Router mode : $ROUTER_MODE"
echo "  Approx mode : $APPROX_MODE"
echo "  HTTP port   : $HTTP_PORT"
echo "  HOSTFILE    : $HOSTFILE"
echo "  DYN_FILE_KV : $DYN_FILE_KV"
echo "  LOG_DIR     : $LOG_DIR"
echo "=========================================="

# -----------------------------------------------------------------------
# 清理：残留 worker 会占显存 + 占端口，且旧注册信息会让 frontend 路由到死实例
# -----------------------------------------------------------------------
if [ "$DO_CLEAN" = true ]; then
    HOSTFILE="$HOSTFILE" bash "$SCRIPT_DIR/stop_all.sh" "$HOSTFILE"
    # 只有全量重启才清 discovery 目录：--no-clean 是"补起 worker"模式，
    # 此时清目录会让已在跑的 worker 从 frontend 视图里消失
    rm -rf "$DYN_FILE_KV"
    mkdir -p "$DYN_FILE_KV"
    rm -f "$LOG_DIR"/worker_*.log
fi
mkdir -p "$DYN_FILE_KV"

source "$VENV/bin/activate"

# -----------------------------------------------------------------------
# frontend（node0）
# --discovery-backend file : 通过 GPFS 共享目录做服务发现，免 etcd
# --request-plane tcp      : 请求直连 worker，免 NATS
# --router-mode            : 切对照组只改这一项并重启 frontend，worker 不动
# --model-path             : 给本地路径，避免 frontend 去 HF 拉 config
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

FRONTEND_PID=""
if [ "$WORKERS_ONLY" = false ]; then
    echo ">>> 启动 frontend (router-mode=$ROUTER_MODE) -> $LOG_DIR/frontend.log"
    "$VENV/bin/python3" -m dynamo.frontend "${FRONTEND_ARGS[@]}" > "$LOG_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
else
    echo ">>> --workers-only：跳过 frontend，只补起 worker"
fi

# -----------------------------------------------------------------------
# 逐节点拉起 worker。端口与卡组由 launch_worker.sh 自行探测（共享机器上必须动态探测）
# 注意：host 先读进数组——若在 while read < hostfile 循环里调 mpirun，
#       mpirun 会吃掉 stdin 剩余的 host 行，只启动第一台机器
# -----------------------------------------------------------------------
mapfile -t HOSTS < <(awk '{print $1}' "$HOSTFILE" | grep -v '^$')
LAUNCH_WORKER="$SCRIPT_DIR/launch_worker.sh"

for HOST in "${HOSTS[@]}"; do
    echo ">>> 启动 $HOST"
    mpirun -n 1 --allow-run-as-root -H "$HOST" \
        -x VENV="$VENV" \
        -x PROFILE_FILE="$PROFILE_FILE" \
        -x DYN_FILE_KV="$DYN_FILE_KV" \
        -x APPROX_MODE="$APPROX_MODE" \
        -x LOG_DIR="$LOG_DIR" \
        -x SKIP_RUNNING="$WORKERS_ONLY" \
        bash "$LAUNCH_WORKER" < /dev/null > "$LOG_DIR/mpirun_${HOST}.log" 2>&1 &
done

echo ""
echo "worker 已在后台拉起（每节点 worker 数 = 节点卡数 / $GPUS_PER_WORKER，见 $LOG_DIR/mpirun_*.log 的 LAUNCHED 行）"
echo "模型加载 3-15 分钟（GPFS 读权重有竞争）。检查注册数量："
echo "  bash $SCRIPT_DIR/check_ready.sh"
echo "日志：tail -f $LOG_DIR/worker_<host>_<slot>.log ; tail -f $LOG_DIR/frontend.log"
echo ""

if [ -n "$FRONTEND_PID" ]; then
    wait "$FRONTEND_PID"
else
    wait
fi
