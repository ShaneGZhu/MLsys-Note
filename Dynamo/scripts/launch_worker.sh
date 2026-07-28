#!/bin/bash
# 单机 worker 启动脚本，由 agg_router_glm_multinode.sh 通过 mpirun 调用
#
# 环境变量（由主脚本通过 mpirun -x 传入）：
#   VENV          : venv 路径
#   PROFILE_FILE  : 模型/并行配置 profile（configs/*.env），必须在跨节点可读路径上
#   DYN_FILE_KV   : file discovery 共享目录（必须在跨节点共享存储上）
#   APPROX_MODE   : true/false，是否跳过 kv-events-config
#   LOG_DIR       : 日志目录（跨节点共享存储）
#
# profile 提供的变量：
#   MODEL_PATH / MODEL_NAME / CHAT_TEMPLATE
#   GPUS_PER_WORKER : 每个 worker 占几张卡（= tp_size × pp_size；dp-attention 复用 TP group，不额外占卡）
#   ENGINE_ARGS[]   : 原样透传给 dynamo.sglang 的引擎参数（--tp-size / --dp-size / --enable-dp-attention /
#                     --ep-size / --moe-a2a-backend / --page-size / --mem-fraction-static ...）
#
# 关键设计（踩过的坑，勿删）：
# 1) 整机按 GPUS_PER_WORKER 顺序切组（8 卡 + 4 = 0-3 / 4-7）。nvidia-smi 里常看到别人的
#    进程占着显存，但不影响本容器起服（实测直接起就能起来），所以不做空闲门控。
# 2) 容器与宿主共享网络命名空间：SGLang 默认用随机端口做 torch.distributed
#    rendezvous（nccl_port = get_free_port()），同机多 worker 并发启动 + 其它租户
#    进程都可能撞端口，报 EADDRINUSE / DistNetworkError。这里显式探测空闲端口并用
#    --nccl-port 钉死，同时错开 5s 启动。
# 3) --port 必须每个 worker 不同：开 --enable-dp-attention 后 PortArgs.init_new() 不再用
#    IPC，而是从 server_args.port + 233 **确定性派生** 5~6 个连续端口（多节点各自独立派生，
#    所以 SGLang 不做可用性搜索）。同机两个 dp-attention worker 若共用默认 port=30000，
#    派生端口会直接撞上。这里按 slot 间隔 1000 分配，并预先校验整个派生区间可 bind。
# 4) ZMQ KV-events 端口无需与 frontend 约定：worker 会把 tcp://*:PORT 解析成本机 IP
#    后注册到 file discovery（$DYN_FILE_KV/v1/event_sources），frontend 自动发现，
#    所以端口可以动态分配。dp-attention 下 SGLang 内部按 base_port + dp_rank 偏移，
#    因此同机各 worker 的 ZMQ base 之间要留出 ≥ dp_size 的间隔。

set -u
LOG_DIR=${LOG_DIR:-/root/paddlejob/workspace/env_run/output/zhushengguang/scripts/dynamo_bench/logs}
mkdir -p "$LOG_DIR"
HOST=$(hostname -s)

source "$VENV/bin/activate"
export DYN_FILE_KV="$DYN_FILE_KV"

# shellcheck source=/dev/null
source "$PROFILE_FILE"
GPUS_PER_WORKER=${GPUS_PER_WORKER:-4}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}

# tp_size × pp_size 必须等于每 worker 卡数，否则 SGLang 会拿不到卡直接被 OOM Kill
_tp=1; _pp=1; _dp=1; _prev=""
for a in "${ENGINE_ARGS[@]}"; do
    case "$_prev" in
        --tp-size|--tp|--tensor-parallel-size) _tp=$a ;;
        --pp-size|--pipeline-parallel-size)    _pp=$a ;;
        --dp-size|--dp|--data-parallel-size)   _dp=$a ;;
    esac
    _prev=$a
done
if [ $((_tp * _pp)) -ne "$GPUS_PER_WORKER" ]; then
    echo "[$HOST] ERROR: tp($_tp) × pp($_pp) != GPUS_PER_WORKER($GPUS_PER_WORKER)，请检查 $PROFILE_FILE"
    exit 1
fi

# ---------------------------------------------------------------------------
# 端口探测：从 start 开始，找第一个「自身 + 所有给定偏移」都能 bind 的基准端口
# 用法：pick_port <start> [offset ...]
# ---------------------------------------------------------------------------
pick_port() {
    "$VENV/bin/python3" - "$@" <<'PY'
import socket, sys
start = int(sys.argv[1])
offsets = [0] + [int(x) for x in sys.argv[2:]]
p = start
while p + max(offsets) < 65500:
    socks = []
    try:
        for off in offsets:
            s = socket.socket()
            s.bind(("0.0.0.0", p + off))
            socks.append(s)
        print(p)
        break
    except OSError:
        p += 1
    finally:
        for s in socks:
            s.close()
PY
}

# ---------------------------------------------------------------------------
# 卡组划分：整机按 GPUS_PER_WORKER 顺序切组，每组起 1 个 worker。
# nvidia-smi 里看到的"他人占用显存"不影响本容器分配（cgpu 显存虚拟化，
# 容器有独立配额），实测直接起任务即可，不需要等卡空闲，所以这里不做空闲门控，
# 只把显存现状打到日志里便于事后排查。
# ---------------------------------------------------------------------------
echo "[$HOST] GPU mem.used = [$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')] MiB"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[$HOST] NGPU=$NGPU GPUS_PER_WORKER=$GPUS_PER_WORKER tp=$_tp pp=$_pp"
# 注意：GROUPS 是 bash 特殊变量（当前用户 group id 数组），赋值会被静默忽略，勿用作变量名
GPU_GROUPS=()
g=0
while [ $((g + GPUS_PER_WORKER)) -le "$NGPU" ]; do
    devs=""
    i=$g
    while [ "$i" -lt $((g + GPUS_PER_WORKER)) ]; do
        devs="${devs:+$devs,}$i"
        i=$((i + 1))
    done
    GPU_GROUPS+=("$devs")
    g=$((g + GPUS_PER_WORKER))
done

if [ ${#GPU_GROUPS[@]} -eq 0 ]; then
    echo "[$HOST] LAUNCHED slots=0（GPU 数 $NGPU < GPUS_PER_WORKER $GPUS_PER_WORKER）"
    exit 0
fi

SLOT=0
for DEVS in "${GPU_GROUPS[@]}"; do
    G=${DEVS%%,*}                       # 该组的首卡 index，用于日志命名，避免"补起 worker"时覆盖旧日志
    # 补起模式（SKIP_RUNNING=true）：该卡组已有活着的 worker 就跳过。
    # CUDA_VISIBLE_DEVICES 只在环境里，不在 cmdline，所以查 /proc/<pid>/environ
    if [ "${SKIP_RUNNING:-false}" = true ]; then
        RUNNING=false
        for pid in $(pgrep -f "dynamo.sglang" 2>/dev/null); do
            if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$DEVS"; then
                RUNNING=true
                break
            fi
        done
        if [ "$RUNNING" = true ]; then
            echo "[$HOST] SKIP worker-gpu$G GPU=$DEVS（已有存活 worker）"
            SLOT=$((SLOT + 1))
            continue
        fi
    fi
    # 端口 band 说明：25557 / 30000 / 34500 都是 Dynamo/SGLang 的**默认值**，
    # 共享宿主网络命名空间下极易被其它租户的同类进程抢占（实测 dp8 加载模型的 2 分钟里
    # 25557+7 被抢，worker 在 bind KV-events ZMQ 时 EADDRINUSE 直接死掉），
    # 所以这里统一挪到 4xxxx 的非默认段。
    _span=$((_dp > 8 ? _dp : 8))                                  # ZMQ 需要校验的连续端口数（base + dp_rank）
    _zmq_offs=$(seq 1 $((_span - 1)) | tr '\n' ' ')
    SYS_PORT=$(pick_port $((18081 + SLOT * 7)))
    # shellcheck disable=SC2086
    ZMQ_PORT=$(pick_port $((45000 + SLOT * 256)) $_zmq_offs)      # dp-attention 下 SGLang 按 base+dp_rank 偏移
    NCCL_PORT=$(pick_port $((47000 + SLOT * 7)))
    SGL_PORT=$(pick_port $((41000 + SLOT * 1000)) 233 234 235 236 237 238 239)
    LOG="$LOG_DIR/worker_${HOST}_gpu${G}.log"

    KV_ARG=()
    if [ "${APPROX_MODE}" != "true" ]; then
        KV_ARG=(--kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$ZMQ_PORT\"}")
    fi
    TPL_ARG=()
    [ -n "$CHAT_TEMPLATE" ] && TPL_ARG=(--chat-template "$CHAT_TEMPLATE")

    echo "[$HOST] 启动 worker-gpu$G GPU=$DEVS sys=$SYS_PORT zmq=$ZMQ_PORT nccl=$NCCL_PORT sgl=$SGL_PORT log=$LOG"

    DYN_SYSTEM_PORT=$SYS_PORT \
    CUDA_VISIBLE_DEVICES=$DEVS \
    "$VENV/bin/python3" -m dynamo.sglang \
      --model-path "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --port "$SGL_PORT" \
      --nccl-port "$NCCL_PORT" \
      --discovery-backend file \
      --request-plane tcp \
      --event-plane zmq \
      "${TPL_ARG[@]}" \
      "${ENGINE_ARGS[@]}" \
      "${KV_ARG[@]}" \
      > "$LOG" 2>&1 &

    SLOT=$((SLOT + 1))
    sleep 5   # 错开启动，进一步降低端口/显存竞争
done

echo "[$HOST] LAUNCHED slots=$SLOT"
wait
