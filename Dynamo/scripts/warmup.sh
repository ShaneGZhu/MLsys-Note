#!/bin/bash
# warmup：用独立的随机合成 prompt 把每个 worker 的 CUDA graph / kernel 暖起来
# （实测首个落到某 worker 的请求 12.8s，稳态 0.4s；不暖会直接进 TTFT 统计）
#
# 为什么不用 aiperf 自带的 --warmup-*：那组参数 benchmark 脚本一个都没透传，
# 而且 warmup 阶段复用的是同一个 dataset，等于把待测前缀提前灌进 KV cache。
#
# 用法：bash warmup.sh [tag]        # tag 默认 warmup，用于区分产物/日志
#
# 环境变量（有默认值，不写死）：
#   DYN_WORKSPACE / DYNAMO_ROOT / VENV / DYN_BENCH_ROOT / DYN_OUT_DIR / LOG_DIR / DYN_HTTP_PORT
#   PROFILE(glm45air_tp4，用来取 MODEL_NAME / MODEL_PATH)
#   WARMUP_REQUESTS(160) / WARMUP_CONCURRENCY(32) / WARMUP_ISL(200) / WARMUP_OSL(50) / WARMUP_SEED(999)
#
# 请求数按 8 worker 估的：并发 32 × 5 轮，足够让每个 worker 都吃到请求。
# 跑完会打一行 frontend 指标，其中 workers_seen 必须等于 worker 总数，否则有 worker 没暖到。

set -u
TAG=${1:-warmup}

S=$(dirname "$(readlink -f "$0")")
W=${DYN_WORKSPACE:-/root/paddlejob/workspace/env_run/output/zhushengguang}
DYNAMO_ROOT=${DYNAMO_ROOT:-$W/kvc/dynamo}
VENV=${VENV:-$DYNAMO_ROOT/.venv}
BENCH_ROOT=${DYN_BENCH_ROOT:-$W/scripts/dynamo_bench}
OUT=${DYN_OUT_DIR:-$BENCH_ROOT/router_bench}
LOG=${LOG_DIR:-$BENCH_ROOT/logs}
PORT=${DYN_HTTP_PORT:-8000}
PROFILE=${PROFILE:-glm45air_tp4}
export PATH="$VENV/bin:$PATH"          
mkdir -p "$OUT" "$LOG"

# MODEL_NAME / MODEL_PATH 从 profile 取，不在本脚本里写死
# shellcheck source=/dev/null
source "$S/configs/$PROFILE.env"

NW=${WARMUP_REQUESTS:-160}
echo ">>> warmup $TAG: $NW 请求 / 并发 ${WARMUP_CONCURRENCY:-32} / ISL ${WARMUP_ISL:-200} -> $LOG/warmup_$TAG.log"
aiperf profile --model "$MODEL_NAME" --tokenizer "$MODEL_PATH" --url "http://localhost:$PORT" \
    --synthetic-input-tokens-mean "${WARMUP_ISL:-200}" --synthetic-input-tokens-stddev 20 \
    --output-tokens-mean "${WARMUP_OSL:-50}" --output-tokens-stddev 10 \
    --concurrency "${WARMUP_CONCURRENCY:-32}" --request-count "$NW" --num-dataset-entries "$NW" \
    --random-seed "${WARMUP_SEED:-999}" --artifact-dir "$OUT/_warmup_$TAG" \
    --endpoint-type chat --streaming --extra-inputs ignore_eos:true --no-gpu-telemetry \
    -H 'Authorization: Bearer NOT USED' -H 'Accept: text/event-stream' \
    > "$LOG/warmup_$TAG.log" 2>&1
RC=$?
pkill -9 -f "[a]iperf"

"$VENV/bin/python3" - "http://localhost:$PORT/metrics" <<'PY'
import sys, urllib.request as u
t = u.urlopen(sys.argv[1]).read().decode()
g = lambda n: sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.split("{", 1)[0] == n)
P = "dynamo_frontend_"
w = {l for l in t.splitlines() if l.startswith(P + "worker_last_input_sequence_tokens")}
print("reqs=%d  prefix_cache_hit=%.4f  mean_ttft=%.3fs  workers_seen=%d" % (
    g(P + "input_sequence_tokens_count"),
    g(P + "cached_tokens_sum") / max(g(P + "input_sequence_tokens_sum"), 1),
    g(P + "time_to_first_token_seconds_sum") / max(g(P + "time_to_first_token_seconds_count"), 1),
    len(w)))
PY

[ $RC -eq 0 ] || echo "warmup 退出码 $RC，看 $LOG/warmup_$TAG.log"
exit $RC
