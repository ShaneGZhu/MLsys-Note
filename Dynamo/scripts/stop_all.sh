#!/bin/bash
# 停止所有 Dynamo frontend / worker 进程（node0 + hostfile 中所有 worker 节点）
# 用途：
#   1. 实验组之间清 KV cache（重启 worker 是清 cache 的手段）
#   2. 启动前清理残留进程，避免端口/显存冲突
#
# 用法：bash stop_all.sh [HOSTFILE]

BENCH_ROOT=${DYN_BENCH_ROOT:-/root/paddlejob/workspace/env_run/output/zhushengguang/scripts/dynamo_bench}
HOSTFILE=${1:-${HOSTFILE:-$BENCH_ROOT/hostfile_4node}}

echo ">>> 停止 node0 上的 frontend 与 launcher"
pkill -f "dynamo.frontend" 2>/dev/null

# 计算自己的祖先 PID 链：本脚本会被 agg_router_glm_multinode.sh 调用，
# 若直接 pkill -f agg_router_glm_multinode.sh 会把调用者（以及自己）一起杀掉
ANCESTORS=" $$ "
p=$$
while [ "$p" != "1" ] && [ -r /proc/$p/stat ]; do
    p=$(awk '{print $4}' /proc/$p/stat)
    ANCESTORS="$ANCESTORS$p "
done

for pid in $(pgrep -f "agg_router_glm_multinode.sh|launch_worker.sh" 2>/dev/null); do
    case "$ANCESTORS" in *" $pid "*) continue ;; esac
    kill "$pid" 2>/dev/null
done

# 远端清理脚本必须放在跨节点共享目录：/tmp 是节点本地的，mpirun 到远端读不到
REMOTE_KILL=$BENCH_ROOT/_remote_kill.sh
mkdir -p "$(dirname "$REMOTE_KILL")"
cat > "$REMOTE_KILL" <<'EOF'
pkill -f "dynamo.sglang" 2>/dev/null
pkill -f "sglang::" 2>/dev/null
sleep 5
pkill -9 -f "dynamo.sglang" 2>/dev/null
pkill -9 -f "sglang::" 2>/dev/null
sleep 3
echo "CLEANED host=$(hostname) remain=$(ps -ef | grep -E 'dynamo.sglang|sglang::' | grep -v grep | wc -l) gpumem=[$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')]"
EOF

# 先把 host 读进数组：如果在 while read < hostfile 的循环体里调 mpirun，
# mpirun 会吃掉 stdin 里剩下的 host 行，导致只清理第一台机器
mapfile -t HOSTS < <(awk '{print $1}' "$HOSTFILE" | grep -v '^$')

for HOST in "${HOSTS[@]}"; do
    echo ">>> 清理 $HOST"
    mpirun -n 1 --allow-run-as-root -H "$HOST" bash "$REMOTE_KILL" < /dev/null 2>&1 | grep CLEANED
done

echo ">>> 完成"
