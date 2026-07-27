#!/bin/bash
# 单机 worker 启动脚本，由 agg_router_glm_multinode.sh 通过 mpirun 调用
#
# 环境变量（由主脚本通过 mpirun -x 传入）：
#   VENV          : venv 路径
#   MODEL_PATH    : 模型本地路径
#   MODEL_NAME    : served model name（HF repo id）
#   CHAT_TEMPLATE : chat template jinja 路径
#   DYN_FILE_KV   : file discovery 共享目录
#   APPROX_MODE   : true/false，是否跳过 kv-events-config
#
# 本机自动起 2 个 worker：
#   Worker 0: GPU 0-3, DYN_SYSTEM_PORT=SYS_PORT0, ZMQ=ZMQ_PORT0
#   Worker 1: GPU 4-7, DYN_SYSTEM_PORT=SYS_PORT1, ZMQ=ZMQ_PORT1
# 端口由主脚本按节点索引计算后通过环境变量传入：
#   SYS_PORT0, SYS_PORT1, ZMQ_PORT0, ZMQ_PORT1

set -e

source "$VENV/bin/activate"
export DYN_FILE_KV="$DYN_FILE_KV"

# -----------------------------------------------------------------------
# 构造 kv-events-config 参数
# approx 模式下不传，worker 不发布 KV events
# -----------------------------------------------------------------------
KV_ARG0=""
KV_ARG1=""
if [ "$APPROX_MODE" != "true" ]; then
    KV_ARG0="--kv-events-config {\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$ZMQ_PORT0\"}"
    KV_ARG1="--kv-events-config {\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$ZMQ_PORT1\"}"
fi

HOST=$(hostname)
echo "[$HOST] 启动 worker-0 (GPU 0-3, sys=$SYS_PORT0, zmq=$ZMQ_PORT0)"

# Worker 0：GPU 0-3
DYN_SYSTEM_PORT=$SYS_PORT0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m dynamo.sglang \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tp-size 4 \
  --page-size 16 \
  --context-length 131072 \
  --mem-fraction-static 0.80 \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-metrics \
  --disable-piecewise-cuda-graph \
  --discovery-backend file \
  --request-plane tcp \
  --event-plane zmq \
  $KV_ARG0 \
  > /tmp/worker_${HOST}_0.log 2>&1 &

echo "[$HOST] 启动 worker-1 (GPU 4-7, sys=$SYS_PORT1, zmq=$ZMQ_PORT1)"

# Worker 1：GPU 4-7
DYN_SYSTEM_PORT=$SYS_PORT1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m dynamo.sglang \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tp-size 4 \
  --page-size 16 \
  --context-length 131072 \
  --mem-fraction-static 0.80 \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-metrics \
  --disable-piecewise-cuda-graph \
  --discovery-backend file \
  --request-plane tcp \
  --event-plane zmq \
  $KV_ARG1 \
  > /tmp/worker_${HOST}_1.log 2>&1 &

# 等待本机两个 worker 都退出
wait
