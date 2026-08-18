#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 聚合部署模式：2 个 SGLang worker + 1 个 KV-aware router frontend
# 适用场景：单机多 worker，验证 KV-aware 路由收益
# GPUs: 2

# 任意子命令失败立即退出；EXIT 时触发 trap 清理所有子进程
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Dynamo 工具函数目录（launch_utils 提供 wait_any_exit 等进程管理函数）
SCRIPT_DIR="/root/paddlejob/workspace/env_run/output/zhushengguang/kvc/dynamo/examples"
# gpu_utils.sh 提供显存限制函数，本脚本不限制显存故注释掉
# source "$SCRIPT_DIR/common/gpu_utils.sh"
source "$SCRIPT_DIR/common/launch_utils.sh" # 提供 print_launch_banner, wait_any_exit

source /root/paddlejob/workspace/env_run/output/zhushengguang/kvc/dynamo/.venv/bin/activate
export PATH="/root/paddlejob/workspace/env_run/output/zhushengguang/kvc/dynamo/.venv/bin:$PATH"
export PYTHONPATH="/root/paddlejob/workspace/env_run/output/zhushengguang/kvc/dynamo:$PYTHONPATH"
# -----------------------------------------------------------------------
# 命令行参数解析
# --enable-otel : 开启 OpenTelemetry 链路追踪（生产排障用，平时不需要）
# --approx      : 近似 KV 路由模式，不发布 KV events，用于对照组消融实验
# -----------------------------------------------------------------------
ENABLE_OTEL=false
APPROX_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        --approx)
            # approx 模式：frontend 加 --no-kv-events，worker 不加 --kv-events-config
            # 用于验证"去掉真实 KV events 后收益下降多少"
            APPROX_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  --approx             Enable approximate KV routing (no KV events)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: System metrics are enabled by default on ports 8081 (worker-1), 8082 (worker-2)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------
# OpenTelemetry 追踪配置（--enable-otel 时生效）
# 需要本地有 OTLP collector 监听 4317 端口
# -----------------------------------------------------------------------
TRACE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    TRACE_ARGS+=(--enable-trace --otlp-traces-endpoint localhost:4317)
fi

# -----------------------------------------------------------------------
# 模型与端口配置
# MODEL     : 模型路径或 HF repo id，worker 和 frontend 需保持一致
# HTTP_PORT : frontend 对外暴露的 HTTP 端口，默认 8000
# -----------------------------------------------------------------------
MODEL="zai-org/GLM-4.5-Air"
MODEL_PATH="/root/paddlejob/workspace/env_run/output/zhushengguang/models/GLM-4.5-Air"

# GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)  # 不限制显存，注释掉

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + KV Routing (2 GPUs)" "$MODEL" "$HTTP_PORT"

# -----------------------------------------------------------------------
# 启动 frontend（路由入口）
# --router-mode kv  : 使用 KV-aware 路由，基于 radix tree 前缀匹配选 worker
# --no-kv-events    : approx 模式下加此参数，退化为近似路由（不消费 ZMQ events）
# DYN_HTTP_PORT     : frontend 监听端口，默认 8000
# -----------------------------------------------------------------------
FRONTEND_ARGS=(
    --discovery-backend file   # file 模式：用共享文件系统做服务发现，无需 etcd
    --request-plane tcp        # 请求直连 worker，无需 NATS
    --router-mode kv           # KV-aware 路由，切对照组只改这一行
    --http-port "$HTTP_PORT"
    --model-name "$MODEL"
    --model-path "$MODEL_PATH" # 提供本地路径避免 frontend 尝试从 HF 拉 model config
)
if [ "$APPROX_MODE" = true ]; then
    FRONTEND_ARGS+=(--no-kv-events)
fi
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}" &

# -----------------------------------------------------------------------
# KV events 配置（approx 模式下置空，worker 不发布 events）
# publisher=zmq  : 通过 ZMQ 直接推送到 frontend，无需 NATS
# topic          : frontend 订阅的 topic 名，两侧必须一致
# endpoint       : worker 监听的 ZMQ 地址，每个 worker 用不同端口（5557/5558）
# -----------------------------------------------------------------------
KV_EVENTS_ARGS_1=()
KV_EVENTS_ARGS_2=()
if [ "$APPROX_MODE" = false ]; then
    KV_EVENTS_ARGS_1=(--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}')
    KV_EVENTS_ARGS_2=(--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558"}')
fi

CHAT_TEMPLATE="$MODEL_PATH/chat_template.jinja"

# -----------------------------------------------------------------------
# 启动 worker-1（GPU 0-3，TP4）
# DYN_SYSTEM_PORT : Dynamo 内部系统端口（服务发现/健康检查），每个 worker 不同
# --tp-size 4     : 4 卡 tensor parallel，GLM-4.5-Air MoE 模型需要
# --page-size 16  : KV cache block 大小（token 数），需与 trace 的 hash_ids 对齐
# --chat-template : GLM-4.5-Air 需要显式指定 jinja 模板，否则输出乱码
# --disable-piecewise-cuda-graph : 避免 MoE 模型 CUDA graph 兼容性问题
# --enable-metrics : 暴露 /metrics 端口，用于采集 SGLang 侧缓存命中率等指标
# --discovery-backend file : 与 frontend 保持一致
# --request-plane tcp      : 与 frontend 保持一致
# --event-plane zmq        : KV events 通过 ZMQ 直接推送，无需 NATS
# -----------------------------------------------------------------------
OTEL_SERVICE_NAME=dynamo-worker-1 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_WORKER1:-8081} \
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m dynamo.sglang \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL" \
  --tp-size 4 \
  --page-size 16 \
  --context-length 131072     \
  --mem-fraction-static 0.80     \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-metrics \
  --disable-piecewise-cuda-graph \
  --discovery-backend file \
  --request-plane tcp \
  --event-plane zmq \
  "${KV_EVENTS_ARGS_1[@]}" \
  "${TRACE_ARGS[@]}" &

# 启动 worker-2（GPU 4-7，TP4），ZMQ 端口偏移 +1，DYN_SYSTEM_PORT 偏移 +1
OTEL_SERVICE_NAME=dynamo-worker-2 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_WORKER2:-8082} \
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m dynamo.sglang \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL" \
  --tp-size 4 \
  --page-size 16 \
  --context-length 131072     \
  --mem-fraction-static 0.80     \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-metrics \
  --disable-piecewise-cuda-graph \
  --discovery-backend file \
  --request-plane tcp \
  --event-plane zmq \
  "${KV_EVENTS_ARGS_2[@]}" \
  "${TRACE_ARGS[@]}" &

# 等待任意子进程退出（worker 崩溃或手动 Ctrl+C），EXIT trap 负责清理其余进程
wait_any_exit
