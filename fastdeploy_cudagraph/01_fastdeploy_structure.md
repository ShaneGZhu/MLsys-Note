# FastDeploy 目录结构与核心模块

## 1. 整体目录结构

FastDeploy 项目位于 `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/`，核心代码在 `fastdeploy/` 目录下。

```
FastDeploy/
├── fastdeploy/                    # 主Python包
│   ├── scheduler/                 # 请求调度与任务管理
│   ├── worker/                    # Worker进程与ModelRunner
│   ├── model_executor/             # 模型执行、层、图优化
│   ├── engine/                     # 核心推理引擎
│   ├── distributed/                # 分布式通信
│   ├── router/                     # 请求路由
│   ├── spec_decode/                # 投机解码
│   └── ...
├── custom_ops/                     # 自定义CUDA/Triton算子
├── benchmarks/                     # Benchmark配置
├── tests/                          # 测试用例
└── examples/                       # 示例应用
```

## 2. 核心模块层次关系

FastDeploy 采用分层架构，从上到下：

```
Engine → (WIP)Executor → Worker → ModelRunner → Model → Layers/Ops
```

### 2.1 Scheduler 层 (`fastdeploy/scheduler/`)

**职责**：管理 incoming requests，处理 queuing、chunked prefill、load balancing

| 文件 | 路径 | 功能 |
|------|------|------|
| `local_scheduler.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/scheduler/local_scheduler.py` | 内存本地调度器，管理请求/响应 |
| `global_scheduler.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/scheduler/global_scheduler.py` | Redis分布式调度器 |
| `dp_scheduler.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/scheduler/dp_scheduler.py` | 数据并行调度器 |
| `splitwise_scheduler.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/scheduler/splitwise_scheduler.py` | Prefill/Decode分离调度器 |

### 2.2 Worker 层 (`fastdeploy/worker/`)

**职责**：设备特定的 worker 进程，管理模型执行

| 文件 | 路径 | 功能 |
|------|------|------|
| `worker_base.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/worker/worker_base.py` | Worker抽象基类 |
| `model_runner_base.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/worker/model_runner_base.py` | ModelRunner抽象基类 |
| `worker_process.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/worker/worker_process.py` | Worker进程实现 (51K) |
| `gpu_worker.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/worker/gpu_worker.py` | GPU Worker实现 |
| `gpu_model_runner.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/worker/gpu_model_runner.py` | **GPU ModelRunner** (142K) |

### 2.3 ModelExecutor 层 (`fastdeploy/model_executor/`)

**职责**：模型执行、神经网络层、图优化

| 子目录/文件 | 路径 | 功能 |
|-------------|------|------|
| `models/` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/models/` | 模型实现 (Qwen, GLM, DeepSeek, etc.) |
| `layers/` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/layers/` | 神经网络层 (attention, linear, normalization) |
| `ops/` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/ops/` | 自定义算子 (gpu, triton backends) |
| `graph_optimization/` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/graph_optimization/` | **图优化 (CUDAGraph, CINN, Dy2St)** |
| `forward_meta.py` | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/forward_meta.py` | Forward元数据 (17K) |

## 3. ModelRunner 核心类

**GPU ModelRunner** 是模型执行的核心入口：

```python
# 文件: fastdeploy/worker/gpu_model_runner.py, line 109
class GPUModelRunner(ModelRunnerBase):
    def __init__(self, fd_config, device, device_id, rank, local_rank):
        # ... 初始化
```

**关键组件**:
- `self.use_cudagraph`: 是否启用CUDAGraph (line 192)
- `self.forward_meta`: Forward元数据，控制prefill/decode执行模式
- `self.attn_backends`: Attention后端列表
- `self.model`: 实际模型对象

## 4. Model 层 (`fastdeploy/model_executor/models/`)

模型文件列表：

| 模型 | 文件路径 |
|------|---------|
| DeepSeek V3 | `deepseek_v3.py` |
| Qwen2/3 | `qwen2.py`, `qwen3.py`, `qwen3moe.py` |
| GLM4 MoE | `glm4_moe.py`, `glm_moe_dsa.py` |
| Ernie4.5 MoE | `ernie4_5_moe.py` |

## 5. Layers 层 (`fastdeploy/model_executor/layers/`)

| 子目录 | 功能 |
|--------|------|
| `attention/` | Attention实现 (MLA, FlashAttention等) |
| `moe/` | MoE层实现 (FusedMoE, EP等) |
| `linear/` | 线性层 (ColumnParallel, RowParallel等) |
| `normalization/` | 归一化层 (RMSNorm等) |
| `rotary_embedding/` | RoPE位置编码 |

## 6. Graph Optimization 层 (`fastdeploy/model_executor/graph_optimization/`)

| 文件 | 功能 |
|------|------|
| `cudagraph_piecewise_backend.py` | **CUDAGraph核心实现** - subgraph级别capture/replay |
| `graph_optimization_backend.py` | 图优化后端封装 (Dy2St + CINN + CUDAGraph) |
| `decorator.py` | @support_graph_optimization装饰器 |
| `dynamic_dims_marker.py` | 动态维度标记 |

## 7. 执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler                                │
│  (local_scheduler.py / global_scheduler.py)                      │
│  - 请求队列管理                                                  │
│  - Chunked Prefill                                               │
│  - Load Balancing                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Worker Process                           │
│  (worker_process.py)                                              │
│  - 多Worker管理                                                   │
│  - 请求分发                                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GPUModelRunner                              │
│  (gpu_model_runner.py)                          ─────────────────│
│  - 输入预处理 (InputBatch)                      │                │
│  - 模型执行                                    │                │
│  - CUDAGraph管理 ──────────────────────────────┘                │
│  - Output处理                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GraphOptBackend                              │
│  (graph_optimization_backend.py)                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  1. Dy2St (Dynamic → Static conversion)                      ││
│  │  2. CINN compilation                                        ││
│  │  3. CUDAGraph capture/replay                               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CudaGraphPiecewiseBackend                   │
│  (cudagraph_piecewise_backend.py)                                │
│  - ConcreteSizeEntry 管理                                        │
│  - capture() / replay() 逻辑                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Model + Layers                           │
│  - DeepseekV3ForCausalLM                                         │
│  - DeepseekV3MLAAttention (Attention)                            │
│  - DeepSeekV3MoE (MoE)                                          │
│  - CUDA Ops (custom_ops/gpu_ops/)                                │
└─────────────────────────────────────────────────────────────────┘
```
