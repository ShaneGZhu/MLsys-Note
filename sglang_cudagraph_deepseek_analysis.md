# SGLang CUDAGraph 与 DeepSeek V3 推理深度解析

> 本文档基于 SGLang 源码，对其核心架构、CUDAGraph 机制、Prefill/Decode 差异以及 DeepSeek V3.2 推理链路进行深度分析。
> 所有代码引用均标注文件路径与行号，路径相对于 `sglang/python/sglang/` 根目录。

---

## 目录

- [第1部分：SGLang 目录结构与核心模块解析](#第1部分sglang-目录结构与核心模块解析)
- [第2部分：CUDAGraph 代码详解](#第2部分cudagraph-代码详解)
- [第3部分：Prefill vs Decode — CUDAGraph 的差异化使用](#第3部分prefill-vs-decode--cudagraph-的差异化使用)
- [第4部分：DeepSeek V3.2 模型推理调用层级](#第4部分deepseek-v32-模型推理调用层级)
- [附录](#附录)

---

# 第1部分：SGLang 目录结构与核心模块解析

## 1.1 顶层目录结构

```
sglang/
├── python/                  # 主 Python 包
│   └── sglang/
│       ├── srt/             # ★ Serving Runtime —— 核心推理引擎（下文重点）
│       ├── lang/            # SGLang 前端语言（解释器、IR、Tracer）
│       ├── jit_kernel/      # JIT 编译的 Triton 内核
│       ├── cli/             # CLI 命令处理
│       ├── eval/            # 评估工具
│       └── utils.py         # 顶层工具函数
├── sgl-kernel/              # 自定义 CUDA 内核包 (sgl_kernel)
│   ├── csrc/                #   C++/CUDA 源码 (attention, gemm, moe, ...)
│   └── src/sgl-kernel/      #   Python 接口
├── sgl-model-gateway/       # 模型网关（外部 API 路由）
├── benchmark/               # 性能基准测试脚本
├── test/                    # 集成/单元测试
├── docker/                  # Dockerfile 和 Compose 配置
├── docs/                    # 文档
├── examples/                # 使用示例
├── scripts/                 # 构建/CI/工具脚本
└── 3rdparty/                # 第三方依赖
```

## 1.2 Serving Runtime (srt/) 核心目录

```
srt/
├── entrypoints/             # API 入口层
│   ├── engine.py            #   Engine 类 — 启动子进程的主入口
│   ├── EngineBase.py        #   EngineBase 抽象基类
│   ├── http_server.py       #   FastAPI HTTP 服务（OpenAI/Anthropic/Ollama 兼容）
│   └── openai/              #   OpenAI 协议定义
├── managers/                # 进程管理层
│   ├── scheduler.py         #   ★ Scheduler — 核心调度器
│   ├── tokenizer_manager.py #   TokenizerManager — 分词管理
│   ├── detokenizer_manager.py # DetokenizerManager — 反分词
│   ├── tp_worker.py         #   TpModelWorker — 张量并行工作器
│   ├── schedule_batch.py    #   Req/ScheduleBatch/ModelWorkerBatch 数据结构
│   ├── schedule_policy.py   #   调度策略（LPM, DFS-weight, FCFS 等）
│   ├── io_struct.py         #   IPC 消息类型
│   └── data_parallel_controller.py # DP 控制器
├── model_executor/          # 模型执行层
│   ├── model_runner.py      #   ★ ModelRunner — 模型前向分发
│   ├── forward_batch_info.py #  ForwardBatch/ForwardMode 定义
│   ├── cuda_graph_runner.py #   ★ CudaGraphRunner (Decode)
│   ├── piecewise_cuda_graph_runner.py # PiecewiseCudaGraphRunner (Prefill)
│   ├── input_buffers.py     #   GraphInputBuffers 静态缓冲区
│   └── model_runner_kv_cache_mixin.py # KV Cache 初始化 Mixin
├── models/                  # 150+ 模型实现
│   ├── registry.py          #   ModelRegistry 模型注册
│   ├── llama.py             #   LLaMA 系列（经典范例）
│   ├── deepseek_v2.py       #   ★ DeepSeek V2/V3/V3.2 模型
│   ├── deepseek_common/     #   DeepSeek 共享工具
│   └── ...                  #   Qwen, Gemma, Mixtral, GLM 等
├── layers/                  # 神经网络构建模块
│   ├── radix_attention.py   #   RadixAttention — 核心注意力层
│   ├── attention/           #   注意力后端（FlashInfer, FA3, Triton, MLA 等）
│   ├── moe/                 #   MoE 混合专家（TopK, FusedMoE, EP）
│   ├── linear.py            #   张量并行线性层
│   ├── logits_processor.py  #   Logits 处理器
│   ├── sampler.py           #   采样器
│   ├── rotary_embedding.py  #   RoPE 实现
│   └── layernorm.py         #   RMSNorm/LayerNorm
├── mem_cache/               # 显存与 KV Cache 管理
│   ├── memory_pool.py       #   ReqToTokenPool, KVCache 显存池
│   ├── radix_cache.py       #   RadixCache 前缀缓存
│   └── allocator.py         #   Token-KV 分配器
├── sampling/                # 采样参数与批处理
├── configs/                 # 模型配置（ModelConfig 等）
├── model_loader/            # 权重加载
├── distributed/             # 分布式通信（parallel_state 等）
├── compilation/             # torch.compile & 分片 CUDAGraph 后端
├── speculative/             # 投机解码（EAGLE v1/v2/v3, N-gram）
├── disaggregation/          # Prefill-Decode 分离部署
├── lora/                    # LoRA 适配器
├── constrained/             # 受限生成（XGrammar, Outlines）
└── server_args.py           # ServerArgs — 全局服务器配置
```

## 1.3 进程架构

SGLang 采用多进程架构，通过 ZMQ 进行进程间通信：

```
                              ┌─────────────────────────────────┐
                              │         Main Process            │
┌──────────────┐              │  ┌───────────────────────────┐  │
│  HTTP Client │──HTTP──────>│  │ http_server.py (FastAPI)  │  │
└──────────────┘              │  └───────────┬───────────────┘  │
                              │              │                  │
                              │  ┌───────────▼───────────────┐  │
                              │  │  TokenizerManager          │  │
                              │  │  (tokenize + 请求管理)     │  │
                              │  └───────────┬───────────────┘  │
                              └──────────────┼──────────────────┘
                                    ZMQ      │
                              ┌──────────────▼──────────────────┐
                              │       Subprocess: Scheduler     │
                              │  ┌───────────────────────────┐  │
                              │  │  Scheduler                 │  │
                              │  │  (调度 + 批处理)            │  │
                              │  └───────────┬───────────────┘  │
                              │              │                  │
                              │  ┌───────────▼───────────────┐  │
                              │  │  TpModelWorker             │  │
                              │  │  (张量并行工作器)           │  │
                              │  └───────────┬───────────────┘  │
                              │              │                  │
                              │  ┌───────────▼───────────────┐  │
                              │  │  ModelRunner               │  │
                              │  │  (模型前向 + CUDAGraph)    │  │
                              │  └───────────┬───────────────┘  │
                              │       ┌──────┴──────┐           │
                              │       │             │           │
                              │  CudaGraph     Piecewise       │
                              │  Runner        CudaGraph       │
                              │  (Decode)      Runner(Prefill) │
                              │       │             │           │
                              │       └──────┬──────┘           │
                              │              │                  │
                              │  ┌───────────▼───────────────┐  │
                              │  │  Model.forward()           │  │
                              │  │  (如 DeepseekV2ForCausalLM)│  │
                              │  └───────────────────────────┘  │
                              └──────────────┬──────────────────┘
                                    ZMQ      │
                              ┌──────────────▼──────────────────┐
                              │   Subprocess: DetokenizerManager │
                              │   (反分词 → 流式返回)            │
                              └─────────────────────────────────┘
```

## 1.4 数据流转

从 HTTP 请求到 Token 生成，数据经历以下转换链：

```
GenerateReqInput          # HTTP 请求体
    │  [TokenizerManager: 分词]
    ▼
TokenizedGenerateReqInput # 含 token_ids
    │  [ZMQ 发送到 Scheduler]
    ▼
Req                       # 单个请求的调度状态
    │  [Scheduler: 批处理组装]
    ▼
ScheduleBatch             # CPU 侧调度批次
    │  [.get_model_worker_batch()]
    ▼
ModelWorkerBatch          # CPU→GPU 传输批次
    │  [ForwardBatch.init_new()]
    ▼
ForwardBatch              # GPU 张量批次（model.forward 的输入）
    │  [Model.forward()]
    ▼
LogitsProcessorOutput     # GPU 张量（logits + hidden_states）
    │  [Sampler.forward()]
    ▼
next_token_ids            # 采样得到的 token
    │  [.cpu() → ZMQ → DetokenizerManager]
    ▼
BatchStrOutput            # 反分词后的文本
```

关键代码位置：
```python
# File: srt/managers/schedule_batch.py
# ScheduleBatch -> ModelWorkerBatch 转换
class ScheduleBatch:
    def get_model_worker_batch(self) -> ModelWorkerBatch: ...

# File: srt/model_executor/forward_batch_info.py
# ModelWorkerBatch -> ForwardBatch 转换
class ForwardBatch:
    @classmethod
    def init_new(cls, batch: ModelWorkerBatch, model_runner: ModelRunner) -> ForwardBatch: ...
```

## 1.5 核心模块详解

### 1.5.1 Engine — 启动入口

```python
# File: srt/entrypoints/engine.py
class Engine(EngineBase):
    """SGLang 主引擎，启动三个组件：
    1. TokenizerManager (主进程)
    2. Scheduler (子进程)
    3. DetokenizerManager (子进程)
    IPC 通过 ZMQ 实现。
    """
    def _launch_subprocesses(self): ...
    async def generate(self, ...): ...
```

`Engine` 是用户使用 SGLang 的 Python API 入口。它负责启动子进程并建立 ZMQ 通信通道。

### 1.5.2 HTTP Server — 对外接口

```python
# File: srt/entrypoints/http_server.py
# FastAPI 服务器，暴露 OpenAI 兼容端点：
# - /v1/chat/completions
# - /v1/completions
# - /v1/embeddings
# 以及 Anthropic、Ollama 兼容端点
```

HTTP Server 接收请求后，将其翻译为 `GenerateReqInput`，发送给 `TokenizerManager`。

### 1.5.3 TokenizerManager — 分词与请求管理

```python
# File: srt/managers/tokenizer_manager.py
class TokenizerManager:
    """运行在主进程中，负责：
    1. 将文本 tokenize 为 token_ids
    2. 处理多模态数据
    3. 通过 ZMQ 发送到 Scheduler
    4. 创建 ReqState 等待异步结果
    """
```

### 1.5.4 Scheduler — 核心调度器

Scheduler 是 SGLang 最核心、最复杂的组件，采用 Mixin 模式组合了 10+ 个功能模块：

```python
# File: srt/managers/scheduler.py, line 253
class Scheduler(
    SchedulerOutputProcessorMixin,     # 输出处理
    SchedulerUpdateWeightsMixin,       # 权重更新
    SchedulerProfilerMixin,            # 性能分析
    SchedulerMetricsMixin,             # 指标收集
    SchedulerDisaggregationDecodeMixin,# PD 分离 Decode
    SchedulerDisaggregationPrefillMixin,# PD 分离 Prefill
    SchedulerMultiplexMixin,           # 多路复用
    SchedulerRuntimeCheckerMixin,      # 运行时检查
    SchedulerPPMixin,                  # 流水线并行
    SchedulerDPAttnMixin,              # 数据并行注意力
    SchedulerDllmMixin,                # 扩散 LLM
):
    """管理张量并行 GPU Worker 的调度器"""
```

核心事件循环：

```python
# File: srt/managers/scheduler.py (event_loop_normal 方法)
def event_loop_normal(self):
    """主事件循环：接收请求 → 获取下一批 → 执行 → 处理结果"""
    while True:
        recv_reqs = self.recv_requests()          # 1. 接收新请求
        self.process_input_requests(recv_reqs)     # 2. 加入等待队列
        batch = self.get_next_batch_to_run()       # 3. 决定跑 prefill 还是 decode
        if batch:
            result = self.run_batch(batch)          # 4. 执行前向传播
            self.process_batch_result(batch, result)# 5. 处理结果、检查完成状态
```

### 1.5.5 TpModelWorker — 张量并行工作器

```python
# File: srt/managers/tp_worker.py
class TpModelWorker(BaseTpWorker):
    """创建 ModelRunner，负责 ModelWorkerBatch → ForwardBatch → 前向推理 → 采样"""

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # 1. 转换为 GPU 张量
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        # 2. 调用 ModelRunner.forward()
        model_output = self.model_runner.forward(forward_batch)
        # 3. 采样得到 next_token_ids
        logits_output = model_output.logits_output
        next_token_ids = self.model_runner.sample(logits_output, forward_batch)
        # 4. 返回 GenerationBatchResult
        return GenerationBatchResult(...)
```

### 1.5.6 ModelRunner — 模型前向分发

```python
# File: srt/model_executor/model_runner.py, line 278
class ModelRunner(ModelRunnerKVCacheMixin):
    """核心职责：
    - 持有模型实例 (self.model)
    - 管理注意力后端 (self.attn_backend)
    - 管理 CudaGraphRunner 和 PiecewiseCudaGraphRunner
    - 根据 ForwardMode 分发 forward 调用
    """
```

关键前向方法调用链：

```python
# File: srt/model_executor/model_runner.py, line 2387
def forward(self, forward_batch: ForwardBatch) -> ModelRunnerOutput:
    output = self._forward_raw(forward_batch, ...)
    return output

# File: srt/model_executor/model_runner.py, line 2443
def _forward_raw(self, forward_batch, ...):
    # 判断是否使用 CUDAGraph
    can_run_graph = (
        forward_batch.forward_mode.is_cuda_graph()  # DECODE/TARGET_VERIFY/IDLE
        and self.graph_runner
        and self.graph_runner.can_run(forward_batch)
    )

    if can_run_graph:
        ret = self.graph_runner.replay(forward_batch, ...)  # ★ CudaGraphRunner
        return ModelRunnerOutput(logits_output=ret, can_run_graph=True)

    # 非 CUDAGraph 路径
    if forward_batch.forward_mode.is_decode():
        ret = self.forward_decode(forward_batch, ...)       # 直接 decode
    elif forward_batch.forward_mode.is_extend():
        ret = self.forward_extend(forward_batch, ...)       # 可能用 PiecewiseCudaGraphRunner
    ...
```

### 1.5.7 ForwardBatch 与 ForwardMode

```python
# File: srt/model_executor/forward_batch_info.py, line 74
class ForwardMode(IntEnum):
    EXTEND = auto()         # Prefill/扩展 — 处理新 token 序列
    DECODE = auto()         # Decode — 每个请求生成 1 个 token
    MIXED = auto()          # 混合模式（Chunked Prefill）
    IDLE = auto()           # 空闲（DP Attention 填充）
    TARGET_VERIFY = auto()  # 投机解码验证
    DRAFT_EXTEND = auto()   # 投机解码草稿扩展
    SPLIT_PREFILL = auto()  # PD 多路复用分片 Prefill
    DLLM_EXTEND = auto()    # 扩散 LLM 扩展

    def is_extend(self):     # EXTEND/MIXED/DRAFT_EXTEND/TARGET_VERIFY/...
    def is_decode(self):     # 仅 DECODE
    def is_cuda_graph(self): # DECODE/TARGET_VERIFY/IDLE/DLLM_EXTEND
```

### 1.5.8 ScheduleBatch 与调度策略

```python
# File: srt/managers/schedule_batch.py
class Req:
    """单个请求的完整状态：token_ids, 采样参数, KV cache 索引等"""

class ScheduleBatch:
    """CPU 侧调度批次，包含多个 Req"""

class ModelWorkerBatch:
    """从 ScheduleBatch 提取的、准备传入 GPU 的批次数据"""
```

```python
# File: srt/managers/schedule_policy.py
class SchedulePolicy:
    """调度策略：
    - Cache-aware: LPM (Longest Prefix Match), DFS-weight
    - Cache-agnostic: FCFS, LOF, Random, Routing-Key
    """
```

### 1.5.9 显存与 KV Cache 管理

```python
# File: srt/mem_cache/memory_pool.py
class ReqToTokenPool:
    """管理请求到 token 槽位的映射"""

class MHATokenToKVPool:
    """标准 MHA 模型的 KV Cache 显存池"""

class MLATokenToKVPool:
    """MLA 模型（如 DeepSeek）的 KV Cache 显存池，存储压缩后的 latent"""
```

```python
# File: srt/mem_cache/radix_cache.py
class RadixCache:
    """基于 Radix Tree 的前缀缓存
    - 允许不同请求共享相同前缀的 KV Cache
    - 显著减少重复计算（如 system prompt）
    """
```

### 1.5.10 注意力层与后端

```python
# File: srt/layers/radix_attention.py
class RadixAttention(nn.Module):
    """核心注意力层，所有模型共用。
    forward() 内部委托给当前选定的 attention backend。
    """
    def forward(self, q, k, v, forward_batch, ...):
        return forward_batch.attn_backend.forward(q, k, v, self, forward_batch, ...)
```

SGLang 支持多种注意力后端（`srt/layers/attention/` 目录）：

| 后端 | 文件 | 说明 |
|------|------|------|
| FlashInfer | `flashinfer_backend.py` | 默认后端，PagedKV |
| FlashInfer MLA | `flashinfer_mla_backend.py` | DeepSeek MLA 专用 |
| FlashAttention 3/4 | `flashattention_backend.py` | FA3/FA4 |
| FlashMLA | `flashmla_backend.py` | DeepSeek FlashMLA |
| CUTLASS MLA | `cutlass_mla_backend.py` | CUTLASS MLA Decode |
| Triton | `triton_backend.py` | Triton 实现 |
| NSA | `nsa_backend.py` | 原生稀疏注意力 |

### 1.5.11 其他重要子系统

| 子系统 | 目录 | 说明 |
|--------|------|------|
| 采样 | `srt/sampling/` | SamplingParams, top-k/top-p/min-p/temperature |
| LoRA | `srt/lora/` | LoRA 适配器管理 |
| 投机解码 | `srt/speculative/` | EAGLE v1/v2/v3, N-gram |
| PD 分离 | `srt/disaggregation/` | Prefill-Decode 分离部署 |
| 量化 | `srt/layers/quantization/` | FP8, GPTQ, AWQ, INT8, MXFP4 等 |
| 受限生成 | `srt/constrained/` | XGrammar, Outlines, LLGuidance |
| 编译 | `srt/compilation/` | torch.compile 和分片 CUDAGraph |
| 分布式 | `srt/distributed/` | TP/PP/DP/EP 并行通信 |

---

# 第2部分：CUDAGraph 代码详解

## 2.1 为什么需要 CUDAGraph？

在 LLM 推理中，**Decode 阶段**每个请求每次只产出 1 个 token，计算量极小，但需要启动大量 CUDA kernel。此时 **CPU 启动开销（kernel launch overhead）** 成为主要瓶颈。

CUDA Graph 的核心思想：
1. **捕获阶段（Capture）**：将一次完整前向传播中所有 CUDA 操作录制成一个"图"
2. **重放阶段（Replay）**：直接回放整个图，**零 CPU 开销**，仅需一次 GPU 提交

SGLang 为此实现了**双 Runner 架构**，分别优化 Decode 和 Prefill 阶段。

## 2.2 双 Runner 架构对比

| 特性 | CudaGraphRunner | PiecewiseCudaGraphRunner |
|------|----------------|--------------------------|
| 文件 | `srt/model_executor/cuda_graph_runner.py` | `srt/model_executor/piecewise_cuda_graph_runner.py` |
| 适用阶段 | **Decode**（及 TARGET_VERIFY, DLLM_EXTEND） | **Extend/Prefill** |
| 索引键 | `batch_size`（请求数） | `num_tokens`（token 总数） |
| 捕获策略 | **整体捕获**：整个 model.forward() 作为一个 graph | **分片捕获**：torch.compile 将模型拆分为多个子图，各自捕获 |
| 配置开关 | 默认启用，`--disable-cuda-graph` 关闭 | `--enable-piecewise-cuda-graph` 显式启用 |
| 捕获粒度 | 每个 batch_size 一个 graph | 每个 num_tokens × 每个模型分片 一个 graph |

## 2.3 CudaGraphRunner 深入分析

### 2.3.1 类定义与初始化

```python
# File: srt/model_executor/cuda_graph_runner.py, line 238
class CudaGraphRunner:
    """用 CUDA Graph 和 torch.compile 运行模型前向传播"""

    def __init__(self, model_runner: ModelRunner):
        # 核心数据结构
        self.graphs = {}           # Dict[int, torch.cuda.CUDAGraph]  bs → 捕获的图
        self.output_buffers = {}   # Dict[int, LogitsProcessorOutput] bs → 输出缓冲

        # 确定捕获的 ForwardMode
        self.capture_forward_mode = ForwardMode.DECODE   # 默认 DECODE
        self.num_tokens_per_bs = 1                        # Decode: 每请求 1 token

        # 投机解码时覆盖
        if model_runner.spec_algorithm.is_eagle():
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY
            self.num_tokens_per_bs = server_args.speculative_num_draft_tokens

        # 确定要捕获的 batch_size 列表
        # 例如: [1, 2, 4, 8, 12, 16, 24, 32, ..., 256]
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner, ...)

        # 初始化注意力后端的 CUDAGraph 状态
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        model_runner.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

        # 创建静态输入缓冲区 — 固定 GPU 地址，CUDA Graph 要求
        self.buffers: GraphInputBuffers = GraphInputBuffers.create(
            max_bs=self.max_bs, max_num_token=self.max_num_token, ...
        )

        # 执行捕获
        with model_capture_mode():
            self.capture()
```

### 2.3.2 GraphInputBuffers — 静态缓冲区

CUDA Graph 要求输入/输出 tensor 的 **GPU 地址在捕获和重放时必须一致**。因此需要预分配一组最大尺寸的静态缓冲区：

```python
# File: srt/model_executor/input_buffers.py, line 16
@dataclass
class GraphInputBuffers:
    input_ids: torch.Tensor          # [max_num_token]     int64
    req_pool_indices: torch.Tensor   # [max_bs]            int32
    seq_lens: torch.Tensor           # [max_bs]            int32, 填充 seq_len_fill_value
    out_cache_loc: torch.Tensor      # [max_num_token]     int32
    positions: torch.Tensor          # [max_num_token]     int64
    num_token_non_padded: torch.Tensor # [1]               int32
    ...

    @classmethod
    def create(cls, *, max_bs, max_num_token, ...):
        """预分配所有静态缓冲区到 GPU 上"""
        with torch.device(device):
            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int32)
            seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int32)
            out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            ...
```

### 2.3.3 capture() — 图捕获流程

```python
# File: srt/model_executor/cuda_graph_runner.py, line 476
def capture(self) -> None:
    # 冻结 GC，防止 GC 干扰 CUDA Graph 录制
    with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream

            # ★ 关键：从大到小逆序捕获
            # 原因：大 batch 先分配显存，小 batch 可以复用该显存
            for bs in reversed(self.capture_bs):
                with patch_model(model, bs in compile_bs, ...):
                    graph, output = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output
```

### 2.3.4 capture_one_batch_size() — 单个 batch_size 捕获

```python
# File: srt/model_executor/cuda_graph_runner.py, line 556
def capture_one_batch_size(self, bs, forward, stream_idx=None):
    graph = torch.cuda.CUDAGraph()
    num_tokens = bs * self.num_tokens_per_bs

    # 1. 从静态缓冲区切片到当前 bs 大小
    input_ids = self.buffers.input_ids[:num_tokens]
    req_pool_indices = self.buffers.req_pool_indices[:bs]
    seq_lens = self.buffers.seq_lens[:bs]
    out_cache_loc = self.buffers.out_cache_loc[:num_tokens]
    ...

    # 2. 构建 ForwardBatch（使用静态缓冲区的切片）
    forward_batch = ForwardBatch(
        forward_mode=self.capture_forward_mode,
        batch_size=bs,
        input_ids=input_ids,
        seq_lens=seq_lens,
        ...
    )

    # 3. 初始化注意力后端的 capture 元数据
    attn_backend.init_forward_metadata_capture_cuda_graph(bs, num_tokens, ...)

    # 4. 定义一次前向传播
    def run_once():
        return forward(input_ids, positions, forward_batch)

    # 5. Warmup：运行两次以确保所有 lazy 初始化完成
    for _ in range(2):
        run_once()
        torch.cuda.synchronize()

    # 6. 初始化全局图显存池（首次时）
    if get_global_graph_memory_pool() is None:
        set_global_graph_memory_pool(torch.cuda.graph_pool_handle())

    # 7. ★ 正式捕获
    with torch.cuda.graph(graph, pool=get_global_graph_memory_pool(), stream=stream):
        out = run_once()

    return graph, out
```

### 2.3.5 全局显存池

```python
# File: srt/model_executor/cuda_graph_runner.py, line 226
global_graph_memory_pool = None  # 全局唯一，所有 Runner 共享

def get_global_graph_memory_pool():
    return global_graph_memory_pool

def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val
```

**设计要点**：
- 单一 `graph_pool_handle()` 被所有 CUDAGraph Runner 共享
- 大 batch 先捕获 → 大显存块先分配 → 小 batch 复用同一块显存
- 避免每个 graph 独立分配显存造成的碎片化

### 2.3.6 can_run() — 判断能否使用 CUDAGraph

```python
# File: srt/model_executor/cuda_graph_runner.py, line 385
def can_run(self, forward_batch: ForwardBatch) -> bool:
    # 基本条件：batch_size 不超过已捕获的最大值
    if forward_batch.batch_size > self.max_bs:
        return False
    # 编码器-解码器模型的额外检查
    if self.is_encoder_decoder:
        if any(l > 0 for l in forward_batch.encoder_lens_cpu):
            return False
    # ... 其他条件检查
    return True
```

### 2.3.7 replay_prepare() — 重放准备

```python
# File: srt/model_executor/cuda_graph_runner.py, line 776
def replay_prepare(self, forward_batch, ...):
    raw_bs = forward_batch.batch_size

    # ★ 核心：将 batch_size 向上对齐到已捕获的值
    # 例如 raw_bs=5 → 对齐到 capture_bs 中的 8
    index = bisect.bisect_left(self.capture_bs, raw_bs)
    bs = self.capture_bs[index]

    # 将真实数据拷贝到静态缓冲区
    seq_lens_cpu = self.buffers.populate_from_forward_batch(
        forward_batch=forward_batch,
        raw_bs=raw_bs,
        bs=bs,                      # 对齐后的 bs
        seq_len_fill_value=self.seq_len_fill_value,  # 填充值（0 或 1）
        ...
    )

    # 更新注意力后端的 replay 元数据
    attn_backend.init_forward_metadata_replay_cuda_graph(
        bs,
        self.buffers.req_pool_indices[:bs],
        self.buffers.seq_lens[:bs],
        forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
        ...
    )

    self.raw_bs = raw_bs
    self.bs = bs
```

### 2.3.8 replay() — 重放执行

```python
# File: srt/model_executor/cuda_graph_runner.py, line 846
def replay(self, forward_batch, ...):
    # 1. 准备：对齐 bs、拷贝数据、更新元数据
    self.replay_prepare(forward_batch, ...)

    # 2. ★ 核心：一次 replay 调用替代整个 forward pass
    self.graphs[self.bs].replay()    # 零 CPU 开销！
    output = self.output_buffers[self.bs]

    # 3. 裁剪输出到真实 batch_size
    if isinstance(output, LogitsProcessorOutput):
        return LogitsProcessorOutput(
            next_token_logits=output.next_token_logits[:self.raw_num_token],
            hidden_states=(
                output.hidden_states[:self.raw_num_token]
                if output.hidden_states is not None else None
            ),
            ...
        )
```

### 2.3.9 Batch Size 捕获策略

```python
# File: srt/server_args.py (get_batch_sizes_to_capture 函数)
# 非投机解码的 batch_size 捕获列表：
# [1, 2, 4, 8, 12] + range(16, 257, step=8) + range(272, 512, step=16) + range(512, max+1, step=32)
#
# 投机解码的更细粒度列表：
# range(1,9,1) + range(10,33,2) + range(40,65,4) + range(72,257,8) + range(272,max+1,16)
#
# 如果 disable_cuda_graph_padding=True:
# list(range(1, max+1))  — 每个 bs 都单独捕获
```

## 2.4 PiecewiseCudaGraphRunner — Prefill 分片捕获

### 2.4.1 与 CudaGraphRunner 的核心差异

Prefill 阶段的特殊性：
- token 数量**变化大**（几个到几千个），不像 Decode 固定为 1
- 序列长度**参差不齐**（ragged），注意力计算模式不同
- MoE 层中的通信（DeepEP all-to-all）会改变 tensor 形状

因此 Prefill 不能像 Decode 那样用单一整体 graph，而是将模型**拆分为多个子图**，每个子图分别捕获。

### 2.4.2 初始化与编译

```python
# File: srt/model_executor/piecewise_cuda_graph_runner.py
class PiecewiseCudaGraphRunner:
    def __init__(self, model_runner: ModelRunner):
        self.capture_forward_mode = ForwardMode.EXTEND  # ★ 用于 Prefill

        # 按 num_tokens 而非 batch_size 索引
        # 例如: [4, 8, 12, ..., 32, 48, 64, ..., 256, 288, ..., 4096]
        self.capture_num_tokens = get_piecewise_capture_sizes(...)

        # 安装 torch.compile，将模型拆分为 "pieces"
        install_torch_compiled(model_runner.model)

        # 预热 torch.compile
        warmup_torch_compile(model_runner, ...)

        # 捕获分片 graph
        self.capture()
```

### 2.4.3 CUDAPiecewiseBackend — 每个分片的后端

```python
# File: srt/compilation/cuda_piecewise_backend.py
class CUDAPiecewiseBackend:
    """torch.compile 的自定义后端，每个编译子图独立管理 CUDA Graph"""

    def __call__(self, *args):
        runtime_shape = args[0].shape[0]  # 当前 token 数

        # 查找匹配的 size entry
        entry = self.concrete_size_entries[runtime_shape]

        if entry.cudagraph is None:
            # 首次：warmup
            if entry.num_warmup < self.warmup_count:
                entry.num_warmup += 1
                return self.compiled_graph_for_general_shape(*args)
            # 达到 warmup 次数：捕获 CUDA Graph
            entry.cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(entry.cudagraph, pool=self.graph_pool, stream=stream):
                entry.output = self.compiled_graph_for_general_shape(*args)
        else:
            # ★ 后续调用：直接 replay
            entry.cudagraph.replay()

        return entry.output
```

### 2.4.4 replay() — Piecewise 重放

```python
# File: srt/model_executor/piecewise_cuda_graph_runner.py
def replay(self, forward_batch, **kwargs):
    # 1. 正常初始化注意力元数据（不使用 cuda_graph 专用接口）
    self.model_runner.attn_backend.init_forward_metadata(forward_batch)

    # 2. 对齐 num_tokens
    index = bisect.bisect_left(self.capture_num_tokens, num_tokens)
    static_num_tokens = self.capture_num_tokens[index]

    # 3. 拷贝数据到静态缓冲区
    self.replay_prepare(forward_batch, static_num_tokens)

    # 4. ★ 调用 model.forward()
    # 内部每个 torch.compile 子图的 CUDAPiecewiseBackend 会自动 replay
    with set_forward_context(forward_batch):
        with set_compiled(True):
            output = self.model_runner.model.forward(
                self.static_input_ids[:static_num_tokens],
                self.static_positions[:static_num_tokens],
                self.static_forward_batch,
                **kwargs,
            )

    # 5. 裁剪输出到真实 token 数
    return output[:raw_num_tokens]
```

## 2.5 Attention Backend 的 CUDAGraph 接口

每个注意力后端必须实现 4 个 CUDAGraph 相关方法：

```python
# File: srt/layers/attention/base_attn_backend.py, line 17
class AttentionBackend(ABC):

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """初始化全局共享的 CUDAGraph 状态（只调用一次）
        FlashInfer: 分配 cuda_graph_kv_indices 缓冲区
        Triton:     分配 cuda_graph_attn_logits, cuda_graph_attn_lse 等
        """

    def init_forward_metadata_capture_cuda_graph(
        self, bs, num_tokens, req_pool_indices, seq_lens, ...
    ):
        """为 CUDAGraph 捕获准备注意力元数据
        FlashInfer: 创建 BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True)
                    存储到 decode_cuda_graph_metadata[bs]
        Triton:     通过 Triton kernel 计算 kv_indptr/kv_indices
        """

    def init_forward_metadata_replay_cuda_graph(
        self, bs, req_pool_indices, seq_lens, seq_lens_sum, ...
    ):
        """为 CUDAGraph 重放更新注意力元数据
        FlashInfer: 调用 indices_updater_decode.update() 更新 KV 索引
        Triton:     重新计算 kv_indptr/kv_indices/num_kv_splits
        """

    def get_cuda_graph_seq_len_fill_value(self):
        """返回填充序列长度的值
        FlashInfer: 返回 1
        Triton:     返回 0
        """
```

## 2.6 ModelRunner 的 CUDAGraph 调度逻辑

```python
# File: srt/model_executor/model_runner.py, line 2443
def _forward_raw(self, forward_batch, ...):
    # ===== Decode 路径：使用 CudaGraphRunner =====
    can_run_graph = (
        forward_batch.forward_mode.is_cuda_graph()   # DECODE/TARGET_VERIFY/IDLE
        and self.graph_runner                          # CudaGraphRunner 已初始化
        and self.graph_runner.can_run(forward_batch)   # batch_size 在范围内
    )
    if can_run_graph:
        ret = self.graph_runner.replay(forward_batch)  # ★ 直接 replay
        return ModelRunnerOutput(logits_output=ret, can_run_graph=True)

    # ===== 非 CUDAGraph 路径 =====
    if forward_batch.forward_mode.is_decode():
        ret = self.forward_decode(forward_batch)        # Decode 但超出 graph 范围
    elif forward_batch.forward_mode.is_extend():
        ret = self.forward_extend(forward_batch)        # Prefill/Extend
    ...

# File: srt/model_executor/model_runner.py, line 2307
def forward_extend(self, forward_batch, ...):
    # ===== Prefill 路径：尝试 PiecewiseCudaGraphRunner =====
    can_run_graph = (
        self.piecewise_cuda_graph_runner is not None
        and self.piecewise_cuda_graph_runner.can_run(forward_batch)
    )
    if can_run_graph:
        return self.piecewise_cuda_graph_runner.replay(forward_batch)  # ★ Piecewise replay

    # ===== 回退：直接 model.forward() =====
    self.attn_backend.init_forward_metadata(forward_batch)
    return self.model.forward(
        forward_batch.input_ids, forward_batch.positions, forward_batch
    )
```

---

# 第3部分：Prefill vs Decode — CUDAGraph 的差异化使用

## 3.1 ForwardMode 枚举回顾

```python
# File: srt/model_executor/forward_batch_info.py, line 74
class ForwardMode(IntEnum):
    EXTEND = auto()          # Prefill: 处理新输入的 token 序列
    DECODE = auto()          # Decode:  每个请求自回归生成 1 个 token
    MIXED = auto()           # 混合:    Chunked Prefill（同时含 extend 和 decode）
    IDLE = auto()            # 空闲:    DP Attention 时的填充 worker
    TARGET_VERIFY = auto()   # 投机解码: 在 target 模型上验证 draft tokens
    DRAFT_EXTEND = auto()    # 投机解码: 在 draft 模型上扩展
    SPLIT_PREFILL = auto()   # PD 多路复用: 分片 prefill
    DLLM_EXTEND = auto()     # 扩散 LLM: 扩展模式
```

**哪些模式使用 CUDAGraph？**

```python
# File: srt/model_executor/forward_batch_info.py
def is_cuda_graph(self):
    """返回 True 的模式会走 CudaGraphRunner.replay()"""
    return (
        self == ForwardMode.DECODE          # ✓ Decode
        or self == ForwardMode.TARGET_VERIFY # ✓ 投机解码验证
        or self == ForwardMode.IDLE          # ✓ 空闲填充
        or self == ForwardMode.DLLM_EXTEND   # ✓ 扩散 LLM
    )
    # 注意: EXTEND (Prefill) 不在此列！Prefill 走 PiecewiseCudaGraphRunner
```

## 3.2 Decode vs Prefill 详细对比

| 维度 | Decode | Prefill/Extend |
|------|--------|----------------|
| **ForwardMode** | `DECODE` | `EXTEND` (或 `MIXED`) |
| **每请求 token 数** | 固定 1 个 | 变化：几个到几千个 |
| **总 token 数** | = batch_size | = sum(各请求输入长度) |
| **批次形状** | 规整 (uniform) | 参差不齐 (ragged) |
| **CUDAGraph Runner** | `CudaGraphRunner` | `PiecewiseCudaGraphRunner` |
| **图索引键** | `batch_size` | `num_tokens` |
| **捕获策略** | 整体捕获 (monolithic) | 分片捕获 (piecewise via torch.compile) |
| **注意力 pattern** | Append-1 (追加单 token) | 变长序列 (ragged attention) |
| **KV Cache 写入** | 写入 1 个 token 到 cache | 写入整个序列到 cache |
| **对齐方式** | `bisect_left(capture_bs, raw_bs)` | `bisect_left(capture_num_tokens, num_tokens)` |
| **注意力后端接口** | `init_forward_metadata_replay_cuda_graph()` | `init_forward_metadata()` (标准接口) |
| **是否默认启用** | 是 | 否，需 `--enable-piecewise-cuda-graph` |

## 3.3 为什么需要不同策略？

### Decode 适合整体捕获

1. **形状固定**：每个请求恰好生成 1 个 token，总 token 数 = batch_size
2. **注意力 uniform**：所有请求都是 "追加一个 token" 模式
3. **无通信边界变化**：MoE 的 dispatch/combine token 数 = batch_size，可预测
4. **GPU 计算少、CPU 启动多**：正是 CUDA Graph 的最佳场景

→ 结论：一个 batch_size 对应一个完整 graph，简单高效。

### Prefill 需要分片捕获

1. **形状高度可变**：不同请求的输入长度差异很大（10 tokens vs 2000 tokens）
2. **注意力 ragged**：每个请求的 QKV 长度不同，需要 ragged attention
3. **MoE 通信改变形状**：DeepEP all-to-all 会重新分配 token 到不同 GPU，形状在通信前后不同
4. **GPU 计算量大、CPU 启动不是瓶颈**：Prefill 的计算量远大于 Decode

→ 结论：在 MoE 通信等边界处拆分模型，每个子图独立捕获，灵活应对形状变化。

### 为什么 Prefill 默认不启用 CUDAGraph？

Prefill 阶段的计算量远大于 CPU 启动开销，因此 CUDA Graph 带来的收益**相对较小**。
只在 MoE 模型（如 DeepSeek V3）且通信开销显著时，分片 CUDA Graph 才有明显价值。

## 3.4 Decode 路径完整调用链

```
Scheduler.event_loop_normal()                    # srt/managers/scheduler.py
  │
  ├─ recv_requests()                              # 接收新请求
  ├─ process_input_requests(recv_reqs)            # 加入等待队列
  ├─ batch = get_next_batch_to_run()              # 返回 running_batch (DECODE mode)
  │
  └─ run_batch(batch)
      │
      ├─ worker_batch = batch.get_model_worker_batch()  # ScheduleBatch → ModelWorkerBatch
      │                                                  # srt/managers/schedule_batch.py
      │
      └─ tp_worker.forward_batch_generation(worker_batch)  # srt/managers/tp_worker.py
          │
          ├─ forward_batch = ForwardBatch.init_new(worker_batch, model_runner)
          │   # ForwardMode = DECODE, batch_size = N, num_tokens = N
          │   # srt/model_executor/forward_batch_info.py
          │
          └─ model_runner.forward(forward_batch)           # srt/model_executor/model_runner.py:2387
              │
              └─ _forward_raw(forward_batch)                # line 2443
                  │
                  ├─ forward_batch.forward_mode.is_cuda_graph() == True  ✓
                  ├─ self.graph_runner.can_run(forward_batch) == True     ✓ (bs ≤ max_bs)
                  │
                  └─ self.graph_runner.replay(forward_batch)  # ★ CudaGraphRunner
                      │                                        # srt/model_executor/cuda_graph_runner.py:846
                      ├─ replay_prepare()                      # line 776
                      │   ├─ bisect_left → 对齐 bs            # 例: raw_bs=5 → bs=8
                      │   ├─ buffers.populate_from_forward_batch()  # 拷贝真实数据到静态缓冲
                      │   └─ attn_backend.init_forward_metadata_replay_cuda_graph()  # 更新 KV 索引
                      │
                      ├─ self.graphs[bs].replay()              # ★ CUDA Graph 重放！零 CPU 开销
                      │
                      └─ output_buffers[bs][:raw_bs]           # 裁剪到真实 batch_size
```

## 3.5 Prefill 路径完整调用链

### 场景 A：启用了 PiecewiseCudaGraphRunner

```
Scheduler.event_loop_normal()                    # srt/managers/scheduler.py
  │
  ├─ batch = get_next_batch_to_run()              # 返回新 prefill batch (EXTEND mode)
  │
  └─ run_batch(batch)
      │
      └─ tp_worker.forward_batch_generation(worker_batch)
          │
          ├─ forward_batch = ForwardBatch.init_new(...)
          │   # ForwardMode = EXTEND, num_tokens = 各请求输入 token 之和
          │
          └─ model_runner.forward(forward_batch)
              │
              └─ _forward_raw(forward_batch)
                  │
                  ├─ forward_batch.forward_mode.is_cuda_graph() == False  ✗ (EXTEND 不算)
                  ├─ forward_batch.forward_mode.is_extend() == True       ✓
                  │
                  └─ forward_extend(forward_batch)               # line 2307
                      │
                      ├─ piecewise_cuda_graph_runner.can_run() == True ✓
                      │
                      └─ piecewise_cuda_graph_runner.replay()    # ★ Piecewise replay
                          │
                          ├─ attn_backend.init_forward_metadata(forward_batch)  # 标准初始化
                          ├─ replay_prepare()
                          │   ├─ bisect_left → 对齐 num_tokens
                          │   └─ 拷贝数据到静态缓冲区
                          │
                          ├─ model.forward()                     # 调用完整 forward
                          │   └─ 内部每个 torch.compile 子图的
                          │      CUDAPiecewiseBackend.__call__()
                          │      └─ entry.cudagraph.replay()     # ★ 分片 replay
                          │
                          └─ output[:raw_num_tokens]             # 裁剪输出
```

### 场景 B：未启用 PiecewiseCudaGraphRunner（默认）

```
...
  └─ forward_extend(forward_batch)
      │
      ├─ piecewise_cuda_graph_runner is None  → can_run = False
      │
      ├─ attn_backend.init_forward_metadata(forward_batch)  # 标准初始化
      │
      └─ self.model.forward(                    # ★ 直接调用，无 CUDA Graph
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch
         )
```

## 3.6 Attention Backend 的模式差异

### Decode 模式（CudaGraphRunner 使用）

注意力后端使用 **专用的 CUDAGraph 接口**：

```python
# FlashInfer 示例:
# File: srt/layers/attention/flashinfer_backend.py

# Capture 时：创建带 use_cuda_graph=True 的 wrapper
def init_forward_metadata_capture_cuda_graph(self, bs, ...):
    wrapper = BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True)
    self.decode_cuda_graph_metadata[bs] = wrapper  # 每个 bs 一个 wrapper

# Replay 时：更新 KV 索引（序列在变化，需要更新）
def init_forward_metadata_replay_cuda_graph(self, bs, req_pool_indices, seq_lens, ...):
    wrapper = self.decode_cuda_graph_metadata[bs]
    wrapper.indices_updater_decode.update(req_pool_indices, seq_lens, ...)

# 填充值：FlashInfer 用 1，Triton 用 0
def get_cuda_graph_seq_len_fill_value(self):
    return 1  # FlashInfer
    # return 0  # Triton
```

### Prefill/Extend 模式（PiecewiseCudaGraphRunner 使用）

注意力后端使用 **标准接口**（非 CUDAGraph 专用）：

```python
# File: srt/layers/attention/flashinfer_backend.py

def init_forward_metadata(self, forward_batch: ForwardBatch):
    """标准初始化 — 处理 ragged 序列，构建 paged KV 索引"""
    if forward_batch.forward_mode.is_extend():
        # 创建 BatchPrefillWithPagedKVCacheWrapper
        # 处理变长序列的 qo_indptr, kv_indptr, kv_indices
        ...
```

**关键区别**：
- Decode 的 replay 接口**只更新动态数据**（KV 索引），元数据结构固定
- Prefill 的标准接口**完全重建**元数据，因为每次的序列数量和长度都不同
- Piecewise 的 graph replay 发生在**模型子图级别**，attention 层本身不感知 CUDAGraph

---

# 第4部分：DeepSeek V3.2 模型推理调用层级

## 4.1 模型文件定位

DeepSeek V2/V3/V3.2 共用一个主文件，通过继承关系区分：

```python
# File: srt/models/deepseek_v2.py, line 2978-2986

class DeepseekV2ForCausalLM(nn.Module):
    """DeepSeek V2 基类 — 包含完整实现"""
    ...

class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass  # V3 直接继承 V2，无额外修改

class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass  # V3.2 也直接继承 V2

# 注册入口
EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
```

相关文件：

| 文件 | 内容 |
|------|------|
| `srt/models/deepseek_v2.py` | 主模型文件：V2/V3/V3.2 |
| `srt/models/deepseek_nextn.py` | MTP (Multi-Token Prediction) 头 |
| `srt/models/deepseek_common/__init__.py` | 共享工具 |
| `srt/models/deepseek_common/attention_backend_handler.py` | 注意力后端分发 |
| `srt/models/deepseek_common/attention_forward_methods/` | 各种注意力前向方法 |
| `srt/models/deepseek_common/deepseek_weight_loader.py` | 权重加载逻辑 |

## 4.2 完整调用层级

以下是一次 Decode 推理的完整调用链（MLA Absorbed 路径 + MoE）：

```
ModelRunner.forward() / graph_runner.replay()
  │
  ▼
DeepseekV2ForCausalLM.forward()                    # deepseek_v2.py:2901
  │
  ├─► self.model(input_ids, positions, forward_batch)  # DeepseekV2Model.forward()
  │     │                                               # deepseek_v2.py:2643
  │     │
  │     ├─ embed_tokens(input_ids)                      # VocabParallelEmbedding
  │     │
  │     ├─ for i in range(start_layer, end_layer):      # 遍历所有 Decoder 层
  │     │     │
  │     │     ▼
  │     │   DeepseekV2DecoderLayer.forward()            # deepseek_v2.py:2351
  │     │     │
  │     │     ├─ layer_communicator.prepare_attn()       # input_layernorm + (reduce_scatter)
  │     │     │     └─ fused_add_rmsnorm()               # sgl-kernel: 融合加法+RMSNorm
  │     │     │
  │     │     ├─► DeepseekV2AttentionMLA.forward()       # deepseek_v2.py:1358
  │     │     │     │
  │     │     │     ├─ forward_prepare()                  # line 1375
  │     │     │     │   └─ dispatch → forward_absorb_prepare()  # MLA Absorbed 路径
  │     │     │     │       ├─ dsv3_fused_a_gemm()        # ★ CUDA kernel: 融合 QKV-A 投影
  │     │     │     │       ├─ q_a_layernorm(q)           # RMSNorm
  │     │     │     │       ├─ kv_a_layernorm(k_nope)     # RMSNorm
  │     │     │     │       ├─ q_b_proj(q)                # ColumnParallelLinear
  │     │     │     │       ├─ bmm_fp8(q_nope, w_kc)      # ★ 权重吸收 (Weight Absorption)
  │     │     │     │       └─ rotary_emb(q_pe, k_pe)     # RoPE 旋转位置编码
  │     │     │     │
  │     │     │     └─ forward_core()                     # line 1447
  │     │     │         └─ forward_absorb_core()
  │     │     │             ├─ attn_mqa(q, k, v, batch)   # ★ RadixAttention → backend
  │     │     │             │   └─ attn_backend.forward_decode()
  │     │     │             │       └─ cutlass_mla_decode() / flash_mla / flashinfer_mla
  │     │     │             ├─ bmm_fp8(attn_out, w_vc)    # ★ 值反吸收 (V De-absorption)
  │     │     │             └─ o_proj(output)              # RowParallelLinear
  │     │     │
  │     │     ├─ layer_communicator.prepare_mlp()         # post_attention_layernorm
  │     │     │     └─ fused_add_rmsnorm()
  │     │     │
  │     │     ├─► DeepseekV2MoE.forward()                 # deepseek_v2.py:624+
  │     │     │     │                                      # (仅稀疏层; 密集层用 DeepseekV2MLP)
  │     │     │     ├─ gate(hidden_states)                 # MoEGate: Router GEMM
  │     │     │     │   └─ dsv3_router_gemm()             # ★ CUDA kernel
  │     │     │     │
  │     │     │     ├─ topk(hidden_states, router_logits)  # TopK 专家选择
  │     │     │     │   └─ topk_sigmoid()                 # ★ CUDA kernel (noaux_tc)
  │     │     │     │
  │     │     │     ├─ experts(hidden_states, topk_out)    # FusedMoE 专家计算
  │     │     │     │   ├─ dispatcher.dispatch()           # Token 分发到专家
  │     │     │     │   ├─ run_moe_core()                  # ★ 专家 GEMM (Triton/DeepGemm/CUTLASS)
  │     │     │     │   └─ dispatcher.combine()            # Token 汇聚
  │     │     │     │
  │     │     │     ├─ shared_experts(hidden_states)       # 共享专家 MLP (可在 alt_stream 并行)
  │     │     │     │   └─ gate_up_proj → SiluAndMul → down_proj
  │     │     │     │
  │     │     │     ├─ final = routed_output * scaling + shared_output
  │     │     │     └─ tensor_model_parallel_all_reduce()  # TP all-reduce
  │     │     │
  │     │     └─ layer_communicator.postprocess_layer()    # (all-reduce / reduce-scatter)
  │     │
  │     └─ norm(hidden_states)                            # 最后一层 RMSNorm
  │
  └─► logits_processor(input_ids, hidden_states, lm_head)  # LogitsProcessor
        └─ F.linear(hidden_states, lm_head.weight)          # 输出投影
```

## 4.3 MLA (Multi-head Latent Attention) 详解

### 4.3.1 MLA 核心思想

MLA 将 KV 压缩到低秩**潜变量空间 (latent space)**，大幅减少 KV Cache 显存：

```
标准 MHA:  KV Cache = num_heads × head_dim × 2           (例: 128×128×2 = 32768)
MLA:       KV Cache = kv_lora_rank + qk_rope_head_dim    (例: 512+64 = 576)
                     ↑ 压缩比约 57:1！
```

### 4.3.2 DeepseekV2AttentionMLA 投影层

```python
# File: srt/models/deepseek_v2.py, line 1067
class DeepseekV2AttentionMLA(nn.Module, DeepseekMHAForwardMixin):
    def __init__(self, config, ...):
        # DeepSeek V3 典型维度:
        # qk_nope_head_dim = 128    q/k 的非 RoPE 部分
        # qk_rope_head_dim = 64     q/k 的 RoPE 部分
        # qk_head_dim = 192         = 128 + 64
        # v_head_dim = 128          v 的维度
        # kv_lora_rank = 512        KV 潜变量维度
        # q_lora_rank = 1536        Q 潜变量维度
        # num_heads = 128

        # ★ 融合 QKV-A 投影: hidden → (q_latent, kv_latent, k_rope)
        self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,                                              # 7168
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,          # 1536+512+64=2112
            bias=False,
        )

        # Q 上投影: q_latent → num_heads × qk_head_dim
        self.q_a_layernorm = RMSNorm(q_lora_rank)           # 1536
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank, num_heads * qk_head_dim,            # 1536 → 128×192=24576
        )

        # KV LayerNorm
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)         # 512

        # KV 上投影 (仅 MHA 路径使用)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim),
        )

        # ★ 两个 RadixAttention 实例
        self.attn_mqa = RadixAttention(
            num_heads=1,  # "单头"注意力，head_dim = kv_lora_rank + qk_rope_head_dim
            head_dim=kv_lora_rank + qk_rope_head_dim,   # 576
            v_head_dim=kv_lora_rank,                     # 512
        )
        self.attn_mha = RadixAttention(
            num_heads=num_local_heads,  # 标准多头注意力 (回退路径)
            head_dim=qk_nope_head_dim + qk_rope_head_dim,  # 192
        )

        # 输出投影
        self.o_proj = RowParallelLinear(num_heads * v_head_dim, hidden_size)

        # ★ 吸收权重矩阵 (在权重加载时从 kv_b_proj 分解得到)
        # w_kc: [num_heads, qk_nope_head_dim, kv_lora_rank]  用于 Q×K 吸收
        # w_vc: [num_heads, kv_lora_rank, v_head_dim]         用于 Attn_out×V 吸收
```

### 4.3.3 注意力前向方法分发

```python
# File: srt/models/deepseek_v2.py, line 1406
def forward_prepare(self, positions, hidden_states, forward_batch, ...):
    attn_forward_method = self.dispatch_attn_forward_method(forward_batch)

    if attn_forward_method == AttnForwardMethod.MHA:
        # 标准 MHA 路径 — 用于不支持 MLA 的后端
        return self.forward_normal_prepare(...)

    elif attn_forward_method == AttnForwardMethod.MLA:
        # ★ MLA Absorbed 路径 — 主要性能路径
        return self.forward_absorb_prepare(...)

    elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE:
        # ROCm 融合 RoPE 路径
        return self.forward_absorb_fused_mla_rope_prepare(...)
    ...
```

### 4.3.4 MLA Absorbed 路径详解（核心性能路径）

```python
# File: srt/models/deepseek_v2.py, line 1525
def forward_absorb_prepare(self, positions, hidden_states, forward_batch, ...):

    # ===== Step 1: 融合 QKV-A 投影 =====
    # 一次 GEMM 同时得到 q_latent, kv_latent, k_rope
    # 小 batch 时使用 dsv3_fused_a_gemm CUDA kernel 优化
    qkv_latent = get_attn_tp_context().fetch_qkv_latent()
    q, latent_cache = qkv_latent.split(
        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
    )
    # q:            [tokens, 1536]
    # latent_cache: [tokens, 576]  = kv_lora_rank(512) + qk_rope_head_dim(64)
    k_nope = latent_cache[..., :self.kv_lora_rank]   # [tokens, 512]

    # ===== Step 2: LayerNorm =====
    q = self.q_a_layernorm(q)            # RMSNorm on q latent
    k_nope = self.kv_a_layernorm(k_nope) # RMSNorm on kv latent

    # ===== Step 3: Q 上投影 =====
    q = self.q_b_proj(q)                 # [tokens, num_local_heads × qk_head_dim]
    q = q.view(-1, self.num_local_heads, self.qk_head_dim)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    # q_nope: [tokens, num_local_heads, 128]
    # q_pe:   [tokens, num_local_heads, 64]

    # ===== Step 4: ★ Q 权重吸收 (Weight Absorption) =====
    # 避免显式构建完整 K 矩阵，直接将 kv_b_proj 的权重"吸收"到 Q 中
    # q_nope_out = q_nope × w_kc   (BatchedMatMul)
    q_nope_out = bmm_fp8(q_nope, self.w_kc)
    # 或使用 DeepGemm: grouped_gemm_nt_f8f8bf16_masked()
    # q_nope_out: [tokens, num_local_heads, kv_lora_rank=512]

    # ===== Step 5: RoPE =====
    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

    return q_nope_out, latent_cache, q_pe, k_pe, ...

# File: srt/models/deepseek_v2.py (forward_absorb_core)
def forward_absorb_core(self, inner_state):
    q_nope_out, latent_cache, q_pe, k_pe, forward_batch = inner_state

    # ===== Step 6: 注意力计算 =====
    # 使用 attn_mqa（"单头"注意力，KV 在 latent space）
    attn_output = self.attn_mqa(
        q=q_nope_out,        # [tokens, num_local_heads, 512]
        k=latent_cache,       # KV Cache 中的 latent: [tokens, 1, 576]
        v=latent_cache,       # 同上
        forward_batch=forward_batch,
        q_rope=q_pe,          # [tokens, num_local_heads, 64]
        k_rope=k_pe,          # [tokens, 1, 64]
    )
    # attn_output: [tokens, num_local_heads, kv_lora_rank=512]

    # ===== Step 7: ★ V 权重反吸收 (V De-absorption) =====
    attn_output = bmm_fp8(attn_output, self.w_vc)
    # attn_output: [tokens, num_local_heads, v_head_dim=128]

    # ===== Step 8: 输出投影 =====
    output = self.o_proj(attn_output.reshape(-1, self.num_local_heads * self.v_head_dim))
    return output
```

### 4.3.5 MHA 标准路径（回退）

当注意力后端不支持 MLA 时，使用标准多头注意力：

```python
# File: srt/models/deepseek_common/attention_forward_methods/forward_mha.py
class DeepseekMHAForwardMixin:
    def forward_normal_prepare(self, positions, hidden_states, forward_batch, ...):
        # 1. 计算 q 和 kv latent（与 absorbed 路径相同）
        q, latent_cache = ...

        # 2. ★ 显式上投影 K 和 V
        kv = self.kv_b_proj(kv_a)  # 解压缩到完整的 K, V 头
        k_nope, v = kv.split(...)

        # 3. 拼接 K
        k = torch.cat([k_nope, k_pe], dim=-1)

        # 4. 标准多头注意力
        attn_output = self.attn_mha(q, k, v, forward_batch)
        return self.o_proj(attn_output)
```

## 4.4 MoE (Mixture of Experts) 层详解

### 4.4.1 MoE 层判断

```python
# File: srt/models/deepseek_v2.py, line 2344
def _is_layer_sparse(self, layer_id, is_nextn):
    """判断某层是否使用 MoE"""
    return is_nextn or (
        config.n_routed_experts is not None
        and layer_id >= config.first_k_dense_replace  # 前几层用密集 MLP
        and layer_id % config.moe_layer_freq == 0     # 每隔 N 层用一次 MoE
    )
```

DeepSeek V3 典型配置：前 1 层密集 MLP，之后每层都是 MoE（256 个路由专家 + 1 个共享专家）。

### 4.4.2 DeepseekV2MoE 结构

```python
# File: srt/models/deepseek_v2.py, line 367
class DeepseekV2MoE(nn.Module):
    def __init__(self, config, layer_id, ...):
        # 路由门控
        self.gate = MoEGate(config, ...)          # 线性层: hidden → num_experts

        # TopK 专家选择
        self.topk = TopK(
            top_k=config.num_experts_per_tok,      # 每 token 选 8 个专家
            num_expert_group=config.n_group,        # 8 个专家组
            topk_group=config.topk_group,           # 每组选 topk_group 个
            correction_bias=self.gate.e_score_correction_bias,  # noaux_tc 偏置
            ...
        )

        # 专家执行 (FusedMoE 或 DeepEPMoE)
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,    # 256 个路由专家
            top_k=config.num_experts_per_tok,       # 8
            hidden_size=config.hidden_size,         # 7168
            intermediate_size=config.moe_intermediate_size,  # 2048
            ...
        )

        # 共享专家 (始终激活)
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            ...
        )
```

### 4.4.3 MoE 前向流程

```python
# File: srt/models/deepseek_v2.py (forward_normal 方法)
def forward_normal(self, hidden_states, forward_batch, ...):
    # 1. 共享专家（可在 alt_stream 上并行执行）
    shared_output = self._forward_shared_experts(hidden_states)

    # 2. 路由门控: hidden_states → router_logits
    router_logits = self.gate(hidden_states)
    # 小 batch 时使用 dsv3_router_gemm() CUDA kernel 加速

    # 3. TopK 专家选择
    topk_output = self.topk(hidden_states, router_logits)
    # 使用 topk_sigmoid() CUDA kernel (DeepSeek V3 的 noaux_tc 方法)
    # 或 topk_softmax() kernel

    # 4. ★ FusedMoE 专家计算
    final_hidden_states = self.experts(hidden_states, topk_output)
    #   内部流程:
    #   ├─ dispatcher.dispatch(hidden_states, topk_output)  # Token 分发到对应专家
    #   ├─ run_moe_core(dispatch_output)                     # 专家 GEMM 计算
    #   │   └─ 根据后端选择:
    #   │       ├─ TritonRunnerCore:    Triton fused_moe kernel
    #   │       ├─ DeepGemmRunnerCore:  DeepGemm FP8 grouped GEMM
    #   │       └─ CUTLASS:             cutlass_moe kernels
    #   └─ dispatcher.combine(combine_input)                 # Token 汇聚

    # 5. 合并
    final_hidden_states *= self.routed_scaling_factor
    final_hidden_states += shared_output

    # 6. TP all-reduce
    tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states
```

### 4.4.4 Expert Parallelism (EP) 后端

当使用 Expert Parallelism 时，MoE 层使用 `DeepEPMoE` 替代 `FusedMoE`：

```python
# File: srt/layers/moe/ep_moe/layer.py
class DeepEPMoE(FusedMoE):
    """使用 DeepEP 协议的 Expert Parallel MoE
    - 通过 all-to-all 通信将 token 分发到拥有对应专家的 GPU
    - 每个 GPU 只持有部分专家的权重
    """
```

Token 分发器：

| 分发器 | 文件 | 说明 |
|--------|------|------|
| `StandardDispatcher` | `srt/layers/moe/token_dispatcher/` | 基础分发/汇聚 |
| `DeepEPDispatcher` | `srt/layers/moe/token_dispatcher/` | DeepEP all-to-all |
| `FlashinferDispatcher` | `srt/layers/moe/token_dispatcher/` | FlashInfer 分发 |
| `MoriEPDispatcher` | `srt/layers/moe/token_dispatcher/` | MORI EP 分发 |

## 4.5 自定义 CUDA 内核 (sgl-kernel/)

### 4.5.1 DeepSeek 专用 GEMM 内核

| 内核 | CUDA 文件 | Python 接口 | 用途 |
|------|----------|-------------|------|
| `dsv3_fused_a_gemm` | `csrc/gemm/dsv3_fused_a_gemm.cu` | `sgl_kernel.dsv3_fused_a_gemm` | 融合 QKV-A 投影 (小 batch 优化) |
| `dsv3_router_gemm` | `csrc/gemm/dsv3_router_gemm_entry.cu` | `sgl_kernel.dsv3_router_gemm` | MoE Router GEMM (256 experts) |
| `bmm_fp8` | `csrc/gemm/bmm_fp8.cu` | `sgl_kernel.bmm_fp8` | FP8 Batched MatMul (权重吸收) |

### 4.5.2 MLA 注意力内核

| 内核 | CUDA 文件 | 用途 |
|------|----------|------|
| `cutlass_mla_decode` | `csrc/attention/cutlass_mla_kernel.cu` | CUTLASS SM90+ MLA Decode |
| `flash_mla_with_kvcache` | `csrc/flashmla_extension.cc` | FlashMLA Decode (DeepSeek 原版) |
| `concat_mla_k` | `csrc/elementwise/concat_mla.cu` | MLA K 拼接: (k_nope, k_pe) → k |
| `concat_mla_absorb_q` | `csrc/elementwise/concat_mla.cu` | MLA Q 吸收拼接 |
| SM100 MLA | `csrc/attention/cutlass_sm100_mla/` | Blackwell 专用 MLA 内核 |

### 4.5.3 MoE 内核

| 内核 | CUDA 文件 | 用途 |
|------|----------|------|
| `topk_sigmoid` | `csrc/moe/moe_topk_sigmoid_kernels.cu` | TopK sigmoid (noaux_tc) |
| `topk_softmax` | `csrc/moe/moe_topk_softmax_kernels.cu` | TopK softmax |
| `moe_fused_gate` | `csrc/moe/moe_fused_gate.cu` | 融合门控 |
| `moe_align_block_size` | `csrc/moe/moe_align_kernel.cu` | Token 对齐到 block |
| `fp8_blockwise_scaled_grouped_mm` | `csrc/moe/fp8_blockwise_moe_kernel.cu` | FP8 Blockwise MoE GEMM |
| `moe_sum` | `csrc/moe/moe_sum.cu` | MoE 求和规约 |
| `prepare_moe_input` | `csrc/moe/prepare_moe_input.cu` | MoE 输入准备 |

### 4.5.4 外部 GEMM 库

除了 sgl-kernel 中的自定义内核外，DeepSeek 推理还使用：

```python
# File: srt/layers/deep_gemm_wrapper.py
# DeepGemm: 外部 FP8 grouped GEMM 库
deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked()  # MoE 专家 GEMM
deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked()  # MLA 权重吸收 BMM
```

## 4.6 DeepSeek V3.2 特有功能

### 4.6.1 NSA (Native Sparse Attention)

DeepSeek V3.2 引入了 **原生稀疏注意力 (NSA)**，减少长序列的注意力计算量：

```python
# File: srt/models/deepseek_v2.py
# V3.2 通过 config 中的 nsa 标记启用
def is_deepseek_nsa(config):
    return getattr(config, "use_nsa", False)

# NSA 组件:
# - Indexer: 选择需要关注的稀疏 token 位置
# - NSA Attention Backend: srt/layers/attention/nsa_backend.py
# - Context Parallelism: NSACPLayerCommunicator
```

### 4.6.2 Context Parallelism (CP)

V3.2 的 NSA 支持 Context Parallelism，将长序列的注意力计算分散到多个 GPU：

```python
# File: srt/models/deepseek_v2.py, line 1101-1107
self.use_nsa = is_deepseek_nsa(config)
self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
if self.nsa_enable_prefill_cp:
    assert self.use_nsa, "CP currently only supports deepseek v3.2 model"
    self.cp_size = get_attention_cp_size()
```

### 4.6.3 注意力后端选择

DeepSeek 模型支持多种注意力后端，通过 `AttentionBackendRegistry` 分发：

```python
# File: srt/models/deepseek_common/attention_backend_handler.py

# 已注册的后端及其前向方法:
# flashinfer → AttnForwardMethod.MLA (Absorbed)
# fa3       → AttnForwardMethod.MHA (Standard)
# flashmla  → AttnForwardMethod.MLA (Absorbed)
# cutlass_mla → AttnForwardMethod.MLA (Absorbed)
# trtllm_mla  → AttnForwardMethod.MLA (Absorbed)
# triton    → AttnForwardMethod.MHA (Standard)
# nsa       → AttnForwardMethod.MLA (Absorbed)
# aiter     → AttnForwardMethod.MLA (Absorbed, AMD)
```

---

# 附录

## A. CUDAGraph 相关配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--disable-cuda-graph` | `False` | 完全禁用 CUDAGraph |
| `--cuda-graph-max-bs` | `None` (自动) | Decode CUDAGraph 最大 batch_size |
| `--cuda-graph-bs` | `None` (自动) | 自定义捕获的 batch_size 列表 |
| `--disable-cuda-graph-padding` | `False` | 禁用 padding，每个 bs 单独捕获 |
| `--enable-torch-compile` | `False` | 对 Decode graph 启用 torch.compile |
| `--torch-compile-max-bs` | `32` | torch.compile 的最大 batch_size |
| `--enable-piecewise-cuda-graph` | `False` | 启用 Prefill 分片 CUDAGraph |
| `--piecewise-cuda-graph-max-tokens` | `None` | Piecewise 最大 token 数 |
| `--piecewise-cuda-graph-tokens` | `None` | 自定义 Piecewise 捕获的 token 列表 |
| `--enable-memory-saver` | `False` | 启用显存节省（CUDAGraph 标签管理） |
| `--enable-cudagraph-gc` | `False` | 允许在 CUDAGraph 捕获时运行 GC |

## B. 完整文件路径索引表

所有路径相对于 `sglang/python/sglang/`：

### 入口与引擎

| 组件 | 路径 |
|------|------|
| Engine | `srt/entrypoints/engine.py` |
| EngineBase | `srt/entrypoints/EngineBase.py` |
| HTTP Server | `srt/entrypoints/http_server.py` |
| ServerArgs | `srt/server_args.py` |

### 调度与管理

| 组件 | 路径 |
|------|------|
| Scheduler | `srt/managers/scheduler.py` |
| TokenizerManager | `srt/managers/tokenizer_manager.py` |
| DetokenizerManager | `srt/managers/detokenizer_manager.py` |
| TpModelWorker | `srt/managers/tp_worker.py` |
| ScheduleBatch / Req | `srt/managers/schedule_batch.py` |
| SchedulePolicy | `srt/managers/schedule_policy.py` |
| IO Struct | `srt/managers/io_struct.py` |
| DataParallelController | `srt/managers/data_parallel_controller.py` |

### 模型执行

| 组件 | 路径 |
|------|------|
| ModelRunner | `srt/model_executor/model_runner.py` |
| ForwardBatch / ForwardMode | `srt/model_executor/forward_batch_info.py` |
| CudaGraphRunner | `srt/model_executor/cuda_graph_runner.py` |
| PiecewiseCudaGraphRunner | `srt/model_executor/piecewise_cuda_graph_runner.py` |
| GraphInputBuffers | `srt/model_executor/input_buffers.py` |
| KV Cache Mixin | `srt/model_executor/model_runner_kv_cache_mixin.py` |

### 模型实现

| 组件 | 路径 |
|------|------|
| DeepSeek V2/V3/V3.2 | `srt/models/deepseek_v2.py` |
| DeepSeek NextN (MTP) | `srt/models/deepseek_nextn.py` |
| DeepSeek Common | `srt/models/deepseek_common/` |
| Attention Backend Handler | `srt/models/deepseek_common/attention_backend_handler.py` |
| MHA Forward Methods | `srt/models/deepseek_common/attention_forward_methods/` |
| Model Registry | `srt/models/registry.py` |
| LLaMA (参考实现) | `srt/models/llama.py` |

### 层与算子

| 组件 | 路径 |
|------|------|
| RadixAttention | `srt/layers/radix_attention.py` |
| AttentionBackend (基类) | `srt/layers/attention/base_attn_backend.py` |
| FlashInfer Backend | `srt/layers/attention/flashinfer_backend.py` |
| FlashInfer MLA Backend | `srt/layers/attention/flashinfer_mla_backend.py` |
| FlashMLA Backend | `srt/layers/attention/flashmla_backend.py` |
| CUTLASS MLA Backend | `srt/layers/attention/cutlass_mla_backend.py` |
| Triton Backend | `srt/layers/attention/triton_backend.py` |
| NSA Backend | `srt/layers/attention/nsa_backend.py` |
| Attention Registry | `srt/layers/attention/attention_registry.py` |
| TopK | `srt/layers/moe/topk.py` |
| FusedMoE | `srt/layers/moe/fused_moe_triton/layer.py` |
| DeepEPMoE | `srt/layers/moe/ep_moe/layer.py` |
| DeepGemm Wrapper | `srt/layers/deep_gemm_wrapper.py` |
| Linear Layers | `srt/layers/linear.py` |
| LogitsProcessor | `srt/layers/logits_processor.py` |
| Sampler | `srt/layers/sampler.py` |
| RoPE | `srt/layers/rotary_embedding.py` |
| RMSNorm | `srt/layers/layernorm.py` |

### 显存管理

| 组件 | 路径 |
|------|------|
| Memory Pool | `srt/mem_cache/memory_pool.py` |
| Radix Cache | `srt/mem_cache/radix_cache.py` |
| Allocator | `srt/mem_cache/allocator.py` |

### 编译

| 组件 | 路径 |
|------|------|
| CUDAPiecewiseBackend | `srt/compilation/cuda_piecewise_backend.py` |
| Piecewise Context | `srt/compilation/piecewise_context_manager.py` |
| Compile Utils | `srt/compilation/compile.py` |

### CUDA 内核 (sgl-kernel/)

| 内核类别 | 路径 |
|----------|------|
| DeepSeek GEMM | `sgl-kernel/csrc/gemm/dsv3_*.cu` |
| BMM FP8 | `sgl-kernel/csrc/gemm/bmm_fp8.cu` |
| CUTLASS MLA | `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu` |
| FlashMLA | `sgl-kernel/csrc/flashmla_extension.cc` |
| MoE TopK | `sgl-kernel/csrc/moe/moe_topk_*.cu` |
| MoE Align | `sgl-kernel/csrc/moe/moe_align_kernel.cu` |
| FP8 MoE | `sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu` |
| MLA Concat | `sgl-kernel/csrc/elementwise/concat_mla.cu` |

---

> 本文档基于 SGLang 源码分析生成，版本时间：2026-03-29。
