# SGLang 设计模式与架构哲学

> 本文档系统性梳理 SGLang 项目中使用的核心设计模式、架构哲学与工程思想。
> 每章由浅入深，包含背景动机、设计哲学、核心代码片段与工作流程。

---

## 第1章：项目整体架构概览

### 背景与动机

SGLang 是一个高性能 LLM 推理服务框架，需要同时满足：
- **高吞吐**：最大化 GPU 利用率
- **低延迟**：快速响应单个请求
- **灵活性**：支持多种模型、多种硬件、多种部署模式

为此，SGLang 采用三层架构 + 多进程流水线设计。

### 设计哲学

> **分层解耦 + 异步流水线**：每层职责单一，通过 IPC 连接形成流水线，CPU/GPU 计算重叠执行。

### 三层架构

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Frontend DSL (sglang.lang)                │
│  - 用户编程接口、Prompt 模板、约束解码语法           │
├─────────────────────────────────────────────────────┤
│  Layer 2: Serving Runtime (sglang.srt)              │
│  - 请求调度、KV 缓存管理、连续批处理、投机解码       │
├─────────────────────────────────────────────────────┤
│  Layer 3: Custom Kernels (sgl-kernel / jit_kernel)  │
│  - CUDA/Triton 高性能算子、FlashMLA、PagedAttention │
└─────────────────────────────────────────────────────┘
```

### 进程模型

```
                    ┌──────────────────┐
                    │  Engine (入口)    │
                    │  engine.py       │
                    └────────┬─────────┘
                             │ ZMQ
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
  │TokenizerMgr  │  │  Scheduler   │  │DetokenizerMgr    │
  │(tokenize)    │  │(调度+批处理)  │  │(output→text)     │
  └──────────────┘  └──────┬───────┘  └──────────────────┘
                           │
                    ┌──────┴───────┐
                    │TpModelWorker │  × tp_size
                    │(模型前向计算) │
                    └──────────────┘
```

### 核心代码

```python
# python/sglang/srt/entrypoints/engine.py (简化)
class Engine(EngineBase):
    """SGLang 推理引擎入口。

    职责：
    1. 解析 ServerArgs，初始化所有子进程
    2. 启动 TokenizerManager / Scheduler / DetokenizerManager
    3. 提供 generate() / encode() 等用户 API
    """
    def __init__(self, server_args: ServerArgs):
        # 1. 启动 Detokenizer 进程
        self.detokenizer_process = run_detokenizer_process(...)

        # 2. 启动 Scheduler 进程 (每个 DP rank 一个)
        for dp_rank in range(dp_size):
            self.scheduler_processes.append(
                run_scheduler_process(server_args, gpu_id, tp_rank, dp_rank, ...)
            )

        # 3. 启动 TokenizerManager (本进程内)
        self.tokenizer_manager = TokenizerManager(server_args, port_args)

    async def generate(self, input: GenerateReqInput) -> Dict:
        """用户调用入口 → TokenizerManager → Scheduler → Model → Detokenizer"""
        return await self.tokenizer_manager.generate_request(input)
```

```python
# python/sglang/srt/managers/scheduler.py:286
class Scheduler(
    SchedulerDisaggregationDecodeMixin,    # PD 分离解码
    SchedulerDisaggregationPrefillMixin,   # PD 分离预填充
    SchedulerMultiplexMixin,               # 多路复用
    SchedulerPPMixin,                      # 流水线并行
    SchedulerDllmMixin,                    # DLLM 支持
    SchedulerMlxOverlapMixin,              # MLX Overlap
):
    """核心调度器 — 管理请求生命周期、内存分配、批处理策略。

    主循环：
    1. recv_requests()     — 从 TokenizerManager 接收新请求
    2. get_next_batch()    — 根据策略组装批次
    3. run_batch()         — 调用 TpModelWorker 执行前向
    4. process_result()    — 处理输出、释放内存、流式返回
    """
```

### 关键文件索引

| 组件 | 文件路径 |
|------|----------|
| Engine 入口 | `python/sglang/srt/entrypoints/engine.py` |
| Scheduler | `python/sglang/srt/managers/scheduler.py` |
| TpModelWorker | `python/sglang/srt/managers/tp_worker.py` |
| ModelRunner | `python/sglang/srt/model_executor/model_runner.py` |
| TokenizerManager | `python/sglang/srt/managers/tokenizer_manager.py` |
| DetokenizerManager | `python/sglang/srt/managers/detokenizer_manager.py` |

### 小结

1. **三层架构**将用户接口、运行时调度、底层算子彻底解耦
2. **多进程模型**利用 ZMQ 进行 IPC，各组件可独立扩展
3. **Scheduler 是核心**，通过 Mixin 组合承载 6 种扩展能力
4. **异步流水线**让 tokenize/schedule/forward/detokenize 并行执行

---

## 第2章：Mixin 组合模式

### 背景与动机

SGLang 中的核心类（如 Scheduler、TokenizerManager、AttentionBackend）功能庞大且需要灵活扩展。传统的深继承链会导致：
- 类之间耦合度过高
- 新增功能必须修改基类
- 代码难以理解和维护

SGLang 选择 **Mixin 组合模式**：将功能按职责拆分到独立的 Mixin 类中，通过 Python 多继承组合。

### 设计哲学

> **组合优于继承**：每个 Mixin 封装一个独立关注点，核心类通过多继承"混入"所需能力。

### 核心代码

#### 示例 1：Scheduler — 6 个 Mixin 组合

```python
# python/sglang/srt/managers/scheduler.py:286
class Scheduler(
    SchedulerDisaggregationDecodeMixin,    # PD 分离 — 解码侧逻辑
    SchedulerDisaggregationPrefillMixin,   # PD 分离 — 预填充侧逻辑
    SchedulerMultiplexMixin,               # 多路复用（多模型共享 GPU）
    SchedulerPPMixin,                      # Pipeline Parallelism 支持
    SchedulerDllmMixin,                    # Draft-LLM 联合推理
    SchedulerMlxOverlapMixin,              # MLX 平台 overlap 支持
):
    """核心调度器。

    每个 Mixin 文件独立维护，互不依赖：
    - scheduler_disagg_decode_mixin.py
    - scheduler_disagg_prefill_mixin.py
    - scheduler_multiplex_mixin.py
    - scheduler_pp_mixin.py
    - scheduler_dllm_mixin.py
    - scheduler_mlx_overlap_mixin.py
    """
```

#### 示例 2：TokenizerManager — 2 个 Mixin

```python
# python/sglang/srt/managers/tokenizer_manager.py:223
class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin):
    """TokenizerManager 通过 Mixin 获得：
    - TokenizerControlMixin: 动态切换 tokenizer、加载新词表
    - TokenizerManagerScoreMixin: scoring/embedding 请求处理
    """
```

#### 示例 3：DeepSeekV4 Attention Backend — 3 个 Mixin

```python
# python/sglang/srt/layers/attention/deepseek_v4_backend.py:332
class DeepseekV4AttnBackend(
    AttentionBackend,           # 抽象基类：定义 forward/init_metadata 接口
    C4IndexerBackendMixin,      # C4 索引器：管理 compressed KV 的索引
    CompressorBackendMixin,     # 压缩器：MLA latent → compressed KV
):
    """三个关注点完全独立：
    - AttentionBackend: 注意力计算的统一接口
    - C4IndexerBackendMixin: 索引管理 (哪些 token 被压缩到哪里)
    - CompressorBackendMixin: KV 压缩算法 (latent projection + RoPE)
    """
```

### Mixin 设计规范

| 规则 | 说明 |
|------|------|
| 单一职责 | 每个 Mixin 只封装一个功能切面 |
| 无状态依赖 | Mixin 不应假设宿主类有哪些属性（通过 `self` 访问时需在 `__init__` 中初始化） |
| 独立文件 | 每个 Mixin 放在独立文件中，文件名以 `_mixin.py` 结尾 |
| MRO 顺序 | 左边的 Mixin 优先级高于右边（Python C3 线性化） |

### 工作流程

```
功能需求来了 → 是否属于已有 Mixin 的职责？
    ├── 是 → 在对应 Mixin 文件中添加方法
    └── 否 → 创建新的 Mixin 类 → 加入宿主类的继承列表
```

### 与其他模式的关联

- Mixin 内部常使用 **Strategy 模式**（如不同的调度策略）
- Mixin 通过 **Plugin/Hook 系统** 可被外部扩展覆盖

### 小结

1. SGLang 的 Scheduler 通过 6 个 Mixin 实现了 PD 分离、PP、多路复用等能力
2. Mixin 让每个功能可以独立开发、测试、review
3. 新增功能只需添加新 Mixin，不修改已有代码（开放-封闭原则）
4. Python 的 MRO 机制确保方法解析顺序明确可预测

---

## 第3章：注册表/工厂模式 (Registry/Factory)

### 背景与动机

SGLang 需要支持：
- 100+ 种模型架构（Llama、Qwen、DeepSeek、Gemma...）
- 15+ 种注意力后端（FlashInfer、Triton、FlashMLA...）
- 多种缓存策略、语法后端、投机算法...

如果用 if-elif 链来分发，每新增一种实现都需修改调度代码。**注册表模式**让新增实现完全不触碰已有代码。

### 设计哲学

> **开放-封闭原则**：对扩展开放，对修改封闭。新增实现只需"注册"，无需修改分发逻辑。

### 核心代码

#### 模型注册表 (Model Registry)

```python
# python/sglang/srt/models/registry.py
@dataclass
class _ModelRegistry:
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def register(self, package_name: str, overwrite: bool = False):
        """扫描 package 下所有模块，收集 EntryClass 属性"""
        new_models = import_model_classes(package_name)
        for arch, cls in new_models.items():
            self.models[arch] = cls

    def resolve_model_cls(self, architectures: List[str]) -> Tuple[Type, str]:
        """根据 HuggingFace config.architectures 查找对应的模型类"""
        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)
        return self._raise_for_unsupported(architectures)

# 自动发现机制：扫描模块，找 EntryClass 属性
def import_model_classes(package_name: str):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            module = importlib.import_module(name)
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                # 支持单个类或列表
                if isinstance(entry, list):
                    for cls in entry:
                        model_arch_name_to_cls[cls.__name__] = cls
                else:
                    model_arch_name_to_cls[entry.__name__] = entry
    return model_arch_name_to_cls

# 全局单例 + 自动注册
ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")  # 扫描内置模型

# 支持外部模型包
if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```

#### 投机算法注册表 (Speculative Algorithm Registry)

```python
# python/sglang/srt/speculative/spec_info.py
class SpeculativeAlgorithm(Enum):
    EAGLE = auto()
    EAGLE3 = auto()
    NGRAM = auto()
    NONE = auto()

    @classmethod
    def register(cls, name: str, *, supports_overlap: bool = False):
        """装饰器 — 注册自定义投机算法插件

        Example:
            @SpeculativeAlgorithm.register("MY_SPEC")
            def _factory(server_args):
                return MySpecWorker
        """
        return _register_algorithm(name, supports_overlap=supports_overlap, ...)

    @classmethod
    def from_string(cls, name: Optional[str]):
        """先查内置 Enum，再查插件注册表"""
        try:
            return cls[name.upper()]
        except KeyError:
            spec = _get_registered_spec(name.upper())
            if spec is not None:
                return spec
            raise ValueError(f"Unknown: {name}")
```

### SGLang 中的注册表汇总

| 注册表 | 文件 | 注册方式 |
|--------|------|----------|
| Model Registry | `models/registry.py` | `EntryClass` 属性自动发现 |
| Attention Backend | `attention/attention_registry.py` | 显式注册函数 |
| Grammar Backend | `constrained/base_grammar_backend.py:216` | 工厂方法 |
| Radix Cache | `mem_cache/registry.py` | 工厂注册 |
| Speculative Algorithm | `speculative/spec_info.py` | `@register()` 装饰器 |
| Storage Backend | `mem_cache/storage/backend_factory.py` | 字符串→类映射 |

### 工作流程

```
新增一个模型实现：
1. 创建文件 python/sglang/srt/models/my_model.py
2. 定义 class MyModelForCausalLM(nn.Module): ...
3. 在文件末尾写 EntryClass = MyModelForCausalLM
4. 无需修改 registry.py — pkgutil 自动发现！
```

### 小结

1. **约定优于配置**：`EntryClass` 属性是隐式注册协议，无需显式 `@register`
2. **外部可扩展**：`SGLANG_EXTERNAL_MODEL_PACKAGE` 让第三方模型包零侵入接入
3. **统一查找接口**：`resolve_model_cls(architectures)` 对上层透明
4. 多个注册表并存，各自管理各自领域

---

## 第4章：Backend 抽象模式 (Strategy Pattern)

### 背景与动机

推理框架需要适配不同的：
- 注意力实现（FlashAttention、FlashInfer、Triton、FlashMLA...）
- KV 缓存分配策略（连续分配、分页分配）
- LoRA 执行后端（BGMV、SGMV、Triton）
- 硬件平台（CUDA、ROCm、XPU、NPU）

如果在调用处 if-else 判断，代码会变得极其脆弱。SGLang 用 **抽象基类 + 多实现** 的 Strategy 模式统一解决。

### 设计哲学

> **面向接口编程**：定义抽象接口，调用者只依赖接口；具体实现在运行时根据配置注入。

### 核心代码

#### AttentionBackend — 注意力计算的统一接口

```python
# python/sglang/srt/layers/attention/base_attn_backend.py:18
class AttentionBackend(ABC):
    """所有注意力后端的抽象基类。

    无论底层是 FlashInfer、Triton 还是 FlashMLA，
    上层 RadixAttention 只调用这三个方法。
    """

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """准备前向计算所需的元数据（page table、seq_lens 等）"""
        raise NotImplementedError()

    @abstractmethod
    def forward(self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch):
        """执行注意力计算 — 核心热路径"""
        raise NotImplementedError()

    @abstractmethod
    def init_cuda_graph_state(self, max_bs: int):
        """为 CUDA Graph 捕获预分配 buffer"""
        raise NotImplementedError()

    @abstractmethod
    def init_forward_metadata_capture_cuda_graph(self, ...):
        """CUDA Graph 捕获时的 metadata 初始化"""
        raise NotImplementedError()

    @abstractmethod
    def init_forward_metadata_replay_cuda_graph(self, ...):
        """CUDA Graph 重放时的 metadata 更新"""
        raise NotImplementedError()
```

#### BaseTokenToKVPoolAllocator — KV 缓存分配抽象

```python
# python/sglang/srt/mem_cache/allocator.py:35
class BaseTokenToKVPoolAllocator(abc.ABC):
    """KV 缓存分配器抽象基类。

    两种主要实现：
    - TokenToKVPoolAllocator: token 级连续分配（简单模型）
    - PagedTokenToKVPoolAllocator: page 级分页分配（长序列优化）
    """

    @abc.abstractmethod
    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        self.size = size
        self.page_size = page_size
        self.free_pages = None
        self.release_pages = None

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    @abc.abstractmethod
    def alloc(self, need_size: int):
        """分配 need_size 个 token 的 KV 存储位置"""
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, indices: torch.Tensor):
        """释放 indices 对应的 KV 存储"""
        raise NotImplementedError()
```

#### KVCache — 物理存储抽象

```python
# python/sglang/srt/mem_cache/memory_pool.py:700
class KVCache(abc.ABC):
    """KV 缓存物理存储抽象。

    实现包括：
    - MHATokenToKVPool: Multi-Head Attention (独立 K/V buffer)
    - MLATokenToKVPool: Multi-Latent Attention (latent buffer)
    - DoubleSparseTokenToKVPool: 稀疏注意力
    """

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_size_bytes(self) -> Union[int, Tuple[int, int]]:
        raise NotImplementedError()
```

### 运行时注入示意

```python
# 在 ModelRunner 初始化时，根据配置选择具体后端
if server_args.attention_backend == "flashinfer":
    attn_backend = FlashInferAttnBackend(...)
elif server_args.attention_backend == "triton":
    attn_backend = TritonAttnBackend(...)
elif model_config.is_deepseek_v4:
    attn_backend = DeepseekV4AttnBackend(...)  # 自动选择

# 模型层只依赖抽象接口
class RadixAttention(nn.Module):
    def forward(self, q, k, v):
        return self.attn_backend.forward(q, k, v, self, forward_batch)
```

### 现有 Backend 实现列表

| 抽象接口 | 实现数量 | 典型实现 |
|----------|----------|----------|
| `AttentionBackend` | 15+ | FlashInfer, Triton, FlashAttention, DSV4, Wave, TorchNative, IntelAMX |
| `BaseTokenToKVPoolAllocator` | 4 | Token级, Paged, SWA双路, HiSparse |
| `KVCache` | 5+ | MHA, MLA, DoubleSparse, DeepSeekV4 四池 |
| `BaseLoRABackend` | 3 | BGMV, SGMV, Triton |
| `BasePrefixCache` | 4 | RadixCache, SWARadixCache, UnifiedRadixCache, ChunkCache |

### 小结

1. **抽象基类定义契约**：`@abstractmethod` 确保所有实现必须提供完整接口
2. **调用者零感知**：模型代码无需知道底层是哪个后端
3. **运行时可切换**：通过 `--attention-backend` 参数动态选择
4. **测试友好**：可以 mock 任何后端进行单元测试

---

## 第5章：Radix Tree 前缀缓存

### 背景与动机

LLM 推理中，大量请求共享相同的 system prompt 或对话前缀。如果每次都重新计算这些前缀的 KV 缓存，将浪费大量 GPU 计算和显存。

SGLang 开创性地使用 **Radix Tree（基数树）** 来管理 KV 缓存的前缀共享——这是 SGLang 最核心的创新之一。

### 设计哲学

> **前缀共享，按需计算**：将 token 序列组织为 Radix Tree，相同前缀的请求共享 KV 缓存节点，只计算未命中的部分。

### 核心数据结构

```python
# python/sglang/srt/mem_cache/radix_cache.py

class TreeNode:
    """Radix Tree 的节点"""
    def __init__(self):
        self.children = defaultdict(TreeNode)  # 子节点字典
        self.parent: TreeNode = None           # 父节点引用
        self.key: RadixKey = None              # 该节点存储的 token 序列片段
        self.value: Optional[torch.Tensor] = None  # KV 缓存物理位置索引
        self.lock_ref = 0                      # 引用计数（活跃请求数）
        self.last_access_time = time.monotonic()   # LRU 时间戳
        self.hit_count = 0                     # 命中次数（LFU）
        self.host_value: Optional[torch.Tensor] = None  # HiCache: CPU 备份

    @property
    def evicted(self):
        """节点的 GPU KV 缓存是否已被驱逐"""
        return self.value is None

class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size

        # 驱逐策略：支持 LRU/LFU/FIFO/SLRU
        if self.eviction_policy == "lru":
            self.eviction_strategy = LRUStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy = SLRUStrategy()
        # ...

        self.root_node = TreeNode()  # 树根
```

### 核心操作

```python
class RadixCache(BasePrefixCache):

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """前缀匹配：沿树向下匹配 token 序列，返回最长公共前缀。

        返回：
        - device_indices: 命中的 KV 缓存物理位置
        - last_device_node: 最后一个有效节点
        """
        # 从根节点开始，逐步匹配 token
        node = self.root_node
        matched_len = 0
        while node.children:
            child_key = key[matched_len:].child_key(self.page_size)
            if child_key not in node.children:
                break
            child = node.children[child_key]
            match_len = key[matched_len:].match(child.key, self.page_size)
            if match_len == 0:
                break
            matched_len += match_len
            node = child
        return MatchResult(device_indices=node.value[:matched_len], ...)

    def insert(self, params: InsertParams) -> InsertResult:
        """请求完成后，将新生成的 KV 缓存插入树中。

        步骤：
        1. 沿已有路径匹配
        2. 在分叉点创建新节点
        3. 将物理 KV 索引绑定到新节点
        """

    def evict(self, params: EvictParams) -> EvictResult:
        """内存不足时，按策略驱逐叶子节点。

        Lock-ref 机制保护活跃请求：
        - lock_ref > 0 的节点不可驱逐
        - 只驱逐 lock_ref == 0 的叶子节点
        """
```

### Lock-Ref 生命周期

```
请求到达 → match_prefix → inc_lock_ref(last_node)
  ↓
请求运行中 → KV 缓存被保护，不会被驱逐
  ↓
请求完成 → insert(new_tokens) → dec_lock_ref(last_node)
  ↓
节点进入 LRU 队列 → 可被 evict
```

### SWA Radix Cache 扩展

```python
# python/sglang/srt/mem_cache/swa_radix_cache.py
class SWARadixCache(RadixCache):
    """专为 Sliding Window Attention 设计的双 LRU 缓存。

    核心创新：
    - 双 LRU 链表：full_lru_list (全注意力) + swa_lru_list (滑动窗口)
    - Tombstone 机制：标记滑动窗口外的 SWA KV 可提前释放
    - dec_swa_lock_only(): decode 推进时，提前释放窗口外 SWA KV
    """
```

### 驱逐策略

| 策略 | 文件 | 特点 |
|------|------|------|
| LRU | `evict_policy.py` | 最近最少使用（默认） |
| LFU | `evict_policy.py` | 最不常使用 |
| SLRU | `evict_policy.py` | 分段 LRU（热/冷分离） |
| FIFO | `evict_policy.py` | 先进先出 |
| Priority | `evict_policy.py` | 优先级感知 |

### 小结

1. **Radix Tree** 天然适合 token 序列的前缀共享，O(n) 匹配
2. **Lock-Ref** 机制实现无锁的引用计数，保护活跃请求
3. **多种驱逐策略** 通过 Strategy 模式可插拔
4. **SWA 扩展** 用 tombstone 实现滑动窗口外内存的提前回收
5. 这是 SGLang 与 vLLM 最大的架构差异之一

---

## 第6章：分层内存池 (Hierarchical Memory Pool)

### 背景与动机

LLM 推理中，KV 缓存是最大的内存消耗项（可占 GPU 显存的 80%+）。动态分配内存（如 `torch.malloc`）会导致：
- 内存碎片化
- 分配延迟不确定
- 无法精确控制内存上限

SGLang 采用**预分配 + 三层池化**方案，在启动时一次性分配全部 KV 缓存内存，运行时只做索引管理。

### 设计哲学

> **预分配 + 分层管理**：消除运行时内存分配，用整数索引代替指针，实现零碎片、O(1) 分配。

### 三层架构

```
┌────────────────────────────────────────────────────┐
│  L1: ReqToTokenPool                                │
│  请求 → token 位置映射 (req_pool_idx → [loc...])    │
│  大小: [max_reqs, max_context_len], dtype=int32    │
├────────────────────────────────────────────────────┤
│  L2: TokenToKVPoolAllocator                        │
│  token 位置 → KV page 分配 (free_pages 管理)       │
│  Paged: page_size=256, Triton kernel 快速分配       │
├────────────────────────────────────────────────────┤
│  L3: KVCache                                       │
│  物理 KV 存储 (预分配的大 tensor)                   │
│  如: [num_pages * page_size, num_heads, head_dim]  │
└────────────────────────────────────────────────────┘
```

### 核心代码

#### L1: ReqToTokenPool — 请求到 Token 的映射

```python
# python/sglang/srt/mem_cache/memory_pool.py:138
class ReqToTokenPool:
    """将请求映射到其 token 在 KV 缓存中的物理位置。

    本质是一个 2D 整数矩阵：req_to_token[req_idx, token_pos] = kv_loc
    """

    def __init__(self, size: int, max_context_len: int, device: str):
        # +1 padding row: CUDA graph 中 padding 的请求索引为 0
        self._alloc_size = size + 1
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(1, self._alloc_size))

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        """为请求分配 pool 槽位"""
        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None  # 内存不足，触发 preemption
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, req: Req):
        """请求完成后归还槽位"""
        self.free_slots.append(req.req_pool_idx)
```

#### L2: PagedTokenToKVPoolAllocator — 分页分配

```python
# python/sglang/srt/mem_cache/allocator.py:360+
class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """分页 KV 缓存分配器。

    关键设计：
    - page_size=256: 每页存储 256 个 token 的 KV
    - Triton kernel 实现高效分配（避免 Python 循环）
    - 两个空闲列表: free_pages (已排序) + release_pages (待合并)
    """

    def alloc_extend(self, prefix_lens, extend_lens, ...):
        """Prefill 分配：为每个请求分配 extend_len 个 token 的 KV 位置。

        Triton kernel (alloc_extend_kernel) 三段式填充：
        1. 旧 page 的剩余空间 (partial tail)
        2. 新的完整 page
        3. 新 page 的部分空间 (partial head)
        """

    def alloc_decode(self, num_reqs, seq_lens, ...):
        """Decode 分配：每个请求只需 1 个新 token。

        优化：只有当 seq_len % page_size == 1 时才需新 page。
        大部分 decode step 只是在已有 page 内追加。
        """
```

#### L3: KVCache — 物理存储

```python
# python/sglang/srt/mem_cache/memory_pool.py:700
class KVCache(abc.ABC):
    """物理 KV 缓存 — 启动时预分配的大 tensor。

    典型实现 MHATokenToKVPool:
      k_buffer: List[torch.Tensor]  # [layer_num] each [num_tokens, head_num, head_dim]
      v_buffer: List[torch.Tensor]  # [layer_num] each [num_tokens, head_num, head_dim]
    """

    def __init__(self, size, page_size, dtype, layer_num, device):
        self.size = size          # 总 token 数
        self.page_size = page_size
        self.layer_num = layer_num

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor: ...

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor: ...
```

### 多级存储扩展 (HiCache)

```
GPU KV Cache (热数据, 快速访问)
    ↕ PCIe / NVLink 异步传输
CPU Memory (温数据, 大容量)
    ↕ 磁盘 I/O
Disk Storage (冷数据, 持久化)
```

关键文件：
- `python/sglang/srt/mem_cache/memory_pool_host.py` — CPU 侧 KV 池
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` — 多级组装
- `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py` — SWA 层 HiCache

### 小结

1. **三层解耦**：请求管理、页分配、物理存储各自独立演进
2. **预分配 + 索引**：零运行时内存分配，确定性延迟
3. **Triton kernel** 实现批量分页分配，比 Python 循环快 100x
4. **HiCache 扩展** 突破 GPU 显存限制，利用 CPU/Disk 扩展容量
5. page_size=256 是吞吐和碎片之间的平衡点

---

## 第7章：Overlap 流水线执行

### 背景与动机

LLM 推理的瓶颈不仅在 GPU 计算本身，还在于 CPU 调度、内存拷贝、采样等辅助操作的延迟。如果这些操作串行执行：

```
[CPU: schedule] → [GPU: forward] → [CPU: process_result] → [CPU: schedule] → ...
                   ↑ GPU 空闲 ↑                             ↑ GPU 空闲 ↑
```

GPU 利用率可能不到 60%。**Overlap** 让 CPU 处理与 GPU 计算并行，消除 GPU 空闲时间。

### 设计哲学

> **隐藏延迟**：让 CPU 调度 batch N+1 和 GPU 执行 batch N 同时进行，最大化硬件利用率。

### 核心代码：CPU-GPU Overlap

```python
# python/sglang/srt/managers/scheduler.py:1538
def event_loop_overlap(self):
    """Overlap 调度循环 — CPU 和 GPU 流水线并行。

    核心思想：
    - GPU 执行 batch[N] 的 forward 时
    - CPU 同时处理 batch[N-1] 的结果 + 调度 batch[N+1]
    """
    self.result_queue: Deque[Tuple[ScheduleBatch, BatchResult]] = deque()

    while True:
        # 1. 接收新请求
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        # 2. 调度下一个 batch (CPU 工作)
        batch = self.get_next_batch_to_run()

        # 3. 判断是否需要禁用 overlap（连续 prefill 时）
        disable_overlap = self.is_disable_overlap_for_batch(batch)
        if disable_overlap:
            # 先处理上一个 batch 的结果
            self.result_queue.popleft() → process_batch_result()

        # 4. 启动当前 batch (GPU 开始工作)
        if batch:
            batch_result = self.run_batch(batch)  # 非阻塞，GPU 异步执行
            self.result_queue.append((batch.copy(), batch_result))

        # 5. 处理上一个 batch 的结果 (CPU 工作，与 GPU 并行)
        if self.last_batch and not disable_overlap:
            self.result_queue.popleft() → process_batch_result()

        # 6. 采样 (依赖上一个 batch 的语法状态)
        self.launch_batch_sample_if_needed(batch_result)

        self.last_batch = batch
```

### 时序图

```
Time →
CPU:  [schedule B1] [process B0 result + schedule B2] [process B1 result + schedule B3]
GPU:  [forward B0 ] [       forward B1              ] [       forward B2              ]
       ↑ 并行 ↑      ↑          并行                ↑
```

### Multi-Stream Overlap (DeepSeek V4)

DeepSeek V4 在单次 forward 内也实现了多流并行：

```python
# python/sglang/srt/models/deepseek_v4.py:476
def _forward_prepare_multi_stream(self, ...):
    """三路 CUDA Stream 并行：

    Stream 1 (Main):     KV 计算 (latent projection)
    Stream 2 (Compress): 压缩器 (compress + norm + rope + store)
    Stream 3 (Indexer):  索引器 (更新 C4 索引表)

    三者操作不同的内存区域，可以完全并行。
    最后在主 stream 上同步。
    """
    with torch.cuda.stream(self.compress_stream):
        self.attn_backend.forward_core_compressor(...)

    with torch.cuda.stream(self.indexer_stream):
        self.attn_backend.forward_core_indexer(...)

    # 主 stream 等待压缩和索引完成
    torch.cuda.current_stream().wait_stream(self.compress_stream)
    torch.cuda.current_stream().wait_stream(self.indexer_stream)
```

### FutureMap — 跨 Iteration 数据传递

```python
# python/sglang/srt/managers/overlap_utils.py
class FutureMap:
    """在 overlap 模式下，batch[N] 的采样结果需要传递给 batch[N+1]。

    FutureMap 是一个 relay buffer：
    - batch[N] 的 forward 完成后，将 hidden states 写入 FutureMap
    - batch[N+1] 的调度读取 FutureMap 获取上一步的 token
    """
```

### Pipeline Parallelism Overlap

```python
# python/sglang/srt/managers/scheduler_pp_mixin.py:1146
# PP 阶段之间的 overlap：
#   Stage[i] 的 send 和 Stage[i+1] 的 recv 可以与计算重叠
#   采用异步 send + 同步 recv 的方式
```

### Overlap 控制策略

| 场景 | Overlap? | 原因 |
|------|----------|------|
| Decode + Decode | Yes | 标准 overlap |
| Prefill + Decode | Yes | 不同 batch 类型可并行 |
| Prefill + Prefill | No (可配置) | 优先降低第一个 prefill 的 TTFT |
| 单 batch | No | 无上一个 batch 可重叠 |

通过 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 环境变量控制。

### 小结

1. **CPU-GPU Overlap** 将调度延迟完全隐藏在 GPU 计算背后
2. **Multi-Stream** 让单次 forward 内的独立操作并行
3. **FutureMap** 解决跨 iteration 的数据依赖
4. Overlap 是 SGLang 高吞吐的关键因素之一（提升 20-30% throughput）

---

## 第8章：环境变量描述符模式

### 背景与动机

SGLang 有 700+ 个可配置的环境变量，用于控制各种运行时行为。传统做法是到处散落 `os.getenv("FOO", "default")`，这带来：
- 类型不安全（环境变量都是字符串）
- 无法集中管理和文档化
- 容易误用（如直接 `if envs.FLAG:` 而非 `.get()`）

SGLang 设计了一套 **描述符（Descriptor）模式** 来统一管理环境变量。

### 设计哲学

> **类型安全 + 集中管理 + 防误用**：用 Python 描述符协议将环境变量封装为类型化对象，集中定义在一个文件中。

### 核心代码

#### 描述符基类

```python
# python/sglang/srt/environ.py:38
class EnvField:
    """环境变量描述符基类。

    利用 Python 的 __set_name__ 协议，自动将属性名作为环境变量名。
    """
    def __init__(self, default: Any):
        self.default = default

    def __set_name__(self, owner, name):
        """Python 描述符协议：类属性赋值时自动调用，获取变量名"""
        self.name = name  # e.g. "SGLANG_ENABLE_OVERLAP"

    def get(self) -> Any:
        """读取环境变量，自动类型转换"""
        value = os.getenv(self.name)
        if value is None:
            return self.default
        return self.parse(value)

    def set(self, value: Any):
        """设置环境变量"""
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        """测试用：临时覆盖环境变量值"""
        backup = os.environ.get(self.name)
        self.set(value)
        yield
        if backup is not None:
            os.environ[self.name] = backup
        else:
            os.environ.pop(self.name, None)

    def __bool__(self):
        """防误用：禁止直接 if envs.FLAG，强制使用 .get()"""
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )
```

#### 类型化子类

```python
# python/sglang/srt/environ.py:109-140
class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        if value.lower() in ["true", "1", "yes"]:
            return True
        if value.lower() in ["false", "0", "no"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean')

class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        return int(value)

class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        return float(value)

class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value

class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())
```

#### 集中定义命名空间

```python
# python/sglang/srt/environ.py:194
class Envs:
    """所有 SGLang 环境变量的集中定义点。

    属性名即环境变量名，值的类型由描述符子类决定。
    支持 IDE 自动补全和类型检查。
    """
    # 调度相关
    SGLANG_DISABLE_OVERLAP_SCHEDULE = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(64)

    # 内存管理
    SGLANG_PAGE_SIZE = EnvInt(256)
    SGLANG_MEM_FRACTION_STATIC = EnvFloat(0.88)

    # 投机解码
    SGLANG_OPT_USE_COMPRESSOR_V2 = EnvBool(True)

    # 模型加载
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr(None)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(())

    # 调试
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvBool(False)
    # ... 700+ 更多变量

envs = Envs()  # 全局单例
```

### 使用方式

```python
# 正确用法
if envs.SGLANG_DISABLE_OVERLAP_SCHEDULE.get():
    self.event_loop_normal()
else:
    self.event_loop_overlap()

# 测试中临时覆盖
with envs.SGLANG_PAGE_SIZE.override(128):
    run_test()

# 错误用法 — 会抛 RuntimeError！
if envs.SGLANG_DISABLE_OVERLAP_SCHEDULE:  # RuntimeError!
    ...
```

### 与其他模式的关联

- **Plugin 系统** 使用 `SGLANG_PLUGINS` 环境变量控制加载哪些插件
- **Registry 模式** 使用 `SGLANG_EXTERNAL_MODEL_PACKAGE` 注册外部模型
- **Backend 选择** 通过环境变量覆盖默认后端

### 小结

1. **描述符协议** 让属性名自动成为环境变量名（零冗余）
2. **类型安全**：`EnvBool/EnvInt/EnvFloat` 在读取时自动转换
3. **防误用**：`__bool__` 重写阻止常见错误
4. **测试友好**：`override()` context manager 实现无副作用的临时覆盖
5. 集中定义让环境变量可搜索、可文档化

---

## 第9章：JIT Kernel 编译模式

### 背景与动机

高性能推理需要大量自定义 CUDA kernel。但如果在安装时编译所有 kernel：
- 安装时间长（10 分钟+）
- 很多 kernel 永远不会被用到（取决于模型和硬件）
- 不同 GPU 架构需要不同的编译参数

SGLang 采用 **按需编译 + 缓存** 的 JIT 策略：kernel 在首次使用时编译，之后从缓存加载。

### 设计哲学

> **按需编译，一次缓存**：避免启动时编译全部 kernel，同时保证热路径零编译延迟。

### 核心代码

#### `@cache_once` 装饰器 — JIT 编译的核心

```python
# python/sglang/jit_kernel/utils.py:46
def cache_once(fn: F) -> F:
    """简单的缓存装饰器（替代 lru_cache，兼容 torch.compile）。

    为什么不用 functools.lru_cache？
    - lru_cache 与 torch.compile 不兼容
    - 我们只需要"编译一次，缓存永久"的语义
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)  # 首次调用时编译
        return result_map[key]  # 后续直接返回缓存

    return wrapper
```

#### 标准 JIT Kernel 模式

```python
# python/sglang/jit_kernel/kvcache.py — 典型的 JIT kernel 模块

# 第一层：@cache_once 工厂 — 按参数编译并缓存 TVM 模块
@cache_once
def _jit_kvcache_module(row_bytes: int) -> Module:
    """按 row_bytes 参数 JIT 编译 KV-Cache 存储 kernel。

    - 不同的 head_dim 产生不同的 row_bytes
    - 每种配置只编译一次
    """
    args = make_cpp_args(row_bytes, is_arch_support_pdl())
    return load_jit(
        "kvcache",
        *args,
        cuda_files=["elementwise/kvcache.cuh"],
        cuda_wrappers=[("store_cache", f"StoreKVCacheKernel<{args}>::run")],
    )

# 第二层：能力探测 — 在运行时检查是否可用
@cache_once
def can_use_store_cache(size: int) -> bool:
    """检查是否支持该 size 的 JIT kernel"""
    if size % 4 != 0:
        return False
    try:
        _jit_kvcache_module(size)
        return True
    except Exception:
        return False

# 第三层：用户调用接口 — 注册为 custom op
@register_custom_op(mutates_args=["k_cache", "v_cache"])
def store_cache(k, v, k_cache, v_cache, indices, *, row_bytes=0):
    """外部调用入口。

    首次调用时触发编译，后续直接执行已编译的 kernel。
    """
    module = _jit_kvcache_module(row_bytes)
    module["store_cache"](k, v, k_cache, v_cache, indices)
```

### JIT 与 AOT 对比

SGLang 同时使用两种 kernel 编译策略：

| 特性 | JIT (jit_kernel/) | AOT (sgl-kernel/) |
|------|------|------|
| 编译时机 | 首次运行时 | pip install 时 |
| 语言 | CUDA + TVM FFI | C++/CUDA |
| 适用场景 | 参数依赖运行时信息 | 性能关键的通用 kernel |
| 缓存位置 | 进程内 `result_map` | 编译后的 .so 文件 |
| 代表 kernel | kvcache, rope, moe_align | flashinfer, mla_decode |

### 目录结构

```
python/sglang/jit_kernel/
├── utils.py              # cache_once, load_jit, make_cpp_args
├── kvcache.py            # KV 缓存写入 kernel
├── rope.py               # RoPE 位置编码 kernel
├── moe_align.py          # MoE token 对齐 kernel
├── flash_attention.py    # FlashAttention Triton 实现
├── norm.py               # LayerNorm/RMSNorm kernel
├── activation.py         # SiLU/GELU activation kernel
└── ...                   # 40+ JIT kernel 模块
```

### 小结

1. **`@cache_once`** 是所有 JIT kernel 的核心装饰器，替代不兼容 torch.compile 的 lru_cache
2. **三层模式**：工厂函数（编译）→ 能力探测 → 用户接口
3. **按参数特化**：不同 head_dim 编译不同的 kernel（如 row_bytes=128 vs 256）
4. **双轨制**：JIT 用于灵活的参数化 kernel，AOT 用于性能关键的固定 kernel

---

## 第10章：TypeBasedDispatcher 类型分发

### 背景与动机

Scheduler 需要处理 30+ 种不同类型的请求（GenerateReq、EmbeddingReq、AbortReq、UpdateWeightsReq...）。传统做法是一个巨大的 if-elif 链：

```python
# 反面示例
if isinstance(req, GenerateReqInput):
    self.handle_generate(req)
elif isinstance(req, EmbeddingReqInput):
    self.handle_embedding(req)
elif isinstance(req, AbortReq):
    self.handle_abort(req)
# ... 30+ elif
```

这违反开放-封闭原则且性能不佳。SGLang 用 **TypeBasedDispatcher** 实现 O(1) 类型分发。

### 设计哲学

> **用字典替代 if-elif**：类型→处理函数的映射，支持精确匹配 + MRO 继承链回退 + 结果缓存。

### 核心代码

```python
# python/sglang/utils.py:624
class TypeBasedDispatcher:
    """基于类型的快速分发器。

    三级查找策略（从快到慢）：
    1. 精确类型匹配 → O(1) dict 查找
    2. MRO 缓存命中 → O(1) dict 查找
    3. 注册顺序遍历 + isinstance 检查 → O(n)，结果缓存
    """

    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        # 有序字典：保持注册顺序（决定优先级）
        self._mapping = OrderedDict(mapping)
        # MRO 缓存：子类类型 → 处理函数
        self._mro_cache = {}
        self._fallback_fn = None

    def __call__(self, obj: Any):
        obj_type = type(obj)

        # 1. 精确匹配 — 最快路径 O(1)
        fn = self._mapping.get(obj_type)
        if fn is not None:
            return fn(obj)

        # 2. MRO 缓存 — 子类首次匹配后缓存
        cached_fn = self._mro_cache.get(obj_type)
        if cached_fn is not None:
            return cached_fn(obj)

        # 3. 遍历注册表，按 MRO 继承链匹配
        for ty, fn in self._mapping.items():
            if isinstance(obj, ty):
                self._mro_cache[obj_type] = fn  # 缓存结果
                return fn(obj)

        # 4. 无匹配 — fallback 或抛异常
        if self._fallback_fn is not None:
            return self._fallback_fn(obj)
        raise ValueError(f"Invalid object: {obj}")

    def __iadd__(self, other: "TypeBasedDispatcher"):
        """合并两个 dispatcher（Mixin 各自注册后合并）"""
        for ty, fn in other._mapping.items():
            if ty not in self._mapping:
                self._mapping[ty] = fn
        self._mro_cache.clear()
        return self
```

### 在 Scheduler 中的使用

```python
# python/sglang/srt/managers/scheduler.py
class Scheduler:
    def init_request_dispatcher(self):
        """初始化请求分发器 — 30+ 种请求类型映射"""
        self.request_dispatcher = TypeBasedDispatcher([
            (GenerateReqInput,      self.handle_generate_request),
            (EmbeddingReqInput,     self.handle_embedding_request),
            (AbortReq,              self.handle_abort_request),
            (UpdateWeightsReqInput, self.handle_update_weights),
            (OpenSessionReqInput,   self.handle_open_session),
            (CloseSessionReqInput,  self.handle_close_session),
            # ... 更多类型
        ])
        # 各 Mixin 追加自己的处理函数
        self.request_dispatcher += self.disagg_decode_dispatcher
        self.request_dispatcher += self.pp_dispatcher

    def process_input_requests(self, recv_reqs):
        """处理所有到达的请求 — 直接分发，无 if-elif"""
        for req in recv_reqs:
            self.request_dispatcher(req)  # O(1) 分发
```

### 设计优势

| 特性 | 说明 |
|------|------|
| O(1) 热路径 | 95%+ 请求走精确匹配路径 |
| MRO 感知 | 子类自动匹配父类的 handler |
| 可组合 | `+=` 运算符让 Mixin 各自注册后合并 |
| 缓存友好 | MRO 查找结果缓存，避免重复遍历 |
| 有序优先级 | 注册顺序决定匹配优先级 |

### 小结

1. **替代 30+ if-elif**，代码清晰且性能优异
2. **三级查找**确保首次调用后都是 O(1)
3. **与 Mixin 模式协作**：每个 Mixin 注册自己的类型处理函数，最后合并
4. 适用于任何"类型→行为"的映射场景

---

## 第11章：Plugin/Hook 系统

### 背景与动机

SGLang 需要支持多种硬件平台（NVIDIA、AMD、Intel、华为 NPU、Apple MLX...），每个平台有不同的实现。如果用 if-else 判断硬件类型，会让核心代码充满条件分支。

SGLang 的解决方案：**Plugin 系统 + Hook 注入**，让平台特定代码完全外置，通过 setuptools entry_points 自动发现和加载。

### 设计哲学

> **允许外部扩展不修改核心代码**：通过 entry_points 自动发现插件，通过 Hook 注入修改/增强核心行为。

### 核心代码

#### Plugin 自动发现

```python
# python/sglang/srt/plugins/__init__.py
PLATFORM_PLUGINS_GROUP = "sglang.srt.platforms"   # 硬件平台插件
GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"      # 通用插件

def load_plugins_by_group(group: str, excluded_dists=None):
    """通过 setuptools entry_points 自动发现并加载插件。

    外部包只需在 pyproject.toml 中注册：
    [project.entry-points."sglang.srt.platforms"]
    xpu = "sglang_xpu:register"

    SGLang 启动时自动扫描并加载。
    """
    # SGLANG_PLUGINS 白名单（逗号分隔）
    allowed_str = envs.SGLANG_PLUGINS.get()
    if allowed_str:
        allowed_set = {x.strip() for x in allowed_str.split(",")}

    # 通过 importlib.metadata 发现所有注册的 entry_points
    discovered = entry_points(group=group)

    plugins = {}
    for ep in discovered:
        if allowed_set and ep.name not in allowed_set:
            continue  # 不在白名单中，跳过
        func = ep.load()       # 动态导入插件
        plugins[ep.name] = (func, ep.dist.name)

    return plugins
```

#### Hook 注册表

```python
# python/sglang/srt/plugins/hook_registry.py

class HookType(Enum):
    """四种 Hook 类型"""
    BEFORE = "before"    # 在原函数之前执行，可修改参数
    AFTER = "after"      # 在原函数之后执行，可修改返回值
    AROUND = "around"    # 包裹原函数，完全控制执行
    REPLACE = "replace"  # 完全替换原函数

class HookRegistry:
    """全局 Hook 注册表。

    线程安全：注册在 load_plugins() 阶段（单线程）。
    apply_hooks() 在引擎启动前调用一次。
    """
    _hooks: dict[str, list[tuple[HookType, Callable, HookSource]]] = defaultdict(list)

    @classmethod
    def register(cls, target: str, hook: Callable, hook_type: HookType):
        """注册 Hook 到目标函数/方法/类。

        Args:
            target: 完全限定路径
                e.g. "sglang.srt.managers.scheduler.Scheduler.schedule"
            hook: Hook 函数
            hook_type: BEFORE/AFTER/AROUND/REPLACE

        Example:
            def my_timer(original_fn, *args, **kwargs):
                start = time.perf_counter()
                result = original_fn(*args, **kwargs)
                print(f"Elapsed: {time.perf_counter() - start:.3f}s")
                return result

            HookRegistry.register(
                "sglang.srt.managers.scheduler.Scheduler.schedule",
                my_timer,
                HookType.AROUND,
            )
        """
        source = _current_plugin_source.get()
        cls._hooks[target].append((hook_type, hook, source))

    @classmethod
    def apply_hooks(cls):
        """引擎启动前，将所有注册的 Hook 应用到目标函数上。

        实现方式：monkey-patching — 替换目标函数为包装后的版本。
        """
        for target, hooks in cls._hooks.items():
            # 解析 target 路径，找到原始函数
            module_path, attr_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            original = getattr(module, attr_name)
            # 按类型包装
            wrapped = _wrap_with_hooks(original, hooks)
            setattr(module, attr_name, wrapped)
```

### 插件开发示例

```toml
# 外部包 pyproject.toml
[project.entry-points."sglang.srt.platforms"]
my_platform = "my_package.plugin:register"

[project.entry-points."sglang.srt.plugins"]
my_hooks = "my_package.hooks:setup"
```

```python
# my_package/hooks.py
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

def setup():
    """插件入口函数 — SGLang 启动时自动调用"""
    HookRegistry.register(
        "sglang.srt.layers.attention.base_attn_backend.AttentionBackend",
        MyCustomBackend,
        HookType.REPLACE,
    )
```

### 环境变量控制

| 变量 | 说明 |
|------|------|
| `SGLANG_PLATFORM` | 选择使用哪个硬件平台插件 |
| `SGLANG_PLUGINS` | 逗号分隔的允许加载的插件白名单 |

### 小结

1. **零侵入扩展**：外部包通过 entry_points 注册，核心代码无需修改
2. **四种 Hook 类型** 覆盖所有注入场景（观察、修改、替换）
3. **自动发现 + 白名单** 确保安全和灵活性并存
4. 硬件平台支持完全外置为独立 pip 包

---

## 第12章：连续批处理与调度策略

### 背景与动机

传统静态批处理（Static Batching）中，batch 内所有请求必须等最长序列完成才能释放。这导致短序列 GPU 空闲、吞吐量低。

**连续批处理（Continuous Batching）** 允许请求在任意 step 加入或离开 batch，最大化 GPU 利用率。SGLang 在此基础上进一步优化：**Chunked Prefill + Mixed Batch**。

### 设计哲学

> **动态组装 batch，持续填满 GPU**：每个 step 都重新决定哪些请求参与计算，让 GPU 永远不空闲。

### 核心代码

#### 调度主循环

```python
# python/sglang/srt/managers/scheduler.py
class Scheduler:
    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """每个 step 的核心决策：组装下一个 batch。

        决策逻辑：
        1. 检查 running batch 中是否有请求完成 → 释放 KV 内存
        2. 检查 waiting queue 中是否有新请求可加入
        3. 根据可用内存，决定 prefill 多少 token
        4. 组装 Mixed Batch: decode 请求 + prefill 请求
        """
        # 处理完成的请求
        if self.running_batch:
            self.running_batch.filter_finished()

        # 如果有等待的请求且内存允许，添加 prefill
        if self.waiting_queue and self.has_available_memory():
            new_batch = self.get_new_prefill_batch()
            if new_batch:
                # Mixed batch: running (decode) + new (prefill)
                return self.merge_batches(self.running_batch, new_batch)

        return self.running_batch  # 纯 decode batch
```

#### Chunked Prefill — 长序列切块

```python
# python/sglang/srt/managers/scheduler.py
def init_chunked_prefill(self, req: Req):
    """将长 prefill 请求切成多个 chunk。

    为什么需要 Chunked Prefill？
    - 一个 8K token 的 prefill 会独占 GPU 数百毫秒
    - 切成 512 token 的 chunk，每个 chunk 只占几毫秒
    - 中间穿插 decode step，降低所有请求的延迟
    """
    chunk_size = self.chunked_prefill_size  # 默认 512
    remaining = req.extend_input_len

    while remaining > chunk_size:
        # 只处理前 chunk_size 个 token
        req.extend_input_len = chunk_size
        remaining -= chunk_size
        req.inflight_middle_chunks += 1
```

#### PrefillAdder — 动态决定 prefill 大小

```python
# python/sglang/srt/managers/schedule_policy.py
class PrefillAdder:
    """动态计算本 step 可以 prefill 多少 token。

    约束条件：
    1. 可用 KV 缓存内存
    2. chunked_prefill_size 上限
    3. running batch 的 decode 需求
    4. 是否开启 mixed batch

    输出：选中的请求列表 + 每个请求的 extend_len
    """
    def add_request(self, req: Req) -> bool:
        """尝试将请求加入 prefill batch"""
        needed_tokens = req.extend_input_len + req.estimated_new_tokens
        if self.remaining_budget >= needed_tokens:
            self.remaining_budget -= needed_tokens
            self.selected.append(req)
            return True
        return False  # 内存不足，放弃
```

#### 调度策略

```python
# python/sglang/srt/managers/schedule_policy.py:125
class CacheAwarePolicy(Enum):
    LPM = "lpm"           # Longest Prefix Match — 优先选前缀命中最长的
    DFS_WEIGHT = "dfs-weight"  # DFS 权重 — 按树深度加权

class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"         # First Come First Serve — 先到先服务
    LOF = "lof"           # Longest Output First — 优先选输出最长的
    RANDOM = "random"     # 随机选择
    ROUTING_KEY = "routing-key"  # 按 routing key 频率优先

class SchedulePolicy:
    def calc_priority(self, waiting_queue, running_batch):
        """根据策略计算等待队列中各请求的优先级"""
        if self.policy == CacheAwarePolicy.LPM:
            # 前缀匹配越长，优先级越高（复用 KV 缓存）
            self._sort_by_longest_prefix(waiting_queue)
        elif self.policy == CacheAgnosticPolicy.FCFS:
            pass  # 保持到达顺序
```

### Preemption — 内存不足时的抢占

```
内存不足 → 暂停低优先级 decode 请求 → 释放其 KV → prefill 高优先级请求
         → 高优先级完成后 → 重新 prefill 被抢占的请求
```

### Batch 类型

| 类型 | 组成 | 场景 |
|------|------|------|
| Decode Batch | 全部 decode 请求 | 正常生成阶段 |
| Extend Batch | 全部 prefill 请求 | 新请求到达时 |
| Mixed Batch | decode + prefill | 默认模式（最高效） |

### 小结

1. **Continuous Batching** 让 GPU 利用率从 60% 提升到 90%+
2. **Chunked Prefill** 将长 prefill 切块，避免阻塞 decode 请求
3. **Mixed Batch** 在同一 forward 中同时处理 decode 和 prefill
4. **LPM 策略** 最大化 Radix Cache 命中率
5. **Preemption** 确保高优先级请求不被阻塞

---

## 第13章：PD 分离 (Prefill-Decode Disaggregation)

### 背景与动机

Prefill 和 Decode 的计算特性截然不同：

| 特性 | Prefill | Decode |
|------|---------|--------|
| 计算强度 | 高（大量 GEMM） | 低（单 token） |
| 内存带宽需求 | 中 | 高（KV 缓存读取） |
| 延迟敏感度 | TTFT | TPOT |
| GPU 利用率 | 高（compute-bound） | 低（memory-bound） |

如果混合部署，prefill 会抢占 decode 的内存带宽，decode 会浪费 prefill 的计算能力。**PD 分离**让两者运行在不同的 GPU 组上。

### 设计哲学

> **计算异构，分而治之**：Prefill 节点负责高计算密度的首次填充，Decode 节点负责高吞吐的逐 token 生成，通过 KV 传输连接。

### 架构图

```
Client Request
      │
      ▼
┌─────────────┐         KV Transfer         ┌──────────────┐
│  Prefill    │  ─────────────────────────▶  │   Decode     │
│  Node(s)    │  NIXL / Mooncake / RDMA      │   Node(s)    │
│             │                              │              │
│  ● Compute  │                              │  ● Iterate   │
│    KV cache │                              │    generate  │
│  ● Return   │                              │  ● Stream    │
│    to decode│                              │    output    │
└─────────────┘                              └──────────────┘
```

### 核心代码

#### Prefill 侧：PrefillBootstrapQueue

```python
# python/sglang/srt/disaggregation/prefill.py
class PrefillBootstrapQueue:
    """Prefill 节点的预填充队列。

    工作流程：
    1. 接收请求
    2. 执行 prefill forward
    3. 将 KV 缓存通过高速网络传输到 Decode 节点
    4. 通知 Decode 节点开始 decode
    """
    def process_prefill_complete(self, req):
        # Prefill 完成后，启动 KV 传输
        self.kv_sender.send_kv_cache(
            req.kv_indices,
            target_decode_node=req.assigned_decode_node
        )
```

#### Decode 侧：DecodePreallocQueue + TransferQueue

```python
# python/sglang/srt/disaggregation/decode.py
class DecodePreallocQueue:
    """Decode 节点的预分配队列。

    在 KV 到达之前，预先分配好 KV 缓存空间。
    """
    def preallocate(self, req):
        # 提前分配 KV 空间，等待传输完成
        indices = self.token_to_kv_pool_allocator.alloc(req.kv_len)
        req.preallocated_indices = indices

class TransferQueue:
    """管理 KV 传输的完成状态。

    使用 poll 模式检查传输是否完成。
    """
    def check_transfer_complete(self) -> List[Req]:
        # 检查哪些请求的 KV 已传输完成
        completed = []
        for req in self.pending:
            if self.kv_receiver.is_complete(req):
                completed.append(req)
        return completed
```

#### KV 传输协议

```python
# python/sglang/srt/disaggregation/nixl/conn.py
class NixlKVSender:
    """基于 NIXL 的 KV 传输（RDMA/GPU Direct）。

    特点：
    - 零拷贝：GPU 内存直接通过 RDMA 传输
    - 异步：不阻塞 GPU 计算
    - 多 TP rank 并行传输
    """

# python/sglang/srt/disaggregation/mooncake/conn.py
class MooncakeKVSender:
    """基于 Mooncake 的 KV 传输（适用于云环境）"""
```

#### 跨 TP Rank 同步

```python
# python/sglang/srt/disaggregation/
def poll_and_all_reduce(self):
    """跨 TP rank 同步传输状态。

    问题：每个 TP rank 独立接收 KV，但必须所有 rank 都收到后
    才能开始 decode（因为注意力计算需要所有 head 的 KV）。

    解决：all_reduce 同步各 rank 的接收状态。
    """
    local_status = self.check_local_transfer()
    global_status = all_reduce(local_status, op=ReduceOp.MIN)
    # MIN 确保只有所有 rank 都完成时才返回 True
```

### 与 Mixin 模式的协作

```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,   # Decode 侧的 PD 分离逻辑
    SchedulerDisaggregationPrefillMixin,  # Prefill 侧的 PD 分离逻辑
    ...
):
    # Mixin 各自实现 PD 分离的一侧
    # 同一个 Scheduler 类通过配置决定是 Prefill 还是 Decode 角色
```

### 小结

1. **异构计算分离**让 Prefill 和 Decode 各自优化硬件利用率
2. **KV 传输**是瓶颈，NIXL/Mooncake 提供高带宽零拷贝方案
3. **预分配 + 异步传输** 消除 Decode 侧等待
4. **all_reduce 同步** 确保 TP 并行的正确性

---

## 第14章：投机解码 (Speculative Decoding)

### 背景与动机

LLM 的 decode 阶段是 memory-bound 的：每个 step 只生成 1 个 token，但要读取全部 KV 缓存。GPU 的计算能力大量闲置。

**投机解码** 的核心思想：用一个小模型（Draft Model）快速预测多个 token，再用大模型（Target Model）一次性验证，从而将多个 decode step 压缩为 1 个 verify step。

### 设计哲学

> **用廉价的猜测 + 批量验证替代逐 token 生成**：利用 GPU 空闲计算能力，用 1 次 target forward 验证 k 个 draft token。

### 核心流程

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Draft (小模型)                                        │
│   输入: last_token                                           │
│   输出: k 个候选 token (如 k=5)                               │
│   时间: ~1ms (小模型快)                                       │
├──────────────────────────────────────────────────────────────┤
│ Step 2: Verify (大模型)                                       │
│   输入: k 个候选 token (作为 prefill 一次性送入)               │
│   输出: 每个位置的 logits → 判断哪些 draft 正确                │
│   时间: ~10ms (但只需 1 次 forward 而非 k 次)                  │
├──────────────────────────────────────────────────────────────┤
│ Step 3: Accept/Reject                                         │
│   correct_drafts: 连续正确的 draft token (不含 bonus)          │
│   bonus_token: target 模型额外给出的 +1 token                  │
│   accept_tokens = correct_drafts + bonus_token                │
└──────────────────────────────────────────────────────────────┘
```

### 核心代码

#### EAGLEWorker — Draft + Verify 流程

```python
# python/sglang/srt/speculative/eagle_worker.py:93
class EAGLEWorker(TpModelWorker):
    """EAGLE 投机解码工作器。

    EAGLE 特点：
    - Draft 模型共享 Target 的 hidden states（不是独立模型）
    - 支持 Tree Draft：k 个位置每个 top-p 个候选 → 树形验证
    - Adaptive Controller 动态调整 draft 步数
    """

    def __init__(self, server_args, target_worker, ...):
        self.topk = server_args.speculative_eagle_topk        # 每步 top-k
        self.speculative_num_steps = server_args.speculative_num_steps  # draft 步数
        self.target_worker = target_worker

        # 共享 allocator（draft 和 target 共用内存管理）
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Adaptive Controller：根据实时 accept_rate 调整 draft 步数
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(self)

    def forward_draft_and_verify(self, batch: ScheduleBatch):
        """完整的一轮投机解码。

        1. Draft: 小模型生成 k 个候选 token
        2. Build Tree: 将候选组织为验证树
        3. Verify: 大模型一次性验证
        4. Accept: 确定哪些 token 被接受
        """
        # Step 1: Draft
        draft_tokens = self.draft_forward(batch)  # [bs, num_steps * topk]

        # Step 2: Build verification tree
        tree = build_tree_kernel_efficient(draft_tokens, self.topk)

        # Step 3: Target verify (一次 forward 验证整棵树)
        verify_output = self.target_worker.forward_verify(tree)

        # Step 4: Accept/Reject
        accept_tokens, bonus_tokens = self.accept_reject(
            verify_output.logits, draft_tokens
        )
        return accept_tokens  # 最终接受的 token 序列
```

#### AdaptiveController — 动态调整 Draft 步数

```python
# python/sglang/srt/speculative/adaptive_runtime_state.py
class AdaptiveController:
    """根据运行时 accept_rate 动态调整 draft 步数。

    思想：
    - accept_rate 高 → 增加 draft 步数 → 更多 token/step
    - accept_rate 低 → 减少 draft 步数 → 减少浪费
    """
    def update(self, num_correct_drafts: int, num_proposed_drafts: int):
        accept_rate = num_correct_drafts / num_proposed_drafts
        if accept_rate > self.high_threshold:
            self.num_steps = min(self.num_steps + 1, self.max_steps)
        elif accept_rate < self.low_threshold:
            self.num_steps = max(self.num_steps - 1, self.min_steps)
```

### 支持的投机算法

| 算法 | 文件 | 特点 |
|------|------|------|
| EAGLE | `eagle_worker.py` | 共享 hidden states，tree draft |
| EAGLE3 | 同上 | 改进版本，支持 hot token 词表压缩 |
| NGRAM | `ngram_worker.py` | 无模型，基于 n-gram 统计 draft |
| DFLASH | — | Flash Decoding 优化版 |
| Standalone | — | 独立 draft 模型 |
| Plugin | `spec_info.py` | `@register()` 自定义算法 |

### 命名规范（来自 .claude/rules）

```
accept_tokens  → 含 bonus token 的最终接受序列
correct_drafts → 仅 draft 中正确的部分（不含 bonus）
bonus_token    → target 模型额外给出的 +1 token
num_accept_tokens → 计数（含 bonus）
num_correct_drafts → 计数（不含 bonus）
accept_rate → 文献约定 α = correct_drafts / proposed_drafts
```

### 小结

1. **EAGLE** 是 SGLang 的主力投机算法（共享 hidden states，低开销）
2. **Tree Draft** 将多个候选组织为树，一次 verify 覆盖所有分支
3. **Adaptive Controller** 让 draft 步数自适应，避免过多/过少猜测
4. **Plugin 注册表** 让外部投机算法可零侵入接入

---

## 第15章：CUDA Graph 捕获与重放

### 背景与动机

LLM Decode 阶段的每个 step 执行相同的计算图（相同的 kernel 序列），但每次都重新调度 kernel 会带来：
- CPU 端 kernel launch overhead（每个 kernel 约 3-10μs）
- 一个 Transformer 层有 10+ 个 kernel，60 层就是 600+ 次 launch
- Decode batch 计算量小（bs=1-256），launch overhead 占比可达 30%+

**CUDA Graph** 将一次完整的前向过程"录制"下来，后续直接"重放"，消除 launch overhead。

### 设计哲学

> **录制一次，重放无限次**：将固定形状的计算图编译为单个 GPU 操作，消除 CPU 端开销。

### 核心代码

#### CudaGraphRunner — 主控类

```python
# python/sglang/srt/model_executor/cuda_graph_runner.py
class CudaGraphRunner:
    """CUDA Graph 管理器。

    核心工作：
    1. 启动时按多个 batch_size 捕获 Graph
    2. 运行时将实际 bs 向上取整到最近的捕获 bs
    3. 用 padding 填充到固定大小后重放 Graph
    """

    def __init__(self, model_runner):
        # 预定义的 batch_size 序列：1,2,4,8,...,max_bs
        self.capture_bs_list = self._compute_capture_bs_list(max_bs)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}

        # 输入/输出 buffer（固定大小，Graph 通过地址引用）
        self.input_buffers = DecodeInputBuffers.create(max_bs=max_bs, ...)
        self.output_buffers = {}

    def capture(self, batch_size: int):
        """捕获指定 batch_size 的 CUDA Graph。

        步骤：
        1. 创建固定大小的输入 buffer
        2. 进入 graph capture 模式
        3. 执行一次完整的 model forward
        4. Graph 记录了所有 kernel 调用
        """
        graph = torch.cuda.CUDAGraph()

        # Warmup（确保所有 lazy 初始化完成）
        self._run_warmup(batch_size)

        # 正式捕获
        with torch.cuda.graph(graph, pool=self.graph_pool):
            output = self.model_runner.forward(
                self.input_buffers,
                forward_mode=ForwardMode.DECODE,
            )

        self.graphs[batch_size] = graph
        self.output_buffers[batch_size] = output

    def replay(self, forward_batch: ForwardBatch):
        """重放 CUDA Graph。

        步骤：
        1. 找到 >= actual_bs 的最小捕获 bs
        2. 将实际数据 copy 到固定 buffer（padding 补齐）
        3. graph.replay() — 一个 GPU 操作完成整个 forward
        4. 从固定 buffer 中取出有效结果
        """
        # 向上取整到最近的捕获 batch_size
        padded_bs = self._get_padded_bs(forward_batch.batch_size)

        # 数据拷贝到固定 buffer
        self.input_buffers.copy_from(forward_batch, padded_bs)

        # 重放！（整个 forward 是一个 GPU 操作）
        self.graphs[padded_bs].replay()

        # 提取有效输出
        return self.output_buffers[padded_bs][:forward_batch.batch_size]
```

#### Breakable CUDA Graph — 条件执行

```python
# python/sglang/srt/model_executor/breakable_cuda_graph/
class BreakableCUDAGraph:
    """支持条件分支的 CUDA Graph。

    标准 CUDA Graph 不支持 if-else（图是静态的）。
    BreakableCUDAGraph 通过 device-side conditional 实现
    "在 Graph 内部根据 GPU 上的条件跳过某些 kernel"。

    用途：
    - 稀疏 MoE 中，根据路由结果跳过未激活的 expert
    - 投机解码中，根据 accept 结果提前终止
    """
```

#### Piecewise CUDA Graph — 分段捕获

```python
# python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
class PiecewiseCudaGraphRunner:
    """分段 CUDA Graph 捕获。

    为什么需要分段？
    - 某些操作不兼容 Graph（如 NCCL 通信、dynamic shape）
    - 将 forward 拆为多段，Graph-compatible 的段用 Graph
    - 不兼容的段用 eager mode

    典型分段：
    Segment 1 (Graph): Attention layers 1-30
    Gap (Eager): NCCL All-Reduce
    Segment 2 (Graph): Attention layers 31-60
    """
```

### CUDA Graph 的 Batch Size 策略

```python
# 捕获的 bs 列表示例
capture_bs_list = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

# 运行时：actual_bs=13 → padded_bs=16
# 13 个真实请求 + 3 个 padding 请求（读写 row 0 的 dummy 数据）
```

### 与其他模式的交互

| 模式 | 交互方式 |
|------|----------|
| Memory Pool | padding 请求的 req_pool_idx=0（dummy row），不浪费真实 KV |
| Attention Backend | 需实现 `init_cuda_graph_state` / `replay` 方法 |
| Overlap | Graph replay 是异步的，与 CPU 调度并行 |
| 投机解码 | Draft 和 Verify 各有独立的 CUDA Graph |

### 小结

1. **消除 launch overhead**：60 层 Transformer 的 600+ kernel launch 压缩为 1 次
2. **Padding 策略**：实际 bs 向上对齐到预捕获的尺寸
3. **Breakable Graph** 解决条件分支问题
4. **Piecewise Graph** 解决 NCCL 等不兼容操作
5. Decode 阶段 CUDA Graph 可带来 10-30% 的延迟降低

---

## 第16章：IPC 与进程间通信

### 背景与动机

SGLang 是多进程架构（TokenizerManager、Scheduler、TpModelWorker、DetokenizerManager），它们之间需要高效通信。不同场景对通信有不同需求：

| 场景 | 数据量 | 延迟要求 | 典型方案 |
|------|--------|----------|----------|
| 请求分发 | 小（KB） | 中 | ZMQ 消息 |
| GPU tensor 共享 | 大（GB） | 极低 | CUDA IPC |
| 多模态特征 | 大（MB） | 中 | Shared Memory |
| TP 集合通信 | 大（GB） | 极低 | NCCL |

### 设计哲学

> **按场景选择最佳 IPC 机制**：轻量消息用 ZMQ，GPU 数据用 CUDA IPC/NCCL，大块 CPU 数据用 Shared Memory。

### 核心代码

#### ZMQ 消息通信 — 请求/控制流

```python
# python/sglang/srt/managers/scheduler_components/ipc_channels.py
class SchedulerChannel:
    """Scheduler 的 ZMQ 通信通道。

    模式：
    - PUSH/PULL: 单向消息流（请求分发）
    - DEALER/ROUTER: 多对一异步通信
    """
    def __init__(self, port_args):
        self.context = zmq.Context()

        # 接收来自 TokenizerManager 的请求
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.bind(f"tcp://*:{port_args.scheduler_input_port}")

        # 发送结果到 DetokenizerManager
        self.send_socket = self.context.socket(zmq.PUSH)
        self.send_socket.connect(f"tcp://localhost:{port_args.detokenizer_port}")

    def recv_requests(self) -> List[Any]:
        """非阻塞接收所有待处理请求"""
        reqs = []
        while self.recv_socket.poll(timeout=0):
            data = self.recv_socket.recv(zmq.NOBLOCK)
            reqs.append(pickle.loads(data))
        return reqs

    def send_result(self, result):
        """发送处理结果"""
        self.send_socket.send(pickle.dumps(result))
```

#### CUDA IPC — GPU Tensor 跨进程共享

```python
# python/sglang/srt/utils/cuda_ipc_transport_utils.py
class CudaIPCTransport:
    """GPU tensor 零拷贝跨进程共享。

    原理：
    - CUDA IPC Handle 是 GPU 内存地址的跨进程引用
    - 不需要数据拷贝，多个进程直接访问同一块 GPU 内存
    - 适用于 TP 分组内的 tensor 共享
    """

    @staticmethod
    def export_tensor(tensor: torch.Tensor) -> bytes:
        """导出 tensor 为 IPC handle（可通过 ZMQ 发送）"""
        return torch.multiprocessing.reduction.rebuild_cuda_tensor(tensor)

    @staticmethod
    def import_tensor(handle: bytes) -> torch.Tensor:
        """从 IPC handle 恢复 tensor（零拷贝）"""
        return torch.multiprocessing.reduction.rebuild_tensor(handle)
```

#### Shared Memory — 多模态数据传输

```python
# 多模态场景：图片/视频特征在 TokenizerManager 中预处理
# 通过 shared memory 传递给 Scheduler，避免 pickle 大数据

import multiprocessing.shared_memory as shm

class SharedMemoryTransport:
    """CPU 大块数据的跨进程共享。

    用途：
    - 图片 embedding（几 MB per image）
    - 音频特征
    - 预处理后的 attention mask
    """

    def create_and_write(self, data: np.ndarray) -> str:
        """创建共享内存并写入数据，返回名称"""
        shared = shm.SharedMemory(create=True, size=data.nbytes)
        buf = np.ndarray(data.shape, dtype=data.dtype, buffer=shared.buf)
        buf[:] = data[:]
        return shared.name  # 将名称通过 ZMQ 发送

    def read_and_close(self, name: str, shape, dtype) -> np.ndarray:
        """通过名称连接共享内存并读取"""
        shared = shm.SharedMemory(name=name)
        data = np.ndarray(shape, dtype=dtype, buffer=shared.buf).copy()
        shared.close()
        shared.unlink()
        return data
```

#### NCCL — TP/PP 集合通信

```python
# TP (Tensor Parallelism): 模型权重和 activation 分片
# PP (Pipeline Parallelism): 层间传递 hidden states

# 初始化
from sglang.srt.distributed import init_distributed_environment
init_distributed_environment(tp_size=8, pp_size=2)

# AllReduce (TP): 合并各 rank 的部分结果
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=tp_group)

# Send/Recv (PP): 层间传递
torch.distributed.send(hidden_states, dst=next_rank, group=pp_group)
torch.distributed.recv(hidden_states, src=prev_rank, group=pp_group)
```

### 通信模式总结

```
TokenizerManager ──ZMQ PUSH──▶ Scheduler ──ZMQ PUSH──▶ Detokenizer
                                    │
                              NCCL AllReduce
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
              TpWorker[0]    TpWorker[1]    TpWorker[2]
                     │              │              │
                     └─────CUDA IPC / NCCL─────────┘
```

### 小结

1. **ZMQ PUSH/PULL** 用于轻量的请求/结果消息传递
2. **CUDA IPC** 实现 GPU tensor 零拷贝跨进程共享
3. **Shared Memory** 用于多模态大数据的 CPU 侧传输
4. **NCCL** 用于 TP/PP 的高性能集合通信
5. 各机制按场景选用，平衡延迟和吞吐

---

## 第17章：硬件平台抽象

### 背景与动机

SGLang 需要运行在多种硬件上：
- NVIDIA CUDA（主力）
- AMD ROCm
- Intel XPU (Gaudi, Ponte Vecchio)
- Huawei Ascend NPU
- Apple MLX (Metal)
- Moore Threads MUSA

每种硬件有不同的：内存管理 API、算子库、CUDA Graph 等效物、通信库。如果在核心代码中 if-else 判断硬件，代码会变得不可维护。

### 设计哲学

> **一套代码适配多硬件**：通过平台抽象层，核心代码只调用统一接口，具体实现由平台插件提供。

### 核心架构

```
DeviceMixin (共享设备操作基类)
├── CudaDeviceMixin(DeviceMixin)        # NVIDIA CUDA 实现
├── RocmDeviceMixin(DeviceMixin)        # AMD ROCm 实现
├── XpuDeviceMixin(DeviceMixin)         # Intel XPU 实现
├── NpuDeviceMixin(DeviceMixin)         # Huawei NPU 实现
└── ...

SRTPlatform(DeviceMixin)               # SRT 推理平台接口
├── CudaSRTPlatform(SRTPlatform, CudaDeviceMixin)
├── RocmSRTPlatform(SRTPlatform, RocmDeviceMixin)
└── MySRTPlatform(SRTPlatform, MyDeviceMixin)  # OOT 插件
```

### 核心代码

#### DeviceMixin — 设备操作基类

```python
# python/sglang/srt/platforms/device_mixin.py
class PlatformEnum(enum.Enum):
    """已知平台枚举"""
    CUDA = auto()
    ROCM = auto()
    CPU = auto()
    XPU = auto()
    MUSA = auto()
    NPU = auto()
    TPU = auto()
    MPS = auto()
    OOT = auto()     # Out-of-tree (外部插件)

class DeviceMixin:
    """共享设备抽象 — 所有平台必须实现的操作。

    方法分为两类：
    - [Active]: SGLang 核心已通过 current_platform 调用
    - [Planned]: 预留接口，待迁移
    """

    # [Active] 设备身份
    def get_device_name(self, device_id: int) -> str: ...
    def get_device_capability(self) -> DeviceCapability: ...
    def get_device_count(self) -> int: ...

    # [Active] 内存管理
    def empty_cache(self): ...
    def get_available_memory(self, device_id: int) -> int: ...
    def memory_stats(self) -> dict: ...

    # [Active] 随机种子
    def set_seed(self, seed: int): ...

    # [Planned] 设备同步
    def synchronize(self): ...
    def current_stream(self): ...
```

#### SRTPlatform — 推理平台接口

```python
# python/sglang/srt/platforms/interface.py
class SRTPlatform(DeviceMixin):
    """SRT 推理平台基类 — 在 DeviceMixin 之上添加推理特有的接口。"""

    # 支持的量化方式
    supported_quantization: list[str] = []

    # 配置生命周期
    def apply_server_args_defaults(self, server_args) -> None:
        """平台特定的默认参数（如 NPU 需要禁用某些功能）"""
        pass

    # 子系统工厂方法
    def get_default_attention_backend(self) -> str:
        """返回该平台默认的注意力后端名"""
        raise NotImplementedError

    def get_graph_runner_cls(self) -> type:
        """返回该平台的 Graph Runner 类（CUDA Graph / NPU Graph / ...）"""
        raise NotImplementedError

    def get_mha_kv_pool_cls(self) -> type:
        """返回该平台的 MHA KV 缓存池类"""
        raise NotImplementedError

    def get_mla_kv_pool_cls(self) -> type:
        """返回该平台的 MLA KV 缓存池类"""
        raise NotImplementedError
```

#### 平台选择与注册

```python
# python/sglang/srt/platforms/__init__.py
def _detect_platform() -> SRTPlatform:
    """自动检测当前硬件平台。

    优先级：
    1. SGLANG_PLATFORM 环境变量显式指定
    2. entry_points 中注册的平台插件
    3. 自动检测（torch.cuda.is_available → CUDA）
    """
    # 检查环境变量
    platform_name = envs.SGLANG_PLATFORM.get()
    if platform_name:
        return load_platform_plugin(platform_name)

    # 自动检测
    if torch.cuda.is_available():
        if is_hip():
            return RocmSRTPlatform()
        return CudaSRTPlatform()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return XpuSRTPlatform()
    # ...

# 全局单例
current_platform: SRTPlatform = _detect_platform()
```

#### 核心代码中的使用

```python
# 核心代码只依赖 current_platform，不关心具体硬件
from sglang.srt.platforms import current_platform

# 获取默认注意力后端
backend_name = current_platform.get_default_attention_backend()

# 获取 Graph Runner
GraphRunnerCls = current_platform.get_graph_runner_cls()
graph_runner = GraphRunnerCls(model_runner)

# 内存查询
available_mem = current_platform.get_available_memory(gpu_id)
```

### OOT 平台插件示例

```toml
# 第三方平台包 pyproject.toml
[project.entry-points."sglang.srt.platforms"]
my_hw = "my_hw_package.platform:MySRTPlatform"
```

```python
# my_hw_package/platform.py
from sglang.srt.platforms.interface import SRTPlatform

class MySRTPlatform(SRTPlatform, MyDeviceMixin):
    supported_quantization = ["fp8", "int8"]

    def get_default_attention_backend(self) -> str:
        return "my_custom_attention"

    def get_graph_runner_cls(self) -> type:
        from my_hw_package.graph_runner import MyGraphRunner
        return MyGraphRunner
```

### 支持的平台

| 平台 | 设备 | 插件方式 |
|------|------|----------|
| CUDA | NVIDIA A100/H100/B200 | 内置 |
| ROCm | AMD MI300X | 内置 |
| XPU | Intel Gaudi/PVC | OOT 插件 |
| NPU | Huawei Ascend 910B | OOT 插件 |
| MLX | Apple M系列 | OOT 插件 |
| MUSA | Moore Threads | OOT 插件 |

### 小结

1. **DeviceMixin + SRTPlatform** 双层抽象：设备操作 + 推理特定接口
2. **工厂方法模式** 让每个平台返回自己的实现类
3. **entry_points 注册** 让外部平台零侵入接入
4. **current_platform 全局单例** 让核心代码完全硬件无关
5. 新增硬件支持只需实现 SRTPlatform 子类 + 注册 entry_point

---

## 总结与学习路径

### 推荐阅读顺序

```
第1章 架构概览 (全局认知)
    ↓
第2章 Mixin 模式 (理解代码组织)
    ↓
第3章 Registry (理解扩展机制)  →  第4章 Backend (理解接口抽象)
    ↓
第5章 Radix Cache (核心创新)  →  第6章 Memory Pool (内存管理)
    ↓
第7章 Overlap (性能优化)  →  第15章 CUDA Graph (延迟优化)
    ↓
第12章 Continuous Batching (调度核心)
    ↓
第14章 Speculative Decoding (高级优化)  →  第13章 PD 分离 (部署优化)
    ↓
第8-11章 工程基础设施 (环境变量、JIT、类型分发、Plugin)
    ↓
第16-17章 系统层 (IPC、硬件抽象)
```

### 核心设计原则总结

| 原则 | 体现 |
|------|------|
| 开放-封闭原则 | Registry、Plugin、Backend 抽象 |
| 单一职责原则 | Mixin、分层内存池 |
| 依赖倒置原则 | 面向接口编程（AttentionBackend ABC） |
| 组合优于继承 | Scheduler 的 6 个 Mixin |
| 按需加载 | JIT Kernel、Plugin 自动发现 |
| 零拷贝 | CUDA IPC、Shared Memory |
| 隐藏延迟 | Overlap、Multi-Stream |
| 预分配 | Memory Pool、CUDA Graph buffer |

### 进阶阅读

- SGLang 论文：[Efficient Programming Model for LLM Serving](https://arxiv.org/abs/2312.07104)
- EAGLE 论文：[EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- DeepSeek V3 技术报告：MLA + Multi-Stream Overlap
- Radix Attention 论文：SGLang 核心前缀缓存方案
