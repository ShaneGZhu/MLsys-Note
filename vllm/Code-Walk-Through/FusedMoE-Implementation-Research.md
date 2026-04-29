# vLLM v1 FusedMoE 实现方式深度调研报告

> **调研目标：** 理解 vLLM v1 FusedMoE 整体架构，为移植/复现到其他框架（如 PaddlePaddle）提供完整参考。
> **调研版本：** vLLM main branch（2025-04）
> **调研方向：** A（模块化内核架构）、B（多后端计算内核）、C（EP/TP/DP 并行策略）、D（高性能通信优化）

---

## 目录

1. [核心文件索引](#核心文件索引)
2. [方向 A：模块化内核架构](#方向-a模块化内核架构四段分离设计)
3. [方向 B：多后端计算内核](#方向-b多后端计算内核)
4. [方向 C：EP/TP/DP 并行策略](#方向-ceptpd-并行策略)
5. [方向 D：高性能通信优化](#方向-d高性能通信优化)
6. [关键设计约束](#关键设计约束移植注意事项)
7. [验证方法](#验证方法)

---

## 核心文件索引

```
vllm/model_executor/layers/fused_moe/
├── layer.py                              # FusedMoE 层入口（★ 移植起点）
├── modular_kernel.py                     # ★ 四段分离架构核心抽象（FusedMoEKernel/Impl/抽象基类）
├── config.py                             # FusedMoEConfig / FusedMoEParallelConfig / FusedMoEQuantConfig
├── fused_moe.py                          # Triton 内核实现（TritonExperts + fused_moe_kernel）
├── fused_moe_method_base.py              # FusedMoEMethodBase 量化方法抽象基类
├── fused_moe_modular_method.py           # FusedMoEModularMethod（对接 quantization 子系统）
├── topk_weight_and_reduce.py             # TopKWeightAndReduce 各实现类
├── activation.py                         # MoEActivation 枚举（SILU/GELU/SwiGLU/...）
├── utils.py                              # moe_kernel_quantize_input / moe_permute 等工具
│
├── router/                               # 路由模块（全部继承 BaseRouter）
│   ├── base_router.py                    # BaseRouter 抽象基类
│   ├── fused_topk_router.py              # 标准 TopK 路由（softmax/sigmoid 两路）
│   ├── grouped_topk_router.py            # ★ 分组 TopK（DeepSeekV3/V4 专用）
│   ├── fused_topk_bias_router.py         # TopK+Bias 路由（Llama4-MoE）
│   ├── router_factory.py                 # 工厂函数：根据 config 选择路由实现
│   └── zero_expert_router.py             # 零专家路由（测试/占位用）
│
├── prepare_finalize/                     # PrepareAndFinalize 各实现
│   ├── no_dp_ep.py                       # ★ 单 Rank 路径（无通信）最简移植起点
│   ├── naive_dp_ep.py                    # DP+EP AllToAll（torch AllReduce/RS 路径）
│   ├── deepep_ht.py                      # DeepEP 高吞吐异步路径（NVLink）
│   ├── deepep_ll.py                      # DeepEP 低延迟 + CUDA Graph 路径
│   ├── batched.py                        # BatchedPrepareAndFinalize（EP batched 格式）
│   ├── nixl_ep.py                        # NIXL EP 路径（跨节点 RDMA）
│   ├── flashinfer_nvlink_one_sided.py    # FlashInfer NVLink 单侧 AllToAll
│   └── flashinfer_nvlink_two_sided.py    # FlashInfer NVLink 双侧 AllToAll
│
├── experts/                              # FusedMoEExpertsModular 各后端实现
│   ├── cutlass_moe.py                    # ★ CUTLASS FP8/FP4/MXFP4/W4A8 系列
│   ├── deep_gemm_moe.py                  # ★ DeepGEMM FP8/FP4（H100/B200+）
│   ├── batched_deep_gemm_moe.py          # DeepGEMM Batched（EP batched 格式）
│   ├── trtllm_fp8_moe.py                 # FlashInfer TrtLLM FP8（Modular/Monolithic）
│   ├── trtllm_bf16_moe.py                # FlashInfer TrtLLM BF16
│   ├── trtllm_mxfp4_moe.py              # FlashInfer TrtLLM MXFP4
│   ├── trtllm_nvfp4_moe.py              # FlashInfer TrtLLM NVFP4
│   ├── flashinfer_cutedsl_moe.py         # FlashInfer CuteDSL（Standard 格式）
│   ├── flashinfer_cutedsl_batched_moe.py # FlashInfer CuteDSL（Batched 格式）
│   ├── xpu_moe.py                        # XPU 专用（BF16/FP8/MXFP4）
│   └── gpt_oss_triton_kernels_moe.py     # OAI GPT-OSS Triton 内核（兼容层）
│
├── oracle/                               # 后端选择 Oracle（按量化格式分文件）
│   ├── unquantized.py                    # BF16/FP16 无量化后端选择
│   ├── fp8.py                            # FP8 系列后端选择
│   ├── int8.py                           # INT8 W8A8 后端选择
│   ├── int_wna16.py                      # WNA16 INT4/INT8 权重量化
│   ├── nvfp4.py                          # NVFP4 后端选择
│   ├── mxfp4.py                          # MXFP4 后端选择
│   └── mxfp8.py                          # MXFP8 后端选择
│
└── runner/                               # MoE 运行时调度
    ├── moe_runner.py                     # ★ MoERunner（主控流程 + custom op 注册）
    ├── moe_runner_interface.py           # MoERunnerInterface 抽象基类
    └── shared_experts.py                 # SharedExperts（与 EP dispatch 重叠执行）

vllm/v1/worker/
├── ubatching.py                          # ★ DBO 微批次双缓冲重叠（UBatchContext + API）
├── workspace.py                          # ★ WorkspaceManager（中间缓冲区统一管理）
└── gpu/eplb_utils.py                     # EPLBController（动态专家负载均衡）
```

### 关键类层次速览

```
FusedMoEKernel（@final，不可继承）
  └─ impl: FusedMoEKernelModularImpl | FusedMoEKernelMonolithicImpl

FusedMoEKernelModularImpl
  ├─ prepare_finalize: FusedMoEPrepareAndFinalizeModular
  │     ├─ MoEPrepareAndFinalizeNoDPEPModular     （单 rank）
  │     ├─ MoEPrepareAndFinalizeNaiveDPEPModular  （naive AllToAll）
  │     ├─ DeepEPHTPrepareAndFinalize             （DeepEP 高吞吐）
  │     └─ DeepEPLLPrepareAndFinalize             （DeepEP 低延迟）
  └─ fused_experts: FusedMoEExpertsModular
        ├─ TritonExperts                          （通用 Triton）
        ├─ CutlassExpertsFp8 / CutlassBatchedExpertsFp8
        ├─ DeepGemmExperts / DeepGemmFP4Experts
        └─ ... （30+ 实现类，全部共享同一 apply() 签名）
```

---

## 方向 A：模块化内核架构（四段分离设计）

### A1. 整体数据流

vLLM v1 最核心的设计创新：将 MoE 前向拆分为**四段独立可替换的模块**，彻底解决了 v0 中 N×M 排列组合式重复实现问题。

```
输入 hidden_states [M, K]  +  router_logits [M, E]
          │
          ▼
    ┌─────────────┐
    │   Router    │  select_experts()
    │             │  → topk_ids [M, topk]，topk_weights [M, topk]
    └──────┬──────┘
           │
           ▼  ┌──────────────────────────────────────────────┐
              │  MoERunner._forward_impl()  主控流程         │
              │  1. gate 线性变换（router_logits）            │
              │  2. 可选 routed_input_transform（如 fp8 量化）│
              │  3. router.select_experts()                  │
              │  4. quant_method.apply()                     │
              │     └→ FusedMoEKernel.apply()               │
              │  5. _maybe_reduce_final_output()             │
              └──────────────────────────────────────────────┘
                    │
                    ▼  FusedMoEKernelModularImpl.apply()
    ┌──────────────────────────────┐
    │  PrepareAndFinalize.prepare()│  量化输入 + EP dispatch（可选）
    │  （或 prepare_async + DBO）  │  → a1q（量化激活）[M,K] 或 [E,max_M,K]
    │                              │  → a1q_scale（per-token/per-block）
    │                              │  → ExpertTokensMetadata（每 expert token 数）
    │                              │  → new_topk_ids / new_topk_weights（EP 后）
    └──────────────┬───────────────┘
                   │
                   ▼  _allocate_buffers() via WorkspaceManager
    ┌──────────────────────────────┐
    │  FusedMoEExperts.apply()     │  GEMM1 → Activation → GEMM2
    │                              │  workspace13: GEMM1 中间结果
    │  各后端（Triton/CUTLASS/     │  workspace2:  activation 中间结果
    │     DeepGEMM/FlashInfer/...）│  fused_out: [M,K] 或 [E,max_M,K]
    └──────────────┬───────────────┘
                   │
                   ▼  finalize_async + DBO（可选）
    ┌──────────────────────────────┐
    │  PrepareAndFinalize.finalize()│  topk 加权求和 + EP combine（可选）
    │  TopKWeightAndReduce 实现类   │  → output [M, K]（in-place 写入）
    └──────────────────────────────┘
          │
          ▼
    输出 [M, K]  + _maybe_reduce_final_output()（条件 TP AllReduce）
          │
          ▼（存在 shared experts 时）
    shared_out = SharedExperts.forward(shared_experts_input)  # 与 EP 重叠
    final = output + tensor_model_parallel_all_reduce(shared_out)
```

**M 的含义在不同阶段发生变化：**
- Router 后：`M = num_input_tokens`
- DeepEP HT dispatch 后：`M = num_tokens_dispatched_to_this_rank`（变长）
- DeepEP LL dispatch 后：`M = max_tokens_per_rank`（固定，CUDA Graph 兼容）

### A2. 关键抽象类 API

#### `FusedMoEPrepareAndFinalizeModular`（`modular_kernel.py:251`）

```python
# ── 同步接口 ──────────────────────────────────────────────────────────────
def prepare(
    self,
    a1: torch.Tensor,                    # 未量化的 MoE 输入，shape=[M, K]
    topk_weights: torch.Tensor,          # top-k 路由权重，shape=[M, topk]
    topk_ids: torch.Tensor,              # top-k expert id，shape=[M, topk]
    num_experts: int,                    # 全局 expert 总数
    expert_map: torch.Tensor | None,     # global→local expert 索引映射，shape=[E]
    apply_router_weight_on_input: bool,  # 是否在量化前将权重乘到输入（仅 topk=1）
    quant_config: FusedMoEQuantConfig,   # 量化配置（包含 w/a scale/dtype/shape）
    defer_input_quant: bool = False,     # 是否延迟量化到 Experts.apply() 内部
) -> tuple:
    # 返回 PrepareResultType = tuple[a1q, a1q_scale, expert_tokens_meta,
    #                                new_topk_ids, new_topk_weights]
    # new_topk_ids/weights 为 None 时上层保持原值不变

def finalize(
    self,
    output: torch.Tensor,                 # 就地写入，shape=[M, K]
    fused_expert_output: torch.Tensor,    # experts 的输出（未加权或已加权）
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool,
    weight_and_reduce_impl: TopKWeightAndReduce,
) -> None

# ── 必须实现的属性/方法 ────────────────────────────────────────────────────
@property
def activation_format(self) -> FusedMoEActivationFormat: ...
    # Standard（2D）或 BatchedExperts（3D），必须与 FusedMoEExperts 一致

def output_is_reduced(self) -> bool: ...
    # True = combine 内已做 AllReduce，上层跳过；HT/LL 返回 True

def max_num_tokens_per_rank(self) -> int | None: ...
    # LL 路径返回预分配最大 token 数；其他返回 None

def topk_indices_dtype(self) -> torch.dtype | None: ...
    # DeepEP HT/LL 要求 int64；其他返回 None（使用默认 int32）

def num_dispatchers(self) -> int: ...
    # EP dispatch 的 rank 数（= dp_size × ep_size）

# ── 异步接口（DBO 通信/计算重叠）──────────────────────────────────────────
def supports_async(self) -> bool:
    return False   # 默认不支持；DeepEP HT/LL 覆盖并返回 True

def prepare_async(self, ...) -> ReceiverType | tuple[Callable, ReceiverType]:
    # HT: 返回单个 receiver 闭包
    # LL: 返回 (hook, receiver) 二元组
    #   hook 由 DBO 注册到【下一批次】延迟执行（接收完成动作）
    #   receiver 直接调用完成量化后处理

def finalize_async(self, ...) -> Callable | tuple[Callable, Callable]:
    # HT: 返回单个 receiver 闭包（等待 event + output.copy_）
    # LL: 返回 (recv_hook, noop) 二元组
```

#### `FusedMoEExpertsModular`（`modular_kernel.py:763`）

```python
# ── 核心计算接口（所有子类共享同一签名）───────────────────────────────────
def apply(
    self,
    output: torch.Tensor,                   # 写入目标 [M,K] 或 [E,max_M,K]
    hidden_states: torch.Tensor,            # 已量化输入 [M,K] 或 [E,max_M,K]
    w1: torch.Tensor,                       # gate+up fused，shape=[E, 2N, K]
    w2: torch.Tensor,                       # down_proj，shape=[E, K, N]
    topk_weights: torch.Tensor,             # [M, topk]（EP 后为 dispatched topk）
    topk_ids: torch.Tensor,                 # [M, topk]，同上
    activation: MoEActivation,              # silu / gelu / swiglu 等
    global_num_experts: int,
    expert_map: torch.Tensor | None,        # shape=[global_E]，-1 表示不属于本 rank
    a1q_scale: torch.Tensor | None,         # 动态激活 scale（来自 prepare）
    a2_scale: torch.Tensor | None,          # 第二段静态 weight scale（来自 QuantConfig）
    workspace13: torch.Tensor,              # GEMM1 中间缓冲区
    workspace2: torch.Tensor,               # activation 中间缓冲区
    expert_tokens_meta: ExpertTokensMetadata | None,
    apply_router_weight_on_input: bool,
) -> None

# ── workspace 形状计算（各后端不同，必须实现）──────────────────────────────
def workspace_shapes(
    self,
    M: int, N: int, K: int, topk: int,
    global_num_experts: int, local_num_experts: int,
    expert_tokens_meta: ExpertTokensMetadata | None,
    activation: MoEActivation,
) -> tuple[tuple, tuple, tuple]:
    # 返回 (workspace13_shape, workspace2_shape, output_shape)

# ── 其他必须实现的抽象方法 ─────────────────────────────────────────────────
@staticmethod
def activation_format() -> FusedMoEActivationFormat: ...

def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce: ...
    # TritonExperts    → TopKWeightAndReduceNoOP（MUL_ROUTED_WEIGHT 内核内部做）
    # CutlassFP8       → TopKWeightAndReduceNoOP（unpermute 内部做）
    # DeepGemmExperts  → TopKWeightAndReduceNoOP
    # BatchedTriton    → TopKWeightAndReduceDelegate（委托给 PrepareAndFinalize）

def moe_problem_size(hidden_states, w1, w2, topk_ids) -> (E, M, N, K, topk): ...
    # 从 tensor shape 中推导问题规模（各后端形状约定不同）
```

#### `TopKWeightAndReduce`（`modular_kernel.py:118`，四个具体实现）

```python
# ── 抽象基类签名 ──────────────────────────────────────────────────────────
def apply(
    self,
    output: torch.Tensor | None,          # None 时内部 alloc
    fused_expert_output: torch.Tensor,    # [M, topk, K] 或 [E, batch, K]
    topk_weights: torch.Tensor,           # [M, topk]
    topk_ids: torch.Tensor,               # [M, topk]
    apply_router_weight_on_input: bool,   # True 时权重已在 prepare 乘过，跳过
) -> torch.Tensor                         # [M, K]

# ── 四个具体实现（topk_weight_and_reduce.py）──────────────────────────────
TopKWeightAndReduceNoOP          # 权重已在 expert kernel 内部应用（Triton/CUTLASS/DeepGEMM）
TopKWeightAndReduceDelegate      # 委托给 PrepareAndFinalize（DeepEP LL 的 combine 内核全包）
TopKWeightAndReduceContiguous    # Standard 格式：(m, topk, K).mul_(w) → ops.moe_sum
TopKWeightAndReduceNaiveBatched  # BatchedExperts 格式：for-loop per expert
```

#### `FusedMoEQuantConfig.make()`（`config.py:484`，量化配置工厂）

```python
FusedMoEQuantConfig.make(
    quant_dtype=None,               # None=BF16，torch.float8_e4m3fn，"nvfp4"，"mxfp4"...
    per_act_token_quant=False,      # True → a_shape=PER_TOKEN（动态 per-token scale）
    per_out_ch_quant=False,         # True → w_shape=PER_TOKEN（per-channel weight scale）
    block_shape=[128, 128],         # FP8 block 量化时指定 [row_block, col_block]
    w1_scale=..., w2_scale=...,     # weight static scale tensors
    a1_scale=..., a2_scale=...,     # activation static scale（None=动态量化）
    g1_alphas=..., g2_alphas=...,   # NVFP4 per-channel gscale
    a1_gscale=..., a2_gscale=...,   # NVFP4 全局 gscale
    w1_zp=..., w2_zp=...,           # INT4 zero point
    w1_bias=..., w2_bias=...,       # GPT-OSS Triton bias
    weight_dtype=None,              # 单独指定权重 dtype（默认与 quant_dtype 相同）
    is_nvfp4_scale_swizzled=True,   # NVFP4 scale 是否已做 swizzle 排列
) -> FusedMoEQuantConfig
```

#### `FusedMoEActivationFormat` 枚举

```python
class FusedMoEActivationFormat(Enum):
    Standard       = ("standard",)        # shape: [num_tokens, hidden_dim]
    BatchedExperts = ("batched_experts",) # shape: [num_experts, max_tokens_per_expert, hidden_dim]
```

- **Standard**：所有 token 连续排列，通过 `sorted_token_ids` 索引找 expert
- **BatchedExperts**：按 expert 分桶，每个 expert 占固定槽位，固定 shape → CUDA Graph 兼容
- `_post_init_setup()` 在 `FusedMoEKernel.__init__` 末尾断言两者一致：
  ```python
  assert prepare_finalize.activation_format == fused_experts.activation_format()
  ```

### A3. ExpertTokensMetadata（三段间的关键元数据）

```python
# 定义位置：modular_kernel.py:97
@dataclass
class ExpertTokensMetadata:
    expert_num_tokens:     torch.Tensor        # GPU，shape=[local_num_experts], int32
                                               # expert_num_tokens[i] = 分配到 local expert i 的 token 数
    expert_num_tokens_cpu: torch.Tensor | None # CPU 副本（DeepGEMM contiguous layout 预计算用）

    @staticmethod
    def make_from_list(nums: list[int], device) -> "ExpertTokensMetadata":
        # DeepEP HT 的 dispatch 返回 expert_num_tokens_per_expert_list（CPU list）
        # 此方法将其转为 GPU tensor，并保留 CPU 副本
        gpu = torch.tensor(nums, device=device, dtype=torch.int32)
        cpu = torch.tensor(nums, dtype=torch.int32)
        return ExpertTokensMetadata(gpu, cpu)
```

**传递链路：**
| 阶段 | 行为 |
|------|------|
| `prepare()` 产生 | EP dispatch 后从 `expert_num_tokens_per_expert_list` 构造；单 rank 路径返回 `None`（kernel 内部从 `topk_ids` 推导）|
| `_allocate_buffers()` 使用 | `workspace_shapes()` 用它计算 DeepGEMM `M_sum`（各 expert 对齐后的 token 总数）|
| `experts.apply()` 消费 | CUTLASS batched grouped GEMM / DeepGEMM contiguous layout 需要变长 M 分组信息 |
| `finalize()` 不用 | 权重规约只需 `topk_ids/topk_weights`，无需 per-expert token 数 |

**`M_sum` 计算（DeepGEMM 专用，`deep_gemm_utils.py`）：**
```python
def compute_aligned_M(M, topk, local_num_experts, block_m, expert_tokens_meta):
    if expert_tokens_meta is None:
        # 单 rank 路径：保守上界
        return round_up(M * topk, block_m)
    # 已知每个 expert 的 token 数，精确计算
    nums = expert_tokens_meta.expert_num_tokens_cpu
    return sum(round_up(int(n), block_m) for n in nums)
    # block_m = 64（DeepGEMM contiguous layout 的 M 对齐粒度）
```

### A4. Modular vs Monolithic 两种执行模式

| 特征 | Modular | Monolithic |
|------|---------|------------|
| 路由输入 | 外部路由（topk_ids + topk_weights） | 内部路由（router_logits → 内核自行 top-k） |
| 异步通信 | 支持（DBO 重叠） | 不支持 |
| workspace 管理 | 使用 WorkspaceManager | 无 |
| SharedExperts 重叠 | ✅（supports_async 时） | ❌ |
| 代表后端 | Triton、CUTLASS、DeepGEMM、DeepEP | FlashInfer TRTLLM BF16/FP8 |
| 移植优先级 | ★★★★★ 优先 | 可后续选择性移植 |

**`FusedMoEKernel.__init__` 选择逻辑（`modular_kernel.py:1484`）：**
```python
@final  # 不允许继承
class FusedMoEKernel:
    def __init__(self, prepare_finalize, fused_experts, shared_experts=None, inplace=False):
        if isinstance(prepare_finalize, FusedMoEPrepareAndFinalizeModular) \
           and isinstance(fused_experts, FusedMoEExpertsModular):
            self.impl = FusedMoEKernelModularImpl(
                prepare_finalize, fused_experts, shared_experts, inplace)
        elif isinstance(prepare_finalize, FusedMoEPrepareAndFinalizeMonolithic) \
             and isinstance(fused_experts, FusedMoEExpertsMonolithic):
            assert not inplace
            self.impl = FusedMoEKernelMonolithicImpl(prepare_finalize, fused_experts)
        else:
            raise ValueError("prepare_finalize 和 fused_experts 必须同为 Modular 或 Monolithic")

        self._post_init_setup()  # 验证 activation_format 一致性

    def _post_init_setup(self):
        self.prepare_finalize.post_init_setup(self.impl.fused_experts)
        # ★ 关键断言：确保 prepare_finalize 和 experts 输出格式一致
        assert (self.prepare_finalize.activation_format
                == self.fused_experts.activation_format())
```

**`FusedMoEKernelModularImpl.apply()` 完整主控流程（`modular_kernel.py:1332`）：**
```python
def apply(self, hidden_states, w1, w2, topk_ids, topk_weights,
          activation, global_num_experts, expert_map,
          apply_router_weight_on_input, shared_experts_input=None):

    # 1. 分配输出缓冲区
    output = hidden_states if self.inplace else torch.empty_like(hidden_states)

    # 2. prepare（含 DBO hook/yield 逻辑）
    a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
        hidden_states, topk_weights, topk_ids,
        global_num_experts, expert_map, apply_router_weight_on_input)

    # 3. expert GEMM（GEMM1 → activation → GEMM2）
    fused_out = self._fused_experts(
        in_dtype=hidden_states.dtype,
        a1q=a1q, a1q_scale=a1q_scale,
        w1=w1, w2=w2,
        topk_weights=topk_weights, topk_ids=topk_ids,
        activation=activation,
        global_num_experts=global_num_experts,
        local_num_experts=w1.shape[0],
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_tokens_meta=expert_tokens_meta,
    )

    # 4. finalize（含 DBO + SharedExperts 重叠）
    return self._finalize(output, fused_out, hidden_states,
                          topk_weights, topk_ids,
                          apply_router_weight_on_input,
                          shared_experts_input=shared_experts_input)
```

**`_prepare()` 的 DBO 分支（`modular_kernel.py:1115`）：**
```python
def _prepare(self, hidden_states, topk_weights, topk_ids, ...):
    if not self.prepare_finalize.supports_async():
        # 同步路径（no_dp_ep / naive_dp_ep）
        assert not dbo_enabled()
        a1q, a1q_scale, meta, ids, weights = self.prepare_finalize.prepare(...)
    else:
        # 异步路径（DeepEP HT / LL）
        dbo_maybe_run_recv_hook()   # 执行上一批次注册的 recv_hook

        prepare_ret = self.prepare_finalize.prepare_async(...)
        # HT 返回 receiver 闭包；LL 返回 (hook, receiver)
        hook, receiver = (prepare_ret if isinstance(prepare_ret, tuple)
                          else (None, prepare_ret))

        if hook is not None:
            if dbo_enabled():
                dbo_register_recv_hook(hook)  # 把 hook 注册到【下一批次】
                dbo_yield()                   # CPU 让出，等另一批次先跑
            else:
                hook()                        # 非 DBO 时直接执行
        a1q, a1q_scale, meta, ids, weights = receiver()
```

### A5. 初始化时机（权重加载后）

```
GPUModelRunner.__init__()
  └─ model_loader.load_model(model_config)        # 加载权重
       └─ prepare_communication_buffer_for_model(model)   # 权重加载完成后调用
            └─ for each FusedMoE layer:
                 └─ layer.maybe_init_modular_kernel(ep_group, ...)
                      ├─ oracle.select_backend(moe_config, quant_config, ...)
                      │    → 选择 PrepareFinalize 实现类
                      │    → 选择 Experts 实现类
                      ├─ FusedMoEQuantConfig.make(w1_scale, w2_scale, ...)
                      ├─ convert_to_xxx_moe_kernel_format()  # 权重格式转换（如 FP8 reorder）
                      └─ FusedMoEKernel(prepare_finalize, fused_experts, ...)
                           → self.quant_method = FusedMoEModularMethod(kernel)
```

**权重格式转换时机（一次性，在 `make_xxx_moe_kernel()` 内）：**
```python
# 以 FP8 为例（oracle/fp8.py）
def convert_to_fp8_moe_kernel_format(layer, backend):
    if backend == Fp8MoeBackend.DEEPGEMM:
        # DeepGEMM 需要转置并按行 shuffle
        layer.w1 = deep_gemm_permute_weights(layer.w1)
        layer.w2 = deep_gemm_permute_weights(layer.w2)
    elif backend == Fp8MoeBackend.AITER:
        # ROCm AITER 需要特殊 shuffle
        layer.w1, layer.w2 = rocm_aiter_shuffle(layer.w1, layer.w2)
```

### A6. workspace 缓冲区 shape（移植时需精确复现）

workspace 命名约定：`workspace13` 因同时被 cache1（GEMM1 输出）和 cache3（最终输出）复用而得名。

| 后端 | 格式 | workspace13_shape | workspace2_shape | output_shape |
|------|------|-------------------|------------------|--------------|
| **TritonExperts** | Standard | `(M, topk, max(N_act, K))` | `(M, topk, max(N, K))` | `(M, K)` |
| **CutlassExpertsFp8** | Standard | `(M×topk, max(N, K))` | `(M×topk, max(N_act, K))` | `(M, K)` |
| **CutlassBatchedExpertsFp8** | Batched | `(E, M×dp, max(N, K))` | `(E, M×dp, max(N_act, K))` | `(E, M, K)` |
| **CutlassExpertsFp4/Mxfp4** | Standard | `(M×topk, max(2N, K))` | `(M×topk, N)` | `(M, K)` |
| **DeepGemmExperts** | Standard | `(M_sum, max(N_act, K))` | `(M_sum, max(N, K))` | `(M, K)` |
| **DeepGemmFP4Experts** | Standard | `(M_sum, max(N_act, K))` | `(M_sum, max(N, K))` | `(M, K)` |

> 符号说明：`N = intermediate_size_per_partition`（即 `w1.shape[1] // 2`，因 gate+up 拼接）；`N_act = adjust_N_for_activation(N, activation)`（SwiGLU 时 = `N//2`，其余 = `N`）；`M_sum = Σ round_up(expert_tokens[i], block_m)` ；`E = local_num_experts`；`dp = num_dispatchers`

**WorkspaceManager 分配流程（`_allocate_buffers()`，`modular_kernel.py:1203`）：**
```python
# 每次 apply() 调用时（非 CUDA Graph 时动态分配）
ws13_shape, ws2_shape, out_shape = self.fused_experts.workspace_shapes(...)
workspace13, workspace2, fused_out = workspace_manager.get_simultaneous(
    (ws13_shape, in_dtype),
    (ws2_shape,  in_dtype),
    (out_shape,  in_dtype),
)
# get_simultaneous 内部：
#   total = Σ round_up(bytes(s,d), 256)  # 256 字节对齐
#   大 uint8 buffer 按 dbo_current_ubatch_id() 选 slot（DBO 时各批次独立）
#   按偏移切片后 .view(dtype).reshape(shape)
```

---

## 方向 B：多后端计算内核

### B1. FusedMoEExpertsModular 实现类全景图

```
移植优先级  类名                              文件                            量化格式         格式
★★★★★  TritonExperts                    fused_moe.py                    BF16/FP8/INT8   Standard
★★★★   CutlassExpertsFp8               experts/cutlass_moe.py          FP8 W8A8        Standard
★★★★   CutlassBatchedExpertsFp8        experts/cutlass_moe.py          FP8 W8A8        Batched
★★★★   DeepGemmExperts                 experts/deep_gemm_moe.py        FP8 block       Standard（H100+）
★★★★   DeepGemmFP4Experts              experts/deep_gemm_moe.py        MXFP4           Standard（SM100+）
★★★    BatchedTritonExperts            fused_batched_moe.py            BF16            Batched
★★★    TrtLlmFp8ExpertsModular         experts/trtllm_fp8_moe.py       FP8             Standard
★★★    TrtLlmBf16ExpertsModular        experts/trtllm_bf16_moe.py      BF16            Standard
★★★    CutlassExpertsFp4               experts/cutlass_moe.py          NVFP4           Standard
★★★    CutlassExpertsMxfp4             experts/cutlass_moe.py          MXFP4           Standard
★★★    CutlassExpertsW4A8Fp8           experts/cutlass_moe.py          W4A8 INT4×FP8   Standard
★★     FlashInferCuteDSLExperts         experts/flashinfer_cutedsl_moe  NVFP4/BF16      Standard
★★     BatchedDeepGemmExperts           experts/batched_deep_gemm_moe   FP8             Batched
★★     MarlinExperts                   fused_marlin_moe.py             WNA16 INT4/INT8  Standard
★      AiterExperts                    rocm_aiter_fused_moe.py         BF16/FP8        Standard（ROCm）
★      XPUExperts / XPUExpertsFp8      experts/xpu_moe.py              BF16/FP8        Standard（XPU）
```

**`finalize_weight_and_reduce_impl()` 返回值速查（决定权重乘法由谁做）：**

| Expert 类 | 返回值 | 原因 |
|-----------|--------|------|
| TritonExperts | `TopKWeightAndReduceNoOP` | Triton kernel 内部 `MUL_ROUTED_WEIGHT` 标志在第二次 GEMM 乘上路由权重 |
| CutlassExpertsFp8（Standard） | `TopKWeightAndReduceNoOP` | `moe_unpermute` 内部完成加权 reduce |
| CutlassBatchedExpertsFp8 | `TopKWeightAndReduceDelegate` | 委托 PrepareAndFinalize（如 DeepEP LL combine 全包）|
| DeepGemmExperts | `TopKWeightAndReduceNoOP` | `deepgemm_unpermute_and_reduce` 内部完成 |
| BatchedTritonExperts | `TopKWeightAndReduceDelegate` | 委托给 Batched/DeepEPLL PrepareAndFinalize |

### B2. Triton 内核机制（最通用路径）

#### token 排列流程

```python
# Step 1: topk_ids [M, topk] → 排列元数据（CUDA C++ kernel，moe_align_block_size.py）
sorted_token_ids,        # shape=[max_num_tokens_padded]，按 expert 排序的 token 索引
expert_ids,              # shape=[max_num_m_blocks]，每个 M-block 对应的 expert id
num_tokens_post_padded   # 标量：对齐后总 token 数（存 GPU tensor，内核通过指针读取）
= moe_align_block_size(topk_ids, BLOCK_SIZE_M=64, num_experts, expert_map)

# 分配规则（C++ kernel 内部逻辑）：
# max_num_tokens_padded = M*topk + num_experts * (BLOCK_SIZE_M - 1)
# max_num_m_blocks = ceil(max_num_tokens_padded / BLOCK_SIZE_M)
# expert_ids[i] = -1 表示该 block 属于本 rank 没有的 expert（kernel 内写零）
```

#### `fused_moe_kernel` Triton 内核核心逻辑（`fused_moe.py:313`）

```python
@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr, b_bias_ptr, a_scale_ptr, b_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_asm, stride_ask, stride_bse, stride_bsk, stride_bsn, stride_bbe, stride_bbn,
    group_n: tl.constexpr, group_k: tl.constexpr,    # block 量化分组大小
    naive_block_assignment: tl.constexpr,             # 不使用 sorted_token_ids（特殊路径）
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,  # True = 在 kernel 内部乘路由权重（第二次 GEMM 用）
    top_k: tl.constexpr,
    compute_type: tl.constexpr,       # 累加类型（bfloat16/float16/float32）
    use_fp8_w8a8: tl.constexpr, use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr, per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # 1. 2D grid → grouped tile 分配（L2 cache 友好）
    pid = tl.program_id(axis=0)
    pid_m, pid_n = grouped_launch_grid_decompose(pid, EM, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # 2. 早退：超出有效 token 范围
    if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr): return

    # 3. 获取 token 索引和 expert ID
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1: write_zeros_to_output(...); return   # 本 rank 无此 expert

    # 4. 计算 A/B 指针（A 按 token_id // top_k 索引原始行）
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 5. 量化 scale 加载（三种模式）
    # block 量化：每 BLOCK_K 步加载一次 a_scale/b_scale（按 group_k/group_n 计算偏移）
    # per-channel：b_scale = b_scale[expert_id, :N]，a_scale = a_scale[token_id]
    # per-tensor：a_scale = scalar，b_scale = b_scale[expert_id]

    # 6. 主循环 GEMM（K 方向分块）
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=...)
        b = tl.load(b_ptrs, mask=...)
        if use_fp8_w8a8 and group_k > 0:   # block 量化：每迭代乘 scale
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        elif use_fp8_w8a8:                 # per-token/per-tensor：一次性 scale
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak; b_ptrs += BLOCK_SIZE_K * stride_bk

    # 7. 反量化（per-tensor/per-channel 模式在此统一乘 scale）
    if use_fp8_w8a8 and not block_quant:
        accumulator = accumulator * a_scale * b_scale

    # 8. 路由权重（MUL_ROUTED_WEIGHT=True 时在 float32 精度下乘）
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask)
        accumulator *= moe_weight[:, None]

    # 9. 写回（精度转换后 store）
    tl.store(c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :],
             accumulator.to(compute_type), mask=c_mask)
```

#### `TritonExperts.apply()` 两次 GEMM 完整流程（`fused_moe.py:1985`）

```python
def apply(self, output, hidden_states, w1, w2, ...):
    E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
    # N = w1.shape[1]（= 2 * intermediate_size_per_partition，因 gate+up 拼接）

    # 分配中间缓冲区（从 workspace 切片）
    intermediate_cache1 = workspace2.view(..., N)            # [M, topk, N] GEMM1 输出（gate+up）
    intermediate_cache2 = workspace13.view(..., N//2)        # [M*topk, N//2] activation 输出
    intermediate_cache3 = workspace2.view(..., K)            # [M, topk, K] GEMM2 输出

    # 获取 expert 分配元数据
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        _prepare_expert_assignment(topk_ids, E, expert_map, ...)

    # GEMM 1：hidden × w1 → intermediate_cache1
    # mul_routed_weight=False（第一次 GEMM 不乘权重）
    invoke_fused_moe_triton_kernel(
        hidden_states, w1, intermediate_cache1,
        a1q_scale, self.w1_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=False, top_k=top_k, ...)

    # 可选 LoRA w13（加到 intermediate_cache1 上）

    # Activation（intermediate_cache1 → intermediate_cache2）
    # 通过 ops.silu_and_mul / gelu_and_mul 等融合内核
    self.activation(activation, intermediate_cache2, intermediate_cache1.view(-1, N))

    # 可选对 intermediate_cache2 做 FP8 requant（a2q, a2q_scale）
    qintermediate, a2q_scale = moe_kernel_quantize_input(intermediate_cache2, ...)

    # GEMM 2：activated × w2 → intermediate_cache3
    # mul_routed_weight=True（第二次 GEMM 内核内乘路由权重），top_k=1
    invoke_fused_moe_triton_kernel(
        qintermediate, w2, intermediate_cache3,
        a2q_scale, self.w2_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=not apply_router_weight_on_input, top_k=1, ...)

    # 可选 LoRA w2

    # Reduce：(M, topk, K) → (M, K)  via ops.moe_sum
    self.moe_sum(intermediate_cache3, output)
```

#### 权重 shape 约定

```
w1 (gate+up fused，w13): [E, 2*N, K]
    E = num_local_experts
    N = intermediate_size_per_partition（= intermediate_size // tp_size）
    K = hidden_size
    维度 1 = 2*N：前 N 行 = gate_proj，后 N 行 = up_proj

w2 (down_proj):          [E, K, N]
    K = hidden_size（输出维度）
    N = intermediate_size_per_partition（gated 激活后维度 = N//2 → 不含因子 2）
    注意：w2 第 2 维是 K（输出），第 3 维是 N（输入），与 w1 相反

weight scale shape:
    per-tensor:  [E]         （每个 expert 一个 scalar）
    per-channel: [E, 2N/K]   （w1 为 [E, 2N]，w2 为 [E, K]）
    per-block:   [E, 2N/block_n, K/block_k]  （DeepGEMM block[128×128]）
```

### B3. CUTLASS FP8 内核接口（`cutlass_moe.py`）

#### Standard 格式（`CutlassExpertsFp8`）执行流程

```python
# run_cutlass_moe_fp8（cutlass_moe.py:53）完整流程
def run_cutlass_moe_fp8(output, hidden_states, w1, w2, topk_ids, activation,
                        global_num_experts, expert_map,
                        w1_scale, w2_scale, a1q_scale, a2_scale,
                        ab_strides1, ab_strides2, c_strides1, c_strides2,
                        workspace13, workspace2, expert_num_tokens,
                        out_dtype, per_act_token, per_out_ch,
                        use_batched_format, topk_weights):

    if not use_batched_format:
        # Standard 格式：先 permute，再 CUTLASS grouped GEMM
        # 1. 按 expert 排序 token（permute）
        a1q_perm, a1q_scale_perm, expert_first_token_offset, inv_perm, _ = moe_permute(
            hidden_states, topk_ids, a1q_scale, ...)
        # 2. 可选 swap_ab（M<=64 时转置 A/B 以提升利用率）
        swap_ab = a1q.size(0) <= 64
        # 3. 计算 CUTLASS problem sizes（每个 expert 的 M 数）
        ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
            expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, swap_ab)
    else:
        # Batched 格式：expert_x 已是 [E, padded_M, K]，直接用
        ops.get_cutlass_batched_moe_mm_data(...)

    # GEMM1: a1q × w1 → mm1_out [M*topk, 2N]
    ops.cutlass_moe_mm(mm1_out, a1q, w1, a1q_scale, w1_scale,
                       expert_offsets, problem_sizes1,
                       ab_strides1, ab_strides1, c_strides1,
                       per_act_token, per_out_ch)

    # Activation（SiLU/GELU/...）
    apply_moe_activation(activation, act_out, mm1_out)

    # 对 activation 输出重新量化（FP8 per-token dynamic）
    a2q, a2q_scale = ops.scaled_fp8_quant(act_out, a2_scale,
                         use_per_token_if_dynamic=per_act_token, output=quant_out)

    # GEMM2: a2q × w2 → mm2_out [M*topk, K]
    ops.cutlass_moe_mm(mm2_out, a2q, w2, a2q_scale, w2_scale,
                       expert_offsets, problem_sizes2,
                       ab_strides2, ab_strides2, c_strides2,
                       per_act_token, per_out_ch)

    # Unpermute + 加权 reduce（仅 Standard 格式）
    moe_unpermute(out=output, permuted_hidden_states=mm2_out,
                  topk_weights=topk_weights, inv_permuted_idx=inv_perm, ...)
```

**CUTLASS strides 初始化（`__init__` 中一次性预分配，`cutlass_moe.py:267`）：**
```python
E = moe_config.num_local_experts
N = moe_config.intermediate_size_per_partition   # = w1.shape[1] // 2
K = moe_config.hidden_dim

ab_strides1_c_strides2 = torch.full((E,), K,   dtype=torch.int64)  # w1 stride-K（也是 mm2 输出 stride）
ab_strides2            = torch.full((E,), N,   dtype=torch.int64)  # w2 stride-N
c_strides1             = torch.full((E,), 2*N, dtype=torch.int64)  # mm1 输出 stride（2*N）
# 注：ab_strides1 和 c_strides2 值相同（都 = K），共用同一 tensor 节省内存
```

**支持的量化组合（`_supports_quant_scheme`）：**

| weight quant | activation quant | `per_act_token` | `per_out_ch` |
|---|---|---|---|
| FP8 per-channel static | FP8 per-token dynamic | True | True |
| FP8 per-tensor static | FP8 per-tensor dynamic | False | False |
| FP8 per-tensor static | FP8 per-tensor static | False | False |

### B4. DeepGEMM FP8 内核（`deep_gemm_moe.py`）

DeepGEMM 是 DeepSeek 开源的针对 H100 Tensor Core 深度优化的 FP8 GEMM 库。

#### 内核使用的格式约定
- 权重必须是 `torch.float8_e4m3fn` 且**行优先（N-T layout）**
- Block quantization shape 必须是 `[128, 128]`（行 128 × 列 128）
- token 按 expert 排列成**连续 layout**（`deepgemm_moe_permute` 返回的格式）

#### `DeepGemmExperts.apply()` 流程（`deep_gemm_moe.py:242`）

```python
def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, ...):
    # 1. Permute：按 expert 排序 token（deepgemm_moe_permute）
    #    返回 a1q[M_sum, K]（连续 layout）+ expert_ids（每个对齐块的 expert）+ inv_perm
    a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(
        hidden_states, topk_ids, a1q_scale, block_shape, expert_tokens_meta, ...)

    # 2. GEMM1（FP8 grouped GEMM，contiguous layout）
    #    m_grouped_fp8_gemm_nt_contiguous 是 DeepGEMM 核心接口
    m_grouped_fp8_gemm_nt_contiguous(
        (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids)
    #    expert_ids 提供每个 block 的 expert 信息（128×128 分块）

    # 3. Activation + FP8 requant（_act_mul_quant）
    #    三条路径由 DeepGemmQuantScaleFMT 决定：
    #    UE8M0 + SiLU → fused_silu_mul_fp8_quant_packed
    #    UE8M0 + other → separate act + per_token_group_quant_fp8_packed_for_deepgemm
    #    Hopper FLOAT32_CEIL_UE8M0 + SiLU → silu_mul_per_token_group_quant_fp8_colmajor
    a2q, a2q_scale = self._act_mul_quant(input=mm1_out.view(-1, N), ...)

    # 4. GEMM2（FP8 grouped GEMM，contiguous layout）
    m_grouped_fp8_gemm_nt_contiguous(
        (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids)

    # 5. Unpermute + 加权 reduce（deepgemm_unpermute_and_reduce）
    deepgemm_unpermute_and_reduce(a=mm2_out, topk_ids=topk_ids,
                                  topk_weights=topk_weights,
                                  inv_perm=inv_perm, expert_map=expert_map, output=output)
```

**`_valid_deep_gemm()` 前置检查（`deep_gemm_moe.py:54`）：**
```python
def _valid_deep_gemm(hidden_states, w1, w2, ...):
    if not has_deep_gemm():   return False, "deep_gemm not installed"
    align = get_mk_alignment_for_contiguous_layout()   # = 128
    # M >= align（至少 128 tokens），N/K % align == 0，N > 512
    if M < align:             return False, f"M={M} < {align}"
    if N % align != 0:        return False, f"N={N} not aligned"
    if K % align != 0:        return False, f"K={K} not aligned"
    if N <= 512:              return False, f"N={N} <= 512"
    if w1.dtype != float8_e4m3fn: return False, "weights not fp8"
    if not all(t.is_contiguous() for t in [hidden_states, w1, w2]): ...
    return True, ""
```

### B5. Oracle 后端选择机制（细化）

#### 完整优先级（CUDA 平台）

```
── 无量化（BF16/FP16）──────────────────────────────────────────────────────
  FlashInfer TRTLLM（优先，除非 dp_size > 1）
  → FlashInfer CUTLASS（dp_size > 1 时降为最低优先）
  → AITER（ROCm）
  → Triton
  → Batched Triton

── FP8 量化 ─────────────────────────────────────────────────────────────────
  FlashInfer TRTLLM（Hopper block-FP8 时降为 Triton 优先）
  → FlashInfer CUTLASS（EP 场景 + block-FP8 时优先）
  → DeepGEMM（SM80+，block_shape=[128,128]）
  → Batched DeepGEMM（BatchedExperts + block-FP8）
  → Marlin（测试 flag VLLM_TEST_FORCE_FP8_MARLIN）
  → vLLM CUTLASS（通用 FP8 W8A8）
  → Batched vLLM CUTLASS
  → Triton（兜底，支持最广泛量化格式）
  → Batched Triton

── NVFP4 ────────────────────────────────────────────────────────────────────
  FlashInfer TrtLLM NVFP4 → FlashInfer CuteDSL → Marlin → Triton emulation

── MXFP4 ────────────────────────────────────────────────────────────────────
  DeepGEMM FP4（SM100+/Blackwell）→ FlashInfer TRTLLM/CUTLASS MXFP4
  → Marlin → Triton emulation

── INT8 W8A8 ────────────────────────────────────────────────────────────────
  Triton（唯一支持）
```

**环境变量覆盖（可强制指定后端）：**
| 环境变量 | 效果 |
|---------|------|
| `VLLM_USE_FLASHINFER_MOE_FP16=1` | 强制无量化使用 FlashInfer |
| `VLLM_USE_FLASHINFER_MOE_FP8=1` | 强制 FP8 使用 FlashInfer |
| `VLLM_USE_DEEP_GEMM=1` | 强制 FP8 使用 DeepGEMM（跳过 FlashInfer）|
| `VLLM_TEST_FORCE_FP8_MARLIN=1` | 强制 FP8 使用 Marlin（测试用）|
| `VLLM_ROCM_USE_AITER=1` | ROCm 平台强制 AITER |
| `VLLM_DEEPEPLL_NVFP4_DISPATCH=1` | LL 路径启用 NVFP4 dispatch |

#### `is_supported_config` 检查流程（每个后端类的静态方法）

```python
@classmethod
def is_supported_config(cls, moe_config, moe_parallel_config,
                         quant_config, activation, activation_format
                         ) -> tuple[bool, str]:
    # 依次检查，任一失败返回 (False, reason_string)
    ok, reason = cls._supports_current_device()          # CUDA/ROCm/XPU 平台
    ok, reason = cls._supports_quant_scheme(w_key, a_key) # 量化格式匹配
    ok, reason = cls._supports_activation(activation)     # silu/gelu/swiglu 等
    ok, reason = cls._supports_parallel_config(config)    # EP/TP/DP 约束
    ok, reason = cls._supports_routing_method(...)        # 路由方式（topk/grouped）
    ok, reason = cls._supports_shape(hidden_dim)          # 形状限制
    # activation_format 一致性（Standard/BatchedExperts）
    if activation_format != cls.activation_format():
        return False, f"activation format mismatch"
    return True, ""
```

#### 量化格式 → 可用后端（移植参考）

| 量化格式（weight_key, act_key） | 可用后端 |
|---------|---------|
| `(None, None)` BF16/FP16 无量化 | FlashInfer TRTLLM/CUTLASS, Triton, AITER(ROCm) |
| `(fp8_static_channel, fp8_dynamic_token)` | CUTLASS FP8, DeepGEMM, FlashInfer, Triton |
| `(fp8_static_128_block, fp8_dynamic_128_sym)` | DeepGEMM（首选）, CUTLASS FP8, Triton |
| `(nvfp4, nvfp4_w4a4)` | FlashInfer TrtLLM/CuteDSL, Marlin, Triton emulation |
| `(mxfp4, mxfp4_w4a4)` | DeepGEMM FP4(SM100+), FlashInfer, Marlin, Triton emulation |
| `(int8_static, int8_dynamic)` | Triton 仅 |
| `(int4_packed, fp8_dynamic)` | CUTLASS W4A8, Triton |
| `(wna16_int4, None)` | Marlin, Triton |

---

## 方向 C：EP/TP/DP 并行策略

### C1. 核心配置结构

#### `FusedMoEParallelConfig`（`config.py`）

```python
@dataclass
class FusedMoEParallelConfig:
    # 并行规模
    tp_size:  int     # Tensor Parallel size（权重按列/行切分）
    ep_size:  int     # Expert Parallel size（expert 按 rank 分配）
    dp_size:  int     # Data Parallel size（token 按 rank 分配）
    pcp_size: int     # Pipeline-CP（Context Parallel）size
    sp_size:  int     # Sequence Parallel size（Ulysses 式序列并行）

    # 各维度当前 rank
    tp_rank: int
    ep_rank: int
    dp_rank: int
    pcp_rank: int

    # 配置开关
    use_ep: bool           # 是否启用 Expert Parallel
    all2all_backend: str   # "allgather_reducescatter" | "deepep_high_throughput"
                           # | "deepep_low_latency" | "nixl_ep" | ...
    enable_eplb: bool      # 是否启用动态专家负载均衡（EPLB）

# ── 重要派生属性（均为 @property）────────────────────────────────────────
@property
def use_all2all_kernels(self):
    return self.dp_size > 1 and self.use_ep
    # True = 需要 AllToAll 通信（token 跨 rank 分发给 expert）

@property
def use_deepep_ht_kernels(self):
    return self.use_all2all_kernels and self.all2all_backend == "deepep_high_throughput"

@property
def use_deepep_ll_kernels(self):
    return self.use_all2all_kernels and self.all2all_backend == "deepep_low_latency"

@property
def use_batched_activation_format(self):
    return self.use_deepep_ll_kernels or self.use_nixl_ep_kernels
    # True → FusedMoEActivationFormat.BatchedExperts（[E, max_M, K]）

@property
def needs_round_robin_routing_tables(self):
    return self.use_deepep_ll_kernels or self.use_nixl_ep_kernels
    # True → expert_map 使用 round_robin 策略（交织分配）
```

**并行维度间的关系：**
```
全局 world_size = tp_size × ep_size × dp_size × pcp_size
EP dispatch 参与 rank 数 = dp_size × ep_size  （num_dispatchers）
每 rank 的 local_num_experts = global_num_experts // ep_size（± 1，linear 策略）
```

### C2. `determine_expert_map`（EP 分配核心）

```python
def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: str = "linear",  # "linear" | "round_robin"
    num_fused_shared_experts: int = 0,           # shared expert 不参与 EP 分配
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    # 返回 (local_num_experts, expert_map, expert_mask)
    # expert_map: shape=[global_num_experts], dtype=int32，-1 表示不属于本 rank
    # expert_mask: shape=[global_num_experts], dtype=bool（可选）
```

**`expert_map` 语义详解：**
- shape = `(global_num_experts,)`，dtype = `torch.int32`
- 值 `>= 0`：该 global expert 在本 rank 的 local 索引（0-based）
- 值 `-1`：该 expert 不属于本 rank（kernel 内部跳过此 expert 的 GEMM）
- 在 Triton 内核中通过 `expert_ids[block] == -1` 判断，命中则写零

**linear 策略（连续分段，默认）：**
```python
base  = global_num_experts // ep_size
extra = global_num_experts % ep_size   # 前 extra 个 rank 各多 1 个 expert
local_num = base + (1 if ep_rank < extra else 0)
start = ep_rank * base + min(ep_rank, extra)
expert_map[start : start + local_num] = torch.arange(local_num)
# 其余位置 = -1
```

示例（16 experts，3 ranks）：
| rank | local experts | global experts | expert_map（仅非 -1 位置）|
|------|---------------|----------------|--------------------------|
| 0 | 6 | 0-5 | `[0,1,2,3,4,5, -1,-1,-1,-1, -1,-1,-1,-1,-1,-1]` |
| 1 | 5 | 6-10 | `[-1,-1,-1,-1,-1,-1, 0,1,2,3,4, -1,-1,-1,-1,-1]` |
| 2 | 5 | 11-15 | `[-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1, 0,1,2,3,4]` |

**round_robin 策略（交织分配，DeepSeekV3 专用）：**
```python
# rank k 得到 global expert: k, k+ep_size, k+2*ep_size, ...
local_experts = torch.arange(ep_rank, global_num_experts, ep_size)
expert_map[local_experts] = torch.arange(len(local_experts))
```

示例（8 experts，2 ranks）：
| rank | local experts | global experts（交织）|
|------|---------------|----------------------|
| 0 | 4 | 0, 2, 4, 6（偶数） |
| 1 | 4 | 1, 3, 5, 7（奇数） |

**global↔physical ID 转换（DeepEP LL 特有）：**
```python
# global_to_physical[global_id] = physical_id（dispatch 时使用）
# physical_to_global[physical_id] = global_id（combine 时恢复）
# DeepEPLLPrepareAndFinalize.__init__ 中存储这两个映射 tensor
dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
combine_topk_ids  = self._map_global_to_physical_ids(topk_ids)  # finalize 中
```

**降级规则（round_robin → linear）：**

round_robin 仅在以下**全部**条件满足时保留，否则自动降级为 linear：
- `num_expert_group > 1`（分组路由，如 DeepSeek）
- `num_redundant_experts == 0`（无冗余 expert 备份）
- `not enable_eplb`（未启用动态负载均衡）
- all2all backend 支持 round_robin 路由表（当前仅 DeepEP LL 和 NIXL EP）

### C3. 单 Rank 路径（`no_dp_ep.py`）—— 移植第一步

```python
class MoEPrepareAndFinalizeNoDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """ep_size=1 或不启用 EP 时的路径，无任何通信操作"""

    @property
    def activation_format(self): return FusedMoEActivationFormat.Standard
    def output_is_reduced(self): return False      # 上层需做 TP AllReduce
    def max_num_tokens_per_rank(self): return None # 不限制 token 数
    def topk_indices_dtype(self): return None      # 使用默认 int32
    def num_dispatchers(self): return 1

    def prepare(self, a1, topk_weights, topk_ids, num_experts,
                expert_map, apply_router_weight_on_input, quant_config, ...):
        # 仅量化，不通信
        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1  # 仅支持 topk=1
            a1 = a1 * topk_weights.to(a1.dtype)  # 非 inplace（shared expert 重叠安全）

        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)
        # 返回 (a1q, a1q_scale, None, None, None)
        # expert_tokens_meta=None → kernel 内部从 topk_ids 推导
        # topk_ids/weights 不变（None = 使用原值）

    def finalize(self, output, fused_expert_output, topk_weights, topk_ids,
                 apply_router_weight_on_input, weight_and_reduce_impl):
        # 仅 topk 加权求和，不通信
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()  # 降级为标准实现
        weight_and_reduce_impl.apply(output=output, ...)
        # output_is_reduced=False → MoERunner 会补做 TP AllReduce

# _quantize_input 内部逻辑
def _quantize_input(a1, quant_config, defer_input_quant):
    if defer_input_quant:
        return a1, None   # AITER/FlashInfer 自己做量化

    input_sf = (quant_config.a1_gscale    # NVFP4 全局 scale
                if quant_config.use_nvfp4_w4a4
                else quant_config.a1_scale) # 其他格式的 activation scale

    a1q, a1q_scale = moe_kernel_quantize_input(
        a1, input_sf,
        quant_dtype=quant_config.quant_dtype,        # float8_e4m3fn / "nvfp4" / None
        per_act_token_quant=quant_config.per_act_token_quant,  # 动态 per-token
        block_shape=quant_config.block_shape,         # [128,128] 或 None
        is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,
    )
    return a1q, a1q_scale
```

### C4. DP+EP naive 路径（`naive_dp_ep.py`）

```python
class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """
    使用 PyTorch AllReduce/ReduceScatter 实现的 naive AllToAll。
    比 DeepEP 慢，但无硬件约束，是 EP 移植的第一步。
    """

    def prepare(self, a1, topk_weights, topk_ids, ...):
        if apply_router_weight_on_input:
            a1 = a1 * topk_weights.to(a1.dtype)

        # ★ 关键：NVFP4 scale 在 dispatch 前不做 swizzle（会改变 shape 破坏 AllToAll）
        a1q, scales = _quantize_and_setup_dispatch(a1, quant_config, defer_input_quant)
        # scales = None（静态量化/延迟量化）或 [a1q_scale]（动态量化随 token dispatch）

        # AllToAll dispatch：token 按路由结果分发到各 rank
        res = get_ep_group().dispatch(
            a1q, topk_weights, topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,   # scale 随 token 一起跨 rank 发送
        )
        # 返回 (dispatched_a1q, new_topk_weights, new_topk_ids[, scales])
        # new_topk_ids 已从 global expert 索引转换为 local expert 索引

        if scales is not None:
            # dispatch 后补做 NVFP4 swizzle（shape 已固定，安全）
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, None, topk_ids_local, topk_weights_dispatched

    def finalize(self, output, fused_expert_output, topk_weights, topk_ids, ...):
        # 1. 本地 topk 加权求和（先 reduce 再 combine）
        out = weight_and_reduce_impl.apply(output=None, fused_expert_output=...)

        # 2. AllToAll combine：将各 rank 计算结果汇总回原 token 位置
        output.copy_(get_ep_group().combine(out, is_sequence_parallel=...))
        # output_is_reduced=False → MoERunner 还需做 TP AllReduce
```

**NVFP4 延迟 swizzle 的原因：**
```python
# swizzle 会把 scales shape 从 [M, K/64] 变为 [M_padded, K_swizzled]（含 padding）
# 这导致 shape 与 hidden_states 不对齐，AllToAll 内核无法将它们配对传输
# 因此：dispatch 前用 is_fp4_scale_swizzled=False 量化，dispatch 后再 swizzle
```

### C5. MoERunner 主控流程（`moe_runner.py`）

```python
class MoERunner(MoERunnerInterface):
    """
    MoE 层前向调度器。注册两个 custom torch op 以绕过 torch.compile 的
    图捕获限制（MoE 层含有动态 shape 和状态，不能直接 trace）。
    """

    def forward(self, hidden_states, router_logits, ...):
        # 可选输入变换（如 fp8 量化）
        if self.routed_input_transform is not None:
            hidden_states = self.routed_input_transform(hidden_states)

        # 可选 gate 线性变换（router_logits → gate_output）
        if self.gate is not None:
            router_logits = self.gate(hidden_states)

        # 通过 custom op 进入 _forward_impl（CUDA Graph 安全）
        if self.shared_experts is not None:
            shared_out, fused_out = torch.ops.vllm.moe_forward_shared(
                hidden_states, router_logits, shared_experts_input, input_ids, layer_name)
        else:
            fused_out = torch.ops.vllm.moe_forward(
                hidden_states, router_logits, None, input_ids, layer_name)

        # 可选输出变换（如 dequant）
        if self.routed_output_transform is not None:
            fused_out = self.routed_output_transform(fused_out)

        return fused_out + shared_out  # shared expert 结果在此加回

    def _forward_impl(self, layer, hidden_states, router_logits, shared_experts_input, ...):
        # 1. Router：计算 topk_ids + topk_weights
        topk_weights, topk_ids = layer.router.select_experts(
            hidden_states, router_logits, ...)

        # 2. ZeroExpertRouter（可选：屏蔽不活跃 expert）
        if self.zero_expert_router is not None:
            topk_ids = self.zero_expert_router.apply(topk_ids, ...)

        # 3. 核心：调用 quant_method（→ FusedMoEKernel.apply()）
        output = layer.quant_method.apply(
            layer, x=hidden_states, router_logits=router_logits,
            top_k=self.moe_config.top_k,
            renormalize=self.moe_config.renormalize,
            topk_group=..., num_expert_group=...,
            global_num_experts=..., expert_map=layer.expert_map,
            shared_experts=layer.shared_experts,
        )

        # 4. 条件 AllReduce
        if (not is_sequence_parallel
            and (tp_size > 1 or ep_size > 1)
            and not output_is_reduced):
            output = tensor_model_parallel_all_reduce(output)

        return output
```

### C6. AllReduce 时机决策矩阵

```python
# MoERunner._maybe_reduce_final_output() 判断逻辑
_fused_output_is_reduced = layer.quant_method.kernel.prepare_finalize.output_is_reduced()

if (not is_sequence_parallel
    and (tp_size > 1 or ep_size > 1)
    and not _fused_output_is_reduced):
    output = tensor_model_parallel_all_reduce(output)
```

| 场景 | `output_is_reduced()` | 谁做 AllReduce |
|------|----------------------|----------------|
| 纯 TP（no_dp_ep） | `False` | MoERunner（TP AllReduce）|
| naive_dp_ep | `False` | MoERunner（TP AllReduce）|
| DeepEP HT | `True` | `buffer.combine()` 内置 |
| DeepEP LL | `True` | `buffer.low_latency_combine()` 内置 |
| SP（Sequence Parallel）| — | `_maybe_dispatch` 做 AllGather/RS |
| shared experts | — | `shared_out` 在 `forward()` 末单独 AllReduce |

**Sequence Parallel 特殊路径（`_maybe_dispatch/_maybe_combine`）：**
```python
# SP 时每个 rank 只看到部分序列
# naive_dp_ep.is_sequence_parallel=True 时：
# prepare: AllGather → 各 rank 获得完整序列 tokens
# finalize: ReduceScatter → 各 rank 只保留属于自己序列位置的结果
```

---

## 方向 D：高性能通信优化

### D1. DeepEP 高吞吐（HT）路径（`deepep_ht.py`）

**核心优化：** 异步 AllToAll（NVLink 优化内核）+ DBO（与另一批次的 GEMM 重叠执行）

#### 初始化与约束

```python
class DeepEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):

    def __init__(self, buffer: deep_ep.Buffer, num_dispatchers, dp_size, rank_expert_offset):
        self.buffer = buffer
        self.handles = [None, None]   # 每个微批次一个 handle，避免 DBO 竞争
        # available_rank_configs = [2, 4, 8, 16, 24, 32, 64, 128, 144, 160]
        # 仅这些 rank 数有预编译 dispatch config；其他 rank 数用 config=None（通用路径）

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size, dtype):
        # DeepEP 传输原子粒度：32（warp）× 16 字节（int4）= 512 字节
        # 例：hidden_size=2880, dtype=bf16(2字节) → 2880×2=5760，round_up(5760,512)=6144
        #     → 实际使用 hidden_size=3072（6144 // 2）
        xfer_atom_size = 512
        return round_up(hidden_size * dtype.itemsize, xfer_atom_size) // dtype.itemsize

    def topk_indices_dtype(self): return torch.int64   # DeepEP 要求 int64
    def output_is_reduced(self): return True           # combine 内置 AllReduce
    def activation_format(self): return Standard
    def max_num_tokens_per_rank(self): return None     # 动态 shape（不兼容 CUDA Graph）
```

#### `_do_dispatch()` 完整流程（`deepep_ht.py:97`）

```python
def _do_dispatch(self, tokens, token_scales, rank_topk_ids, rank_topk_weights,
                 num_experts, a1_scale, quant_config, defer_input_quant):
    has_scales = token_scales is not None

    # ── Step 1: 在 compute stream 上录制事件（DBO 通信/计算边界）──────────
    previous_event = dbo_get_previous_event(self.buffer.capture)
    # buffer.capture 在 compute_stream 上录制 DeepEP event
    # 必须在 yield 之前调用，确保 event 仅覆盖本批次的 compute 工作

    # ── Step 2: CPU yield，让另一批次排队 compute 工作 ────────────────────
    # CPU yield 前切到 comm stream，GPU compute → comm 有序性由 event 保证
    dbo_yield_and_switch_from_compute_to_comm()
    # 此后 CPU 在 comm stream 上排队指令；另一批次 CPU 则排队其 GEMM 指令

    # ── Step 3: CPU blocking —— 获取 dispatch layout（AllToAll 元数据）─────
    # get_dispatch_layout 是 CPU 密集型操作（计算各 rank 的 token 分配）
    # 在 comm stream 上执行，不阻塞另一批次的 compute stream
    (num_tokens_per_rank, num_tokens_per_rdma_rank,
     dispatch_expert_num_tokens, is_token_in_rank, event) = \
        self.buffer.get_dispatch_layout(
            topk_idx=rank_topk_ids,
            num_experts=num_experts,
            previous_event=previous_event,  # 等待上一批次 compute 完成
            async_finish=False,
            allocate_on_comm_stream=False,
        )

    # ── Step 4: 发出 AllToAll dispatch（NVLink 优化内核）─────────────────
    token_data = (tokens, token_scales) if has_scales else tokens
    (dispatched_token_data, expert_topk_ids, expert_topk_weights,
     expert_num_tokens_list, handle, event) = self.buffer.dispatch(
        x=token_data,
        handle=None,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=dispatch_expert_num_tokens,
        topk_idx=rank_topk_ids,
        topk_weights=rank_topk_weights,
        expert_alignment=1,              # token 数不做额外对齐（动态 shape）
        config=self._get_dispatch_config(),  # None 或预编译 config
        previous_event=previous_event,
        async_finish=self.async_prepare and not dbo_enabled(),  # 非 DBO 时异步
        allocate_on_comm_stream=False,
    )
    # handle 保存 dispatch 状态，combine 时需要
    self.handles[dbo_current_ubatch_id()] = handle

    # ── Step 5: 切回 compute stream（等待 comm 完成）─────────────────────
    dbo_switch_to_compute_sync()
    # = _signal_comm_done() + update_stream(compute) + _wait_comm_done()
    # compute stream 等待 comm stream 的 AllToAll 完成事件

    # 返回 receiver 闭包（延迟执行量化后处理）
    return lambda: self._receiver(event, has_scales, dispatched_token_data,
                                  expert_topk_ids, num_experts,
                                  expert_num_tokens_list, expert_topk_weights,
                                  a1_scale, quant_config, defer_input_quant)
```

#### `_receiver()` 量化后处理（`deepep_ht.py:183`）

```python
def _receiver(self, event, has_scales, token_data, expert_topk_ids, num_experts,
              expert_num_tokens_list, expert_topk_weights, a1_scale, quant_config, ...):

    if event.event is not None:
        event.current_stream_wait()   # 等待异步 AllToAll 完成

    if has_scales:
        expert_x, expert_x_scale = token_data
    else:
        expert_x, expert_x_scale = token_data, None

    # expert_topk_ids 从 local 索引 → global 索引（+rank_expert_offset）
    # -1 位置替换为安全的哑 expert id（kernel 通过 expert_map 过滤）
    expert_topk_ids = torch.where(
        expert_topk_ids == -1,
        num_experts - 1 if self.rank_expert_offset == 0 else 0,
        expert_topk_ids + self.rank_expert_offset,
    )

    # CPU list → GPU tensor（含 CPU 副本，供 DeepGEMM 使用）
    expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
        expert_num_tokens_list, device=expert_x.device)

    # 非 block 量化：dispatch 在 BF16，收到后再量化（DeepEP 仅支持 block scale dispatch）
    if not quant_config.is_block_quantized and not defer_input_quant:
        expert_x, expert_x_scale = moe_kernel_quantize_input(
            expert_x, a1_scale, quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=False,   # 注：HT 路径不支持 per-token scale dispatch
            block_shape=quant_config.block_shape, ...)

    return (expert_x, expert_x_scale, expert_tokens_meta,
            expert_topk_ids, expert_topk_weights)
```

#### `_finalize()` combine 流程（`deepep_ht.py:336`）

```python
def _finalize(self, output, fused_expert_output, topk_weights, topk_ids,
              apply_router_weight_on_input, weight_and_reduce_impl, do_async):

    handle = self.handles[dbo_current_ubatch_id()]

    # 1. 本地 topk 加权求和（在 compute stream）
    if fused_expert_output.numel() != 0:  # M=0 时跳过（本 rank 无 token 分配）
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        fused_expert_output = weight_and_reduce_impl.apply(
            output=None, fused_expert_output=fused_expert_output, ...)

    # 2. DBO yield：切到 comm stream
    previous_event = dbo_get_previous_event(self.buffer.capture)
    dbo_yield_and_switch_from_compute_to_comm()

    # ★ 关键约束：combine 仅支持 BF16 输入
    assert fused_expert_output.dtype == torch.bfloat16

    # 3. AllToAll combine（NVLink，内置 AllReduce）
    combined_x, _, event = self.buffer.combine(
        x=fused_expert_output,
        handle=handle,
        topk_weights=None,             # 权重已在本地乘过
        config=self._get_combine_config(),
        previous_event=previous_event,
        async_finish=do_async and not dbo_enabled(),
        allocate_on_comm_stream=False,
    )

    # 4. 切回 compute stream
    dbo_switch_to_compute()

    if do_async:
        def _receiver():
            if event.event is not None: event.current_stream_wait()
            dbo_switch_to_comm()
            output.copy_(combined_x, non_blocking=True)
            dbo_yield_and_switch_from_comm_to_compute()
        return _receiver
    else:
        output.copy_(combined_x, non_blocking=True)
        return None
```

### D2. DeepEP 低延迟（LL）路径（`deepep_ll.py`）

**核心特性：** 固定 shape 输出（BatchedExperts 格式）→ CUDA Graph 兼容；更低通信延迟

#### 约束与配置

```python
class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):

    # 编译时硬编码的 hidden_size 列表（LL kernel 仅为这些值编译）
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

    def __init__(self, buffer, max_tokens_per_rank, num_dispatchers, use_fp8_dispatch=False,
                 global_to_physical=None, physical_to_global=None, local_expert_global_ids=None):
        self.max_tokens_per_rank = max_tokens_per_rank  # 预分配固定值（CUDA Graph 关键！）
        self.use_fp8_dispatch = use_fp8_dispatch  # True = dispatch FP8 量化输入
        self.use_ue8m0_dispatch = False           # post_init_setup 后可能改为 True
        # global_to_physical/physical_to_global：round_robin 时 global↔physical expert ID 映射

    def post_init_setup(self, fused_experts):
        # 根据 expert 后端能否处理 packed UE8M0 scale 决定是否在 dispatch 时使用
        if fused_experts.supports_packed_ue8m0_act_scales() and self.use_fp8_dispatch:
            self.use_ue8m0_dispatch = True   # DeepGEMM 等 UE8M0 内核的优化路径

    @property
    def activation_format(self): return BatchedExperts   # [E, max_M, K]
    def output_is_reduced(self): return True
    def max_num_tokens_per_rank(self): return self.max_tokens_per_rank
    def topk_indices_dtype(self): return torch.int64
```

#### `prepare_async()` 流程（`deepep_ll.py:230`）

```python
def prepare_async(self, a1, topk_weights, topk_ids, num_experts, ...):
    hidden_size = a1.size(1)
    assert hidden_size in self.SUPPORTED_HIDDEN_SIZES  # 必须是 8 个值之一

    a2a_idx = dbo_current_ubatch_id()

    if apply_router_weight_on_input:
        a1 = a1 * topk_weights.to(a1.dtype)

    # global expert ID → physical expert ID（round_robin 映射）
    dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)

    # low_latency_dispatch 参数：
    (expert_x,          # [E, max_tokens_per_rank, hidden]  ← BatchedExperts 固定 shape！
     expert_num_tokens, # [E] GPU tensor，实际 token 数（≤ max_tokens_per_rank）
     handle,            # 保存 dispatch 状态，combine 时使用
     _,
     hook               # 轻量 recv_hook（等待接收完成的最小操作）
    ) = self.buffer.low_latency_dispatch(
        a1,
        dispatch_topk_ids,
        self.max_tokens_per_rank,    # 固定最大 token 数，输出 shape 不变
        num_experts,
        use_fp8=self.use_fp8_dispatch,
        round_scale=self.use_ue8m0_dispatch,
        use_ue8m0=self.use_ue8m0_dispatch,
        async_finish=False,
        return_recv_hook=True,       # 返回 hook（注册给下一批次延迟执行）
    )
    self.handles[a2a_idx] = handle

    # 返回 (hook, receiver) 二元组（与 HT 路径不同！）
    return (
        hook,  # 由 DBO 注册到【下一批次】延迟执行
        lambda: self._receiver(expert_x, expert_num_tokens, quant_config.a1_scale,
                               a1.dtype, quant_config)
    )
```

#### `_receiver()` 量化（`deepep_ll.py:346`）

```python
def _receiver(self, expert_x, expert_num_tokens, a1_scale, a1_dtype, quant_config):
    # _do_quant 内部的三条路径：
    # 1. use_fp8_dispatch + block_k==128：DeepEP 内核已量化，直接 unpack
    # 2. use_fp8_dispatch + block_k≠128：先 dequant → BF16，再重新量化
    # 3. 其他（BF16 dispatch）：对 [E, max_M, K] reshape 成 [E*max_M, K] 后量化
    expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)

    expert_tokens_meta = mk.ExpertTokensMetadata(
        expert_num_tokens=expert_num_tokens,  # GPU [E]
        expert_num_tokens_cpu=None,           # LL 路径不提供 CPU 副本
    )
    return expert_x, expert_x_scale, expert_tokens_meta, None, None
    # topk_ids/weights 返回 None（combine kernel 内部直接处理）
```

#### `_finalize()` combine（`deepep_ll.py:390`）

```python
def _finalize(self, output, fused_expert_output, topk_weights, topk_ids, ...):
    # ★ 关键：LL 路径的权重归约必须由 combine kernel 完成（委托模式）
    assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate)

    handle = self.handles[dbo_current_ubatch_id()]

    # 处理 apply_router_weight_on_input：权重已乘则 combine 权重全设为 1
    combine_topk_weights = (torch.ones_like(topk_weights)
                            if apply_router_weight_on_input else topk_weights)

    # physical expert ID（combine 同样使用 physical ID）
    combine_topk_ids = self._map_global_to_physical_ids(topk_ids)

    # 先执行本批次注册的 recv_hook（上一批次 dispatch 的接收完成动作）
    dbo_maybe_run_recv_hook()

    # low_latency_combine 内置：topk weighted sum + AllReduce（写入 out）
    _, _, recv_hook = self.buffer.low_latency_combine(
        fused_expert_output,   # [E, max_M, K] BatchedExperts 格式
        combine_topk_ids,
        combine_topk_weights,
        handle,
        async_finish=False,
        zero_copy=False,
        return_recv_hook=dbo_enabled() or do_async,
        out=output,            # in-place 写入最终结果
    )

    return recv_hook, lambda: None  # (hook 给下一批次, noop receiver)
```

**与 HT 路径关键对比：**

| 特性 | DeepEP HT | DeepEP LL |
|------|-----------|-----------|
| hidden_size 约束 | 512 字节对齐（动态 round up） | 仅 8 个固定值 |
| 输出格式 | Standard `[M, K]`（M 动态）| BatchedExperts `[E, max_M, K]`（固定）|
| CUDA Graph | ❌ 不兼容（动态 shape）| ✅ 兼容（固定 shape）|
| dispatch 预处理 | `get_dispatch_layout()` CPU blocking | 直接 `low_latency_dispatch`（更低延迟）|
| expert ID 类型 | global ID + `rank_expert_offset` | physical ID（`global_to_physical` 映射）|
| 量化 dispatch | block-FP8 = 量化前 dispatch；其他 = BF16 dispatch 后量化 | `use_fp8_dispatch` 参数控制 |
| finalize 权重归约 | 本地 `TopKWeightAndReduceContiguous` + combine AllReduce | `low_latency_combine` 全包 |
| DBO hook 返回 | `prepare_async` 返回单 receiver | 返回 `(hook, receiver)` 二元组 |
| 支持 per-token scale dispatch | ❌（HT 路径 per-token quant 只能 dispatch 后做）| ❌（has_per_token_scales 时 assert 失败）|

### D3. DBO（双批次重叠）机制（`v1/worker/ubatching.py`）

**核心思想：** 批次 A 做 EP AllToAll 通信时，批次 B 同步做 Expert GEMM 计算，实现通信/计算流水线。

#### `UBatchContext` 完整字段（`ubatching.py:20`）

```python
class UBatchContext:
    id: int                              # 微批次 ID（0 或 1）
    comm_stream: torch.cuda.Stream       # AllToAll 通信专用 CUDA stream
    compute_stream: torch.cuda.Stream    # Expert GEMM 计算专用 CUDA stream
    forward_context: ForwardContext      # 每批次独立的 forward context（layer 状态等）
    ready_barrier: threading.Barrier     # 两批次线程启动时的一次性同步栅栏
    cpu_wait_event: threading.Event      # 本批次 CPU 等待事件（调用 wait() 挂起）
    cpu_signal_event: threading.Event    # 另一批次的 wait_event（本批次调用 set() 唤醒对方）
    gpu_comm_done_event: torch.Event     # GPU 事件：comm stream 完成标志
    gpu_compute_done_event: torch.Event  # GPU 事件：compute stream 完成标志
    current_stream: torch.cuda.Stream    # 当前活跃 stream（随时可查）
    recv_hook: Callable | None           # 注册的延迟接收 hook（跨批次传递！）
    schedule: str                        # 调度策略（"default"）
```

**`cpu_wait_event` / `cpu_signal_event` 的交叉设计（`make_ubatch_contexts` 第 234 行）：**
```python
# 两个 Event 对象，每批次各持有一对（wait/signal 互换）
cpu_events = [threading.Event() for _ in range(2)]
# 批次 0: cpu_wait_event=cpu_events[0], cpu_signal_event=cpu_events[1]
# 批次 1: cpu_wait_event=cpu_events[1], cpu_signal_event=cpu_events[0]
# 批次 0 set(events[1]) 唤醒批次 1；批次 1 set(events[0]) 唤醒批次 0
```

#### `UBatchContext` 六个核心方法

```python
def yield_(self):
    """CPU 让出执行权给另一批次（保持当前 stream 不变）"""
    self.current_stream = current_stream()
    self._cpu_yield()              # set(signal) → wait(self_wait) → restore context
    self.update_stream(self.current_stream)

def yield_and_switch_from_compute_to_comm(self):
    """让出 CPU + 从 compute stream 切到 comm stream（dispatch 前调用）"""
    assert current_stream() == self.compute_stream
    self._signal_compute_done()    # 在 compute_stream 录制 gpu_compute_done_event
    self._cpu_yield()              # CPU yield → 另一批次运行
    self.update_stream(self.comm_stream)    # 恢复后切到 comm stream
    self._wait_compute_done()      # comm_stream.wait(gpu_compute_done_event)

def yield_and_switch_from_comm_to_compute(self):
    """让出 CPU + 从 comm stream 切到 compute stream（combine 后调用）"""
    assert current_stream() == self.comm_stream
    self._signal_comm_done()       # 在 comm_stream 录制 gpu_comm_done_event
    self._cpu_yield()
    self.update_stream(self.compute_stream)
    self._wait_comm_done()         # compute_stream.wait(gpu_comm_done_event)

def switch_to_comm_sync(self):
    """同步切到 comm（非 yield，不让出 CPU）"""
    self._signal_compute_done()
    self.update_stream(self.comm_stream)
    self._wait_compute_done()

def switch_to_compute_sync(self):
    """同步切回 compute（非 yield）"""
    self._signal_comm_done()
    self.update_stream(self.compute_stream)
    self._wait_comm_done()

def maybe_run_recv_hook(self):
    if self.recv_hook is not None:
        self.recv_hook(); self.recv_hook = None
```

#### 全局 API 函数（线程安全，通过 thread-local dict 路由）

```python
# 全局字典：thread_id → ubatch_id（判断是否在 DBO 模式）
_THREAD_ID_TO_CONTEXT: dict = {}
_CURRENT_CONTEXTS: list[UBatchContext | None] = []

dbo_enabled()                          # len(_THREAD_ID_TO_CONTEXT) > 0
dbo_current_ubatch_id()                # _THREAD_ID_TO_CONTEXT[thread_ident]
dbo_yield()                            # ctx.yield_()
dbo_yield_and_switch_from_compute_to_comm()  # ctx.yield_and_switch_from_compute_to_comm()
dbo_yield_and_switch_from_comm_to_compute()  # ctx.yield_and_switch_from_comm_to_compute()
dbo_switch_to_comm()                   # ctx.switch_to_comm()
dbo_switch_to_compute()                # ctx.switch_to_compute()
dbo_switch_to_comm_sync()              # ctx.switch_to_comm_sync()
dbo_switch_to_compute_sync()           # ctx.switch_to_compute_sync()
dbo_maybe_run_recv_hook()              # ctx.maybe_run_recv_hook()

def dbo_register_recv_hook(recv_hook):
    # ★ hook 注册给【下一批次】而非当前批次！
    ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
    next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
    next_ctx.recv_hook = recv_hook

def dbo_get_previous_event(func, *args, **kwargs):
    # 在本批次 compute_stream 上执行 func（如 buffer.capture）
    # 用于录制 DeepEP event，确保在正确 stream 上录制
    ctx = _CURRENT_CONTEXTS[dbo_current_ubatch_id()]
    with torch.cuda.stream(ctx.compute_stream):
        return func(*args, **kwargs)
```

#### 完整执行时序（以 DeepEP LL + DBO 为例）

```
时间轴 ─────────────────────────────────────────────────────────────────▶
                 T1            T2            T3             T4
批次 A CPU   [route+quant] [dispatch]   [sleeping]   [recv+GEMM]  [combine]
批次 B CPU   [sleeping]   [recv_hook]   [dispatch]   [sleeping]   [recv+GEMM]
                               ↑                         ↑
                          A dispatch                B dispatch
                         完成，A yield             完成，B yield
                         注册 hook→B              注册 hook→A

批次 A GPU   [GEMM...]    ──────── dispatch comm ──────── [expert GEMM] [combine comm]
批次 B GPU   [GEMM...]    ──── [expert GEMM] ──── dispatch comm ────────── [expert GEMM]
             ↑                  ↑                  ↑
          前一批次           B 在 A 通信           A 在 B 通信
          GEMM 计算         期间做 GEMM            期间做 GEMM
                                 （重叠！）              （重叠！）
```

**UBatchContext.__enter__ / __exit__ 生命周期：**
```python
def __enter__(self):
    _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
    _CURRENT_CONTEXTS[self.id] = self
    self.ready_barrier.wait()   # 等待两个批次线程都 ready（同步启动）
    self.cpu_wait_event.wait()  # 等待被唤醒（由另一批次或主线程 set）
    self._restore_context()     # 恢复 forward_context（moe_layer_index 等）
    self.update_stream(self.compute_stream)  # 默认从 compute stream 开始

def __exit__(self, ...):
    self.maybe_run_recv_hook()  # 执行可能还未执行的 recv_hook
    self.cpu_signal_event.set() # 唤醒另一批次（通知自己完成）
```

### D4. WorkspaceManager（`v1/worker/workspace.py`）

```python
class WorkspaceManager:
    """
    统一管理 MoE 中间缓冲区（workspace13, workspace2, fused_out）。
    DBO 时为每个微批次维护独立 buffer slot，确保两批次不共享内存。
    """

    def __init__(self, device, num_ubatches=None):
        self._device = device
        self._num_ubatches = num_ubatches or 1    # DBO 时 = 2
        # 每个 slot 是一个连续 uint8 大 buffer（惰性分配）
        self._current_workspaces: list[torch.Tensor | None] = \
            [None] * self._num_ubatches
        self._locked = False   # CUDA Graph capture 完成后 lock，禁止扩容

    def get_simultaneous(self, *shapes_and_dtypes: tuple[tuple, torch.dtype]
                         ) -> list[torch.Tensor]:
        """
        从同一个 uint8 buffer 中分配多个 tensor（256 字节对齐切分）。
        通过 dbo_current_ubatch_id() 自动选择对应批次的 buffer slot。
        """
        total_bytes = sum(
            round_up(prod(shape) * dtype.itemsize, 256)
            for shape, dtype in shapes_and_dtypes
        )
        buf = self._ensure_workspace_size(total_bytes)  # 惰性扩容

        tensors = []
        offset = 0
        for shape, dtype in shapes_and_dtypes:
            nbytes = prod(shape) * dtype.itemsize
            nbytes_aligned = round_up(nbytes, 256)
            tensors.append(buf[offset:offset + nbytes].view(dtype).reshape(shape))
            offset += nbytes_aligned
        return tensors

    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        slot_idx = dbo_current_ubatch_id()
        ws = self._current_workspaces[slot_idx]
        if ws is None or ws.nbytes < required_bytes:
            if self._locked:
                raise AssertionError(
                    f"WorkspaceManager locked: cannot grow workspace from "
                    f"{ws.nbytes if ws else 0} to {required_bytes} bytes"
                )  # CUDA Graph 捕获后不允许扩容（会导致 graph replay 地址失效）
            # 扩容：分配新的对齐 buffer，只影响当前 slot
            ws = torch.empty(required_bytes, dtype=torch.uint8, device=self._device)
            self._current_workspaces[slot_idx] = ws
        return ws

    def lock(self):
        """CUDA Graph warmup 完成后调用，冻结 buffer size"""
        self._locked = True

    def unlock(self):
        """Elastic EP 扩缩容时（expert 数量变化）需要重新分配，解锁后扩容"""
        self._locked = False
```

**DBO 独立性保证（关键设计）：**
```
DBO 场景（num_ubatches=2）：

批次 A 的 workspace slot = _current_workspaces[0]
批次 B 的 workspace slot = _current_workspaces[1]

当批次 A 在 GEMM 时（使用 slot[0]），批次 B 的 dispatch 不会动 slot[0]
当批次 B 需要扩容时，只扩 slot[1]，不影响 slot[0] 的地址

CUDA Graph 兼容：
- LL 路径在 warmup 阶段确定最大 workspace 大小
- lock() 后 graph 录制，replay 时 workspace 地址稳定
- HT 路径因 M 动态变化不兼容 CUDA Graph（不调用 lock）
```

---

## 移植路线图（优先级排序）

> **总体原则：** 优先验证功能正确性，再逐步叠加性能优化；每一阶段结束后务必通过对应的验证方法（见下节）再进入下一阶段。

### 第一阶段：最小可运行版本（单卡，无量化，无通信）

**目标：** 在单 GPU 上完整跑通 MoE 前向，与 PyTorch naive 实现对齐。

| 步骤 | 内容 | 关键接口 / 文件 | 注意事项 |
|------|------|----------------|---------|
| 1 | 实现 `FusedMoEConfig`：`hidden_size, intermediate_size, num_experts, top_k, dtype, ...` | `config.py:FusedMoEConfig` | dtype 影响 workspace 内存计算，需准确 |
| 2 | 实现 `FusedMoEParallelConfig`：`tp_size=1, ep_size=1, dp_size=1, use_ep=False` | `config.py:994` | 先固定为 1-rank 配置，后续扩展 |
| 3 | 实现 `determine_expert_map(ep_size=1, ep_rank=0, global_num_experts=E)` → `expert_map=[0,1,...,E-1]` | `layer.py:71` | 单 rank 时 expert_map 是 identity 映射，无 -1 |
| 4 | 实现 `ExpertTokensMetadata` 数据结构 + 从 topk_ids 统计 `expert_num_tokens` | `modular_kernel.py:97` | GPU/CPU 两份计数；DeepGEMM 需要 CPU 版本 |
| 5 | 实现 `FusedMoEPrepareAndFinalizeModular` 抽象接口（所有方法返回 `NotImplementedError`） | `modular_kernel.py:251` | 接口签名必须严格一致，为后续实现打基础 |
| 6 | 实现 `NoDpEpPrepareAndFinalize`（仅量化，无通信） | `no_dp_ep.py` | `prepare()` 返回 5-元组；`finalize()` 调 `weight_and_reduce_impl.apply()` |
| 7 | 实现 `FusedMoEExpertsModular` 抽象接口 + `workspace_shapes()` 计算逻辑 | `modular_kernel.py:763` | workspace shape 因后端不同；先用 Triton 公式 |
| 8 | 实现 `TritonExperts`（或框架等效的 grouped GEMM）：`moe_align_block_size` → `invoke_fused_moe_triton_kernel` | `fused_moe.py:TritonExperts` | 需要 CUDA kernel `moe_align_block_size`（C++）|
| 9 | 实现 `TopKWeightAndReduce.Contiguous`：`fused_expert_output [M, topk, K]` → `output [M, K]` | `topk_weight_and_reduce.py` | gated 激活时输出 dim=K；注意 apply_router_weight_on_input 标志 |
| 10 | 实现 `WorkspaceManager`（单批次，基础版）：`get_simultaneous()` + 惰性扩容 | `workspace.py` | 256 字节对齐；先不实现 lock/unlock |
| 11 | 实现 `FusedMoEKernel.__init__` 中的类型检查 + `FusedMoEKernelModularImpl.apply()` 主流程 | `modular_kernel.py:FusedMoEKernelModularImpl` | 依赖步骤 5-10 全部就绪 |
| 12 | 实现 `MoERunner.forward()` 顶层主控（调用路由 → kernel → 可选 AllReduce） | `moe_runner.py` | 先跳过 AllReduce；`_maybe_reduce_final_output()` 先做空实现 |

**验收标准：**
```python
# 与 naive 实现对比（无量化，BF16）
naive_out = sum(topk_weights[:, i] * linear(x, w2[topk_ids[:, i]]
               @ F.silu(linear(x, w1[topk_ids[:, i]])[:, :N//2])
               * linear(x, w1[topk_ids[:, i]])[:, N//2:])
               for i in range(topk))
assert torch.allclose(fused_out, naive_out, atol=1e-3, rtol=1e-3)
```

---

### 第二阶段：多卡并行（TP + EP + Router 完整化）

**目标：** 支持单机多卡 tensor parallel 和 expert parallel，完善路由实现。

| 步骤 | 内容 | 关键接口 / 文件 | 注意事项 |
|------|------|----------------|---------|
| 13 | 实现 `determine_expert_map(linear 策略)` 完整版：`base = E//ep_size`，余数分布 | `layer.py:71` | 注意 `-1` 值表示非本 rank；传给 kernel 内部跳过 |
| 14 | 实现 `NaiveDpEpPrepareAndFinalize.prepare()`：AllGather + AllToAll dispatch，`topk_ids` 转 local | `naive_dp_ep.py` | dispatch 结果 `expert_tokens_meta` 需从 AllToAll recv 统计 |
| 15 | 实现 `NaiveDpEpPrepareAndFinalize.finalize()`：本地 topk reduce + AllToAll combine | `naive_dp_ep.py` | combine 后 output 已包含来自其他 rank 的贡献 |
| 16 | 实现 `_maybe_reduce_final_output()`：根据 `output_is_reduced` 标志决定是否发 TP AllReduce | `moe_runner.py:357` | EP+TP 混合时，`output_is_reduced=False` 仍需 AllReduce |
| 17 | 实现 `FusedTopKRouter`（softmax/sigmoid normalization + topK 选择）| `router/fused_topk_router.py` | 注意 `norm_topk_prob` 参数对 topk_weights 的影响 |
| 18 | 实现 `GroupedTopKRouter`（DeepSeekV3 分组 TopK）：`num_expert_group` 约束 | `router/grouped_topk_router.py` | `group_topk` 阶段先做每 group 取 max，再全局 topK；顺序不可颠倒 |
| 19 | 实现 Oracle 后端选择基础框架：`get_priority_backends()` + 条件降级链 | `oracle/unquantized.py` | 先仅支持 BF16/FP16 无量化路径 |
| 20 | 完善 `WorkspaceManager.lock()/unlock()` + CUDA Graph 安全 | `workspace.py` | 与步骤 26（CUDA Graph）配合 |

**验收标准：**
```bash
# 2-GPU EP 正确性测试（参考 vLLM 测试）
torchrun --nproc-per-node=2 test_ep_moe.py --ep-size=2 --num-experts=8 --top-k=2
# 期望：EP 输出与单 GPU 参考实现在 atol=1e-2 范围内一致
```

---

### 第三阶段：量化内核（FP8 / NVFP4）

**目标：** 支持 FP8 W8A8 量化，大幅提升计算吞吐。

| 步骤 | 内容 | 关键接口 / 文件 | 注意事项 |
|------|------|----------------|---------|
| 21 | 实现 `FusedMoEQuantConfig.make()`：解析 `quant_type`，构建 a1_scale/a2_scale/block_shape | `config.py:FusedMoEQuantConfig` | per-channel vs per-token vs block 三种 scale 形状不同 |
| 22 | 实现 `moe_kernel_quantize_input()`：动态 FP8 量化（per-token absmax）| `utils.py` | 量化结果 shape 与原始 hidden_states 一致；scale shape 取决于 per_act_token |
| 23 | 实现 `NoDpEpPrepareAndFinalize` 的量化路径（`defer_input_quant` 分支）| `no_dp_ep.py` | `defer_input_quant=True` 时跳过 prepare 量化，由 experts 内部处理 |
| 24 | 实现 `CutlassExpertsFp8`（或框架对应 grouped FP8 GEMM）：stride tensor 预构建 + `cutlass_moe_fp8` 调用 | `experts/cutlass_moe.py` | stride tensor 形状 `[E]`，全部填相同值；需要 permute/unpermute 辅助 |
| 25 | 实现 `DeepGemmExperts`（H100+）：`_valid_deep_gemm()` 检查 + `m_grouped_fp8_gemm_nt_contiguous` | `experts/deep_gemm_moe.py` | 需要 N/K 可被 128 整除；block scale 格式需 uint8 pack |
| 26 | 完善 Oracle 选择逻辑：FP8 格式降级链（DeepGEMM → CUTLASS → Triton）| `oracle/fp8.py` | 验证各约束条件与目标框架 GPU 型号对应关系 |

**验收标准：**
```python
# FP8 与 BF16 参考实现误差
assert torch.allclose(fp8_out.float(), bf16_out.float(), atol=0.05)
# 吞吐比较（同等 batch_size=512, num_experts=256, top_k=8）
# 期望 FP8 比 BF16 快 ~1.5-2x（因 Tensor Core FP8 吞吐翻倍）
```

---

### 第四阶段：高性能通信优化（DeepEP + DBO）

**目标：** 实现 communication-computation overlap，最大化 GPU 利用率。

| 步骤 | 内容 | 关键接口 / 文件 | 注意事项 |
|------|------|----------------|---------|
| 27 | 实现 `UBatchContext`（DBO 核心）：CPU 事件同步 + GPU stream 切换 | `ubatching.py` | `cpu_wait_event` 与 `cpu_signal_event` 是相邻批次共享的（i 与 i+1 的关系）|
| 28 | 实现所有 `dbo_*` 全局 API：`dbo_yield`、`dbo_register_recv_hook` 等 | `ubatching.py:160+` | `recv_hook` 注册到**下一个**批次的 context（`(idx+1) % N`）|
| 29 | 实现 `WorkspaceManager` DBO 扩展：`_num_ubatches=2`，各批次独立 slot | `workspace.py` | slot 隔离是 DBO 正确性的关键，不可复用 |
| 30 | 实现 `DeepEPHTPrepareAndFinalize`：`_do_dispatch()` 中的 CPU yield + stream 切换 | `deepep_ht.py` | HT dispatch 需先调 `get_dispatch_layout()`（CPU blocking），时序关键 |
| 31 | 实现 `DeepEPHTPrepareAndFinalize.finalize_async()`：local reduce + `_do_combine()` | `deepep_ht.py` | HT combine 要求 BF16；`output_is_reduced=True` 需向上层传递 |
| 32 | 实现 `determine_expert_map(round_robin 策略)` + 降级逻辑 | `layer.py` | round_robin 条件：`num_expert_group > 1 and no_redundant and no_eplb and ll_or_nixl` |
| 33 | 实现 `DeepEPLLPrepareAndFinalize`：`low_latency_dispatch` + `low_latency_combine` | `deepep_ll.py` | LL 输出格式为 BatchedExperts；`TopKWeightAndReduceDelegate` 不做本地 reduce |
| 34 | 实现 `EPLBController`（动态专家负载均衡，可选）：监控 token 分布 + 动态迁移 expert | `eplb_utils.py` | EPLB 与 round_robin 互斥；启用 EPLB 时强制 linear 策略 |

**验收标准：**
```python
# DBO 正确性：关闭/开启 DBO 输出一致
assert torch.allclose(dbo_off_out, dbo_on_out, atol=1e-5)

# 吞吐提升验证（期望 DBO 可带来 10-30% 提升，取决于 compute/comm 比例）
# 使用 nsight systems 确认通信和计算有效重叠
```

---

### 依赖关系图（阶段间）

```
阶段1（基础单卡）
  └── 阶段2（多卡并行）
       ├── 阶段3（量化内核）
       │    └── 阶段4（高性能通信）
       └── 阶段4 的步骤 27-29（DBO 基础设施）可并行于阶段3
```

---

## 关键设计约束（移植注意事项）

### 内存布局与对齐约束

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| **workspace 256 字节对齐** | `get_simultaneous()` 每个 tensor 起始地址必须是 256 字节边界 | CUDA kernel 地址未对齐，性能下降或崩溃 |
| **workspace13 与 output 可复用** | `workspace13_shape` 与 `output_shape` 取 `max(nbytes)` 分配同一块内存 | 两者生命周期不重叠（output 在 finalize 写入），可安全复用 |
| **w1 shape 含 gated 因子 2** | `w1=[E, 2*N, K]`（gate proj + up proj fused）；w2 输入维度是 `N`（不是 `2N`）| GEMM2 K 维度对不齐，计算结果错误 |
| **DeepGEMM N/K 对齐** | N 和 K 必须被 128 整除；N > 512（确保 SM 有效利用率）| `_valid_deep_gemm()` 检查失败，自动降级到 CUTLASS/Triton |
| **CUTLASS stride tensor** | 需预构建 `[E]` 形状的 stride tensor，值全为同一常数（例如 `K` 或 `2N`）| grouped GEMM 每个 expert 起始地址计算错误 |

### 数据类型约束

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| **`topk_ids` dtype** | DeepEP LL 要求 `int64`；Triton/CUTLASS 通常用 `int32`；类型转换在 `topk_indices_dtype()` 中声明 | LL kernel 索引溢出或地址计算错误 |
| **HT combine 要求 BF16** | `fused_expert_output` 传入 `buffer.combine()` 前必须是 BF16；代码中有 `assert dtype == bfloat16` | FP8 量化路径需在 combine 前反量化 |
| **LL `use_fp8_dispatch` 搭配** | FP8 dispatch 时 `use_ue8m0_dispatch` 由 `post_init_setup()` 动态决定（取决于 experts 是否支持 packed ue8m0）| ue8m0 scale 格式不匹配，精度损失超预期 |
| **NVFP4 delayed swizzle** | `naive_dp_ep.py` 在 dispatch 后延迟 swizzle：`_delay_swizzle_topk_ids()` → receiver 中 `_swizzle_topk_ids()` | dispatch 传输的是 raw topk_ids；combine 用的是 swizzled ids；顺序颠倒会拿到错误 expert 结果 |

### 并行策略约束

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| **`expert_map` 中 -1 不可省略** | kernel 内部靠 `-1` 判断跳过非本 rank expert；expert_map=None 时单 rank 路径不需要（等效为 all-local）| 多 rank 时本 rank 的 expert 错误地做了不属于本 rank 的 expert 的 GEMM |
| **`output_is_reduced` 标志** | `DeepEPHT` 和 `DeepEPLL` 的 combine kernel 内置 AllReduce；上层必须检查此标志，避免重复 AllReduce | 输出值翻倍（被 AllReduce 两次）|
| **round_robin 自动降级** | round_robin 策略需要同时满足 4 个条件（见 C 节），任一不满足则静默降级为 linear | 路由表和实际 expert 分布不一致，输出结果错误（静默错误）|
| **Shared expert 的独立 AllReduce** | 共享专家（如 DeepSeekV3 的 shared_expert）在 MoE 输出后需单独做 AllReduce；不能与 MoE 部分合并 | 共享专家输出只是单 rank 的贡献，加总后幅度偏小 |

### CUDA Graph 约束

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| **LL 路径才支持 CUDA Graph** | LL 输出 `[E, max_tokens_per_rank, K]` 形状固定；HT 路径因 `M` 动态变化，不兼容 CUDA Graph | HT 路径若强行 graph capture，replay 时 buffer 形状不匹配 |
| **workspace lock 顺序** | CUDA Graph warmup（调 `workspace.lock()`）必须在 graph capture 前执行；Elastic EP 扩缩容时 `unlock() → 扩容 → lock()` | lock 后若 workspace 扩容，原有 tensor 地址失效，graph replay 读到错误地址 |
| **LL `hidden_size` 硬约束** | DeepEP LL kernel 仅为以下 hidden_size 编译：`{2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}`；不在列表中则需 `maybe_roundup_layer_hidden_size()` pad | 超出最大 8192 时直接抛出 `ValueError`；中间值向上 pad 有轻微计算浪费 |

### 激活格式约束

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| **`activation_format` 必须一致** | `PrepareAndFinalize.activation_format` 返回值必须与 `FusedMoEExperts` 期望的输入格式一致 | Standard 格式 `[M, K]` 传入 BatchedExperts 内核（期望 `[E, max_M, K]`），维度不匹配 |
| **BatchedExperts `max_tokens_per_rank` 固定** | LL dispatch 输出的 expert_x 第二维固定为 `max_tokens_per_rank`，与实际 token 数无关 | 配套 experts 内核必须只处理 `expert_num_tokens[i]` 个有效 token，忽略 padding |
| **Monolithic 内核不走 modular 流程** | `FusedMoEKernel.__init__` 类型检查：若 prepare_finalize/fused_experts 不是 Modular 子类，则走 Monolithic 路径（单一 `apply()` 接口）| 将 Monolithic 实现的 experts 传入 Modular prepare/finalize，接口不兼容 |

---

## 验证方法

> 验证体系遵循"由浅入深"原则：先验证每个模块的独立正确性，再验证模块组合后的系统正确性，最后通过性能基准确认优化效果。

### V1. 单 Rank 功能正确性

```python
# 测试脚本框架（无量化，BF16，单 GPU）
import torch
import torch.nn.functional as F

def naive_moe_forward(x, w1, w2, topk_ids, topk_weights):
    """PyTorch 参考实现（无优化，逐 expert 计算）"""
    M, K = x.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    out = torch.zeros(M, K, device=x.device, dtype=x.dtype)
    for m in range(M):
        for ki in range(topk_ids.size(1)):
            e = topk_ids[m, ki].item()
            gate_up = x[m] @ w1[e].T        # [2N]
            gate, up = gate_up[:N], gate_up[N:]
            activated = F.silu(gate) * up    # [N]
            out[m] += topk_weights[m, ki] * (activated @ w2[e].T)  # [K]
    return out

# 对比（atol=1e-3 适合 BF16 精度）
ref = naive_moe_forward(x, w1, w2, topk_ids, topk_weights)
fused = fused_moe_forward(x, w1, w2, topk_ids, topk_weights)
assert torch.allclose(ref.float(), fused.float(), atol=1e-3, rtol=1e-2), \
    f"Max diff: {(ref - fused).abs().max()}"
```

**测试矩阵：**

| 参数 | 测试值 |
|------|--------|
| M（token 数） | 1, 16, 128, 512, 2048 |
| E（expert 数） | 8, 64, 256 |
| top_k | 1, 2, 8 |
| hidden_size K | 1024, 4096, 7168 |
| intermediate N | 1024, 3584, 14336 |
| 激活函数 | SwiGLU, GELU |

---

### V2. 多 Rank EP 正确性

```bash
# 启动 2-GPU EP 测试（验证 dispatch/combine 往返正确性）
torchrun --nproc-per-node=2 tests/test_ep_correctness.py \
    --ep-size=2 --num-experts=16 --top-k=2 --hidden-size=4096

# 验证逻辑：
# 1. rank 0 和 rank 1 各保留一半 expert
# 2. 所有 token 经过 dispatch → experts → combine 后
# 3. 与单 rank（所有 expert 在本地）的输出对比，误差 < 1e-2
```

**关键验证点：**
- `dispatch` 后每个 rank 收到的 `expert_num_tokens` 之和 = 原始 `M * top_k`
- `combine` 后每个 token 的权重之和 ≈ 1.0（若 `norm_topk_prob=True`）
- `expert_map` 中 `-1` 对应的 expert slot 的输出不参与最终结果

---

### V3. 量化精度验证

```python
# FP8 W8A8 精度对比
def test_fp8_precision(x_bf16, w1_bf16, w2_bf16, topk_ids, topk_weights):
    # BF16 参考
    ref = fused_moe_bf16(x_bf16, w1_bf16, w2_bf16, topk_ids, topk_weights)

    # FP8 量化版本
    x_fp8, x_scale = quantize_fp8(x_bf16)
    w1_fp8, w1_scale = quantize_fp8_per_channel(w1_bf16)
    w2_fp8, w2_scale = quantize_fp8_per_channel(w2_bf16)
    fp8_out = fused_moe_fp8(x_fp8, w1_fp8, w2_fp8, x_scale, w1_scale, w2_scale,
                            topk_ids, topk_weights)

    # FP8 精度通常在 atol=0.05（相对误差 ~1%）以内
    rel_err = (ref - fp8_out).abs() / (ref.abs() + 1e-6)
    assert rel_err.mean() < 0.02, f"Mean relative error: {rel_err.mean()}"
    assert rel_err.max() < 0.1, f"Max relative error: {rel_err.max()}"
```

**FP8 特殊验证：**
- `per_act_token=True` vs `per_act_token=False`：两种 scale 形状的正确性
- `block_shape=[128, 128]` block 量化的正确性（scale reshape 逻辑）
- `ue8m0` packed scale 的 unpack 逻辑（DeepEP LL 特有）

---

### V4. DBO（双批次重叠）正确性

```python
# DBO 正确性：控制变量测试
def test_dbo_correctness(input_batches, model_config):
    # 基准：单批次顺序执行（无 DBO）
    ref_outputs = [run_single_batch(b, model_config) for b in input_batches]

    # DBO：双批次交织执行
    dbo_outputs = run_with_dbo(input_batches, model_config, num_ubatches=2)

    for i, (ref, dbo) in enumerate(zip(ref_outputs, dbo_outputs)):
        assert torch.allclose(ref, dbo, atol=1e-5), \
            f"Batch {i}: DBO output differs, max diff = {(ref - dbo).abs().max()}"
```

**DBO 特殊验证（并发安全性）：**

```python
# 验证 recv_hook 注册顺序
# 批次 A 的 recv_hook 必须在批次 B 执行时被调用（而非批次 A 自身）
ctx_a.register_recv_hook(lambda: assert_hook_run_by_batch_b())

# 验证 workspace slot 隔离
# 批次 A 的 workspace[0] 和批次 B 的 workspace[1] 不重叠
assert not tensors_overlap(workspace_slot_0, workspace_slot_1)
```

---

### V5. CUDA Graph 兼容性

```python
# CUDA Graph capture + replay 稳定性测试
def test_cuda_graph_stability(model, input_batch, num_replays=100):
    # Warmup（触发 workspace 分配 + lock）
    for _ in range(3):
        out_warmup = model(input_batch)
    model.workspace_manager.lock()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_captured = model(input_batch)

    # Replay N 次，验证输出稳定
    for i in range(num_replays):
        g.replay()
        assert torch.allclose(out_warmup, out_captured, atol=1e-5), \
            f"Graph replay {i}: output unstable"

    # 验证 lock 后无 workspace 扩容
    try:
        model.workspace_manager._ensure_workspace_size(10**9)  # 尝试扩容
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass  # 预期行为
```

---

### V6. 性能基准

**吞吐测试（prefill 场景）：**

```bash
# 基准命令（参考 vLLM benchmark_throughput.py）
python benchmarks/benchmark_throughput.py \
    --model deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --num-scheduler-steps 1 \
    --max-num-batched-tokens 131072 \
    --input-len 1024 --output-len 1

# 期望指标对比（H100 x8，单机）：
# BF16（Triton）:     ~50k tokens/s
# FP8（CUTLASS/DeepGEMM）: ~80k tokens/s
# FP8 + DeepEP HT：   ~100k tokens/s
# FP8 + DeepEP LL + DBO：~120k tokens/s
```

**延迟测试（decode 场景，batch_size=1）：**

```bash
python benchmarks/benchmark_latency.py \
    --model deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --input-len 1024 --output-len 128

# 期望指标：
# DeepEP LL 应显著优于 HT（目标：LL 比 HT 快 20-40%）
# DeepEP LL + DBO 应比无 DBO 快 10-30%（取决于 M 大小）
```

**Profiling 工具：**

```bash
# NSight Systems：确认 comm/compute 重叠
nsys profile --trace=cuda,nvtx python run_moe.py

# NSight Compute：分析 Triton/CUTLASS kernel 利用率
ncu --metrics sm__throughput.avg.pct python run_moe.py

# vLLM 内置 profiler
VLLM_TORCH_PROFILER_DIR=/tmp/profile python run_vllm.py
tensorboard --logdir /tmp/profile
```

---

### V7. 回归测试集构建建议

| 测试类别 | 覆盖场景 | 工具/框架 |
|---------|---------|---------|
| 单元测试 | 每个 PrepareAndFinalize / Experts 实现 | pytest + parametrize |
| 集成测试 | 完整 MoE 层（router + prepare + experts + finalize）| pytest + torchrun |
| 精度测试 | BF16/FP8/NVFP4 各量化格式 | 参考实现对比 |
| 并行测试 | EP=2/4/8, TP=1/2/4 的所有组合 | torchrun 多进程 |
| 压力测试 | M=1 到 M=65536 的全范围 token 数 | pytest + parametrize |
| 回归检查 | 与 vLLM 原始输出逐 tensor 对比 | pickle 保存参考输出 |

---

*报告生成时间：2026-04-29（基于 vLLM main branch 2025-04 调研）*
*基于 vLLM 仓库路径：`/root/paddlejob/share-storage/gpfs/system-public/zhushengguang/jobspace/vllm/`*
