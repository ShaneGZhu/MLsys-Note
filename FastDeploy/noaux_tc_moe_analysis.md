# FastDeploy MoE noaux_tc 路由机制与调用链路分析

> 基于 FastDeploy `develop` 分支，GLM4-MoE 模型为切入点。

---

## 目录

1. [FusedMoE 类参数说明](#1-fusedmoe-类参数说明)
2. [noaux_tc 算法原理](#2-noaux_tc-算法原理)
   - [topk_group 的作用](#topk_group-的作用)
3. [CUDA Kernel 实现](#3-cuda-kernel-实现)
   - [warp_topk 工具库速查](#warptopk-工具库速查noauxtc_kernelh46)
4. [Python 层调用链路](#4-python-层调用链路)
5. [各后端完整调用链路](#5-各后端完整调用链路)
   - [5.1 Cutlass 后端（w16a16）](#51-cutlass-后端w16a16)
   - [5.2 DeepGemm 后端（block_wise_fp8）](#52-deepgemm-后端block_wise_fp8)
   - [5.3 Triton 后端（wfp8afp8）](#53-triton-后端wfp8afp8)
   - [5.4 EP 路径](#54-ep-路径)
6. [量化路径全览](#6-量化路径全览)
7. [FP8 量化 CUDA Kernel](#7-fp8-量化-cuda-kernel)
8. [Triton GEMM Kernel 内部逻辑](#8-triton-gemm-kernel-内部逻辑)
9. [Triton vs Cutlass 关键差异](#9-triton-vs-cutlass-关键差异)
10. [文件索引](#10-文件索引)

---

## 1. FusedMoE 类参数说明

**文件：** `fastdeploy/model_executor/layers/moe/moe.py`

| 参数 | 类型 | 说明 |
|---|---|---|
| `fd_config` | FDConfig | 全局推理配置（并行、量化、调度等） |
| `hidden_size` | int | 模型隐层维度，如 7168 |
| `reduce_results` | bool | TP 模式下最后是否执行 AllReduce |
| `renormalize` | bool | 是否对 topk weights 归一化使 sum=1（`norm_topk_prob`） |
| `moe_intermediate_size` | int | FFN 中间层宽度，按 `tp_size` 切分 |
| `num_experts` | int | 全局 expert 总数 |
| `expert_id_offset` | int | EP 场景下当前 rank 的 expert 起始编号 |
| `top_k` | int | 每个 token 路由到的 expert 数，如 8 |
| `topk_method` | str | topk 选择算法，`"noaux_tc"` 为 DeepSeek-V3/GLM4 风格带纠偏路由 |
| `topk_group` | int | `noaux_tc` 专用：每个 token 最多从几个 group 中选 expert，作为多样性约束（见 §2 topk_group 说明） |
| `n_group` | int | `noaux_tc` 专用：expert 分成的组数，如 8 |
| `routed_scaling_factor` | float | 路由权重全局缩放因子 |
| `layer_idx` | int | 当前 MoE 层索引，用于 routing replay、EP 初始化 |
| `gate_correction_bias` | Tensor | `e_score_correction_bias`：专家负载纠偏向量（DeepSeek-V3 核心） |
| `redundant_table_manger` | object | EPLB 冗余专家管理器（负载均衡场景） |
| `weight_key_map` | dict | checkpoint key 到内部 param name 的映射 |
| `topk_reduce_func` | Callable | renormalize 时的自定义归一化函数 |

**GLM4-MoE 中的实例化**（`glm4_moe.py:169`）：

```python
self.experts = FusedMoE(
    fd_config,
    hidden_size=fd_config.model_config.hidden_size,
    reduce_results=not self.merge_ffn_tp,
    renormalize=self.norm_topk_prob,
    moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
    num_experts=fd_config.model_config.n_routed_experts,
    top_k=fd_config.model_config.num_experts_per_tok,
    topk_method="noaux_tc",
    topk_group=fd_config.model_config.topk_group,
    n_group=fd_config.model_config.n_group,
    routed_scaling_factor=fd_config.model_config.routed_scaling_factor,
    layer_idx=layer_id,
    gate_correction_bias=self.gate.e_score_correction_bias,
    topk_reduce_func=lambda x: x.sum(axis=-1, keepdim=True) + 1e-20,
)
```

---

## 2. noaux_tc 算法原理

### 名称含义

**No Auxiliary Loss + Temperature Correction**

- **No Auxiliary Loss**：路由无需辅助平衡损失，通过 `e_score_correction_bias` 在推理时动态平衡负载
- **Temperature Correction**：bias 项对 expert 分数进行纠偏，抑制过热 expert、激励冷门 expert

来源：DeepSeek-V3 论文，GLM4-MoE 复用了该路由策略。

### 算法步骤

```
gating_output [N, num_experts]  （gate 线性层输出）
        │
        │ sigmoid
        ▼
scores  [N, num_experts]        （原始路由分数，无偏，用于最终权重）
        │
        ├── + e_score_correction_bias  →  scores_with_bias  （用于选择 expert）
        │
        ▼
【两阶段分组 TopK（CUDA kernel）】

  阶段一：组分数计算
    每个 group 的 score = top1 + top2  （该 group 内所有 expert 分数的前两名之和）

  阶段二：两轮筛选
    1. 从 n_group 个 group 中，按组分数选出 topk_group 个 group
    2. 在选中 group 内，按 scores_with_bias 选出最终 top_k 个 expert
    3. 输出权重取原始 scores（不含 bias），可选 renormalize

        ▼
topk_weights [N, top_k]         （路由权重，无偏）
topk_ids     [N, top_k]         （路由 expert id）
```

### topk_group 的作用

`topk_group` 控制**每个 token 最多可以从几个 group 里选 expert**，是一个强制多样性的约束。

**示例配置（GLM4-MoE 典型值）：**

```
num_experts = 256,  n_group = 8   →  每组 32 个 expert
topk_group  = 3                   →  每个 token 只从 3 个组里选
top_k       = 8                   →  最终选 8 个 expert
```

**为什么要限制 topk_group：**

```
不限制 group：8 个 expert 可能全来自同一组（32 个里选 8 个）
              → expert 高度集中，负载不均，语义多样性差

限制 topk_group=3：8 个 expert 必须分散在 3 个组里
              → 强制 expert 多样性，每组最多贡献 ~3 个 expert
```

**在 kernel 中的实现**（`noauxtc_kernel.h:550`）：

```cpp
// want_neg_inf_num = WARP_SIZE - n_group + topk_group = 32 - 8 + 3 = 27
// 即让 27 个 lane 变成 -inf，剩 5 个 lane 持有有效分数 → 选出了 topk_group=3 个组
int32_t want_neg_inf_num = WARP_SIZE - n_group + topk_group;
while (neg_inf_num < want_neg_inf_num) {
    topk_group_value = cg::reduce(tile, value, cg::greater<T>());
    if (value == topk_group_value) value = neg_inf<T>();  // 逐轮排除最高组
    ...
}
// 之后只有"被选中组"的 expert 才进入 WarpSelect topk 候选
```

`topk_group` 本质上是 noaux_tc 区别于普通 topk 的核心设计之一，配合 `e_score_correction_bias` 共同实现**无辅助 loss 的负载均衡**。

---

### 与标准 softmax-topk 的区别

| 对比项 | 标准 softmax-topk | noaux_tc |
|---|---|---|
| 激活函数 | softmax | sigmoid |
| 选择依据 | softmax 概率 | sigmoid + bias |
| 输出权重 | softmax 概率 | 原始 sigmoid（无 bias） |
| 平衡方式 | 辅助 loss（训练期） | bias 纠偏（推理期动态） |
| 分组约束 | 无 | 分组选 topk_group 个组，保证多样性 |

### Python 层实现

**文件：** `fastdeploy/model_executor/layers/moe/moe.py:81`

```python
def get_moe_scores(gating_output, n_group, topk_group, top_k,
                   routed_scaling_factor, e_score_correction_bias,
                   renormalize, ...):
    scores = paddle.nn.functional.sigmoid(gating_output)
    scores_with_bias = scores + e_score_correction_bias

    if expert_id_to_ep_rank_array is None:
        # 标准版本
        scores, topk_values, topk_idx = noaux_tc(
            scores, scores_with_bias,
            n_group, topk_group, top_k,
            renormalize, routed_scaling_factor
        )
    else:
        # EPLB 冗余 expert 版本
        scores, topk_values, topk_idx, _ = noaux_tc_redundant(...)

    return scores, topk_values, topk_idx
```

---

## 3. CUDA Kernel 实现

### 文件位置

```
custom_ops/gpu_ops/
├── noauxtc_kernel.h          # kernel 完整实现（header-only）
├── noaux_tc.cu               # Paddle custom op 注册
└── noaux_tc_redundant.cu     # EPLB 冗余 expert 版本注册
```

### Op 注册（noaux_tc.cu）

```cpp
std::vector<paddle::Tensor> NoauxTc(
    paddle::Tensor& scores,            // sigmoid 原始分数 [N, num_experts]
    paddle::Tensor& scores_with_bias,  // 加 bias 后的分数 [N, num_experts]
    int n_group, int topk_group, int topk,
    bool renormalize, float routed_scaling_factor)
{
    invokeNoAuxTc<float, int64_t>(
        scores.data<float>(), group_scores.data<float>(),
        topk_values.data<float>(), topk_indices.data<int64_t>(),
        scores_with_bias.data<float>(),
        num_tokens, num_experts, n_group, topk_group, topk,
        renormalize, routed_scaling_factor, stream);
    return {scores, topk_values, topk_indices};
}

PD_BUILD_STATIC_OP(noaux_tc)
    .Inputs({"scores", "scores_with_bias"})
    .Outputs({"output_tensor", "topk_values", "topk_indices"})
    .Attrs({"n_group: int", "topk_group: int", "topk: int",
            "renormalize: bool", "routed_scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(NoauxTc));
```

### 阶段一 Kernel：`topk_with_k2_kernel`（行 475-499）

```
grid  = ceil(num_tokens * n_group / 16)    # 16 warps/block
block = 512
一个 warp 处理一个 (token, group) 对
```

**核心逻辑：**

```
一个 warp（32 threads）处理该 group 内所有 expert 的分数
1. warp reduce 找 max1（最大值）
2. __ballot_sync + __popc 判断 max1 是否唯一
3. 若唯一：将该 lane 置 -inf，再 reduce 得 max2
4. 若不唯一：max2 = max1
5. group_score = max1 + max2  →  写回 group_scores[token][group]
```

> SM >= 900（Hopper）使用 `griddepcontrol.wait/launch_dependents` 汇编实现 PDL 流水线，两阶段 kernel 无需 CPU 同步。

### 阶段二 Kernel：`group_idx_and_topk_idx_kernel`（行 501-668）

```
grid  = ceil(num_tokens / 16)
block = 512
smem  = calc_smem_size_for_block_wide(16_warps, top_k)
一个 warp 处理一个 token
```

**核心逻辑（5步）：**

```
Step 1: 组筛选（warp reduce loop）
  每个 lane 持有一个 group 的 group_score
  循环 reduce + 置 -inf，找出得分最高的 topk_group 个 group
  （处理相同分数的 tie-breaking 细节）

Step 2: 组内 Expert 选择（WarpSelect 双调排序）
  遍历每个选中 group，从其 scores_with_bias 子向量中
  用 WarpSelect 选出 topk 个 expert → s_topk_idx[topk]

Step 3: 读取原始权重（无 bias）
  value = scores[s_topk_idx[i]]       ← 原始 sigmoid，不含 bias
  topk_sum += reduce(value)            ← 用于 renormalize

Step 4: 清零 scores 矩阵
  for i in num_experts: scores[i] = 0  ← 先全部清零

Step 5: 写回最终结果
  if renormalize:
      weight = (value / topk_sum) * routed_scaling_factor
  else:
      weight = value * routed_scaling_factor

  scores[expert_id] = weight           ← sparse scatter 写回
  topk_indices[i]   = expert_id
  topk_values[i]    = weight

异常保底：
  若 topk_group_value == -inf（全部 group 分数异常）
  → 默认选前 top_k 个 expert，权重均等 1/topk
```

### Launcher（行 859-952）

```cpp
void invokeNoAuxTc(..., cudaStream_t stream) {
    // 用 cudaLaunchKernelEx + cudaLaunchAttributeProgrammaticStreamSerialization
    // 将两个 kernel 串联在同一 stream，GPU 自动保证顺序，无需 CPU 同步

    // kernel 1：计算 group_scores
    cudaLaunchKernelEx(&config, topk_with_k2_kernel, ...);

    // kernel 2：group 筛选 + expert topk
    cudaLaunchKernelEx(&config, group_idx_and_topk_idx_kernel, ...);
}
```

### EPLB 冗余 Expert 版本（`noaux_tc_redundant`）

在 Step 5 额外处理：

```cpp
// 同一逻辑 expert 可在多个 EP rank 上存在（冗余副本）
int len = expert_in_rank_num_list[expert_topk];     // 该 expert 在几个 rank 上
int select = xorwow_moe(state) % len;               // xorwow 伪随机（以 token_id 为 seed）
int selected_rank = expert_id_to_ep_rank_array[expert_topk * redundant_ep_rank_num_plus_one + select];
atomicAdd(&tokens_per_expert_stats_list[expert_topk], 1);  // 负载统计
topk_indices[i] = selected_rank;  // 输出 EP rank id，而非 expert id
```

### warp_topk 工具库速查（`noauxtc_kernel.h:46`）

> 开发 `topk_with_k2_kernel` 和 `group_idx_and_topk_idx_kernel` 时的工具参考。

#### 工具函数

| 函数 | 签名 | 用途 |
|---|---|---|
| `round_up_to_multiple_of<N>(len)` | `__host__ __device__ constexpr T` | 对 N 向上取整，用于循环边界对齐到 WARP_SIZE |
| `calc_smem_size_for_block_wide<T,idxT>(num_warps, k)` | `int` | 计算 `WarpSelect` 需要的动态 smem 字节数，传给 kernel 启动参数 |
| `isPowerOf2(v)` | `constexpr bool` | 编译期检查 2 的幂次（`WarpSelect capacity` 约束） |
| `neg_inf<T>()` | `__device__ T` | 返回类型 T 的负无穷（via float cast，bf16/fp16 安全） |

#### WarpSelect — 核心 top-k 选择器

```cpp
// 模板参数
WarpSelect<
    int capacity,      // 内部缓存大小，必须是 2^n 且 >= 32（WARP_SIZE）
                       // topk <= capacity；两个 kernel 均用 capacity=WARP_SIZE=32
    bool greater,      // true=选最大 top-k，false=选最小
    typename T,        // 分数类型（float / __nv_bfloat16）
    typename idxT,     // 索引类型（int32_t）
    bool is_stable     // true=相同分数时用 smaller index 打破平局
>
```

**构造**

```cpp
WarpSelect<WARP_SIZE, true, T, int32_t, true> queue(
    (int32_t)topk,   // k：最终保留前 k 个
    neg_inf<T>()     // dummy：初始填充值，选最大时用负无穷
);
```

**API 调用顺序（必须严格遵守）**

```
1. queue.add(val, idx)      // 可调用任意多次，每次添加一个候选
   或
   queue.add(in_ptr, start, end)  // 批量添加 [start, end) 范围
                                  // 内部自动对齐到 WARP_SIZE 边界

2. queue.done()             // 必须调用！内含 __syncthreads()
                            //   → 整个 block 所有 warp 必须都走到此处
                            //   → done() 之后 smem 被 WarpSelect 用于 warp 间归并
                            //   → 不要在 done() 后、dump 前向 smem 写其他数据

3. queue.dumpIdx(out_idx)   // 只导出索引（两个 kernel 实际用法）
   或
   queue.dump(out, out_idx) // 同时导出值和索引
```

**约束汇总**

| 约束 | 说明 |
|---|---|
| `capacity` 必须是 2 的幂次且 ≥ 32 | static_assert 保证，编译期报错 |
| `topk ≤ capacity` | 两个 kernel 均用 `capacity=WARP_SIZE=32`，故 `topk ≤ 32` |
| `done()` 含 `__syncthreads()` | block 内所有 warp 必须无条件走到 `done()`，用 `if_proceed_next_topk` 保护 `add()` 但不能保护 `done()` |
| `add()` 的 idx 是全局 expert 编号 | `dumpIdx` 导出的也是全局编号，可直接作为 `scores[]` 下标 |
| smem 复用时序 | `group_idx_and_topk_idx_kernel` 里 `s_topk_idx` / `s_topk_value` 与 `WarpSelect` 共用同一块 `smem_buf`；`WarpSelect` 在 `done()` 后才使用 smem，因此 `dumpIdx` 写完 `s_topk_idx` 之后才能继续使用 |

#### BitonicSort / BitonicMerge — 内部排序（不直接调用）

`WarpSelect` 的内部实现，开发者**无需直接调用**。只需知道：
- warp 内双调排序，寄存器级别，无需额外 smem
- `capacity=32` 时退化为纯 warp shuffle 实现（`BitonicSort<32,...>`）

#### smem 布局示意（`group_idx_and_topk_idx_kernel`）

```
smem_buf  [0 ...]
│
├── WarpSelect 内部区域（val_smem_ / idx_smem_）
│     由 WarpSelect 构造函数自动计算偏移，大小 = calc_smem_size_for_block_wide(NUM_WARPS_PER_BLOCK, topk)
│
└── 额外区域（kernel 手动管理，与 WarpSelect smem 复用，时序安全）
      s_topk_idx   [NUM_WARPS_PER_BLOCK * topk]  int32_t
      s_topk_value [warp_id * topk ... (warp_id+1)*topk]  T
                                   ↑ 每个 warp 独立一段，避免竞争
```

---

## 4. Python 层调用链路

### noaux_tc Python 绑定

**文件：** `fastdeploy/model_executor/ops/gpu/fastdeploy_ops/__init__.py:4974`

```python
@unified
def static_op_noaux_tc(scores, scores_with_bias, n_group, topk_group, topk,
                        renormalize, routed_scaling_factor):
    outs = _C_ops._run_custom_op("static_op_noaux_tc",
        scores, scores_with_bias,
        n_group, topk_group, topk, renormalize, routed_scaling_factor)
    return output_tensor, topk_values, topk_indices
```

底层通过 `_C_ops._run_custom_op(op_name, ...)` 调用注册在 `fastdeploy_ops_pd_.so` 中的 CUDA kernel。

### 并行模式决策树

```
FusedMoE.forward(x, gate)
│
├── EP > 1 && TP > 1 && tokens >= tp_size
│     └─► forward_split_allgather()
│
├── EP > 1 && enable_chunked_moe
│     └─► forward_chunked_moe()
│
└── 其他
      └─► forward_normal()
            └─► quant_method.apply()
                   ├── EP > 1 && prefill  →  apply_ep_prefill()
                   ├── EP > 1 && decode   →  apply_ep_decode()
                   └── EP == 1            →  apply_tp()  /  apply() (Triton)
```

**所有路径的路由逻辑均一致：**

```python
# Cutlass / DeepGemm / Triton 所有后端 apply() 中：
if layer.topk_method == "noaux_tc":
    gate_out, topk_weights, topk_ids = get_moe_scores(...)  # ← 共用
else:
    topk_ids, topk_weights = moe_topk_select(...)           # ← 其他方法
```

---

## 5. 各后端完整调用链路

### 5.1 Cutlass 后端（w16a16）

**文件：** `fused_moe_cutlass_backend.py:336`（`apply_tp`）

```
FusedMoE.forward(x, gate)
  └─ forward_normal()
       └─ CutlassMoEMethod.apply_tp()
            │
            │ Step 1: 路由
            ├─ gate(x) → gate_out [N, num_experts]
            ├─ get_moe_scores() → topk_weights, topk_ids
            │    ├─ sigmoid(gate_out)
            │    ├─ + e_score_correction_bias
            │    └─ static_op_noaux_tc()  ← CUDA kernel
            │
            │ Step 2: Token 重排（物理 gather）
            ├─ moe_expert_dispatch(x, gate_out, topk_idx, topk_only_mode=True)
            │    └─ static_op_moe_expert_dispatch()  ← CUDA kernel
            │    → permute_input [N*top_k, hidden]   （expert-major 顺序）
            │    → token_nums_per_expert, permute_indices, topk_weights, topk_idx
            │
            │ Step 3: Group GEMM + SwiGLU
            ├─ moe_expert_ffn(permute_input, token_nums_per_expert,
            │                  up_gate_proj_weight, down_proj_weight)
            │    └─ static_op_moe_expert_ffn()  ← Cutlass Group GEMM CUDA kernel
            │       a. GEMM: [N*top_k, hidden] @ [E, hidden, mid*2] → [N*top_k, mid*2]
            │       b. SwiGLU: silu(a) * b → [N*top_k, mid]
            │       c. GEMM: [N*top_k, mid] @ [E, mid, hidden] → [N*top_k, hidden]
            │
            │ Step 4: 加权聚合
            └─ moe_expert_reduce(ffn_out, topk_weights, permute_indices)
                 └─ static_op_moe_expert_reduce()  ← CUDA kernel
                 → [N, hidden]  （按 topk_weights 加权求和）
```

### 5.2 DeepGemm 后端（block_wise_fp8）

**文件：** `fused_moe_deepgemm_backend.py`

```
DeepGemmFusedMoeMethod.apply_tp()
  │
  ├─ gate(x) + get_moe_scores()             → topk_weights, topk_ids（同上）
  │
  ├─ per_token_quant(x, block_size=128)     → x_fp8, x_scale  （FP8 blockwise 量化）
  │
  ├─ ep_moe_expert_dispatch_fp8(...)        → permute_input_fp8, permute_scale
  │
  ├─ m_grouped_fp8_gemm_nt_contiguous(...)  ← DeepGemm FP8 Group GEMM（SM90 H100 优化）
  │    up_gate_proj：[N*top_k, mid*2]
  │
  ├─ fused_mask_swiglu_fp8_quant(...)       ← SwiGLU + FP8 再量化（融合 kernel）
  │
  ├─ m_grouped_fp8_gemm_nt_contiguous(...)  ← down_proj
  │    [N*top_k, hidden]
  │
  └─ ep_moe_expert_combine(...)             → [N, hidden]
```

**FP8 量化说明：**
- `block_size=128`：每 128 个元素共用一个 scale
- scale 在 `apply_tp` 中做 transpose（`fused_moe_deepgemm_backend.py:580`）：DeepGemm 要求 activation scale 为列主序

### 5.3 Triton 后端（wfp8afp8）

**文件：** `fused_moe_triton_backend.py:679`（`Wfp8Afp8MoEMethod.apply`）

```
Wfp8Afp8MoEMethod.apply(layer, x, gate)
  │
  │ Step 1: 路由（与 Cutlass 后端完全相同）
  ├─ gate(x) + get_moe_scores() → topk_weights, topk_ids
  │
  │ Step 2: Preprocess（排序索引，无物理拷贝）
  ├─ tritonmoe_preprocess_func(topk_ids, num_experts, BLOCK_SIZE_M)
  │    → sorted_token_ids  [N*top_k + padding]  （按 expert 排序的 token 下标）
  │    → expert_ids        [num_blocks]          （每个 tile 对应的 expert）
  │    → num_tokens_post_padded
  │
  │ Step 3: up_gate_proj（FP8 w8a8）
  ├─ x_q, x_scale = scaled_fp8_quant(x)        （per-token 动态 FP8 量化）
  ├─ fused_moe_kernel_paddle[grid](
  │      x_q, up_gate_proj_weight, out,
  │      x_scale, up_gate_proj_weight_scale,
  │      use_fp8_w8a8=True, per_channel_quant=True,
  │      MUL_ROUTED_WEIGHT=False                 ← up_gate 不乘路由权重
  │  )  ← Triton JIT kernel
  │
  │ Step 4: SwiGLU
  ├─ paddle.incubate.nn.functional.swiglu(up_gate_proj_out)
  │
  │ Step 5: down_proj（FP8 w8a8）
  ├─ x_q, x_scale = scaled_fp8_quant(down_proj_input)   （再量化）
  ├─ fused_moe_kernel_paddle[grid](
  │      x_q, down_proj_weight, out,
  │      x_scale, down_proj_weight_scale,
  │      topk_weights,                             ← 传入路由权重
  │      use_fp8_w8a8=True, per_channel_quant=True,
  │      MUL_ROUTED_WEIGHT=True                    ← down_proj 直接乘路由权重
  │  )
  │
  │ Step 6: 聚合
  └─ out.reshape_([N, top_k, hidden]).sum(axis=1)
       ← 直接 sum，因 down_proj 已乘 topk_weights
```

**Triton 配置自适应：**

```python
config = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, ...}  # 默认高吞吐
if token_num <= num_experts:
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, ...}  # decode 低延迟
```

### 5.4 EP 路径

```
apply_ep_decode()  （低延迟，DeepEP）
  ├─ get_moe_scores()                     → topk_ids, topk_weights
  ├─ ep_decoder_runner.dispatch(x, topk_idx)
  │    └─ deep_ep.Buffer.low_latency_dispatch()   ← DeepEP all-to-all CUDA kernel
  ├─ compute_ffn(...)                     ← 同 TP 路径的 moe_expert_ffn
  └─ ep_decoder_runner.combine(ffn_out)
       └─ deep_ep.Buffer.low_latency_combine()    ← DeepEP all-to-all CUDA kernel

apply_ep_prefill()  （高吞吐，DeepEP）
  ├─ get_moe_scores()
  ├─ deep_ep.Buffer.dispatch()            ← all-to-all
  ├─ ep_moe_expert_dispatch() + compute_ffn()
  ├─ ep_moe_expert_combine()
  └─ deep_ep.Buffer.combine()             ← all-to-all（结果收集）
```

---

## 6. 量化路径全览

### 量化方法决策链

```
FusedMoE.__init__
  │
  ├── moe_quant_config 存在？
  │     ├── "wfp8afp8"         → Wfp8Afp8MoEMethod        (Triton,  per-token FP8)
  │     ├── "tensor_wise_fp8"  → TensorWiseFP8MoEMethod    (Triton,  tensor-level FP8)
  │     ├── "block_wise_fp8"
  │     │     ├── SM100        → BlackwellGemmFusedMoeMethod
  │     │     ├── DeepGemm/EP  → DeepGemmFusedMoeMethod    (DeepGemm, block FP8)
  │     │     └── 其他         → BlockWiseFP8MoEMethod      (Triton)
  │     ├── "w4a8"             → CutlassW4A8MoEMethod       (Cutlass)
  │     ├── "w4afp8"           → CutlassW4AFP8MoEMethod     (Cutlass)
  │     └── "weight_only"      → CutlassWeightOnlyMoEMethod (Cutlass)
  │
  └── 无量化 → get_moe_method() → CutlassMoEMethod          (Cutlass, w16a16)
```

### FP8 量化路径对比

| 量化名称 | scale 粒度 | MoE 后端 | activation 量化时机 |
|---|---|---|---|
| `wfp8afp8` | per-token（激活）+ per-channel（权重） | Triton | `scaled_fp8_quant()`，每次 GEMM 前 |
| `tensor_wise_fp8` | tensor-level（全局一个 scale） | Triton | 静态 scale，加载时确定 |
| `block_wise_fp8` | per-block 128 元素 | DeepGemm / Triton | `per_token_quant(block_size=128)` |

---

## 7. FP8 量化 CUDA Kernel

**文件：** `custom_ops/gpu_ops/per_token_quant_fp8.cu`

### `quant_per_token_per_block` kernel

```
输入:  BF16/FP16  [token_num, hidden_size]
输出:  FP8-E4M3   [token_num, hidden_size]
       scale       [token_num, hidden_size/128]   (float32 或 UE8M0)

配置:
  每个 warp 负责 128 元素的一个 block（4 elements/thread）
  gridx = min(132*8, token_num)
  blockx = min(1024, hidden_size/128 * 32)
```

**Scale 计算逻辑：**

```
1. 每个 thread 读 4 个元素，计算本地 |x|_max
2. warp reduce（5次 __shfl_down_sync）得 block 内 |x|_max
3. broadcast 到所有 lane（__shfl_sync(..., 0)）

标准路径（UseUE8M0=false）：
  scale = max_value / 448.0       (448 = FP8-E4M3 最大值)
  x_fp8 = round(x * 448 / max_value)

UE8M0 路径（SM100 Blackwell）：
  scale = exp2(ceil(log2(max_value / 448)))   ← 对齐到 2 的整数次幂
  x_fp8 = round(x / scale)
  存储：取浮点指数位 (scale_bits >> 23) & 0xFF，4个 scale 打包成 1个 int32

环境变量 PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE=1：
  max_value *= 7.0f               ← 扩大量化范围（减溢出，稍降精度）
```

**`PerTokenQuantPadding` 变体**（DeepGemm 专用）：
- scale 以**列主序**存储 `[hidden_size_scale, padded_token_num]`
- 对齐到 TMA（Tensor Memory Accelerator）的 16-byte 边界
- 在 `fused_moe_deepgemm_backend.py:580` 中通过 transpose 调整布局

---

## 8. Triton GEMM Kernel 内部逻辑

**文件：** `fastdeploy/model_executor/layers/moe/triton_moe_kernels.py:27`

```python
@triton.jit
def fused_moe_kernel_paddle(
    a_ptr,           # activation [N, K]，通过 sorted_token_ids 间接寻址
    b_ptr,           # weight     [E, N_out, K]（stacked expert weights）
    c_ptr,           # output     [N*top_k, N_out]
    a_scale_ptr,     # activation scale
    b_scale_ptr,     # weight scale
    topk_weights_ptr,
    sorted_token_ids_ptr,  # token 排序索引（按 expert 分组）
    expert_ids_ptr,        # 每个 tile 对应的 expert
    ...
):
    # 1. 用 expert_ids[pid_m] 选择当前 tile 对应的 expert 权重
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + ...

    # 2. 通过 sorted_token_ids 间接寻址 activation（无物理 gather）
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    a_ptrs = a_ptr + (offs_token // top_k) * stride_am + ...

    # 3. scale 加载（3种量化模式）
    #   per_channel_quant：a_scale[token]，b_scale[expert][channel]
    #   block_wise：a_scale[token][k_block]，b_scale[expert][n_block][k_block]
    #   tensor_wise：a_scale[expert]，b_scale[expert]（各一个标量）

    # 4. GEMM 主循环（沿 K 维 tiling）
    for k in range(0, K // BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask)
        b = tl.load(b_ptrs)
        if use_fp8_w8a8 and block_wise:
            accumulator += tl.dot(a, b) * a_scale * b_scale  # 每 block 一次 dequant
        else:
            accumulator = tl.dot(a, b, acc=accumulator)       # 末尾统一 dequant

    # 5. 路由权重乘法（仅 down_proj，MUL_ROUTED_WEIGHT=True）
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token)
        accumulator = accumulator * moe_weight[:, None]

    # 6. Dequant（末尾统一，仅 per_channel_quant 模式）
    if use_fp8_w8a8 and per_channel_quant:
        accumulator = (accumulator * a_scale * b_scale).to(bfloat16)

    tl.store(c_ptrs, accumulator)
```

---

## 9. Triton vs Cutlass 关键差异

| 对比项 | Triton 路径 | Cutlass 路径 |
|---|---|---|
| **路由（noaux_tc）** | 完全相同，均调用 `get_moe_scores()` | 相同 |
| **Token 重排方式** | `sorted_token_ids` 排序索引，无物理拷贝，kernel 内间接寻址 | `moe_expert_dispatch` 物理 gather，生成 `permute_input` 新 buffer |
| **GEMM 实现** | Triton JIT（Python 编写，JIT 编译），灵活但稍慢 | Cutlass C++ CUDA（高度优化，更高吞吐） |
| **输出聚合** | down_proj 时 `MUL_ROUTED_WEIGHT=True` 直接乘权重，最后 `sum(axis=1)` | 独立 `moe_expert_reduce` CUDA op |
| **FP8 量化时机** | `scaled_fp8_quant()`，每次 GEMM 前实时计算 | `per_token_quant()`，dispatch 前一次性完成 |
| **适用场景** | 量化方案灵活，适合新格式快速支持 | 高吞吐推理，生产环境首选 |

---

## 10. 文件索引

| 功能 | 文件路径 |
|---|---|
| **noaux_tc CUDA kernel** | `custom_ops/gpu_ops/noauxtc_kernel.h` |
| **noaux_tc op 注册** | `custom_ops/gpu_ops/noaux_tc.cu` |
| **noaux_tc redundant op 注册** | `custom_ops/gpu_ops/noaux_tc_redundant.cu` |
| **FP8 per-token 量化 kernel** | `custom_ops/gpu_ops/per_token_quant_fp8.cu` |
| **Python noaux_tc 绑定** | `fastdeploy/model_executor/ops/gpu/fastdeploy_ops/__init__.py:4974` |
| **get_moe_scores / FusedMoE 类** | `fastdeploy/model_executor/layers/moe/moe.py` |
| **Cutlass 后端（w16a16/w4a8/w4afp8）** | `fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py` |
| **DeepGemm 后端（block_wise_fp8）** | `fastdeploy/model_executor/layers/moe/fused_moe_deepgemm_backend.py` |
| **Triton 后端（wfp8afp8/tensor_wise/block_wise）** | `fastdeploy/model_executor/layers/moe/fused_moe_triton_backend.py` |
| **Triton GEMM kernel** | `fastdeploy/model_executor/layers/moe/triton_moe_kernels.py` |
| **EP 运行器** | `fastdeploy/model_executor/layers/moe/ep.py` |
| **量化配置注册** | `fastdeploy/model_executor/layers/quantization/__init__.py` |
| **wfp8afp8 量化配置** | `fastdeploy/model_executor/layers/quantization/wfp8afp8.py` |
| **GLM4-MoE 模型（入口）** | `fastdeploy/model_executor/models/glm4_moe.py` |
