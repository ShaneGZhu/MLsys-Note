# MoE Scoring 前处理：FastDeploy `get_moe_scores` vs vLLM `grouped_topk`

> **调研目标：** 找到 vLLM 中与 FastDeploy `get_moe_scores` 等价的前处理实现，并深入分析其 CUDA kernel 实现细节。
> **对应模型：** DeepSeek-V2 / V3（`topk_method="noaux_tc"`）
> **调研日期：** 2026-04-29

---

## 目录

1. [FastDeploy `get_moe_scores` 功能概述](#1-fastdeploy-get_moe_scores-功能概述)
2. [vLLM 对应实现：调用链](#2-vllm-对应实现调用链)
3. [逐步骤对照](#3-逐步骤对照)
4. [CUDA Kernel 深度解析](#4-cuda-kernel-深度解析)
5. [两个 CUDA Kernel 的选路逻辑](#5-两个-cuda-kernel-的选路逻辑)
6. [通用路径：`grouped_topk_fused_kernel`](#6-通用路径grouped_topk_fused_kernel)
7. [优化路径：`grouped_topk_fused_small_expert_count_kernel`](#7-优化路径grouped_topk_fused_small_expert_count_kernel)
8. [与 FastDeploy `noaux_tc` 的关键差异](#8-与-fastdeploy-noaux_tc-的关键差异)

---

## 1. FastDeploy `get_moe_scores` 功能概述

**文件位置：**
```
FastDeploy/fastdeploy/model_executor/layers/moe/moe.py:81~146
```

**函数签名：**
```python
def get_moe_scores(
    gating_output,          # [M, n_expert]  router GEMM 的原始 logits
    n_group,                # expert 分组数
    topk_group,             # 每次选多少个组
    top_k,                  # 最终选几个 expert
    routed_scaling_factor,  # 输出权重乘以的全局 scale
    e_score_correction_bias,# [1, n_expert] fp32，选组用的偏置
    renormalize,            # 是否对 topk 权重 sum-normalize
    ...                     # 冗余 expert 扩展参数
) -> (scores, topk_values, topk_idx)
```

**核心逻辑（9 步）：**

```python
# Step 1: sigmoid 激活
scores = sigmoid(gating_output)                        # [M, E]

# Step 2: 加 correction bias（仅用于 expert/group 选择，不影响最终权重值）
scores_with_bias = scores + e_score_correction_bias    # [M, E]

# Step 3~7: noaux_tc kernel 内部完成
# 3. 将 E 个 expert 分成 n_group 组
# 4. 每组计算组分数（FastDeploy: 每组最大 biased score）
# 5. 按组分数选 topk_group 个组
# 6. 从选中组的 expert 中（按 biased score）选全局 top_k 个 expert
# 7. 取这 top_k 个 expert 对应的原始 sigmoid 分数作为权重（不含 bias）
topk_values, topk_idx = noaux_tc(scores, scores_with_bias, ...)

# Step 8: 可选 renormalize
if renormalize:
    topk_values /= sum(topk_values) + 1e-20

# Step 9: 乘全局 scale
topk_values *= routed_scaling_factor
```

---

## 2. vLLM 对应实现：调用链

```
模型层
  deepseek_v2.py:274  →  if topk_method == "noaux_tc": 注册 e_score_correction_bias
                      →  GroupedTopKRouter(scoring_func="sigmoid", ...)

Python 层
  grouped_topk_router.py:81   grouped_topk()           ← Python fallback
  grouped_topk_router.py:29   fused_grouped_topk()     ← CUDA 优化入口
  _custom_ops.py:2544         ops.grouped_topk()       ← 转发到 C++ op

C++ 绑定层
  csrc/moe/torch_bindings.cpp:130  _moe_C.grouped_topk  ← torch custom op 注册

CUDA Kernel 层
  csrc/moe/grouped_topk_kernels.cu:888   invokeNoAuxTc()
    ├─ grouped_topk_fused_small_expert_count_kernel   (SM90 PDL 优化路径)
    └─ grouped_topk_fused_kernel                      (通用路径)
```

> **来源注释（文件第 3 行）：**
> `Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu`
> — 与 FastDeploy 的 `noaux_tc` kernel **同源**，均来自 TRT-LLM。

---

## 3. 逐步骤对照

| 步骤 | FastDeploy `get_moe_scores` | vLLM `grouped_topk()` | 代码位置 |
|------|----------------------------|-----------------------|---------|
| ① sigmoid | `scores = sigmoid(gating_output)` | `scores = gating_output.sigmoid()` | `grouped_topk_router.py:116` |
| ② 保存原始分数 | `noaux_tc` kernel 内部处理 | `original_scores = scores`（加 bias 前保存）| `grouped_topk_router.py:124` |
| ③ 加 bias | `scores_with_bias = scores + e_score_correction_bias` | `scores = scores + e_score_correction_bias.unsqueeze(0)` | `grouped_topk_router.py:125` |
| ④ 计算组分数 | 每组 **最大** biased score | 每组 **top-2 biased scores 之和**（`topk(2)[0].sum(-1)`）| `grouped_topk_router.py:126~128` |
| ⑤ 选 topk_group 个组 | `noaux_tc` 内部 WarpSelect | `group_idx = topk(group_scores, topk_group)[1]` | `grouped_topk_router.py:136~138` |
| ⑥ 遮掉未选中组 | `noaux_tc` 内部 mask | `masked_fill(~score_mask, -inf)` | `grouped_topk_router.py:141~146` |
| ⑦ 全局 topK（按 biased 选位置） | `noaux_tc` 返回 topk_idx | `topk_ids = topk(tmp_scores, topk)[1]` | `grouped_topk_router.py:149` |
| ⑧ 取权重（用原始 sigmoid 分数） | `noaux_tc` 返回 topk_values | `topk_weights = original_scores.gather(1, topk_ids)` | `grouped_topk_router.py:151` |
| ⑨ renormalize | `topk_values /= sum + 1e-20` | `topk_weights /= topk_weights.sum(dim=-1, keepdim=True)` | `grouped_topk_router.py:157~158` |
| ⑩ 乘 scale | `topk_values *= routed_scaling_factor` | `topk_weights *= routed_scaling_factor` | `grouped_topk_router.py:160~161` |

**步骤 ④ 组分数计算差异说明：**

| 实现 | 组分数计算方式 | 对应模型 |
|------|--------------|---------|
| FastDeploy `noaux_tc` | 每组最大 biased score（top-1）| DeepSeek-V2 风格 |
| vLLM Python fallback（有 bias 时）| 每组 top-2 biased scores 之和 | DeepSeek-V3 官方实现 |
| vLLM Python fallback（无 bias 时）| 每组最大 score（max）| DeepSeek-V2 风格 |
| vLLM CUDA kernel | **top-2 biased scores 之和**（与 V3 官方对齐）| DeepSeek-V3 |

> 两种方式在语义上都是"优先选择 biased 分数高的组"，top-2 sum 对于 expert 数 > 1 的组更稳健，是 V3 的推荐实现。

---

## 4. CUDA Kernel 深度解析

### 4.1 文件结构

```
csrc/moe/grouped_topk_kernels.cu
├── 工具类（行 47~411）
│   ├── BitonicMerge / BitonicSort   —— warp 内双调排序
│   └── WarpSort / WarpSelect        —— warp 级 top-K 选择（带 SMEM 暂存）
│
├── 辅助函数（行 413~513）
│   ├── apply_scoring<SF>()          —— 按 scoring_func 做激活（sigmoid/none）
│   └── topk_with_k2()               —— 单 warp 计算一组的 top-2 biased scores 之和
│
├── Kernel 实现（行 516~885）
│   ├── grouped_topk_fused_kernel                        —— 通用路径
│   └── grouped_topk_fused_small_expert_count_kernel     —— SM90+ 优化路径
│
└── 启动函数 + 实例化（行 888~999）
    ├── invokeNoAuxTc<T, BiasT, IdxT, SF>()
    └── INSTANTIATE_NOAUX_TC 宏（18 个类型组合）
```

### 4.2 sigmoid 实现

```cuda
// 用 tanh 近似代替 1/(1+exp(-x))，避免 exp 在大负数时的数值不稳定
__device__ inline float sigmoid_accurate(float x) {
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}
```

> **等价性：** `sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5` 是数学恒等式，精度和稳定性优于直接计算。

### 4.3 ScoringFunc 枚举

```cuda
enum ScoringFunc {
    SCORING_NONE    = 0,  // 不做激活（softmax 已在 Python 层计算）
    SCORING_SIGMOID = 1   // kernel 内部 fused sigmoid（sigmoid 路径）
};
```

`fused_grouped_topk()` Python 层的调用约定：
- `scoring_func="sigmoid"` → 传原始 logits + `scoring_func=1`（kernel 内做 sigmoid）
- `scoring_func="softmax"` → Python 层先做 softmax，传 scores + `scoring_func=0`（无激活）

---

## 5. 两个 CUDA Kernel 的选路逻辑

位置：`invokeNoAuxTc()`，行 888~973

```
                invokeNoAuxTc()
                      │
          ┌───────────┴───────────┐
          │                       │
  is_single_group          is_multi_group
  n_group==1, topk_group==1,   n_group>1, E≤256,
  E≤512, topk≤8 (or ==22)     epg≤32, topk_group≤4,
                                topk≤8
          │                       │
          └──────────┬────────────┘
                     │
        grouped_topk_fused_small_expert_count_kernel
        （SM90 PDL 优化，shared memory 版，线程数 = E）
                     │
              ┌──────┴──────┐
          否则（大 E 或超限参数）
              │
    grouped_topk_fused_kernel
    （通用，动态 smem，每 warp = 一组）
```

**模型与 kernel 路径对应：**

| 模型 | E | n_group | topk | topk_group | kernel 路径 |
|------|---|---------|------|------------|------------|
| DeepSeek-V3 | 256 | 8 | 8 | 3 | `small_expert_count`（`is_multi_group`）|
| Kimi-K2 | 384 | — | — | — | `small_expert_count`（`NumKimiK2Experts`）|
| Nemotron | 512 | 1 | 22 | 1 | `small_expert_count`（`MaxSupportedTopExperts=22`）|
| 其他（E>512 等）| >512 | — | — | — | `grouped_topk_fused_kernel` |

---

## 6. 通用路径：`grouped_topk_fused_kernel`

**位置：** 行 516~670

**启动配置：**
```
grid  = num_tokens          （每 block 处理 1 个 token）
block = n_group × 32        （每 warp 负责 1 个 expert group）
smem  = WarpSelect 暂存区 + s_group_scores[n_group]
```

### Phase 1：各 warp 并行计算组分数（行 567~572）

每个 warp（= 一个 group）调用 `topk_with_k2<SF>()`：

```cuda
// 每 lane 遍历组内部分 experts，找局部 top-2
for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
    T value = apply_scoring<SF>(input[i]);  // sigmoid(logit[i])
    value = value + static_cast<T>(bias[i]); // + correction_bias
    // 维护局部 top-1 / top-2
}
// warp reduce：全局 top-1 和 top-2
T max1 = cg::reduce(tile, largest, cg::greater<T>());
T max2 = ...;
// 写入 shared memory
s_group_scores[warp_id] = max1 + max2;  // 组分数 = top2 biased scores 之和
```

### Phase 2：warp 0 完成 group 选择 + expert 选择（行 574~669）

```
Step A: WarpSelect 从 n_group 个组分数中选出 topk_group 个组
Step B: 遍历被选中的组，对每个 expert 重新计算 sigmoid + bias
        用 WarpSelect 从候选 experts 中选出全局 topk 个
Step C: 对选中的 topk 个 expert，取原始（无 bias）sigmoid 分数
        lane_unbiased = sigmoid(scores_token[selected_expert_id])
Step D: 可选 renorm：topk_sum = Σ lane_unbiased + 1e-20，scale /= topk_sum
Step E: 输出 topk_values[i] = lane_unbiased * scale
        输出 topk_indices[i] = selected_expert_id
```

**SM90+ PDL 优化标记（行 561~564, 667~669）：**
```cuda
// kernel 入口：等待上游 kernel（通常是 router GEMM）完成
asm volatile("griddepcontrol.wait;");

// kernel 出口：通知下游 kernel（dispatch/GEMM）可以启动
asm volatile("griddepcontrol.launch_dependents;");
```

---

## 7. 优化路径：`grouped_topk_fused_small_expert_count_kernel`

**位置：** 行 675~885

**核心区别（相较通用路径）：**

| 特性 | 通用路径 | 优化路径 |
|------|---------|---------|
| expert score 存储 | 寄存器 | **shared memory**（`smemScoreSigmoid` / `smemScoreBias`）|
| TopK 实现 | `WarpSelect`（warp + SMEM 暂存）| `reduce_topk::reduceTopK`（更轻量）|
| 启动配置 | `block = n_group * 32` | `block = num_experts`（每线程一个 expert）|
| 动态 SMEM | 需要（WarpSelect 暂存）| **不需要**（静态 SMEM）|
| SM90 PDL | `griddepcontrol.wait/launch_dependents` | `cudaGridDependencySynchronize` / `cudaTriggerProgrammaticLaunchCompletion` |

**数据流：**
```
每线程加载 1 个 expert 的 logit
  → apply_scoring<SF>()           → smemScoreSigmoid[threadExpert]
  → scoreSigmoid + bias           → smemScoreBias[threadExpert]

# UseGroups=true 时（DeepSeek-V3）：
每 warp = 1 group：
  reduce_topk::reduceTopK → 取组内 top-2 biased scores
  warp lane 0 写 smemGroupScores[warpIdx] = top1 + top2

__syncthreads()

warp 0：
  从 smemGroupScores 选 topk_group 个组（reduceTopK）
  对被选中组的每个 expert 从 smemScoreBias 取值，选全局 topk（reduceTopK）
  从 smemScoreSigmoid 取原始分数（无 bias）→ 乘 scale → 输出
```

---

## 8. 与 FastDeploy `noaux_tc` 的关键差异

| 维度 | vLLM `grouped_topk_kernels.cu` | FastDeploy `noaux_tc` |
|------|-------------------------------|----------------------|
| **代码来源** | TRT-LLM `noAuxTcKernels.cu`（直接改编）| 自研 CUDA kernel |
| **组分数计算** | **top-2 biased scores 之和**（V3 官方做法）| 每组**最大** biased score（V2 做法）|
| **SM90+ 优化** | PDL（Programmatic Dependent Launch）✅ | 无 |
| **冗余 expert** | 不支持（无 `noaux_tc_redundant` 等价）| `noaux_tc_redundant` 支持 EP 冗余副本 |
| **sigmoid 实现** | `tanh` 近似（`0.5*tanh(0.5*x)+0.5`，数值更稳定）| 标准 sigmoid |
| **输出格式** | `float32 values + int32 indices` | 相同 |
| **支持的模型** | DeepSeek-V3/V2、Kimi-K2、Nemotron（硬编码阈值）| 通用（由调用方传参）|
| **Python fallback** | 有（`grouped_topk()` 纯 PyTorch 实现）| 无（仅 CUDA kernel）|

---

## 附录：关键常量与类型实例化

```cuda
// csrc/moe/grouped_topk_kernels.cu

// 硬编码的模型 expert 数阈值
static constexpr int NumNemotronExperts  = 512;
static constexpr int NumKimiK2Experts   = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount = 512;   // max(512,384,256)
static constexpr int MaxNumExpertsUnit  = 128;        // 无组路径的单 warp 处理上限
static constexpr int NumTopGroupScores  = 2;          // 每组取 top-2
static constexpr int DefaultMaxNumTopExperts = 8;     // 默认最大 topk
static constexpr int MaxSupportedTopExperts  = 22;    // Nemotron 特化
static constexpr int MaxNumTopGroups    = 4;          // topk_group 上限

// 实例化的类型组合（scores_dtype × bias_dtype × idx_dtype × scoring_func）
// SCORING_SIGMOID: {float,half,bfloat16} × {float,half,bfloat16} × int32_t
// SCORING_NONE:    {float,half,bfloat16} × {float,half,bfloat16} × int32_t
```

---

*文档生成时间：2026-04-29*
*参考文件：*
- `FastDeploy/fastdeploy/model_executor/layers/moe/moe.py:81~146`
- `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:29~162`
- `vllm/_custom_ops.py:2544~2580`
- `vllm/csrc/moe/grouped_topk_kernels.cu`（全文）
- `vllm/csrc/moe/torch_bindings.cpp:128~134`
