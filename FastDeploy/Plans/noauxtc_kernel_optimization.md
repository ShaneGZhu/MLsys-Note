# NoAuxTc Kernel 优化方案

> 文件：`custom_ops/gpu_ops/noauxtc_kernel.h`
> 目标 Kernel：`group_idx_and_topk_idx_kernel`
> 更新时间：2026-04-28

---

## 背景

`group_idx_and_topk_idx_kernel` 是 DeepSeek MoE 路由中的核心 CUDA Kernel，负责：
1. 计算每个 token 在各 expert group 的 top2 得分（group score）
2. 选出 topk_group 个最优 group
3. 在选中的 group 内做 topk expert 选择
4. 可选 renormalize 后写出最终路由权重和索引

---

## 问题清单与优化方案

### P0（已复核）：`topk_with_k2_kernel` 内嵌调用分析

**代码位置**：L528-535

```cpp
topk_with_k2_kernel(
    group_scores, scores_with_bias,
    num_tokens, n_group, num_experts, num_experts_per_group
);
// 调用返回后，主 kernel 再做自己的指针偏移
scores_with_bias += case_id * num_experts;
group_scores     += case_id * n_group;
```

**结论：无冗余计算，无竞争写，逻辑正确。**

原因分析：
- `topk_with_k2_kernel` 是 `__device__` 函数，参数为**值传递**，函数内部 `input += offset` / `output += offset` 只修改局部栈上的指针副本，不影响调用方的 `group_scores` / `scores_with_bias` 变量
- 函数内部 `token_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id`，继承主 kernel 的上下文，每个 warp 只负责自己的 token，不存在多 warp 写同一位置的竞争
- 主 kernel 调用返回后，`group_scores` 仍指向全局基址，L538 的偏移从基址出发，完全正确
- `cg::tiled_partition<32>` 的 tile 只覆盖当前 warp 的 32 个线程，不控制跨 warp

**潜在关注点（非 bug）**：
- 注释掉的 2-kernel 方案（L889-918）原本想将 group score 计算独立出去，当前折叠进主 kernel 是一种设计选择，节省了一次 kernel launch 开销
- SM90 以下无 `griddepcontrol`，同一 warp 内 device 函数调用本身是串行的，读写顺序有保证，无需额外同步

---

### P1：算术与分支优化

#### 问题 2：`routed_scaling_factor` 使用 `double` 传参

**代码位置**：L520, L663-666

```cpp
double routed_scaling_factor   // 函数参数

// 实际运算
value = cuda_cast<float, T>(s_topk_value[i]) / topk_sum * routed_scaling_factor;
```

**问题分析**：
- GPU 上 double 算术吞吐比 float 低 8-32x（视架构）
- 实际精度需求为 float 级别，`double` 参数只是被隐式转换使用

**优化方案**：将参数类型改为 `float`

---

#### 问题 3：`renormalize` 运行时分支未特化

**代码位置**：L662-667

```cpp
if (renormalize) {
    value = ... / topk_sum * routed_scaling_factor;
} else {
    value = ... * routed_scaling_factor;
}
```

**问题分析**：
- `renormalize` 在实际推理中几乎是常量，每次 kernel 调用值固定
- 造成 warp-level 的条件判断开销

**优化方案**：将 `renormalize` 提升为模板参数

```cpp
template <typename T, typename IdxT, bool kRenormalize>
__global__ void group_idx_and_topk_idx_kernel(...) {
    ...
    if constexpr (kRenormalize) {
        value = ... / topk_sum * routed_scaling_factor;
    } else {
        value = ... * routed_scaling_factor;
    }
}
```

---

### P2：内存访问优化

#### 问题 4：Shared Memory Bank Conflict（已复核：无问题）

**代码位置**：L544-553

smem 分两个阶段复用：

**阶段一 WarpSelect**（L596~L625）：
- `val_smem_[warp_id * WARP_SIZE + lane_id]`，32 lane 访问 32 个连续 int32，正好覆盖 32 个 bank，无 conflict

**阶段二 s_topk_idx/s_topk_value**（L625之后）：
- 同一 warp 内 lane 访问 `s_topk_idx[warp_id*topk + lane_id]`，是连续地址，无 conflict
- **不同 warp 之间不存在 bank conflict 概念**（各 warp 执行时间独立，bank conflict 只发生在同一 warp 内部的不同 lane 同时访问同一 bank 时）

**结论：当前代码无 bank conflict 问题，此条优化项取消。**

---

#### 问题 5：`scores` 数组清零开销（已评估，不采纳）

**代码位置**：L651-655

**评估结论**：优化方向（收窄清零范围、下沉到 caller）均存在以下问题：
- 收窄清零范围需要跟踪 group 选中状态，代码改动分散、可读性下降
- 省略清零需要确认 `moe_expert_dispatch` C++ 实现不读未写位置，链路较长、风险难以控制
- 性价比不足，**不采纳**

---

#### 问题 6：`isfinite` 检查有隐式转换

**代码位置**：L614

```cpp
isfinite(cuda_cast<float, T>(scores_with_bias[offset + i]))
```

**优化方案**：
- 对 `T=__nv_bfloat16`，直接用位操作判断 exponent 全1：
  ```cpp
  // bfloat16: sign(1) + exp(8) + mantissa(7)
  // inf/nan 判断：exp 全1 即 0x7F80
  inline __device__ bool is_finite_bf16(__nv_bfloat16 val) {
      return (__bfloat16_as_ushort(val) & 0x7F80) != 0x7F80;
  }
  ```

---

### P1：`n_group` 模板特化 + `invokeNoAuxTc` Dispatch

#### 现状

**代码位置**：L529

```cpp
T group_scores_tmp[8];  // 硬编码为 8，n_group=1 时浪费寄存器，n_group>8 时越界
topk_with_k2_kernel(group_scores_tmp, ...);
group_scores = group_scores_tmp;
```

已经用了寄存器数组的思路，但 `[8]` 是硬编码，存在两个问题：
- `n_group=1`（GLM-4.5-Air）时多占 7 个寄存器
- 未来 `n_group=16` 的模型会直接越界

#### `n_group` 的取值特性（已确认）

- 来源：模型 `config.json`，启动时一次性加载，**运行时不会动态修改**
- 当前已知取值：`1`（GLM-4.5-Air）、`8`（DeepSeek-V3/R1）
- 语义约束：`n_group` 必须整除 `num_experts`，实践中均为 2 的幂次

因此可以安全地将 `n_group` 提升为编译期模板参数。

#### 方案设计

**Step 1：kernel 增加 `kNGroup` 模板参数**

```cpp
template <typename T, typename IdxT, int kNGroup>
__global__ void group_idx_and_topk_idx_kernel(
    T* scores,
    T* group_scores,        // 保留接口兼容性，kernel 内部不再使用
    T* topk_values,
    IdxT* topk_indices,
    const T* scores_with_bias,
    int64_t const num_tokens,
    // n_group 参数删除，改为 kNGroup
    int64_t const topk_group,
    int64_t const topk,
    int64_t const num_experts,
    int64_t const num_experts_per_group,
    bool const renormalize,
    double routed_scaling_factor) {

  // 编译期已知大小 → 全部进寄存器，编译器可完全 unroll topk_with_k2 内循环
  T group_scores_tmp[kNGroup];

  topk_with_k2_kernel(group_scores_tmp, scores_with_bias,
                      num_tokens, kNGroup, num_experts, num_experts_per_group);
  __syncwarp();

  scores_with_bias += case_id * num_experts;
  scores           += case_id * num_experts;
  group_scores      = group_scores_tmp;   // 重定向到寄存器数组

  // 后续所有 n_group 替换为 kNGroup（编译器静态折叠）
  int32_t want_neg_inf_num = WARP_SIZE - kNGroup + topk_group;
  if (lane_id < kNGroup && ...) { value = group_scores[lane_id]; }
  int neg_inf_num = WARP_SIZE - kNGroup;
  for (int i_group = 0; i_group < kNGroup; i_group++) { ... }
}
```

**Step 2：dispatch 宏，覆盖 1~16 的 2 次幂**

```cpp
#define DISPATCH_NGROUP_KERNEL(KNAME, T, IdxT, N_GROUP, GRID, BLOCK, SMEM, STREAM, ...) \
  do {                                                                                    \
    switch (N_GROUP) {                                                                    \
      case 1:  KNAME<T, IdxT,  1><<<GRID, BLOCK, SMEM, STREAM>>>(__VA_ARGS__); break;   \
      case 2:  KNAME<T, IdxT,  2><<<GRID, BLOCK, SMEM, STREAM>>>(__VA_ARGS__); break;   \
      case 4:  KNAME<T, IdxT,  4><<<GRID, BLOCK, SMEM, STREAM>>>(__VA_ARGS__); break;   \
      case 8:  KNAME<T, IdxT,  8><<<GRID, BLOCK, SMEM, STREAM>>>(__VA_ARGS__); break;   \
      case 16: KNAME<T, IdxT, 16><<<GRID, BLOCK, SMEM, STREAM>>>(__VA_ARGS__); break;   \
      default:                                                                            \
        PADDLE_THROW("Unsupported n_group=%d, must be power-of-2 in [1,16]",             \
                     (int)(N_GROUP));                                                     \
    }                                                                                     \
  } while (0)
```

**Step 3：`invokeNoAuxTc` 改造**

```cpp
template <typename T, typename IdxT>
void invokeNoAuxTc(..., int64_t const n_group, ...) {
  int64_t num_blocks = (num_tokens - 1) / NUM_WARPS_PER_BLOCK + 1;
  size_t  smem_bytes = warp_topk::calc_smem_size_for_block_wide<T, int32_t>(
                           NUM_WARPS_PER_BLOCK, topk);

#ifdef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
  // METAX 路径：直接 <<<>>> 调用
  DISPATCH_NGROUP_KERNEL(
      group_idx_and_topk_idx_kernel, T, IdxT, n_group,
      num_blocks, BLOCK_SIZE, smem_bytes, stream,
      scores, group_scores, topk_values, topk_indices, scores_with_bias,
      num_tokens, topk_group, topk, num_experts, num_experts / n_group,
      renormalize, routed_scaling_factor);
#else
  // NVIDIA 路径：cudaLaunchKernelEx，需要函数指针，用 lambda 封装
  auto launch = [&](auto* kernel_ptr) {
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim          = num_blocks;
    config.blockDim         = BLOCK_SIZE;
    config.dynamicSmemBytes = smem_bytes;
    config.stream           = stream;
    attrs[0].id             = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = false;
    config.numAttrs = 1;
    config.attrs    = attrs;
    cudaLaunchKernelEx(&config, kernel_ptr,
        scores, group_scores, topk_values, topk_indices, scores_with_bias,
        num_tokens, topk_group, topk, num_experts, num_experts / n_group,
        renormalize, routed_scaling_factor);
  };
  switch (n_group) {
    case 1:  launch(&group_idx_and_topk_idx_kernel<T, IdxT,  1>); break;
    case 2:  launch(&group_idx_and_topk_idx_kernel<T, IdxT,  2>); break;
    case 4:  launch(&group_idx_and_topk_idx_kernel<T, IdxT,  4>); break;
    case 8:  launch(&group_idx_and_topk_idx_kernel<T, IdxT,  8>); break;
    case 16: launch(&group_idx_and_topk_idx_kernel<T, IdxT, 16>); break;
    default: PADDLE_THROW("Unsupported n_group=%d", (int)n_group);
  }
#endif
}
```

#### 编译产物与预期收益

| kNGroup | 适用模型 | group_scores_tmp 寄存器 | 额外收益 |
|---------|---------|------------------------|---------|
| 1 | GLM-4.5-Air | 1 个 | 分组循环完全消除，编译器折叠为 if/直通 |
| 8 | DeepSeek-V3/R1 | 8 个 | 8 次全局内存广播读 → 寄存器；循环被 unroll |
| 2/4/16 | 未来模型预备 | 2/4/16 个 | 同上 |

`topk_with_k2_kernel` 内部的 `for (loc_group_idx < kNGroup)` 循环在编译期常量下被 `#pragma unroll` 完全展开，循环控制开销也消除。

#### `group_scores` 参数的处理

- `group_idx_and_topk_idx_kernel`：保留参数但内部不再使用（加 `[[maybe_unused]]` 注释），保持 caller 接口不变
- `group_idx_and_topk_idx_redundant_kernel`：不受影响，仍从外部接收 `group_scores`

---

### P3：`topk_sum` 累加逻辑检查

**代码位置**：L634-646

```cpp
float topk_sum = 1e-20;
for (int i = lane_id; i < round_up_to_multiple_of<WARP_SIZE>(topk); i += WARP_SIZE) {
    T value = i < topk ? scores[s_topk_idx[i]] : 0.0f;
    ...
    topk_sum += cg::reduce(tile, cuda_cast<float, T>(value), cg::plus<float>());
}
```

**问题分析**：
- `cg::reduce` 返回 warp-level 求和，每个 lane 都加完整 reduce 结果 → 单轮（`topk <= 32`）时各 lane 的 `topk_sum` 相同，结果正确
- 若 scores 中含负值（理论上不应该，但异常输入时），`topk_sum` 可能 < `1e-20`，renormalize 结果异常放大

**优化方案**：renormalize 前增加守卫：
```cpp
topk_sum = max(topk_sum, 1e-20f);
```

---

## 优化优先级汇总

| 优先级 | 问题 | 预期收益 | 难度 |
|--------|------|---------|------|
| **P1** | `n_group` 模板特化 + dispatch（1/2/4/8/16） | 寄存器精确分配 + 循环 unroll + 消除全局内存读 | 中 |
| **P1** | `routed_scaling_factor` double→float | 算术吞吐提升 | 低 |
| **P1** | `renormalize` 模板特化 | 消除分支 | 低 |
| ~~P2~~ | ~~smem bank conflict padding~~ | ~~已复核：无 bank conflict，此项取消~~ | - |
| ~~P2~~ | ~~scores 清零优化~~ | ~~已评估：风险高、可读性下降，不采纳~~ | - |
| **P3** | `isfinite` 位操作替代 | 小幅 | 低 |
| **P3** | `topk_sum` 防护 | 鲁棒性 | 低 |

---

## 待讨论事项

- [x] `n_group` 动态性确认：与 model config 绑定，运行时不变，可安全做模板特化
- [x] P0 `topk_with_k2_kernel` 冗余调用：已复核，逻辑正确，指针值传递，无竞争写，无冗余计算
- [ ] P0 拆分方案：是用 2-kernel + stream dependency，还是用 `cudaLaunchKernelEx` programmatic serialization？
- [ ] smem 生命周期分析：WarpSelect 和 `s_topk_idx/s_topk_value` 是否真的可以复用同一块 smem？
- [ ] `scores` 清零是否可以下沉到 caller 侧？

---

*方案持续更新中...*
