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

## 代码复核结论

### `topk_with_k2_kernel` 内嵌调用（L528-535）

**结论：无冗余计算，无竞争写，逻辑正确。**

- `topk_with_k2_kernel` 是 `__device__` 函数，参数**值传递**，内部指针偏移不影响调用方变量
- `token_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id`，每个 warp 只写自己的 token，无竞争
- `cg::tiled_partition<32>` 的 tile 只覆盖当前 warp 的 32 个线程

### Shared Memory Bank Conflict（L544-553）

**结论：无 bank conflict 问题。**

- WarpSelect 阶段：`warp_id * WARP_SIZE + lane_id`，32 lane 访问 32 个连续元素，完美覆盖 32 个 bank
- `s_topk_idx/s_topk_value` 阶段：同一 warp 内连续访问，无 conflict；不同 warp 间访问时间独立，bank conflict 概念不适用

---

## 优化方案

### P1-A：`n_group` 模板特化 + Dispatch

#### 现状问题

**代码位置**：L529

```cpp
T group_scores_tmp[8];  // 硬编码为 8：n_group=1 时浪费寄存器，n_group>8 时越界
```

#### `n_group` 取值特性（已确认）

- 来源：模型 `config.json`，启动时一次性加载，**运行时不会动态修改**
- 当前已知取值：`1`（GLM-4.5-Air）、`8`（DeepSeek-V3/R1）
- 实践中均为 2 的幂次，可安全做模板特化

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
  group_scores      = group_scores_tmp;

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
  DISPATCH_NGROUP_KERNEL(
      group_idx_and_topk_idx_kernel, T, IdxT, n_group,
      num_blocks, BLOCK_SIZE, smem_bytes, stream,
      scores, group_scores, topk_values, topk_indices, scores_with_bias,
      num_tokens, topk_group, topk, num_experts, num_experts / n_group,
      renormalize, routed_scaling_factor);
#else
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

#### 预期收益

| kNGroup | 适用模型 | 收益 |
|---------|---------|------|
| 1 | GLM-4.5-Air | 分组循环完全消除，寄存器精确为 1 个 |
| 8 | DeepSeek-V3/R1 | 循环 unroll，寄存器精确为 8 个 |
| 2/4/16 | 未来模型 | 同上，防止越界 |

`topk_with_k2_kernel` 内部 `for (loc_group_idx < kNGroup)` 循环在编译期常量下被完全展开。

#### 注意事项

- `group_idx_and_topk_idx_kernel`：`group_scores` 参数保留（加 `[[maybe_unused]]`），保持 caller 接口不变
- `group_idx_and_topk_idx_redundant_kernel`：不受影响，仍从外部接收 `group_scores`

---

### P1-B：`routed_scaling_factor` double → float

**代码位置**：L522, L668-671

```cpp
double routed_scaling_factor   // 当前：double 参数
// 实际只做 float 精度的乘法运算
value = cuda_cast<float, T>(s_topk_value[i]) / topk_sum * routed_scaling_factor;
```

GPU 上 double 算术吞吐比 float 低 8-32x，精度需求为 float 级别。

**方案**：将 kernel 参数类型改为 `float`（`invokeNoAuxTc` 的调用方已经传 `float`）

---

### P1-C：`renormalize` 模板特化

**代码位置**：L667-672

```cpp
if (renormalize) {          // 运行时分支，值固定但每次执行都判断
    value = ... / topk_sum * routed_scaling_factor;
} else {
    value = ... * routed_scaling_factor;
}
```

`renormalize` 在推理中为常量，可提升为模板参数消除运行时分支：

```cpp
template <typename T, typename IdxT, int kNGroup, bool kRenormalize>
__global__ void group_idx_and_topk_idx_kernel(...) {
    if constexpr (kRenormalize) {
        value = ... / topk_sum * routed_scaling_factor;
    } else {
        value = ... * routed_scaling_factor;
    }
}
```

dispatch 宏在 P1-A 基础上再加一层 `renormalize` 的 bool 特化（2×5=10 个特化版本）。

---

### P3-A：`topk_sum` 负值防护

**代码位置**：L639, L668

`topk_sum` 初始化为 `1e-20` 用于防除零，但若 scores 出现负值（异常输入），`topk_sum` 可能仍小于 `1e-20`。

**方案**：renormalize 前加一行守卫

```cpp
topk_sum = max(topk_sum, 1e-20f);
```

---

### P3-B：`isfinite` 位操作替代

**代码位置**：L614, L619

```cpp
isfinite(cuda_cast<float, T>(scores_with_bias[offset + i]))  // 每次隐式转换 bf16→float
```

对 `T=__nv_bfloat16` 可直接用位操作判断，避免转换：

```cpp
inline __device__ bool is_finite_bf16(__nv_bfloat16 val) {
    // bfloat16: sign(1) + exp(8) + mantissa(7)，exp 全1 即 inf/nan
    return (__bfloat16_as_ushort(val) & 0x7F80) != 0x7F80;
}
```

---

## 优化优先级汇总

| 优先级 | 方案 | 预期收益 | 实现难度 |
|--------|------|---------|---------|
| **P1** | P1-A：`n_group` 模板特化 + dispatch | 寄存器精确分配 + 循环 unroll + 防越界 | 中 |
| **P1** | P1-B：`routed_scaling_factor` double→float | 算术吞吐提升 | 低 |
| **P1** | P1-C：`renormalize` 模板特化 | 消除运行时分支 | 低 |
| **P3** | P3-A：`topk_sum` 负值防护 | 鲁棒性提升 | 低 |
| **P3** | P3-B：`isfinite` 位操作替代 | 小幅性能提升 | 低 |

P1-A/B/C 可捆绑为一个 PR，P3 单独处理。

---

*方案持续更新中...*
