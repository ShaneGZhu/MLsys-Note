# NoAuxTC 端到端融合设计：sigmoid → bias_add → topk routing

> 文档基于 `custom_ops/gpu_ops/noauxtc_kernel.h` / `noauxtc_kernel_dev.h`
> 与 `fastdeploy/model_executor/layers/moe/moe.py` 的 `get_moe_scores` 函数。

---

## 1. 背景：现状与问题

### 1.1 当前调用链

```
Python (get_moe_scores)
  ①  scores           = sigmoid(gating_output)       # Paddle kernel, 写 global [N,E]
  ②  scores_with_bias = scores + bias                # Paddle kernel, 写 global [N,E]
  ③  noaux_tc(scores, scores_with_bias, ...)
        └─ invokeNoAuxTc (noauxtc_kernel.h)
              ├─ topk_with_k2_kernel                  # 读 scores_with_bias
              └─ group_idx_and_topk_idx_kernel        # 读 scores_with_bias (再次)
                                                      # 读 scores (gather topk_values)
                                                      # 写 topk_values, topk_indices
```

### 1.2 问题量化（T=128, E=256, fp32）

| 指标 | 值 |
|---|---|
| kernel launch 次数 | 3（sigmoid / bias_add / noaux_tc） |
| global mem 读 | gating×1 + scores×2 + sbias×2 = **5 × [N,E]** |
| global mem 写 | scores×1 + sbias×1 + topk_values + topk_idx = **2 × [N,E]** + 小 |
| 中间 buffer | scores \[N,E\] + scores_with_bias \[N,E\] = **2 × 128KB = 256KB** |
| 总 launch overhead（实测） | **~25 µs**（kernel launch dominated） |

`scores` 和 `scores_with_bias` 写入 global memory 后立刻被 noaux_tc 读回，
两次 round-trip 是纯浪费。

### 1.3 `noauxtc_kernel_dev.h` vs `noauxtc_kernel.h`

| 项 | `noauxtc_kernel.h` | `noauxtc_kernel_dev.h` |
|---|---|---|
| topk_with_k2 | 独立 kernel，单独 launch | **内联进** `group_idx_and_topk_idx_kernel` |
| kernel 数（noaux_tc 阶段） | 2 | **1** |
| `griddepcontrol` (SM90) | 有 | 有 |
| sigmoid / bias_add | 外部计算，传入已算好 fp32 | 同左，尚未融合 |

`_dev.h` 已完成 topk_with_k2 → group_idx 的内核合并，
本文档讨论在此基础上进一步融合 sigmoid 与 bias_add。

---

## 2. 方案 A：最小改动——在 kernel 入口重算 sigmoid+bias

### 2.1 思路

把 `group_idx_and_topk_idx_kernel` 的输入从
`(scores*, scores_with_bias*)` 改为 `(gating_output*, bias*)`，
在 kernel 的 prologue 阶段（per-lane 遍历整行时）直接计算：

```cuda
// 在 warp 的 prologue 循环中，将原来的：
//   T sbias = scores_with_bias[case_id * num_experts + i];
// 改为：
float g   = gating_output[case_id * num_experts + i];
float s   = 1.f / (1.f + __expf(-g));   // sigmoid，register only
float sb  = s + bias[i];                 // bias_add，register only
// 直接送进 topk_with_k2 与 WarpSelect，不再写 global
```

**gather topk_values 问题**：第 637 行（`noauxtc_kernel_dev.h`）：
```cuda
T value = scores[s_topk_idx[i]];  // unbiased sigmoid score
```
若 `scores` 不再写 global，改为对 topk=8 个索引重算 sigmoid：
```cuda
float g_topk = gating_output[case_id * num_experts + s_topk_idx[i]];
T value      = 1.f / (1.f + __expf(-g_topk));   // 仅 8 次 __expf，代价极小
```

### 2.2 收益

- 消除 sigmoid + bias_add 两次 kernel launch（省 ~12 µs）
- 消除 `scores_with_bias[N,E]` global buffer（节省 `N×E×4` 字节）
- `scores[N,E]` 仍需保留（后续 gather 用），或改为重算节省该 buffer

### 2.3 局限

- 对 `topk_with_k2`（group_scores 阶段）无法利用 smem 缓存——sbias 在 register 中是一过性的，
  如果 n_group > 1 且 num_experts_per_group > WARP_SIZE，需要多次 load gating_output（每个 group 的 lane 扫描），
  导致 gating_output 被读 n_group 遍（vs 当前 sbias 读 n_group 遍），两者等价。
- `scores[N,E]` 如果改为重算，则 gating_output 被读两遍（group_scores 阶段 + gather 阶段），
  不如方案 C 用 smem 缓存来得彻底。

---

## 3. 方案 B：两阶段 kernel，sigmoid+bias+topk_with_k2 合并

### 3.1 思路

将三个 kernel 重组为两个：

```
Kernel 1（新）：sigmoid_bias_groupscore_kernel
  输入：gating_output [N,E]，bias [1,E]
  输出：scores [N,E]（unbiased，写 global），group_scores [N,G]
  每个 warp 处理一个 token：
    - 遍历列计算 sigmoid → 写 scores[row]
    - 同时 sigmoid+bias → 纯 register，喂入 topk_with_k2 warp reduce → 写 group_scores[row]
    - scores_with_bias 完全不落 global

Kernel 2（几乎不变）：group_idx_and_topk_idx_kernel
  输入：scores [N,E]，group_scores [N,G]
  WarpSelect 读 sbias 的部分改为从 gating_output 重算（仅 topk_group×num_experts_per_group 个元素）
  gather topk_values 读 scores（unbiased，已在 Kernel 1 写好）
```

### 3.2 收益 vs 方案 A

- global 写操作：scores×1（Kernel 1）+ topk_values + topk_idx（Kernel 2）
- `scores_with_bias[N,E]` buffer 彻底消除
- Kernel 2 中 WarpSelect 阶段读 sbias 改为重算，仅读 gating_output 中 topk_group 个 group 的元素（E/n_group × topk_group 次），对 n_group=8、topk_group=4 相当于原来的一半读量

### 3.3 局限

- 两阶段之间 `scores[N,E]` 仍需落 global（Kernel 1 → Kernel 2 的数据传递）
- 仍有 2 次 kernel launch

---

## 4. 方案 C（推荐）：单 kernel 全流水，消除全部中间 buffer

### 4.1 核心思想

在 `_dev.h` 已合并 topk_with_k2 的基础上，
把 sigmoid + bias_add 一并内联，
并用 **shared memory 缓存整行 unbiased scores**，
最终实现 **1 次 kernel launch、gating_output 只读 1 遍、无中间 global buffer**。

### 4.2 smem 布局设计

参数基线（SM90，`BLOCK_SIZE=512`，`NUM_WARPS_PER_BLOCK=16`）：

```
E=256, topk=8, n_group=8, topk_group=4
```

| 区域 | 大小 | 说明 |
|---|---|---|
| `smem_scores[NUM_WARPS × E]` | 16 × 256 × 4B = **16 KB** | 缓存每 warp/token 的 unbiased sigmoid scores |
| `WarpSelect staging`（val+idx） | 16 × 32 × (4+4)B = **4 KB**（已有） | WarpSelect 的 smem staging buffer |
| `s_topk_idx[NUM_WARPS × topk]` | 16 × 8 × 4B = **512 B**（已有） | topk 专家索引 |
| `s_topk_value[NUM_WARPS × topk]` | 16 × 8 × 4B = **512 B**（已有） | topk 专家分值 |
| **合计** | **~21 KB** | SM90 smem 上限 228 KB，剩余充裕 |

> E=128 时 smem_scores 缩减为 8 KB，总计 ~13 KB，更宽松。

### 4.3 单 kernel 执行流程

每个 warp 负责一个 token（`case_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id`）：

```
Phase 0：Load & sigmoid & bias_add（一次遍历完成三件事）
──────────────────────────────────────────────────────
for (int i = lane_id; i < num_experts; i += WARP_SIZE) {
    float g  = gating_output[case_id * E + i];
    float s  = 1.f / (1.f + __expf(-g));        // sigmoid
    float sb = s + bias[i];                      // bias_add（bias 可从 const mem 读）
    smem_scores[warp_id * E + i] = s;            // 缓存 unbiased score → smem
    sbias_reg[i/WARP_SIZE] = sb;                 // sbias 留在 register file
    // 同时喂进 topk_with_k2 累积器（见 Phase 1）
}
// 此阶段：gating_output 只读一次，scores/scores_with_bias 无 global 写
```

```
Phase 1：topk_with_k2（group_scores）— 与 Phase 0 合并到同一遍历
──────────────────────────────────────────────────────────────────
// 在 Phase 0 的 for 循环中，按 group 维度并行累积 top-2 biased scores
// （与 _dev.h 中 topk_with_k2_kernel 逻辑相同，但输入从 smem_sbias 取）
// 结果写入 group_scores_reg[n_group]（register，不写 global）
```

```
Phase 2：选 topk_group 个 group — 与 _dev.h 逻辑完全一致
──────────────────────────────────────────────────────────
// warp reduce 找 topk_group_value
// 确定 group_mask（哪些 group 参与 WarpSelect）
// group_scores 全在 register，无 global 读写
```

```
Phase 3：WarpSelect topk 个 expert（biased scores）
────────────────────────────────────────────────────
// 对 topk_group 个 selected group：
for (int i_group : selected_groups) {
    int offset = i_group * experts_per_group;
    for (int i = lane_id; i < align_epg; i += WARP_SIZE) {
        // sbias 从 register file 取（Phase 0 已存入 sbias_reg）
        // 若 register 溢出，改为从 gating_output 重算 sigmoid+bias
        //   代价：topk_group/n_group × E 次 __expf = 4/8 × 256 = 128 次
        float sb = sbias_reg[offset/WARP_SIZE + i/WARP_SIZE]; // 或重算
        queue.add(sb, offset + i);
    }
}
queue.done();
queue.dumpIdx(s_topk_idx);   // 写 smem
```

```
Phase 4：gather topk_values（unbiased scores）
──────────────────────────────────────────────
// 从 smem_scores 读（Phase 0 已写好），无 global 读
for (int i = lane_id; i < topk; i += WARP_SIZE) {
    T value = smem_scores[warp_id * E + s_topk_idx[i]];
    s_topk_value[i] = value;
    topk_sum += value;
}
```

```
Phase 5：renormalize + scale，写输出
─────────────────────────────────────
for (int i = lane_id; i < topk; i += WARP_SIZE) {
    float v = s_topk_value[i];
    if (renormalize) v = v / topk_sum * routed_scaling_factor;
    else             v = v * routed_scaling_factor;
    topk_values[i]  = v;                     // 写 global（小，topk=8）
    topk_indices[i] = s_topk_idx[i];         // 写 global（小，topk=8）
}
// scores[N,E] 的稀疏写（原 kernel 的 "trick"）同样可消除，
// 若下游不再使用 scores 稀疏矩阵，直接删去
```

### 4.4 register file 压力分析

Phase 0 每个 lane 需要在 register 中同时持有：
- `sbias_reg[E/WARP_SIZE]` = `256/32 = 8` 个 float（供 Phase 3 WarpSelect 使用）
- `group_scores_reg[n_group]` = 8 个 float（group 累积结果）
- top-2 中间变量 `largest`, `second_largest`：2 个 float

合计约 **18 个 float per lane**（72 B），加上 CUDA 编译器本身的临时变量，
总 register 用量预估 **~32-40 register/thread**，远低于 SM90 的 255 上限，
不会触发 register spill。

> 若 E > 256（如 E=512），`sbias_reg` 增到 16 个 float，仍在安全范围内。
> 若 n_group > 8，`group_scores_reg` 相应增加，需重新评估。

### 4.5 bias 的读取策略

`bias[1, E]` 在所有 token 间共享，有两种加速读取的方式：

| 策略 | 说明 | 适用场景 |
|---|---|---|
| `__constant__ memory` | E≤256 (1KB)，broadcast 读，无 L1 竞争 | E≤65535（constant mem 上限 64KB）|
| L1 cache 自然命中 | 多 warp 顺序读同一行，L1 line 复用 | E 任意，默认推荐 |
| smem 预加载 | block 内 thread 协作加载 bias → smem，再广播 | E 较大、L1 miss 率高时 |

对 E=128/256 的 MoE 配置，**L1 cache 自然命中**即可（bias 行仅 512B/1KB），
不需要额外 constant memory 声明。

### 4.6 新 kernel 签名

```cuda
template <typename T, typename IdxT>
__global__ void noaux_tc_fused_kernel(
    const T* __restrict__ gating_output,   // [N, E]  输入，只读一次
    const T* __restrict__ bias,            // [1, E]  bias，只读一次
    T*       __restrict__ topk_values,     // [N, topk]  输出
    IdxT*    __restrict__ topk_indices,    // [N, topk]  输出
    // 可选：sparse scores 输出（若下游需要）
    // T* __restrict__ scores_sparse,      // [N, E]  稀疏 scores
    int64_t const num_tokens,
    int64_t const num_experts,
    int64_t const n_group,
    int64_t const topk_group,
    int64_t const topk,
    bool    const renormalize,
    double  const routed_scaling_factor
);
```

Python 侧调用简化为：
```python
# 之前
scores = paddle.nn.functional.sigmoid(gating_output)
scores_with_bias = scores + e_score_correction_bias
scores_out, topk_values, topk_idx = noaux_tc(
    scores, scores_with_bias, n_group, topk_group, top_k, renormalize, scale
)

# 融合后
topk_values, topk_idx = noaux_tc_fused(
    gating_output, e_score_correction_bias,
    n_group, topk_group, top_k, renormalize, scale
)
```

### 4.7 与 `_dev.h` 的对比：改动 delta

```diff
- 函数参数：(scores*, scores_with_bias*, ...)
+ 函数参数：(gating_output*, bias*, ...)

- Phase 0（现无）：直接使用传入的 scores/scores_with_bias

+ Phase 0（新增）：
+   for lane in [0, E):
+       s  = sigmoid(gating_output[row*E + lane])
+       sb = s + bias[lane]
+       smem_scores[warp*E + lane] = s          // smem 写
+       sbias_reg[lane/32] = sb                  // register 保留

  Phase 1（topk_with_k2）：
-   input = scores_with_bias[row]（global 读）
+   input = sbias_reg（register 直接用）

  Phase 3（WarpSelect）：
-   candidates = scores_with_bias[offset + i]（global 读）
+   candidates = sbias_reg[offset/32 + i/32]（register 读）

  Phase 4（gather topk_values）：
-   value = scores[s_topk_idx[i]]（global 读）
+   value = smem_scores[warp*E + s_topk_idx[i]]（smem 读）

- smem 新增：smem_scores[NUM_WARPS * E * sizeof(T)]
```

### 4.8 预期性能收益

以 DeepSeek-V3 配置（T=128, E=256, n_group=8, topk=8）为例：

| 指标 | 当前（3 kernels） | 方案 C（1 kernel） | 节省 |
|---|---|---|---|
| kernel launch 数 | 3 | **1** | -2 |
| global mem 读量 | 5 × 128KB | **1 × 128KB** | **-80%** |
| global mem 写量 | 2 × 128KB + 8KB | **~8KB** | **-97%** |
| 中间 buffer | 256 KB | **0** | -256 KB |
| 显存占用（routing） | N×E×8B（scores+sbias） | **0** | N×E×8B |
| 预估总延迟 | ~25 µs | **~13 µs** | ~12 µs |

> 注：预估基于实测 kernel launch overhead ~12 µs/launch，
> 融合后单 kernel 的计算量不变，IO 大幅减少，瓶颈转为 gating_output 的单次读取。

---

## 5. 实施建议

### 5.1 优先级与风险

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 验证 `_dev.h` 在 SM90 上的正确性 | 低 | 低（有测试覆盖） |
| 方案 A：kernel 入口替换 sigmoid+bias | 中 | 低（仅改参数传递 + 重算 8 次 sigmoid） |
| 方案 C：增加 smem_scores 区域 | 中 | 中（需校验 smem 大小计算 + Phase 3 register spill 情况） |
| Python 侧接口更新（moe.py） | 低 | 低 |

### 5.2 测试策略

1. **精度测试**：对齐 `get_moe_scores` 的 Python 参考实现（见 `test_get_moe_scores.py`）
2. **性能测试**：用 `test_sigmoid_add_fused.py` 基准框架扩展，新增 `noaux_tc_fused` 对比项
3. **smem 溢出检查**：编译时断言 `sizeof(smem) <= 动态 smem 上限`，运行时用 `cudaFuncGetAttributes` 验证

### 5.3 约束条件

- 要求 E 能被 `WARP_SIZE(32)` 整除（现有配置均满足：128/256/512）
- `n_group ≤ WARP_SIZE(32)`（现有配置均满足：1/4/8）
- smem_scores 大小：`NUM_WARPS_PER_BLOCK × E × sizeof(T) ≤ 可用 smem`
  - SM90 动态 smem 最大 ~228KB，E=256 时 smem_scores=64KB，叠加已有 smem ~5KB，总计 ~69KB，仍在范围内
  - E=512：smem_scores=128KB，需降低 BLOCK_SIZE（减少 NUM_WARPS_PER_BLOCK）至 8 warps，smem_scores=64KB

---

## 6. 总结

| 方案 | kernel 数 | 中间 buffer | 主要改动 | 推荐场景 |
|---|---|---|---|---|
| 现状 | 3 | scores + sbias [N,E] | — | — |
| 方案 A | 1 | scores [N,E] | 参数替换 + 8 次重算 | 快速验证 |
| 方案 B | 2 | scores [N,E] | 新增 sigmoid_bias_groupscore kernel | 过渡方案 |
| **方案 C** | **1** | **无** | smem_scores + register sbias | **最终推荐** |

方案 C 在 SM90（H800/H100）上 smem 完全充裕，
实现 sigmoid → bias_add → group_scores → topk_group 选择 → WarpSelect → gather → renorm
的全流程单 kernel 化，
可将端到端 routing 延迟从 ~25 µs 压缩至 ~13 µs，同时消除 `N×E×8B` 的中间显存占用。
