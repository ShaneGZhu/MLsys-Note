# Warp TopK 工具参考手册 (Warp TopK Tool Reference)

> **来源**: vLLM `grouped_topk_kernels.cu` (vllm::moe::warp_topk namespace)
> **版本**: 基于 v1.3.0rc2 TensorRT-LLM 改编
> **文档日期**: 2026-05-09

---

## 目录

1. [概述 (Overview)](#1-概述-overview)
2. [核心组件 (Core Components)](#2-核心组件-core-components)
3. [深度原理详解 (Deep Dive: Principles)](#3-深度原理详解-deep-dive-principles)
4. [API 详细参考 (Detailed API Reference)](#4-api-详细参考-detailed-api-reference)
5. [使用示例 (Usage Examples)](#5-使用示例-usage-examples)
6. [依赖和兼容性 (Dependencies & Compatibility)](#6-依赖和兼容性-dependencies--compatibility)
7. [实际应用案例 (Real-world Use Cases)](#7-实际应用案例-real-world-use-cases)
8. [最佳实践 (Best Practices)](#8-最佳实践-best-practices)

---

## 1. 概述 (Overview)

`warp_topk` 命名空间提供了一套高效、编译期优化的 **warp 级排序和 Top-K 选择**工具。这些工具专门为 CUDA warp（32 个线程）设计，利用：
- 双调排序算法（Bitonic Sort）
- warp shuffle 指令（`__shfl_xor_sync`）
- Cooperative Groups API
- 静态模板编译优化

### 核心功能

| 功能 | 描述 |
|------|------|
| **WarpSort** | 全排序，将整个 warp 内的元素排序 |
| **WarpSelect** | Top-K 选择，只保留最大的 K 个元素（按值或按值+索引）|
| **BitonicSort/Merge** | 底层双调排序网络，可用于任意 2ⁿ 元素排序 |

### 支持的操作

- 排序方向：升序（`ascending=true`）/ 降序（`greater=true`）
- 稳定性：稳定排序（相等元素保持原始顺序）/ 不稳定排序
- 索引伴随：可同时排序值和对应的索引

---

## 2. 核心组件 (Core Components)

```
namespace warp_topk {
    // 工具函数
    round_up_to_multiple_of()
    isPowerOf2()
    is_better_than()

    // 双调排序模板
    BitonicMerge<capacity, ascending, reverse, T, idxT, is_stable>
    BitonicSort<capacity, ascending, T, idxT, is_stable>

    // 主要类
    WarpSort<capacity, greater, T, idxT, is_stable>
    WarpSelect<capacity, greater, T, idxT, is_stable>
}
```

### 2.1 工具函数 (Utility Functions)

| 函数 | 功能 |
|------|------|
| `round_up_to_multiple_of<size>(len)` | 将 len 向上取整到 size 的倍数 |
| `isPowerOf2(v)` | 判断 v 是否是 2 的幂 |
| `is_better_than<greater>(val, baseline)` | 比较函数，支持稳定排序（相等时比较索引）|

### 2.2 双调排序模板 (Bitonic Sort Templates)

双调排序是一种比较排序算法，特别适合并行硬件实现：
- `BitonicMerge`: 将双调序列归并为有序序列
- `BitonicSort`: 将任意序列排序为双调/有序序列

### 2.3 WarpSort 类

基于双调排序的 warp 级全排序器。

### 2.4 WarpSelect 类

基于 WarpSort 的 Top-K 选择器，支持增量添加元素，适用于流式场景。

---

## 3. 深度原理详解 (Deep Dive: Principles)

### 3.1 双调排序算法 (Bitonic Sort)

#### 数学定义

**双调序列 (Bitonic Sequence)**：一个序列先单调递增后单调递减，或循环旋转后满足此性质。

```
有效双调序列示例：
1, 3, 5, 7, 8, 6, 4, 2  (先升后降)
7, 8, 6, 4, 2, 1, 3, 5  (旋转后仍为双调)

非双调序列：
1, 5, 2, 8, 3, 6, 4, 7  (不满足单调性质)
```

#### 算法流程

双调排序通过递归构造和归并双调序列来实现排序：

```
BitonicSort(n):
    if n == 1: return
    BitonicSort(n/2, ascending)      // 前半部分升序
    BitonicSort(n/2, !ascending)     // 后半部分降序 → 得到双调序列
    BitonicMerge(n, ascending)       // 归并为有序序列
```

#### 归并网络 (Bitonic Merge Network)

对于 8 元素（2³）的归并网络：

```
Stage 1 (stride=4):  [0↔4, 1↔5, 2↔6, 3↔7]
Stage 2 (stride=2):  [0↔2, 1↔3, 4↔6, 5↔7]
Stage 3 (stride=1):  [0↔1, 2↔3, 4↔5, 6↔7]
```

每 stage 的比较是独立的，可以并行执行。

#### 算法复杂度

- **比较次数**: O(log² n)
- **深度**: O(log² n) 比较层级
- **并行度**: n/2 个比较可同时执行

对于 n=32（warp size）：
- 总比较次数 = 5 × (5+1) / 2 = 80
- 深度 = 5 stages × 5 sub-stages = 25 级比较

#### Warp 级并行优化策略

在 warp 环境下，每个 lane 执行一个比较：

```cuda
// Stage 1: 比较 lane i 与 lane (i ^ 4)
other_val = __shfl_xor_sync(mask, my_val, 4);
if (my_val > other_val && ascending) {
    my_val = other_val;  // 交换
}
```

`__shfl_xor_sync` 实现无 memory 访问的数据交换，非常高效。

---

### 3.2 Warp 级并行机制

#### __shfl_xor_sync 原理

Warp Shuffle 指令允许线程间直接交换寄存器值，无需经过 shared memory：

```cuda
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=WARP_SIZE);
```

| 参数 | 说明 |
|------|------|
| `mask` | 参与交换的线程掩码（通常 0xffffffff）|
| `var` | 要交换的变量 |
| `laneMask` | XOR 掩码，确定交换目标线程 |
| `width` | 子 warp 大小（通常 32）|

**工作机制**：

```
lane 0: laneMask=4 → 与 lane 4 交换
lane 1: laneMask=4 → 与 lane 5 交换
lane 4: laneMask=4 → 与 lane 0 交换
...
```

#### Cooperative Groups 接口

CG 提供更高层的抽象：

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

auto tile = cg::tiled_partition<32>(block);  // 创建 32 线程的 group
auto reduced = cg::reduce(tile, value, cg::plus<float>{});  // 规约操作
```

#### Shared Memory 暂存策略

WarpSelect 使用 shared memory 作为缓冲区：

```
+-----------------------+-----------------------+
|   val_smem_ (32 T)   |   idx_smem_ (32 idx) |
|   (分数缓冲区)        |   (索引缓冲区)        |
+-----------------------+-----------------------+

工作流程：
1. warp 各 lane 收集 32 个候选 → 写入 val_smem_/idx_smem_
2. 对缓冲区进行 BitonicSort
3. 将排序结果与当前 Top-K 合并
4. 重复步骤 1-3
```

#### 线程间数据交换模式

双调排序中的比较-交换模式：

```
Stride = 16 (最大比较距离)
[0]↔[16], [1]↔[17], ..., [15]↔[31]

Stride = 8
[0]↔[8], [1]↔[9], ..., [7]↔[15], [16]↔[24], ..., [23]↔[31]

Stride = 4
...

Stride = 2
...

Stride = 1 (最终排序)
[0]↔[1], [2]↔[3], ..., [30]↔[31]
```

---

### 3.3 Top-K 选择算法

#### 维护 K-th 元素阈值策略

WarpSelect 持续维护当前第 K 大的元素（`k_th_`）：

```
策略：
1. 新元素 value 与 k_th_ 比较
2. value > k_th_ 则保留，否则丢弃
3. 保留的元素积累到 32 个后归并进 Top-K
4. 更新 k_th_ = Top-K 中最小（或第 K 个）元素
```

#### 动态缓冲区管理

```cpp
__device__ void add(T val, idxT idx) {
    // 步骤 1: 决定是否保留
    bool do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);

    // 步骤 2: 使用 ballot 统计保留数量
    uint32_t mask = __ballot_sync(FULL_WARP_MASK, do_add);
    if (mask == 0) return;  // 全部丢弃

    // 步骤 3: 计算写入位置（前缀和）
    int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));

    // 步骤 4: 写入 shared memory
    if (do_add && pos < WARP_SIZE) {
        val_smem_[pos] = val;
        idx_smem_[pos] = idx;
    }

    // 步骤 5: 积累计数，满 32 个时归并
    smem_buf_len_ += __popc(mask);
    if (smem_buf_len_ >= WARP_SIZE) {
        merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
        smem_buf_len_ -= WARP_SIZE;
    }
}
```

#### 批量插入与归并优化

```cpp
// 批量添加数组中的元素
__device__ void add(T const* in, idxT start, idxT end) {
    idxT end_for_fullwarp = round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
        T val = (i < end) ? in[i] : dummy_;
        add(val, i);  // 复用单个 add 逻辑
    }
}
```

---

## 4. API 详细参考 (Detailed API Reference)

### 4.1 工具函数

#### round_up_to_multiple_of

```cuda
template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len);
```

**作用**: 将 len 向上取整到 size 的倍数。

**示例**:
```cuda
round_up_to_multiple_of<32>(50)   // → 64
round_up_to_multiple_of<256>(512)  // → 512
round_up_to_multiple_of<32>(0)    // → 0
```

**用途**: Shared memory 对齐、批量处理大小计算。

---

#### isPowerOf2

```cuda
template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v);
```

**作用**: 判断 v 是否是 2 的幂。

**实现**:
```cuda
return (v && !(v & (v - 1)));
```

**示例**:
```cuda
isPowerOf2(32)   // → true
isPowerOf2(24)   // → false
isPowerOf2(1)    // → false (bitwise trick，0 视为 false)
```

---

#### is_better_than

```cuda
template <bool greater, typename T>
__forceinline__ __device__ bool is_better_than(T val, T baseline);

template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val, T baseline,
                                               idxT index, idxT baseline_index);
```

**作用**: 比较两个值是否"更好"，支持稳定排序。

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `greater` | bool | true=降序（大的更好），false=升序（小的更好）|
| `val` | T | 待比较值 |
| `baseline` | T | 基准值 |
| `index` / `baseline_index` | idxT | 索引（仅稳定排序需要）|

**返回值**:
- 无索引版本：`val > baseline`（降序）或 `val < baseline`（升序）
- 有索引版本（稳定排序）：值相等时，索引更小的优先

---

### 4.2 BitonicMerge 模板

```cuda
template <int size, bool ascending, bool reverse, typename T,
          typename idxT, bool is_stable>
struct BitonicMerge {
    __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr);
};
```

**模板参数**:
| 参数 | 约束 | 说明 |
|------|------|------|
| `size` | 必须 ≥ 2×WARP_SIZE，且为 2 的幂 | 数组大小 |
| `ascending` | bool | true=升序，false=降序 |
| `reverse` | bool | 排序方向（用于递归调用）|
| `T` | 支持浮点/整数 | 值类型 |
| `idxT` | 整数类型 | 索引类型 |
| `is_stable` | bool | 是否稳定排序 |

**约束条件**:
```cuda
static_assert(isPowerOf2(size));
static_assert(size >= 2 * WARP_SIZE);
```

**使用场景**: 通常由 BitonicSort 调用，一般不直接使用。

---

### 4.3 BitonicSort 模板

```cuda
template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
    __device__ static void sort(T* __restrict__ val_arr,
                             idxT* __restrict__ idx_arr);
};
```

**模板参数**:
| 参数 | 约束 | 说明 |
|------|------|------|
| `size` | 必须 ≥ 2×WARP_SIZE，且为 2 的幂 | 数组大小 |
| `ascending` | bool | true=升序，false=降序 |
| `T` | 支持浮点/整数 | 值类型 |
| `idxT` | 整数类型 | 索引类型 |
| `is_stable` | bool | 是否稳定排序 |

**特殊化**: `size=32` 时有特殊化实现，使用 warp shuffle 优化。

---

### 4.4 WarpSort 类

#### 构造函数

```cuda
template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy);
};
```

**模板参数**:
| 参数 | 约束 | 说明 |
|------|------|------|
| `capacity` | 必须 ≥ WARP_SIZE 且为 2 的幂 | 最大排序容量 |
| `greater` | bool | true=降序，false=升序 |
| `T` | 值类型 | float, half, int 等 |
| `idxT` | 索引类型 | int32_t 等 |
| `is_stable` | bool | 是否稳定排序 |

**构造参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `k` | idxT | 要保留的元素数量（通常 `k = min(capacity, actual_n)`）|
| `dummy` | T | 初始填充值，用于无效数据 |

**内部结构**:
```cpp
static constexpr int max_arr_len_ = capacity / WARP_SIZE;  // 每线程管理的元素数
T val_arr_[max_arr_len_];   // 每线程的值数组
idxT idx_arr_[max_arr_len_]; // 每线程的索引数组
```

#### 公共方法

##### load_sorted

```cuda
__device__ void load_sorted(T const* __restrict__ in,
                           idxT const* __restrict__ in_idx,
                           idxT start);
```

**作用**: 加载已排序的 k 个元素并合并到当前 Top-K 中。

**参数**:
| 参数 | 说明 |
|------|------|
| `in` | 已排序的值数组 |
| `in_idx` | 对应的索引数组 |
| `start` | 起始位置 |

---

##### dump

```cuda
__device__ void dump(T* __restrict__ out, idxT* __restrict__ out_idx) const;
__device__ void dumpIdx(idxT* __restrict__ out_idx) const;
```

**作用**: 将结果写入输出数组。

---

##### get_val / get_idx

```cuda
__device__ __forceinline__ idxT get_idx(int i = 0) const;
__device__ __forceinline__ T get_val(int i = 0) const;
```

**作用**: 获取每线程负责的第 i 个元素。

**注意**: 对于 `capacity == WARP_SIZE`，`max_arr_len_ == 1`，应使用 `i == 0`。

---

### 4.5 WarpSelect 类

WarpSelect 继承自 WarpSort，支持增量添加元素。

#### 构造函数

```cuda
template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  __device__ WarpSelect(idxT k, T dummy);
};
```

**额外要求**: 需要 extern shared memory 作为缓冲区。

```cuda
extern __shared__ char smem_buf[];  // 必须在外部声明
```

#### 公共方法

##### add (单个元素)

```cuda
__device__ void add(T val, idxT idx);
```

**作用**: 添加单个元素到 Top-K 选择器。

**工作流程**:
1. 比较元素与当前 k-th 阈值
2. 如果更好，写入 shared memory 缓冲区
3. 缓冲区满 32 个元素时归并
4. 更新 k-th 阈值

---

##### add (批量)

```cuda
__device__ void add(T const* in, idxT start, idxT end);
```

**作用**: 批量添加数组 [start, end) 中的所有元素。

---

##### done

```cuda
__device__ void done();
```

**作用**: 处理缓冲区中剩余的元素，完成最终选择。

**必须调用**: 在添加完所有元素后调用，确保缓冲区清空。

---

##### 继承的方法

WarpSelect 继承 WarpSort 的所有方法：
- `load_sorted()`
- `dump()` / `dumpIdx()`
- `get_val()` / `get_idx()`

---

## 5. 使用示例 (Usage Examples)

### 5.1 简单排序示例 (WarpSort)

```cuda
__global__ void warp_sort_example(float* input, float* output, int n) {
    int lane = threadIdx.x % 32;

    // 每个线程加载一个值
    float my_val = (threadIdx.x < n) ? input[threadIdx.x] : -INFINITY;
    int my_idx = threadIdx.x;

    // 使用 WarpSort 对整个 warp 排序（32 个元素）
    warp_topk::WarpSort<32, /*greater*/ true, float, int, /*stable*/ false>
        sorter(32, -INFINITY);

    // 由于 capacity == WARP_SIZE，只需单次归并
    sorter.load_sorted(&my_val, &my_idx, 0);

    // 输出结果
    sorter.dump(output, reinterpret_cast<int*>(output + n));  // 注意：idx 存储在第二个数组
}
```

### 5.2 Top-K 选择示例 (WarpSelect)

```cuda
__global__ void topk_select_example(float* data, int n,
                                   float* topk_vals, int* topk_idxs, int k) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 每个 warp 处理一个 token 的专家选择
    float* warp_data = data + blockIdx.x * n;

    extern __shared__ char smem_buf[];
    warp_topk::WarpSelect<32, /*greater*/ true, float, int, /*stable*/ true>
        selector(k, -INFINITY);  // 稳定 Top-K

    // 批量添加所有元素
    selector.add(warp_data, 0, n);

    // 完成最终选择
    selector.done();

    // 输出 Top-K 结果
    if (lane < k) {
        topk_vals[blockIdx.x * k + lane] = selector.get_val(0);
        topk_idxs[blockIdx.x * k + lane] = selector.get_idx(0);
    }
}

// 启动配置
// grid: num_tokens
// block: 32 threads per warp
// dynamic smem: needed for WarpSelect
```

### 5.3 完整 kernel 示例 (包含 shared memory 计算)

```cuda
__global__ void grouped_topk_kernel(float* scores, float* bias,
                                   float* topk_values, int* topk_indices,
                                   int num_tokens, int num_experts,
                                   int n_group, int topk_group, int topk) {
    int32_t warp_id = threadIdx.x / 32;
    int32_t lane_id = threadIdx.x % 32;

    // 每个线程处理一个 expert 的分数
    int expert_per_group = num_experts / n_group;

    extern __shared__ char smem_buf[];
    float* s_group_scores = reinterpret_cast<float*>(smem_buf);

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    // Phase 1: 计算每组分数
    if (warp_id < n_group) {
        float group_score = 0.0f;
        float max1 = -INFINITY, max2 = -INFINITY;

        for (int i = lane_id; i < expert_per_group; i += 32) {
            int global_expert = warp_id * expert_per_group + i;
            float val = scores[blockIdx.x * num_experts + global_expert];
            val = sigmoid(val) + bias[global_expert];  // sigmoid + bias

            // 维护 Top-2
            if (val > max1) { max2 = max1; max1 = val; }
            else if (val > max2) { max2 = val; }
        }

        // warp reduce: 每个 warp 的最大值和次大值
        max1 = cg::reduce(tile, max1, cg::greater<float>());

        // 处理唯一最大值情况
        bool has_max1 = (max1 == max1);  // 当前线程有 max1
        if (has_max1 && __popc(__ballot_sync(0xffffffff, has_max1)) == 1) {
            float local_max2 = (has_max1) ? max2 : max1;
            max2 = cg::reduce(tile, local_max2, cg::greater<float>());
        } else {
            max2 = max1;  // 无需额外计算
        }

        if (lane_id == 0) {
            s_group_scores[warp_id] = max1 + max2;  // 组分数
        }
    }

    __syncthreads();

    // Phase 2: warp 0 选择 Top-K 组
    if (warp_id == 0) {
        warp_topk::WarpSelect<32, /*greater*/ true, float, int, /*stable*/ true>
            group_sel(topk_group, -INFINITY);

        float my_score = (lane_id < n_group) ? s_group_scores[lane_id] : -INFINITY;
        group_sel.add(my_score, lane_id);
        group_sel.done();

        // Phase 3: 从选中组中选择全局 Top-K 专家
        warp_topk::WarpSelect<32, /*greater*/ true, float, int, /*stable*/ true>
            expert_sel(topk, -INFINITY);

        for (int g = 0; g < topk_group; ++g) {
            int selected_group = group_sel.get_idx(0);

            // 广播组 ID
            selected_group = __shfl_sync(0xffffffff, selected_group, g);

            int group_offset = selected_group * expert_per_group;
            for (int i = lane_id; i < expert_per_group; i += 32) {
                int global_expert = group_offset + i;
                float val = scores[blockIdx.x * num_experts + global_expert];
                float biased_val = sigmoid(val) + bias[global_expert];
                expert_sel.add(biased_val, global_expert);
            }
        }
        expert_sel.done();

        // 输出结果
        if (lane_id < topk) {
            int expert_id = expert_sel.get_idx(0);
            topk_indices[blockIdx.x * topk + lane_id] = expert_id;
            topk_values[blockIdx.x * topk + lane_id] =
                sigmoid(scores[blockIdx.x * num_experts + expert_id]);
        }
    }
}
```

---

## 6. 依赖和兼容性 (Dependencies & Compatibility)

### 6.1 头文件依赖

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/limits>
```

### 6.2 CUDA 版本要求

| 特性 | 最低 CUDA 版本 |
|------|----------------|
| `__shfl_xor_sync` | CUDA 9.0 |
| `cooperative_groups` | CUDA 9.0 |
| `__nv_bfloat16` | CUDA 11.0 |
| `cuda::std::numeric_limits` | CUDA 12.8+ |

### 6.3 设备架构要求

- **WARP_SIZE**: 固定为 32（当前 NVIDIA GPU）
- **Compute Capability**: >= 7.0（推荐 8.0+ 以获得最佳性能）

### 6.4 数据类型支持

| 类型 | 支持 | 说明 |
|------|------|------|
| `float` | ✅ | 原生支持 |
| `half` (fp16) | ✅ | 需 `cuda_fp16.h` |
| `__nv_bfloat16` | ✅ | 需 `cuda_bf16.h`，CC >= 8.0 |
| `int32_t` | ✅ | 索引类型 |
| `int64_t` | ✅ | 索引类型（如需要）|

### 6.5 编译选项

```cmake
# 推荐编译选项
set(CUDA_ARCHITECTURES 80;86;90)  # 根据目标 GPU 调整
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr")
```

### 6.6 与其他库的兼容性

| 库 | 兼容性 | 说明 |
|------|--------|------|
| TensorRT-LLM | ✅ | 原始来源 |
| FastDeploy | ⚠️ | 有不同实现（reduceTopK）|
| PyTorch | ✅ | 可通过 custom op 集成 |

---

## 7. 实际应用案例 (Real-world Use Cases)

### Case 1: grouped_topk_fused_kernel 中的使用

**位置**: `grouped_topk_kernels.cu:586-644`

**场景**: DeepSeek V3 MoE 路由，从多个专家组中选择 Top-K 专家。

**使用模式**:

```cuda
// 1. 选择 Top-K 个组
warp_topk::WarpSelect<32, true, T, int32_t, true>
    group_sel(topk_group, neg_inf<T>());

// 2. 每个线程贡献一个组分数
T gscore = (lane_id < n_group) ? s_group_scores[lane_id] : neg_inf<T>();
group_sel.add(gscore, lane_id);
group_sel.done();

// 3. 选择全局 Top-K 专家
warp_topk::WarpSelect<32, true, T, int32_t, true>
    expert_sel(topk, neg_inf<T>());

// 4. 从选中组的每个专家中添加候选
for (int32_t g = 0; g < topk_group; ++g) {
    int32_t gid = __shfl_sync(FULL_WARP_MASK, sel_gid_lane, g);
    // 遍历组内所有专家
    for (int32_t i = lane_id; i < num_experts_per_group; i += 32) {
        int32_t idx = gid * num_experts_per_group + i;
        T cand = sigmoid(scores_token[idx]) + bias[idx];
        expert_sel.add(cand, idx);
    }
}
expert_sel.done();
```

**关键点**:
- 使用 `capacity=32`（warp size），每个线程管理 1 个元素
- 稳定排序（`is_stable=true`），保证结果可预测
- 多轮选择：先选组，再从组中选专家

---

### Case 2: MoE 路由中的 Expert 选择

**场景**: 为每个 token 分配到 K 个最优的专家。

**简化版实现**:

```cuda
__global__ void moe_routing_topk(float* gating_output, int* expert_assignment,
                                 int num_tokens, int num_experts, int k) {
    int token_id = blockIdx.x;
    int lane = threadIdx.x;

    extern __shared__ char smem_buf[];
    warp_topk::WarpSelect<32, /*greater*/ true, float, int, /*stable*/ false>
        selector(k, -INFINITY);

    // 每个线程负责一部分专家的分数计算
    int experts_per_thread = (num_experts + 31) / 32;
    int start = lane * experts_per_thread;
    int end = min(start + experts_per_thread, num_experts);

    for (int i = start; i < end; ++i) {
        float score = sigmoid(gating_output[token_id * num_experts + i]);
        selector.add(score, i);
    }
    selector.done();

    // 输出分配结果
    if (lane < k) {
        expert_assignment[token_id * k + lane] = selector.get_idx(0);
    }
}
```

**优化点**:
- 直接处理分数，无需额外的 bias 计算阶段
- 使用不稳定排序提升性能（`is_stable=false`）
- 批量添加减少 kernel 启动开销

---

### Case 3: 多组 Top-K 合并场景

**场景**: 合并多个 pre-sort chunk 的 Top-K 结果。

```cuda
__device__ void merge_pre_sorted_chunks(
    float* sorted_chunks, int num_chunks, int chunk_size,
    int k, float* output_vals, int* output_idxs) {

    extern __shared__ char smem_buf[];
    warp_topk::WarpSelect<32, /*greater*/ true, float, int, /*stable*/ true>
        merger(k, -INFINITY);

    // 逐个 chunk 加载
    for (int c = 0; c < num_chunks; ++c) {
        float* chunk_vals = sorted_chunks + c * chunk_size * 2;
        int* chunk_idxs = reinterpret_cast<int*>(chunk_vals + chunk_size);

        // 使用 load_sorted 加载预排序结果
        merger.load_sorted(chunk_vals, chunk_idxs, 0);
    }

    merger.dump(output_vals, output_idxs);
}
```

**适用场景**:
- 分布式 Top-K 的 reduce 阶段
- 分层搜索的最后合并步骤

---

## 8. 最佳实践 (Best Practices)

### 8.1 容量选择 (capacity 参数) 指南

| 场景 | 推荐配置 | 理由 |
|------|----------|------|
| 标准 warp 操作 | `capacity=32` | 与 WARP_SIZE 匹配，每线程 1 个元素 |
| Top-K 且 K<16 | `capacity=32` | 仍为 32，避免模板特殊化 |
| 需要 >32 元素 | `capacity=64, 128, ...` | 使用多个 warp 或更大容量 |
| 固定小 K 值 | `capacity=32` | 性能损失小，代码简单 |

**权衡**:
- `capacity` 越大，每线程管理的 `max_arr_len_` 越大
- `capacity > 32` 时，需要多个 warp 的协作

---

### 8.2 稳定 vs 不稳定排序的选择场景

| 场景 | 推荐 | 原因 |
|------|------|------|
| MoE 路由（DeepSeek V3）| 稳定 | 结果可复现，便于调试 |
| 通用 Top-K（无特殊要求）| 不稳定 | 更快，减少比较操作 |
| 需要确定性输出 | 稳定 | 相等元素顺序可预测 |
| 性能敏感 | 不稳定 | 省去索引比较 |

**示例对比**:

```cuda
// 不稳定排序 - 值相等时，输出顺序可能不同
WarpSelect<32, true, float, int, /*is_stable*/ false>

// 稳定排序 - 值相等时，索引小的排在前面
WarpSelect<32, true, float, int, /*is_stable*/ true>
```

---

### 8.3 Shared Memory 使用优化建议

**1. WarpSelect 的 smem 布局**:

```cuda
// WarpSelect 自动使用以下布局：
extern __shared__ char smem_buf[];
// [val_smem_: num_warps * 32 * sizeof(T)]
// [idx_smem_: num_warps * 32 * sizeof(idx_t)]

// 用户可以在这之后放置自定义 smem：
T* custom_smem = reinterpret_cast<T*>(
    smem_buf + num_warps * 32 * (sizeof(T) + sizeof(idx_t)));
```

**2. 对齐要求**:

```cuda
// WarpSelect 内部使用 256 字节对齐
size_t val_bytes_aligned =
    warp_topk::round_up_to_multiple_of<256>(val_bytes);

// 用户自定义 smem 建议使用 16B 或 32B 对齐
uintptr_t ptr_u = reinterpret_cast<uintptr_t>(smem_buf + offset);
ptr_u = (ptr_u + 15) & ~static_cast<uintptr_t>(15);  // 16B 对齐
```

**3. Bank Conflict 避免**:

```cuda
// 避免：连续线程访问连续元素（可能 bank conflict）
__shared__ float data[32];  // 4 banks
data[threadIdx.x];  // ❌ 可能 conflict

// 建议：交错访问或使用更大的数据类型
__shared__ float2 data[16];  // 8 banks
data[threadIdx.x / 2];  // ✅ 更好
```

---

### 8.4 性能调优技巧

#### 1. 使用 `__restrict__` 指针

```cuda
void add(T const* __restrict__ in, ...)  // 告诉编译器指针无别名
```

#### 2. 使用 `__forceinline__` 关键函数

```cuda
__forceinline__ __device__ bool is_better_than(...)  // 减少调用开销
```

#### 3. 模板常量化分支

```cuda
if constexpr (is_stable) {  // 编译期分支，运行时无开销
    // 稳定排序代码
} else {
    // 不稳定排序代码
}
```

#### 4. 合并循环以减少迭代

```cuda
// 不好：多次循环
for (int i = lane; i < n; i += 32) {
    process_one(i);
}

// 好：处理多个元素
for (int i = lane; i < n; i += 32) {
    val[i] = compute_val(i);      // 合并访存
    idx[i] = compute_idx(i);
    process_pair(val[i], idx[i]); // 合并计算
}
```

#### 5. 使用 `__ballot_sync` 替代 `__syncthreads`

```cuda
// Warp 内同步，不需要 block 级同步
uint32_t mask = __ballot_sync(FULL_WARP_MASK, condition);
```

---

### 8.5 常见陷阱和解决方案

| 陷阱 | 症状 | 解决方案 |
|------|------|---------|
| 忘记调用 `done()` | 部分结果丢失 | 在 `add()` 完成后必须调用 `done()` |
| shared memory 冲突 | 性能下降 | 检查访问模式，使用交错访问 |
| `capacity` 非 2 的幂 | 编译错误 | 确保为 32, 64, 128 等 |
| lane 假设 | 非全 warp 情况错误 | 使用 `__shfl_sync` 全 mask 而非假设 |
| 浮点 NaN 比较 | 排序错误 | 使用 `is_finite()` 过滤 NaN |

---

### 8.6 调试技巧

```cuda
// 1. 打印 warp 内状态
if (lane_id == 0) {
    printf("Warp %d: k_th_=%f, smem_buf_len_=%d\n",
           warp_id, k_th_, smem_buf_len_);
}

// 2. 验证结果一致性
__syncthreads();  // 确保所有线程完成
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Top-1: idx=%d, val=%f\n", get_idx(0), get_val(0));
}

// 3. 检查 shared memory
__shared__ bool smem_valid[32];
smem_valid[lane_id] = (val_smem_[lane_id] != dummy);
__syncthreads();
if (lane_id == 0) {
    printf("Valid entries: %d\n",
           __popc(__ballot_sync(0xffffffff, smem_valid[lane_id])));
}
```

---

## 附录 A: 快速参考卡 (Quick Reference Card)

```
┌─────────────────────────────────────────────────────────────────┐
│                    WarpSelect 快速参考                        │
├─────────────────────────────────────────────────────────────────┤
│  声明:                                                     │
│  WarpSelect<32, true, float, int, true> sel(k, -INFINITY); │
│                                                         │
│  使用流程:                                                  │
│  1. extern __shared__ char smem_buf[];  // 必须            │
│  2. WarpSelect<...> selector(k, dummy);                     │
│  3. selector.add(val, idx);  // 或 selector.add(arr, s, e) │
│  4. selector.done();           // 必须                    │
│  5. selector.dump(out, out_idx);                          │
│                                                         │
│  关键参数:                                                │
│  - capacity: 建议 32（warp size）                          │
│  - greater: true=降序, false=升序                         │
│  - is_stable: true=稳定排序（相等时比索引）               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 附录 B: 术语表 (Glossary)

| 术语 | 英文 | 解释 |
|------|------|------|
| 双调序列 | Bitonic Sequence | 先单调递增后单调递减的序列 |
| Warp | Warp | CUDA 的基本执行单元，32 个线程 |
| Lane | Lane | Warp 中的单个线程（0-31）|
| 稳定排序 | Stable Sort | 相等元素保持原始相对顺序 |
| 降序/升序 | Descending/Ascending | 大的在前 / 小的在前 |
| Shared Memory | 共享内存 | Block 内线程共享的快速内存 |
| Shuffle | Shuffle 指令 | Warp 内线程间直接交换数据 |

---

*文档结束*

如有问题或需要补充，请参考源代码：`grouped_topk_kernels.cu:47-411`
