# Warp TopK 工具参考手册 (Warp TopK Tool Reference)

> **来源**: vLLM `grouped_topk_kernels.cu` (vllm::moe::warp_topk namespace)
> **版本**: 基于 v1.3.0rc2 TensorRT-LLM 改编
> **文档日期**: 2026-05-09

---

## 目录

1. [完整工具代码 (Complete Code)](#1-完整工具代码-complete-code)
2. [快速使用指南 (Quick Start)](#2-快速使用指南-quick-start)
3. [使用示例 (Usage Examples)](#3-使用示例-usage-examples)
4. [依赖和兼容性 (Dependencies & Compatibility)](#4-依赖和兼容性-dependencies--compatibility)
5. [最佳实践 (Best Practices)](#5-最佳实践-best-practices)
6. [API 快速参考 (API Quick Reference)](#6-api-快速参考-api-quick-reference)

---

## 1. 完整工具代码 (Complete Code)

> 直接复制以下代码到你的项目中即可使用。

```cuda
// ============== warp_topk.h ==============
#pragma once
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace warp_topk {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
static constexpr int WARP_SIZE = 32;

// ============== 工具函数 ==============
template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) return 0;
  return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
__forceinline__ __device__ bool is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val, T baseline, idxT index,
                                               idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}

// ============== 双调排序模板 ==============
template <int size, bool ascending, bool reverse, typename T, typename idxT, bool is_stable>
struct BitonicMerge {
  __device__ static void merge(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable) {
        is_better = is_better_than<ascending>(val, other_val, idx_arr[i], idx_arr[other_i]);
      } else {
        is_better = is_better_than<ascending>(val, other_val);
      }
      if (is_better) {
        T tmp = val; val = other_val; other_val = tmp;
        idxT tmp2 = idx_arr[i]; idx_arr[i] = idx_arr[other_i]; idx_arr[other_i] = tmp2;
      }
    }
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(val_arr, idx_arr);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(val_arr + arr_len / 2, idx_arr + arr_len / 2);
  }
};

template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  __device__ static void sort(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(val_arr, idx_arr);
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(val_arr + arr_len / 2, idx_arr + arr_len / 2);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(val_arr, idx_arr);
  }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  __device__ static void sort(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;
        T other = __shfl_xor_sync(FULL_WARP_MASK, *val_arr, stride);
        idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, *idx_arr, stride);
        bool is_better;
        if constexpr (is_stable) {
          if constexpr (ascending) {
            is_better = ((*val_arr > other) || ((*val_arr == other) && (*idx_arr < other_idx))) != (reverse != is_second);
          } else {
            is_better = ((*val_arr > other) || ((*val_arr == other) && (*idx_arr > other_idx))) != (reverse != is_second);
          }
        } else {
          is_better = (*val_arr != other && (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) { *val_arr = other; *idx_arr = other_idx; }
      }
    }
    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(val_arr, idx_arr);
  }
};

template <bool ascending, bool reverse, typename T, typename idxT, bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  __device__ static void merge(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      T& val = *val_arr;
      T other = __shfl_xor_sync(FULL_WARP_MASK, val, stride);
      idxT& idx = *idx_arr;
      idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, stride);
      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending) {
          is_better = ((*val_arr > other) || ((*val_arr == other) && (*idx_arr < other_idx))) == (reverse != is_second);
        } else {
          is_better = ((*val_arr > other) || ((*val_arr == other) && (*idx_arr > other_idx))) == (reverse != is_second);
        }
      } else {
        is_better = (val != other && ((val > other) == (ascending != is_second)));
      }
      if (is_better) { val = other; idx = other_idx; }
    }
  }
};

// ============== WarpSort 类 (全排序) ==============
template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy)
      : lane_(threadIdx.x % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));
    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  __device__ void load_sorted(T const* __restrict__ in, idxT const* __restrict__ in_idx, idxT start) {
    idxT idx = start + WARP_SIZE - 1 - lane_;
    for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WARP_SIZE) {
      if (idx < start + k_) {
        T t = in[idx];
        bool is_better;
        if constexpr (is_stable) {
          is_better = is_better_than<greater>(t, val_arr_[i], in_idx[idx], idx_arr_[i]);
        } else {
          is_better = is_better_than<greater>(t, val_arr_[i]);
        }
        if (is_better) { val_arr_[i] = t; idx_arr_[i] = in_idx[idx]; }
      }
    }
    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(val_arr_, idx_arr_);
  }

  __device__ void dump(T* __restrict__ out, idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) { out[out_i] = val_arr_[i]; out_idx[out_i] = idx_arr_[i]; }
    }
  }

  __device__ void dumpIdx(idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) { out_idx[out_i] = idx_arr_[i]; }
    }
  }

  __device__ __forceinline__ idxT get_idx(int i = 0) const { return idx_arr_[i]; }
  __device__ __forceinline__ T get_val(int i = 0) const { return val_arr_[i]; }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;
  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];
  int const lane_;
  idxT const k_;
  T const dummy_;
};

// ============== WarpSelect 类 (Top-K 选择) ==============
template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  __device__ WarpSelect(idxT k, T dummy)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy),
        k_th_(dummy), k_th_idx_(0), k_th_lane_((k - 1) % WARP_SIZE) {
    extern __shared__ char smem_buf[];
    int const num_of_warp = blockDim.x / WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;
    val_smem_ = reinterpret_cast<T*>(smem_buf) + warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(smem_buf + round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE)) + warp_id * WARP_SIZE;
  }

  __device__ void add(T const* in, idxT start, idxT end) {
    idxT const end_for_fullwarp = round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i);
    }
  }

  __device__ void add(T val, idxT idx) {
    bool do_add;
    if constexpr (is_stable) {
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    } else {
      do_add = is_better_than<greater>(val, k_th_);
    }
    uint32_t mask = __ballot_sync(FULL_WARP_MASK, do_add);
    if (mask == 0) return;

    int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));
    if (do_add && pos < WARP_SIZE) {
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += __popc(mask);
    if (smem_buf_len_ >= WARP_SIZE) {
      __syncwarp();
      merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      pos -= WARP_SIZE;
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
    __syncwarp();
  }

  __device__ void done() {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx);
    }
  }

 private:
  __device__ void set_k_th_() {
    k_th_ = __shfl_sync(FULL_WARP_MASK, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable) {
      k_th_idx_ = __shfl_sync(FULL_WARP_MASK, idx_arr_[max_arr_len_ - 1], k_th_lane_);
    }
  }

  __device__ void merge_buf_(T val, idxT idx) {
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(&val, &idx);
    T& old = val_arr_[max_arr_len_ - 1];
    bool is_better;
    if constexpr (is_stable) {
      is_better = is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    } else {
      is_better = is_better_than<greater>(val, old);
    }
    if (is_better) { old = val; idx_arr_[max_arr_len_ - 1] = idx; }
    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(val_arr_, idx_arr_);
    set_k_th_();
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;
  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};

}  // namespace warp_topk
```

---

## 2. 快速使用指南 (Quick Start)

### 核心类对比

| 类 | 用途 | 何时使用 |
|------|------|---------|
| `WarpSort` | 全排序 | 需要完整排序结果 |
| `WarpSelect` | Top-K 选择 | 只需要最大的 K 个元素 |

### WarpSelect 标准用法

```cuda
__global__ void topk_kernel(float* data, float* out_vals, int* out_idxs,
                           int n, int k) {
    int lane = threadIdx.x % 32;
    int token_id = blockIdx.x;

    extern __shared__ char smem_buf[];  // 必须！

    // 1. 创建 WarpSelect
    warp_topk::WarpSelect<32, /*greater*/ true, float, int, /*stable*/ true>
        selector(k, -INFINITY);

    // 2. 添加元素
    float* token_data = data + token_id * n;
    selector.add(token_data, 0, n);

    // 3. 完成选择（必须调用！）
    selector.done();

    // 4. 输出结果
    if (lane < k) {
        out_vals[token_id * k + lane] = selector.get_val(0);
        out_idxs[token_id * k + lane] = selector.get_idx(0);
    }
}

// 启动配置:
// grid: num_tokens
// block: 32 threads (1 warp)
// dynamic smem: 2 * num_warps * 32 * sizeof(element_type)
```

---

## 3. 使用示例 (Usage Examples)

### 3.1 简单 Top-K 选择

```cuda
__global__ void simple_topk(float* input, float* output, int n, int k) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    extern __shared__ char smem_buf[];

    warp_topk::WarpSelect<32, true, float, int, true>
        selector(k, -INFINITY);

    float* warp_input = input + blockIdx.x * n;
    selector.add(warp_input, 0, n);
    selector.done();

    if (lane < k) {
        output[blockIdx.x * k + lane] = selector.get_val(0);
    }
}
```

### 3.2 分组 Top-K (MoE 风格)

```cuda
__global__ void grouped_topk(float* scores, float* topk_vals, int* topk_idxs,
                            int num_experts, int n_group, int topk_group, int k) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int experts_per_group = num_experts / n_group;

    extern __shared__ char smem_buf[];
    float* s_group_scores = reinterpret_cast<float*>(smem_buf);

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    // Phase 1: 计算每组分数 (warp 并行)
    if (warp_id < n_group) {
        float max1 = -INFINITY, max2 = -INFINITY;
        for (int i = lane; i < experts_per_group; i += 32) {
            float val = scores[blockIdx.x * num_experts + warp_id * experts_per_group + i];
            if (val > max1) { max2 = max1; max1 = val; }
            else if (val > max2) { max2 = val; }
        }
        max1 = cg::reduce(tile, max1, cg::greater<float>());
        if (lane == 0) s_group_scores[warp_id] = max1 + max2;
    }
    __syncthreads();

    // Phase 2: 选择 Top-K 组
    if (warp_id == 0) {
        warp_topk::WarpSelect<32, true, float, int, true>
            group_sel(topk_group, -INFINITY);
        float my_score = (lane < n_group) ? s_group_scores[lane] : -INFINITY;
        group_sel.add(my_score, lane);
        group_sel.done();

        // Phase 3: 从选中组选择全局 Top-K
        warp_topk::WarpSelect<32, true, float, int, true>
            expert_sel(k, -INFINITY);
        for (int g = 0; g < topk_group; ++g) {
            int gid = __shfl_sync(0xffffffff, group_sel.get_idx(0), g);
            for (int i = lane; i < experts_per_group; i += 32) {
                int global_expert = gid * experts_per_group + i;
                float val = scores[blockIdx.x * num_experts + global_expert];
                expert_sel.add(val, global_expert);
            }
        }
        expert_sel.done();

        if (lane < k) {
            topk_idxs[blockIdx.x * k + lane] = expert_sel.get_idx(0);
            topk_vals[blockIdx.x * k + lane] = expert_sel.get_val(0);
        }
    }
}
```

### 3.3 合并多个已排序数组

```cuda
__global__ void merge_sorted_chunks(float** chunks, int num_chunks, int chunk_size,
                                  float* out_vals, int* out_idxs, int k) {
    int lane = threadIdx.x % 32;

    extern __shared__ char smem_buf[];

    warp_topk::WarpSelect<32, true, float, int, true>
        merger(k, -INFINITY);

    for (int c = 0; c < num_chunks; ++c) {
        merger.load_sorted(chunks[c], chunks[c] + chunk_size, 0);
    }

    merger.dump(out_vals, out_idxs);
}
```

---

## 4. 依赖和兼容性 (Dependencies & Compatibility)

### 必需头文件

```cuda
#include <cooperative_groups.h>
```

### 最低 CUDA 版本

| 特性 | 最低版本 |
|------|---------|
| `__shfl_xor_sync` | CUDA 9.0 |
| `__ballot_sync` | CUDA 9.0 |
| `cooperative_groups` | CUDA 9.0 |

### 数据类型支持

| 类型 | 支持 |
|------|------|
| `float` | ✅ |
| `half` (fp16) | ✅ |
| `__nv_bfloat16` | ✅ (CC >= 8.0) |
| `int32_t` | ✅ (索引类型) |

### 编译要求

```cmake
set(CUDA_ARCHITECTURES 80;86;90)  # 根据目标 GPU
```

---

## 5. 最佳实践 (Best Practices)

### 5.1 容量 (capacity) 选择

| 场景 | 推荐配置 |
|------|---------|
| 标准 warp 操作 | `capacity=32` |
| Top-K 且 K<16 | `capacity=32` |
| 需要 >32 元素 | `capacity=64, 128, ...` |

### 5.2 稳定 vs 不稳定排序

| 场景 | 推荐 |
|------|------|
| MoE 路由（需要可复现）| `is_stable=true` |
| 通用 Top-K | `is_stable=false` |
| 性能敏感 | `is_stable=false` |

### 5.3 Shared Memory 布局

```cuda
// WarpSelect 自动使用以下 smem 布局:
// [val_smem_: num_warps * 32 * sizeof(T)]
// [idx_smem_: num_warps * 32 * sizeof(idx_t)]

// 用户可以在这之后放置自定义 smem:
T* custom_smem = reinterpret_cast<T*>(
    smem_buf + num_warps * 32 * (sizeof(T) + sizeof(idx_t)));
```

### 5.4 常见陷阱

| 陷阱 | 解决方案 |
|------|---------|
| 忘记调用 `done()` | 在 `add()` 完成后必须调用 |
| capacity 非 2 的幂 | 使用 32, 64, 128 等值 |
| 忘记 extern shared memory | 添加 `extern __shared__ char smem_buf[];` |

### 5.5 性能优化

```cuda
// 使用 __restrict__ 告诉编译器指针无别名
void add(T const* __restrict__ in, ...)

// 使用模板常量化分支（编译期优化）
if constexpr (is_stable) { ... }

// 使用 warp 同步而非 block 同步
uint32_t mask = __ballot_sync(FULL_WARP_MASK, condition);
```

---

## 6. API 快速参考 (API Quick Reference)

### WarpSelect 模板参数

```cuda
WarpSelect<capacity, greater, T, idxT, is_stable> selector(k, dummy);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `capacity` | int | 容量（必须 ≥32 且为 2 的幂）|
| `greater` | bool | true=降序, false=升序 |
| `T` | 值类型 | float, half 等 |
| `idxT` | 索引类型 | int32_t 等 |
| `is_stable` | bool | 稳定排序 |

### WarpSelect 方法

```cuda
// 添加单个元素
void add(T val, idxT idx);

// 批量添加数组元素
void add(T const* in, idxT start, idxT end);

// 完成选择（必须调用）
void done();

// 获取结果
T get_val(int i = 0) const;
idxT get_idx(int i = 0) const;
void dump(T* out, idxT* out_idx) const;
```

---

*文档结束*

如需查看原始实现，参考：`grouped_topk_kernels.cu:47-411`
