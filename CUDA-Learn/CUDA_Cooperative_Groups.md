# CUDA Cooperative Groups 学习笔记

> 来源：[CUDA Programming Guide 4.4 Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) | v13.2 | Updated 2026-03-04

---

## 目录

1. [简介](#1-简介)
2. [Group Handle 与成员函数](#2-group-handle-与成员函数)
3. [隐式 Group（默认行为）](#3-隐式-group默认行为)
4. [创建 Cooperative Groups（分区操作）](#4-创建-cooperative-groups分区操作)
5. [同步机制](#5-同步机制)
6. [集合操作（Collective Operations）](#6-集合操作collective-operations)
7. [异步数据移动（Async Memcpy）](#7-异步数据移动async-memcpy)
8. [大规模 Group（跨 Grid 同步）](#8-大规模-group跨-grid-同步)
9. [最佳实践与注意事项](#9-最佳实践与注意事项)

---

## 1. 简介

**Cooperative Groups** 是 CUDA 编程模型的扩展，核心目标是让开发者能够**精确控制线程协作的粒度**。

### 解决了什么问题？

在 Cooperative Groups 之前，CUDA 仅有一个同步原语：
```cuda
__syncthreads();  // 只能同步整个 thread block
```

这导致开发者不得不自行实现 warp 内或跨 block 的不安全的临时同步机制，代码脆弱、难维护、跨 GPU 世代不可移植。

Cooperative Groups 提供了：
- 安全的、面向未来的同步机制
- 内置常用并行原语（scan、reduce 等）
- 灵活的线程分组粒度（sub-warp → warp → block → cluster → grid）

---

## 2. Group Handle 与成员函数

所有 Cooperative Groups 通过 **Group Handle** 管理，Handle 使线程能查询自己在组内的位置和组的信息。

```cpp
namespace cg = cooperative_groups;  // 标准命名空间别名（全文通用）
```

### 核心成员函数

| 函数 | 返回值 | 说明 |
|------|--------|------|
| `thread_rank()` | `uint32_t` | 当前线程在组内的排名（0-indexed） |
| `num_threads()` | `uint32_t` | 组内线程总数 |
| `thread_index()` | `dim3` | 线程在 launched block 内的 3D 索引 |
| `dim_threads()` | `dim3` | launched block 的 3D 维度 |

### 使用示例

```cpp
namespace cg = cooperative_groups;

__global__ void kernel() {
    cg::thread_block block = cg::this_thread_block();

    // 查询当前线程在 block 中的排名
    int rank = block.thread_rank();      // 等价于 threadIdx.x（1D block 情况下）
    int size = block.num_threads();      // 等价于 blockDim.x（1D block 情况下）

    printf("Thread rank: %d / %d\n", rank, size);
}
```

> **完整 API 文档**：参考 [Cooperative Groups API Reference](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups-api)

---

## 3. 隐式 Group（默认行为）

CUDA Runtime 根据 kernel 启动配置**自动创建**以下隐式 Group，作为进一步分区的起点：

| 访问函数 | 返回 Group 范围 | 约束条件 |
|---------|----------------|---------|
| `cg::this_thread_block()` | 当前 thread block 内所有线程 | 无 |
| `cg::this_grid()` | 整个 grid 内所有线程 | 需配合 `cudaLaunchCooperativeKernel` |
| `cg::coalesced_threads()` | warp 中当前活跃线程 | 不保证确定性，不保证全程合并 |
| `cg::this_cluster()` | 当前 cluster 内所有线程 | 计算能力 ≥ 9.0（Hopper+） |

### 示例：获取各级隐式 Group

```cpp
__global__ void implicit_groups_demo() {
    namespace cg = cooperative_groups;

    // Block 级 Group（最常用）
    cg::thread_block block = cg::this_thread_block();

    // Grid 级 Group（需要特殊启动方式）
    cg::grid_group grid = cg::this_grid();

    // Warp 中的活跃线程组（适用于有分支的 warp）
    cg::coalesced_group active = cg::coalesced_threads();

    // Cluster 级 Group（仅 Hopper 及以上）
    // cg::cluster_group cluster = cg::this_cluster();
}
```

### ⚠️ 重要提示

- `coalesced_threads()` 返回的是**当前时刻活跃的线程集合**，不保证哪些线程被选中，也不保证它们后续保持合并状态
- `this_cluster()` 在非 cluster 启动时假设 1×1×1 cluster

---

## 4. 创建 Cooperative Groups（分区操作）

分区操作将父 Group 切分为子 Group。**分区是 collective 操作**，父 Group 中所有线程必须参与。

### 分区方式总览

| 分区类型 | 分配方式 | 典型用途 |
|---------|---------|---------|
| `tiled_partition` | 连续块划分（row-major） | sub-warp 级并行原语 |
| `stride_partition` | 循环轮询分配 | 跨步访问模式 |
| `labeled_partition` | 按整型标签分组 | 自定义条件分组 |
| `binary_partition` | 按 0/1 标签分组 | `labeled_partition` 的特化版本 |

---

### 4.1 `tiled_partition`（最常用）

将 parent group 切分为大小固定的连续子组，子组按 row-major 排列。

```cpp
namespace cg = cooperative_groups;

__global__ void tiled_demo() {
    // Step 1: 获取 block 级隐式 Group（尽早创建，避免分支后创建）
    cg::thread_block block = cg::this_thread_block();

    // Step 2: 将 block 切分为大小为 8 的 tile（编译期常量，性能最佳）
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);

    // Step 3: 在 tile 内操作
    int my_rank_in_tile = tile8.thread_rank();  // 0~7
    int tile_id = block.thread_rank() / 8;      // 当前 tile 的编号

    // tile 内同步
    tile8.sync();

    // tile 内 warp-level 原语（shfl、vote 等）
    int val = threadIdx.x;
    int shuffled = tile8.shfl(val, 0);  // broadcast tile[0] 的值到所有成员
}
```

> **编译期 vs 运行期模板参数**：
> - `tiled_partition<N>(g)` — N 为编译期常量，启用 warp-level 原语（`shfl`、`vote` 等），**推荐**
> - `tiled_partition(g, n)` — n 为运行期参数，不支持 warp-level 原语

---

### 4.2 `labeled_partition`

按整型标签将线程分入不同子组：

```cpp
__global__ void labeled_demo(int* data) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    // 根据数据值的奇偶分组
    int label = data[threadIdx.x] % 4;  // 0~3，最多 4 个子组
    auto subgroup = cg::labeled_partition(block, label);

    // 同一 label 的线程在同一 subgroup 内协作
    int sum = cg::reduce(subgroup, data[threadIdx.x], cg::plus<int>());
}
```

---

### 4.3 `binary_partition`

`labeled_partition` 的 0/1 特化，适用于二分条件：

```cpp
__global__ void binary_demo(int* data) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    bool is_positive = (data[threadIdx.x] > 0);
    auto subgroup = cg::binary_partition(block, is_positive);

    // 正数线程和负数线程分别在各自的 subgroup 内
    int count = subgroup.num_threads();
}
```

---

### ⚠️ 避免 Group 创建危险（Hazard）

```cpp
// ❌ 危险：只有部分线程进入分支，collective 操作会死锁或数据损坏
if (threadIdx.x < 16) {
    auto subgroup = cg::tiled_partition<8>(block);  // 只有 16 个线程执行了这行！
    subgroup.sync();  // DEADLOCK！
}

// ✅ 正确：先完成分区，再条件分支
auto subgroup = cg::tiled_partition<8>(block);
if (threadIdx.x < 16) {
    subgroup.sync();  // OK，此时整个 block 都已完成分区
}
```

---

## 5. 同步机制

### 5.1 `sync()`

`cg::sync()` 是 `__syncthreads()` 的 CG 泛化版本，适用于**任意 Group 粒度**。

**保证：**
1. 同步点之前的所有内存访问（读/写），在同步点之后对组内所有线程可见
2. 所有组内线程到达同步点后，才允许任何线程继续执行

```cpp
namespace cg = cooperative_groups;

__global__ void sync_demo(int* shared_buffer) {
    cg::thread_block block = cg::this_thread_block();

    // 等价于 __syncthreads()
    cg::sync(block);

    // 也可以通过 handle 调用
    block.sync();

    // 子组同步（仅同步 tile 内的线程）
    auto tile = cg::tiled_partition<16>(block);
    tile.sync();
}
```

> `cg::sync(grid)` 可同步整个 grid，但需要使用 `cudaLaunchCooperativeKernel` 启动

---

### 5.2 Barriers（屏障 API）

Cooperative Groups 提供类似 `cuda::barrier` 的屏障 API，支持**到达与等待分离**（arrive-wait split），允许隐藏同步延迟。

**与 `cuda::barrier` 的关键区别：**
- CG barrier 自动初始化，无需手动 init
- 每个 phase 内，组内所有线程必须 **各调用一次** arrive 和 wait
- `barrier_arrive()` 返回 `arrival_token`，必须传入对应的 `barrier_wait()`（一次性消耗）

```cpp
namespace cg = cooperative_groups;

__global__ void barrier_demo() {
    cg::thread_block block = cg::this_thread_block();

    // Phase 1: 线程 arrive（声明"我已到达"），返回 token
    auto token = block.barrier_arrive();

    // ── 可在此处做本地计算，隐藏同步延迟 ──
    int local_result = do_local_work();  // 不能用 shared memory 或 collective 操作

    // Phase 2: 等待所有线程都 arrive（消耗 token）
    block.barrier_wait(std::move(token));

    // 现在所有线程都已 arrive，可安全访问共享数据
    // 注意：barrier_wait 不保证所有线程都已执行完 barrier_wait
    // 只保证所有线程都已执行了 barrier_arrive
}
```

**⚠️ 使用 Barrier 的注意事项：**

```
arrive ──► [本地工作] ──► wait
             ↑
    此区间内禁止：
    - 任何 collective 操作（sync、reduce 等）
    - 访问被 barrier 保护的共享内存

barrier_wait 保证：所有线程都已 arrive
barrier_wait 不保证：所有线程都已 wait（即 wait 之后只能安全读，不能认为所有人都过了 wait）
```

---

## 6. 集合操作（Collective Operations）

集合操作需要组内**所有线程参与**，且同一参数对应位置传入的值必须相同（除非 API 明确允许）。

---

### 6.1 `reduce`（并行归约）

对组内每个线程的数据执行归约。

**支持的 Operator：**

| Operator | 计算 | 类型限制 |
|----------|------|---------|
| `cg::plus<T>` | 求和 | 算术类型 |
| `cg::less<T>` | 最小值 | 可比较类型 |
| `cg::greater<T>` | 最大值 | 可比较类型 |
| `cg::bit_and<T>` | 按位与 | 整数类型 |
| `cg::bit_or<T>` | 按位或 | 整数类型 |
| `cg::bit_xor<T>` | 按位异或 | 整数类型 |

> 硬件加速：计算能力 ≥ 8.0（Ampere+）对 4 字节类型有硬件加速；旧硬件自动降级为软件实现

```cpp
namespace cg = cooperative_groups;

__global__ void block_reduce_sum(int* data, int* result) {
    cg::thread_block block = cg::this_thread_block();

    int val = data[threadIdx.x];

    // Block 内所有线程的 val 求和，每个线程都得到结果
    int sum = cg::reduce(block, val, cg::plus<int>());

    // 只需要一个线程写回结果（通常选 rank 0）
    if (block.thread_rank() == 0) {
        result[blockIdx.x] = sum;
    }
}
```

**使用 tiled 子组的 reduce（实现分层归约）：**

```cpp
__global__ void warp_reduce_demo(int* data, int* result) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    // 先在 warp 内归约（tile 大小 = 32）
    auto warp = cg::tiled_partition<32>(block);
    int val = data[threadIdx.x];
    int warp_sum = cg::reduce(warp, val, cg::plus<int>());

    // warp leader 写入 shared memory
    __shared__ int warp_results[32];  // 假设最多 32 个 warp
    if (warp.thread_rank() == 0) {
        warp_results[threadIdx.x / 32] = warp_sum;
    }
    block.sync();

    // 再在 block 级别做最终归约（这里简化为单线程）
    if (block.thread_rank() == 0) {
        int total = 0;
        for (int i = 0; i < block.num_threads() / 32; i++)
            total += warp_results[i];
        result[blockIdx.x] = total;
    }
}
```

---

### 6.2 `inclusive_scan` / `exclusive_scan`（前缀扫描）

对任意大小 Group 执行前缀扫描，支持上方所有 Reduction Operator。

| 函数 | 语义 | 示例（输入 [1,2,3,4]） |
|------|------|----------------------|
| `inclusive_scan` | 包含自身的前缀和 | → [1, 3, 6, 10] |
| `exclusive_scan` | 不含自身的前缀和（首元素为 identity） | → [0, 1, 3, 6] |

```cpp
namespace cg = cooperative_groups;

__global__ void scan_demo(int* data, int* result) {
    cg::thread_block block = cg::this_thread_block();

    int val = data[block.thread_rank()];

    // Exclusive scan：每个线程得到其之前所有线程的和
    int prefix_sum = cg::exclusive_scan(block, val, cg::plus<int>());

    // 结果：result[i] = data[0] + data[1] + ... + data[i-1]
    result[block.thread_rank()] = prefix_sum;
}
```

```cpp
// Inclusive scan 示例
__global__ void inclusive_scan_demo(int* data, int* result) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    int val = data[block.thread_rank()];

    // Inclusive：result[i] = data[0] + ... + data[i]
    int inclusive_sum = cg::inclusive_scan(block, val, cg::plus<int>());
    result[block.thread_rank()] = inclusive_sum;
}
```

---

### 6.3 `invoke_one` / `invoke_one_broadcast`

适用于组内只需要**一个线程**执行串行工作（如 I/O、原子操作）的场景。

| 函数 | 行为 |
|------|------|
| `invoke_one(g, fn, args...)` | 选一个线程执行 fn，其他线程不执行，**无返回值广播** |
| `invoke_one_broadcast(g, fn, args...)` | 选一个线程执行 fn，**返回值广播给所有线程** |

> **线程选择不确定**：哪个线程被选中不保证确定性

```cpp
namespace cg = cooperative_groups;

__global__ void invoke_one_demo(int* counter) {
    cg::thread_block block = cg::this_thread_block();

    // ── 示例 1：只有一个线程打印 ──
    cg::invoke_one(block, []() {
        printf("Hello from one thread in the block!\n");
    });
    cg::sync(block);  // 等待打印完成

    // ── 示例 2：只有一个线程做原子加，结果广播给所有人 ──
    int ticket = cg::invoke_one_broadcast(block, [counter]() -> int {
        return atomicAdd(counter, 1);  // 返回旧值
    });
    // 此时 block 内所有线程都拿到了 ticket（相同值）
    printf("Thread %d got ticket: %d\n", block.thread_rank(), ticket);
}
```

**⚠️ 约束：**
- invocable 内部**不允许**对调用组进行 collective 操作或同步
- 允许与调用组**之外**的线程通信

---

## 7. 异步数据移动（Async Memcpy）

`memcpy_async` 提供 global memory → shared memory 的**异步拷贝**，核心用途是"预取"数据，将数据传输延迟与计算重叠。

### API

| 函数 | 作用 |
|------|------|
| `cg::memcpy_async(g, dst, src, size)` | 发起异步拷贝（非阻塞），由组内线程协作完成 |
| `cg::wait(g)` | 等待组内所有异步拷贝完成（必须所有线程都调用） |

### 基础使用示例

```cpp
namespace cg = cooperative_groups;

__global__ void prefetch_demo(int* input, int* output, int N) {
    cg::thread_block block = cg::this_thread_block();

    __shared__ int shared_data[256];  // 假设 blockDim.x = 256

    // 发起异步拷贝：global → shared
    // 每个线程负责拷贝对应 index 的一个 int
    cg::memcpy_async(
        block,
        shared_data + block.thread_rank(),   // dst：shared memory
        input + blockIdx.x * blockDim.x + block.thread_rank(),  // src：global memory
        sizeof(int)
    );

    // ── 可在此处做与 shared_data 无关的计算，隐藏 memcpy 延迟 ──
    int local_val = do_independent_work();

    // 等待异步拷贝完成（必须！）
    cg::wait(block);

    // 现在 shared_data 中的数据已就绪
    output[blockIdx.x * blockDim.x + block.thread_rank()] =
        shared_data[block.thread_rank()] + local_val;
}
```

### Double Buffering 模式（高级用法）

```cpp
__global__ void double_buffer_demo(int* input, int* output, int N) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    __shared__ int buf[2][256];  // 双缓冲
    int cur = 0;

    // 预取第一批
    cg::memcpy_async(block, buf[cur], input, 256 * sizeof(int));
    cg::wait(block);

    for (int i = 256; i < N; i += 256) {
        int next = 1 - cur;
        // 预取下一批（异步）
        cg::memcpy_async(block, buf[next], input + i, 256 * sizeof(int));

        // 同时处理当前批（计算与传输重叠）
        process(buf[cur], output);

        cg::wait(block);  // 等待下一批预取完成
        cur = next;
    }
    process(buf[cur], output);  // 处理最后一批
}
```

### ⚠️ 对齐要求

| 条件 | 效果 |
|------|------|
| src（global）和 dst（shared）均 ≥ 4 字节对齐 | 真正异步执行 |
| 不满足对齐条件 | 退化为同步拷贝 |
| **推荐** | 16 字节对齐，性能最佳 |

---

## 8. 大规模 Group（跨 Grid 同步）

### `cudaLaunchCooperativeKernel`

当需要 **Grid 级别同步**（`cg::this_grid().sync()`）时，必须使用该 API 启动 kernel。

**普通 `<<<>>>` 启动无法支持 grid-level 同步。**

#### 第一步：检查设备支持

```cpp
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

if (!supportsCoopLaunch) {
    fprintf(stderr, "Device does not support cooperative launch!\n");
    return;
}
// 计算能力 ≥ 6.0 才支持
```

#### 第二步：使用 `cudaLaunchCooperativeKernel` 启动

```cpp
void* args[] = { &d_input, &d_output, &N };

cudaLaunchCooperativeKernel(
    (void*)my_kernel,   // kernel 函数指针
    gridDim,            // grid 尺寸
    blockDim,           // block 尺寸
    args,               // 参数数组
    0,                  // shared memory bytes
    stream              // CUDA stream
);
```

#### Kernel 内部使用 Grid-level 同步

```cpp
__global__ void grid_sync_kernel(int* data, int N) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1：所有线程写数据
    data[tid] = tid;

    // Grid-level 同步：确保所有 block 都写完
    grid.sync();

    // Phase 2：所有线程读邻居数据（安全，因为已同步）
    int neighbor = (tid + 1 < N) ? data[tid + 1] : data[0];
}
```

### 平台支持要求

| 平台 | 要求 |
|------|------|
| Linux（无 MPS） | 计算能力 ≥ 6.0 |
| Linux（有 MPS） | 计算能力 ≥ 7.0 |
| Windows | 最新版本 |

> ⚠️ **CUDA 13 起**：已移除多设备（multi-device）Cooperative Launch API，不再支持跨设备同步。

---

## 9. 最佳实践与注意事项

### 性能建议

```
1. 尽早创建 Group Handle（在任何分支之前）：
   ✅ auto block = cg::this_thread_block();  // kernel 入口处
   ❌ if (cond) { auto block = cg::this_thread_block(); ... }

2. 传递 Group Handle 时使用引用：
   ✅ void foo(cg::thread_block& block) { ... }
   ❌ void foo(cg::thread_block block) { ... }  // 拷贝构造不推荐

3. tiled_partition 优先使用编译期常量大小（启用 warp primitives）：
   ✅ cg::tiled_partition<32>(block)
   ❌ cg::tiled_partition(block, n)  // 运行期大小，失去 shfl/vote 等能力

4. memcpy_async 使用 16 字节对齐以达到最佳性能
```

### 常见陷阱

```
❌ 陷阱 1：在条件分支内执行 collective 操作
   if (some_condition) {
       cg::sync(block);  // 若不是所有线程都进入分支 → 死锁
   }

❌ 陷阱 2：barrier_arrive 和 barrier_wait 之间使用 collective 操作
   auto token = block.barrier_arrive();
   int x = cg::reduce(block, val, cg::plus<int>());  // 非法！
   block.barrier_wait(std::move(token));

❌ 陷阱 3：invoke_one 内部对调用组做 collective 操作
   cg::invoke_one(block, [&]() {
       block.sync();  // 非法！
   });

❌ 陷阱 4：用普通 <<<>>> 启动使用 grid.sync() 的 kernel
   my_kernel<<<grid, block>>>();  // grid.sync() 行为未定义！
   // 必须用 cudaLaunchCooperativeKernel
```

### 快速选型指南

```
需要同步整个 block？                → cg::sync(block) 或 block.sync()
需要 warp 内 shuffle/vote？         → tiled_partition<32>(block) + warp-level primitives
需要任意大小子组归约？              → tiled_partition<N>(block) + cg::reduce()
需要按条件分组？                   → labeled_partition / binary_partition
只让一个线程做串行工作？            → invoke_one / invoke_one_broadcast
需要隐藏 global→shared 传输延迟？   → memcpy_async + wait
需要跨 block 全局同步？             → cudaLaunchCooperativeKernel + this_grid().sync()
```

---

*文档来源：NVIDIA CUDA Programming Guide v13.2，2026-03-04*
