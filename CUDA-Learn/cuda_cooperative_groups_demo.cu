/**
 * CUDA Cooperative Groups 示例演示
 *
 * 覆盖章节：
 *   §2  Group Handle 与成员函数
 *   §3  隐式 Group
 *   §4  分区操作（tiled / labeled / binary）
 *   §5  同步机制（sync / barrier arrive-wait）
 *   §6  集合操作（reduce / scan / invoke_one）
 *   §7  异步数据移动（memcpy_async / double buffering）
 *   §8  大规模 Group（grid-level sync）
 *
 * 目标机器：H800（sm_90）
 * 编译：nvcc -std=c++17 -arch=sm_90 cuda_cooperative_groups_demo.cu -o cg_demo
 * 运行：./cg_demo
 *
 * ──────────────────────────────────────────────────────────────
 *  重要 API 约束（与文档描述存在的差异）：
 *  • cg::reduce / inclusive_scan / exclusive_scan
 *      只支持 thread_block_tile / coalesced_group，不支持 thread_block
 *  • labeled_partition / binary_partition
 *      第一个参数须为 coalesced_group 或 thread_block_tile
 *  • invoke_one / invoke_one_broadcast（CUDA 12.4+）
 *      同上，不支持 thread_block
 *  解决方案：先用 tiled_partition<32> 切出 warp tile 或
 *           用 coalesced_threads() 获取 coalesced_group
 * ──────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ═══════════════════════════════════════════════════════════════════
// §2  Group Handle 与成员函数
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_group_handle()
{
    cg::thread_block block = cg::this_thread_block();

    uint32_t rank = block.thread_rank();   // 组内排名 (0-indexed)
    uint32_t size = block.num_threads();   // 组内线程总数
    dim3     idx  = block.thread_index();  // 3D 线程索引（= threadIdx）
    dim3     dim  = block.dim_threads();   // block 尺寸（= blockDim）

    if (rank == 0)
        printf("[§2] block.num_threads=%u  block.dim_threads=(%u,%u,%u)\n",
               size, dim.x, dim.y, dim.z);
    if (rank < 4)
        printf("[§2] thread_rank=%u  thread_index=(%u,%u,%u)\n",
               rank, idx.x, idx.y, idx.z);
}

// ═══════════════════════════════════════════════════════════════════
// §3  隐式 Group
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_implicit_groups()
{
    // 最常用：当前 thread block 内所有线程
    cg::thread_block block = cg::this_thread_block();

    // warp 中当前活跃线程（有分支时不等于完整 warp）
    cg::coalesced_group active = cg::coalesced_threads();

    if (block.thread_rank() == 0)
        printf("[§3] thread_block.num_threads=%u  coalesced_threads=%u\n",
               block.num_threads(), active.num_threads());

    // grid_group 和 cluster_group 需要特殊启动，见 §8
}

// ═══════════════════════════════════════════════════════════════════
// §4.1  tiled_partition（最常用）
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_tiled_partition()
{
    // ✅ 在任何分支前创建 Group Handle
    cg::thread_block block = cg::this_thread_block();

    // 编译期常量大小 → 启用 warp-level 原语（shfl / vote 等）
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);

    int global_rank = (int)block.thread_rank();
    int tile_rank   = (int)tile8.thread_rank();  // 0~7
    int tile_id     = global_rank / 8;

    tile8.sync();   // 仅同步 tile8 内的 8 个线程

    int val      = threadIdx.x;
    int shuffled = tile8.shfl(val, 0);  // 广播 tile 内 rank-0 的值

    if (global_rank < 16)  // 只打印前 2 个 tile
        printf("[§4.1] global=%2d  tile_id=%d  tile_rank=%d"
               "  val=%2d  shfl_from_rank0=%2d\n",
               global_rank, tile_id, tile_rank, val, shuffled);
}

// ═══════════════════════════════════════════════════════════════════
// §4.2  labeled_partition
// 约束：第一个参数须为 coalesced_group 或 thread_block_tile
//       不能直接传 thread_block
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_labeled_partition(int* data)
{
    cg::thread_block block = cg::this_thread_block();

    // 用 tiled_partition<32> 切出 warp tile（编译期常量，尺寸须 <= 32）
    // 这里 blockDim.x = 16，用 tile<16> 覆盖整个 block
    auto tile = cg::tiled_partition<16>(block);

    int  label    = data[tile.thread_rank()] % 4;   // 0~3
    auto subgroup = cg::labeled_partition(tile, label);

    // reduce 也只支持 tile / coalesced_group
    int sum = cg::reduce(subgroup, data[tile.thread_rank()], cg::plus<int>());

    if (subgroup.thread_rank() == 0)
        printf("[§4.2] label=%d  subgroup_size=%u  reduce_sum=%d\n",
               label, subgroup.num_threads(), sum);
}

// ═══════════════════════════════════════════════════════════════════
// §4.3  binary_partition
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_binary_partition(int* data)
{
    cg::thread_block block = cg::this_thread_block();
    auto             tile  = cg::tiled_partition<16>(block);

    bool is_positive = (data[tile.thread_rank()] > 0);
    auto subgroup    = cg::binary_partition(tile, is_positive);

    if (subgroup.thread_rank() == 0)
        printf("[§4.3] is_positive=%d  subgroup_size=%u\n",
               (int)is_positive, subgroup.num_threads());
}

// ═══════════════════════════════════════════════════════════════════
// §5.1  sync()
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_sync()
{
    cg::thread_block block = cg::this_thread_block();

    __shared__ int shared_val;

    if (block.thread_rank() == 0)
        shared_val = 42;

    cg::sync(block);   // 等价于 __syncthreads()，block 内全部线程参与
    // block.sync();  // 完全等价的成员函数写法

    // 子组同步：仅同步 tile 内的线程
    auto tile16 = cg::tiled_partition<16>(block);
    tile16.sync();

    if (block.thread_rank() == 0)
        printf("[§5.1] shared_val=%d  (cg::sync OK)\n", shared_val);
}

// ═══════════════════════════════════════════════════════════════════
// §5.2  Barriers — arrive / wait 分离
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ int local_compute(int rank) { return rank * rank; }

__global__ void demo_barrier()
{
    cg::thread_block block = cg::this_thread_block();

    // Phase 1：声明"我已到达"，拿到一次性 token
    auto token = block.barrier_arrive();

    // ── arrive → wait 之间：只做本地计算，禁止任何 collective ──
    int result = local_compute((int)block.thread_rank());

    // Phase 2：等待所有线程都调用 barrier_arrive（消耗 token）
    block.barrier_wait(std::move(token));

    // barrier_wait 保证：所有线程都已 arrive（写完 shared 数据）
    // 不保证：所有线程都已执行完 barrier_wait
    if (block.thread_rank() == 0)
        printf("[§5.2] barrier arrive-wait passed  result[rank=0]=%d\n", result);
}

// ═══════════════════════════════════════════════════════════════════
// §6.1  reduce
// 约束：cg::reduce 只支持 thread_block_tile / coalesced_group
// 方案：先 warp-level reduce，再 shared memory 汇总
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_block_reduce(const int* data, int* result)
{
    cg::thread_block     block = cg::this_thread_block();
    // 每个 warp 切成 tile<32>，在 tile 内做 reduce
    auto warp = cg::tiled_partition<32>(block);

    int val      = data[threadIdx.x];
    int warp_sum = cg::reduce(warp, val, cg::plus<int>());   // warp 内归约

    __shared__ int warp_results[32];   // 最多 blockDim.x/32 个 warp
    if (warp.thread_rank() == 0) {
        int wid = (int)threadIdx.x / 32;
        warp_results[wid] = warp_sum;
    }
    block.sync();

    // 线程 0 汇总各 warp 结果
    if (block.thread_rank() == 0) {
        int total     = 0;
        int num_warps = (int)block.num_threads() / 32;
        for (int i = 0; i < num_warps; i++) total += warp_results[i];
        result[blockIdx.x] = total;
        printf("[§6.1] block=%d  sum=%d\n", blockIdx.x, total);
    }
}

// §6.1  也演示 min / max / bit_and 等其他 operator
__global__ void demo_reduce_ops(const int* data, int* out_max, int* out_and)
{
    cg::thread_block block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int val     = data[threadIdx.x];
    int warp_max = cg::reduce(warp, val, cg::greater<int>());
    int warp_and = cg::reduce(warp, val, cg::bit_and<int>());

    if (warp.thread_rank() == 0 && blockIdx.x == 0) {
        printf("[§6.1-ops] warp_max=%d  warp_bit_and=0x%X\n",
               warp_max, (unsigned)warp_and);
        out_max[threadIdx.x / 32] = warp_max;
        out_and[threadIdx.x / 32] = warp_and;
    }
}

// ═══════════════════════════════════════════════════════════════════
// §6.2  exclusive_scan / inclusive_scan
// 约束：只支持 thread_block_tile（N 须为 2 的幂，且 ≤ 32）
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_exclusive_scan(const int* data, int* result)
{
    cg::thread_block block = cg::this_thread_block();
    // blockDim.x = 8，用 tile<8>
    auto tile = cg::tiled_partition<8>(block);

    int val        = data[tile.thread_rank()];
    int prefix_sum = cg::exclusive_scan(tile, val, cg::plus<int>());
    result[tile.thread_rank()] = prefix_sum;
}

__global__ void demo_inclusive_scan(const int* data, int* result)
{
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<8>(block);

    int val           = data[tile.thread_rank()];
    int inclusive_sum = cg::inclusive_scan(tile, val, cg::plus<int>());
    result[tile.thread_rank()] = inclusive_sum;
}

// ═══════════════════════════════════════════════════════════════════
// §6.3  invoke_one / invoke_one_broadcast（CUDA 12.4+）
// 约束：只支持 coalesced_group / thread_block_tile，不支持 thread_block
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_invoke_one(int* counter)
{
    cg::thread_block block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);   // warp tile

    // 只选 warp 内一个线程打印，其余不执行，无广播
    cg::invoke_one(warp, []() {
        printf("[§6.3] invoke_one: exactly ONE thread in this warp prints\n");
    });
    warp.sync();

    // 只选一个线程执行原子加，返回值广播给 warp 内所有线程
    int ticket = cg::invoke_one_broadcast(warp, [counter]() -> int {
        return atomicAdd(counter, 1);   // 返回旧值（操作前的值）
    });

    // warp 内所有 32 个线程都拿到相同的 ticket
    if (warp.thread_rank() < 4)
        printf("[§6.3] invoke_one_broadcast: warp_rank=%u  ticket=%d\n",
               warp.thread_rank(), ticket);
}

// ═══════════════════════════════════════════════════════════════════
// §7  memcpy_async — 基础预取
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ int independent_work(int rank) { return rank + 100; }

__global__ void demo_memcpy_async(const int* __restrict__ input, int* output)
{
    cg::thread_block block = cg::this_thread_block();
    __shared__ int   smem[256];   // blockDim.x == 256

    // 整块异步拷贝：组内线程协作，将 blockDim.x 个 int 从 global 拷到 shared
    // size 参数是总字节数，由 block 内所有线程共同完成
    cg::memcpy_async(
        block,
        smem,
        input + blockIdx.x * blockDim.x,
        blockDim.x * sizeof(int)   // 整块大小，非单元素
    );

    // ── 与传输重叠的本地计算（不依赖 smem）──
    int local_val = independent_work((int)block.thread_rank());

    // 等待所有异步拷贝完成（组内所有线程必须调用）
    cg::wait(block);

    // smem 数据已就绪
    output[blockIdx.x * blockDim.x + block.thread_rank()] =
        smem[block.thread_rank()] + local_val;
}

// ═══════════════════════════════════════════════════════════════════
// §7  memcpy_async — Double Buffering
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_double_buffer(const int* __restrict__ input,
                                   int* output, int N)
{
    cg::thread_block block = cg::this_thread_block();
    const int        CHUNK = (int)blockDim.x;   // == 256

    __shared__ int buf[2][256];
    int cur       = 0;
    int num_chunk = (N + CHUNK - 1) / CHUNK;

    // 预取第一批（等待完成，作为循环起点）
    int first_size = min(CHUNK, N);
    cg::memcpy_async(block, buf[cur], input, first_size * (int)sizeof(int));
    cg::wait(block);

    for (int c = 1; c < num_chunk; c++) {
        int next        = 1 - cur;
        int next_offset = c * CHUNK;
        int next_size   = min(CHUNK, N - next_offset);

        // 异步预取下一批
        cg::memcpy_async(block, buf[next],
                         input + next_offset, next_size * (int)sizeof(int));

        // 同时处理当前批（计算与传输重叠）
        int cur_offset = (c - 1) * CHUNK;
        int cur_size   = min(CHUNK, N - cur_offset);
        for (int i = (int)block.thread_rank(); i < cur_size;
             i += (int)block.num_threads())
            output[cur_offset + i] = buf[cur][i] * 2;

        cg::wait(block);   // 等待下一批预取完成
        cur = next;
    }

    // 处理最后一批
    int last_offset = (num_chunk - 1) * CHUNK;
    int last_size   = min(CHUNK, N - last_offset);
    for (int i = (int)block.thread_rank(); i < last_size;
         i += (int)block.num_threads())
        output[last_offset + i] = buf[cur][i] * 2;
}

// ═══════════════════════════════════════════════════════════════════
// §8  Grid-level sync（须用 cudaLaunchCooperativeKernel 启动）
// ═══════════════════════════════════════════════════════════════════
__global__ void demo_grid_sync(int* data, int N)
{
    cg::grid_group grid = cg::this_grid();
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    // Phase 1：所有线程写数据
    if (tid < N) data[tid] = tid;

    // Grid-level 同步：确保所有 block 完成写入
    // 普通 <<<>>> 启动时此处行为未定义，必须用 CooperativeKernel
    grid.sync();

    // Phase 2：安全读跨 block 边界的邻居数据
    if (tid < N) {
        int neighbor = (tid + 1 < N) ? data[tid + 1] : data[0];
        if (tid < 4)
            printf("[§8] grid_sync: tid=%d  data[tid]=%d  neighbor=%d\n",
                   tid, data[tid], neighbor);
    }
}

// ───────────────────────────────────────────────────────────────────
// 辅助
// ───────────────────────────────────────────────────────────────────
static void print_arr(const char* label, const int* arr, int n)
{
    printf("%s [", label);
    for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", arr[i]);
    printf("]\n");
}

static bool arr_eq(const int* a, const int* b, int n)
{
    return memcmp(a, b, (size_t)n * sizeof(int)) == 0;
}

// ═══════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════
int main()
{
    // 打印设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("============================================================\n");
    printf("  CUDA Cooperative Groups Demo\n");
    printf("  Device: %s  (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("============================================================\n\n");

    // ── §2  Group Handle ─────────────────────────────────────────
    printf("── §2  Group Handle ────────────────────────────────────\n");
    demo_group_handle<<<1, 8>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");

    // ── §3  Implicit Groups ──────────────────────────────────────
    printf("── §3  Implicit Groups ─────────────────────────────────\n");
    demo_implicit_groups<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");

    // ── §4.1  tiled_partition ────────────────────────────────────
    printf("── §4.1 tiled_partition (tile<8>, 32 threads) ──────────\n");
    demo_tiled_partition<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");

    // ── §4.2  labeled_partition ──────────────────────────────────
    printf("── §4.2 labeled_partition (label = val%%4) ──────────────\n");
    {
        const int N = 16;
        int h[N]; for (int i = 0; i < N; i++) h[i] = i;   // 0..15
        int* d; CUDA_CHECK(cudaMalloc(&d, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_labeled_partition<<<1, N>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d));
    }
    printf("\n");

    // ── §4.3  binary_partition ───────────────────────────────────
    printf("── §4.3 binary_partition (positive / non-positive) ────\n");
    {
        const int N = 16;
        int h[N];
        for (int i = 0; i < N; i++) h[i] = (i % 3 == 0) ? -(i + 1) : (i + 1);
        int* d; CUDA_CHECK(cudaMalloc(&d, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_binary_partition<<<1, N>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d));
    }
    printf("\n");

    // ── §5.1  sync() ─────────────────────────────────────────────
    printf("── §5.1 sync() ─────────────────────────────────────────\n");
    demo_sync<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");

    // ── §5.2  Barriers ───────────────────────────────────────────
    printf("── §5.2 Barriers arrive-wait ───────────────────────────\n");
    demo_barrier<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");

    // ── §6.1  reduce ─────────────────────────────────────────────
    printf("── §6.1 reduce (warp-level + shared-mem aggregation) ───\n");
    {
        const int N = 64;   // 2 warps
        int h[N], hr[1];
        int expected = 0;
        for (int i = 0; i < N; i++) { h[i] = i + 1; expected += h[i]; }
        int *d, *dr;
        CUDA_CHECK(cudaMalloc(&d,  N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dr, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_block_reduce<<<1, N>>>(d, dr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hr, dr, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[main] sum=%d  expected=%d  %s\n",
               hr[0], expected, hr[0] == expected ? "PASS" : "FAIL");
        CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(dr));
    }
    printf("\n");

    printf("── §6.1 reduce operators (max / bit_and) ───────────────\n");
    {
        const int N = 32;
        int h[N];
        for (int i = 0; i < N; i++) h[i] = i + 1;   // 1..32
        int *d, *dmax, *dand;
        CUDA_CHECK(cudaMalloc(&d,    N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dmax, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dand, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_reduce_ops<<<1, N>>>(d, dmax, dand);
        CUDA_CHECK(cudaDeviceSynchronize());
        // 1&2&...&32 = 0  (因为 32 是 0b100000，其他数都没有该位，但低位又有 0)
        // max(1..32) = 32
        printf("[main] expected max=32, bit_and=0\n");
        CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(dmax)); CUDA_CHECK(cudaFree(dand));
    }
    printf("\n");

    // ── §6.2  exclusive_scan ──────────────────────────────────────
    printf("── §6.2 exclusive_scan ─────────────────────────────────\n");
    {
        const int N = 8;
        int h[N]        = {1,2,3,4,5,6,7,8};
        int expected[N] = {0,1,3,6,10,15,21,28};
        int hr[N];
        int *d, *dr;
        CUDA_CHECK(cudaMalloc(&d,  N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dr, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_exclusive_scan<<<1, N>>>(d, dr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hr, dr, N * sizeof(int), cudaMemcpyDeviceToHost));
        print_arr("[main] result  ", hr, N);
        print_arr("[main] expected", expected, N);
        printf("[main] %s\n", arr_eq(hr, expected, N) ? "PASS" : "FAIL");
        CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(dr));
    }
    printf("\n");

    printf("── §6.2 inclusive_scan ─────────────────────────────────\n");
    {
        const int N = 8;
        int h[N]        = {1,2,3,4,5,6,7,8};
        int expected[N] = {1,3,6,10,15,21,28,36};
        int hr[N];
        int *d, *dr;
        CUDA_CHECK(cudaMalloc(&d,  N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dr, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_inclusive_scan<<<1, N>>>(d, dr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hr, dr, N * sizeof(int), cudaMemcpyDeviceToHost));
        print_arr("[main] result  ", hr, N);
        print_arr("[main] expected", expected, N);
        printf("[main] %s\n", arr_eq(hr, expected, N) ? "PASS" : "FAIL");
        CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(dr));
    }
    printf("\n");

    // ── §6.3  invoke_one ──────────────────────────────────────────
    printf("── §6.3 invoke_one / invoke_one_broadcast ──────────────\n");
    {
        int h = 0, *d;
        CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice));
        demo_invoke_one<<<1, 32>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[main] counter=%d  expected=1  %s\n",
               h, h == 1 ? "PASS" : "FAIL");
        CUDA_CHECK(cudaFree(d));
    }
    printf("\n");

    // ── §7  memcpy_async 基础 ──────────────────────────────────────
    printf("── §7  memcpy_async (basic prefetch) ───────────────────\n");
    {
        const int N = 256;
        int h_in[N], h_out[N];
        for (int i = 0; i < N; i++) h_in[i] = i;
        int *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_memcpy_async<<<1, N>>>(d_in, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
        // output[i] = smem[i] + (i + 100) = h_in[i] + i + 100 = 2i + 100
        bool ok = true;
        for (int i = 0; i < N; i++)
            if (h_out[i] != 2 * i + 100) { ok = false; break; }
        printf("[main] %s\n", ok ? "PASS" : "FAIL");
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }
    printf("\n");

    // ── §7  Double Buffering ──────────────────────────────────────
    printf("── §7  memcpy_async (double buffering, 3 chunks) ───────\n");
    {
        const int N = 768;   // 3 × 256
        int *h_in  = new int[N];
        int *h_out = new int[N];
        for (int i = 0; i < N; i++) h_in[i] = i;
        int *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
        demo_double_buffer<<<1, 256>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
        // output[i] = input[i] * 2
        bool ok = true;
        for (int i = 0; i < N; i++)
            if (h_out[i] != h_in[i] * 2) { ok = false; break; }
        printf("[main] %s\n", ok ? "PASS" : "FAIL");
        delete[] h_in; delete[] h_out;
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }
    printf("\n");

    // ── §8  Grid-level sync ───────────────────────────────────────
    printf("── §8  Grid-level sync (cudaLaunchCooperativeKernel) ───\n");
    {
        int supportsCoopLaunch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch,
                                          cudaDevAttrCooperativeLaunch, 0));
        if (!supportsCoopLaunch) {
            printf("[§8] Device does not support cooperative launch, skipped.\n");
        } else {
            const int N     = 64;
            const int BLOCK = 32;
            const int GRID  = (N + BLOCK - 1) / BLOCK;
            int *d;
            CUDA_CHECK(cudaMalloc(&d, N * sizeof(int)));
            int  n    = N;
            void* args[] = { &d, &n };
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)demo_grid_sync,
                dim3(GRID), dim3(BLOCK),
                args, 0, nullptr));
            CUDA_CHECK(cudaDeviceSynchronize());
            int h[N];
            CUDA_CHECK(cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost));
            bool ok = true;
            for (int i = 0; i < N; i++) if (h[i] != i) { ok = false; break; }
            printf("[main] data write check: %s\n", ok ? "PASS" : "FAIL");
            CUDA_CHECK(cudaFree(d));
        }
    }
    printf("\n");

    printf("============================================================\n");
    printf("  All demos completed.\n");
    printf("============================================================\n");
    return 0;
}
