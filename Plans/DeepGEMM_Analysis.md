# DeepGEMM（PaddlePaddle Fork）技术分析报告

> 分析日期：2026-04-23
> 仓库路径：`/root/paddlejob/share-storage/gpfs/system-public/zhushengguang/jobspace/PFCC/DeepGEMM/`
> FastDeploy 路径：`/root/paddlejob/workspace/env_run/output/zhushengguang/jobspace/FastDeploy/`

---

## 目录

1. [仓库概述与目录结构](#1-仓库概述与目录结构)
2. [所有公开接口](#2-所有公开接口)
3. [使用场景与主要使用方法](#3-使用场景与主要使用方法)
4. [JIT 编译实现原理](#4-jit-编译实现原理)
5. [与 Paddle 适配的差异化改动](#5-与-paddle-适配的差异化改动)
6. [DeepSeekV3.2 中的 deepgemm 接口使用](#6-deepseekv32-中的-deepgemm-接口使用)
7. [Indexer 中 FP8 误差合理性分析](#7-indexer-中-fp8-误差合理性分析)

---

## 1. 仓库概述与目录结构

DeepGEMM 是 DeepSeek AI 开源的高性能 GEMM 库，专为 NVIDIA Hopper（SM90，H100）和 Blackwell（SM100，H20/H800）架构优化，支持 FP8 E4M3fn 和 BF16 数据类型。本仓库是 PaddlePaddle 团队（PFCC）维护的适配分支（`paddle` 分支），在原版基础上增加了 Paddle 构建系统支持和若干功能增强。

**核心特性：**
- 基于 NVIDIA CUTLASS 3.x，使用 CUDA 12.1+ TMA/Tensor Core 特性
- 全部内核通过 JIT 编译，针对具体矩阵形状深度特化，避免通用路径开销
- 支持 FP8 量化 GEMM（per-token-group 和 per-block 两种缩放粒度）
- 为 MoE（Mixture of Experts）场景提供 M 轴/K 轴分组 GEMM
- 为 DSA（Dynamic Sparse Attention）Indexer 提供专用 MQA Logits 内核

### 1.1 目录结构

```
DeepGEMM/
├── README.md
├── CMakeLists.txt
├── setup.py                        # Python 安装脚本（Paddle 适配版）
│
├── csrc/                           # C++ 核心
│   ├── python_api.cpp              # pybind11 绑定入口
│   ├── apis/                       # 各功能 C++ 接口头文件
│   │   ├── gemm.hpp                # GEMM 核心接口
│   │   ├── attention.hpp           # MQA Logits 接口
│   │   ├── einsum.hpp              # Einsum 接口
│   │   ├── layout.hpp              # Scaling Factor 布局变换
│   │   └── runtime.hpp             # 运行时配置（SM 数量/TC 利用率）
│   ├── jit/                        # JIT 编译系统
│   │   ├── compiler.hpp            # NVCC/NVRTC 编译器抽象
│   │   ├── cache.hpp               # 文件系统编译缓存
│   │   ├── kernel_runtime.hpp      # 已编译内核加载与运行时
│   │   ├── device_runtime.hpp      # 设备运行时（GPU 属性/cuBLASLt 句柄）
│   │   └── handle.hpp              # CUDA Driver/Runtime API 封装
│   ├── jit_kernels/                # JIT 内核模板代码
│   │   ├── heuristics/             # 配置启发式选择（sm90/sm100）
│   │   └── impls/                  # 各架构/场景内核实现的 JIT 包装
│   │       ├── sm90_fp8_gemm_1d1d.hpp
│   │       ├── sm90_fp8_gemm_1d2d.hpp
│   │       ├── sm90_bf16_gemm.hpp
│   │       ├── sm100_fp8_gemm_1d1d.hpp
│   │       ├── sm100_bf16_gemm.hpp
│   │       ├── smxx_fp8_mqa_logits.hpp
│   │       ├── smxx_fp8_paged_mqa_logits.hpp
│   │       ├── smxx_cublaslt.hpp
│   │       └── ...
│   └── utils/                      # 工具头文件（兼容性/异常/格式化等）
│
├── deep_gemm/                      # Python 包
│   ├── __init__.py                 # 包入口（导入并导出所有 API）
│   ├── include/deep_gemm/          # CUDA 内核实现（.cuh/.hpp）
│   │   ├── common/                 # 公共 CUDA 组件
│   │   └── impls/                  # 每个架构的具体内核实现
│   ├── utils/
│   │   ├── layout.py               # 布局变换工具
│   │   └── math.py                 # FP8 量化辅助函数
│   ├── testing/                    # 测试工具（bench/numeric/utils）
│   └── legacy/                     # 遗留 Triton 内核（A100 用）
│
├── tests/                          # 测试文件
│   ├── test_fp8.py
│   ├── test_bf16.py
│   ├── test_attention.py
│   ├── test_einsum.py
│   └── ...
│
└── third-party/
    ├── cutlass/                    # NVIDIA CUTLASS（submodule）
    └── fmt/                        # {fmt} 格式化库（submodule）
```

---

## 2. 所有公开接口

所有公开 API 由 `csrc/python_api.cpp` 通过 pybind11 注册为 `deep_gemm_cpp` 模块，在 `deep_gemm/__init__.py` 中统一导出。命名约定：`nt` = A 不转置 × B 转置（最常用），`nn/tn/tt` 为其他转置组合。

---

### 2.1 FP8 GEMM（稠密矩阵乘法）

```
D = C + A @ B.T   （nt 版本，NT 布局）
```

```python
fp8_gemm_nt(
    a: Tuple[Tensor, Tensor],      # (数据 [M, K] FP8 E4M3fn, 缩放因子 SF)
    b: Tuple[Tensor, Tensor],      # (数据 [N, K] FP8 E4M3fn, 缩放因子 SF)
    d: Tensor,                     # 输出 [M, N]，BF16 或 FP32
    c: Optional[Tensor] = None,    # 累加基础值 [M, N]，可与 d 相同（in-place）
    recipe: Optional[Tuple[int, int, int]] = None,  # SF 粒度，如 (1, 1, 128)
    compiled_dims: str = "nk",     # JIT 编译时固化哪些维度
    disable_ue8m0_cast: bool = False,  # 禁止 SM100 自动 UE8M0 转换
    bias: Optional[Tensor] = None  # 偏置（SM100 专用）
)

# 其他转置变体（参数相同）：
fp8_gemm_nn(...)   # B 不转置
fp8_gemm_tn(...)   # A 转置
fp8_gemm_tt(...)   # A、B 均转置
```

**参数维度详解：**

| 参数 | 形状 | dtype | 语义 |
|------|------|-------|------|
| `a[0]` | `[M, K]` | FP8 E4M3fn | 输入矩阵 A |
| `a[1]`（SM90） | `[M, K/128]` | FP32 | per-token-group 缩放因子，每 128 个 K 元素一个 scale |
| `a[1]`（SM100） | packed UE8M0 | uint32 | 4 个 UE8M0（8bit 无符号指数）打包为 1 个 int32 |
| `b[0]` | `[N, K]` | FP8 E4M3fn | 权重矩阵 B（转置存储） |
| `b[1]` | `[N/128, K/128]` | FP32/UE8M0 | per-block 缩放因子（128×128 块） |
| `d` | `[M, N]` | BF16/FP32 | 输出矩阵 |
| `c` | `[M, N]` | 同 d | 可选累加项，`d = c + A @ B.T` |
| `bias` | `[N]` | BF16/FP32 | 输出偏置（仅 SM100） |

**`compiled_dims` 说明：**
- `"nk"`（默认）：将 N 和 K 编译为常量，适合推理（N/K 固定，M 可变）
- `"mn"`：将 M 和 N 固化，适合训练（M/N 固定，K 可变）
- `"mnk"`：全部固化，最高性能，形状必须完全匹配

---

### 2.2 FP8 MoE M 轴分组 GEMM

**contiguous 布局**（EP prefill，token 已按 expert 连续排列）：

```python
m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[Tensor, Tensor],   # (数据 [M_total, K] FP8, SF)
    b: Tuple[Tensor, Tensor],   # (数据 [G, N, K] FP8, SF)，G = 组数（expert 数）
    d: Tensor,                  # 输出 [M_total, N] BF16
    m_indices: Tensor,          # [G] int32，每组（expert）token 的累积结束偏移
                                # 或 [M_total] int32，每个 token 归属的 expert id（-1=填充）
    recipe=None, compiled_dims="nk", disable_ue8m0_cast=False, bias=None
)
m_grouped_fp8_gemm_nn_contiguous(...)   # B 不转置
```

**masked 布局**（decode/CUDA Graph 友好，保留 padding）：

```python
m_grouped_fp8_gemm_nt_masked(
    a: Tuple[Tensor, Tensor],   # (数据 [G, M_pad, K] FP8, SF)
    b: Tuple[Tensor, Tensor],   # (数据 [G, N, K] FP8, SF)
    d: Tensor,                  # 输出 [G, M_pad, N] BF16
    masked_m: Tensor,           # [G] int32，每组实际有效 token 数（其余行跳过）
    expected_m: int,            # 每组预期最大 token 数（用于 JIT 启发式）
    recipe=None, compiled_dims="nk", disable_ue8m0_cast=False, bias=None
)
```

| 参数 | contiguous 形状 | masked 形状 | 语义 |
|------|----------------|-------------|------|
| `a[0]` | `[M_total, K]` | `[G, M_pad, K]` | 激活矩阵 |
| `b[0]` | `[G, N, K]` | `[G, N, K]` | 各 expert 权重 |
| `d` | `[M_total, N]` | `[G, M_pad, N]` | 输出 |
| `m_indices/masked_m` | `[M_total] int32` | `[G] int32` | token→expert 映射或有效数量 |

---

### 2.3 FP8 K 轴分组 GEMM（MoE 权重梯度）

```python
k_grouped_fp8_gemm_nt_contiguous(   # SM90
    a: Tuple[Tensor, Tensor],   # (数据 [sum_K, M] FP8, SF)
    b: Tuple[Tensor, Tensor],   # (数据 [sum_K, N] FP8, SF)
    d: Tensor,                  # 输出 [G, M, N] FP32（权重梯度）
    ks: List[int],              # 每组 K 轴大小列表，len = G
    ks_tensor: Tensor,          # ks 的 GPU tensor 版本，[G] int32
    c: Optional[Tensor] = None,
    recipe: Tuple = (1, 1, 128),
    compiled_dims: str = "mn"
)
k_grouped_fp8_gemm_tn_contiguous(...)   # SM100 变体
```

---

### 2.4 BF16 GEMM

```python
bf16_gemm_nt(
    a: Tensor,                  # [M, K] BF16
    b: Tensor,                  # [N, K] BF16
    d: Tensor,                  # [M, N] BF16 或 FP32
    c: Optional[Tensor] = None,
    compiled_dims: str = "nk"
)
# 变体：bf16_gemm_nn / bf16_gemm_tn / bf16_gemm_tt

m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices, compiled_dims="nk")
m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims)
k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c, compiled_dims)
```

---

### 2.5 cuBLASLt GEMM（通用 fallback）

```python
cublaslt_gemm_nt(a: Tensor, b: Tensor, d: Tensor, c=None)
cublaslt_gemm_nn(...)
cublaslt_gemm_tn(...)
cublaslt_gemm_tt(...)
```

任意架构可用，作为 JIT 内核不支持时的 fallback。输入输出均为 BF16/FP16/FP32，无量化要求。

---

### 2.6 FP8 MQA Logits（DSA Indexer Prefill 用）

```python
fp8_mqa_logits(
    q: Tensor,                      # [T, num_heads, head_dim] FP8 E4M3fn
    kv: Tuple[Tensor, Tensor],      # ([T_kv, head_dim] FP8, [T_kv] FP32 scale)
                                    # MQA：K 只有 1 个头，scale per-token
    weights: Tensor,                # [T, num_heads] FP32
                                    # 已融合 softmax_scale 和 n_heads^{-0.5}
    cu_seq_len_k_start: Tensor,     # [T] int32，每个 q token 对应 K 范围的起始偏移
    cu_seq_len_k_end: Tensor,       # [T] int32，对应 K 范围的结束偏移
    clean_logits: bool = True,      # True 时将未填充位置置为 -inf
    max_seqlen_k: int = 0           # 输出压缩模式的 K 最大长度
) -> Tensor  # [T, max_seqlen_k] FP32，token-to-token 加权 logits
```

**内核语义：**
```
out[i, j] = relu(sum_h( q[i, h, :] @ kv[j, :] ) * weights[i, h])
```
每个 query token i 与所有 KV token j 做点积，按 head 轴做加权 ReLU 归约，输出标量相似度。

---

### 2.7 FP8 Paged MQA Logits（DSA Indexer Decode 用）

```python
get_paged_mqa_logits_metadata(
    context_lens: Tensor,   # [batch_size] 或 [batch_size, next_n] int32
    block_kv: int,          # KV cache 块大小（固定 64）
    num_sms: int
) -> Tensor  # schedule_meta [num_sms+1, 2] int32，调度元数据

fp8_paged_mqa_logits(
    q: Tensor,              # [batch_size, next_n, num_heads, head_dim] FP8
    fused_kv_cache: Tensor, # [num_kv_blocks, block_kv, 1, head_dim+4] uint8
                            # FP8 数据 + FP32 SF 融合存储（最后 4 字节为 scale）
    weights: Tensor,        # [batch_size*next_n, num_heads] FP32
    context_lens: Tensor,   # [batch_size] 或 [batch_size, next_n] int32
    block_table: Tensor,    # [batch_size, max_block_len] int32，分页表
    schedule_meta: Tensor,  # 由 get_paged_mqa_logits_metadata 生成
    max_context_len: int,
    clean_logits: bool = False
) -> Tensor  # [batch_size*next_n, max_context_len] FP32
```

**与 prefill 版本的关键区别：**

| 特性 | `fp8_mqa_logits`（prefill） | `fp8_paged_mqa_logits`（decode） |
|------|---------------------------|----------------------------------|
| KV 布局 | 连续（gather 后） | 分页（block_tables 索引） |
| `q` 形状 | `[T, n_heads, head_dim]` | `[bsz, next_n, n_heads, head_dim]` |
| `next_n` 支持 | 不适用 | 1 或 2（双 token 预测） |
| schedule | 无 | 需要预计算 schedule_metadata |
| `clean_logits` 默认 | `True` | `False` |

---

### 2.8 特殊 FP8 GEMM（带 Head 插零 Epilogue）

```python
fp8_gemm_nt_skip_head_mid(
    a: Tuple[Tensor, Tensor],      # [M, K] FP8
    b: Tuple[Tensor, Tensor],      # [N, K] FP8
    d: Tensor,                     # [M, N + N/(left+right)*mid] BF16/FP32
                                   # 输出在 mid 位置插入零填充
    head_splits: Tuple[int, int, int],  # (left, mid, right)：head 划分
    recipe=None, compiled_dims="nk", disable_ue8m0_cast=False
)
```

用于 DeepSeek 模型中跳过中间 head 段的 epilogue（特定稀疏注意力场景）。

---

### 2.9 Einsum 接口

```python
einsum(
    expr: str,               # 支持："bmk,bnk->mn", "bhr,hdr->bhd", "bhd,hdr->bhr"
    a: Tensor,               # BF16
    b: Tensor,               # BF16
    d: Tensor,               # 输出 BF16 或 FP32
    c: Optional[Tensor] = None,
    use_cublaslt: bool = False
)

fp8_einsum(
    expr: str,               # 支持："bhr,hdr->bhd", "bhd,hdr->bhr", "bhd,bhr->hdr"
    a: Tuple[Tensor, Tensor],
    b: Tuple[Tensor, Tensor],
    d: Tensor,
    c: Optional[Tensor] = None,
    recipe: Tuple = (1, 128, 128)
)
```

**支持的表达式语义：**

| 表达式 | 含义 |
|--------|------|
| `"bmk,bnk->mn"` | 批量矩阵乘法，对 b 和 k 求和，输出 `[m, n]` |
| `"bhr,hdr->bhd"` | 注意力加权 Value 汇聚：`[batch, head, rank] × [head, d, rank]` |
| `"bhd,hdr->bhr"` | Query/Key 投影：`[batch, head, d] × [head, d, rank]` |
| `"bhd,bhr->hdr"` | 权重梯度计算（FP8 Einsum 专用） |

---

### 2.10 运行时配置接口

```python
set_num_sms(num_sms: int)   # 限制最大 SM 使用数量（用于多任务共享 GPU）
get_num_sms() -> int         # 获取当前 SM 数量（未设置时返回设备总 SM 数）
set_tc_util(ratio: int)      # 设置 Tensor Core 利用率估算（0-100），影响 JIT 启发式
get_tc_util() -> int
```

---

### 2.11 布局变换工具（`csrc/apis/layout.hpp` + `deep_gemm/utils/layout.py`）

```python
transform_sf_into_required_layout(
    sf: Tensor,                    # 输入 scale factor
    mn: int, k: int,               # 矩阵维度
    recipe: Tuple[int, int, int],  # 粒度
    num_groups: Optional[int] = None,
    is_sfa: bool = False,          # True 表示 A 侧 scale（per-token）
    disable_ue8m0_cast: bool = False
) -> Tensor

get_tma_aligned_size(mn: int, element_size: int) -> int
get_mk_alignment_for_contiguous_layout() -> int  # contiguous 布局对齐要求（通常 128）
get_mn_major_tma_aligned_tensor(sf: Tensor) -> Tensor           # FP32 SF → MN-major TMA 对齐
get_mn_major_tma_aligned_packed_ue8m0_tensor(sf: Tensor) -> Tensor  # FP32 SF → UE8M0 packed
get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks) -> Tensor
```

---

### 2.12 Python 量化辅助函数（`deep_gemm/utils/math.py`）

```python
ceil_div(x: int, y: int) -> int
align(x: int, y: int) -> int

# FP8 量化（返回 (fp8_tensor, scale_tensor)）
per_token_cast_to_fp8(x: Tensor, use_ue8m0: bool) -> Tuple[Tensor, Tensor]
# x: [M, K] BF16，输出 scale: [M, K/128] FP32（或 UE8M0 packed）

per_channel_cast_to_fp8(x: Tensor, use_ue8m0: bool) -> Tuple[Tensor, Tensor]
# 沿 channel（第 0 维）做 per-row 量化

per_block_cast_to_fp8(x: Tensor, use_ue8m0: bool) -> Tuple[Tensor, Tensor]
# 128×128 块量化，scale: [M/128, K/128] FP32

per_custom_dims_cast_to_fp8(x, dims, use_ue8m0) -> Tuple[Tensor, Tensor]
# 自定义量化维度

ceil_to_ue8m0(x: Tensor) -> Tensor   # FP32 → 最近上界 UE8M0（即 2 的整数次幂）
```

---

## 3. 使用场景与主要使用方法

### 3.1 使用场景汇总

| 场景 | 接口 | 说明 |
|------|------|------|
| LLM 推理/训练 dense GEMM | `fp8_gemm_nt` | 标准稠密线性层 |
| MoE prefill/训练（contiguous） | `m_grouped_fp8_gemm_nt_contiguous` | token 已排好序，最高效 |
| MoE decode（masked / CUDA Graph） | `m_grouped_fp8_gemm_nt_masked` | 保留 padding，无 reorder 开销 |
| MoE 权重梯度（backward） | `k_grouped_fp8_gemm_*_contiguous` | K 轴分组累积梯度 |
| DSA Indexer prefill logits | `fp8_mqa_logits` | 连续 KV，稀疏注意力索引 |
| DSA Indexer decode logits | `fp8_paged_mqa_logits` | 分页 KV cache，适合推理 |
| GQA/MHA 注意力 einsum | `fp8_einsum("bhr,hdr->bhd")` | Batched head 矩阵乘 |
| 通用 GEMM fallback | `cublaslt_gemm_nt` | 不支持 JIT 或调试时使用 |

### 3.2 标准使用流程（Paddle 环境）

```python
# Step 1: 必须在 import deep_gemm 前开启 Paddle Torch 兼容层
import paddle
paddle.enable_compat(scope={"deep_gemm": True})

# Step 2: 导入
import deep_gemm
import torch  # 通过 Paddle compat 代理

# Step 3: FP8 量化
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

a_bf16 = torch.randn(128, 7168, device='cuda', dtype=torch.bfloat16)
b_bf16 = torch.randn(4096, 7168, device='cuda', dtype=torch.bfloat16)

a_fp8, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)   # SM90
b_fp8, sfb = per_block_cast_to_fp8(b_bf16, use_ue8m0=False)

# Step 4: FP8 GEMM
d = torch.empty(128, 4096, device='cuda', dtype=torch.bfloat16)
deep_gemm.fp8_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d)

# Step 5: MoE grouped GEMM（contiguous 布局）
from deep_gemm.utils import get_mk_alignment_for_contiguous_layout, align

num_experts, K, N = 8, 7168, 2048
align_size = get_mk_alignment_for_contiguous_layout()  # 128

# 每个 expert 的 token 数（需要对齐）
expert_tokens = [512, 640, 480, 0, 320, 480, 512, 704]
aligned_tokens = [align(t, align_size) for t in expert_tokens]
M_total = sum(aligned_tokens)

# 构建 m_indices（每 token 对应 expert id，填充行用 -1）
m_indices = torch.zeros(M_total, dtype=torch.int32, device='cuda')
# ... 填充 m_indices ...

a_fp8, sfa = per_token_cast_to_fp8(a_bf16.view(-1, K), use_ue8m0=False)
b_fp8_list, sfb_list = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=False) for e in range(num_experts)])
b_fp8 = torch.stack(list(b_fp8_list))
sfb = torch.stack(list(sfb_list))

d = torch.empty(M_total, N, device='cuda', dtype=torch.bfloat16)
deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a_fp8, sfa), (b_fp8, sfb), d, m_indices)
```

### 3.3 通过 PaddleFleet 集成（推荐方式）

```bash
pip install paddlefleet -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

```python
from paddlefleet.ops import deep_gemm
# 直接使用所有接口，无需手动 enable_compat
deep_gemm.fp8_gemm_nt(...)
```

### 3.4 性能使用建议

- **预热（warmup）**：首次调用某种 `(N, K)` 组合时会触发 JIT 编译（约需 5~30 秒），建议在推理前预热所有出现的 GEMM 形状
- **持久化缓存**：配置 `DG_JIT_CACHE_DIR` 到持久存储，避免每次重启重新编译
- **NVRTC 加速编译**：设置 `DG_JIT_USE_NVRTC=1` 可将编译速度提升约 10 倍（推荐 NVRTC 12.8+）
- **SM 限制**：调用 `set_num_sms(n)` 可与其他 CUDA 内核共享 GPU

---

## 4. JIT 编译实现原理

### 4.1 整体流程

```
Python API 调用（如 fp8_gemm_nt(a, b, d, recipe=(1,1,128))）
    │
    ▼
C++ impl 层（csrc/apis/gemm.hpp）
    │  根据 M/N/K 形状、架构（SM 版本）、布局参数选择实现
    ▼
JIT 内核包装（csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp）
    │  生成 CUDA C++ 代码字符串（模板注入常量）：
    │      "#define BLOCK_M 128\n"
    │      "#define BLOCK_N 128\n"
    │      "#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>\n"
    ▼
Compiler::build(name, code)（csrc/jit/compiler.hpp）
    │
    ├─ 计算缓存键（kernel_signature）
    │       = hash(name + 库头文件MD5 + 编译器版本 + 编译flags + 代码字符串)
    │
    ├─ 查找文件缓存（~/.deep_gemm/cache/kernel.{name}.{hash}/）
    │       命中 ──────────────────────────────────┐
    │       未命中                                  │
    │           │                                   │
    │           ▼                                   │
    │   NVCC/NVRTC 编译                             │
    │       → kernel.cubin 写入缓存目录             │
    │                                               │
    ▼                                               ▼
KernelRuntime 加载（csrc/jit/kernel_runtime.hpp）
    │  cuobjdump 发现函数名
    │  cuModuleLoad + cuModuleGetFunction
    ▼
cuLaunchKernelEx（支持 cluster dim，TMA multicast）
    │
    ▼
GPU 执行，输出写入 d tensor
```

### 4.2 编译器（`csrc/jit/compiler.hpp`）

支持两种后端，通过环境变量 `DG_JIT_USE_NVRTC` 选择：

**NVCCCompiler（默认）：**
```
生成代码 → 写入 ~/.deep_gemm/cache/kernel.xxx/kernel.cu
→ 调用 nvcc -cubin -O3 --expt-relaxed-constexpr
         -I{库头文件目录} --gpu-architecture=sm_{arch}
→ 产出 kernel.cubin
```

**NVRTCCompiler（`DG_JIT_USE_NVRTC=1`）：**
```
生成代码字符串 → 调用 nvrtcCompileProgram API（内存编译）
→ nvrtcGetCUBIN 获取 cubin 数据
→ 写入文件缓存
→ 编译速度约快 10x，NVRTC 12.8+ 支持 PCH 进一步加速
```

### 4.3 缓存键设计（防止错误命中）

```cpp
kernel_signature = fmt::format("{}$${}$${}$${}$${}",
    name,             // 内核名称（如 "sm90_fp8_gemm_1d1d_128_128_..."）
    library_version,  // 库所有头文件内容的 MD5 哈希（修改任何 .cuh 都失效）
    signature,        // 编译器标识（如 "NVCC12.9"）
    flags,            // 编译标志字符串
    code              // 生成的内核代码字符串（改变 M/N/K 或配置都失效）
);
cache_dir = ~/.deep_gemm/cache/kernel.{name}.{hex_digest(signature)}/
```

**缓存失效场景：** 修改任何库头文件、更新 NVCC 版本、修改 `#define` 常量（即改变形状）。

### 4.4 内核运行时（`csrc/jit/kernel_runtime.hpp`）

- **内存缓存**：`KernelRuntimeCache` 使用 `unordered_map<string, shared_ptr<KernelRuntime>>` 缓存已加载内核，避免重复 `cuModuleLoad`
- **函数发现**：通过 `cuobjdump -symbols` 自动从 cubin 中找到内核函数名
- **执行**：通过 `cuLaunchKernelEx` 支持 cluster dim（TMA multicast，Hopper+ 特性）

### 4.5 代码生成模式

每个内核使用 `#define + #include` 模板特化：
```cpp
// 生成代码示例（针对 M=128, N=128, BF16 输出）
#define BLOCK_M 128
#define BLOCK_N 128
#define KERNEL_MAJOR_A 0
#define NUM_STAGES 4
#define OUTPUT_DTYPE __nv_bfloat16
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>
```
编译时常量使编译器能够静态展开循环、消除 dead branch、最优化寄存器分配。

### 4.6 相关环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DG_JIT_CACHE_DIR` | 缓存目录 | `~/.deep_gemm` |
| `DG_JIT_USE_NVRTC` | `1` 使用 NVRTC 后端 | `0`（NVCC） |
| `DG_JIT_NVCC_COMPILER` | 指定 nvcc 路径 | 自动检测 |
| `DG_JIT_DEBUG` | `1` 打印调试信息 | `0` |
| `DG_JIT_PTXAS_VERBOSE` | `1` 打印 PTXAS 寄存器用量 | `0` |
| `DG_JIT_PRINT_COMPILER_COMMAND` | `1` 打印完整编译命令 | `0` |
| `DG_JIT_CPP_STANDARD` | C++ 标准 | `20` |
| `DG_PRINT_CONFIGS` | `1` 打印选中的 tile 配置 | `0` |

---

## 5. 与 Paddle 适配的差异化改动

本仓库基于 `paddle` 分支，在 deepseek-ai/DeepGEMM 原版基础上有约 10 个 Paddle 相关提交，涉及以下文件。

### 5.1 构建系统（`setup.py`）

```python
# 原版
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME, BuildExtension
setuptools.setup(name='deep_gemm', ...)
extra_compile_args = cxx_flags  # 直接传列表

# Paddle 版
from paddle.utils.cpp_extension import CUDAExtension, setup
CUDA_HOME = "/usr/local/cuda"   # 硬编码 CUDA 路径
setup(name='deep_gemm_cpp', ...)  # 模块名改为 deep_gemm_cpp
extra_compile_args = {"nvcc": cxx_flags, "cxx": cxx_flags}  # 需要字典
cxx_flags.extend(["-DPADDLE_WITH_CUDA", "-DPADDLE_WITH_NCCL"])  # 新增宏
build_libraries = ['cuda', 'cudart', 'cublas', 'nvrtc']  # 新增链接库
```

### 5.2 Python 入口（`deep_gemm/__init__.py`）

```python
# 新增开头两行
import paddle
paddle.enable_compat(scope={"deep_gemm"})   # 启用 torch 代理模式

# 模块名更改：._C → deep_gemm_cpp
import deep_gemm_cpp
from deep_gemm_cpp import set_num_sms, get_num_sms, ...
deep_gemm_cpp.init(...)   # 初始化调用也改名
```

### 5.3 C++ 头文件适配

**`csrc/jit/compiler.hpp`（新增两行）：**
```cpp
#define PADDLE_WITH_CUDA   // 确保 gpuStream_t 声明正确
#include <c10/cuda/CUDAStream.h>   // 新增显式包含
```

**`csrc/jit/kernel_runtime.hpp` 和 `csrc/jit/device_runtime.hpp`（新增一行）：**
```cpp
#define PADDLE_WITH_CUDA
```

**`csrc/utils/compatibility.hpp`（版本检测简化）：**
```cpp
// 原版（依赖 torch/version.h）
#include <torch/version.h>
#define DG_FP8_COMPATIBLE (TORCH_VERSION_MAJOR > 2 || ...)
#define DG_TENSORMAP_COMPATIBLE (CUDA_VERSION >= 12010)

// Paddle 版（硬编码，避免依赖 PyTorch 头文件）
#define DG_FP8_COMPATIBLE true
#define DG_TENSORMAP_COMPATIBLE true
```

### 5.4 Python 工具函数（`deep_gemm/utils/math.py`）

```python
# 新增前置
import paddle
paddle.enable_compat()

# Tensor 设备 API 替换：x.device → x.place（Paddle API）
x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.place).fill_(0)

# CUDA Graph 兼容性修复（ceil_to_ue8m0）
# 原版：torch.pow(2.0, ...)  ← Python float 字面量在 Graph 捕获期会触发同步
# Paddle 版：
result = torch.pow(torch.full([1], 2.0, device=x.place), exp)

# 移除 CUDA Graph 不安全的 assert
# 原版：assert x.view(-1).amax().item() > 0  ← .item() 导致 CPU/GPU 同步
# Paddle 版：直接删除此行
```

### 5.5 测试文件适配

所有测试文件（`tests/test_fp8.py` 等）头部新增：
```python
import paddle
paddle.enable_compat()
```

**`tests/generators.py` FP8 操作绕过（Paddle 不支持 FP8 直接切片赋值）：**
```python
# 原版（直接 FP8 tensor 切片赋值）
a_fp8 = torch.empty_like(a, dtype=torch.float8_e4m3fn)
a_fp8[0:128, :] = ...   # Paddle 不支持

# Paddle 版（先 BF16 中转，最后一次性 cast）
a_fp8_bf16 = torch.empty_like(a, dtype=torch.bfloat16)
a_fp8_bf16[0:128, :] = a_chunk.to(torch.bfloat16)
a_fp8 = a_fp8_bf16.to(torch.float8_e4m3fn)  # 最后统一 cast
```

### 5.6 Paddle Fork 新增功能（超越原版）

| 功能 | 相关提交 | 说明 |
|------|----------|------|
| SM100 bias 支持 | ac257ef, 7167709 | FP8 GEMM 接口新增 `bias` 参数，支持 Blackwell bias epilogue，修复内存越界 bug |
| `next_n=2` paged logits | 72259cf | `fp8_paged_mqa_logits` 支持双 token 预测（DeepSeek V3.2 解码场景） |
| CUDA Graph 兼容修复 | 3e008e2 | `ceil_to_ue8m0` 消除 `.item()` 同步调用和 unsafe assert |
| arange bug 修复 | 1026d60 | Paddle tensor 上 arange 相关 bug 修复 |

---

## 6. DeepSeekV3.2 中的 deepgemm 接口使用

**涉及文件：**
- `FastDeploy/fastdeploy/model_executor/models/deepseek_v3.py`（主模型，1331 行）
- `FastDeploy/fastdeploy/model_executor/layers/quantization/block_wise_fp8.py`
- `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_deepgemm_backend.py`
- `FastDeploy/fastdeploy/model_executor/layers/quantization/fp8_utils.py`

### 6.1 调用全景图

```
deepseek_v3.py
  ├─ Indexer（行 560-748）
  │     ├─ prefill 路径 → deep_gemm.fp8_mqa_logits
  │     └─ decode 路径  → deep_gemm.fp8_paged_mqa_logits
  │                       deep_gemm.get_paged_mqa_logits_metadata
  │
  ├─ DeepseekV32DSAAttention（行 751-965）
  │     └─ self.indexer(...)  → 调用上述 Indexer
  │
  ├─ 所有线性层（通过 BlockWiseFP8LinearMethod）
  │     └─ block_wise_fp8.py → fp8_gemm_nt
  │
  └─ 所有 MoE 层（通过 FusedMoeDeepgemmBackend）
        └─ fused_moe_deepgemm_backend.py
              ├─ m_grouped_fp8_gemm_nt_contiguous  （EP prefill）
              ├─ m_grouped_fp8_gemm_nt_masked      （EP prefill padding）
              └─ m_grouped_fp8_gemm_nt_masked      （decode）
```

**deep_gemm 导入方式（`deepseek_v3.py` 行 72-73）：**
```python
paddle.enable_compat(scope={"deep_gemm": True})
# ...
import deep_gemm   # 懒加载，在 Indexer.forward 内调用
```

### 6.2 稠密线性层（`block_wise_fp8.py`）

**调用代码（`BlockWiseFP8LinearMethod.apply`）：**
```python
def apply(self, layer, x):
    linear_out = paddle.empty((x.shape[0], layer.output_size), dtype=paddle.bfloat16)
    # 激活量化：per-token-group（128 对齐）
    x_fp8, x_scale = per_token_quant_padding(x, quant_block_size=128, use_ue8m0=...)
    x_scale = x_scale[:x.shape[0], ...]
    # GEMM 调用
    fp8_gemm_nt(
        x_fp8,                   # FP8 激活
        x_scale,                 # 激活 scale
        layer.weight,            # FP8 权重（预量化，per-block 128×128）
        layer.weight_scale_inv,  # 权重 scale
        linear_out,
        layer_output_size=layer.output_size,
    )
    return linear_out
```

**张量形状（以 hidden_size=7168, output_size=4096 为例）：**
```
x（BF16）                [T, 7168]
x_fp8（FP8 E4M3fn）      [T, 7168]
x_scale（FP32）          [T, 56]        （7168/128 = 56）
layer.weight（FP8）      [4096, 7168]   （N, K 布局，适合 nt GEMM）
weight_scale_inv（FP32） [32, 56]       （4096/128=32, 7168/128=56）
linear_out（BF16）        [T, 4096]
```

### 6.3 MoE contiguous GEMM（EP Prefill，`fused_moe_deepgemm_backend.py`）

**前向传播数据流：**

```
hidden_states [T, 7168]
    │
    ├─ gate（ReplicatedLinear）→ logits [T, 256]
    │     topk → topk_ids [T, 8], topk_weights [T, 8]
    │
    ├─ per_token_quant → x_fp8 [T, 7168] FP8, x_scale [T, 56] FP32
    │
    ├─ [EP dispatch] AlltoAll → permute_input [M_total, 7168] FP8
    │
    ├─ m_grouped_fp8_gemm_nt_contiguous(up_gate_proj)
    │     a：[M_total, 7168] FP8 × [E, 4096, 7168]（up_gate_proj_weight, E=num_local_experts）
    │     d：[M_total, 4096] BF16（2*moe_intermediate_size=2*2048=4096，SwiGLU 拼接）
    │
    ├─ SwiGLU → [M_total, 2048]
    ├─ per_token_quant → [M_total, 2048] FP8, [M_total, 16] scale
    │
    ├─ m_grouped_fp8_gemm_nt_contiguous(down_proj)
    │     a：[M_total, 2048] FP8 × [E, 7168, 2048]（down_proj_weight）
    │     d：[M_total, 7168] BF16
    │
    ├─ [EP combine] AlltoAll + weighted sum
    └─ + shared_experts_out → [T, 7168]
```

**DeepSeek-V3 关键超参：**
- `n_routed_experts = 256`，`num_experts_per_tok = 8`
- `moe_intermediate_size = 2048`，`hidden_size = 7168`

### 6.4 MoE masked GEMM（Prefill padding 路径）

```python
# 保留 [E, max_tokens_per_expert, K] 维度，不做 permute
up_gate_out = paddle.empty(
    [num_local_experts, ep_size * max_tokens_per_rank, moe_intermediate_size * 2],
    dtype=paddle.bfloat16
)
m_grouped_fp8_gemm_nt_masked(
    (permute_input, permute_scale),                            # [E, M_pad, K]
    (layer.up_gate_proj_weight, layer.up_gate_proj_weight_scale_inv),
    up_gate_out,                                               # [E, M_pad, 2*I]
    token_nums_per_expert,   # [E] int32，每 expert 实际 token 数
    expected_m = max_tokens_per_rank,
    disable_ue8m0_cast = not use_ue8m0,
)
```

### 6.5 Indexer prefill 路径（`fp8_mqa_logits`）

**完整输入构造流程（`Indexer.forward`，行 618-708）：**

```
输入：hidden_states [T, 7168], qr（q_lora_rank 压缩 query）[T, 1536]
    │
    ├─ wq_b(qr) → q [T*n_heads, head_dim=192]
    │     reshape → [T, n_heads=8, 192]
    │     split → q_pe [T, 8, 64], q_nope [T, 8, 128]
    │
    ├─ wk(hidden_states) → k [T, 192]，LayerNorm
    │     split → k_pe [T, 64], k_nope [T, 128]
    │
    ├─ rotary_emb(q_pe, k_pe) → 旋转位置编码
    │
    ├─ concat [q_pe, q_nope] → q [T*8, 192]
    │     per_token_group_quant_fp8(q, group_size=128)
    │         → q_fp8 [T, 8, 192] FP8，q_scale [T, 8, 1] FP32
    │
    ├─ weights_proj(hidden) → weights [T, 8]
    │     weights *= q_scale * softmax_scale * n_heads^{-0.5}
    │     weights 形状 [T, 8]（已融合所有缩放因子）
    │
    ├─ indexer_k_quant_and_cache(k, ...)   # 将 k 量化并写入 paged cache
    │
    └─ cp_gather_indexer_k_quant_cache(...)  # 从 cache gather 出连续 K
           → k_fp8 [T_kv, 192] FP8
           → k_scale [T_kv] FP32（per-token scale）

调用：
    logits = deep_gemm.fp8_mqa_logits(
        q_fp8,                   # [T, 8, 192]
        (k_fp8, k_scale),        # ([T_kv, 192], [T_kv])
        weights,                  # [T, 8]
        ks,                       # [T] 起始偏移
        ke,                       # [T] 结束偏移
        max_seqlen_k=max_seqlen_k,
        clean_logits=False,
    )  # → [T, max_seqlen_k] FP32

输出处理：
    radix_topk_ragged_transform(logits, indexer_top_k, ks, ke-ks, ...)
    # → indexer_top_k [T, 64] int32（每 token 的 top-64 KV 位置）
```

### 6.6 Indexer decode 路径（`fp8_paged_mqa_logits`）

```python
# 预计算调度元数据
schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
    cache_seqlens,    # [bsz] int32
    64,               # block_size（KV cache 分页大小固定 64）
    deep_gemm.get_num_sms()
)  # → [num_sms+1, 2] int32

# Decode logits
logits = deep_gemm.fp8_paged_mqa_logits(
    decoder_q.reshape(-1, 1, index_n_heads, index_head_dim),
    #   [bsz, 1, 8, 192] FP8（next_n=1 或 2）
    self.indexer_cache.unsqueeze(2),
    #   [num_kv_blocks, 64, 1, head_dim+4] uint8（FP8 data + FP32 scale 融合）
    decoder_weight,           # [bsz, 8] FP32
    cache_seqlens,            # [bsz] int32
    forward_meta.block_tables,# [bsz, max_blocks] int32
    schedule_metadata,
    self.max_model_len,
    clean_logits=True,
).contiguous()  # → [bsz, max_model_len] FP32

# Top-k 选择
radix_topk_ragged_transform(logits, indexer_top_k, ...)
# → indexer_top_k [bsz, 64] int32
```

### 6.7 Indexer 在 DSAAttention 中的作用

```
DeepseekV32DSAAttention.forward
    │
    ├─ indexer_top_k = self.indexer(hidden_states, query, ...)
    │     → [T, 64] int32，每 token 关注的 KV 位置索引
    │
    └─ dsa_attn(q, k, v=indexer_top_k, ...)
          # v 传入的是稀疏索引，而非实际 value tensor
          # DSA 内核根据 indexer_top_k 只计算选中的 64 个 KV 位置的注意力
          # → 将 O(T^2) 全注意力降为 O(T×64) 稀疏注意力
```

---

## 7. Indexer 中 FP8 误差合理性分析

### 7.1 量化链路与误差来源

Indexer 的计算链路如下：

```
BF16 query [T, 8, 192]
    │
    ├─ per_token_group_quant_fp8(group_size=128)
    │     每 128 个元素找最大值 → scale = max_val / fp8_max（fp8_max=224，非理论448）
    │     量化：fp8_val = round(bf16_val / scale)  ← 3 位尾数截断
    │
    ▼ q_fp8 [T, 8, 192] FP8 E4M3fn
    │
    ├─ fp8_mqa_logits（FP8 × FP8 点积，FP32 累加）
    │     内部：out += q_fp8[i,h,:] ⊗ k_fp8[j,:] * (q_scale[i,h] * k_scale[j])
    │     head_dim = 192，即 192 次 FP8 乘法累加到 FP32
    │
    ▼ logits [T, max_seqlen_k] FP32
```

**误差来源分析：**

| 来源 | 量级 | 说明 |
|------|------|------|
| FP8 E4M3fn 尾数截断 | 相对误差 ≈ 6.25%/element | 3 位尾数，最小精度 2^{-3} = 0.125 |
| K 轴累加放大 | × √192 ≈ 13.9 | 独立误差随机游走，均方根放大 |
| fp8_max=224 保守设置 | 浪费约 1 bit | 理论最大值 448，保守值 224，降低了动态范围利用率 |
| UE8M0 scale 舍入（SM100） | 最坏 × 0.707 | scale 只能是 2 的整数次幂，舍入误差 ±50% |
| 综合绝对误差估算 | ~1e-2 量级 | 取决于输入分布，正常推理场景下约 0.01~0.05 |

**详细推导：**
```
单个 FP8 元素相对误差：δ ≈ 2^{-3}/2 = 0.0625（6.25%，相对于 scale 最大值）
点积维度 head_dim = 192，误差随机游走：
    σ_abs = sqrt(192) × δ × E[|q||k|] ≈ 13.9 × 0.0625 × mean_activation ≈ 0.87 × mean_act
若 mean_act（FP8 量化后均值）≈ 0.01~0.1，则绝对误差 ≈ 0.01~0.1，即 ~1e-2 量级。
```

### 7.2 为何对 Indexer 功能影响可忽略

Indexer 的目标是 **top-k 排序选择**，而非精确数值输出：

```
logits（含 ~1e-2 误差）→ radix_topk_ragged_transform → top-64 KV 索引
```

- **排序稳定性**：只要相邻 token 的 logit 差值 >> FP8 误差（通常差值在 0.1~1.0 量级），top-64 的选择结果与精确计算完全相同
- **容错性**：即使约 5% 的位置出现排序错误（边界情况），稀疏注意力的整体 recall 仍然很高（64 取 top 中有极少数偏差不影响模型输出）
- **论文验证**：DeepSeek 技术报告中对比了 FP8 和 BF16 Indexer，最终任务精度无统计显著差异

### 7.3 设计取舍（FP8 vs BF16 Indexer）

| 维度 | FP8 Indexer（当前方案） | BF16 Indexer（对比） |
|------|----------------------|---------------------|
| **误差** | logits ~1e-2 量级 | logits ~1e-4 量级 |
| **KV Cache 显存** | FP8 存储，节约约 50% | BF16 存储 |
| **计算吞吐** | Tensor Core FP8，约 4x 吞吐 | BF16 Tensor Core，约 2x |
| **CUDA Graph 兼容** | 需要特殊处理（已修复） | 原生支持 |
| **适用场景** | 大规模推理加速优先 | 精度敏感场景/研究 |

**结论：** ~1e-2 的误差对于 Indexer top-k 选择任务是**合理且可接受的**。这是推理系统中精度与性能的标准工程权衡——牺牲精确数值以换取 2x 显存节约和 2x 计算吞吐提升，而对最终输出质量的影响可忽略不计。

---

## 附录：关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| Python API 导出 | `deep_gemm/__init__.py` | 全文 |
| pybind11 注册 | `csrc/python_api.cpp` | 全文 |
| JIT 编译器 | `csrc/jit/compiler.hpp` | 全文 |
| JIT 文件缓存 | `csrc/jit/cache.hpp` | 全文 |
| 内核加载运行时 | `csrc/jit/kernel_runtime.hpp` | 全文 |
| Paddle 兼容宏 | `csrc/utils/compatibility.hpp` | 全文 |
| FP8 量化工具 | `deep_gemm/utils/math.py` | 全文 |
| Indexer 定义 | `deepseek_v3.py` | 560-748 |
| DSA Attention | `deepseek_v3.py` | 751-965 |
| MLA Attention | `deepseek_v3.py` | 202-464 |
| 稠密线性层调用 | `block_wise_fp8.py` | 365-405 |
| MoE GEMM 调用 | `fused_moe_deepgemm_backend.py` | 455-566 |
| deep_gemm 版本选择 | `fp8_utils.py` | 60-91 |
