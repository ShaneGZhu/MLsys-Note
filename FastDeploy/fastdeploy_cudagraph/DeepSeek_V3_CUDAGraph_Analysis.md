# FastDeploy CUDA Graph 适配开发指南 - DeepSeek V3.2 Decode 阶段

## 目录

1. [项目架构概览](#1-项目架构概览)
2. [CUDA Graph 核心机制](#2-cuda-graph-核心机制)
3. [Graph Break 根源分析](#3-graph-break-根源分析)
4. [DeepSeek V3 特殊性分析](#4-deepseek-v3-特殊性分析)
5. [Decode 阶段 CUDA Graph 适配方案](#5-decode-阶段-cuda-graph-适配方案)
6. [快速上手路线图](#6-快速上手路线图)
7. [关键文件索引](#7-关键文件索引)

---

## 1. 项目架构概览

### 1.1 目录结构

```
/root/paddlejob/workspace/env_run/output/zhushengguang/FastDeploy/fastdeploy/model_executor/
├── graph_optimization/                    # CUDA Graph 核心实现
│   ├── cudagraph_piecewise_backend.py     # 图捕获/重放管理
│   ├── graph_optimization_backend.py       # 顶层封装
│   ├── decorator.py                        # @support_graph_optimization
│   ├── dynamic_dims_marker.py              # 动态维度标记
│   └── utils.py                           # warmup 模式守卫
├── models/                                # 模型实现
│   ├── deepseek_v3.py                     # DeepSeek V3 主模型
│   ├── qwen3moe.py, qwen3.py, qwen2.py    # 参考模型
│   └── ernie4_5_moe.py                    # 参考模型
├── layers/                                # 层实现
│   ├── attention/                         # 注意力后端
│   │   ├── attention.py                   # 统一入口
│   │   ├── mla_attention_backend.py       # MLA 注意力 ⭐
│   │   ├── flash_attn_backend.py          # FlashAttention
│   │   └── ...
│   └── linear.py                          # 线性层（含 KVBatchLinear）
└── forward_meta.py                        # ForwardMeta 数据类
```

### 1.2 模型继承关系

```
nn.Layer
└── DeepSeekV3PretrainedModel
    └── DeepseekV3ForCausalLM (主模型类, @support_graph_optimization)
        └── DeepseekV32ForCausalLM

    └── DeepSeekV3Model (backbone)
        └── DeepSeekV3DecoderLayer
            ├── DeepseekV3MLAAttention / DeepseekV32DSAAttention
            └── DeepSeekV3MoE / DeepSeekV3MLP
```

---

## 2. CUDA Graph 核心机制

### 2.1 图捕获触发条件

核心逻辑在 `cudagraph_piecewise_backend.py:154-166`：

```python
should_skip_cudagraph = not kwargs["forward_meta"].step_use_cudagraph

if not should_skip_cudagraph:
    if exist_prefill:
        # Prefill 或 Mixed 阶段 - 跳过 CUDA Graph
        should_skip_cudagraph = True
    else:
        # Decode 阶段 - 使用 CUDA Graph
        should_skip_cudagraph = real_shape > self.max_captre_size
```

**关键结论**：Decode 阶段（无 prefill）且 token 数量在捕获范围内时，才会尝试捕获 CUDA Graph。

### 2.2 三种执行模式

| 模式 | `graph_opt_level` | `full_cuda_graph` | Prefill | Decode |
|------|------------------|-------------------|---------|--------|
| **Dynamic** | 0 | - | Dynamic | Dynamic + CUDAGraph |
| **Static Full Graph** | >0 | True | Dynamic | Static + CUDAGraph |
| **Static Split Graph** | >0 | False | Static + CUDAGraph | Dynamic + CUDAGraph |

### 2.3 图捕获完整流程

```
GraphOptBackend.__call__(**kwargs)
    │
    ├─► 检查 use_cudagraph 配置
    │
    ├─► 判断 should_skip_cudagraph
    │       (基于 step_use_cudagraph, exist_prefill, real_shape)
    │
    └─► CudaGraphPiecewiseBackend.__call__(**kwargs)
            │
            ├─► 确定 padding_real_shape
            │
            ├─► 获取/创建 ConcreteSizeEntry
            │
            ├─► 首次捕获：
            │       warmup (默认2次)
            │       graphs.CUDAGraph.capture_begin()
            │       capture_custom_allreduce():
            │           entry.runnable(**kwargs)
            │       graphs.CUDAGraph.capture_end()
            │
            └─► 重放：
                    entry.cuda_graph.replay()
```

### 2.4 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_cudagraph` | True (GPU) | 是否启用 CUDA Graph |
| `graph_opt_level` | 0 | 0=dynamic, 1=static, 2=static+CINN |
| `full_cuda_graph` | True | 完整图 vs 分割图 |
| `cudagraph_capture_sizes` | None | Decode 捕获的 token 数量列表 |
| `cudagraph_num_of_warmups` | 2 | warmup 次数 |
| `max_capture_size` | 512 | 最大捕获尺寸 |

---

## 3. Graph Break 根源分析

### 3.1 什么是 Graph Break

Graph Break 是指 CUDA Graph 无法捕获的执行路径中的某些操作，导致图被分割成多段，降低性能甚至完全无法使用 CUDA Graph。

### 3.2 Graph Break 的主要成因

#### 3.2.1 GPU→CPU 同步操作（最严重）

| 操作 | 文件位置 | 影响 |
|------|----------|------|
| `.item()` | flash_attn_backend.py:392,415,420,450 | 强制 CPU 分支 |
| `.cpu()` | append_attn_backend.py:94-109 | D2H 复制 |
| `.max().cpu()` | moba_attention_backend.py:107-108 | D2H 复制 |

**示例代码**（flash_attn_backend.py:420）：
```python
use_fa_do_prefill = forward_meta.max_len_tensor_cpu[1].item() > 0  # ❌ graph break
```

#### 3.2.2 Python 动态条件分支

```python
# mla_attention_backend.py:590
if k is not None:     # ❌ Python 条件，CUDA Graph 无法静态捕获
    # Prefill 分支
    ...
# Decode 分支
...
```

#### 3.2.3 环境变量读取

```python
# mla_attention_backend.py:605
if int(os.getenv("USE_FLASH_MLA", "0")) == 0:   # ❌ 环境变量导致不确定性
    fmha_out = multi_head_latent_attention(...)
else:
    fmha_out = flash_mla_impl(...)
```

#### 3.2.4 Custom Ops 黑名单

`append_attn_backend.py:180-196` 将某些 custom op 加入黑名单：

```python
paddle.set_flags({
    flag: ",".join(set(
        paddle.get_flags(flag)[flag].split(",") + [
            "custom_op.static_op_append_attention_with_output_",
            "custom_op.static_op_get_block_shape_and_split_kv_block",
        ]
    ))
})
```

### 3.3 Graph Break 检测机制

`graph_optimization_backend.py:73-75`：

```python
# Check generated code has no break graph
new_code = new_guarded_codes[0][0][0]
if any(name.startswith("$") for name in new_code.co_names):
    raise RuntimeError("Model has breakgraph, please set env SOT_LOG_LEVEL=3 to check it.")
```

### 3.4 常见 Graph Break 操作一览

| 操作 | 是否支持 CUDA Graph | 说明 |
|------|---------------------|------|
| `.item()` | ❌ 不支持 | GPU→CPU 同步 |
| `.cpu()` / `.numpy()` | ❌ 不支持 | D2H 复制 |
| `paddle.where` | ✅ 支持 | 条件操作 |
| `index_select` | ✅ 支持 | 索引选择 |
| `paddle.matmul` | ✅ 支持 | 矩阵乘法 |
| `any()` / `all()` | ⚠️ 条件支持 | 在捕获区域外可用 |
| For 循环（可变边界） | ⚠️ 条件支持 | 静态边界可用 |

---

## 4. DeepSeek V3 特殊性分析

### 4.1 MLA 架构特点

DeepSeek V3 使用 **Multi-head Latent Attention (MLA)**，与传统 MHA 的区别：

```
传统 MHA:
  Q, K, V → Attention → Output
  (K, V 需要完整存储用于 decode)

MLA:
  compressed_kv = down_proj(KV)
  decode: Q + compressed_kv + RoPE(K_pe) → Attention
  (只需存储 compressed_kv，大幅减少 KV 缓存）
```

### 4.2 Decode 阶段关键路径

`deepseek_v3.py:406-440`：

```python
if need_do_decode:  # max_dec_len_this_time
    # Q 投影
    q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]), proj_type="k")
    q_input = paddle.concat([q_nope_out, query_pe], axis=-1)

    # MLA Attention（使用 compressed_kv）
    fmha_out_decode = self.mla_attn(
        q=q_input,
        k=None, v=None,  # decode 不需要完整 K/V
        qkv=None,
        compressed_kv=compressed_kv,  # ⭐ 压缩后的 KV
        k_pe=key_pe,                  # ⭐ RoPE 位置编码
        forward_meta=forward_meta,
    )
```

### 4.3 MLAAttentionBackend 的问题

`mla_attention_backend.py` 存在多处 graph break：

```python
# 第 590 行 - 动态条件分支
if k is not None:     # ❌ prefill vs decode 动态分支
    prefill_mla_write_cache(...)
    fmha_out = self.flash_attn_func(...)
    return fmha_out

# Decode 路径
decode_mla_write_cache(...)

# 第 605 行 - 环境变量不确定性
if int(os.getenv("USE_FLASH_MLA", "0")) == 0:   # ❌
    fmha_out = multi_head_latent_attention(...)
else:
    fmha_out = flash_mla_impl(...)
```

### 4.4 KVBatchLinear 的特殊性

`layers/linear.py:946` 的 `KVBatchLinear`：

```python
def forward_k_b(self, x: paddle.Tensor) -> paddle.Tensor:
    out = paddle.bmm(x, self.k_b_proj_weight)  # batch matmul
    return out
```

使用 `paddle.bmm` 进行批量矩阵乘法，这是 decode 阶段的关键操作。

---

## 5. Decode 阶段 CUDA Graph 适配方案

### 5.1 适配策略

#### 策略 1：消除 Graph Break（推荐）

**目标**：修改 attention backend，移除所有导致 graph break 的操作。

**修改点**：

1. **移除 `.item()` 调用**
   - 位置：`mla_attention_backend.py`
   - 方法：将条件判断提前到图捕获前，使用静态布尔值

2. **消除动态条件分支**
   - 位置：`mla_attention_backend.py:590`
   - 方法：使用 `paddle.where` 或创建两个独立的 capture entry

3. **消除环境变量依赖**
   - 位置：`mla_attention_backend.py:605`
   - 方法：在初始化时读取环境变量，缓存为实例属性

#### 策略 2：纯 Decode 路径隔离

**目标**：确保纯 Decode 请求（`need_do_prefill=False`）走稳定路径。

**修改点**：

1. 在 `gpu_model_runner.py` 中正确设置 `forward_meta`
2. 确保 `exist_prefill=False` 时 `step_use_cudagraph=True`
3. 验证 Decode batch size 在 `cudagraph_capture_sizes` 范围内

### 5.2 详细修改方案

#### 5.2.1 修改 MLAAttentionBackend

**文件**：`layers/attention/mla_attention_backend.py`

```python
# Before (line 590):
if k is not None:
    # Prefill 分支
    ...

# After:
# 使用 forward_meta.forward_mode.is_decode() 替代动态检查
is_decode_mode = forward_meta.forward_mode.is_decode()
```

**文件**：`layers/attention/mla_attention_backend.py`

```python
# Before (line 605):
if int(os.getenv("USE_FLASH_MLA", "0")) == 0:
    fmha_out = multi_head_latent_attention(...)
else:
    fmha_out = flash_mla_impl(...)

# After:
# 在 __init__ 中读取并缓存
self.use_flash_mla = int(os.getenv("USE_FLASH_MLA", "0")) == 0

# 在 forward 中使用
if self.use_flash_mla:
    fmha_out = multi_head_latent_attention(...)
else:
    fmha_out = flash_mla_impl(...)
```

#### 5.2.2 修改 ForwardMeta

**文件**：`forward_meta.py`

添加静态标志用于 CUDA Graph 决策：

```python
@dataclass
class ForwardMeta:
    # ... existing fields ...

    # 新增：静态决策标志
    is_pure_decode: bool = False  # 纯 decode 模式标志
    decode_batch_size: int = 0    # decode batch 大小
```

### 5.3 验证方法

#### 5.3.1 单元测试

```python
# test_mla_cudagraph.py
def test_pure_decode_cudagraph():
    """测试纯 Decode 阶段的 CUDA Graph 捕获"""
    # 1. 设置模型为纯 decode 模式
    # 2. 执行多次 warmup
    # 3. 捕获 CUDA Graph
    # 4. 验证 replay 输出一致
```

#### 5.3.2 集成测试

```bash
# 1. 运行 benchmark
python benchmark_serving.py \
    --model /root/paddlejob/workspace/env_run/output/models/DeepSeek-V3.2-Exp-BF16-5layers \
    --backend hotteNN \
    --base_url http://localhost:18002 \
    --num-requests 100

# 2. 检查日志
# 预期：Initial test run successful
# 预期：无 "TransferEncodingError"
```

---

## 6. 快速上手路线图

### 阶段 1：理解代码（1-2 天）

1. **阅读核心文件**
   - [cudagraph_piecewise_backend.py](fastdeploy/model_executor/graph_optimization/cudagraph_piecewise_backend.py) - 图捕获/重放逻辑
   - [graph_optimization_backend.py](fastdeploy/model_executor/graph_optimization/graph_optimization_backend.py) - 顶层封装
   - [mla_attention_backend.py](fastdeploy/model_executor/layers/attention/mla_attention_backend.py) - MLA 实现

2. **理解执行流程**
   - 理解 `ForwardMeta` 如何传递
   - 理解 `step_use_cudagraph` 决策点
   - 理解 Decode 阶段的特殊处理

### 阶段 2：问题定位（1 天）

1. **运行单步调试**
   ```bash
   export SOT_LOG_LEVEL=3
   python -c "from deepseek_v3 import *; ..."
   ```

2. **检查 Graph Break**
   - 启用 break graph 检测
   - 找出所有 `.item()` 和 `.cpu()` 调用

3. **分析 Worker 崩溃原因**
   - 检查 dmesg 是否有 OOM
   - 检查 nvidia-smi 显存使用

### 阶段 3：修改实现（2-3 天）

1. **消除 Graph Break**
   - 修改 `mla_attention_backend.py`
   - 确保 Decode 路径无动态分支

2. **验证 CUDA Graph 捕获**
   - 添加日志确认捕获成功
   - 验证 replay 正确性

### 阶段 4：性能优化（1-2 天）

1. **优化捕获尺寸**
   - 设置合理的 `cudagraph_capture_sizes`
   - 避免过多的小 batch capture entry

2. **Benchmark 对比**
   - 对比 Dynamic vs CUDAGraph 性能
   - 验证 Decode 延迟改善

---

## 7. 关键文件索引

### 7.1 CUDA Graph 核心

| 文件 | 行数 | 说明 |
|------|------|------|
| `graph_optimization/cudagraph_piecewise_backend.py` | 300 | 图捕获/重放核心逻辑 |
| `graph_optimization/graph_optimization_backend.py` | 180 | 顶层封装 |
| `graph_optimization/decorator.py` | 134 | `@support_graph_optimization` |
| `graph_optimization/dynamic_dims_marker.py` | 200 | 动态维度标记 |

### 7.2 Attention 实现

| 文件 | 说明 | Graph Break 风险 |
|------|------|-----------------|
| `layers/attention/attention.py` | 统一入口 | 低 |
| `layers/attention/mla_attention_backend.py` | MLA 注意力 | **高** |
| `layers/attention/flash_attn_backend.py` | FlashAttention | **高** |
| `layers/attention/append_attn_backend.py` | Append 模式 | 中 |

### 7.3 DeepSeek V3 模型

| 文件 | 行数 | 说明 |
|------|------|------|
| `models/deepseek_v3.py` | 1100+ | 主模型实现 |
| `layers/linear.py` | 1100+ | 包含 KVBatchLinear |
| `layers/rotary_embedding.py` | 600+ | RoPE 实现 |

### 7.4 其他参考模型

| 文件 | 说明 |
|------|------|
| `models/qwen3moe.py` | MoE + CUDA Graph 参考 |
| `models/ernie4_5_moe.py` | Ernie MoE 参考 |
| `models/glm4_moe.py` | GLM MoE 参考 |

---

## 附录：常见问题

### Q1: 是否需要熟悉所有算子实现？

**不需要**。重点关注：
1. **Attention 后端** - 最容易导致 graph break
2. **动态shape操作** - 如 `.item()`, `.cpu()`
3. **条件分支** - Python if vs paddle.where

### Q2: 什么算子约束了图的捕获？

最关键的约束来自：
1. **GPU→CPU 同步操作** - `.item()`, `.cpu()`
2. **动态条件分支** - `if tensor_value > 0`
3. **环境变量读取** - `os.getenv()`

### Q3: 如何判断能否进入 CUDA Graph？

检查 `CudaGraphPiecewiseBackend.__call__()` 的决策逻辑：

```python
# 满足以下条件可以进入：
# 1. step_use_cudagraph = True
# 2. 不是 prefill 阶段 (exist_prefill = False)
# 3. token 数量在捕获范围内 (real_shape <= max_capture_size)
```

---

*文档版本：1.0*
*最后更新：2026-04-07*
