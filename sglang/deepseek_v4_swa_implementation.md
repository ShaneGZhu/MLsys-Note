# DeepSeek V4 SWA (Sliding Window Attention) 实现分析

## 1. 架构概览

DeepSeek V4 使用**压缩稀疏注意力**，每层通过 `compress_ratio` 定义行为：

| compress_ratio | 层类型 | 注意力范围 |
|---|---|---|
| 0 (c0) | 纯 SWA 层 | 仅最近 128 tokens |
| 4 (c4) | SWA + 4x 压缩 | SWA(128) + 全局 4x 压缩 KV |
| 128 (c128) | SWA + 128x 压缩 | SWA(128) + 全局 128x 压缩 KV |

**所有层都维护 SWA KV cache（窗口大小 128）**，c4/c128 层额外维护压缩后的全局 KV。

配置来源：`python/sglang/srt/configs/deepseek_v4.py`
```python
window_size: int = 128
compress_ratios: List[int]  # 每层的 compress_ratio，如 [0, 4, 128, 4, 128, ...]
```

---

## 2. SWA 内存池

### 2.1 核心类层次

```
BaseSWAKVPool (base_swa_memory_pool.py)
└── DeepSeekV4TokenToKVPool (deepseek_v4_memory_pool.py)
    ├── swa_kv_pool: DeepSeekV4SingleKVPool   # 所有层共用的 SWA KV 存储
    ├── c4_kv_pool: HiSparseC4DevicePool      # c4 压缩 KV
    ├── c128_kv_pool: DeepSeekV4SingleKVPool   # c128 压缩 KV
    ├── full_to_swa_index_mapping: Tensor      # full 索引 → SWA 索引映射
    └── cached_loc: Optional[Tensor]           # 缓存的翻译结果
```

### 2.2 SWA KV 存储格式

每个 token 占 **584 bytes**（`DeepSeekV4SingleKVPool.create_buffer`）：

```
584 bytes/token:
├── qk_nope_head_dim FP8:  448 bytes
├── qk_rope_head_dim BF16: 128 bytes (dim=64, bf16=2 bytes)
├── FP8 量化 scales:         7 bytes (448 / block_size_64)
└── scale padding:            1 byte
```

### 2.3 内存池大小计算

`python/sglang/srt/model_executor/pool_configurator.py` 中 `DSV4PoolConfigurator`：

```
SWA pool size = full_pool_tokens × swa_full_tokens_ratio (默认 0.1)
c4 pool size  = full_pool_tokens / 4
c128 pool size = full_pool_tokens / 128
```

### 2.4 索引映射机制

SWA pool 和 full pool 使用不同的索引空间。通过 `full_to_swa_index_mapping` 做翻译：

```python
# deepseek_v4_memory_pool.py:505-508
def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
    assert self.full_to_swa_index_mapping is not None
    return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)
```

该映射在 allocator 初始化时注册（`register_mapping`），在 token 分配/回收时更新。

### 2.5 SWA 翻译缓存优化

环境变量 `SGLANG_OPT_CACHE_SWA_TRANSLATION=True` 启用后，每次 forward 只在第一个 SWA 层做一次索引翻译，后续层复用：

```python
# deepseek_v4_memory_pool.py:756-761
if self._should_cache_swa:
    if layer_id == self.start_layer or self.cached_loc is None:
        self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
    swa_loc = self.cached_loc
else:
    swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
```

失效场景：
- `register_mapping()` 被调用时（mapping 变化）
- 手动调用 `invalidate_loc_cache()`

---

## 3. SWA KV 写入路径

### 3.1 标准路径（HIP/AMD）

```python
# deepseek_v4.py:582-605 (_forward_prepare 中 _is_hip=True 分支)
token_to_kv_pool = get_token_to_kv_pool()
swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(forward_batch.out_cache_loc)
swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]

# 融合 kernel: q_norm + kv_norm + rope + SWA cache 写入
q = fused_qk_norm_rope_swa_store(
    q=q, kv=kv,
    swa_cache=swa_cache,
    swa_loc=swa_loc,
    swa_page_size=swa_page_size,
    ...
)
```

### 3.2 Fused 路径（CUDA）

`set_swa_key_buffer_radix_fused()`：将 norm + rope + quantize + store 融合为单个 JIT kernel：

```python
# deepseek_v4_memory_pool.py (set_swa_key_buffer_radix_fused)
fused_k_norm_rope_flashmla(
    input=cache_k,
    output=swa_cache,
    indices=swa_loc,
    ...
)
```

底层调用 `sglang/jit_kernel/dsv4/fused_k_norm_rope_flashmla` 和 `fused_store_cache`。

### 3.3 store_cache（Backend 入口）

```python
# deepseek_v4_backend.py:982
if save_kv_cache:
    self.store_cache(layer_id, swa_k, forward_batch)
```

---

## 4. SWA KV 读取与 Attention 计算

### 4.1 Decode 路径

```python
# deepseek_v4_backend.py:983-1065
swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
# reshape: [num_pages, bytes_per_page] → [num_pages, swa_window, 1, kv_dim]
swa_k_cache = swa_k_cache[:, :swa_window_size * k_cache_total_dim].view(
    swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
)

# c4/c128 层额外获取压缩 KV
if compress_ratio == 4:
    extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
elif compress_ratio == 128:
    extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)

# Flash MLA attention: SWA + 压缩 KV 联合计算
o = flash_mla.flash_mla_with_kvcache(
    q=q,
    k_cache=swa_k_cache,              # SWA 部分（所有层）
    indices=swa_page_indices,          # SWA page 索引
    topk_length=swa_topk_lengths,      # 有效 SWA token 数（≤128）
    extra_k_cache=extra_k_cache,       # 压缩部分（仅 c4/c128）
    extra_indices_in_kvcache=extra_indices,
    extra_topk_length=extra_topk_lengths,
    attn_sink=attn_sink,               # attention sink token
    ...
)
```

### 4.2 SWA Page 索引计算

```python
# deepseek_v4_backend.py:1176-1192
def get_swa_page_indices(self, seq_lens_casual, req_pool_indices_repeated):
    pos_causal = seq_lens_casual - 1
    # 计算窗口内每个位置的 offset
    offsets = pos_causal.unsqueeze(1) - torch.arange(SWA_WINDOW).unsqueeze(0)
    invalid_offset_mask = offsets < 0
    offsets.masked_fill_(invalid_offset_mask, 0)
    # 从 req_to_token 获取 raw indices
    raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
    raw_indices.masked_fill_(invalid_offset_mask, -1)
    # 翻译到 SWA 索引空间
    swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
    return swa_indices
```

### 4.3 SWA topk_length

```python
swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)  # ≤ 128
```

当序列长度 < 128 时，实际有效 SWA token 数 = seq_len。

---

## 5. SWA Radix Cache

`python/sglang/srt/mem_cache/swa_radix_cache.py` 实现了 SWA 的 radix tree cache，支持前缀复用：

- SWA 节点与 full attention 节点共享 radix tree 结构
- SWA 数据超出窗口后可被 "tombstone"（逻辑删除），释放 SWA pool 空间
- full attention 数据不受 SWA tombstone 影响

相关环境变量：
```python
SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = False   # compact 有 bug on retract
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = False
SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = False
SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN = False
```

---

## 6. 关键文件索引

| 文件 | 职责 |
|---|---|
| `python/sglang/srt/configs/deepseek_v4.py` | 模型配置：`window_size=128`, `compress_ratios` |
| `python/sglang/srt/models/deepseek_v4.py` | Attention forward，KV 写入调用 |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | SWA/压缩 KV pool，索引翻译 |
| `python/sglang/srt/mem_cache/base_swa_memory_pool.py` | SWA pool 基类 |
| `python/sglang/srt/mem_cache/swa_radix_cache.py` | SWA radix tree cache |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | Attention backend，`SWA_WINDOW=128`，decode/prefill 元数据 |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | DSV4 默认参数 (`swa_full_tokens_ratio=0.1`) |
| `python/sglang/srt/model_executor/pool_configurator.py` | 内存池大小计算 |
| `python/sglang/srt/environ.py` | `SGLANG_OPT_CACHE_SWA_TRANSLATION` 等 |
| `sglang/jit_kernel/dsv4/` | Fused norm+rope+store CUDA kernels |

---

## 7. 数据流总结

```
Forward Pass (per layer):
┌──────────────────────────────────────────────────────────┐
│ 1. x → wkv projection → kv (bf16)                       │
│ 2. kv → kv_norm → rope → quantize_fp8 → SWA cache 写入  │
│    └─ swa_loc = translate(out_cache_loc)                 │
│    └─ swa_kv_pool.kv_buffer[layer_id][swa_loc] = kv_fp8 │
│ 3. c4/c128 层额外: kv → compress → 压缩 pool 写入        │
│ 4. Decode attention:                                     │
│    ├─ SWA: flash_mla(q, swa_k_cache, swa_page_indices)  │
│    └─ 压缩: flash_mla(..., extra_k_cache, extra_indices) │
└──────────────────────────────────────────────────────────┘
```
