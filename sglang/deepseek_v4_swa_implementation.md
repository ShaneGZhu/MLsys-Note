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

#### 2.3.1 `full_pool_tokens` 的计算链路

```
GPU 总显存
  │
  ▼ 模型加载后剩余显存 × mem_fraction_static
  │
  ▼ _profile_available_bytes()  (model_runner_kv_cache_mixin.py:62-76)
  │
available_bytes (字节)
  │
  ▼ full_token = available_bytes / bytes_per_full_token
  │                                     ↑
  │                      _get_bytes_per_full_token() 计算的每 full token 总字节成本
  │
full_token (page 对齐后) ← 即 full_pool_tokens
```

计算可用字节（`model_runner_kv_cache_mixin.py:62-76`）：

```python
def _profile_available_bytes(self, pre_model_load_memory):
    post_model_load_memory = get_available_gpu_memory(...)  # 模型加载后剩余 (GB)
    # rest = mem_fraction_static 比例的空闲 GPU 显存
    rest_memory = post_model_load_memory - pre_model_load_memory * (1 - self.mem_fraction_static)
    return int(rest_memory * (1 << 30))  # GB → bytes
```

#### 2.3.2 `bytes_per_full_token`：每个 full token 的总成本

一个 full token 的 "成本" 是其在**所有 pool** 中摊销费用之和（`pool_configurator.py:372-410`）：

```python
def _get_bytes_per_full_token(self) -> float:
    kv_bytes = qk_nope_head_dim + qk_rope_head_dim * 2 + 8  # = 448 + 128 + 8 = 584
    # ... indexer_bytes, state_bytes 等

    return (
        swa_ratio × kv_bytes × num_layers_total           # SWA pool (每 full token 摊 0.1 份)
      + 1/4 × kv_bytes × num_layers_ca4                   # c4 KV pool
      + 1/128 × kv_bytes × num_layers_ca128               # c128 KV pool
      + 1/4 × indexer_bytes × num_layers_ca4              # indexer pool
      + swa_ratio × (c4_ring/swa_page) × c4_state_bytes × num_layers_ca4    # c4 state
      + swa_ratio × (c128_ring/swa_page) × c128_state_bytes × num_layers_ca128  # c128 state
      + swa_ratio × (c4_ring/swa_page) × c4_indexer_state_bytes × num_layers_ca4
    )
```

**关键洞察：`bytes_per_full_token` 已经把 SWA、c4、c128、state 全部摊进去了。** 因此 `available_bytes / bytes_per_full_token` 直接得到 `full_token`，其余 pool 大小按固定比例推导。

#### 2.3.3 `_compute_dsv4_sizes`：从 full_token 推导所有 pool

```python
# pool_configurator.py:412-422
def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
    full_token = full_token // page_size * page_size                        # page 对齐
    swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size  # SWA pool
    return _DSV4PoolSizes(
        full_max_total_num_tokens=full_token,                               # radix tree 主索引
        swa_max_total_num_tokens=swa_tokens,                                # 滑动窗口 KV
        c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),  # c4 压缩 KV
        c128_max_total_num_tokens=full_token // 128,                        # c128 压缩 KV
        c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,   # c4 ring buffer
        c128_state_pool_size=swa_tokens // self.swa_page_size * self.c128_ring_size, # c128 ring buffer
    )
```

各 pool 之间的固定比例关系：

```
full_token = N (基准, 由 available_bytes / bytes_per_full_token 得出)
  │
  ├── swa_tokens       = N × 0.1              ← swa_full_tokens_ratio
  ├── c4_tokens        = N / 4                ← compress_ratio=4
  ├── c128_tokens      = N / 128              ← compress_ratio=128
  ├── c4_state_slots   = (N×0.1/128) × 8     ← swa_pages × ring_size
  └── c128_state_slots = (N×0.1/128) × 128   ← swa_pages × ring_size
```

比例来源：
- **`/4`, `/128`**：压缩比，论文定义（每 4/128 个 token 压缩成 1 个）
- **`×0.1`**：`swa_full_tokens_ratio`，CLI 可配（`--swa-full-tokens-ratio`）
- **state**：从 SWA pages 数 × ring_size 得出（见 §8.7）

#### 2.3.4 投机解码的额外膨胀

使用 speculative decoding 时，`bytes_per_full_token` 按 `(T+D)/T` 比例膨胀：

```python
# pool_configurator.py:353-360
if self.is_speculative:
    draft_layers = 1
    target_layers = self.num_layers_total
    self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers
```

效果：同样的显存，full_token 数量略减，为 draft worker 预留 KV cache 空间。

#### 2.3.5 用户约束覆盖

若用户设置 `--max-total-tokens`，跳过 profiling，直接以用户值作为 `full_token`：

```python
# model_runner_kv_cache_mixin.py:887-891
constrained = self._apply_token_constraints(config.max_total_num_tokens)
if constrained != config.max_total_num_tokens:
    config = configurator.calculate_pool_sizes_from_max_tokens(constrained, page_size)
```

#### 2.3.6 数值示例

假设 80GB A100，模型加载后剩余 40GB，`mem_fraction_static=0.88`，61 层 DSV4：

```
available_bytes ≈ 35.2 GB ≈ 37.6 × 10⁹ bytes
bytes_per_full_token ≈ 9,700 bytes (示意值，视层配置而异)

full_token ≈ 37.6G / 9,700 ≈ 3,870,000 tokens (page 对齐后)
  → swa_tokens   ≈ 387,000
  → c4_tokens    ≈ 967,500
  → c128_tokens  ≈ 30,200
  → c4_state     ≈ 387,000/128 × 8 = 24,187 slots/layer
  → c128_state   ≈ 387,000/128 × 128 = 387,000 slots/layer
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

### 3.1 调用入口

`python/sglang/srt/models/deepseek_v4.py:578-605`：

```python
from sglang.srt.layers.fused_qk_norm_rope_store import fused_qk_norm_rope_swa_store

token_to_kv_pool = get_token_to_kv_pool()
swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(forward_batch.out_cache_loc)
swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]
swa_page_size = token_to_kv_pool.swa_kv_pool.page_size

q = fused_qk_norm_rope_swa_store(
    q=q, kv=kv,
    kv_norm_weight=self.kv_norm.weight,
    swa_cache=swa_cache,       # 目标: [num_pages, bytes_per_page] uint8 buffer
    swa_loc=swa_loc,           # [num_tokens] int32, 翻译后的 SWA 物理索引
    swa_page_size=swa_page_size,
    cos_cache=self.cos_cache,
    sin_cache=self.sin_cache,
    positions=positions,
    ...
)
```

### 3.2 Fused Triton Kernel

写入在 `python/sglang/srt/layers/fused_qk_norm_rope_store.py:199-267` 的 Triton kernel 中完成，
与 Q/KV RMSNorm + RoPE **融合在同一个 kernel** 中，避免额外的 global memory round-trip。

融合的步骤：
```
kv (BF16, hidden_size → head_dim=512 经 wkv 投影)
  │
  ▼ RMSNorm (kv_norm_weight)
  │
  ├── nope 部分 (前 448 维): 写回 kv in-place
  └── rope 部分 (后 64 维):  → RoPE 旋转 → 写回 kv in-place
  │
  ▼ Paged SWA Store (在同一 kernel 中)
  │
  ├── nope: per-tile FP8 量化 → 写入 swa_cache
  ├── rope: BF16 直接写入 swa_cache
  └── scales: uint8 写入 swa_cache
```

### 3.3 每 Token 存储布局（584 字节）

SWA cache 以 paged 方式组织：`swa_kv_pool.kv_buffer[layer_id]` 形状为 `[num_pages, bytes_per_page_padded]`。

每个 page 内按 token 顺序存储（`swa_page_size` 个 token），每 token 布局：

```
Page 布局:
┌─────── Values 区域: page_size × 576 bytes ───────┐
│ Token 0: [448B FP8 nope | 128B BF16 rope]        │
│ Token 1: [448B FP8 nope | 128B BF16 rope]        │
│ ...                                               │
│ Token N: [448B FP8 nope | 128B BF16 rope]        │
├─────── Scales 区域: page_size × 8 bytes ─────────┤
│ Token 0: [7B scales | 1B pad]                    │
│ Token 1: [7B scales | 1B pad]                    │
│ ...                                               │
└───────────────────────────────────────────────────┘

每 token 总计: 576 + 8 = 584 字节
```

各分量详解：

| 分量 | 维度 | 存储类型 | 大小 | 说明 |
|------|------|----------|------|------|
| nope | 448 (qk_nope_head_dim) | **float8_e4m3fn** | 448 字节 | KV 经 RMSNorm 后量化 |
| rope | 64 (qk_rope_head_dim) | **bfloat16** | 128 字节 | KV 经 RoPE 后,不量化 |
| scales | 7 tiles (448/64) | **uint8** (指数编码) | 7 字节 | per-tile 量化 scale |
| pad | — | — | 1 字节 | 对齐 |

### 3.4 FP8 量化策略

**Per-tile absmax + power-of-2 scale**（`fused_qk_norm_rope_store.py:224-258`）：

```python
# 每 tile = 64 维, 共 7 tiles (448/64)
for tile_i in range(7):
    tile_data = kv_normed[tile_start : tile_start + 64]  # [BLOCK_M, 64] float32

    # 1. 求 tile 内最大绝对值
    abs_max = max(abs(tile_data))

    # 2. 计算 2 的幂次 scale (保证 dequant 是精确乘法)
    scale_pow2 = exp2(ceil(log2(abs_max / FP8_MAX)))

    # 3. 量化到 FP8
    x_fp8 = clamp(tile_data / scale_pow2, FP8_MIN, FP8_MAX).to(float8_e4m3fn)

    # 4. scale 编码为 uint8 (biased exponent: ceil_log2 + 127)
    scale_uint8 = (ceil_log2 + 127).to(uint8)

    # 写入
    store(swa_cache + value_base + tile_start, x_fp8)       # 64 bytes FP8
    store(swa_cache + scale_base + tile_i, scale_uint8)     # 1 byte scale
```

**为什么 scale 用 2 的幂次**：dequant 时 `x_real = x_fp8 * scale_pow2`，乘 2 的幂次在浮点中是精确的（只改指数位），无舍入误差。

### 3.5 RoPE 部分为什么不量化

- rope 维度只有 64（vs nope 的 448），BF16 只需 128 字节，占比小
- rope 编码位置信息，量化会损失位置区分度（频率分辨率）
- 总开销：128B BF16 vs 64B FP8 + 1B scale，节省有限但精度损失显著

### 3.6 Kernel 内的寻址计算

```python
# Triton kernel 中的 page 内寻址
loc = tl.load(swa_loc_ptr + token_id)         # SWA 物理索引
page_id = loc // SWA_PAGE_SIZE                 # 属于哪个 page
page_off = loc % SWA_PAGE_SIZE                 # page 内第几个 token

VALUE_STRIDE = 448 + 64*2  # = 576 bytes/token (nope + rope)
SCALE_BYTES = 7 + 1        # = 8 bytes/token (scales + pad)

value_base = page_id * page_stride + page_off * VALUE_STRIDE
scale_base = page_id * page_stride + SWA_PAGE_SIZE * VALUE_STRIDE + page_off * SCALE_BYTES
```

### 3.7 store_cache 备用路径（Backend 入口）

非融合路径，用于 Context Parallelism 等场景：

```python
# deepseek_v4_backend.py
if save_kv_cache:
    self.store_cache(layer_id, swa_k, forward_batch)
```

此路径调用 `set_swa_key_buffer_radix_fused()` 或 `fused_k_norm_rope_flashmla` JIT kernel，
功能相同（norm + rope + FP8 quant + paged store）但通过不同代码路径触发。

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

## 5. Prefix Caching（SWA Radix Cache）

DSV4 的 prefix caching 核心挑战：**三个 KV pool（SWA / c4 / c128）生命周期不同**。SWA 数据只需最近 128 tokens，过期即可回收；而 c4/c128 压缩数据需要长期保留。

### 5.1 核心设计：单索引 Radix Tree

Radix tree 只存一套索引（full-pool indices），SWA 和压缩池的索引通过映射/算术推导：

```
Radix Tree Node
    └─ value: [full-pool indices]   ← 树中唯一存储的索引
                  │
                  ├── full_to_swa_index_mapping[indices]  → SWA pool 位置
                  │
                  ├── indices // 4  → c4 pool 位置
                  │
                  └── indices // 128 → c128 pool 位置
```

### 5.2 双分配器（SWATokenToKVPoolAllocator）

`python/sglang/srt/mem_cache/swa_memory_pool.py`

```python
class SWATokenToKVPoolAllocator:
    full_attn_allocator   # 管理 full-pool 索引（c4/c128 用）
    swa_attn_allocator    # 管理 SWA pool 索引
    full_to_swa_index_mapping  # full→SWA 映射表 [full_size + page_size + 1]
```

**分配流程**（`alloc_extend`）：
1. 从 `full_attn_allocator` 分配 N 个 full 索引
2. 从 `swa_attn_allocator` 分配 N 个 SWA 索引
3. 写入映射：`mapping[full_indices] = swa_indices`
4. 返回 full 索引（写入 radix tree）

**释放流程**（`free`）：
1. `full_attn_allocator.free(full_indices)` — 回收 full 槽位
2. 查映射表得到对应 SWA 索引，`swa_attn_allocator.free(swa_indices)`
3. 清零映射表

**SWA 单独释放**（`free_swa`）：
- 只回收 SWA 槽位，full 索引保留
- 用于 decode 时释放窗口外的 SWA 数据

### 5.3 树节点结构

`python/sglang/srt/mem_cache/swa_radix_cache.py`

```python
class TreeNode:
    value: torch.Tensor        # full-pool indices
    swa_tombstone: bool        # SWA 数据是否已被回收

    # 双锁
    full_lock_ref: int         # full pool 保护引用计数
    swa_lock_ref: int          # SWA pool 保护引用计数

    # 双 LRU 链表指针
    prev / next                # full LRU
    swa_prev / swa_next        # SWA LRU

    swa_uuid: int              # SWA 锁边界标记
```

**Tombstone 机制**：节点的 SWA 数据被回收但 full 数据（c4/c128）仍有效时，标记 `swa_tombstone = True`。节点保留在 tree 中，prefix 共享不受影响。

### 5.4 双 LRU 驱逐策略

```
evict(full_needed, swa_needed):

Phase 1: Full 驱逐（释放 full + SWA）
    遍历 full_lru_list → 找未锁定的叶子节点
    → free(node.value)  // 同时释放 full 和 SWA
    → 从 tree 中删除节点
    → 递归删除变成 childless 的 tombstone 父节点

Phase 2: SWA-only 驱逐（只释放 SWA）
    遍历 swa_lru_list → 找未 SWA-锁定的节点（可以是内部节点!）
    → free_swa(node.value)  // 只释放 SWA 槽位
    → node.swa_tombstone = True
    → 节点保留在 tree 中，c4/c128 数据不受影响
```

**SWA 驱逐可以 tombstone 内部节点**（不止叶子），c4/c128 的 prefix 共享不会因 SWA pool 压力而被破坏。

### 5.5 Tombstone 感知的 Prefix 匹配

```python
# swa_radix_cache.py: _match_prefix_helper
def _match_prefix_helper(self, key, ...):
    match_len_since_tombstone = 0

    for each matched node:
        if node.swa_tombstone:
            match_len_since_tombstone = 0  # 重置
        else:
            match_len_since_tombstone += len(node.value)

        # 有效匹配条件: tombstone 之后累积 ≥ sliding_window_size 的非 tombstone tokens
        if match_len_since_tombstone >= sliding_window_size:
            best_match = current_position  # 窗口内 SWA 数据完整
```

保证命中的 prefix 在滑动窗口内有完整的 SWA 数据，否则需要 recompute。

### 5.6 双锁机制

**`inc_lock_ref(node)`**：
- 从 node 到 root，递增 `full_lock_ref`
- 仅对底部 `sliding_window_size` tokens 递增 `swa_lock_ref`
- 记录 `swa_uuid_for_lock` 标记 SWA 锁边界

**`dec_lock_ref(node, swa_uuid)`**：
- 从 node 到 root，递减 `full_lock_ref`
- 从 node 到 swa_uuid 边界，递减 `swa_lock_ref`

**`dec_swa_lock_only(node, swa_uuid)`**：
- Decode 推进时提前释放 SWA 锁
- 内部节点：正常解锁进入 evictable
- 叶子节点：立即 `free_swa` + tombstone（因为 full_lock_ref > 0 不能完全删除）

### 5.7 Decode 阶段 SWA 提前释放

```python
# schedule_batch.py: maybe_evict_swa()
# 当 decode position > swa_evicted_seqlen + sliding_window + page_size:
#   1. 释放 req_to_token 中窗口外的 SWA 索引
#   2. 调用 dec_swa_lock_only(node) 释放 tree 中的 SWA 锁
#   3. 更新 req.swa_evicted_seqlen
```

长序列 decode 时 SWA pool 不会被无限占用，只保留最近 window 的 SWA 数据。

### 5.8 请求完整生命周期

```
1. match_prefix(tokens)
   └─ 遍历 SWARadixCache，tombstone-aware 匹配
   └─ 返回 (prefix_indices, last_node)
   └─ inc_lock_ref(last_node) → 锁住 full + SWA(仅窗口内)

2. alloc_extend(new_tokens)
   └─ 双分配器: full + SWA 同时分配
   └─ 写入 full_to_swa_index_mapping

3. forward pass
   └─ SWA 层: translate(full_indices) → SWA pool 读写
   └─ c4/c128 层: full_indices // ratio → compressed pool 读写

4. decode 推进
   └─ maybe_evict_swa(): 释放窗口外 SWA 数据
   └─ dec_swa_lock_only(): 提前释放 tree SWA 锁

5. 请求完成 → cache_finished_req()
   └─ insert(full_indices) 到 tree
   └─ 前 swa_evicted_seqlen 的节点标记为 tombstone
   └─ dec_lock_ref(node): 释放 full + SWA 锁
   └─ 节点进入 LRU，供后续请求 prefix 复用
```

### 5.9 Prefix 共享原理

当多个请求共享同一前缀时：
1. 第一个请求将 KV indices 插入 tree
2. 后续请求 `match_prefix` 匹配到已有节点，直接复用 full-pool indices
3. `full_to_swa_index_mapping` 保证这些共享索引的 SWA 位置正确
4. c4/c128 的压缩数据通过算术推导也自动共享
5. **一套 full 索引，三个 pool 同时共享**

### 5.10 相关环境变量

```python
SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = False   # compact 有 bug on retract
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = False
SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = False
SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN = False
```

### 5.11 SWA Pool 容量限制与流控

#### 5.11.1 总容量上限

```python
# pool_configurator.py:414
swa_tokens = int(full_token * self.swa_ratio)
# swa_ratio = 0.1 (DSV4 hook 设置)
```

举例：如果 full pool 有 100,000 tokens，则 SWA pool 最多存 **10,000 tokens**，所有请求共享。

#### 5.11.2 SWA Pool 内部状态划分

```
SWA pool 总容量 = swa_size (如 10,000 tokens)
    │
    ├── protected（被锁定）：正在活跃使用的请求
    │     每个 decode 请求锁住 ≤ sliding_window_size (128) tokens
    │     每个 prefill 请求锁住其 extend 长度
    │
    ├── evictable（可驱逐）：tree 中未锁定节点的 SWA 数据
    │     即 swa_evictable_size_
    │
    └── available（空闲）：尚未分配的空槽位
          即 swa_attn_allocator.available_size()
```

#### 5.11.3 四级流控机制

**卡口 1：调度准入**（`schedule_policy.py`）

```python
@property
def rem_swa_tokens(self):
    return (
        token_to_kv_pool_allocator.swa_available_size()  # SWA 空闲
        + tree_cache.swa_evictable_size()                 # SWA 可驱逐
        - rem_swa_token_offset                            # 已预留给本轮其他请求
    )

def budget_state(self):
    if self.rem_swa_tokens <= 0:
        return AddReqResult.NO_TOKEN  # 拒绝新请求入队
```

每个请求的 SWA 预算计算：
```python
def _swa_budget_for_req(self, extend_input_len):
    alloc = min(extend_input_len, self.rem_chunk_tokens)
    return max(alloc, sliding_window_size) + page_size
    # 至少预留 128 + 256 = 384 tokens
```

**卡口 2：分配时双检查**（`swa_memory_pool.py`）

```python
# SWATokenToKVPoolAllocator.alloc_extend()
if num_new_pages > self.full_attn_allocator.available_size() // page_size:
    return None  # full pool 不够
if num_new_pages > self.swa_attn_allocator.available_size() // page_size:
    return None  # SWA pool 不够 → 触发驱逐
```

`available_size()` 取两个 pool 的最小值：
```python
def available_size(self):
    return min(
        self.full_attn_allocator.available_size(),
        self.swa_attn_allocator.available_size(),
    )
```

**卡口 3：LRU 驱逐回收**（`swa_radix_cache.py`）

```python
# evict Phase 2: SWA-only 驱逐
while swa_num_evicted < swa_num_tokens:
    x = swa_lru_list.get_lru()  # 最久未使用

    if x is internal or (leaf with full_lock > 0):
        free_swa(x.value)       # 只释放 SWA 槽位
        x.swa_tombstone = True  # 标记 tombstone
        swa_evictable_size_ -= len(x.value)

    elif x is unlocked leaf:
        free(x.value)           # 释放 full + SWA
        delete_leaf(x)          # 从 tree 删除
```

**卡口 4：Decode 自动释放**（`schedule_batch.py`）

```python
# maybe_evict_swa()
# 当 decode position > swa_evicted_seqlen + sliding_window + page_size:
#   1. free_swa(窗口外的旧 SWA tokens)
#   2. dec_swa_lock_only(tree_node) → 释放 tree SWA 锁
#   3. req.swa_evicted_seqlen += freed_len
```

#### 5.11.4 并发能力估算

```
假设: swa_size = 10,000, sliding_window = 128, page_size = 256

每个 decode 请求实际占用 = 1 page = 256 tokens
  (因为 maybe_evict_swa 会释放超出窗口的部分)

最大并发 decode ≈ 10,000 / 256 ≈ 39 个请求

如果同时有 prefill:
  1 个 prefill 请求 extend 2048 tokens → 占用 2048 SWA tokens
  剩余: (10,000 - 2048) / 256 ≈ 31 个 decode 请求
```

#### 5.11.5 限制总结

| 限制层 | 位置 | 机制 |
|---|---|---|
| 硬上限 | `pool_configurator.py` | `swa_size = full_tokens × 0.1` |
| 调度准入 | `schedule_policy.py` | `rem_swa_tokens ≤ 0` 拒绝新请求 |
| 分配检查 | `swa_memory_pool.py` | `swa_available < need` → 返回 None |
| 驱逐回收 | `swa_radix_cache.py` | SWA LRU → tombstone/delete 回收空间 |
| 提前释放 | `schedule_batch.py` | `maybe_evict_swa()` decode 超窗口自动释放 |

### 5.12 Tombstone 感知匹配的具体场景

#### 场景示例

```
Radix Tree 路径:
  Node A [tokens 0-127] tombstoned → Node B [128-255] alive → Node C [256-383] alive
```

新请求 tokens 在 position 200 处分歧，tree 匹配只能到 200：

```
匹配过程:
  Node A (tombstoned): match_len_since_tombstone = 0      (重置)
  Node B (走到 200):   match_len_since_tombstone = 72     (200-128=72 < 128)
                       → 无效匹配点!

回退: best_match = 0 (无有效点，不复用 prefix)
```

#### 不同匹配点的安全性分析

| 匹配到 | 下次 decode position | SWA 窗口需要 | 安全? |
|---|---|---|---|
| token 127 (Node A 末尾) | 128 | tokens 0-127 (Node A tombstoned!) | 不安全 |
| token 200 (Node B 中间) | 201 | tokens 73-200 (含 Node A 部分!) | 不安全 |
| token 255 (Node B 末尾) | 256 | tokens 128-255 (全在 alive Node B) | **安全** |
| token 383 (Node C 末尾) | 384 | tokens 256-383 (全在 alive Node C) | **安全** |

#### 匹配失败后的处理

当 best_match = 0 时，请求按"无 prefix 命中"处理：
1. `alloc_extend` 为全部 tokens 分配新的 full + SWA 槽位
2. Forward pass 重新计算所有 KV 并写入新分配的 pool
3. 请求完成后 `insert` 到 tree，覆盖 Node A 的 tombstone（de-tombstone）
4. 后续相同 prefix 的请求可以正常复用

**本质 tradeoff**：积极回收 SWA pool（提高并发/吞吐）→ 后续相同 prefix 可能需要 recompute。

### 5.13 设计决策总结

| 设计决策 | 原因 |
|---|---|
| Tree 只存 full-pool indices | 避免多套索引同步，简化 tree 逻辑 |
| `full_to_swa_index_mapping` 查表 | O(1) 推导 SWA 位置，无需 tree 存储 |
| Tombstone 机制 | SWA 生命周期短，允许独立回收不破坏 prefix 共享 |
| 双 LRU | Full 和 SWA 独立驱逐，SWA 可 tombstone 内部节点 |
| SWA-aware prefix matching | 保证命中的 prefix 在窗口内有完整 SWA 数据 |
| Decode 提前释放 SWA 锁 | 长序列不占 SWA pool，提高并发 |
| 四级流控 | 调度→分配→驱逐→自动释放，保证不 OOM |

---

## 6. HiCache：多级 KV Cache（GPU / CPU / Disk）

SGLang 通过 **HiCache（Hierarchical Cache）** 实现了 DSV4 论文中描述的 on-disk cache 优化，支持三级存储层次。

### 6.1 三级存储架构

```
┌─────────────────────────────────────────────────────────────────┐
│ L1: GPU HBM (device)                                            │
│   - 热数据，直接参与 attention 计算                                │
│   - 容量: 由 GPU 显存决定（如 80GB H100 → ~100K tokens）          │
│   - 延迟: ~ns                                                    │
├─────────────────────────────────────────────────────────────────┤
│ L2: CPU DRAM (host)                                             │
│   - 温数据，被 GPU 驱逐但仍在内存中                                │
│   - 容量: GPU × hicache_ratio（默认 2x → ~200K tokens）           │
│   - 延迟: ~μs (PCIe/NVLink DMA)                                  │
├─────────────────────────────────────────────────────────────────┤
│ L3: Storage (disk/网络/分布式)                                    │
│   - 冷数据，持久化存储                                            │
│   - 容量: 近乎无限（取决于后端）                                    │
│   - 延迟: ~ms (SSD) 到 ~10ms (网络)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 启用方式

```bash
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V4 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \                          # L2 = L1 × 2
    --hicache-write-policy write_through_selective \ # GPU→CPU 写策略
    --hicache-io-backend kernel \                    # GPU↔CPU IO 方式
    --hicache-storage-backend hf3fs \               # L3 后端
    --hicache-storage-backend-extra-config '{"path": "/mnt/3fs/kvcache"}'
```

### 6.3 DSV4 多 Pool HiCache 栈

DSV4 的 HiCache 不只是简单的 KV offload，而是**对所有 7 个子 pool 都建立了 GPU↔CPU 的 host 副本**：

```python
# hybrid_pool_assembler.py: build_deepseek_v4_hicache_stack()

Pool Name                    │ Device Pool              │ Host Pool 类型
─────────────────────────────┼──────────────────────────┼──────────────────────
PoolName.KV                  │ kvcache (full indices)   │ LogicalHostPool
PoolName.SWA                 │ swa_kv_pool              │ DeepSeekV4PagedHostPool
PoolName.DEEPSEEK_V4_C4      │ c4_kv_pool               │ DeepSeekV4PagedHostPool
PoolName.DEEPSEEK_V4_C4_INDEXER │ c4_indexer_kv_pool    │ DeepSeekV4PagedHostPool
PoolName.DEEPSEEK_V4_C128    │ c128_kv_pool             │ DeepSeekV4PagedHostPool
PoolName.DEEPSEEK_V4_C4_STATE│ compress_state_pools[c4] │ DeepSeekV4StateHostPool
PoolName.DEEPSEEK_V4_C4_INDEXER_STATE │ indexer_compress_state_pools[c4] │ DeepSeekV4StateHostPool
PoolName.DEEPSEEK_V4_C128_STATE │ compress_state_pools[c128] │ DeepSeekV4StateHostPool
```

**layer_mapping 构造**：
```python
# 每种 pool 有独立的 layer mapping
full_layer_mapping = {0: 0, 1: 1, ..., N: N}  # 所有层
swa_layer_mapping  = {0: 0, 1: 1, ..., N: N}  # 所有层
c4_layer_mapping   = {layer_id: compress_layer_id for c4 layers}
c128_layer_mapping = {layer_id: compress_layer_id for c128 layers}
c4_state_mapping   = {layer_id: local_id for c4 layers}
```

### 6.4 Host Pool 大小计算

```python
# _deepseek_v4_num_host_pages()
device_full_pages = ceil(full_pool_size / page_size)
device_swa_pages = ceil(swa_size / swa_page_size)

ratio = server_args.hicache_ratio  # 默认 2.0

full_host_pages = max(int(device_full_pages * ratio), device_full_pages + 1)
swa_host_pages  = max(int(device_swa_pages * ratio), device_swa_pages + 1)
```

c4/c128/state 的 host pool 复用 `full_host_pages` 或 `swa_host_pages`。

### 6.5 数据流动路径

#### Write-through（GPU → CPU）

```python
# HiRadixCache.write_backup() / HybridCacheController.write()
# 触发条件: 节点的 lock_ref 下降到 write_through_threshold 以下

# 对 DSV4 的每个 pool 都会同步写:
for pool_entry in host_pool_group.entries:
    # GPU buffer[layer][device_indices] → CPU buffer[layer][host_indices]
    transfer_kv_all_layer_mla(
        src_layers=device_pool.data_ptrs,
        dst_layers=host_pool.buffers,
        src_indices=device_indices,
        dst_indices=host_indices,
        ...
    )
node.host_value = host_indices
node.backuped = True
```

#### Evict（GPU 释放，CPU 保留）

```python
# HiRadixCache._evict_backuped()
# 触发: GPU pool 容量不足时，优先驱逐已 backup 到 CPU 的节点

cache_controller.evict_device(node.value)  # 释放 GPU 所有 pool 的对应槽位
node.value = None  # GPU indices 清空
# node.host_value 仍然有效 → 后续可 load_back
# 树中保留该节点（evicted=True），prefix 匹配时会遇到
```

#### Load-back（CPU → GPU）

```python
# HiRadixCache.load_back()
# 触发: match_prefix 命中了 evicted 节点（数据在 CPU 中）

# 1. 在 GPU 分配新槽位
device_indices = cache_controller.load(host_indices=host_indices, ...)
# 2. 对每个 pool 做 CPU → GPU DMA
for pool_entry in host_pool_group.entries:
    transfer_kv_all_layer_mla(
        src=host_pool, dst=device_pool,
        src_indices=host_indices, dst_indices=device_indices,
    )
# 3. 恢复节点
node.value = device_indices
node.evicted = False
```

#### Prefetch（Storage → CPU）

```python
# HiRadixCache.prefetch_from_storage()
# 触发: 新请求到来时，预测性地从 L3 加载可能命中的 prefix

# 1. 在 host pool 分配空间
host_indices = mem_pool_host.alloc(prefetch_length)
# 2. 异步从 storage backend 读取
operation = cache_controller.prefetch(req_id, host_indices, key, ...)
# 3. 完成后插入 tree (check_prefetch_progress)
```

#### Backup（CPU → Storage）

```python
# 当 CPU pool 也满时，将最久未用的 host 数据写入 L3
# HiRadixCache.evict_host() 会驱逐 CPU 节点
# 之前 write_backup 时如果 enable_storage=True，数据已持久化到 L3
```

### 6.6 Radix Tree 节点状态机

```
                    ┌────────────────┐
                    │   Active (L1)  │
                    │ value = GPU idx│
                    │ backuped=False │
                    └───────┬────────┘
                            │ write_through (lock_ref 降低)
                            ▼
                    ┌────────────────┐
                    │  Backed-up(L1) │
                    │ value = GPU idx│
                    │ host_value=CPU │
                    │ backuped=True  │
                    └───────┬────────┘
                            │ evict (GPU 压力)
                            ▼
                    ┌────────────────┐
                    │  Evicted (L2)  │
                    │ value = None   │
                    │ host_value=CPU │
                    │ evicted=True   │
                    └───────┬────────┘
         load_back ↑        │ evict_host (CPU 压力)
         (CPU→GPU) │        ▼
                    │ ┌────────────────┐
                    │ │  Cold (L3)     │
                    │ │ 树节点被删除    │
                    │ │ 仅 Storage 有   │
                    │ └────────────────┘
                    │        │ prefetch (新请求预测)
                    │        ▼
                    │ ┌────────────────┐
                    └─│ Prefetched(L2) │
                      │ host_value=CPU │
                      │ → load_back    │
                      └────────────────┘
```

### 6.7 Storage Backends（L3 实现）

| 后端 | 路径 | 适用场景 |
|---|---|---|
| **hf3fs** | `storage/hf3fs/` | 火山 3FS 高性能文件系统 |
| **mooncake** | `storage/mooncake_store/` | Mooncake 分布式 KV 缓存 |
| **nixl** | `storage/nixl/` | NVIDIA NIXL（GPU Direct Storage） |
| **simm** | `storage/simm/` | Shared Inference Memory Manager |
| **aibrix** | `storage/aibrix_kvcache/` | AIBrix KV Cache 服务 |
| **eic** | `storage/eic/` | EIC 存储引擎 |
| **lmcache** | `storage/lmcache/` | LMCache 集成 |
| **dynamic** | `storage/backend_factory.py` | 动态加载自定义后端 |

所有后端通过 `HiCacheStorageBackend` 抽象接口注册，支持：
- `batch_exists_v2()`: 检查 prefix 是否在 storage 中
- `batch_get()`: 从 storage 读取 KV 数据
- `batch_set()`: 写入 KV 数据到 storage
- `batch_delete()`: 删除 storage 中的数据

### 6.8 HiSparse（DSV4 c4 层专属优化）

HiSparse 是针对 c4 压缩层的 **sparse attention + host offload** 优化：

```
原理: c4 层做全局注意力时只需 top-k 个 token 的 KV，不需要全量

优化: GPU 上只存 1/N 的 c4 KV (device_buffer_size = full / host_to_device_ratio)
     CPU 上存完整 c4 KV
     每次 decode 前由 indexer 选出 top-k pages 从 CPU 搬到 GPU
```

启用方式：
```bash
--enable-hisparse \
--hisparse-config '{"top_k": 2048, "host_to_device_ratio": 2}'
```

关键类：
```python
# hisparse_memory_pool.py
class HiSparseDSV4TokenToKVPool(DSV4TokenToKVPool):
    # c4_kv_pool 在 GPU 上只分配 c4_size / host_to_device_ratio
    # HiSparseHostPoolMixin 管理 CPU 端完整副本
    # 运行时: indexer score → top-k selection → CPU→GPU transfer → attention
```

```python
# sparsity/core/sparse_coordinator.py
class SparseCoordinator:
    # 协调 indexer 计算 → top-k 选择 → page 搬运 → attention 计算
    # 实现了 "compute indexer → select pages → transfer → attend" pipeline
```

### 6.9 HiCache + SWA Radix Cache 的交互

对于 DSV4，HiCache 和 SWARadixCache 是**两个独立的缓存层次**，分别处理不同问题：

| | SWARadixCache | HiCache |
|---|---|---|
| 解决的问题 | SWA pool 有限，需要 tombstone 管理 | GPU 显存有限，需要分级缓存 |
| 管理的维度 | SWA vs Full (水平：同 token 不同 pool) | GPU vs CPU vs Disk (垂直：同数据不同层级) |
| 驱逐粒度 | 可以只驱逐 SWA 数据 | 整个节点的所有 pool 一起 offload |
| 适用场景 | 非 HiCache 模式下的标准部署 | 启用 `--enable-hierarchical-cache` |

当两者同时启用时（通过 UnifiedRadixCache）：
- SWA tombstone 处理 SWA pool 的生命周期管理
- HiCache 处理整体 GPU↔CPU↔Storage 的数据搬运
- `UnifiedRadixCache` 统一协调两者

### 6.10 性能关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hicache-ratio` | 2.0 | CPU pool = GPU pool × ratio |
| `--hicache-write-policy` | `write_through_selective` | GPU→CPU 写策略 |
| `--hicache-io-backend` | `kernel` | GPU↔CPU 传输方式 |
| `--hicache-storage-backend` | None | L3 后端（不设则只有 L1+L2） |
| `--hicache-storage-prefetch-policy` | `timeout` | L3 prefetch 策略 |
| `write_through_threshold` | 1-2 | lock_ref 降到多少时触发 write-through |
| `load_back_threshold` | 10 | 至少多少 tokens 才值得 load_back |
| `prefetch_threshold` | 256 | 至少多少 tokens 才触发 L3 prefetch |

### 6.11 关键文件索引

| 文件 | 职责 |
|---|---|
| `mem_cache/hiradix_cache.py` | 三级感知的 radix tree（HiRadixCache） |
| `mem_cache/unified_radix_cache.py` | 统一 radix cache（支持 SWA + HiCache 同时启用） |
| `managers/cache_controller.py` | `HiCacheController`：GPU↔CPU 传输调度 |
| `mem_cache/hybrid_cache/hybrid_cache_controller.py` | `HybridCacheController`：多 pool 传输协调 |
| `mem_cache/hybrid_cache/hybrid_pool_assembler.py` | DSV4 HiCache 栈构建（7 个 pool 的 host 副本） |
| `mem_cache/memory_pool_host.py` | CPU 端 pool 实现（`HostKVCache`, `DeepSeekV4PagedHostPool`） |
| `mem_cache/hicache_storage.py` | Storage 抽象接口、PoolName 枚举 |
| `mem_cache/storage/backend_factory.py` | Storage backend 动态注册/加载 |
| `mem_cache/storage/hf3fs/` | 3FS 后端实现 |
| `mem_cache/storage/nixl/` | NVIDIA NIXL 后端 |
| `mem_cache/hisparse_memory_pool.py` | HiSparse：c4 层 GPU/CPU 分级 |
| `mem_cache/sparsity/` | Sparse attention 协调器 |

---

## 7. 关键文件索引

| 文件 | 职责 |
|---|---|
| `python/sglang/srt/configs/deepseek_v4.py` | 模型配置：`window_size=128`, `compress_ratios` |
| `python/sglang/srt/models/deepseek_v4.py` | Attention forward，KV 写入调用 |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | SWA/压缩 KV pool，索引翻译 |
| `python/sglang/srt/mem_cache/base_swa_memory_pool.py` | SWA pool 基类 |
| `python/sglang/srt/mem_cache/swa_radix_cache.py` | SWA radix tree cache（tombstone、双 LRU、双锁） |
| `python/sglang/srt/mem_cache/swa_memory_pool.py` | `SWATokenToKVPoolAllocator`（双分配器 + mapping） |
| `python/sglang/srt/mem_cache/common.py` | `evict_from_tree_cache`（双 pool 驱逐调度） |
| `python/sglang/srt/mem_cache/registry.py` | `is_hybrid_swa=True` 时选择 `SWARadixCache` |
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | Attention backend，`SWA_WINDOW=128`，decode/prefill 元数据 |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | DSV4 默认参数 (`swa_full_tokens_ratio=0.1`) |
| `python/sglang/srt/model_executor/pool_configurator.py` | 内存池大小计算 |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | Pool 初始化，创建 `DeepSeekV4TokenToKVPool` + `SWATokenToKVPoolAllocator` |
| `python/sglang/srt/managers/schedule_policy.py` | `PrefillAdder` 检查 `rem_total_tokens` + `rem_swa_tokens` |
| `python/sglang/srt/managers/schedule_batch.py` | `maybe_evict_swa`（decode 窗口外 SWA 释放） |
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

---

## 8. Compress State Ring Buffer 机制

### 8.1 State Pool 的作用

c4/c128 层在压缩时需要凑够 `compress_ratio` 个 token 才能输出一个 compressed KV。在此之前，已到达但尚未凑齐的 token 的中间状态需要暂存——这就是 **State Ring Buffer** 的职责。

关键文件：
- `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` — `CompressStatePool` 定义
- `python/sglang/srt/layers/attention/dsv4/compressor.py` — `Compressor` 类（wkv_gate 投影 + 压缩调度）
- `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` — V2 压缩路径（JIT C++ kernel）
- `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh` — c4 CUDA 压缩 kernel
- `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh` — c128 CUDA 压缩 kernel
- `python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh` — c128 在线压缩 kernel

### 8.2 kv 和 score 的计算来源

每个 token 的 state 由 **`wkv_gate` 线性投影** 产生：

```python
# compressor.py:317-320
self.wkv_gate = ReplicatedLinear(
    self.dim,                    # 输入: hidden_size (7168)
    2 * coff * self.head_dim,    # 输出: 2 × coff × head_dim
    bias=False,
)

# compressor.py:353-354
def compute_kv_score(self, x, forward_batch):
    kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)  # [num_tokens, 2*coff*head_dim]
    return kv_score
```

其中 `coff = 1 + overlap`：c4 层 overlap=True → coff=2；c128 层 overlap=False → coff=1。

**输出维度拆解**（以 c4 为例, head_dim=512）：

```
wkv_gate 输出: [num_tokens, 2 × 2 × 512] = [num_tokens, 2048]
                           ├── kv 部分 (前 1024): coff × head_dim — token 的「内容表示」
                           └── score 部分 (后 1024): coff × head_dim — token 的「重要性权重」
```

语义对比：

| 分量 | 维度 | 含义 |
|------|------|------|
| **kv** | coff × head_dim | 该 token 对压缩 KV 的**内容贡献**（"我携带什么信息"） |
| **score** | coff × head_dim | 该 token 对压缩 KV 的**权重贡献**（"我有多重要"） |

### 8.3 压缩公式：加权 softmax 求和

凑够 `ratio` 个 token 后执行压缩：

```python
# 核心公式 (compressor 内部)
# kv_and_score_to_compress: [num_windows, ratio, head_dim]

# 1. 加上可学习的位置偏置 (APE)
kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

# 2. softmax 加权求和
kv_compressed = (
    kv_and_score_to_compress.kv                     # [windows, ratio, head_dim]
    * kv_and_score_to_compress.score.softmax(dim=1)  # softmax over ratio 维度
).sum(dim=1)                                         # → [windows, head_dim]
```

数学表达：

$$\text{compressed\_kv} = \sum_{i=0}^{\text{ratio}-1} \text{softmax}(\text{score} + \text{APE})_i \cdot \text{kv}_i$$

- `APE`（Absolute Position Encoding）：可学习参数 `[ratio, coff × head_dim]`，为窗口内不同位置提供位置偏置
- softmax 使得压缩是**内容自适应**的——重要 token 贡献更大权重

压缩完成后：`compressed_kv` 经过 `RMSNorm → RoPE → 存入 c4/c128 KV pool`。

### 8.4 完整数据流

```
hidden_state x  [num_tokens, 7168]
     │
     ▼ wkv_gate (Linear: 7168 → 2048)
     │
     ├─────────────────────────────────┐
     ▼                                 ▼
   kv (1024 dim)                   score (1024 dim)
  "内容表示"                        "重要性权重"
     │                                 │
     ▼───── 写入 State Ring Buffer ────▼
     │       (等待凑够 ratio=4 个)       │
     │                                 │
     │  ┌──── 凑够 4 个 token ─────┐   │
     │  ▼                           ▼   │
     │  kv[0..3]               score[0..3]
     │     │                        │
     │     │                        ▼ + APE (位置偏置)
     │     │                        │
     │     │                        ▼ softmax(dim=ratio)
     │     │                     weights[0..3]
     │     │                        │
     │     ▼──── 加权求和 ──────────┘
     │
     ▼
compressed_kv [1, 512]
     │
     ▼ RMSNorm + RoPE
     │
     ▼ 存入 c4_kv_pool (持久化)
```

### 8.5 Ring Buffer 物理布局

**索引公式**（`deepseek_v4_compress_state.py:135`）：

```python
def translate_from_swa_loc_to_state_loc(self, swa_loc: torch.Tensor) -> torch.Tensor:
    swa_pages = swa_loc // self.swa_page_size      # 属于哪个 SWA page
    state_loc = swa_pages * self.ring_size + (swa_loc % self.ring_size)
    state_loc = torch.where(swa_loc < 0, -1, state_loc)
    return state_loc
```

**Ring size 选择**（`deepseek_v4_memory_pool.py:30-43`）：

```python
def get_compress_state_ring_size(compress_ratio: int, is_speculative: bool = False) -> int:
    if compress_ratio == 128 and ONLINE_C128:
        return 1   # 在线模式只需 1 个 slot
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128
```

| | c4 | c128 | c128 (在线) |
|---|---|---|---|
| compress_ratio | 4 | 128 | 128 |
| ring_size (正常) | 8 | 128 | 1 |
| ring_size (投机) | 16 | 256 | — |
| ring / compress_ratio | 2 (有 overlap) | 1 | 1/128 |

**每层的物理 tensor**：

```
c4_state_pool per layer:  [num_swa_pages × 8, 2048]   (FP32)
c128_state_pool per layer: [num_swa_pages × 128, 1024] (FP32)
c128_online per layer:     [num_swa_pages × 1, 1536]   (FP32)
```

每个 state slot 的 last_dim 拆解：

| 模式 | last_dim | 公式 | 布局 |
|------|----------|------|------|
| c4 (overlap) | 2048 | `2×(1+1)×512` | `[kv_overlap \| kv_current \| score_overlap \| score_current]` |
| c128 (非在线) | 1024 | `2×(1+0)×512` | `[kv \| score]` |
| c128 (在线) | 1536 | `3×512` | `[max \| sum \| kv]` (online softmax 三元组) |

### 8.6 Ring 在 SWA Page 内的循环覆写

以 c4 `ring_size=8`, `swa_page_size=128` 为例：

```
SWA page (128 tokens) → 只有 8 个 state slot
→ ring 在一个 page 内循环 128/8 = 16 圈

swa_loc (页内偏移)  →  swa_loc % 8  →  state slot
────────────────────────────────────────────────
0                       0               slot 0
1                       1               slot 1
...
7                       7               slot 7
8                       0               slot 0  ← 覆写
...
127                     7               slot 7
```

**为什么覆写是安全的**：

- compress_ratio=4：每 4 个 token 完成一次压缩，compressed KV 已存入 c4_kv_pool
- ring_size=8 = 2 × compress_ratio：同时保留**当前窗口 + 前一窗口** 的数据
- c4 有 **overlap 机制**——压缩 token[4..7] 时需要读取 token[0..3] 的 kv/score
- 当 token[8..11] 开始覆写 slot 0-3 时，token[4..7] 的 slot 4-7 仍完好

```
┌─── 窗口 0 (token 0-3) ───┐┌─── 窗口 1 (token 4-7) ───┐
│ slot 0  slot 1  slot 2  slot 3 ││ slot 4  slot 5  slot 6  slot 7 │
└── 压缩完成, 输出 c4[0] ─────┘└── 压缩时读取窗口 0 overlap ──┘
                                              ↕
                                  ┌─── 窗口 2 (token 8-11) ──┐
                                  │ slot 0-3 被覆写 (安全)     │
                                  │ 窗口 1 的 slot 4-7 仍有效  │
                                  └───────────────────────────┘
```

### 8.7 每层的 Ring 数量

每个 SWA page 绑定一个 ring，因此：

```
num_rings_per_layer = swa_tokens / swa_page_size
```

Pool sizing 来自 `pool_configurator.py:420-421`：

```python
c4_state_pool_size = swa_tokens // self.swa_page_size * self.c4_ring_size
c128_state_pool_size = swa_tokens // self.swa_page_size * self.c128_ring_size
```

典型数值（full_tokens=100,000, swa_ratio=0.1）：

```
swa_tokens = 10,000
num_swa_pages = 10,000 / 128 = 78

每个 c4 层:  78 rings × 8 slots/ring  = 624 state slots
每个 c128 层: 78 rings × 128 slots/ring = 9,984 state slots
```

### 8.8 State Ring Buffer 与 Prefix Caching 的适配

#### 核心设计：State 不独立管理，寄生于 SWA 生命周期

```
┌─────────────────────────────────────────┐
│  Radix Tree (full-pool indices)         │  ← 持久化层
│  - 负责 cache 管理、eviction、match      │
├─────────────────────────────────────────┤
│  SWA Pool (full_to_swa_mapping)         │  ← 半持久层
│  - 跟随 tree node 生命周期              │
│  - tombstone 可独立释放                 │
├─────────────────────────────────────────┤
│  State Pool (swa_loc → state_loc)       │  ← 瞬态层
│  - 完全依附 SWA slot 有效性             │
│  - 无独立分配/释放 API                  │
│  - ring 覆写 = 隐式回收                 │
└─────────────────────────────────────────┘
```

#### 情况 A：Prefix 命中，SWA 未被 tombstone → 完美复用

```python
# compress_hip.py:205-215 (prefill extend 时恢复 state)
pre_state_indices = compute_state_len_indices(prefix_len, ratio)
raw_loc = req_to_token[req_pool_indices[i], pre_state_indices]
swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(raw_loc)
state_loc = state_pool.translate_from_swa_loc_to_state_loc(swa_loc)
pre_kv_state = state_pool.get_state_by_state_loc(state_loc)  # 读取 partial state
kv_and_score_buffer = KVAndScore.cat([pre_kv_state, kv_and_score], dim=0)  # 拼接后继续累积
```

请求 B 匹配了请求 A 的前缀 → tree node 仍持有 SWA lock → SWA slot 有效 → state slot 有效 → **直接从 partial state 恢复累积，无需重计算**。

示例：
```
prefix_len = 13, ratio = 4:
- 已完成 3 个压缩窗口 (token 0-11) → compressed_kv[0..2] 已在 c4_kv_pool
- 未完成: token 12 (窗口内第 1 个)
- State buffer 存了 token 12 的 (kv, score)
- 新请求从 token 13 继续，读取 state 后与 token 13,14,15 拼接 → 完成窗口 → 输出 compressed_kv[3]
```

#### 情况 B：Prefix 节点被 SWA tombstone → 匹配被截断

SWA slot 已归还 → state 内容不可信（可能被其他请求覆盖）。

`_match_prefix_helper` 中的 tombstone 感知（`swa_radix_cache.py`）确保不会使用失效的 state：

```python
if child.swa_tombstone:
    match_len_since_tombstone = 0  # 重置计数器

# 只有 tombstone 后有 ≥ sliding_window_size 个有效 token 才接受匹配
if match_len_since_tombstone >= self.sliding_window_size:
    best_value_len = len(value)
```

结果：prefix 匹配被截断到最后一个有完整 SWA/state 数据的位置，新请求从截断点重新 prefill。

#### 情况 C：完全驱逐 → 从头计算

full + SWA 都释放 → tree node 删除 → 无匹配 → 全量 prefill。

### 8.9 State Pool 的设计哲学

| 设计决策 | 效果 |
|----------|------|
| 无独立 allocator | slot 与 SWA slot 一一映射，零管理开销 |
| 无引用计数 | 跟随 SWA pool 生命周期 |
| 无 eviction 策略 | ring 覆写 = 隐式回收 |
| 确定性索引公式 | 任何时候从 swa_loc 可直接算出 state_loc |

Prefix cache 兼容性完全由 SWA tombstone 机制间接保证——只要 SWA slot 活着，state 就有效；SWA 被释放，compressor 不会尝试读取对应 state。

### 8.10 投机解码翻倍 Ring Size 的原因

```
正常 c4:  ring_size = 8  = 2 × compress_ratio → 保留 2 个窗口
投机 c4:  ring_size = 16 = 4 × compress_ratio → 保留 4 个窗口
```

Draft 产生的多个 token 可能在 verify 后被拒绝，需要 **回退到更早的 state 状态**。额外 2 个窗口的缓冲保证回退时 partial state 仍然可读。

### 8.11 c128 在线模式的特殊 State

在线 c128（`SGLANG_OPT_USE_ONLINE_COMPRESS`）使用 incremental online softmax，每个 slot 存储三元组：

```
[max | sum | kv]  各 512 维 (head_dim)，共 1536 FP32
```

更新公式（`c128_online_v2.cuh`）：
```cpp
// 每到达一个新 token：
new_max = fmaxf(old_max, new_score);
old_sum = sum * expf(old_max - new_max);
new_exp = expf(new_score - new_max);
new_sum = old_sum + new_exp;
out_kv = (old_kv * old_sum + new_kv * new_exp) / new_sum;
out_max = new_max;
out_sum = new_sum;
```

优势：`ring_size=1`，只需 1 个 slot 即可完成 128 token 的增量压缩。
限制：不支持投机解码（无法回退增量状态）。

---

## 9. 普通 c128 vs Online c128 详细对比

### 9.1 核心差异：「存原始数据再一次性算」vs「逐 token 增量更新」

| 维度 | 普通 c128 | Online c128 |
|------|-----------|-------------|
| State 存储 | 原始 (kv, score) × 128 个 | (max, sum, kv) × 1 个 |
| last_dim | 2 × head_dim = 1024 | 3 × head_dim = 1536 |
| ring_size | 128 | 1 |
| 每 SWA page 内存 | 128 × 1024 × 4B = **512 KB** | 1 × 1536 × 4B = **6 KB** |
| 内存节省 | — | **~85×** per page |
| 计算模式 | 存满 128 个后一次性 softmax | 每个 token 增量更新 |
| 支持投机解码 | 是 (ring_size=256) | 否 |
| 支持 CUDA Graph | 是 | 否 |
| 数值精度 | 一次性 softmax（数值最稳定） | 增量 softmax（理论等价，浮点误差略大） |

### 9.2 普通 c128 工作流程

**State Buffer 形状**：`[num_slots, 128, head_dim × 2]`

每个 slot 存原始 (kv, score) 对：
```
[0, head_dim)     → kv    (该 token 的内容向量)
[head_dim, 2*hd)  → score (该 token 的权重向量)

一个完整 state "page" = 128 个 slot = 一整个压缩窗口的原始数据
```

**Decode 流程**（`c128_v2.cuh`）：

```cpp
// 1. 先把当前 token 的 (kv, score) 写入 buffer 对应位置
c128_write_decode(kv_dst, kv_src);

// 2. 判断是否凑齐 128 个
if (plan.write_loc % 128 == 127) {
    // 一次性读取 128 个 (kv, score)，执行压缩
    c128_forward(kv_buf, kv_src, kv_out, score_bias, 128);
}
```

**压缩算法（`c128_forward`）**：

```cpp
// 从 buffer 加载 128 个 (kv, score)
for (j = 0..127):
    kv[j] = load(buffer[j].kv)
    score[j] = load(buffer[j].score) + bias[j]    // + APE

// 分 16 个 warp, 每个 warp 处理 8 个位置:
// local max → local sum → local product
// 再 cross-warp reduction:
global_val_max = warp::reduce_max<16>(local_val_max)
rescale = expf(local_val_max - global_val_max)
global_exp_sum = warp::reduce_sum<16>(local_exp_sum * rescale)
final_scale = rescale / global_exp_sum
global_product = warp::reduce_sum<16>(local_product * final_scale)

// 输出: kv_out = Σ(kv[i] * softmax(score[i] + APE[i]))
kv_out = global_product
```

**Prefill 流程** — 两个独立 kernel launch：

```
1. write_c128_prefill  — 把尾部 partial token 的 (kv, score) 写入 buffer
2. flash_c128_prefill  — 对完整 128-chunk 一次性 softmax 压缩输出
```

### 9.3 Online c128 工作流程

**State Buffer 形状**：`[num_slots, 1, head_dim × 3]`

每个 slot 存 running state 三元组：
```
[0, head_dim)       → max  (running max of scores, 逐元素)
[head_dim, 2*hd)    → sum  (running sum of exp(score - max), 逐元素)
[2*hd, 3*hd)        → kv   (running weighted-average kv, 逐元素)
```

**形状解读**：中间维度 `1` = ring_size=1（每个 SWA page 只需 1 个时间槽）；
last_dim = `head_dim × 3` 因为三元组是**逐元素**的（每个维度 d 独立维护自己的 max/sum/kv），详见 §9.7。

**Decode 流程**（`c128_online_v2.cuh:76-113`）：

```cpp
if (pos_in_chunk != 0) {
    // Mid-chunk: 增量合并当前 token 到 running state
    old_max = load(buffer.max);
    old_sum = load(buffer.sum);
    old_kv  = load(buffer.kv);

    new_score = kv_score_input.score + bias[pos_in_chunk];
    new_max = fmaxf(old_max, new_score);
    old_sum_rescaled = old_sum * expf(old_max - new_max);   // 稳定化
    new_exp = expf(new_score - new_max);
    new_sum = old_sum_rescaled + new_exp;
    out_kv = (old_kv * old_sum_rescaled + new_kv * new_exp) / new_sum;
    out_max = new_max;
    out_sum = new_sum;
} else {
    // First token of chunk: 初始化 state
    out_kv  = new_kv;
    out_max = new_score + bias[0];
    out_sum = 1.0f;
}

if (pos_in_chunk == 127) {
    // Chunk closed: state.kv 就是最终 compressed_kv, 直接输出
    store(kv_output, out_kv);
    // 不写回 buffer (state 自然清零, 下一个 chunk 重新 init)
} else {
    // 写回 running state
    store(buffer.max, out_max);
    store(buffer.sum, out_sum);
    store(buffer.kv,  out_kv);
}
```

**数学等价性**：Online softmax 与 batch softmax 计算结果相同：

$$\text{out\_kv} = \frac{\sum_{i=0}^{n} \text{kv}_i \cdot e^{s_i - m}}{\sum_{i=0}^{n} e^{s_i - m}}$$

其中 $m = \max(s_0, ..., s_n)$，增量维护 $(m, \sum e^{s-m}, \text{weighted\_kv})$ 三元组即可。

**Prefill 流程**（`c128_online_v2.cuh:234-377`）— 也是两个 pass：

```
Pass 1 - Compress pass (kWrite=false):
  → 对跨越 128 边界的 segment:
    a. segment 内做 warp-tile softmax → 得到 (seg_kv, seg_max, seg_sum)
    b. 与 prior partial state (从 buffer 读取) 做 online merge:
       new_max = fmaxf(buf_max, seg_max)
       new_s1 = buf_sum * expf(buf_max - new_max)
       new_s2 = seg_sum * expf(seg_max - new_max)
       new_sum = new_s1 + new_s2
       new_kv = (buf_kv * new_s1 + seg_kv * new_s2) / new_sum
    c. 输出 compressed_kv

Pass 2 - Write pass (kWrite=true):
  → 对 trailing partial segment (未凑齐 128):
    a. 同样 segment-internal softmax
    b. 与 prior state merge
    c. 结果写回 state buffer (等后续 token 继续累积)
```

### 9.4 为什么 Online 模式不支持投机解码

```
投机解码流程:
  Draft:  生成 token [n, n+1, n+2, n+3, n+4]
  Verify: target model 验证, 假设 token n+2 被拒绝
  回退:   需要恢复到 token n+1 时的 state

普通 c128:
  buffer 里存着所有 raw (kv, score) —— 位置独立
  → 回退 = 忽略 n+2..n+4 的 buffer 位置, 直接从 n+1 继续
  → ring_size=256 保证足够的回退窗口

Online c128:
  state = (max, sum, kv) 是所有已见 token 的聚合结果
  → token n+2 的 kv/score 已经融入 running sum
  → 无法从聚合中"减去"某个 token 的贡献
  → 回退 = 只能从 chunk 起点重新计算 → 破坏了 online 的意义
```

### 9.5 Kernel 级性能差异

| | 普通 c128 | Online c128 |
|---|---|---|
| **Decode 带宽** | 写 1024B/token + 凑齐时读 128×1024B | 读 1536B + 写 1536B (每 token) |
| **Decode 计算** | 凑齐时：128-wide softmax reduction | 每 token：简单 fmaxf + expf + div |
| **Prefill 并行度** | 高：write + compress 独立 kernel | 较低：compress + write 都需要串行读 prior state |
| **Latency per token** | 大部分 token 只做 write (极低延迟) | 每个 token 都做 read-compute-write |
| **Throughput** | 批量 softmax 利于 GPU 并行 | 单 token 计算量小但频率高 |

### 9.6 使用场景选择

```
选普通 c128 (默认):
  ✓ 需要投机解码 (MTP/EAGLE)
  ✓ 需要 CUDA Graph (高吞吐场景)
  ✓ 内存不是瓶颈
  ✓ 追求最高计算吞吐

选 Online c128:
  ✓ 内存极度紧张 (节省 ~85× state 空间)
  ✓ 不使用投机解码
  ✓ 可以接受无 CUDA Graph
  ✓ 长序列场景 (state 内存随 SWA pages 线性增长, online 模式增长极慢)
```

启用方式：
```bash
export SGLANG_OPT_USE_ONLINE_COMPRESS=1
```

关键代码路径：
- 判断逻辑：`python/sglang/srt/layers/attention/dsv4/compressor_v2.py:28-30`
- 普通 c128 kernel：`python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`
- Online c128 kernel：`python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh`
- State pool 初始化：`python/sglang/srt/mem_cache/deepseek_v4_compress_state.py:96-103`
- Ring size 选择：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:37-38`

### 9.7 论文公式与代码的对应：Per-Element Softmax

DSV4 论文（HCA 部分）公式：

$$C = H \cdot W_{KV}, \quad Z = H \cdot W_Z \quad \text{(Eq. 20-21)}$$

$$S_{m'i:m'(i+1)-1} = \text{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1} + B) \quad \text{(Eq. 22)}$$

$$C^{\text{Comp}}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j \quad \text{(Eq. 23)}$$

其中 $\odot$ 为 Hadamard（逐元素）乘积。

#### `Softmax_row` 的含义

$Z_{m'i:m'(i+1)-1} + B$ 是 $[m' \times c]$ 矩阵。`Softmax_row` 表示**对每一列（每个维度 d）独立做 softmax，归一化方向是行轴（position）**：

```
       d=0    d=1    d=2   ...  d=511
pos 0: z₀₀    z₀₁    z₀₂        z₀,₅₁₁
pos 1: z₁₀    z₁₁    z₁₂        z₁,₅₁₁
  ...
pos 127: z₁₂₇,₀  ...              z₁₂₇,₅₁₁
         ↓      ↓      ↓            ↓
     softmax softmax softmax    softmax   ← 每列独立归一化, Σⱼ S[j,d]=1
```

因此对输出的每个维度 d：

$$C^{\text{Comp}}_i[d] = \sum_{j=0}^{127} S_j[d] \cdot C_j[d], \quad \text{其中} \sum_{j=0}^{127} S_j[d] = 1$$

**每个维度 d 有自己独立的 softmax 归一化因子**——这就是压缩操作的核心特征。

#### 论文到代码的映射

| 论文 | 代码 | 说明 |
|------|------|------|
| $W_{KV} \in \mathbb{R}^{d \times c}$ | `wkv_gate` 前半部分输出 | 产生 kv (内容) |
| $W_Z \in \mathbb{R}^{d \times c}$ | `wkv_gate` 后半部分输出 | 产生 score (权重) |
| $C_j$ (KV entry) | `kv_score_input[..., :head_dim]` | state 中的 "kv" |
| $Z_j$ (权重) | `kv_score_input[..., head_dim:]` | state 中的 "score" |
| $B$ (位置偏置) | `self.ape` | 可学习 APE 参数 |
| $\text{Softmax}_{\text{row}}$ | per-element softmax across positions | 代码中逐维度归一化 |
| $\odot$ (Hadamard) | `kv[j][i] * exp_score` | 逐元素加权 |

代码中 `wkv_gate` 将论文的两次投影合并为一次：
```python
# 论文: C = H·W_KV, Z = H·W_Z  (两次矩阵乘)
# 代码: [C, Z] = H·wkv_gate     (一次矩阵乘, 输出拼接)
self.wkv_gate = ReplicatedLinear(dim, 2 * coff * head_dim)
```

#### 为什么 Online 三元组必须是 per-element

由于每个维度 d 的 softmax 归一化因子不同：

```
维度 d=0:  Σⱼ exp(score_j[0] - max_0) = sum_0    ← 独立的 (max₀, sum₀)
维度 d=1:  Σⱼ exp(score_j[1] - max_1) = sum_1    ← 独立的 (max₁, sum₁)
...
维度 d=511: Σⱼ exp(score_j[511] - max_511) = sum_511
```

Online 增量计算需要为每个维度维护独立的 running state：

```
max[512]  — 每个维度各自的 running max（数值稳定化）
sum[512]  — 每个维度各自的 running Σexp(s-max)
kv[512]   — 每个维度各自的 weighted average result
```

共 3 × 512 = 1536 floats → `[num_slots, 1, head_dim × 3]`。

如果压缩公式不用 $\odot$ 而是标量权重 $S_j \cdot C_j$（所有维度共享同一归一化因子），则 state 只需
`(1 scalar max, 1 scalar sum, 512-dim kv)` = 514 floats。但论文的 **per-element $\odot$ + Softmax\_row 组合
决定了 state 必须是 3×head_dim 维的**。

#### CUDA kernel 中的对应

普通 c128 kernel（`c128_v2.cuh:143-166`）：
```cpp
// j 遍历 128 个 position (每 warp 8 个), i 遍历 tile elements (head_dim 分片)
for (int32_t i = 0; i < kTileElements; ++i) {         // ← 逐元素
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {  // ← 跨 position
        score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }
    // 对 j 方向做 max → exp → sum → weighted sum
    float max_value = score_fp32[i][0];
    for (j = 1..7): max_value = fmaxf(max_value, score_fp32[i][j]);
    for (j = 0..7):
        exp_score = expf(score_fp32[i][j] - max_value);
        sum_product += kv[j][i] * exp_score;      // ← ⊙ 对应
        sum_exp_value += exp_score;
}
// 然后 cross-warp reduction 合并 16 个 warp 的 partial results
```

Online c128 kernel（`c128_online_v2.cuh:82-93`）：
```cpp
for (uint32_t i = 0; i < kVecSize; ++i) {     // ← 逐元素 (每个维度独立)
    new_max = fmaxf(old_max, new_score);       // 该维度的 running max
    old_sum = sum_score_vec[i] * expf(old_max - new_max);
    new_exp = expf(new_score - new_max);
    new_sum = old_sum + new_exp;               // 该维度的 running sum
    out_kv_vec[i] = (old_kv * old_sum + new_kv * new_exp) / new_sum;  // 该维度的 avg
}
```

两种实现数学上等价，都是 per-element softmax across positions。
