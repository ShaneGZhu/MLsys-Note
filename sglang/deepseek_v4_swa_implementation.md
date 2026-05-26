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
