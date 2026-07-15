# R3 多级缓存改造逐行解读

- **Commit**: `f98b2a15adf89d4bd7538341ec704566af2dbbda`
- **标题**: `NewRLInfra-2382 r3 support multi-level cache`
- **作者**: liuyi39@baidu.com
- **日期**: 2026-07-14
- **规模**: 13 files, +1045 / -38

## 0. 阅读前提：术语与背景

### 0.1 R3 = Routing Replay

R3 是 RLHF 训练阶段的**路由复放**机制：训练侧要在 replay 时严格按照 rollout 采样时刻 MoE 层选中的 expert 索引来重放前向，才能得到与采样一致的 logprob。所以推理端在每次 rollout 结束时需要**把每一层 MoE 的 topk expert 索引落到存储**（本地文件或 RDMA p2pstore），训练端再拉取。

历史实现（本 PR 之前）只有 device → host（TopkCapturer 的 `BaseHostCache`）→ 通过独立的 `StoreWrapper` 后台进程写盘/写 RDMA 一条路径。routing 数据的生命周期完全由 R3 插件自己管理，跟 SGLang 的 KV 多级缓存管线是**平行的**，没关系。

### 0.2 HiCache 多级缓存

SGLang 的 HiCache 把 KV 分为三级：

- **L1 Device**：GPU 上的 `token_to_kv_pool`（`MHATokenToKVPool` / `MLATokenToKVPool` 等）。
- **L2 Host**：CPU pinned memory 上的 `token_to_kv_pool_host`，容量比 L1 大很多。
- **L3 Storage**：外部存储后端，如 file / mooncake / 3fs / nixl 等，容量更大且跨机可读。

KV 命中路径：
- 前缀命中 device：直接用。
- 前缀命中 host：`load_back`（H2D）到 device 再用。
- 前缀命中 storage：先 `prefetch`（storage → host），后续 `load_back` 到 device。

写回路径：
- 请求 evict device 时 `write_backup`（D2H）到 host。
- host 满或触发策略时 `page_set`（H2L3）到 storage。

### 0.3 本 PR 的目标

把 R3 routing topk 数据**接入这套 HiCache 管线**，作为一个 "side pool" 跟着 KV 一起在 device/host/storage 三级之间流动。这样 replay 命中 KV cache 时，路由数据也随之被找回，不需要每次都从头 forward。

关键抽象：**Host KV Side State** —— 挂在 host KV pool 生命周期上的额外 host 侧状态。routing topk 是第一个这样的 side state。

## 1. 全局架构

```
                    ┌──────────────────────────────────────────────┐
                    │            RoutedExpertsCapturer             │
                    │   (state_capturer/routed_experts.py)         │
                    │                                              │
                    │  device_cache: [max_bs, L, topk]  (GPU)      │
                    │  host_cache:   [num_host_tokens, L, topk]    │
                    │                                              │
                    │  host_kv_routing_state (新增, owner-only):   │
                    │     HostKVTopkState 与 host KV pool 同尺寸    │
                    │        (num_host_tokens, L, topk)            │
                    └──────────────────────────────────────────────┘
                              ▲                       ▲
                              │ D2H/H2D               │ page get/set
                              │                       │
        ┌─────────────────────┴────────────────────┐  │
        │       HostKVSideStateManager             │  │
        │   (mem_cache/host_kv_side_state.py)      │  │
        │                                          │  │
        │   Providers: [RoutingTopkProvider,...]   │  │
        │   sync_device_to_host(dev_idx, host_idx) │  │
        │   sync_host_to_device(host_idx, dev_idx) │  │
        │   page_set/get(hash_values, host_idx)    │  │
        └──────────────────────────────────────────┘  │
              ▲                       ▲               │
   D2H/H2D    │                       │  register     │
   在这里发生  │                       │               │
              │                       │               ▼
   ┌──────────┴────────┐   ┌──────────┴────────┐  ┌──────────────────┐
   │  HiRadixCache     │   │ UnifiedRadixCache │  │ HiCacheController │
   │ write_backup/     │   │ D2H commit /      │  │ _page_backup /   │
   │  load_back        │   │  H2D commit       │  │ _page_get piggy- │
   └───────────────────┘   └───────────────────┘  │ back page_set/get │
                                                  └───────────────────┘
                                                            │
                                                            ▼
                                                  L3 storage (mooncake/file)
                                                  PoolName.ROUTING key suffix
```

要点：
- **RoutingTopkProvider** 是 `HostKVSideStateProvider` 的实现，只在 attn TP rank 0（"routing owner"）启用。
- 所有 D2H / H2D 都通过 provider 转调 `RoutedExpertsCapturer.copy_routing_*`。
- L3 只在 attach storage 后自动注册；page get / set 是 best-effort 附着在 KV 的 `_page_backup` / `_page_get_or_wait_ready` 里。
- routing 是 **owner-only sidecar pool**：只有 rank 0 有数据，非 owner rank 需要在 all-reduce 里塞中性值以保持形状一致。

## 2. 新增文件：`state_capturer/host_kv_topk_state.py`

**作用**：定义"跟 host KV pool 大小一致的 topk 存储池 + 页级搬运接口"。这是 R3 routing L2 层的**数据容器**（不含策略）。

完整代码（78 行）：

```python
import torch

from sglang.srt.state_capturer.base import BaseHostCache


class HostKVTopkState:
    def __init__(
        self,
        num_host_tokens: int,
        num_layers: int,
        topk_size: int,
        page_size: int,
        name: str,
    ):
        self.cache = BaseHostCache(num_host_tokens, num_layers, topk_size, name=name)
        self.page_size = page_size
        self.size = num_host_tokens
        self.num_layers = num_layers
        self.topk_size = topk_size
```

**逐段解读**：

- **`self.cache = BaseHostCache(...)`**：复用 `state_capturer/base.py:54` 里的 `BaseHostCache`——它是 pinned CPU 张量 `[num_host_tokens, num_layers, topk_size]`，dtype 固定 `int32`。这里 `num_host_tokens` 直接和 host KV pool 的 token 数对齐，等于对每个 host slot 都能存下一份 topk。
- **`page_size`**：HiCache 的分页粒度（比如 64 或 128 token），后面 L3 打包时按 page 拆。

```python
    @property
    def buffer(self) -> torch.Tensor:
        return self.cache.buffer

    @property
    def kv_buffer(self) -> torch.Tensor:
        return self.buffer
```

- **两个 property 都指向底层 buffer**。`kv_buffer` 是为了兼容 storage 后端对"pool 对象"的鸭子类型要求（诸如 `MHATokenToKVPool` 有 `kv_buffer` 属性）。这样 `mooncake_store.register_mem_host_pool_v2` 拿到这个对象就能像操作 KV pool 一样操作它。

```python
    def copy_from_device_slot_cache(
        self,
        device_slot_cache: BaseHostCache,
        device_indices: torch.Tensor,
        host_indices: torch.Tensor,
    ):
        self.buffer[host_indices.cpu()] = device_slot_cache.buffer[device_indices.cpu()]

    def copy_to_device_slot_cache(
        self,
        device_slot_cache: BaseHostCache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ):
        device_slot_cache.buffer[device_indices.cpu()] = self.buffer[host_indices.cpu()]
```

- **命名细节**：这里 `device_slot_cache` 类型标注是 `BaseHostCache` 而**不是** `BaseDeviceCache`——因为 R3 的一个特点是：**routing topk 已经在 forward 结束时通过 `TopkCaptureOutput.finalize()` 从 GPU 拷到 pinned host 上** (`state_capturer/base.py:94`)，也就是说 R3 视角里"device 侧"实际上就是 `BaseHostCache`（`self.host_cache`，`num_host_tokens = max_bs` 或 request pool 大小）。
- 所以这里的搬运其实是**"每请求 slot 上的短存储" → "hicache host token 大池"**，都在 CPU 上做，索引均 `.cpu()`。
- `host_indices` / `device_indices` 从 hicache 的调度层传下来，形状是 1D 的 `[num_transfer_tokens]`。

```python
    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return [self.buffer]

    def get_ksize_per_token(self) -> int:
        return self.num_layers * self.topk_size * 4
```

- **`get_hybrid_pool_buffer`**：mooncake / 其他 storage 后端在注册 pool 时要拿到"一组底层张量"（比如 MLA pool 是 `[k_buffer]`，标准 MHA 是 `[k_buffer, v_buffer]`）。routing 只有一个 buffer。
- **`get_ksize_per_token = num_layers * topk_size * 4`**：单 token 字节数。`* 4` 是因为 dtype 是 `int32`（4 字节）。这个数字被 mooncake 用来计算 page 的字节偏移。

```python
    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        page = self.buffer[index : index + self.page_size]
        if flat:
            return page.flatten().contiguous()
        return page
```

- **page = 一段连续 `page_size` 个 token 的 slice**。`flat=True` 时展平成 1D 用于按字节打包（mooncake 的 `_batch_set` 需要连续 bytes）；`flat=False` 保留 3D 用于本地内存直取。

```python
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.empty(
            self.page_size * self.num_layers * self.topk_size,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
```

- **给 storage 后端**用的"空 page 缓冲区"。有些后端在 `get`（拉取）路径下需要先给它一个 pinned buffer 让它填。`page_size * num_layers * topk_size` 是单 page 的 int32 元素数。

```python
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor):
        self.buffer[index : index + self.page_size] = data_page.view(
            self.page_size, self.num_layers, self.topk_size
        )
```

- 与 `get_data_page(flat=True)` 对偶：storage 拉回 1D 数据后 reshape 回 3D 写回 buffer。

```python
    def get_page_buffer_meta(self, host_indices: torch.Tensor):
        ptr_list = []
        element_size_list = []
        page_bytes = self.page_size * self.get_ksize_per_token()
        for index in host_indices[:: self.page_size].cpu().tolist():
            page = self.buffer[index : index + self.page_size]
            ptr_list.append(page.data_ptr())
            element_size_list.append(page_bytes)
        return ptr_list, element_size_list
```

- 给 **mooncake zero-copy 路径**用：mooncake 用 `(void* ptr, size_t nbytes)` 数组来做 RDMA 传输描述符。
- `host_indices[:: self.page_size]`：只取每个 page 的第一个 token 索引作为 page 起点。
- `page_bytes = page_size * num_layers * topk_size * 4`：每页的字节数。

**总结**：`HostKVTopkState` 就是一个"配了页面/切片语义的 pinned int32 大张量"，接口对齐 storage 后端对"pool"的鸭子类型。它自己不管什么时候搬、搬多少 —— 那都是上层 `HostKVSideStateManager` 和 `HiCacheController` 的事。

## 3. 修改：`state_capturer/routed_experts.py`

### 3.1 imports 与 logger

```python
+import logging
 from typing import Optional

 import numpy as np
 ...
 from sglang.srt.layers.dp_attention import (
     attn_tp_all_gather_into_tensor,
     get_attention_tp_size,
+    get_attn_tensor_model_parallel_rank,
     get_dp_local_slice_cpu,
     is_dp_attention_enabled,
 )
 ...
 from sglang.srt.state_capturer.base import BaseTopkCapturer
+from sglang.srt.state_capturer.host_kv_topk_state import HostKVTopkState
+
+logger = logging.getLogger(__name__)
```

- 引入 **`get_attn_tensor_model_parallel_rank`** 用于判定 owner。
- 引入 **`HostKVTopkState`**，就是上一节新加的类。

### 3.2 构造函数里新增两行

```python
+        # Routing data follows the scheduling lifecycle (each DP rank manages
+        # its own radix tree and host/L3 cache), so use attn TP rank which
+        # is 0 on every DP rank under DP-attention and equals MoE TP rank
+        # otherwise.
+        self._is_routing_owner: bool = get_attn_tensor_model_parallel_rank() == 0
+        self.host_kv_routing_state: Optional[HostKVTopkState] = None
```

**关键设计决策**：**"routing owner" 用 attn TP rank == 0，而不是 MoE TP rank 或全局 rank**。原因：
- 调度层（scheduler / radix tree / hicache）是按 **attn TP** 组织的。DP-attention 场景下，每个 DP rank 内部还有一个 attn TP 组，各自维护自己的 radix 树、host/L3 cache。
- 如果用 MoE TP rank == 0，DP rank > 0 的 attn TP=0 rank 就会漏掉 owner 身份，导致该 DP rank 的 radix 树没人写 routing side state。
- 非 DP-attention 场景下 attn TP rank 与 MoE TP rank 等价，语义不变。

`host_kv_routing_state` 初始化为 `None`，实际分配在 `init_host_kv_routing_state()` 里，因为构造 `RoutedExpertsCapturer` 时还不知道 host KV pool 的最终大小。

### 3.3 `init_host_kv_routing_state`

```python
+    def init_host_kv_routing_state(self, num_host_tokens: int, page_size: int):
+        """Pre-allocate owner-only Host KV routing topk state.
+
+        R3 routed experts are an output-side auxiliary state. Only TP rank 0 owns
+        the Host HiCache/L3 routing cache; non-owner TP ranks keep no host-slot
+        routing cache and skip Host/L3 synchronization.
+        """
+        if self.host_kv_routing_state is not None:
+            return
+        if not self.is_routing_owner():
+            return
+
+        shape = (num_host_tokens, self.num_layers, self.topk_size)
+        self.host_kv_routing_state = HostKVTopkState(
+            num_host_tokens=num_host_tokens,
+            num_layers=self.num_layers,
+            topk_size=self.topk_size,
+            page_size=page_size,
+            name="routed_experts_host_kv_topk",
+        )
+        logger.info(
+            "[R3 routing] owner host_kv_routing_state initialized: shape=%s", shape
+        )
```

- **幂等**：已初始化就直接返回。因为 `HiRadixCache._init_host_kv_side_states` 和 `UnifiedRadixCache._init_host_kv_side_states` 都会调用，构造时机可能重复。
- **非 owner 直接跳过**：这是"owner-only sidecar"约定的落地点。非 owner 上 `host_kv_routing_state` 永远是 `None`。
- 命名 `routed_experts_host_kv_topk` 只是用于日志/pool name。

### 3.4 三个搬运/查询方法

```python
+    def copy_routing_device_to_hicache_host(
+        self, device_indices: torch.Tensor, host_indices: torch.Tensor
+    ):
+        if not self.is_routing_owner() or self.host_kv_routing_state is None:
+            return
+        self.host_kv_routing_state.copy_from_device_slot_cache(
+            self.host_cache, device_indices, host_indices
+        )
+
+    def copy_routing_hicache_host_to_device(
+        self, host_indices: torch.Tensor, device_indices: torch.Tensor
+    ):
+        if not self.is_routing_owner() or self.host_kv_routing_state is None:
+            return
+        self.host_kv_routing_state.copy_to_device_slot_cache(
+            self.host_cache, host_indices, device_indices
+        )
```

- 都以 `is_routing_owner()` 兜底为 no-op，方便上层无脑调用。
- 注意方法名里的 "device" **不是 GPU 而是 R3 视角的 `self.host_cache`**（前面 `HostKVTopkState.copy_from_device_slot_cache` 解释过）。R3 数据早已 D2H 到 pinned CPU。所以整个 side-state 搬运其实是 **"CPU → CPU"**，速度很快。

```python
+    def is_routing_owner(self) -> bool:
+        return self._is_routing_owner
+
+    def can_collect_routing_output(self) -> bool:
+        return self.is_routing_owner()
```

- 两个查询方法。`can_collect_routing_output` 单独存在是为了未来可能加更多条件（例如"routing 已初始化且未 shutdown"）；当前实现就是 owner 判定。
- **`can_collect_routing_output` 会被 `batch_result_processor._maybe_collect_routed_experts` 用**——非 owner rank 不做 routing 数据收集（因为它拿到的路由数据不完整或空）。

**至此 `RoutedExpertsCapturer` 的改动结束**。`capture()` 方法和后面所有原有逻辑没动。

## 4. 新增文件：`mem_cache/host_kv_side_state.py`

这是本 PR 引入的**核心抽象层**：把 provider（谁提供 side state）、manager（怎么调度）、pool spec（怎么和 HiCache 集成）三件事分开。185 行，四个类。

### 4.1 抽象基类 `HostKVSideStateProvider`

```python
class HostKVSideStateProvider:
    pool_name: PoolName

    def enabled(self) -> bool:
        raise NotImplementedError

    def storage_supported(self, storage_backend_type: str) -> bool:
        return True

    def init_host_state(self, num_host_tokens: int, page_size: int) -> None:
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def sync_device_to_host(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> None:
        raise NotImplementedError

    def sync_host_to_device(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> None:
        raise NotImplementedError
```

**六个抽象方法约定了 provider 生命周期**：

| 方法 | 时机 | 用途 |
|------|------|------|
| `enabled()` | 每次操作前 | provider 是否 active（rank/config 决策） |
| `storage_supported(backend)` | attach storage 时 | 检查 L3 后端是否兼容 |
| `init_host_state(N, page)` | host KV pool ready 后 | 分配 host 侧张量 |
| `get_state()` | attach storage / manager 查询 | 拿到底层 pool 对象（要能被 storage 注册） |
| `sync_device_to_host` | KV D2H 时 piggyback | 搬 L1→L2 |
| `sync_host_to_device` | KV H2D 时 piggyback | 搬 L2→L1 |

- `pool_name: PoolName` 是类属性，用于 storage 侧 key 命名。
- `storage_supported` 默认 `True`，让通用 provider 不必重写。

### 4.2 具体 provider：`RoutingTopkProvider`

```python
class RoutingTopkProvider(HostKVSideStateProvider):
    pool_name = PoolName.ROUTING

    def _capturer(self):
        return get_global_experts_capturer()

    def enabled(self) -> bool:
        capturer = self._capturer()
        return capturer is not None and capturer.is_routing_owner()

    def storage_supported(self, storage_backend_type: str) -> bool:
        return storage_backend_type in {"file", "mooncake"}

    def init_host_state(self, num_host_tokens: int, page_size: int) -> None:
        capturer = self._capturer()
        if capturer is None:
            return
        capturer.init_host_kv_routing_state(num_host_tokens, page_size)

    def get_state(self):
        capturer = self._capturer()
        if capturer is None or not capturer.is_routing_owner():
            return None
        return capturer.host_kv_routing_state

    def sync_device_to_host(self, device_indices, host_indices) -> None:
        capturer = self._capturer()
        if capturer is not None:
            capturer.copy_routing_device_to_hicache_host(device_indices, host_indices)

    def sync_host_to_device(self, host_indices, device_indices) -> None:
        capturer = self._capturer()
        if capturer is not None:
            capturer.copy_routing_hicache_host_to_device(host_indices, device_indices)
```

**逐点解读**：

- **`pool_name = PoolName.ROUTING`**：绑到本 PR 在 `hicache_storage.py` 新加的枚举值。
- **`_capturer()` 每次调用**（而不是构造时缓存）：因为 `get_global_experts_capturer()` 有可能在 provider 创建之后才注册。这样在 `RoutedExpertsCapturer` 还未就绪时 provider 也能安全存活。
- **`enabled()` = capturer 存在 && 是 routing owner**：非 owner rank 上 provider 存在但永不 active。
- **`storage_supported = {"file", "mooncake"}`**：只有这两个 L3 后端认领了 `PoolName.ROUTING` 的读写逻辑。3fs / nixl / hf3fs 等还没适配。
- **`init_host_state`**：`enabled()` 内会判 owner，这里的 capturer 判空是为了健壮性（比如测试注入的空环境）。
- **`get_state`**：非 owner 返回 `None`，让 manager 的 `register_with_storage` 直接跳过。
- **`sync_*` 委托给 capturer**：注意这里**没有再判 owner**，是因为 capturer 内部方法自己判过（`copy_routing_*` 都有 `if not self.is_routing_owner()` 兜底）。

### 4.3 `HostKVSideStateManager` 构造

```python
class HostKVSideStateManager:
    def __init__(self, providers: Optional[list[HostKVSideStateProvider]] = None):
        self.providers = providers if providers is not None else [RoutingTopkProvider()]
        self.registered_pool_names: set[PoolName] = set()
```

- **默认 providers = `[RoutingTopkProvider()]`**：使得 HiCacheController 在没显式传时也自动获得 routing 能力。
- **`registered_pool_names`**：跟踪已成功注册到 storage 的 pool，避免 detach/attach 时重复注册。

### 4.4 生命周期方法

```python
    def reset_storage_registration(self) -> None:
        self.registered_pool_names.clear()

    def init_host_states(self, num_host_tokens: int, page_size: int) -> None:
        for provider in self.providers:
            if provider.enabled():
                provider.init_host_state(num_host_tokens, page_size)
```

- **`reset_storage_registration`**：detach storage 时清空——重新 attach 时会再次注册。
- **`init_host_states`**：批量调所有 enabled provider 的 `init_host_state`。仅在 host KV pool ready 时（在 HiRadixCache / UnifiedRadixCache 里）调用一次。

```python
    def sync_device_to_host(self, device_indices, host_indices) -> None:
        for provider in self.providers:
            if provider.enabled():
                provider.sync_device_to_host(device_indices, host_indices)

    def sync_host_to_device(self, host_indices, device_indices) -> None:
        for provider in self.providers:
            if provider.enabled():
                provider.sync_host_to_device(host_indices, device_indices)
```

- 广播式转发。当前只有一个 provider，未来加更多 provider（比如 indexer topk）时不需要改上层。

### 4.5 storage 注册

```python
    def register_with_storage(
        self, storage_backend, storage_backend_type: Optional[str], enable_storage: bool
    ) -> None:
        if not enable_storage:
            return
        for provider in self.providers:
            pool_name = provider.pool_name
            if pool_name in self.registered_pool_names:
                continue
            if not provider.enabled():
                continue
            state = provider.get_state()
            if state is None:
                continue
            if storage_backend_type is None or not provider.storage_supported(
                storage_backend_type
            ):
                logger.warning(
                    "Host KV side state L3 disabled: pool=%s backend=%s is unsupported.",
                    pool_name,
                    storage_backend_type,
                )
                continue
            storage_backend.register_mem_host_pool_v2(state, pool_name)
            self.registered_pool_names.add(pool_name)
```

**流程**（每次 attach storage 时被 `HiCacheController._maybe_register_host_kv_side_states_with_storage` 调用）：

1. `enable_storage=False` 直接跳出（L3 关掉时不注册）。
2. 每个 provider 检查：
   - 已注册过？跳。
   - 未 enabled（如 non-owner）？跳。
   - `get_state()` 是 `None`？跳。
   - backend 不支持？warn + 跳。
3. 满足全部才调 `storage_backend.register_mem_host_pool_v2(state, pool_name)`。
4. 加入 `registered_pool_names`，避免重复注册。

**关键 API**：`register_mem_host_pool_v2(state, pool_name)` 是 storage backend 的公用注册入口。它把这个 `HostKVTopkState` 当作一个 "pool" 挂到后端，后续 batch_set_v2 / batch_get_v2 用 `pool_name` 来路由。

### 4.6 `registered_sidecar_specs` — 给 UnifiedRadixCache 用

```python
    def registered_sidecar_specs(self) -> list[SidecarPoolSpec]:
        specs: list[SidecarPoolSpec] = []
        for pool_name in self.registered_pool_names:
            if pool_name == PoolName.ROUTING:
                specs.append(
                    SidecarPoolSpec(
                        pool_name=PoolName.ROUTING,
                        indices_from_pool=PoolName.KV,
                        hit_policy=PoolHitPolicy.ALL_PAGES,
                    )
                )
        return specs
```

- **`SidecarPoolSpec`** 是 unified radix cache 里的辅助 pool 声明——告诉它"这个 pool 是随 KV pool 走的，取用同一批 host_indices（`indices_from_pool=KV`），命中策略 `ALL_PAGES` 表示所有页都算命中不做单独检查"。
- 只对 `ROUTING` 有效——将来添加新 side state 需要在这里增加分支或改成 provider 自己返回 spec。这里目前是 hard-code，代码坐姿是"最小可用"。

### 4.7 L3 读写 piggyback

```python
    def page_set(self, storage_backend, hash_values, host_indices) -> None:
        for pool_name in self.registered_pool_names:
            try:
                storage_backend.batch_set_v2(
                    [
                        PoolTransfer(
                            name=pool_name,
                            host_indices=host_indices,
                            keys=list(hash_values),
                        )
                    ]
                )
            except Exception:
                logger.debug(
                    "Host KV side state L3 write failed: pool=%s",
                    pool_name,
                    exc_info=True,
                )

    def page_get(self, storage_backend, hash_values, host_indices) -> None:
        # 结构完全对称，改 batch_set_v2 → batch_get_v2
        ...
```

**要点**：
- **`hash_values`** 是 KV backup/prefetch 时算出来的 page hash（每个 page 一个哈希）。routing 用同一套 hash 作为 key，语义是"给定 KV prefix，其对应的 routing topk"。
- **`host_indices`** 是 host KV pool 里对应的 slot indices，routing 也复用这些 indices（因为 `HostKVTopkState.size == host_kv_pool.size`）。
- **best-effort**：包 `try/except`，异常只 `logger.debug`，绝不影响主 KV 路径。这是"L3 副作用"，宁可 miss 也不 hang 主流程。
- **`PoolTransfer`** 是 storage 层通用传输描述符 `(pool_name, host_indices, keys)`。

## 5. 小改动：`mem_cache/hicache_storage.py`

```python
 class PoolName(str, Enum):
     ...
     # Draft KV pool
     DRAFT = "draft"

+    # R3 MoE routing pool
+    ROUTING = "routing"

     def __str__(self) -> str:
         return self.value
```

- 仅新增枚举 `ROUTING = "routing"`。用作 storage 后端里 pool key 的一部分（比如 `_mla_suffix_routing`）。
- 其他 PoolName 值（KV / DRAFT / DEEPSEEK_V4_C4_STATE 等）保持不变。

## 6. 小改动：`mem_cache/storage/mooncake_store/mooncake_store.py`

```python
             PoolName.DEEPSEEK_V4_C4_STATE,
             PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
             PoolName.DEEPSEEK_V4_C128_STATE,
+            PoolName.ROUTING,
         ):
-            # DSA indexer and DeepSeek V4 side pools are page-packed
-            # single-object pools.
+            # DSA indexer, DeepSeek V4 side pools, and R3 routing are
+            # page-packed single-object pools.
             suffixes = [f"_{self.mla_suffix}_{pool_name}"]
```

**含义**：mooncake_store 在生成 storage key 时，根据 pool 类型选不同的 key 后缀策略：
- 标准 KV pool（MHA）分 `_k` / `_v` 双 suffix；
- MLA-like / side pool 只用一个 `_{mla_suffix}_{pool_name}` 后缀。

routing 是"每 page 一个连续 int32 数组"的 single-object pool，语义和 DSA indexer / DeepSeek V4 side state 一样，所以并入同一组。

## 7. 修改：`managers/cache_controller.py`

### 7.1 导入 & 构造函数

```python
+from sglang.srt.mem_cache.host_kv_side_state import HostKVSideStateManager
 ...

 class HiCacheController:
     def __init__(
         self, ...,
         enable_storage_metrics: bool = False,
+        host_kv_side_state_manager: Optional[HostKVSideStateManager] = None,
     ):
         ...
+        # Host KV side states support (best-effort piggyback on target L3 ops).
+        self.host_kv_side_state_manager = (
+            host_kv_side_state_manager or HostKVSideStateManager()
+        )
```

- 参数可选：外部（HiRadixCache）传一个共享 manager，或不传时自造一个。
- **"共享 manager"设计的必要性**：HiRadixCache 和 HiCacheController 都需要访问同一个 manager（一个负责初始化和 D2H/H2D，一个负责 L3 piggyback），所以 HiRadixCache 在自己构造时就先建 manager 再传给 controller。

### 7.2 attach storage 时挂钩

```python
             self.page_set_func = self._page_set_zero_copy

         self._maybe_register_draft_with_storage()
+        self._maybe_register_host_kv_side_states_with_storage()

         # Ensure stop_event is clear before starting threads.
         self.storage_stop_event.clear()
```

- attach 成功后，紧跟 draft pool 注册之后，注册 side states。

```python
         except Exception:
             ...
             self.page_set_func = self._generic_page_set
             self.draft_page_get_func = None
             self.draft_page_set_func = None
+            self.host_kv_side_state_manager.reset_storage_registration()
             raise
```

- attach 失败清理：清空 `registered_pool_names`，下次 attach 从头再来。

### 7.3 detach storage

```python
     def detach_storage_backend(self):
         ...
         self.page_set_func = self._generic_page_set
         self.draft_page_get_func = None
         self.draft_page_set_func = None
+        self.host_kv_side_state_manager.reset_storage_registration()
         # Now it's safe to clear the stop event for future re-attach.
         self.storage_stop_event.clear()
```

- detach 时同步清空注册状态。

### 7.4 显式注册入口

```python
+    def register_host_kv_side_states_if_ready(self) -> None:
+        self._maybe_register_host_kv_side_states_with_storage()
+
+    def _maybe_register_host_kv_side_states_with_storage(self) -> None:
+        """Register Host KV side states with storage when ready."""
+        self.host_kv_side_state_manager.register_with_storage(
+            self.storage_backend,
+            self.storage_backend_type,
+            self.enable_storage,
+        )
```

- `register_host_kv_side_states_if_ready` 是**给外部（radix cache）用的公有接口**：因为 host KV pool 的 `init_host_states` 是 radix cache 触发的，触发之后需要通知 controller "现在可以注册到 storage 了"。
- `_maybe_register_host_kv_side_states_with_storage` 是内部实现，attach 时也会调。两次调用是幂等的（manager 内部有 `registered_pool_names` 去重）。

### 7.5 `_page_backup` 里的 piggyback set

在 `_page_backup` 函数里（原来负责 KV 从 host backup 到 storage），一处新增：

```python
             if self.has_draft:
                 self._draft_page_set(batch_hashes, batch_host_indices)

+            # Best-effort Host KV side state L3 write alongside target.
+            self.host_kv_side_state_manager.page_set(
+                self.storage_backend, batch_hashes, batch_host_indices
+            )
+
             if prefix_keys and len(prefix_keys) > 0:
                 prefix_keys += batch_hashes
             operation.completed_tokens += self.page_size * len(batch_hashes)
```

- 紧跟 draft page_set 之后：draft 有自己的分支，side state 走 manager。
- 复用同一批 `batch_hashes` 和 `batch_host_indices`——保证 routing 数据的 L3 key 与 KV 的 key 天然对齐。

### 7.6 `prefetch` 循环里的 piggyback get

在 `prefetch()` 的批处理循环里（原来负责从 storage 读回 host），一处新增：

```python
             if self.has_draft:
                 self._draft_page_get(batch_hashes, batch_host_indices)

+            # Best-effort Host KV side state L3 read before publishing target completion.
+            self.host_kv_side_state_manager.page_get(
+                self.storage_backend, batch_hashes, batch_host_indices
+            )

             prev_completed_tokens = operation.completed_tokens
             # Get one batch token, and update the completed_tokens if succeed
             extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
```

- **顺序关键**："before publishing target completion" 是说：先把 side state 数据拉回来，再更新 `operation.completed_tokens`，保证外部感知到"prefetch 到位"时 routing 也已到位。

## 8. 修改：`mem_cache/hiradix_cache.py`

### 8.1 构造函数改动

```python
 from sglang.srt.mem_cache.hicache_storage import (
     PoolTransfer,
     PrefetchTimeoutConfig,
 )
+from sglang.srt.mem_cache.host_kv_side_state import HostKVSideStateManager
```

```python
         self.load_cache_event = threading.Event()
+        self.host_kv_side_state_manager = HostKVSideStateManager()
```

- HiRadixCache 自己 hold manager，然后往下传。

```python
         if isinstance(self.kv_cache, DSATokenToKVPool):
             attach_hybrid_dsa_pool_to_hiradix_cache(...)
+            self.cache_controller.host_kv_side_state_manager = (
+                self.host_kv_side_state_manager
+            )
         else:
             self.cache_controller = HiCacheController(
                 ...
                 storage_backend_extra_config=extra_config,
                 enable_storage_metrics=self.enable_storage_metrics,
+                host_kv_side_state_manager=self.host_kv_side_state_manager,
             )
```

- 两个分支的处理方式不同：
  - **DSA 分支** (`attach_hybrid_dsa_pool_to_hiradix_cache`)：controller 已经在函数里构造，事后覆盖它的 `host_kv_side_state_manager`。
  - **标准分支**：构造 controller 时直接注入。
- 这样保证不管哪个 codepath，controller 和 radix cache 用同一个 manager。

### 8.2 构造末尾触发 init

```python
         self._apply_storage_runtime_config(...)
+        self._init_host_kv_side_states()

         # record the nodes with ongoing write through
         self.ongoing_write_through = {}
```

- `_apply_storage_runtime_config` 之后（此时 storage backend 已经 attach 完毕），触发 side state init。

### 8.3 三个私有辅助方法

```python
+    def _init_host_kv_side_states(self) -> None:
+        """Initialize Host KV side states once after host KV pool is ready."""
+        if self.token_to_kv_pool_host is None:
+            return
+        self.host_kv_side_state_manager.init_host_states(
+            self.token_to_kv_pool_host.size,
+            self.token_to_kv_pool_host.page_size,
+        )
+        register = getattr(
+            self.cache_controller, "register_host_kv_side_states_if_ready", None
+        )
+        if register is not None:
+            register()
```

- 直接从 `token_to_kv_pool_host` 拿 `size` 和 `page_size` 传给 manager。
- 之后用 `getattr` 拿 `register_host_kv_side_states_if_ready`——**用 getattr 是防御式**：某些旧版本 controller 可能没有这个方法（DSA 分支的 controller 是外部函数注入的）。有则调，没有则算了（此时如果没 attach storage 也 OK，因为 attach 时会自动注册）。

```python
+    def _sync_host_kv_side_states_device_to_host(
+        self, device_indices, host_indices
+    ) -> None:
+        self.host_kv_side_state_manager.sync_device_to_host(
+            device_indices, host_indices
+        )
+
+    def _sync_host_kv_side_states_host_to_device(
+        self, host_indices, device_indices
+    ) -> None:
+        self.host_kv_side_state_manager.sync_host_to_device(
+            host_indices, device_indices
+        )
```

- 单纯的转发。之所以做成实例方法而不是直接在 caller 里调 manager，是为了以后可能加"HiRadixCache 自己的额外逻辑"时不用改 caller。

### 8.4 `write_backup` 里插入 D2H sync

```python
     def write_backup(self, node: TreeNode, write_back=False) -> int:
         # Backup invariant ...
         ...
         if host_indices is not None:
             node.host_value = host_indices.clone()
             assert len(node.host_value) > 0
+            self._sync_host_kv_side_states_device_to_host(node.value, node.host_value)
             self._track_write_through_node(node, len(node.key))
             if not write_back:
                 self.inc_lock_ref(node)
```

**位置解读**：
- `node.value` 是 device 侧（KV pool 上）的 slot indices，`node.host_value` 是 host 侧 slot indices。刚 `clone()` 完 host_indices，两个都是有效 tensor。
- 在 `_track_write_through_node`（做 KV D2H bookkeeping）**之前**插入 side state 同步，保证 routing D2H 和 KV D2H 逻辑上是原子的（后续 write_through / L3 backup 时能拿到一致的数据）。
- 记住：routing 的 D2H 实际是 CPU→CPU（前面 3.4 解释过），所以这次调用非常快，几乎不阻塞。

### 8.5 `load_back` 里插入 H2D sync

```python
             ...
             return None

+        self._sync_host_kv_side_states_host_to_device(host_indices, device_indices)
+
         self.ongoing_load_back[last_hit_node.id] = last_hit_node
         offset = 0
         for node in nodes_to_load:
```

- **在 `ongoing_load_back` 注册之前** 就把 routing H2D 做了。
- 这里 `host_indices` 和 `device_indices` 都是当前这次 load_back 涉及的 slot indices，已经确认非 None。

## 9. 修改：`mem_cache/unified_radix_cache.py`

UnifiedRadixCache 的 hicache 集成模式与 HiRadixCache 不同——它用 "prefetch/backup 组件 + sidecar pool" 的模式管理多池。改动**更复杂**，涉及 6 个地方。

### 9.1 `ongoing_prefetch` tuple 结构扩展

```python
         ] = {}
         self.ongoing_backup: dict[int, tuple[UnifiedTreeNode, DecLockRefParams]] = {}
```

上方的 `ongoing_prefetch` 类型注解从

```python
tuple[
    prefetch_key, last_host_node, host_indices,
    PrefetchOperation, DecLockRefParams,
    dict[ComponentType, list[PoolTransfer]],
]
```

改为多一列：

```python
tuple[
    prefetch_key, last_host_node, host_indices,
    PrefetchOperation, DecLockRefParams,
    dict[ComponentType, list[PoolTransfer]],
+   list[PoolTransfer],                            # aux_xfers
]
```

**为什么加一列**：UnifiedRadixCache 原本的 `comp_xfers`（第 6 列）按 **component 类型**（`ComponentType`，如 base / draft / hybrid）分组存所有 transfer。但 **routing side state 不属于任何 component**——它是"手工挂在 KV component 生命周期外的辅助"。加一列 `aux_xfers` 专门存这些**"跟 KV 同批次但不归 component 管的额外 pool"**。

### 9.2 `register_sidecar_pool` 加去重

```python
     def register_sidecar_pool(self, spec: SidecarPoolSpec) -> None:
-        self.sidecar_pool_specs.append(spec)
+        if spec not in self.sidecar_pool_specs:
+            self.sidecar_pool_specs.append(spec)
```

- 防止多次 attach / init 时重复注册同一个 spec（幂等）。
- 依赖 `SidecarPoolSpec` 是 `@dataclass` 或有 `__eq__` 的 comparable 类型。

### 9.3 `_init_host_kv_side_states`

```python
+    def _init_host_kv_side_states(self) -> None:
+        """Initialize Host KV side states (e.g. routing topk) after host pool is ready."""
+        if self.cache_controller is None:
+            return
+        cc = self.cache_controller
+        cc.host_kv_side_state_manager.init_host_states(
+            cc.mem_pool_host.size,
+            cc.mem_pool_host.page_size,
+        )
+        register = getattr(cc, "register_host_kv_side_states_if_ready", None)
+        if register is not None:
+            register()
+        for spec in cc.host_kv_side_state_manager.registered_sidecar_specs():
+            self.register_sidecar_pool(spec)
```

**与 HiRadixCache 的区别**：多做一步——把 manager 里的 `registered_sidecar_specs()` 结果注册到自己的 `sidecar_pool_specs`。这是因为 UnifiedRadixCache 的 prefetch/backup 组件需要通过 `sidecar_pool_specs` 得知"有哪些 pool 是 sidecar 类型的"来做特殊路径。

**调用时机**：

```python
             storage_prefetch_threshold=storage_prefetch_threshold,
         )

+        # Host KV side state initialization (routing topk D2H/H2D sync)
+        self._init_host_kv_side_states()

         # State initialization
         self.write_through_threshold = (
```

- 在 storage 相关配置完成之后立即调。

### 9.4 D2H commit 里 sync（同 HiRadixCache 的 write_backup）

```python
         if host_indices is None:
             return 0

+        # Sync host KV side states (e.g. routing topk) D2H
+        self.cache_controller.host_kv_side_state_manager.sync_device_to_host(
+            device_value, host_indices
+        )
+
         # Commit
         kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
         self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
```

- 在 `PoolTransfer` 构造和 commit 之前先 sync。这样后续 commit 触发 L3 write 时 routing 已在 host。

### 9.5 H2D commit 里 sync（同 HiRadixCache 的 load_back）

```python
         if device_indices is None:
             return False

+        # Sync host KV side states (e.g. routing topk) H2D
+        self.cache_controller.host_kv_side_state_manager.sync_host_to_device(
+            kv_xfer.host_indices, device_indices
+        )
+
         # Commit: each component gets only its own transfers
         kv_xfer.device_indices = device_indices
         self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
```

### 9.6 prefetch 起 tuple 时把 aux_xfers 塞进去

```python
             operation,
             anchor_lock_params,
             comp_xfers,
+            aux_xfers,
         )
         self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)
```

- 这里 `aux_xfers` 变量在上面（本 diff 未显示的地方）已经准备好了，就是 side state manager 为本次 prefetch 贡献的 PoolTransfer 列表。

### 9.7 最大改动点：all-reduce 形状对齐

原始代码：

```python
         if self.tp_world_size > 1:
             # Reduce full completed tokens together with the sidecar pools that
             # this prefetch actually transferred, in one all_reduce.
             sidecar_pools = [t.name for xfers in comp_xfers.values() for t in xfers]
             packed = torch.tensor(
                 [completed_tokens] + [hit_pages.get(p, 0) for p in sidecar_pools],
                 dtype=torch.int,
             )
             self._all_reduce_attn_groups(packed, torch.distributed.ReduceOp.MIN)
             min_completed_tokens = int(packed[0].item())
             for i, p in enumerate(sidecar_pools, start=1):
                 hit_pages[p] = int(packed[i].item())
```

新代码：

```python
         if self.tp_world_size > 1:
             # Reduce full completed tokens together with every extra pool in a
             # fixed order. Ranks without an owner-only sidecar (e.g. ROUTING)
             # contribute a neutral page count so tensor shapes stay identical.
             local_extra_pools = {t.name for t in aux_xfers}
             reduce_pools = [p for p in PoolName if p != PoolName.KV]
             completed_pages = completed_tokens // self.page_size
             packed = torch.tensor(
                 [completed_tokens]
                 + [
                     hit_pages.get(p, 0) if p in local_extra_pools else completed_pages
                     for p in reduce_pools
                 ],
                 dtype=torch.int,
             )
             self._all_reduce_attn_groups(packed, torch.distributed.ReduceOp.MIN)
             min_completed_tokens = int(packed[0].item())
             for i, p in enumerate(reduce_pools, start=1):
                 if p in local_extra_pools:
                     hit_pages[p] = int(packed[i].item())
```

**逐行解读**（这段是本 PR 最微妙的部分）：

- **`local_extra_pools = {t.name for t in aux_xfers}`**：本 rank 实际有 transfer 的 aux pool 集合。只有 routing owner rank 的这个集合里会包含 `PoolName.ROUTING`；非 owner rank 里没有。
- **`reduce_pools = [p for p in PoolName if p != PoolName.KV]`**：**所有 rank 一致**的固定顺序枚举，排除 KV（KV 已经用 `completed_tokens` 单独 reduce 了）。
- **`completed_pages = completed_tokens // self.page_size`**：中性填充值。为什么这个值是中性的？因为下一步是 `MIN` all-reduce，中性值必须 **>= 任何 rank 的真实值** 才不影响结果——`completed_tokens // page_size` 是"本 rank 已完成的最大页数"，任意 pool 的 hit_pages 都 ≤ 这个上界。
- **`packed = torch.tensor([completed_tokens] + [...])`**：**所有 rank 上 tensor shape 完全一致**（长度 = 1 + `len(reduce_pools)`），这是 collective 不 hang 的必要条件。
- **`for p in reduce_pools`**：本 rank 有的 pool 用 `hit_pages.get(p, 0)`（真实值），没有的 pool 用 `completed_pages`（中性值）。
- **`self._all_reduce_attn_groups(packed, MIN)`**：在 attn TP 组内做 MIN 归约。取 MIN 是因为 hicache 采取"所有 rank 都命中的最小公共前缀"作为可用长度。
- **回填 `hit_pages`**：只回填本 rank 有的 pool，避免把中性值误写为真实命中。

**为什么原来的实现有 bug**：原来的 `sidecar_pools` 是 `comp_xfers.values()` 平铺得来，routing 是 owner-only sidecar 但不属于任何 component，所以本身就漏了。即使加进去，不同 rank 的 `sidecar_pools` 长度也不同（owner 有 ROUTING、其他 rank 没有），all_reduce shape 不一致会 hang。新方案通过"固定 pool 枚举 + 中性填充"解决。

### 9.8 `terminate_prefetch` / prefetch 完成路径的 tuple 拆解

```python
     def terminate_prefetch(self, req_id: str) -> None:
         if req_id not in self.ongoing_prefetch:
             return
-        _, _, _, operation, _, _ = self.ongoing_prefetch[req_id]
+        _, _, _, operation, _, _, _ = self.ongoing_prefetch[req_id]
         if operation.host_indices is None:
             return
```

- 因 tuple 多一列，解构也要多一个 `_`。

```python
             host_indices,
             operation,
             anchor_lock_params,
-            comp_xfers,
+            _comp_xfers,
+            aux_xfers,
         ) = self.ongoing_prefetch[rid]
         if operation.host_indices is None:
             return
         ...
         self.cache_controller.append_host_mem_release(
             host_indices=host_indices[:completed_tokens],
-            extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
+            extra_pools=aux_xfers,
         )
```

**关键语义修正**：`append_host_mem_release` 是"prefetch 结束/取消后归还 host slot 引用计数"的地方。它需要知道**这次实际转发过的额外 pool 都有哪些**，好一起 dec ref。

- **旧代码**：把 `comp_xfers.values()` 平铺当 `extra_pools`。但 `comp_xfers` 已经涵盖 `BASE_COMPONENT_TYPE`（KV）等，而且这些是"component 内的 pool"，不该在这里被当作 extra pool 归还（会导致 KV 双重 dec）。
- **新代码**：只用 `aux_xfers`——严格意义上的"额外 pool"，也就是 routing 这类。语义正确。

### 9.9 flush 分支同款修正

```python
                     _host_indices,
                     _operation,
                     anchor_lock_params,
-                    comp_xfers,
+                    _comp_xfers,
+                    aux_xfers,
                 ) = info
-                cc.append_host_mem_release(
-                    extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
-                )
+                cc.append_host_mem_release(extra_pools=aux_xfers)
```

- 同样的语义修正，另一处 flush 路径。

## 10. 修改：`managers/scheduler.py`（兜底 flush）

只两处小改，都是往关键状态转换点插入 `flush_deferred_release()`：

### 10.1 `check_memory` 空闲检查前

```python
         if not self.is_fully_idle():
             return

+        # No further decode iteration will follow, so no lagging finalize can
+        # land — release any KV slots parked by the overlap deferred-free path.
+        self.batch_result_processor.flush_deferred_release()
+
         # memory leak check (skipped for hisparse — pool counters intentionally
         # diverge during host-backup, see _get_swa_token_info clamp).
         if not self.enable_hisparse:
```

- **动机**：`check_memory` 在完全空闲时会做 KV pool 计数检查（"是否 leak"）。如果这时 `deferred_release` 里还挂着 slot，pool 计数会显示"占用"，触发假告警。
- **正确性**：`is_fully_idle()` 已经保证没有下一轮 decode 会来落 finalize，所以直接 flush 是安全的。

### 10.2 `handle_pause_generation` 处理前

```python
         self.last_batch = None
         self.cur_batch = None

+        # The last batch's results are processed; no further decode iteration
+        # will follow while paused, so flush any deferred KV releases.
+        self.batch_result_processor.flush_deferred_release()
+
         if recv_req.mode == "retract" and not self.running_batch.is_empty():
             self.running_batch.filter_batch(v1_spec_info_filtered=True)
             if len(self.running_batch.reqs) != 0:
```

- **动机**：暂停生成时，即将进入长时间的 idle。同理不会有下一轮 finalize。
- 位于清空 `last_batch` / `cur_batch` 之后，正是"上一批次的 finalize 已经处理完"的状态点。

## 11. 修改：`managers/scheduler_components/batch_result_processor.py`（overlap deferred release）

这是本 PR **最关键的正确性修复**——不是 R3 直接功能，而是补上 overlap 模式下 R3 topk 落盘时机的漏洞。

### 11.1 dataclass 加字段

```python
 from __future__ import annotations

 import logging
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 from typing import (
     TYPE_CHECKING,
     Callable,
```

```python
     output_streamer: "SchedulerOutputStreamer"
     abort_request: Callable

+    # Under overlap, a finished req's KV slot must not be freed until the NEXT
+    # decode iteration's finalize() has landed its lagging EOS/last-token topk
+    # write. Finished reqs are parked here (with is_insert) and released after
+    # the following iteration's finalize. See process_batch_result_decode.
+    deferred_release: List[Tuple["Req", bool]] = field(default_factory=list)
```

- 用 `field(default_factory=list)` 是 dataclass 里给可变默认值的标准写法（不能直接 `= []`）。
- 元素 `(Req, is_insert: bool)`：`is_insert` 是 `release_kv_cache` 的参数，指示是否是"边释放边插入 radix tree"的模式。

### 11.2 `_maybe_collect_routed_experts` 硬化

```python
     def ...(self, req):
         """...
-        Logs a soft warning if the resulting tensor's row count differs from
+        Raises ValueError if the resulting tensor's row count differs from
         the expected `seqlen - 1 - start_len`, to catch silent regressions.
         """
         if not req.capture_routed_experts:
             return
         capturer = get_global_experts_capturer()
-        if capturer is None:
+        if capturer is None or not capturer.can_collect_routing_output():
             return
         start_len = req.routed_experts_start_len
```

- **新增 `not capturer.can_collect_routing_output()`**：非 routing owner rank 直接跳过（前面 3.4/3.5 讲过）。
- 效果：多卡场景下只有 owner rank 会把 routing 数据从 host_cache slot 抽出来赋给 `req.routed_experts`，其他 rank `req.routed_experts` 保持 None。

```python
         if (
             req.routed_experts is not None
             and req.routed_experts.shape[0] != expected_rows
         ):
-            logger.warning(
+            raise ValueError(
                 "routed_experts row-count mismatch for req %s: got %d, expected %d "
                 "(seqlen=%d, raw_seqlen=%d, cached_tokens=%d, start_len=%s). "
-                "This indicates a silent bug.",
-                req.rid,
-                req.routed_experts.shape[0],
-                expected_rows,
-                seqlen,
-                req.seqlen,
-                req.cached_tokens,
-                req.routed_experts_start_len,
+                "This indicates a silent bug."
+                % (
+                    req.rid,
+                    req.routed_experts.shape[0],
+                    expected_rows,
+                    seqlen,
+                    req.seqlen,
+                    req.cached_tokens,
+                    req.routed_experts_start_len,
+                )
             )
```

- **`logger.warning` → `raise ValueError`**：本 PR 决定这种行数不一致必须硬失败——因为它意味着 routing 数据和 request tokens 对不上，即使继续跑也会拿到错的 replay 结果。
- 语法从 `logger.warning(fmt, *args)` 改成 `raise ValueError(fmt % (args,))`——因为 exception 不接受 %-style lazy formatting。

### 11.3 `flush_deferred_release`

```python
+    def flush_deferred_release(self):
+        """Release all deferred KV slots unconditionally.
+
+        Called at idle/pause when no further decode iteration (hence no lagging
+        finalize) will follow, so the parked finished reqs can be freed safely.
+        """
+        if not self.deferred_release:
+            return
+        for req, is_insert in self.deferred_release:
+            release_kv_cache(req, self.tree_cache, is_insert=is_insert)
+        self.deferred_release.clear()
```

- 供 scheduler 的 `check_memory` 和 `handle_pause` 调用（前面 10.1/10.2）。
- 无条件释放：调用方已保证没有 pending finalize 会用这些 slot。

### 11.4 `process_batch_result_decode` 关键改动

```python
     def process_batch_result_decode(self, batch, result):
         if result.copy_done is not None:
             result.copy_done.synchronize()
+        # Under overlap, a req that finished on the *previous* loop iteration
+        # shares this batch snapshot (batch.copy() is a shallow reqs copy). Its
+        # EOS/last-token topk finalize lags one iteration and lands here. To keep
+        # that write valid we do NOT free finished reqs' slots immediately (see
+        # _handle_finish_state_updated_req, which parks them in deferred_release);
+        # we free them only AFTER this iteration's finalize below, by which point
+        # the lagging write has already landed on the still-owned slot.
         if result.routed_experts_output is not None:
             result.routed_experts_output.finalize()
             result.routed_experts_output = None
```

**这段大注释是理解本次修复的关键**。它描述了 bug 的物理机制：
1. Overlap 模式下 scheduler 采用**双流水线**：CPU 在 batch N 尚未完成时就已经开始筹备 batch N+1 的输入（`batch.copy()` 是 shallow copy，reqs 是同一批对象引用）。
2. Batch N 里某个 req 在 CPU 端已经标记 finished 并"发出 EOS"，但**GPU 端 batch N 的最后一层 forward 还没算完**——正在算 EOS 那个 token 的 routing topk。
3. Batch N+1 开始执行 forward，同一批 reqs 会走 `_maybe_collect_routed_experts` 之类逻辑。当 batch N+1 的 finalize 到来时，它才把 batch N 那个 EOS token 的 topk 从 GPU 拷回 pinned host（`TopkCaptureOutput.finalize()` = `host_cache.buffer[out_cache_loc] = topk`）。
4. 如果 batch N 结束时立刻 free 掉这个 req 的 KV slot，`out_cache_loc` 指向的 slot 可能被别的 req 拿去用了——**batch N+1 的 finalize 就会写到别的 req 的 slot 上**，导致 routing 数据错位。

修复思路：**推迟一轮释放**。

### 11.5 释放推迟到 finalize 之后

`finalize` 完之后新增：

```python
         if result.indexer_topk_output is not None:
             result.indexer_topk_output.finalize()
             result.indexer_topk_output = None

+        # Now that this iteration's lagging finalize has landed, release the KV
+        # slots of reqs that finished on the previous iteration.
+        if (self.enable_overlap or self.enable_overlap_mlx) and self.deferred_release:
+            for req, is_insert in self.deferred_release:
+                release_kv_cache(req, self.tree_cache, is_insert=is_insert)
+            self.deferred_release.clear()
```

- **触发条件**：`enable_overlap or enable_overlap_mlx`。非 overlap 模式没这个问题（forward 和 CPU 处理同步）。
- **顺序**：先调 `routed_experts_output.finalize()`（把 batch N 的 EOS topk 写进 host_cache），再释放 batch N 已 finish 的 req。此时 slot 还归 req 所有，写入是安全的。

### 11.6 finish 时不再立即 release

原来的代码路径（`_handle_finish_state_updated_req` 之类）里：

```python
                 is_insert = (
                     ...
                     if get_global_server_args().enable_mamba_extra_buffer_lazy()
                     else True
                 )
-                release_kv_cache(req, self.tree_cache, is_insert=is_insert)
+                if self.enable_overlap or self.enable_overlap_mlx:
+                    # Defer the free by one decode iteration so the lagging
+                    # finalize (which writes this req's EOS/last-token topk on
+                    # the NEXT iteration) still lands on the slot while the req
+                    # owns it. Draining happens after that finalize; idle/pause
+                    # flush covers the tail where no next iteration follows.
+                    self.deferred_release.append((req, is_insert))
+                else:
+                    release_kv_cache(req, self.tree_cache, is_insert=is_insert)

             req.time_stats.set_completion_time()
```

- **overlap 分支**：不立即 release，挂到 deferred_release。
- **非 overlap 分支**：保留原逻辑立即 release。

**释放路径全景**：

| 情况 | 释放时机 |
|------|----------|
| 非 overlap | `_handle_finish_state_updated_req` 里立即释放 |
| overlap，有下一轮 decode | 下一轮 decode 的 `process_batch_result_decode` finalize 后释放 |
| overlap，无下一轮（idle） | `Scheduler.check_memory` 空闲检查前 `flush_deferred_release` |
| overlap，无下一轮（pause） | `Scheduler.handle_pause_generation` 里 `flush_deferred_release` |

三个 flush 点组合起来保证任何 tail 场景都不会 slot 泄漏。

## 12. 修改：`internal/r3_plugin/routing_store.py`

### 12.1 `_on_request_finished` 里对 warmup/health 请求宽松处理

```python
     def _on_request_finished(self, req):
         """Transform replay tensors and submit to store."""
         rollout_id = req.rid

         if self.only_last_turn:
-            _r3_validate_rollout_id(rollout_id)
+            try:
+                _r3_validate_rollout_id(rollout_id)
+            except ValueError as e:
+                # Non-RLHF requests (warmup / health probes / any request whose
+                # rid lacks the R3 '<prefix>:<gen_id>:<turn_id>:<segment_id>'
+                # structure) are not part of replay. Skip them instead of
+                # failing, even under routing_debug -- the debug fail-hard is
+                # reserved for real routing-data corruption
+                # (_verify_debug_routing_data), not for non-training rids.
+                logger.debug(f"[R3] skip non-RLHF rid: {e}")
+                return
```

**改动动机**：以前 `only_last_turn=True` 遇到 rid 不符合 R3 格式（例如 sglang server 启动时的 warmup 请求 rid 是随机短串、健康检查探针 rid 以 `HEALTH_CHECK` 开头等）就直接抛异常，导致训练管线跑不通。改成 debug 日志 + return，让非训练流量安静路过。

**为什么在 `routing_debug=True` 时也宽松**：注释解释得很清楚——`routing_debug` 的"hard fail"语义仅面向真实的路由数据损坏（在 `_verify_debug_routing_data` 里检查），不该被无关 rid 触发。

### 12.2 `_verify_debug_routing_data` 报错信息扩展

原来只报第一个不一致坐标，改后**枚举全部**：

```python
-        mismatch = np.argwhere(routing_data != expected)[0]
-        mismatch_idx = tuple(mismatch)
+        # Surface EVERY mismatch (not just the first) so corruption patterns --
+        # e.g. confined to specific layers / tokens / expert slots -- stay visible.
+        diff_mask = routing_data != expected
+        coords = np.argwhere(diff_mask)
+        total_mismatches = len(coords)
+        bad_layers = np.unique(coords[:, 0]).tolist()
+        bad_tokens = np.unique(coords[:, 1]).tolist()
+        bad_ks = np.unique(coords[:, 2]).tolist()
```

- `argwhere(diff_mask)`：得到所有不一致的 `(layer, token, k)` 三元组数组。
- `np.unique(axis)`：按各维度聚合，方便看"损坏是否集中在某几层/某几个 token/某个 topk 槽位"。这对定位 kernel bug 至关重要（比如"只有 layer 3 的 expert 128 出错"暗示某个特定的 permute 逻辑）。

```python
+        def _compact_axis(values):
+            n = len(values)
+            head = ", ".join(str(v) for v in values[:20])
+            tail = "" if n <= 20 else f", ...(+{n - 20} more)"
+            return f"{n}:[{head}{tail}]"
+
+        max_dump = 100
+        dumped = min(max_dump, total_mismatches)
+        details = []
+        for ci in range(dumped):
+            li, ti, ki = coords[ci].tolist()
+            details.append(
+                f"\n      (layer={li}, token={ti}, k={ki}) "
+                f"got={routing_data[li, ti, ki]}, expected={expected[li, ti, ki]}"
+            )
+        trailing = (
+            ""
+            if total_mismatches <= max_dump
+            else f"\n      ...{total_mismatches - max_dump} more mismatch(es) omitted"
+        )
```

- **`_compact_axis`**：把长的 unique 列表压缩成 `n:[前 20 个, ...(+X more)]` 形式，避免日志爆炸但保留数量。
- **`max_dump = 100`**：具体坐标最多打印 100 条，其余用 `...N more mismatches omitted` 收尾。

```python
         raise ValueError(
             "[R3 debug] Data corruption! "
             f"rid={rollout_id}, start_pos={start_pos}, "
             f"shape={tuple(routing_data.shape)}, total_mismatches={total_mismatches}\n"
             f"    affected layers {_compact_axis(bad_layers)}\n"
             f"    affected tokens {_compact_axis(bad_tokens)}\n"
             f"    affected k slots {_compact_axis(bad_ks)}\n"
             f"    first {dumped}/{total_mismatches} mismatches:{''.join(details)}"
             f"{trailing}"
         )
```

- 最终异常信息包含四层：总览、每维聚合、前 100 条明细、trailing 溢出提示。

## 13. 测试改动

### 13.1 `test_internal/r3_plugin/test_routing_store.py`（+12 行）

只改了一个测试用例名和断言，因为 12.1 那处行为改成"跳过"了：

```python
-    def test_invalid_rid_raises(self):
+    def test_invalid_rid_is_skipped(self):
         indexer = np.array([[[1]]], dtype=np.int32)
         wrapper = _make_wrapper(replay_kinds={"indexer"}, only_last_turn=True)
         req = _make_req(indexer_topk=indexer, rid="bad_id")

-        with self.assertRaises(ValueError):
-            wrapper._on_request_finished(req)
+        # Non-RLHF rids (warmup / health probes) are silently skipped under
+        # only_last_turn=True; no exception, no submission.
+        submitted = []
+        wrapper._submit_to_store = (
+            lambda data, rollout_id, replay_kind: submitted.append(rollout_id)
+        )
+        wrapper._on_request_finished(req)
+        self.assertEqual(submitted, [])
```

- 用 monkey-patch 把 `_submit_to_store` 替成记录器，验证"没抛异常且没提交"。

### 13.2 `test/srt/unit/mem_cache/test_r3_multi_cache_unittest.py`（新，472 行）

端到端单元测试，覆盖：
- HostKVSideStateManager 的初始化/搬运/注册流程；
- RoutingTopkProvider 在 owner / non-owner 场景的行为；
- unified radix cache 里 aux_xfers 参与 all-reduce 的形状对齐；
- overlap 模式 deferred release 的正确释放时机；
- L3 read/write piggyback 是 best-effort（异常吞掉）。

因文件较长，此处不逐行列举——建议直接读代码。所有 test class 都用 `CustomTestCase`（`sglang.test.test_utils`），可以用 `python -m pytest test/srt/unit/mem_cache/test_r3_multi_cache_unittest.py` 单独跑。

## 14. 关键概念回顾

### 14.1 "Owner-only Sidecar Pool"

- **sidecar pool**：不属于 KV pool 本身但生命周期与 KV pool 严格绑定的辅助 pool（routing、indexer topk、draft、SWA v_buffer 等）。它们随 KV 一起 D2H/H2D/L3。
- **owner-only**：这个 pool 在多 rank 场景下只由某个特定 rank（这里是 attn TP rank 0）持有真实数据，其他 rank 上是空。
- 为了不让 collective 挂起，"owner-only + all-reduce" 必须用**中性值填充**+**固定 pool 顺序**的技巧。

### 14.2 Best-effort L3 Piggyback

- routing 是"附加信息"，主流程（KV L3 读写）不能因为它挂掉而 fail。
- 因此 manager 的 `page_get` / `page_set` 都套 `try/except`，异常只 debug log。

### 14.3 Overlap 模式下的一轮延迟释放

- Overlap = 上一 batch 的 GPU 后处理与下一 batch 的 CPU 调度重叠。
- 副作用：一个 batch 的 finalize（把 GPU 上的 topk 拷回 host_cache）在**下一个 batch 的 result 处理时**才会真正落地。
- 因此 KV slot 的释放必须延迟一轮，让 lagging finalize 能落到"仍归 finished req 所有"的 slot 上。
- 空闲 / pause 时需要**显式 flush**兜底。

### 14.4 attn TP rank vs MoE TP rank

- **attn TP** 组织"attention 计算 + 调度状态"。
- **MoE TP** 组织"expert 分片"。
- **DP-attention** 场景下这两个是**不同的组**。routing 数据以调度视角为主（跟 radix / host cache 走），所以用 attn TP rank == 0 判定 owner。

## 15. 需要额外注意的坑

1. **`can_collect_routing_output` 返回 False 时 `_maybe_collect_routed_experts` 早退**——这个 return 是本 PR 加的。之前非 owner rank 也在收集，会拿到 shape=0 或 shape 错的数据。改后非 owner rank 上 `req.routed_experts` 永远是 None。下游代码（比如 `StoreWrapper.process_finished_requests`）已经支持 `routed_experts is None` 直接跳过。
2. **`register_mem_host_pool_v2` 依赖 backend 有 v2 API**——如果 backend 只有 v1（旧接口），这行会 AttributeError。这就是为什么 `RoutingTopkProvider.storage_supported` 严格白名单 `{file, mooncake}`。
3. **`sidecar_pool_specs` 去重依赖 `SidecarPoolSpec` 的 `__eq__`**——要确保它是 `@dataclass(frozen=False)` 或已定义 `__eq__`。否则 `if spec not in ...` 会因引用相等而失效。
4. **overlap 的 deferred release 只覆盖 decode 路径**——如果 prefill 阶段就 finished（罕见但存在），走的是不同 codepath，本 PR 未处理，需要观察是否引入 leak。
5. **all-reduce 中性值 `completed_pages`**——依赖 `completed_tokens` 已经是所有 rank 的最小值下界。如果哪天上游改了 pack 顺序，中性值假设需要重新论证。

## 16. 完整改动清单速查

| 文件 | 增/删 | 主要内容 |
|------|-------|----------|
| `state_capturer/host_kv_topk_state.py` | 新 78 | Host KV 侧 topk pool 容器 |
| `state_capturer/routed_experts.py` | +58 | owner 判定 + host_kv_routing_state + D2H/H2D 方法 |
| `mem_cache/host_kv_side_state.py` | 新 185 | Provider / Manager / L3 piggyback 抽象 |
| `mem_cache/hicache_storage.py` | +3 | 新增 `PoolName.ROUTING` |
| `mem_cache/hiradix_cache.py` | +38 | 挂 manager + write_backup/load_back sync |
| `mem_cache/unified_radix_cache.py` | +52/-16 | ongoing_prefetch 加 aux_xfers + all-reduce 形状对齐 |
| `mem_cache/storage/mooncake_store/...` | +3/-2 | `PoolName.ROUTING` 归入 single-object 分支 |
| `managers/cache_controller.py` | +31 | manager 参数 + attach/detach + page_set/get piggyback |
| `managers/scheduler.py` | +8 | idle / pause 时 flush |
| `managers/scheduler_components/batch_result_processor.py` | +55/-13 | deferred_release + overlap 释放推迟一轮 |
| `internal/r3_plugin/routing_store.py` | +45/-10 | warmup rid 宽松 + debug 报错全量枚举 |
| `test_internal/r3_plugin/test_routing_store.py` | +12 | 用例名改 + 断言方向反过来 |
| `test/srt/unit/mem_cache/test_r3_multi_cache_unittest.py` | 新 472 | 端到端单测 |

**总规模**：13 files, +1045 / -38。

## 17. 推荐学习路径

1. 先读 `state_capturer/base.py`（本 PR 未改）搞清 `BaseHostCache` / `BaseTopkCapturer` / `TopkCaptureOutput` 三个原语。
2. 再读 `state_capturer/host_kv_topk_state.py`（本 PR 新增）—— 上一步的自然扩展。
3. 读 `mem_cache/host_kv_side_state.py`（本 PR 新增）—— Provider/Manager 抽象。
4. 读 `state_capturer/routed_experts.py` 的 diff —— provider 的具体实现来源。
5. 读 `mem_cache/hiradix_cache.py` 的 diff —— 简单版接线（先看这个）。
6. 读 `mem_cache/unified_radix_cache.py` 的 diff —— 复杂版接线，重点看 aux_xfers 和 all-reduce。
7. 读 `managers/cache_controller.py` 的 diff —— L3 piggyback。
8. 读 `batch_result_processor.py` + `scheduler.py` 的 diff —— overlap 正确性修复。
9. 最后读 `routing_store.py` 的两处改动 —— 小的运行期改进。
10. 参照测试 `test_r3_multi_cache_unittest.py` 反向验证理解。

