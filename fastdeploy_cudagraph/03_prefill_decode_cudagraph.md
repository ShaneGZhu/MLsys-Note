# Prefill/Decode 阶段中的 CUDAGraph

## 1. ForwardMeta 关键字段

**文件**: `forward_meta.py`, line 62-159

```python
@dataclass
class ForwardMeta:
    # Input tokens IDs of removed padding
    ids_remove_padding: paddle.Tensor

    # Use cuda graph in this step or not.
    # 关键: 控制该step是否使用cudagraph
    step_use_cudagraph: bool = False

    # Flag indicating exist prefill in this step
    exist_prefill: bool = False

    # Forward mode used during attention
    forward_mode: ForwardMode = ForwardMode.MIXED
```

**ForwardMode 枚举** (`forward_meta.py`, line 31-59):

```python
class ForwardMode(IntEnum):
    EXTEND = auto()   # Prefill and Extend mode
    DECODE = auto()   # Decode mode
    MIXED = auto()    # Mixed mode (prefill + decode同时存在)
    NATIVE = auto()   # Native mode
```

**max_len_tensor_cpu 索引说明** (`forward_meta.py`, line 111`):
```python
# max_len_tensor_cpu数组索引:
# [0] = total tokens
# [1] = enc_len (prefill length)
# [2] = dec_len (decode length)
# [5] = kv_len
```

## 2. GPUModelRunner 中决定 CUDAGraph 使用

**文件**: `gpu_model_runner.py`, line 1310-1331

```python
# 1. 判断当前batch类型
only_decode_use_cudagraph = self.use_cudagraph and if_only_decode

# 2. 判断是否prefill-only且启用cudagraph_only_prefill
only_prefill_use_cudagraph = self.use_cudagraph and self.cudagraph_only_prefill and self.only_prefill()

# 3. 设置 step_use_cudagraph 标志
self.forward_meta.step_use_cudagraph = (
    only_prefill_use_cudagraph
    if self.cudagraph_only_prefill
    else only_decode_use_cudagraph and self.forward_meta.ids_remove_padding.shape[0] > 0
)

# 4. 静态分割图模式: 强制启用CUDAGraph
if (
    hasattr(self, "graph_opt_config")
    and self.use_cudagraph
    and self.graph_opt_config.graph_opt_level > 0
    and not self.graph_opt_config.full_cuda_graph
):
    self.forward_meta.step_use_cudagraph = True

# 5. 设置 exist_prefill 标志
self.forward_meta.exist_prefill = self.exist_prefill()
```

## 3. exist_prefill() 判断逻辑

**文件**: `gpu_model_runner.py`, line 285-289

```python
def exist_prefill(self):
    """check whether prefill stage exist"""
    return self.exist_prefill_flag
```

**设置位置**: `gpu_model_runner.py`, line 339-353

```python
def only_prefill(self):
    """check whether prefill only"""
    if_only_prefill = True
    decode_exists = None

    # EP模式下检查所有rank
    if self.fd_config.parallel_config.use_ep and self.fd_config.scheduler_config.splitwise_role == "mixed":
        only_prefill_batch_list = []
        decode_exists = self.exist_decode()
        paddle.distributed.all_gather_object(only_prefill_batch_list, not decode_exists)
        if_if_only_prefill = all(only_prefill_batch_list)

    if_only_prefill = if_only_prefill and not (decode_exists if decode_exists is not None else self.exist_decode())
    return if_only_prefill
```

## 4. Prefill vs Decode 的核心区别

### 4.1 Prefill 阶段特征

| 特征 | 说明 |
|------|------|
| **输入** | 多个token的序列 (prompt) |
| **计算量** | 大 (O(n²) attention, n = sequence length) |
| **KV Cache** | 需要写入 |
| **形状** | variable length, 通常较大 |
| **CUDAGraph** | 较难捕获 (形状变化大) |

### 4.2 Decode 阶段特征

| 特征 | 说明 |
|------|------|
| **输入** | 单个token |
| **计算量** | 小 (O(1) attention) |
| **KV Cache** | 只需读取，写入单个位置 |
| **形状** | 固定batch size，固定shape |
| **CUDAGraph** | 易于捕获和重放 |

## 5. Prefill 阶段的 CUDAGraph 处理

### 5.1 Prefill Capture 配置

**文件**: `config.py`

```python
# Prefill阶段捕获的token数列表
cudagraph_capture_sizes_prefill: list[int] = [1, 2, 4, 8]

# Prefill最大捕获size
max_capture_shape_prefill: int = 512
```

### 5.2 Prefill 的 Capture 逻辑

**文件**: `cudagraph_piecewise_backend.py`, line 165-170

```python
# Static split graph mode: use Static + CUDAGraph for prefill/mixed phase
static_cudagraph_for_prefill = exist_prefill and not self.full_cuda_graph and self.dy2st
```

**触发条件**:
1. `exist_prefill == True` (存在prefill)
2. `full_cuda_graph == False` (使用分割图模式)
3. `dy2st == True` (启用静态图优化)

### 5.3 Prefill 执行路径

```python
# cudagraph_piecewise_backend.py, line 156-248
def __call__(self, **kwargs):
    exist_prefill = kwargs["forward_meta"].exist_prefill

    # Prefill专用shape映射
    if static_cudagraph_for_prefill:
        padding_real_shape = self.real_shape_to_captured_size_prefill[real_shape]
    else:
        padding_real_shape = self.real_shape_to_captured_size[real_shape]

    # 获取entry
    entry = self.concrete_size_entries.get((padding_real_shape, static_cudagraph_for_prefill))

    # Prefill走静态图路径
    if static_cudagraph_for_prefill or static_cudagraph_for_decode:
        return self.run_static_model(entry, **kwargs)  # 静态图capture/replay
```

## 6. Decode 阶段的 CUDAGraph 处理

### 6.1 Decode Capture 配置

**文件**: `config.py`

```python
# Decode阶段捕获的batch sizes
cudagraph_capture_sizes: list[int] = [1, 2, 4, 8, 16, 32, ...]
```

### 6.2 Decode 执行路径

```python
# 条件: 只有decode，无prefill
static_cudagraph_for_decode = not exist_prefill and self.full_cuda_graph and self.dy2st

# 非静态图模式下的decode
if entry.cuda_graph is None:
    # Warmup
    for n in range(entry.num_finished_warmup, self.warm_up_size):
        entry.runnable(**kwargs)

    # Capture
    new_grpah = graphs.CUDAGraph(pool_id=self.unique_memory_pool_id)
    with capture_custom_allreduce():
        new_grpah.capture_begin()
        outputs = entry.runnable(**kwargs)
        new_grpah.capture_end()

# Replay
entry.cuda_graph.replay()
```

## 7. Mixed 模式 (Prefill + Decode)

当一个batch中同时存在prefill请求和decode请求时：

```python
# forward_meta.py, line 40
MIXED = auto()  # Mixed mode

# gpu_model_runner.py, line 1342
self.forward_meta.exist_prefill = self.exist_prefill()

# 如果exist_prefill为True但不是纯prefill，则是Mixed模式
```

**Mixed模式处理**:
- 如果 `full_cuda_graph == False`: 使用静态图 + CUDAGraph
- 如果 `full_cuda_graph == True`: 跳过CUDAGraph (因为形状不规则)

## 8. capture_model 触发流程

**文件**: `gpu_model_runner.py`, line 1857-1948

```python
@sot_warmup_guard(True)
def capture_model(self) -> None:
    """Trigger CUDA Graph capture for all shapes in cuda graph capture list"""
    if not self.use_cudagraph:
        return

    # 模式1: cudagraph_only_prefill - 只捕获prefill
    if self.fd_config.graph_opt_config.cudagraph_only_prefill:
        for num_tokens in sorted(capture_sizes, reverse=True):
            self._dummy_run(
                num_tokens=num_tokens,
                batch_size=self.scheduler_config.max_num_seqs,
                in_capturing=True,
                capture_prefill=True,
            )

    # 模式2: 默认捕获decode
    else:
        for batch_size in sorted(capture_sizes, reverse=True):
            self._dummy_run(
                num_tokens=self.fd_config.get_max_chunk_tokens(),
                batch_size=batch_size,
                in_capturing=True,
            )

@sot_warmup_guard(True)
def capture_model_prefill_and_mixed(self) -> None:
    """Trigger CUDA Graph capture for prefill/mixed phase in static split graph mode."""
    for capture_size in sorted(capture_sizes, reverse=True):
        self._dummy_run(
            num_tokens=capture_size,
            batch_size=1,
            in_capturing=True,
            capture_prefill=True,
        )
```

## 9. 流程图总结

```
┌──────────────────────────────────────────────────────────────┐
│                    GPUModelRunner.forward()                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  1. 判断 batch 类型                                          │
│     - exist_prefill = True?  → Prefill/Mixed                │
│     - exist_prefill = False? → Decode                       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  2. 设置 forward_meta.step_use_cudagraph                    │
│     - decode: 始终使用 (形状固定)                             │
│     - prefill: 仅在 full_cuda_graph=False 时使用             │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  3. GraphOptBackend.__call__()                               │
│     - 根据 step_use_cudagraph 决定是否启用                   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  4. CudaGraphPiecewiseBackend.__call__()                     │
│     - 获取/创建 ConcreteSizeEntry                            │
│     - 执行 capture() 或 replay()                             │
└──────────────────────────────────────────────────────────────┘
```

## 10. 关键代码位置总结

| 功能 | 文件 | 行号 |
|------|------|------|
| ForwardMeta定义 | forward_meta.py | 62-159 |
| step_use_cudagraph设置 | gpu_model_runner.py | 1318-1331 |
| exist_prefill判断 | gpu_model_runner.py | 285-289 |
| Prefill capture | cudagraph_piecewise_backend.py | 170-171 |
| Decode capture | cudagraph_piecewise_backend.py | 196-248 |
| capture触发 | gpu_model_runner.py | 1857-1948 |
