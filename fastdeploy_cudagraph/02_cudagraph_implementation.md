# CUDAGraph 实现深度分析

## 1. CUDAGraph 核心文件

| 文件 | 路径 | 功能 |
|------|------|------|
| **cudagraph_piecewise_backend.py** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/graph_optimization/cudagraph_piecewise_backend.py` | **CUDAGraph核心实现** |
| graph_optimization_backend.py | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/graph_optimization/graph_optimization_backend.py` | 图优化后端封装 |
| communication.py | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/distributed/communication.py` | capture_custom_allreduce |

## 2. 核心数据结构

### 2.1 ConcreteSizeEntry

**文件**: `cudagraph_piecewise_backend.py`, line 37-55

```python
@dataclass
class ConcreteSizeEntry:
    """Record the concrete information corresponding to the current shape(num_tokens)"""

    # Concrete shape
    real_shape: int
    # The size is in cudagraph_capture_sizes
    use_cudagraph: bool = True
    # Has runtime-bs been captured before
    captured: bool = False

    # Need to be captured callable object（dynamic graph or static graph backend）
    runnable: Callable = None
    # Number of completed warmups
    num_finished_warmup: int = 0
    # Captured cuda graph object corresponding to the current real shape
    cuda_graph: Optional[graphs.CUDAGraph] = None
    # Output buffers of cudagraph
    output_buffers: List[Optional[paddle.Tensor]] = field(default_factory=list)
```

**说明**:
- `real_shape`: 实际token数量
- `captured`: 是否已捕获
- `cuda_graph`: Paddle的CUDAGraph对象
- `output_buffers`: 存储capture时的输出buffer，用于replay时返回结果

### 2.2 Dy2StCudaGraphManager (静态图模式)

**文件**: `cudagraph_piecewise_backend.py`, line 58-86

```python
class Dy2StCudaGraphManager:
    def __init__(self):
        self.state = jit_utils.CUDAGraphState.DISABLE
        self.captured_batch_size = set()
        self.batch_size = -1

    def run_impl(self, original_run_impl, inputs, parameters, attrs):
        run_state = self.state
        prog_attrs, cuda_graph_attrs = attrs
        if run_state == jit_utils.CUDAGraphState.REPLAY:
            if self.batch_size not in self.captured_batch_size:
                run_state = jit_utils.CUDAGraphState.DISABLE
        elif run_state == jit_utils.CUDAGraphState.CAPTURE:
            self.captured_batch_size.add(self.batch_size)

        cuda_graph_attrs |= {
            "cuda_graph_state": run_state,
            "cuda_graph_dispatch_key": self.batch_size if run_state != jit_utils.CUDAGraphState.DISABLE else 0,
        }
        return original_run_impl(inputs, parameters, (prog_attrs, cuda_graph_attrs))

    @contextmanager
    def run_impl_guard(self):
        with paddle.jit.dy2static.pir_partial_program.replace_run_impl_guard(
            self.run_impl,
        ):
            yield
```

**CUDAGraphState 枚举**:
- `DISABLE`: 禁用CUDAGraph
- `CAPTURE`: 捕获模式
- `REPLAY`: 重放模式

## 3. Capture 逻辑 (动态图模式)

**文件**: `cudagraph_piecewise_backend.py`, line 196-241

```python
# Capture a new cuda graph
if entry.cuda_graph is None:
    assert (
        real_shape == padding_real_shape
    ), f"real_shape:{real_shape} is not equal to padding_real_shape:{padding_real_shape} when capture new graph."

    # 1. Warmup the model - 预热模型
    for n in range(entry.num_finished_warmup, self.warm_up_size):
        entry.num_finished_warmup += 1
        entry.runnable(**kwargs)

    # 2. Store input addresses for debug
    input_addresses = [x.data_ptr() for (_, x) in kwargs.items() if isinstance(x, paddle.Tensor)]
    entry.input_addresses = input_addresses

    # 3. Create new CUDAGraph object
    new_grpah = graphs.CUDAGraph(pool_id=self.unique_memory_pool_id)
    paddle.device.synchronize()

    # 4. Capture - 捕获开始
    with capture_custom_allreduce():  # 处理自定义all-reduce通信
        new_grpah.capture_begin()
        outputs = entry.runnable(**kwargs)  # 执行forward
        if isinstance(outputs, paddle.Tensor):
            outputs = [outputs]
        new_grpah.capture_end()  # 捕获结束

    # 5. Store output buffer - 存储输出buffer用于replay
    entry.cuda_graph = new_grpah
    for output in outputs:
        if output is not None:
            output_buffer = paddle.zeros_like(output)
            output._share_buffer_to(output_buffer)
            output._clear()
            entry.output_buffers.append(output_buffer)
        else:
            entry.output_buffers.append(None)

    paddle.device.synchronize()
```

## 4. Replay 逻辑

**文件**: `cudagraph_piecewise_backend.py`, line 243-248

```python
# Replay - 重放CUDAGraph
entry.cuda_graph.replay()
logger.debug(f"[CUDA GRAPH][ID:{id(self)}] CUDAGraph replayed for real shape {padding_real_shape}")
if len(entry.output_buffers) == 1:
    return entry.output_buffers[0]
return entry.output_buffers
```

## 5. 静态图模式的 Capture/Replay

**文件**: `cudagraph_piecewise_backend.py`, line 126-154

```python
def run_static_model(self, entry: ConcreteSizeEntry, **kwargs):
    if not entry.captured:
        # Warmup the model
        for n in range(entry.num_finished_warmup, self.warm_up_size):
            entry.num_finished_warmup += 1
            entry.runnable(**kwargs)

        # Capture
        self.cuda_graph_manager.state = jit_utils.CUDAGraphState.CAPTURE
        self.cuda_graph_manager.batch_size = entry.real_shape
        entry.captured = True
        with capture_custom_allreduce():
            with self.cuda_graph_manager.run_impl_guard():
                entry.runnable(**kwargs)

    # Replay
    self.cuda_graph_manager.state = jit_utils.CUDAGraphState.REPLAY
    self.cuda_graph_manager.batch_size = entry.real_shape
    with self.cuda_graph_manager.run_impl_guard():
        return entry.runnable(**kwargs)
```

**关键区别**: 静态图模式使用 `run_impl_guard()` 上下文管理器，通过 `Dy2StCudaGraphManager.run_impl()` 动态设置CUDAGraph状态。

## 6. 主入口 __call__

**文件**: `cudagraph_piecewise_backend.py`, line 156-248

```python
def __call__(self, **kwargs) -> List[paddle.Tensor] | paddle.Tensor:
    # Get real shape (total num tokens)
    ids_remove_padding: paddle.Tensor = kwargs["forward_meta"].ids_remove_padding
    real_shape = ids_remove_padding.shape[0]

    exist_prefill = kwargs["forward_meta"].exist_prefill

    # Static split graph mode: use Static + CUDAGraph for prefill/mixed phase
    static_cudagraph_for_prefill = exist_prefill and not self.full_cuda_graph and self.dy2st
    # Static full graph mode: use Static + CUDAGraph for decode phase only
    static_cudagraph_for_decode = not exist_prefill and self.full_cuda_graph and self.dy2st

    # Get padding_real_shape based on mode
    if static_cudagraph_for_prefill:
        padding_real_shape = self.real_shape_to_captured_size_prefill[real_shape]
    else:
        padding_real_shape = self.real_shape_to_captured_size[real_shape]

    # Get or create entry
    entry = self.concrete_size_entries.get((padding_real_shape, static_cudagraph_for_prefill))

    if not entry.use_cudagraph:
        return entry.runnable(**kwargs)

    # Execution modes with CUDAGraph:
    # - Static split graph mode: Static + CUDAGraph for prefill/mixed, Dynamic + CUDAGraph for decode
    # - Static full graph mode: Dynamic for prefill/mixed, Static + CUDAGraph for decode
    # - Dynamic mode: Dynamic + CUDAGraph for decode only
    if static_cudagraph_for_prefill or static_cudagraph_for_decode:
        return self.run_static_model(entry, **kwargs)

    # Dynamic graph capture logic (line 196-248)
    ...
```

## 7. 与 Custom AllReduce 的协作

**文件**: `communication.py`, line 46-53

```python
@contextmanager
def capture_custom_allreduce():
    global _TP_AR
    ar_context = nullcontext()
    if _TP_AR is not None:
        ar_context = _TP_AR.capture()  # 获取自定义all-reduce的捕获上下文
    with ar_context:
        yield
```

**作用**: 确保自定义 all-reduce 操作也被纳入 CUDAGraph 捕获范围，避免capture时通信被遗漏。

## 8. 配置参数

**文件**: `config.py`, line 1042-1098

```python
class GraphOptimizationConfig:
    # 是否启用CUDAGraph
    self.use_cudagraph: bool = False if paddle.is_compiled_with_xpu() else True

    # Decode阶段捕获的batch sizes列表
    self.cudagraph_capture_sizes: Optional[list[int]] = None

    # Prefill阶段捕获的token数列表
    self.cudagraph_capture_sizes_prefill: list[int] = [1, 2, 4, 8]

    # 预热次数
    self.cudagraph_num_of_warmups: int = 2

    # 是否只捕获prefill
    self.cudagraph_only_prefill: bool = False

    # 完整图模式 vs 分割图模式
    # True: 完整静态图 + CUDAGraph for decode
    # False: 分割静态图 + CUDAGraph for prefill/mixed/decode
    self.full_cuda_graph: bool = True

    # Prefill最大捕获size
    self.max_capture_shape_prefill: int = 512
```

## 9. 执行模式总结

| Mode | Prefill + Mixed | Decode |
|------|-----------------|--------|
| **Dynamic (graph_opt_level=0)** | Dynamic | Dynamic + CUDAGraph |
| **Static Full Graph (full=True)** | Dynamic | Static + CUDAGraph |
| **Static Split Graph (full=False)** | Static + CUDAGraph | Dynamic + CUDAGraph |
