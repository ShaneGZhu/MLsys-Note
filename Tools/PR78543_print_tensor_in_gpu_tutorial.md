# PR #78543 使用指南

## 📌 PR 概述

**标题**: `[phi_ops]Add _print_tensor_in_gpu for CUDA Graph safe GPU tensor debugging`

**作者**: DanielSun11

**状态**: ✅ 已合并 (Merged)

**仓库**: https://github.com/PaddlePaddle/Paddle/pull/78543

**变更统计**: `+1,666 additions, -1 deletion`，涉及 8 个文件

---

## 🎯 核心功能

这个 PR 为 PaddlePaddle 添加了一个 **CUDA Graph 安全的 GPU 张量调试函数**，可以在不触发 host/device 内存传输的情况下直接在 GPU 上打印张量数据。

### 关键特性

- ✅ **CUDA Graph 安全**：可以在 CUDA Graph 捕获期间调用
- ✅ **零内存拷贝**：使用 device-side printf，无需 D2H 传输
- ✅ **完整数据打印**：支持打印 dtype、shape、所有数据值
- ✅ **支持 10 种数据类型**：float32/64, float16, bfloat16, int32/64/16/8, uint8, bool

---

## 📖 使用方法

### 1️⃣ 基础用法

```python
import paddle
from paddle.utils.gpu_utils import _print_tensor_in_gpu

# 设置 GPU 设备
paddle.device.set_device('gpu')

# 创建 GPU 张量
x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])

# 打印张量（输出到 stdout）
_print_tensor_in_gpu(x)
```

**输出格式**：
```
[TensorDebug] dtype : FLOAT32
[TensorDebug] shape : [2, 2]
[TensorDebug] numel : 4
[TensorDebug] data  :
[[1, 2],
 [3, 4]]
```

---

### 2️⃣ 在 CUDA Graph 中使用

这是这个 API 最大的优势——可以在 CUDA Graph 捕获期间安全调用：

```python
import paddle

# 定义包含打印的函数
def my_function(x):
    y = x + 1
    paddle.utils.gpu_utils._print_tensor_in_gpu(y)  # ✅ CUDA Graph 安全
    return y * 2

# CUDA Graph 捕获
x = paddle.randn([2, 3])
graph = paddle.jit.to_static(my_function, input_spec=[x])

# 执行（打印会在 graph 执行时发生）
result = graph(x)
```

**为什么 CUDA Graph 安全？**

因为该 API 自动处理了 stream 选择逻辑：
- 在 CUDA Graph 捕获期间：使用 DeviceContext 的捕获流
- 正常执行时：使用当前 stream

这避免了 `cudaErrorStreamCaptureImplicit (error 906)` 错误。

---

### 3️⃣ 捕获输出到变量

由于输出是通过 CUDA kernel 的 `printf` 直接写入 stdout，需要使用文件描述符重定向来捕获：

```python
import os
import sys
import tempfile
import paddle
from paddle.utils.gpu_utils import _print_tensor_in_gpu

def capture_gpu_print(tensor):
    """捕获 GPU print 输出到字符串"""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        path = f.name

    # 重定向 stdout 到文件
    sys.stdout.flush()
    old_fd = os.dup(1)
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
    os.dup2(fd, 1)
    os.close(fd)

    # 执行打印
    _print_tensor_in_gpu(tensor)
    paddle.device.synchronize()  # 等待 GPU 完成

    # 恢复 stdout
    sys.stdout.flush()
    os.dup2(old_fd, 1)
    os.close(old_fd)

    # 读取输出
    text = open(path).read()
    os.remove(path)
    return text

# 使用示例
x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
output_text = capture_gpu_print(x)
print(output_text)
```

---

### 4️⃣ 从输出文本重建张量

PR 提供了配套的解析函数，可以从打印输出重建张量：

```python
from paddle.utils.gpu_utils import _parse_tensor_from_gpu_print

# 从捕获的文本重建张量
text = """
[TensorDebug] dtype : FLOAT32
[TensorDebug] shape : [2, 2]
[TensorDebug] numel : 4
[TensorDebug] data  :
[[1, 2],
 [3, 4]]
"""

reconstructed_tensor = _parse_tensor_from_gpu_print(text)
print(reconstructed_tensor)
# Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), ...)
```

**解析函数的特性**：
- 支持所有 10 种数据类型
- 支持 0-D（标量）、1-D、2-D、N-D 张量
- 支持空张量（numel == 0）
- 自动处理 bfloat16（转为 float32）
- 自动处理 bool 类型（True/False）

---

## 🔧 技术实现细节

### Stream 选择逻辑（C++ 层）

```cpp
cudaStream_t stream = nullptr;
if (phi::backends::gpu::CUDAGraph::IsCapturing()) {
    // CUDA Graph 捕获中：使用 DeviceContext 的 stream
    auto* dev_ctx = static_cast<phi::GPUContext*>(
        phi::DeviceContextPool::Instance().Get(dense.place()));
    stream = dev_ctx->stream();
} else {
    // 正常执行：使用当前 stream
    const auto device_id = dense.place().GetDeviceId();
    stream = paddle::platform::get_current_stream(device_id)->raw_stream();
}
phi::DebugPrintGPUTensor(dense, stream);
```

**关键设计点**：

1. **Stream 感知**：自动检测是否在 CUDA Graph 捕获期间
2. **零内存传输**：形状信息通过寄存器传递（最多支持 9 维）
3. **Device-side printf**：使用 CUDA kernel 的 printf，直接输出到 stdout
4. **单线程 kernel**：只使用一个 CUDA 线程执行打印，避免输出混乱

### 支持的数据类型映射

| PaddlePaddle dtype | 内部枚举 | NumPy dtype | 特殊处理 |
|-------------------|---------|-------------|---------|
| float32 | FLOAT32 | np.float32 | - |
| float64 | FLOAT64 | np.float64 | - |
| float16 | FLOAT16 | np.float16 | - |
| bfloat16 | BFLOAT16 | np.float32 | ⚠️ NumPy 无 bfloat16，升级为 float32 |
| int32 | INT32 | np.int32 | - |
| int64 | INT64 | np.int64 | - |
| int16 | INT16 | np.int16 | - |
| int8 | INT8 | np.int8 | - |
| uint8 | UINT8 | np.uint8 | - |
| bool | BOOL | np.bool_ | ⚠️ 输出 True/False 字符串 |

---

## 📂 修改的文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `paddle/fluid/pybind/eager_functions.cc` | +55 | 添加 Python 绑定 `_print_tensor_in_gpu` |
| `paddle/phi/kernels/legacy/gpu/tensor_debug.cu` | +325 | CUDA kernel 实现 |
| `paddle/phi/kernels/legacy/gpu/tensor_debug.h` | +43 | C++ API 头文件 |
| `python/paddle/utils/gpu_utils.py` | +291 | Python API 和解析函数 |
| `python/paddle/utils/__init__.py` | +1 | 导出模块 |
| `test/legacy_test/test_print_tensor_in_gpu.py` | +954 | 单元测试（954 行） |
| `paddle/phi/kernels/CMakeLists.txt` | +1 | CMake 配置 |
| `test/legacy_test/CMakeLists.txt` | +1 | 测试配置 |

---

## ⚠️ 使用限制

### 1. 仅支持 GPU 张量

```python
# ❌ 错误：CPU 张量
x_cpu = paddle.to_tensor([1, 2, 3])
_print_tensor_in_gpu(x_cpu)  # InvalidArgument

# ✅ 正确：先移动到 GPU
x_gpu = x_cpu.cuda()
_print_tensor_in_gpu(x_gpu)
```

**错误信息**：
```
InvalidArgument: _print_tensor_in_gpu only supports GPU tensors. Please call tensor.cuda() first.
```

### 2. 仅支持 DenseTensor

```python
# ❌ 错误：稀疏张量不支持
sparse_x = paddle.sparse.sparse_coo_tensor(...)
_print_tensor_in_gpu(sparse_x)  # InvalidArgument

# ✅ 正确：使用稠密张量
dense_x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
_print_tensor_in_gpu(dense_x)
```

**错误信息**：
```
InvalidArgument: _print_tensor_in_gpu only supports DenseTensor.
```

### 3. 需要 CUDA 支持

```python
# CPU-only 安装会报错
if not paddle.is_compiled_with_cuda():
    raise ValueError(
        "paddle.utils._print_tensor_in_gpu is not supported in "
        "CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU "
        "support to call this API."
    )
```

### 4. 最大支持 9 维张量

```cpp
static constexpr int kMaxDims = 9;
```

形状信息通过寄存器传递，最多支持 9 维。超过 9 维的张量会报错。

---

## 🎯 适用场景

### ✅ 推荐使用

| 场景 | 说明 |
|------|------|
| **CUDA Graph 调试** | 可在 graph 捕获期间调用，不会破坏 graph |
| **高性能推理服务** | 避免 D2H 传输，零性能损耗 |
| **在线调试** | 生产环境快速打印 GPU 张量状态 |
| **性能分析** | 检查中间结果而不影响性能 |
| **单元测试** | 验证 GPU kernel 输出 |

### ❌ 不推荐使用

| 场景 | 替代方案 |
|------|----------|
| **普通 CPU 调试** | 直接用 `print(tensor)` |
| **需要格式化输出** | 输出格式固定，无法自定义 |
| **非 DenseTensor** | 先转为 DenseTensor |
| **需要持久化日志** | 需要手动捕获 stdout |

---

## 🧪 测试覆盖

单元测试 `test_print_tensor_in_gpu.py` 覆盖了以下场景：

1. **所有数据类型**：float32/64, float16, bfloat16, int32/64/16/8, uint8, bool
2. **各种形状**：标量（0-D）、1-D、2-D、3-D
3. **边界情况**：空张量（numel == 0）
4. **Stream 顺序**：打印在计算 op 之后执行
5. **CUDA Graph 捕获/重放**
6. **CUDA Graph 数据更新 + 重放**
7. **CUDA Graph 混合 op（add + print）**
8. **CUDA Graph 多数据类型**
9. **错误处理**：CPU 张量、非 DenseTensor
10. **Python API 可达性**：`paddle.utils.gpu_utils._print_tensor_in_gpu`

---

## 🔍 实现原理

### CUDA Kernel 结构

```cpp
template <typename T>
__global__ void PrintTensorKernel(
    const T* data,
    ShapeInfo shape_info,  // 通过寄存器传递
    DebugDtype dtype_tag
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 单线程执行，避免输出混乱
        printf("[TensorDebug] dtype : %s\n", DtypeName(dtype_tag));
        printf("[TensorDebug] shape : [");
        // ... 打印 shape
        printf("[TensorDebug] numel : %lld\n", shape_info.numel);
        printf("[TensorDebug] data  :\n");
        // ... 递归打印数据
    }
}
```

### Python 绑定层

```cpp
static PyObject* eager_api_print_tensor_in_gpu(PyObject* self,
                                               PyObject* args,
                                               PyObject* kwargs) {
    auto tensor = CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0);

    // 检查是否为 DenseTensor
    PADDLE_ENFORCE_EQ(tensor.is_dense_tensor(), true, ...);

    // 检查是否在 GPU 上
    PADDLE_ENFORCE_EQ(dense.place().GetType() == phi::AllocationType::GPU, true, ...);

    // 选择正确的 stream
    cudaStream_t stream = nullptr;
    if (phi::backends::gpu::CUDAGraph::IsCapturing()) {
        stream = dev_ctx->stream();  // 捕获流
    } else {
        stream = get_current_stream(device_id)->raw_stream();  // 当前流
    }

    // 调用 CUDA kernel
    phi::DebugPrintGPUTensor(dense, stream);
    RETURN_PY_NONE
}
```

### Python 解析层

`_parse_tensor_from_gpu_print` 函数的工作流程：

1. **正则提取元数据**：
   ```python
   dtype_m = re.search(r'\[TensorDebug\] dtype\s*:\s*(\w+)', text)
   shape_m = re.search(r'\[TensorDebug\] shape\s*:\s*\[([^\]]*)\]', text)
   numel_m = re.search(r'\[TensorDebug\] numel\s*:\s*(\d+)', text)
   data_m = re.search(r'\[TensorDebug\] data\s*:(.*)', text, re.DOTALL)
   ```

2. **处理特殊情况**：
   - 空张量：`numel == 0`，返回空数组
   - 标量：`shape == []`，直接解析单个值
   - N-D 张量：展平所有括号，按逗号分割

3. **重建张量**：
   ```python
   arr = np.array(flat_values, dtype=np_dtype).reshape(shape)
   t = paddle.to_tensor(arr, dtype=paddle_dtype, place=gpu_place)
   ```

---

## 💡 最佳实践

### 1. 在 CUDA Graph 中使用

```python
import paddle

class MyModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear(x)
        # ✅ 调试中间结果
        paddle.utils.gpu_utils._print_tensor_in_gpu(y)
        return paddle.nn.functional.relu(y)

# 使用 CUDA Graph 加速
model = MyModel()
x = paddle.randn([32, 10])

# 转为静态图（自动使用 CUDA Graph）
static_model = paddle.jit.to_static(model, input_spec=[x])
output = static_model(x)
```

### 2. 性能分析中使用

```python
import paddle
import time

x = paddle.randn([1024, 1024]).cuda()
y = paddle.randn([1024, 1024]).cuda()

# 记录开始时间
paddle.device.synchronize()
start = time.time()

# 计算
z = paddle.matmul(x, y)

# ✅ 打印中间结果（不影响性能）
paddle.utils.gpu_utils._print_tensor_in_gpu(z)

# 记录结束时间
paddle.device.synchronize()
end = time.time()

print(f"Time: {end - start:.6f}s")
```

### 3. 单元测试中使用

```python
import unittest
import paddle
from paddle.utils.gpu_utils import _print_tensor_in_gpu, _parse_tensor_from_gpu_print

class TestMyKernel(unittest.TestCase):
    def test_kernel_output(self):
        x = paddle.randn([2, 3]).cuda()

        # 捕获输出
        output_text = capture_gpu_print(x)

        # 重建张量
        y = _parse_tensor_from_gpu_print(output_text)

        # 验证
        np.testing.assert_allclose(x.numpy(), y.cpu().numpy(), rtol=1e-5)
```

---

## 🔗 相关链接

- **PR 链接**: https://github.com/PaddlePaddle/Paddle/pull/78543
- **代码文件**:
  - C++ 实现: `paddle/phi/kernels/legacy/gpu/tensor_debug.cu`
  - Python API: `python/paddle/utils/gpu_utils.py`
  - 单元测试: `test/legacy_test/test_print_tensor_in_gpu.py`

---

## 📝 总结

这个 PR 提供了一个**生产级的 GPU 张量调试工具**，特别适合在 CUDA Graph 等高性能场景下使用。

**核心优势**：
- ✅ **CUDA Graph 兼容**：填补了 PaddlePaddle 在 GPU 调试工具上的空白
- ✅ **零性能损耗**：无需 D2H 传输，不影响推理性能
- ✅ **完整数据输出**：支持所有主流数据类型
- ✅ **配套解析工具**：可从输出重建张量，便于测试验证

**快速上手**：
```python
import paddle
from paddle.utils.gpu_utils import _print_tensor_in_gpu

x = paddle.randn([2, 3]).cuda()
_print_tensor_in_gpu(x)  # 直接打印到 stdout
```

**适用场景**：CUDA Graph 调试、高性能推理服务、在线调试、性能分析

---

**文档生成时间**: 2026-04-17

**作者**: dodo AI 助手
