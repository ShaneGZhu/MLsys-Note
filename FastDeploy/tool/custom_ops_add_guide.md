# FastDeploy 添加自定义 CUDA 算子全流程指南

> 梳理从 CUDA Kernel 编写到 Python 调用的完整链路，适用于任意自定义 GPU 算子。

---

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Python 调用层                                    │
│  from fastdeploy.model_executor.ops.gpu import my_op                │
│  result = my_op(input_tensor, attr1, attr2, ...)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                  Python 暴露层 (双路径)                              │
│  路径1: pybind11 → fastdeploy_ops.my_op()  (动态模式)               │
│  路径2: PD_BUILD_STATIC_OP → _C_ops._run_custom_op()  (图模式)     │
│  → import_custom_ops() 合并为统一 my_op() 函数                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                  编译产物                                            │
│  fastdeploy_ops_pd_.so  (包含 PD_BUILD_STATIC_OP + PYBIND11_MODULE)│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                  构建系统                                            │
│  setup_ops.py (CUDAExtension) + build.sh                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│               C++/CUDA 算子层                                       │
│  .h (kernel声明) → .cu (注册+wrapper) → cpp_extensions.cc (pybind) │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: 编写 CUDA Kernel 代码

### 1.1 文件组织方式

有两种组织方式，根据算子复杂度选择：

**方式 A：单文件自包含**（适用于简单算子）

将所有代码放在一个 `.cu` 文件中：

```
custom_ops/gpu_ops/gelu_tanh.cu
```

```cpp
#include "helper.h"

// 1) __device__ 辅助函数
__device__ float gelu_tanh_func(float x) { ... }

// 2) __global__ kernel
__global__ void gelu_tanh_kernel(...) { ... }

// 3) Host wrapper + 注册（见 Step 2）
```

**方式 B：头文件 + 注册文件分离**（适用于复杂算子、模板算子）

将 kernel 代码放在 `.h` 中，注册代码放在 `.cu` 中：

```
custom_ops/gpu_ops/xxx_kernel.h    # CUDA kernel 实现
custom_ops/gpu_ops/xxx.cu          # C++ wrapper + 算子注册
```

> **何时选择方式 B**：kernel 代码量大、使用模板需要显式实例化、或需要被多个 `.cu` 文件复用时。

### 1.2 Kernel 头文件模板（方式 B）

```cpp
// custom_ops/gpu_ops/xxx_kernel.h
#pragma once
#include <cooperative_groups.h>
#include "helper.h"    // WARP_SIZE, PDTraits 等公共定义

// 1) 常量定义
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

// 2) __device__ 工具函数/类
template <typename T>
__device__ inline T my_util_func(T val) { ... }

// 3) __global__ CUDA Kernel
template <typename T>
__global__ void my_op_kernel(T* output, const T* input, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = my_util_func(input[idx]);
    }
}

// 4) Host 端 Kernel 启动器
//    职责：计算 grid/block 维度、shared memory 大小，然后 launch kernel
template <typename T>
void invokeMyOp(T* output, const T* input, int64_t n,
                cudaStream_t stream) {
    int64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    my_op_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input, n);
}

// 5) 显式模板实例化（确保 .cu 文件能链接到符号）
template void invokeMyOp<float>(float*, const float*, int64_t, cudaStream_t);
// 如需其他类型：
// template void invokeMyOp<half>(half*, const half*, int64_t, cudaStream_t);
```

### 1.3 关键设计要点

- **Kernel 和 launcher 都放在 `.h` 中**：因为使用模板，需要在头文件中实现以便实例化
- **`invokeXxx` 是 host 函数**：负责计算 grid/block 维度、动态 shared memory 大小，然后调用 `<<<>>>` 启动 kernel
- **显式模板实例化**：在 `.h` 末尾用 `template void invokeXxx<T>(...)` 实例化所需的类型组合，确保链接时符号可见
- **依赖 `helper.h`**：提供 `WARP_SIZE`、`PDTraits`、`GetEmptyTensor`、`PD_BUILD_STATIC_OP` 等公共定义
- **PDL (Programmatic Dependent Launch) 支持**：对于 Hopper 架构 (SM90+)，可使用 `helper.h` 中的 `launchWithPdlWhenEnabled()` 替代直接 `<<<>>>` 调用

---

## Step 2: 编写 C++ Wrapper + 算子注册 (`.cu` 文件)

### 2.1 文件位置

```
custom_ops/gpu_ops/xxx.cu
```

### 2.2 文件内容模板

```cpp
// custom_ops/gpu_ops/xxx.cu
#include "helper.h"
#include "xxx_kernel.h"    // 如果使用方式 B；方式 A 不需要此行

// ==================== 1. C++ Wrapper 函数 ====================
// 职责: paddle::Tensor <-> 原始指针转换，调用 invokeXxx()
std::vector<paddle::Tensor> MyOp(paddle::Tensor& input,
                                 int attr1,
                                 float attr2) {
    // a) 从输入 Tensor 提取形状/类型/设备信息
    auto input_shape = input.shape();
    PD_CHECK(input_shape.size() == 2);    // 可选：维度检查
    int64_t num_tokens = input_shape[0];
    int64_t num_cols = input_shape[1];
    auto input_type = input.dtype();
    auto place = input.place();

    // b) 分配输出 Tensor
    auto output = paddle::empty({num_tokens, num_cols}, input_type, place);
    // 也可以用 GetEmptyTensor (需 -DPADDLE_DEV 编译选项):
    // auto output = GetEmptyTensor(common::DDim({...}), input_type, place);

    // c) 获取 CUDA stream
    auto stream = input.stream();

    // d) 调用 kernel launcher（传入原始指针）
    invokeMyOp<float>(output.data<float>(),
                      input.data<float>(),
                      input.numel(),
                      stream);

    // e) 返回输出 Tensor 列表
    return {output};
}

// ==================== 2. InferShape 函数 ====================
// 给定输入形状和属性，推导输出形状
// 参数顺序: 每个输入的 shape, 然后是所有 Attr (按 PD_BUILD_STATIC_OP 中的声明顺序)
std::vector<std::vector<int64_t>> MyOpInferShape(
    const std::vector<int64_t>& input_shape,
    int attr1) {
    return {input_shape};  // 输出形状与输入相同
}

// ==================== 3. InferDtype 函数 ====================
// 给定输入类型，推导输出类型
// 参数: 每个输入的 dtype
std::vector<paddle::DataType> MyOpInferDtype(
    const paddle::DataType& input_dtype) {
    return {input_dtype};  // 输出类型与输入相同
}

// ==================== 4. PD_BUILD_STATIC_OP 注册 ====================
PD_BUILD_STATIC_OP(my_op)
    .Inputs({"input"})                        // 输入 Tensor 名称列表
    .Outputs({"output"})                      // 输出 Tensor 名称列表
    .Attrs({"attr1: int", "attr2: float"})    // 属性列表，格式: "名称: 类型"
    .SetKernelFn(PD_KERNEL(MyOp))             // 核心计算函数
    .SetInferShapeFn(PD_INFER_SHAPE(MyOpInferShape))  // 形状推导
    .SetInferDtypeFn(PD_INFER_DTYPE(MyOpInferDtype)); // 类型推导
```

### 2.3 Wrapper 的核心模式

| 步骤 | 代码 | 说明 |
|------|------|------|
| 接受输入 | `paddle::Tensor&` | 接受 Paddle Tensor 引用 |
| 提取信息 | `.shape()`, `.dtype()`, `.place()` | 获取形状、类型、设备 |
| 分配输出 | `paddle::empty({dims}, dtype, place)` | 在 GPU 上分配输出空间 |
| 获取指针 | `.data<T>()` | 获取原始 GPU 指针 |
| 获取 stream | `.stream()` | 获取 CUDA stream |
| 调用 kernel | `invokeXxx<T>(ptr..., stream)` | 传入指针和 stream |
| 返回 | `std::vector<paddle::Tensor>` | 返回输出 Tensor 列表 |

### 2.4 PD_BUILD_STATIC_OP 宏说明

```cpp
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
```

- 注册名会被自动加上 `static_op_` 前缀，即 `my_op` -> `static_op_my_op`
- Python 端通过 `_C_ops._run_custom_op("static_op_my_op", ...)` 调用
- Paddle 构建系统会**自动生成**对应的 Python wrapper 函数
- `.Inputs()` / `.Outputs()` 中的名称仅用于标识，需要与 InferShape/InferDtype 的参数顺序对应
- `.Attrs()` 格式为 `"名称: 类型"`，支持的类型：`int`, `float`, `bool`, `str`, `int64_t` 等

### 2.5 Inplace 操作 (可选)

对于需要原地修改输入的算子，使用 `SetInplaceMap`：

```cpp
PD_BUILD_STATIC_OP(my_op_inplace)
    .Inputs({"input", "state"})
    .Outputs({"output", "state_out"})
    .SetInplaceMap({{"state", "state_out"}})   // state 原地更新
    .SetKernelFn(PD_KERNEL(MyOpInplace))
    .SetInferShapeFn(PD_INFER_SHAPE(MyOpInplaceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MyOpInplaceInferDtype));
```

### 2.6 多输出算子

返回多个 Tensor 时，确保 `.Outputs()` 顺序与 `return {...}` 顺序一致：

```cpp
std::vector<paddle::Tensor> MyMultiOutOp(paddle::Tensor& input, ...) {
    auto output1 = paddle::empty(...);
    auto output2 = paddle::empty(...);
    auto output3 = paddle::empty(...);
    // 调用 kernel...
    return {output1, output2, output3};  // 顺序必须与 .Outputs() 一致
}

PD_BUILD_STATIC_OP(my_multi_out_op)
    .Inputs({"input"})
    .Outputs({"output1", "output2", "output3"})  // 顺序与 return 一致
    // ...
```

### 2.7 多类型分发

如果算子需要支持多种数据类型，可以使用 helper.h 中的 `DISPATCH_FLOAT_FP6_DTYPE` 宏：

```cpp
std::vector<paddle::Tensor> MyOpDispatch(paddle::Tensor& input, ...) {
    auto dtype = input.dtype();
    DISPATCH_FLOAT_FP6_DTYPE(dtype, c_type, {
        // c_type 在此处被定义为对应的 C++ 类型
        invokeMyOp<c_type>(output.data<c_type>(), input.data<c_type>(), ...);
    })
    return {output};
}
```

---

## Step 3: 注册 pybind11 绑定 (`cpp_extensions.cc`)

### 3.1 文件位置

```
custom_ops/gpu_ops/cpp_extensions.cc
```

### 3.2 需要做的两件事

**1) 前向声明**（文件顶部，与其他前向声明放在一起）：

```cpp
std::vector<paddle::Tensor> MyOp(paddle::Tensor& input,
                                 int attr1,
                                 float attr2);
```

**2) 模块注册**（文件底部 `PYBIND11_MODULE` 块中）：

```cpp
PYBIND11_MODULE(fastdeploy_ops, m) {
    // ... 其他算子注册 ...

    m.def("my_op", &MyOp, "my_op description");

    // ... 其他算子注册 ...
}
```

### 3.3 为什么需要双注册

| 维度 | pybind11 (`m.def`) | PD_BUILD_STATIC_OP |
|------|-------------------|--------------------|
| 注册名 | `my_op` | `static_op_my_op` |
| Python 调用 | `fastdeploy_ops.my_op(...)` | `_C_ops._run_custom_op("static_op_my_op", ...)` |
| 适用模式 | 动态图模式 | 静态图/图模式 |
| 自动生成 | 无 (手动注册) | 自动生成 Python wrapper |
| InferShape | 不需要 | 必须提供 |

> **注意**：两者都要注册，确保动态模式和图模式都能调用。`import_custom_ops()` 机制会自动合并两个路径。

---

## Step 4: 编译构建 (`setup_ops.py`)

### 4.1 将源文件加入 sources 列表

在 `custom_ops/setup_ops.py` 的 CUDA sources 列表中添加 `.cu` 文件：

```python
# setup_ops.py, 在 paddle.is_compiled_with_cuda() 分支下的 sources 列表
sources = [
    # ... 已有源文件 ...
    "gpu_ops/my_op.cu",          # <-- 添加新算子的 .cu 文件
    # ... 其他源文件 ...
]
```

> **注意**：
> - 不需要将 `.h` 文件加入 sources，只需加入 `.cu` 和 `.cc` 文件
> - `cpp_extensions.cc` 已在列表中（包含 pybind11 模块），无需重复添加
> - 如果是 MetaX GPU 平台，还需在对应的 `sources` 列表中添加
> - 如果算子仅在特定 SM 版本可用，应放在对应的 `if cc >= XX:` 条件块内

### 4.2 关键编译参数（已预配置）

```python
nvcc_compile_args += [
    "-DPADDLE_DEV",               # 启用 GetEmptyTensor 等 helper 函数
    "-DPADDLE_ON_INFERENCE",      # 推理模式
    "-Igpu_ops",                  # 头文件搜索路径 (使 #include "xxx_kernel.h" 生效)
    "-Ithird_party/cutlass/include",
    # ... gencode flags (根据目标 GPU 架构自动生成) ...
]
```

### 4.3 构建命令

```bash
# 通过 build.sh 构建 (推荐)
bash build.sh

# 或直接调用 setup_ops.py
FD_BUILDING_ARCS='[80]' python setup_ops.py install
```

### 4.4 编译产物

构建完成后，产物被安装到：

```
fastdeploy/model_executor/ops/gpu/
  ├── fastdeploy_ops/              # Python 包目录
  │   ├── __init__.py              # 自动生成的 static_op_* wrapper 函数
  │   └── fastdeploy_ops_pd_.so    # 编译好的共享库
  └── __init__.py                  # import_custom_ops 入口
```

对于特定 SM 版本，可能有 `fastdeploy_ops_90/` 等子目录。

---

## Step 5: Python 侧自动暴露与调用

### 5.1 自动生成的 Python Wrapper

编译后，Paddle 自定义算子系统会自动在 `fastdeploy_ops/__init__.py` 中生成：

```python
@unified
def static_op_my_op(input, attr1, attr2):
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("static_op_my_op", input, attr1, attr2)
        res = []
        start_idx = 0
        res.append(outs[start_idx]); start_idx += 1  # output
        return res[0] if len(res)==1 else res
    else:
        # 静态图模式: 使用 LayerHelper 构建 op
        helper = LayerHelper("static_op_my_op", **locals())
        helper.append_op(type="static_op_my_op", ...)
```

> 此文件由 Paddle 构建系统自动生成，**不要手动修改**。

### 5.2 import_custom_ops 统一调度机制

`fastdeploy/model_executor/ops/gpu/__init__.py` 的加载流程：

```python
from fastdeploy.import_ops import import_custom_ops

PACKAGE = "fastdeploy.model_executor.ops.gpu"
module_path = decide_module()  # 根据 SM 版本选择模块路径
import_custom_ops(PACKAGE, module_path, globals())
```

`import_custom_ops()` 内部逻辑：

1. `importlib.import_module()` 导入 `fastdeploy_ops` 模块
2. 遍历模块中所有函数，加入 `globals()` 命名空间
3. `preprocess_static_op()` 后处理：
   - 找到所有 `static_op_*` 前缀的函数
   - 去掉 `static_op_` 前缀得到算子名（如 `static_op_my_op` -> `my_op`）
   - 如果同时存在 pybind11 函数 `my_op` 和 static op `static_op_my_op`
   - 用 `wrap_unified_op()` 合并：**动态模式走 pybind11，图模式走 static op**
   - 最终 `globals()["my_op"]` 指向统一调度函数

### 5.3 最终 Python 调用方式

```python
# 在应用代码中直接 import
from fastdeploy.model_executor.ops.gpu import my_op

# 调用
result = my_op(input_tensor, attr1=1, attr2=0.5)
```

---

## 完整文件清单与职责

| 文件 | 职责 | 需要新建/修改 |
|------|------|-------------|
| `custom_ops/gpu_ops/xxx_kernel.h` | CUDA kernel 实现 (device 函数 + global kernel + host launcher) | **新建** (方式 B) |
| `custom_ops/gpu_ops/xxx.cu` | C++ wrapper + InferShape + InferDtype + PD_BUILD_STATIC_OP 注册 | **新建** |
| `custom_ops/gpu_ops/cpp_extensions.cc` | pybind11 前向声明 + `m.def()` 注册 | **修改** (添加2处) |
| `custom_ops/gpu_ops/helper.h` | 公共工具 (WARP_SIZE, PDTraits, GetEmptyTensor 等) | 通常不修改 |
| `custom_ops/setup_ops.py` | 编译配置 (sources 列表) | **修改** (添加 .cu 文件) |
| `fastdeploy/model_executor/ops/gpu/__init__.py` | 自动加载算子 | 通常不修改 |
| `fastdeploy/import_ops.py` | import_custom_ops 统一调度机制 | 通常不修改 |

---

## 新增算子完整 Checklist

### 1. 编写 CUDA Kernel

**方式 A (简单算子)**：在 `.cu` 文件中直接实现 kernel

**方式 B (复杂/模板算子)**：新建 `.h` 文件实现 kernel

```cpp
// custom_ops/gpu_ops/xxx_kernel.h
#pragma once
#include "helper.h"

template <typename T>
__global__ void my_op_kernel(T* output, const T* input, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = input[idx] * 2;
}

template <typename T>
void invokeMyOp(T* output, const T* input, int64_t n, cudaStream_t stream) {
    int64_t num_blocks = (n + 255) / 256;
    my_op_kernel<T><<<num_blocks, 256, 0, stream>>>(output, input, n);
}

// 显式模板实例化
template void invokeMyOp<float>(float*, const float*, int64_t, cudaStream_t);
```

### 2. 编写 Wrapper + 注册

新建 `.cu` 文件：

```cpp
// custom_ops/gpu_ops/xxx.cu
#include "helper.h"
#include "xxx_kernel.h"

std::vector<paddle::Tensor> MyOp(paddle::Tensor& input) {
    auto shape = input.shape();
    auto dtype = input.dtype();
    auto place = input.place();
    auto output = paddle::empty(shape, dtype, place);
    auto stream = input.stream();

    invokeMyOp<float>(output.data<float>(), input.data<float>(),
                      input.numel(), stream);
    return {output};
}

std::vector<std::vector<int64_t>> MyOpInferShape(
    const std::vector<int64_t>& input_shape) {
    return {input_shape};
}

std::vector<paddle::DataType> MyOpInferDtype(
    const paddle::DataType& input_dtype) {
    return {input_dtype};
}

PD_BUILD_STATIC_OP(my_op)
    .Inputs({"input"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(MyOp))
    .SetInferShapeFn(PD_INFER_SHAPE(MyOpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MyOpInferDtype));
```

### 3. 注册 pybind11

修改 `custom_ops/gpu_ops/cpp_extensions.cc`：

```cpp
// 文件顶部：前向声明
std::vector<paddle::Tensor> MyOp(paddle::Tensor& input);

// PYBIND11_MODULE 块中：注册
m.def("my_op", &MyOp, "my_op description");
```

### 4. 加入编译列表

修改 `custom_ops/setup_ops.py`，在 `sources` 列表中添加：

```python
"gpu_ops/xxx.cu",
```

### 5. 编译 & 验证

```bash
# 编译
bash build.sh

# Python 验证
python -c "
from fastdeploy.model_executor.ops.gpu import my_op
import paddle
paddle.set_device('gpu')
x = paddle.randn([4, 8])
y = my_op(x)
print(y)
"
```

---

## 关键公共依赖 (`helper.h`)

`helper.h` 提供了所有算子共用的基础设施：

| 功能 | 说明 |
|------|------|
| `WARP_SIZE` | 32 (NVIDIA) 或 64 (CoreX) |
| `PD_BUILD_STATIC_OP(name)` | 注册宏，自动加 `static_op_` 前缀 |
| `PDTraits<D>` | Paddle DataType -> C++ 类型映射 (float/half/bf16/fp8/int8) |
| `GetEmptyTensor()` | 在指定 place 上分配 Tensor (需 `-DPADDLE_DEV`) |
| `GetNumBlocks()` | 根据 SM 数量计算最优 block 数 |
| `GetSMVersion()` | 获取当前 GPU 的 SM 版本 |
| `CUDA_CHECK()` | CUDA 错误检查宏 |
| `AlignedVector<T, Size>` | 向量化内存访问 |
| `launchWithPdlWhenEnabled()` | 支持 PDL (Programmatic Dependent Launch) 的 kernel 启动器 |
| `DISPATCH_FLOAT_FP6_DTYPE` | 数据类型分发宏 |
| `DISPATCH_BOOL_DTYPE` | 布尔编译期分发宏 |
| `cudaCheckError()` | kernel launch 后错误检查 |

---

## 数据流总结

```
Python: my_op(input_tensor, attr1, attr2, ...)
    │
    ├─ 动态模式 -> fastdeploy_ops.my_op()  [pybind11]
    │                -> MyOp()  [C++ wrapper in xxx.cu]
    │                   -> invokeMyOp<T>(..., stream)  [host launcher in xxx_kernel.h]
    │                      -> my_op_kernel<<<...>>>(...)  [CUDA kernel]
    │
    └─ 图模式 -> _C_ops._run_custom_op("static_op_my_op", ...)
                  -> MyOp() -> invokeMyOp() -> kernel<<<...>>>()
```

---

## 注意事项

1. **模板实例化**：如果 kernel 使用模板，必须在 `.h` 末尾显式实例化所需类型组合，否则链接时找不到符号
2. **`-Igpu_ops` 编译选项**：`setup_ops.py` 已配置，确保 `#include "xxx_kernel.h"` 能找到头文件
3. **`-DPADDLE_DEV` 编译选项**：必须定义，否则 `GetEmptyTensor` 等 helper 函数不可用
4. **输出顺序**：`PD_BUILD_STATIC_OP` 的 `.Outputs()` 列表顺序必须与 C++ wrapper 的 `return {...}` 顺序一致
5. **InferShape/InferDtype**：必须实现，否则图模式下无法推导输出形状和类型
6. **pybind11 和 PD_BUILD_STATIC_OP 双注册**：两者都要做，确保动态模式和图模式都能调用
7. **构建后需重启 Python**：修改 `.cu` 文件后需要重新编译，Python 进程需要重启才能加载新的 `.so`
8. **SM 版本条件编译**：如果算子需要特定 SM 版本的指令（如 SM90 的 PDL），在 `setup_ops.py` 中使用 `if cc >= XX:` 条件添加源文件和宏定义
9. **多 GPU 平台**：如果算子需在 MetaX GPU、ROCm 等平台运行，需在 `setup_ops.py` 对应平台的 sources 列表中也添加，并处理平台相关的条件编译宏（如 `PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU`）
