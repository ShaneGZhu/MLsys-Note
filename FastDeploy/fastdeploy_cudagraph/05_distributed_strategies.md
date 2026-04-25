# DeepseekV3.2 分布式策略 (TP/DP/EP)

## 1. 核心文件位置

| 类别 | 文件路径 |
|------|---------|
| **分布式通信** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/distributed/communication.py` |
| **MoE分布式** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/layers/moe/moe.py` |
| **EP实现** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/layers/moe/ep.py` |
| **DeepseekV3MoE** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/models/deepseek_v3.py` |

## 2. 分布式配置

**文件**: `config.py` 中的 `ParallelConfig`

```python
class ParallelConfig:
    # Tensor Parallel (TP) - 张行并行
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    tp_group: Optional["paddle.distributed.Group"] = None

    # Data Parallel (DP) - 数据并行
    data_parallel_size: int = 1

    # Expert Parallel (EP) - Expert并行
    expert_parallel_size: int = 1
    expert_parallel_rank: int = 0
    ep_group: Optional["paddle.distributed.Group"] = None

    # Sequence Parallel - 序列并行
    use_sequence_parallel: bool = False

    # Pipeline Parallel (PP) - 流水线并行
    pipeline_parallel_size: int = 1
```

## 3. Tensor Parallel (TP) 详解

### 3.1 TP 核心思想

```
标准AllReduce: 所有GPU计算相同部分，结果合并
TP AllReduce: 每个GPU计算不同部分（如不同的attention heads），结果需要AllReduce
```

### 3.2 DeepSeekV3 中的 TP 配置

**文件**: `deepseek_v3.py`, line 135-140

```python
class DeepSeekV3MoE(nn.Layer):
    def __init__(self, fd_config, layer_id, prefix):
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.ep_size = fd_config.parallel_config.expert_parallel_size

        # EP和TP互斥 - MoE只支持EP或TP，不同时支持
        if self.ep_size > 1:
            self.tp_size = 1
```

### 3.3 TP AllReduce 操作

**文件**: `communication.py`

```python
def tensor_model_parallel_all_reduce(tensor: paddle.Tensor) -> paddle.Tensor:
    """TP AllReduce - 合并各TP rank的计算结果"""
    if paddle.distributed.get_world_size() == 1:
        return tensor

    tensor_shape = tensor.shape
    output = paddle.empty_like(tensor)

    paddle.distributed.all_reduce(
        tensor.reshape([-1]),
        op=paddle.distributed.ReduceOp.SUM,
        group=tp_group,
    )
    return tensor

def tensor_model_parallel_all_reduce_custom(tensor: paddle.Tensor) -> paddle.Tensor:
    """使用自定义通信组的TP AllReduce"""
    global _TP_AR
    if _TP_AR is None:
        return tensor
    return _TP_AR.all_reduce(tensor)
```

### 3.4 TP 在 MoE 中的使用

**文件**: `deepseek_v3.py`, line 193-205

```python
def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
    """DeepSeekV3MoE forward"""
    # 1. Shared experts 计算
    shared_experts_out = self.shared_experts(hidden_states)

    # 2. EP模式下，shared_experts_out需要TP AllReduce
    if self.attn_tp_size > 1 and self.ep_size > 1:
        shared_experts_out = tensor_model_parallel_all_reduce(shared_experts_out)

    # 3. MoE experts 计算
    moe_out = self.experts(hidden_states, self.gate, forward_meta)

    # 4. 合并结果
    moe_out = moe_out + shared_experts_out

    # 5. TP AllReduce - 在sum of experts之后
    if self.tp_size > 1:
        moe_out = tensor_model_parallel_all_reduce(moe_out)

    return moe_out
```

### 3.5 TP 在 Attention 中的使用

**文件**: `deepseek_v3.py`, line 220-240

```python
class DeepseekV3MLAAttention(nn.Layer):
    def __init__(self, fd_config, layer_id, prefix=""):
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads
        # 每个TP rank负责部分attention heads
        self.num_attention_heads_tp = self.num_attention_heads // self.tp_size
```

```python
# Attention后向传播时会自动进行TP AllReduce
# 通过ColumnParallelLinear和RowParallelLinear实现
```

## 4. Expert Parallel (EP) 详解

### 4.1 EP 核心思想

```
标准MoE: 所有expert在每个GPU上
EP MoE: 每个GPU只有部分expert，通过通信获取其他GPU的expert

例如: 8 experts, 4 GPUs with EP=4
- GPU 0: experts [0, 1]
- GPU 1: experts [2, 3]
- GPU 2: experts [4, 5]
- GPU 3: experts [6, 7]

Token分发: 每个token被路由到top-k experts，可能分布在不同GPU上
```

### 4.2 EP 在 MoE 中的配置

**文件**: `deepseek_v3.py`, line 132-140

```python
class DeepSeekV3MoE(nn.Layer):
    def __init__(self, fd_config, layer_id, prefix):
        self.ep_size = fd_config.parallel_config.expert_parallel_size
        self.tp_size = fd_config.parallel_config.tensor_parallel_size

        # EP模式下，MoE的TP强制设为1
        if self.ep_size > 1:
            self.tp_size = 1
```

### 4.3 EP 通信原语

**文件**: `ep.py`

```python
class DeepEPEngine:
    """DeepEP引擎 - EP通信封装"""

    @staticmethod
    def load_balance(running_batch_size, n_active_experts, ep_size):
        """负载均衡 - 确保各EP rank负载均匀"""
        ...

    @staticmethod
    def dispatch(hidden_states, topk_ids, num_experts_per_ep, ep_size):
        """分发 - 将token分发到对应expert所在的EP rank"""
        ...

    @staticmethod
    def combine(expert_outputs, topk_ids, num_experts_per_ep, ep_size):
        """合并 - 将各EP rank的expert输出合并"""
        ...
```

### 4.4 EP Runner

**文件**: `ep.py`

```python
class EPRunner:
    """EP运行基类"""
    pass

class EPPrefillRunner(EPRunner):
    """Prefill阶段EP Runner"""
    def forward(self, hidden_states, topk_ids, topk_weights):
        # 1. Dispatch tokens to EP ranks
        dispatched_hidden = low_latency_dispatch(hidden_states, topk_ids, ...)

        # 2. Local expert computation
        local_expert_out = self.experts(dispatched_hidden)

        # 3. Combine outputs from all EP ranks
        combined_out = low_latency_combine(local_expert_out, topk_ids, ...)

        return combined_out

class EPDecoderRunner(EPRunner):
    """Decode阶段EP Runner"""
    def forward(self, hidden_states, topk_ids, topk_weights):
        # 类似的dispatch/combine流程
        ...
```

### 4.5 EP 通信函数

**文件**: `ep.py`

```python
def low_latency_dispatch(hidden_states, topk_ids, expert_ids, num_experts_per_ep):
    """
    Low-latency dispatch for EP
    使用all-to-all通信将tokens发送到对应expert所在GPU
    """
    # All-to-all dispatch
    dispatched = paddle.distributed.alltoall(
        tensor=hidden_states,
        group=ep_group,
        ...
    )

def low_latency_combine(expert_outputs, topk_ids, expert_ids, num_experts_per_ep):
    """
    Low-latency combine for EP
    将各GPU的expert输出合并回原始token顺序
    """
    # All-to-all combine
    combined = paddle.distributed.alltoall(
        tensor=expert_outputs,
        group=ep_group,
        ...
    )
```

### 4.6 EP vs TP 在 MoE 中的对比

| 特性 | EP (Expert Parallel) | TP (Tensor Parallel) |
|------|----------------------|----------------------|
| **并行维度** | Expert维度 | Attention/Linear维度 |
| **通信模式** | All-to-All | All-Reduce |
| **显存节省** | Expert卸载到多GPU | 需复制完整expert到每GPU |
| **通信量** | Token路由通信 | 激活值AllReduce |
| **适用场景** | MoE层 | Attention + FFN层 |

## 5. Data Parallel (DP) 详解

### 5.1 DP 核心思想

```
DP: 每个GPU运行完整的模型副本，处理不同的数据batch
- 无需通信（前向/后向各自独立）
- 仅在梯度同步时需要AllReduce
```

### 5.2 DP 配置

```python
# config.py
data_parallel_size: int = 1  # 默认关闭

# 使用场景
# world_size = TP × PP × DP
# 例如: 8 GPUs, TP=2, PP=2 → DP=2
```

### 5.3 DP 与 EP 的组合

```python
# 使用场景: 超大规模MoE模型
# world_size = TP × EP × DP
# 例如: 16 GPUs, EP=4, DP=2 → 4个EP组，每个组内2路DP
```

## 6. MoE 中的 Chunked MoE (EP显存优化)

**文件**: `moe.py`, line 749-800

```python
def forward_chunked_moe(self, hidden_states, gate, forward_meta):
    """
    Chunked MoE - EP模式下分块处理MoE以节省显存
    将token分成多个chunk，每个chunk独立做MoE计算
    """
    chunk_size = self.fd_config.parallel_config.chunked_moe_size

    # 分块处理
    num_chunks = (hidden_states.shape[0] + chunk_size - 1) // chunk_size

    all_outputs = []
    for i in range(num_chunks):
        chunk = hidden_states[i * chunk_size : (i + 1) * chunk_size]
        chunk_out = self.moe_method.forward(chunk, ...)
        all_outputs.append(chunk_out)

    return paddle.concat(all_outputs, axis=0)
```

## 7. Sequence Parallel (SP) 详解

### 7.1 SP 核心思想

```
标准: 每个GPU持有完整序列
SP: 序列被分割到多个GPU

例如: sequence_length=1024, TP=4
- 标准: 每GPU持有完整1024 tokens
- SP: 每GPU持有256 tokens，序列维度进行AllGather/AllReduce
```

### 7.2 SP 在 MoE 中的使用

**文件**: `moe.py`, line 730-731

```python
# MoE输出后需要进行sequence parallel的all-gather
if self.fd_config.parallel_config.use_sequence_parallel_moe:
    out = self.norm.allgather(out, group=sp_group)
```

## 8. 分布式策略组合示例

### 8.1 单节点 8 GPU 配置

```python
# TP=8 - 张量并行
fd_config.parallel_config.tensor_parallel_size = 8

# TP=4, EP=2 - 混合并行
fd_config.parallel_config.tensor_parallel_size = 4
fd_config.parallel_config.expert_parallel_size = 2

# TP=2, EP=4 - EP优先
fd_config.parallel_config.tensor_parallel_size = 2
fd_config.parallel_config.expert_parallel_size = 4
```

### 8.2 多节点配置

```python
# 16节点 × 8 GPU = 128 GPUs
# TP=8, EP=8, DP=2
world_size = 128

fd_config.parallel_config.tensor_parallel_size = 8
fd_config.parallel_config.expert_parallel_size = 8
fd_config.parallel_config.pipeline_parallel_size = 1
# data_parallel_size = world_size / (TP × EP × PP) = 2
```

## 9. 通信原语总结

| 原语 | 用途 | 参与者 |
|------|------|--------|
| **AllReduce** | TP并行结果汇总 | TP group内所有rank |
| **AllGather** | SP序列汇总 | SP group内所有rank |
| **AllToAll** | EP token路由分发/合并 | EP group内所有rank |
| **Broadcast** | 配置/权重同步 | master rank → others |

## 10. 关键代码位置总结

| 功能 | 文件 | 行号 |
|------|------|------|
| ParallelConfig | config.py | ~500 |
| tensor_model_parallel_all_reduce | communication.py | ~100 |
| DeepSeekV3MoE TP/EP配置 | deepseek_v3.py | 132-140 |
| MoE TP AllReduce | deepseek_v3.py | 193-205 |
| DeepEPEngine | ep.py | ~50 |
| EPPrefillRunner | ep.py | ~150 |
| EPDecoderRunner | ep.py | ~200 |
| low_latency_dispatch | ep.py | ~250 |
| low_latency_combine | ep.py | ~280 |
| forward_chunked_moe | moe.py | 749-800 |
| use_sequence_parallel_moe | moe.py | 730-731 |
