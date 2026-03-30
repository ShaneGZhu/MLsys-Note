# DeepseekV3.2 模型推理层次

## 1. 核心文件位置

| 类别 | 文件路径 |
|------|---------|
| **模型定义** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/models/deepseek_v3.py` |
| **MLA Attention后端** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/layers/attention/mla_attention_backend.py` |
| **MoE层** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/layers/moe/moe.py` |
| **EP实现** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/fastdeploy/model_executor/layers/moe/ep.py` |
| **MLA CUDA算子** | `/root/paddlejob/workspace/env_run/output/shengguang/FastDeploy/custom_ops/gpu_ops/multi_head_latent_attention.cu` |

## 2. 模型类层次结构

**文件**: `deepseek_v3.py`

```
DeepseekV32ForCausalLM (主入口)
    │
    └── DeepSeekV3Model
            │
            ├── DeepSeekV3DecoderLayer (×num_layers)
            │       │
            │       ├── DeepseekV3MLAAttention (Attention)
            │       │       │
            │       │       └── MLAAttentionBackend
            │       │
            │       └── DeepSeekV3MoE (MoE)
            │               │
            │               └── FusedMoE
            │
            ├── VocabParallelEmbedding (Embedding)
            │
            └── RMSNorm / Linear (Output)
```

## 3. DeepseekV3ForCausalLM 主入口

**文件**: `deepseek_v3.py`

```python
# line ~600
class DeepseekV32ForCausalLM(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.model = DeepSeekV3Model(fd_config)
        self.lm_head = ParallelLMHead(...)

    def forward(self, input_ids, ...):
        hidden_states = self.model(input_ids, ...)
        logits = self.lm_head(hidden_states)
        return logits
```

## 4. DeepSeekV3Model 前向流程

**文件**: `deepseek_v3.py`, line 350-450

```python
class DeepSeekV3Model(nn.Layer):
    def forward(self, input_ids, forward_meta, ...):
        # 1. Embedding
        hidden_states = self.embed_tokens(input_ids)

        # 2. Decoder Layers
        for layer in self.layers:
            # Attention + MoE
            hidden_states = layer(hidden_states, forward_meta)

        # 3. Final Norm
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

## 5. Attention 实现 - MLA (Multi-Head Latent Attention)

### 5.1 MLA 核心思想

DeepSeek V3 采用 Multi-Head Latent Attention (MLA)，通过低秩压缩减少 KV Cache 存储：

```
标准MHA:
- Key: [batch, seq, num_heads, head_dim]
- Value: [batch, seq, num_heads, head_dim]
- KV Cache: O(batch * seq * num_heads * head_dim)

MLA:
- 压缩的KV: [batch, seq, kv_lora_rank] << 标准MHA
- 解压后: [batch, seq, num_heads, head_dim]
```

### 5.2 DeepseekV3MLAAttention 类

**文件**: `deepseek_v3.py`, line 208-320

```python
class DeepseekV3MLAAttention(nn.Layer):
    def __init__(self, fd_config, layer_id, prefix=""):
        # MLA 配置
        self.q_lora_rank = fd_config.model_config.q_lora_rank        # Q压缩rank
        self.kv_lora_rank = fd_config.model_config.kv_lora_rank        # KV压缩rank
        self.qk_nope_head_dim = fd_config.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = fd_config.model_config.qk_rope_head_dim
        self.v_head_dim = fd_config.model_config.v_head_dim

        # 投影层
        self.qkv_a_proj_with_mqa = MergedReplicatedLinear(...)
        self.q_a_layernorm = RMSNorm(...)
        self.q_b_proj = ColumnParallelLinear(...)
        self.kv_a_proj_with_mqa = MergedReplicatedLinear(...)
        self.k_b_proj = ColumnParallelLinear(...)
        self.o_proj = RowParallelLinear(...)

    def forward(self, hidden_states, forward_meta):
        # 1. Q投影和压缩
        q = self.q_a_layernorm(self.q_b_proj(self.qkv_a_proj_with_mqa(hidden_states)[0]))

        # 2. KV投影和压缩
        # ...

        # 3. 调用MLA Attention后端
        fmha_out = self.attn_backend.forward_extend(...)  # Prefill
        # 或
        fmha_out = self.attn_backend.forward_decode(...)   # Decode
```

### 5.3 MLA Attention Backend

**文件**: `mla_attention_backend.py`

```python
class MLAAttentionBackend(AttentionBackend):
    def forward_extend(self, q, k, v, ..., forward_meta):
        """Prefill/Extend阶段"""
        # 1. 写入KV Cache
        prefill_mla_write_cache(compressed_kv, k_pe, latent_cache, ...)

        # 2. Flash Attention计算
        fmha_out = flash_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)

        return fmha_out

    def forward_decode(self, q, k, v, ..., forward_meta):
        """Decode阶段"""
        # 1. 写入KV Cache (单token)
        decode_mla_write_cache(compressed_kv, k_pe, latent_cache, ...)

        # 2. 从Latent Cache读取并计算Attention
        fmha_out = multi_head_latent_attention(q, latent_cache, latent_cache, ...)

        return fmha_out
```

### 5.4 MLA CUDA 算子

**文件**: `custom_ops/gpu_ops/multi_head_latent_attention.cu`

```cuda
// 核心Kernel
__global__ void BatchMLAWithPagedKVCacheKernel(...)  // Prefill批量处理
__global__ void DecodeMLAAttentionKernel(...)         // Decode单token处理

// Cache读写
__global__ void prefill_mla_write_cache(...)          // Prefill写cache
__global__ void decode_mla_write_cache(...)           // Decode写cache
```

## 6. MoE 实现 (Mixture of Experts)

### 6.1 DeepSeekV3MoE 类

**文件**: `deepseek_v3.py`, line 127-205

```python
class DeepSeekV3MoE(nn.Layer):
    def __init__(self, fd_config, layer_id, prefix):
        # Tensor Parallel配置
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.ep_size = fd_config.parallel_config.expert_parallel_size

        # EP模式下，MoE的TP设为1
        if self.ep_size > 1:
            self.tp_size = 1

        # Gate网络
        self.gate = ReplicatedLinear(
            input_size=hidden_size,
            output_size=n_routed_experts,
            ...
        )

        # Expert并行: 每个rank有完整的expert集合
        self.experts = FusedMoE(
            num_experts=n_routed_experts,
            top_k=num_experts_per_tok,
            ...
        )

        # 共享Expert
        self.shared_experts = DeepSeekV3MLP(...)

    def forward(self, hidden_states, forward_meta):
        # 1. 共享Expert计算
        shared_experts_out = self.shared_experts(hidden_states)

        # 2. MoE Expert计算
        moe_out = self.experts(hidden_states, self.gate, forward_meta)

        # 3. 合并结果
        moe_out = moe_out + shared_experts_out

        # 4. TP AllReduce (如果启用)
        if self.tp_size > 1:
            moe_out = tensor_model_parallel_all_reduce(moe_out)

        return moe_out
```

### 6.2 FusedMoE 类

**文件**: `moe.py`, line 129-300

```python
class FusedMoE(nn.Layer):
    def forward(self, hidden_states, gate, forward_meta):
        # 1. Gate计算
        gating_output = gate(hidden_states)

        # 2. TopK Expert选择
        scores, topk_values, topk_idx = get_moe_scores(
            gating_output,
            n_group=self.n_group,
            topk_group=self.topk_group,
            top_k=self.top_k,
            renormalize=self.renormalize,
            ...
        )

        # 3. Expert计算
        moe_out = self.moe_method.forward(
            hidden_states,
            topk_weights=topk_values,
            topk_ids=topk_idx,
            ...
        )

        return moe_out
```

### 6.3 MoE CUDA 算子

**文件**: `custom_ops/gpu_ops/moe/`

| 文件 | 功能 |
|------|------|
| `moe_topk_select.cu` | TopK Expert选择 |
| `fused_moe.cu` | 融合MoE计算 |
| `moe_dispatch.cu` | Expert dispatch |
| `moe_reduce.cu` | Expert reduce |

```cuda
// moe_topk_select.cu
__global__ void moe_topk_select_kernel(
    const float* gating_output,    // [num_tokens, num_experts]
    float* scores,                 // [num_tokens, top_k]
    int* topk_idx,                 // [num_tokens, top_k]
    ...
)
```

## 7. Prefill/Decode 阶段区分

### 7.1 Prefill 阶段

```python
# deepseek_v3.py, line 370-404
def forward(self, forward_meta, ...):
    need_do_prefill = forward_meta.max_len_tensor_cpu[1] > 0  # enc_len > 0

    if need_do_prefill:
        # Prefill计算
        fmha_out_prefill = self.mla_attn(
            q=query, k=key, v=value, ...
        )
```

### 7.2 Decode 阶段

```python
# deepseek_v3.py, line 406-440
def forward(self, forward_meta, ...):
    need_do_decode = forward_meta.max_len_tensor_cpu[2] > 0  # dec_len > 0

    if need_do_decode:
        # Decode计算 - q是单个token
        fmha_out_decode = self.mla_attn(
            q=q_input, k=None, v=None, ...
        )
```

## 8. 完整推理流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeepseekV32ForCausalLM                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Embedding Layer                                              │
│     hidden_states = embed_tokens(input_ids)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Decoder Layers (for each layer)                             │
│     ┌─────────────────────────────────────────────────────────┐│
│     │  a. MLA Attention                                       ││
│     │     ├── qkv_a_proj (Q压缩)                               ││
│     │     ├── kv_a_proj (KV压缩)                              ││
│     │     ├── q_a_layernorm + q_b_proj (Q解压)                ││
│     │     ├── k_b_proj (K解压)                                ││
│     │     └── attn_backend.forward_extend/decode()             ││
│     │            │                                             ││
│     │            ├── prefill_mla_write_cache / decode_mla_...  ││
│     │            └── flash_attn / multi_head_latent_attn       ││
│     │                                                         ││
│     │  b. MoE Layer                                           ││
│     │     ├── gate(hidden_states) → gating_output              ││
│     │     ├── get_moe_scores() → topk_idx, scores             ││
│     │     ├── experts(hidden_states, topk_idx)                ││
│     │     └── shared_experts + moe_out                        ││
│     └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Final Norm + LM Head                                         │
│     hidden_states = norm(hidden_states)                         │
│     logits = lm_head(hidden_states)                            │
└─────────────────────────────────────────────────────────────────┘
```

## 9. 关键代码位置总结

| 功能 | 文件 | 行号 |
|------|------|------|
| DeepseekV32ForCausalLM | deepseek_v3.py | ~600 |
| DeepSeekV3Model.forward | deepseek_v3.py | 350-450 |
| DeepseekV3MLAAttention | deepseek_v3.py | 208-320 |
| MLA Backend forward_extend | mla_attention_backend.py | 392-444 |
| MLA Backend forward_decode | mla_attention_backend.py | 446-532 |
| DeepSeekV3MoE | deepseek_v3.py | 127-205 |
| FusedMoE | moe.py | 129-300 |
| get_moe_scores | moe.py | 81-126 |
| MLA CUDA Kernel | multi_head_latent_attention.cu | - |
| MoE TopK CUDA | moe_topk_select.cu | - |
