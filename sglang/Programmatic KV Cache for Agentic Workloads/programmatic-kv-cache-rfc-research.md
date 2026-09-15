# Programmatic KV Cache for Agentic Workloads：SGLang RFC #27574 调研报告

> 作者：@ishandhanani, @hzh0425  
> RFC Issue：https://github.com/sgl-project/sglang/issues/27574  
> Phase 1 实现：PR #29436 (first-class session identity)  
> 整理日期：2026-07-20

---

## 一、问题背景：为什么现有 KV Cache 机制在 Agentic 工作负载下失效

### 1.1 现有机制的本质假设

当前 LLM 推理引擎（包括 SGLang、vLLM）管理 KV cache 的方式是**请求局部的 LRU（request-local LRU）**：

- 缓存单元 = 一个 token block，以 block hash 索引
- 淘汰策略 = 引用计数 + LRU
- 可见范围 = 单个引擎实例内部

这个设计对于短会话、无状态请求是合理的，但 Agentic 工作负载打破了它的所有假设。

### 1.2 Agentic 工作负载的结构性特征

一个典型的 Agentic workload 具有如下结构：

```
System Prompt (共享前缀，所有 session 共享)
    └── Tool Schema (半共享前缀，同类 agent 共享)
            └── Session History (per-session，随 turn 增长)
                    └── Current Turn Input (每次请求唯一)
```

关键观察：
1. **Router/Orchestrator 具有全局视野**：它知道哪些 session 还活跃、哪些 token 范围是共享前缀、工具调用间隔是 10ms 还是 10 分钟
2. **Engine 只有局部视野**：它只看到 block hash 和引用计数，完全不知道上层语义
3. **这种信息不对称导致四类典型失效**：

| 失效场景 | 具体表现 |
|---------|---------|
| 跨 worker 前缀共享 | 另一个 worker 已有所需前缀的 KV，但当前 worker 不知道，重新计算 |
| 即将恢复的 session | Orchestrator 知道某个 session 10s 后会继续，Engine 却在内存压力下把它淘汰了 |
| Session 终止未释放 | Session 结束后 KV 应该尽快释放，但 Engine 不知道 session 语义，等 LRU 慢慢淘汰 |
| 工具调用间隙 | 长时间工具调用期间 KV 被淘汰，恢复时需要完整 prefill 重算 |

### 1.3 量化数据支撑（MiniMax M2.7 H100 实测）

在有内存压力的场景下（24.18 GB / 95,232 keys 被淘汰）：

- **未保留（unretained）的 cold-worker probe**：重新计算了 10,032 tokens
- **保留一小时（retained probe）**：从 L3 存储恢复 10,016 tokens，仅重新计算 16 tokens

这组数据展示了 hint 机制的潜在价值：在内存压力下，hint 引导的策略将重算开销降低了约 **630 倍**。

---

## 二、RFC 设计方案

### 2.1 核心设计哲学

RFC 提出的解法是一个**窄带、由 Router 发起的 hint 接口**，核心原则：

```
Orchestrator owns policy; Engine executes.
（编排者负责策略；引擎负责执行）
```
整个系统的正确性不应该依赖于hint被强制执行，而是提供策略，Engine判断如何执行；

四条设计原则：

1. **策略在外，执行在内**：Router 拥有全局 KV 索引、overlap/load 路由、HA 和轨迹感知；Engine 拥有 GPU 内存映射和调度权
2. **零 overhead when unused**：没有 hint 的请求行为与今天完全一致，不引入任何额外开销
3. **Hints are soft, bounded, safe to reject**：客户端发送的任何 hint 都不能无限 pin 内存或死锁 scheduler，Engine 可以 accept / clip / defer / reject
4. **Router 主导**：Router 有全局 KV 索引，Orchestrator 负责 hint 生成，这是信息流的自然方向

### 2.2 整体架构

```
         ┌─────────────────┐
         │   Router /      │
         │  Orchestrator   │  ← 全局视野：session活跃性、共享前缀、工具调用间隔
         └────────┬────────┘
                  │ KvHintEnvelope
                  ▼
         ┌─────────────────┐
         │  SGLang         │
         │  Scheduler      │  ← 接收 hint，决定如何执行（接受/裁剪/延迟/拒绝）
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │    HiCache      │  ← 分层缓存管理
         ├────────┬────────┤
         │ L1 GPU │ L2 Host│
         └────────┴────────┘
                  │ L3 interface
         ┌────────▼────────┐
         │  Mooncake /     │
         │  Shared Storage │  ← 跨 worker 的共享 KV 存储
         └─────────────────┘
```

### 2.3 Hint Taxonomy（提示分类）

RFC 定义了五类 hint，构成一个完整的缓存策略表达体系：

#### Share（共享）
**作用**：复用另一个 worker 或共享层（L3）上已有的前缀 KV  
**地位**：基础原语（foundational primitive），跨 worker KV 移动的核心能力  
**典型场景**：多 worker 部署中，将热门 System Prompt 的 KV 通过 L3 共享，避免每个 worker 独自计算

#### Prefetch（预取）
**作用**：在 KV 被请求之前，将其移入更热的层级（如 L2→L1 GPU）  
**依赖**：依赖 Share 的基础设施  
**典型场景**：Orchestrator 预测下一个 turn 即将到来，提前将 session KV 预热到 GPU

#### Demote（降级）
**作用**：将 KV 移入更冷的层级而非直接丢弃  
**典型场景**：长时间工具调用期间（如等待外部 API 响应），将 session KV 从 GPU 降级到 host memory 或 L3，而不是直接淘汰  
**与 Prefetch 配合**：Demote + Prefetch 形成完整的 pause-resume 生命周期管理

#### Pin（锁定）
**作用**：在有界 TTL 内保护高价值前缀不被淘汰  
**约束**：Pin 不代表永久 HBM 驻留，只是有时间上界的保护  
**POC 实现**：当前 POC 将 Pin 映射到 Mooncake L3 lease  
**典型场景**：多 agent 共享的 System Prompt + Tool Schema 前缀，在高内存压力下需要保护

POC = Proof of Concept，概念验证。具体含义是：一个功能可行但不生产就绪的原型实现。
#### Retain（保留偏向）
**作用**：不是 lease-protect，而是调整淘汰优先级（attach 相对优先级 + 可选时长），让引擎在内存压力下优先淘汰低优先级 KV  
**与 Pin 的区别**：Pin 是强保证（bounded TTL lease），Retain 是软偏向（eviction order bias）  
**典型场景**：Orchestrator 知道某些 session 明显比其他 session 更重要，通过 Retain 引导淘汰顺序


---

## 三、Phase 1：Session-based KV Cache

RFC 将实现分为两个阶段，Phase 1 专注于 session 身份建立。

### 3.1 Phase 1 包含的工作

| PR / TODO | 内容 |
|-----------|------|
| **PR #29436** | 顶层 `session_id` 字段，first-class session identity |
| **PR #27058** | `session_id` 标记进 KV blocks，实现 session 级 radix 寻址 |
| **TODO** | HiCache 层的 `SessionRadixCache` |

### 3.2 PR #29436 深度分析：First-Class Session Identity

这是 Phase 1 的第一个 merged 实现，核心贡献是将 session 的**身份（identity）**与**生命周期管理（lifecycle）**解耦。

#### 动机

旧机制要求调用方在发送请求前必须先调用 `/open_session`，这在以下场景下很笨拙：
- OpenAI 兼容接口（无状态调用，难以插入前置 open）
- 无状态 Agentic 框架（每个请求都经过独立的编排器实例）

#### 核心设计

引入 `session_id: Optional[str]`——一个稳定的标量身份令牌：

```python
@dataclass
class GenerateReqInput:
    session_id: Optional[str] = None
    """Stable identity shared by requests in the same session.
    Unlike session_params, this does not alter or reconstruct the prompt."""
```

关键语义约束：
```python
# session_id 和 session_params 互斥
if self.session_id is not None and self.session_params is not None:
    raise ValueError("session_id and session_params are mutually exclusive")
```

#### 改动层次（全栈透传）

```
Proto/gRPC
    ↓ optional string session_id
Entry Points (EngineBase, Engine, http_server_engine, runtime_endpoint)
    ↓ session_id: Optional[str] = None
OpenAI 兼容层 (CompletionRequest, ChatCompletionRequest, ResponsesRequest)
    ↓ session_id: Optional[str] = None
io_struct.py (GenerateReqInput, TokenizedGenerateReqInput)
    ↓
Scheduler ← 核心简化（-43/+9 行）
    ↓ radix_native_session = recv_req.session_id
SessionRadixCache ← implicit open, rename release_session→release_radix_session
```

#### Scheduler 层的简化

这是最有价值的改动。旧代码中 `radix_native_session` 需要从 `session_params` 派生，并且需要一个验证块来拒绝 `session_params` 中 radix-native 不支持的字段（`rid`/`offset`/`replace`）。

新代码：
```python
# scheduler.py — 直接读取顶层字段
radix_native_session = recv_req.session_id
```

旧的验证块整体删除——因为 `session_id` 和 `session_params` 的互斥约束在更上层已经保证。

**代码质量结论**：删除代码比新增代码更有价值。这次改动净减少 51 行（scheduler -43，session_radix_cache -8）。

#### 使用流程对比

旧流程（session_params 路径，仍然保留）：
```
1. POST /open_session  → 显式注册
2. POST /generate      → session_params.id → radix_native_session
3. POST /close_session → 释放 KV
```

新流程（session_id 路径）：
```
1. POST /generate (session_id="xyz") → 自动 implicit open
2. POST /generate (session_id="xyz") → radix cache 复用
3. POST /close_session               → tree_cache.release_radix_session("xyz")
```

对于 OpenAI SDK 用户：
```python
# 无需任何预注册，直接在请求体带 session_id 即可
response = client.completions.create(
    model="llama-3.2-1b",
    prompt="...",
    extra_body={"session_id": "my-agent-session-001"}
)
```

#### 实测验证

对 Llama 3.2 1B 的测试：
- 无 `/open_session` 直接发 `session_id` 请求 → 成功复用 **11 个缓存 token**
- `/close_session` 后 KV 内存正常回收

---

## 四、Phase 2：Router-initiated Hint API

### 4.1 KvHintEnvelope

Phase 2 的核心载体是 `KvHintEnvelope`，一个由 Orchestrator 通过请求预处理传入 Engine 的信封结构。

当前 POC 已实现的 shape：
```python
KvHintEnvelope {
    retention: [
        {
            prefix_tokens: [...],  # 需要保留的 token 范围
            ttl_seconds: 3600      # 有界 TTL
        }
    ]
}
```

这是第一个被实际测试的 schema shape，RFC 明确说明这不是最终分类。

### 4.2 L3 Pin 的 POC 实现

POC 将 Pin hint 映射到 Mooncake L3 的有界 TTL lease：

```
Router 发现 session A 有高价值前缀
    → 在请求中携带 KvHintEnvelope{retention: [{prefix, ttl=3600}]}
    → Scheduler 接收并传入 HiCache
    → HiCache 向 Mooncake L3 申请 committed page group + TTL lease
    → 内存压力下：此 page group 受保护，不被淘汰
    → TTL 到期或 Orchestrator 发送 demote hint：释放 lease
```

### 4.3 Phase 2 TODO 清单

- [ ] Productionize admission（入场控制）
- [ ] Telemetry（可观测性）
- [ ] Expiry 管理（TTL 到期处理）
- [ ] Namespace（多租户隔离）
- [ ] Version-skew handling（Router 和 Engine 版本不一致时的容错）
- [ ] 在 request-time L3 restoration 测量完成前，不添加 proactive Prefetch/Demote 执行 API

---

## 五、与相关工作的关系

### 5.1 SGLang 内部三层架构

这个 RFC 是三个相互关联的 initiative 的一部分：

```
#21846 (Distributed KVCache System)
    → 机制层：HiCache 分层存储、PD incremental transfer、storage prefetch 接口

RFC #27574 (Programmatic KV Cache)
    → 策略层：骑在 #21846 的基础设施上，决定什么时候使用它

#24656 (Agent-Aware KV Cache Phase 1)
    → 请求作用域的信封：agent_hints, cache_ttl_ms/reuse_hint，映射到 Pin/Retain
      （提供了 client-driven 的 in-band 路径，是 RFC #27574 out-of-band 路径的补充）
```

### 5.2 与业界其他方案的对比

| 系统 | 方案 | 与本 RFC 的关系 |
|------|------|----------------|
| **vLLM #37003** | Context-Aware KV-Cache Retention API，优先级淘汰 | 对应 RFC 中的 Retain hint，思路相似 |
| **vLLM #37168** | 长期 agent 的 active coordination + two-zone scheduling | 更激进，直接在引擎内部 schedule，RFC 选择了更保守的 hint 路径 |
| **TensorRT-LLM** | `KvCacheRetentionConfig`：token range retention with priority 0–100 + duration_ms | 最接近 RFC Retain hint 的实现，已生产 |
| **Dynamo #6213/#6571** | Anthropic-style `cache_control` normalization | in-band，client 驱动，RFC 是 out-of-band，Router 驱动 |
| **Dynamo #8789/#9140** | `agent_context` / ATIF | 更完整的 agentic inference 框架，RFC 是其中 KV 策略的 SGLang 实现 |
| **Mooncake #2835** | TTL-bounded `retain_groups`，自动到期，future-member 继承，有界 group admission | RFC 的 L3 Pin 直接基于此实现 |

---

## 六、关键研究文献

| 论文 | 核心贡献 | 与本 RFC 的关系 |
|------|---------|----------------|
| **KVCache in the Wild** (arXiv 2506.02634) | 阿里巴巴生产 trace 分析，量化 KV cache miss 的成本分布 | 提供了 agentic workload 失效的实证依据 |
| **Continuum** (arXiv 2511.02230) | 多 turn agent 的 KV cache TTL 机制 | 直接启发了 Pin/Retain hint 的 TTL 设计 |
| **KVFlow** (arXiv 2507.07400) | workflow-aware prefix caching | 与 RFC 的 workflow-level hint 思路一致 |
| **MARCONI** (arXiv 2411.19379) | hybrid LLM 的 prefix caching | 跨模型/跨层的 KV 共享，是 Share hint 的理论基础 |
| **Tail-Optimized Caching** (arXiv 2510.15152) | LLM 推理中针对尾延迟优化的缓存策略 | 提供了 Retain hint 的优先级理论基础 |

---

## 七、总结与展望

### RFC 的核心价值主张

1. **信息对称**：让 Engine 能感知 Router/Orchestrator 的语义（session 活跃性、工具调用间隔、重要性），而不是盲目 LRU
2. **控制反转**：策略在外（Orchestrator），执行在内（Engine），符合系统设计的关注点分离原则
3. **渐进式**：Phase 1（session identity）→ Phase 2（hint API）→ 未来（proactive Prefetch/Demote），每个阶段都独立可用

### Phase 1 已完成的价值（PR #29436）

对 Agentic 框架开发者而言，PR #29436 已经带来了直接可用的改进：
- OpenAI SDK 用户可以零侵入地获得 session-level KV cache 复用，只需在请求 body 里加一个 `session_id` 字符串
- 代码反而更简洁（scheduler 净减少 43 行）
- 旧用户（使用 `session_params`）零迁移成本

### 尚待解决的核心问题

1. **L3 Pin 的生产就绪**：admission control、telemetry、expiry、namespace、version-skew 都还是 TODO
2. **Proactive Prefetch/Demote**：需要先测量 request-time L3 restoration 的延迟，才能判断 proactive 预取的 trade-off 是否值得
3. **KvHintEnvelope schema 稳定化**：当前的 `{retention: [{prefix_tokens, ttl_seconds}]}` 是第一个 shape，还需要验证其他 hint 类型的 schema 设计
4. **HiCache 层的 SessionRadixCache**：Phase 1 的最后一块 TODO

### 设计上值得借鉴的思路

> **Hints are soft, bounded, and safe to reject.**

这个设计原则值得在任何"外部策略驱动内部执行"的系统设计中参考。它将 hint 的语义定义为**意图表达**而非**指令**，允许 Engine 在保证安全的前提下尽力执行，从而在不改变内部一致性保证的前提下引入外部优化机会。
