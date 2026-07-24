# Dynamo KV Cache Router 技术调研

> 基于 NVIDIA Dynamo `main` 分支（对应稳定版 ****v1.2.1****，2026-06-13 发布），源码引用格式为 `文件路径 :: 函数名` 或 `文件路径`（Dynamo 是 Rust 主体 + Python CLI 的复合仓库，行号版本敏感故不写死）。所有官方文档引用均给出 URL；结论若源自源码则给出 crate + 文件路径。
> 
> ****本文的定位****：本仓库 `dynamo/` 目录下已经有两篇高深度的姊妹调研——`RL-Router调度策略.md`（PaddleRL InferRouter 三个策略，源码级）和 `verl KVC Aware Rollout Router 技术调研.md`（verl `#6712 + #6940` PR，源码级）。同目录旧文档 `Dynamo-KV-Cache-Router-机制详解.md` 在****决策分支细节、变量来源可追溯性、参数默认值、源码位置****上明显浅于那两篇。****本文是完全重写的替代版本****，写作时对齐两篇姊妹文档的深度和节奏；旧文档保留供对照，但其中若干技术结论已由本文纠正（详见第 13 节"常见误区"和第 12 节"横向对比"里的修正条目）。
> 
> ****推荐阅读顺序****：如果只关心"Dynamo 是怎么选 worker 的"，读 §3 → §6 → §8；如果关心"Dynamo 怎么知道谁有 KV cache"，读 §4 → §5；如果关心与 verl / RL-Router / SGLang 的差异，读 §12。

---

## 目录

1. ﻿[Dynamo KV Router 到底是什么](#1-dynamo-kv-router-到底是什么)﻿
    
2. ﻿[系统边界：Router、Indexer、KVBM、HiCache、NIXL、Planner、Event Plane、Discovery](#2-系统边界)﻿
    
3. ﻿[一条 KV-aware 请求的完整路由流程](#3-一条-kv-aware-请求的完整路由流程)﻿
    
4. ﻿[Indexer 与 Overlap Scores（信息来源之一：本地历史索引）](#4-indexer-与-overlap-scoresinfo-source-1)﻿
    
5. ﻿[KV Events 协议与传输（信息来源之二：worker 事件流）](#5-kv-events-协议与传输info-source-2)﻿
    
6. ﻿[Cost Function 逐项拆解](#6-cost-function-逐项拆解)﻿
    
7. ﻿[Filter 与 Scoring 的边界](#7-filter-与-scoring-的边界)﻿
    
8. ﻿[决策分支全景（源码级 if-else）](#8-决策分支全景)﻿
    
9. ﻿[P/D 分离下的路由分工](#9-pd-分离下的路由分工)﻿
    
10. ﻿[系统级能力：Discovery、Multi-Router、冷启动、Migration、Cancellation、Multimodal](#10-系统级能力)﻿
    
11. ﻿[Observability：Router 全部 Prometheus 指标](#11-observability)﻿
    
12. ﻿[与 verl / RL-Router / SGLang Gateway 的横向对比](#12-横向对比)﻿
    
13. ﻿[常见误区（含对旧文档结论的修正）](#13-常见误区)﻿
    
14. ﻿[调参与排障建议](#14-调参与排障建议)﻿
    
15. ﻿[参考资料与源码索引](#15-参考资料与源码索引)﻿
    

---

## 1. Dynamo KV Router 到底是什么

一句话：****Dynamo KV Router 是一个跨 worker 的 prefix cache 亲和调度器****——它同时权衡两件事：

1. 这次请求的 prompt 前缀，在哪些 worker 上已经有可复用的 KV cache（****cache 收益****）；
    
2. 这些 worker 现在有多忙，请求打过去会不会因排队/资源不足导致端到端延迟反而更差（****负载成本****）。
    

Router 的最终决策是把这两件事融进一个 cost function，选出 cost 最低的 worker（详见 §6）。它****不直接读 GPU 显存里的 KV tensor****，而是维护一份基于 worker 主动上报事件的"KV block 元信息全局视图"，这份视图和"每个 worker 有多忙"的负载视图一起，作为决策的输入。

### 1.1 Router mode 全景

Dynamo 的 Router 支持多种 routing mode，KV-aware 只是其中一种。CLI 参数 `--router-mode` 的选项完整列表（`components/src/dynamo/common/configuration/groups/router_args.py`）：

|   |   |   |
|---|---|---|
|`--router-mode`|说明|用途|
|`round-robin`|顺序轮询，****Dynamo 默认值****|忽略负载与 cache，最简单|
|`random`|均匀随机|无状态基线|
|`power-of-two`|随机取两个 worker 挑负载较低者|经典 shortest-queue 近似|
|`least-loaded`|挑当前 potential decode load 最低者|纯负载均衡|
|`direct`|客户端通过 header 指定 worker|外部 EPP（External Prefill Planner）先决策的场景|
|`device-aware-weighted`|按 encoder GPU/CPU 加权（`normalized_load = total_inflight / (instance_count × throughput_weight)`）|编码器异构场景（`DYN_ENCODER_CUDA_TO_CPU_RATIO=8` 默认）|
|`****kv****`|****本文所讲的 KV-aware Router****|长上下文 / 多轮 prefill 复用|

  

****关键背景****：Dynamo 默认不启用 KV Router——用户必须显式 `--router-mode kv`（或用 `--load-aware` preset 隐式开启，见 §7.5）。这一点和 SGLang Model Gateway 默认就是 `cache_aware` 是明显不同的策略取向。

> 来源：`components/src/dynamo/common/configuration/groups/router_args.py`；`docs/components/router/router-concepts.md`。

### 1.2 与姊妹方案的一句话对比

|   |   |   |   |   |
|---|---|---|---|---|
|方案|决策依据|信息实时性|引擎依赖|合入状态|
|****Dynamo KV Router****（本文）|事件驱动的全局 KV block 索引 + 4 层 tier credit + 负载|worker 事件推送 + 少量 HTTP polling|vLLM / TRT-LLM / SGLang（tier-aware 需 SGLang ≥ 0.5.11）|生产可用（v1.2.1）|
|verl`KVCAwareBalancer`|vLLM ZMQ KV events 反向索引 + Prometheus 负载|事件推送 + 5s polling|仅 vLLM|`#6712 + #6940` 未合并|
|RL-Router`pd_cache_aware`|本地 radix（历史反推）+ 归一负载|本地 counter，sub-ms|与后端解耦（默认 vLLM 格式指标）|生产运行中|
|SGLang Gateway`cache_aware`|Gateway 侧近似前缀树（请求文本历史）|每请求更新，强实时|任意后端|SGLang 默认策略|

  

后续章节会在每个关键决策点给出细粒度的对比小节。

---

## 2. 系统边界

Dynamo 文档里同时出现的名词较多（Router / KVIndexer / KVBM / HiCache / NIXL / Event Plane / Planner / Discovery），容易混淆。先给一张边界清单：

|   |   |   |
|---|---|---|
|组件|负责什么|不负责什么|
|****KV Router / KvScheduler****|选 worker，基于 cache overlap + 负载打分|不搬 KV tensor、不管理 GPU 分配器|
|****Indexer****（`KvIndexer` / `ConcurrentRadixTreeCompressed` / `Remote` / `None`）|维护"哪个 block hash 被哪个 worker 缓存"的全局前缀树|不做打分，不推事件（只消费）|
|****ActiveSequences / PrefillLoadTracker / BlockTracker****|维护每 worker 的活跃请求、prefill 排队 tokens、decode blocks|不含真实 GPU 内存 pointer|
|****Event Plane****|传输 worker 上报的`RouterEvent`（Stored / Removed / Cleared）|不传输真实 KV tensor 数据|
|****KVBM****（KV Block Manager）|Dynamo 自研的多层 KV block 生命周期管理（G1/G2/G3/G4 tier）|与 HiCache****互斥关系****（见下）|
|****HiCache****|SGLang 自带的多层 KV cache（GPU / host-pinned / external）|由 SGLang runtime 而不是 Dynamo 管理|
|****NIXL****|跨 GPU / RDMA / NVLink / GDS 的高速数据传输|不决定路由|
|****Planner****|autoscaling 控制器（按 TTFT/ITL SLA 扩缩 prefill/decode 副本数）|不决定单请求路由|
|****Discovery****（etcd / K8s CRD）|worker 注册、lease、健康探测|不做请求转发|

  

### 2.1 KVBM 与 HiCache 的关系（澄清一个常见误解）

官方原文（`docs/backends/sglang/sglang-hicache.md`）：

> "KVBM is described as Dynamo's own block manager, ****an alternative to HiCache****."

也就是说，****KVBM 与 HiCache 是并列的两条 tier 管理路线，不是叠加关系****。SGLang 场景下走 HiCache（SGLang 自带的分层 KV cache），Dynamo Router 通过消费 HiCache 上报的 KV events 感知分层状态；vLLM / TRT-LLM 场景下则可能走 KVBM。这一点在 Feature Matrix 里也能看出来：截至 v1.2.0，SGLang 在 "KV Block Manager" 一列是 🚧，因为 SGLang 用户不走 KVBM connector——不是能力缺失，是路线不同。

> 来源：`docs/backends/sglang/sglang-hicache.md`；`docs/components/kvbm/README.md`；`docs.nvidia.com/dynamo/resources/feature-matrix`。

### 2.2 Planner 与 Router 的关系

Planner 是 ****autoscaling 控制器****，不是请求调度器。官方文档明确写道（`docs/design-docs/planner-design.md`）：

> "Dynamo's autoscaling controller."  
> "The Planner operates on aggregate signals (traffic shape, FPM, queue depths)."

Planner 与 Router 是****并列的两个组件****——层级部署里可以有一个控制平面 DGD 同时挂着 `Frontend`、`GlobalRouter`、`GlobalPlanner`。Planner 消费 Router 侧的 metrics（`throughput_metrics_source` 可以是 `frontend` 或 `router`）来判断是否要扩缩副本，但****不会替代 Router 做单请求的 worker 选择****。SLA 目标（TTFT `ttft_ms=500ms` 默认、ITL `itl_ms=50ms` 默认）只影响 Planner 的扩缩容决策，Router 本身****不直接感知 SLA****（这与 verl 的 RL-aware scheduling 讨论方向不同，见 §12）。

> 来源：`docs/design-docs/planner-design.md`；`docs.nvidia.com/dynamo/components/planner/planner-guide`。

### 2.3 一张边界简图

                    ┌──────────────────────────────────┐

Client ───► Frontend│  tokenize / assemble metadata    │

                    └──────────────┬───────────────────┘

                                   ▼

              ┌─────────────────────────────────────────┐

              │   KvRouter (Rust)                        │

              │  ┌───────────────────────────────────┐  │

              │  │ Indexer (RadixTree per tier)       │  │

              │  │  ← consumes KV events              │  │

              │  └───────────────────────────────────┘  │

              │  ┌───────────────────────────────────┐  │

              │  │ ActiveSequences (per worker)       │  │

              │  │  prefill_tracker / block_tracker   │  │

              │  └───────────────────────────────────┘  │

              │  ┌───────────────────────────────────┐  │

              │  │ Scheduler + DefaultWorkerSelector  │  │

              │  │  cost = w_prefill · adj + decode   │  │

              │  └───────────────────────────────────┘  │

              └──────────┬──────────────────────────────┘

                         │  selected worker

                         ▼

              ┌────────────┬──────────────┬────────────┐

              │ vLLM / SGLang / TRT-LLM Workers        │

              │  · execute prefill / decode            │

              │  · publish KV events (Stored/Removed)  │

              │  · publish HTTP /metrics               │

              │  · (P/D) NIXL transfer of KV blocks    │

              └────────────────────────────────────────┘

                         │

                         ▼

                    Event Plane

                (ZMQ default, NATS optional)

---

## 3. 一条 KV-aware 请求的完整路由流程

以 `--router-mode kv` 为例，一次请求从 Frontend 收到到 Router 返回选中 worker 的完整时序：

sequenceDiagram

  participant C as Client

  participant F as Frontend

  participant R as KvRouter

  participant I as Indexer (radix tree)

  participant S as SharedKvCache (optional)

  participant A as ActiveSequences

  participant W as Selected Worker

  participant E as Event Plane

  C->>F: OpenAI-compatible request

  F->>F: tokenize + normalize

  F->>R: find_best_match_details_with_policy_class(tokens, hints)

  R->>R: compute_block_hash_for_seq(tokens)  ← XXH3-64 per block

  par parallel query

    R->>I: query_tiered_matches(hashes)

    I-->>R: {worker → (device/host/disk overlap blocks)}

  and

    R->>S: query shared pool (if --shared-cache-type=hicache)

    S-->>R: {shared_overlap_blocks}

  end

  R->>R: narrow_allowed_by_lora(candidates)

  R->>A: for each worker: read active_prefill_tokens / decode_blocks

  R->>R: worker_logit(each candidate) → argmin or softmax(T)

  R-->>F: FindBestMatchOutcome::Routed { worker, overlap, cached_tokens }

  F->>W: forward request

  W-->>C: stream tokens

  W->>E: KV events (Stored / Removed / Cleared)

  E->>I: apply_event (update radix tree)

7 个关键步骤：

1. ****Tokenize****：Frontend 对请求做 tokenize（因为 KV cache 是按 token 序列组织的，raw text 无法用于命中判定）。
    
2. ****Block hashing****：Router 用 `compute_block_hash_for_seq(...)`（`lib/llm/src/kv_router.rs` tracing span `"kv_router.compute_block_hashes"`）把 token 序列切成 block、逐 block 算 `LocalBlockHash`（XXH3-64；详见 §4.3）。
    
3. ****并行查询 Indexer + SharedKvCache****：`query_tiered_matches(...)` 并发问两个数据源——本地 Indexer 返回 device/host/disk 三个 tier 的 overlap blocks，可选的 shared pool（Mooncake 等）返回 shared_overlap_blocks。
    
4. ****LoRA filter****：`narrow_allowed_by_lora(...)` 收窄候选 worker 集合——****LoRA-pinned worker 若不在原候选集合中会被丢弃，不会扩大集合****（这是硬约束，不是打分惩罚）。
    
5. ****读取每 worker 负载****：从 `ActiveSequences`（每 worker 一份，见 §6.2）取 `active_prefill_tokens` 与 `potential_decode_blocks()`。
    
6. ****打分与选择****：对每个候选算 `worker_logit`（详见 §6），temperature=0 时 argmin + reservoir tie-break，temperature>0 时 softmax 采样。
    
7. ****返回决策 + 事件回环****：Router 把 `FindBestMatchOutcome::Routed { worker, overlap_blocks, cached_tokens, ... }` 交给 Frontend；请求转发到 worker 后，worker 在 prefill/decode 过程中通过 event plane 反馈 KV block 状态变化，Indexer 相应更新。
    

> 主入口源码：`lib/llm/src/kv_router.rs :: KvRouter::find_best_match_details_with_policy_class`。所有 `find_best_match*` 方法最终都调用它。

---

## 4. Indexer 与 Overlap Scores（信息来源之一：本地历史索引）

Dynamo KV Router 的 cache 感知信号，****主要来自 Indexer 维护的全局前缀树****——它订阅 worker 发出的 KV 事件，把"哪个 block hash 被哪个 worker 缓存"实时记下来。这一层是打分公式里 `device/host/disk_overlap_blocks` 的来源。

### 4.1 顶层：`Indexer` 是一个 enum，不是 struct

一个容易踩的坑：`Indexer` 在 Dynamo 源码里****是 enum，四个变体****（`lib/llm/src/kv_router/indexer/mod.rs`）：

|   |   |
|---|---|
|变体|用途|
|`KvIndexer`|单线程 actor，`std::thread` 持有 `RadixTree`，无锁；通过 `tokio::sync::mpsc` 通信|
|`Concurrent`|底层是`ConcurrentRadixTreeCompressed`，包了一层 `ThreadPoolIndexer`；默认使用（`--router-event-threads=4`）|
|`Remote`|Router 不自己维护 index，而是通过 RPC 向独立的 indexer 服务查询（`--use-remote-indexer`）|
|`None`|Indexer 关闭；用于 basic modes（round-robin 等）|

  

每个变体持有：一个 primary indexer（`Device` tier）、一组 `LowerTierIndexers`（host / disk tier）、可选 `SideIndexer`（"approximate mode" 下的自预测覆盖，`--no-router-kv-events` 才启用）、以及 `primary_records_routing_decisions: bool`。

﻿`apply_event` 分发逻辑：

- ﻿`Cleared` 事件广播到 primary + 所有 lower tier；
    
- GPU tier 事件走 primary；
    
- 非 GPU 事件走 `lower_tier.get_or_create(event.storage_tier)`；
    
- ﻿`Remote` / `None` 忽略事件。
    

> 来源：`lib/llm/src/kv_router/indexer/mod.rs`。

### 4.2 KvIndexer actor 模型：为什么没有原子计数器

这里是一个****与 verl / RL-Router 截然不同的设计选择****：Dynamo 内部****没有集中的原子 in-flight 计数器****（verl / RL-Router 都用了 `AtomicI64` 或 `map + Mutex`）。原因是 Dynamo 走的是"actor + 消息 + `&mut self`"模型：

- ﻿`lib/kv-router/src/indexer/kv_indexer.rs` 里，`KvIndexer` 用 `std::thread` 起一个后台线程，线程内部 `let mut trie = RadixTree::new();`——****没有**** `****Arc<Mutex>>****`，因为只有这一个线程能写。
    
- 所有对 trie 的读写都通过 `tokio::sync::mpsc` 传入 `MatchRequest` / event / control op。
    
- Channel 容量：events `16384`, routing decisions `2048`, matches `128`, control ops `16`。
    
- 主循环用 `tokio::select! { biased; ... }`，`biased` 确保 cancellation / worker removal 分支优先，避免长事件流饥饿控制信号。
    
- Clone 安全通过 `_ref_count: Arc<()>` 实现，`Drop` 里 `Arc::strong_count(...) == 1` 才真正 cancel 后台线程。
    

这个设计的直接好处是****无锁读写****——查询和事件消费不会互相阻塞。代价是所有跨线程通信都要走 channel，延迟受 mpsc 调度影响。Router 通过 `dynamo_router_overhead_indexer_find_matches_ms` histogram 监控这条链路的实际延迟（详见 §11）。

> 来源：`lib/kv-router/src/indexer/kv_indexer.rs`。

### 4.3 Block hash 算法：per-block 独立 XXH3-64，不是 chained

****这是本文与旧**** `****Dynamo-KV-Cache-Router-机制详解.md****` ****的一个重要纠正****：block hash 是****per-block 独立****的 XXH3-64，****不是**** chained rolling hash；而 `SequenceHash` 才是 chained（把上一个 block 的 hash 当 parent 混入）。这两个概念在 event schema 里分别对应 `LocalBlockHash` 和 `ExternalSequenceBlockHash`。

****源码事实****（`lib/kv-router/src/protocols.rs`）：

// XXH3 seed 基线常量

const XXH3_SEED: u64 = 1337;

// 实际 seed 由 namespace + LoRA 派生 salt

seed = dynamo_kv_hashing::compute_salt_hash(cache_namespace, lora_name)

- ﻿`****LocalBlockHash****`：单个 block 的独立 XXH3-64，只哈希"当前 block 的 token bytes（+ 可选的 mm_hash）"。相邻 block 的 hash 之间****没有依赖****。
    
- ﻿`****SequenceHash****`：chained rolling hash——第一个 block 的 SequenceHash 等于其 LocalBlockHash，后续通过 `compute_next_sequence_hash(parent_seq_hash, current_block_hash)` 递推。
    
- ﻿`****ExternalSequenceBlockHash****`：engine 侧（如 vLLM/SGLang runtime）自己算的 sequence-level hash；Router 只当作 `u64` 存，不重新计算。
    

****Block 切分规则****：

|   |   |   |   |
|---|---|---|---|
|场景|Window (stride)|Block 数估算|尾部处理|
|非 Eagle 模型|`kv_block_size`|`tokens.len() / stride`|不足一个 window 的丢弃|
|Eagle speculative decoding|`stride + 1`（相邻 block 1 token 重叠）|`(len - 1) / stride`|同上|
|`kv_block_size == 0`|—|返回空 vec|—|

  

****输入字节****：little-endian 平台直接把 `&[u32]` 当 bytes 用；否则逐个 `to_le_bytes` 写入。

****Multimodal 支持****：一个 block 的 `mm_hash` 集合会先 `sort_unstable`、再按 `u64` LE 追加到 token bytes 之后再 hash。源码注释原文——__"blocks with identical tokens but different multimodal objects produce different hashes."__

****默认 block_size****：`DEFAULT_KV_CACHE_BLOCK_SIZE = 16`（`lib/llm/src/local_model.rs`）；CLI 通过 `--kv-cache-block-size` / `DYN_KV_CACHE_BLOCK_SIZE` 覆盖。注意这个参数****必须与后端引擎的 KV block size 对齐****——vLLM 的 `--block-size`、SGLang 的 `--page-size` 都要匹配，否则命中率信号会失真。

> 来源：`lib/kv-router/src/protocols.rs`；`lib/llm/src/local_model.rs`。

### 4.4 单 Router 内部的两种 RadixTree 后端

﻿`--router-event-threads` 控制 Indexer 内部的并发模型（`docs/design-docs/router-design.md`）：

|   |   |   |
|---|---|---|
|`--router-event-threads`|Indexer 实现|特性|
|`=1`|单线程`RadixTree`|也支持 approximate mode 下的 TTL expiration|
|`>1`（****默认 4****）|`ConcurrentRadixTreeCompressed` + `ThreadPoolIndexer`|****thread-safe****，采用 ****sticky worker routing****——同一 worker 的事件恒定落到同一 thread 上串行化，读则可以并发进行|

  

Sticky 分片的关键含义：****同一 worker 的事件在同一 thread 上按 event_id 顺序应用****，避免 out-of-order 事件破坏 parent_hash 链条；不同 worker 之间的事件可以真正并行处理。

> 来源：`docs/design-docs/router-design.md`；`lib/kv-router/src/indexer/concurrent_radix_tree_compressed/`。

### 4.5 Overlap analysis 的输出格式

﻿`query_tiered_matches(...)` 的输出是一个 `OverlapAnalysis`，包含每个候选 worker 的四类 overlap blocks：

|   |   |   |
|---|---|---|
|字段|单位|来源|
|`device_overlap_blocks`|KV blocks|primary Indexer（GPU tier）|
|`host_overlap_blocks`|KV blocks|LowerTierIndexers 的 host 层（如 SGLang HiCache`CPU_PINNED`）|
|`disk_overlap_blocks`|KV blocks|LowerTierIndexers 的 disk 层|
|`shared_beyond_blocks`|KV blocks|SharedKvCache（Mooncake 等）—— 是"共享池命中中超出当前 worker 本地 device overlap 的部分"|

  

﻿`****shared_beyond_blocks****` ****的语义比较微妙****：源码用 `shared_cache_hits.hits_beyond(device_overlap_blocks)` 表达——****只减 device 部分，不减 host / disk****。这与 §6.13 讲的 device/host/disk 三层内部互斥的机制****是两回事****，具体差异与潜在的 double-count 问题详见 §6.14。

****⚠️ 关于 device / host / disk 三项是否也有类似去重****：****是的，有去重****——通过 Indexer 层的 `query_lower_tiers` 用 continuation offset 机制实现（详见 §6.13）。`tier_overlap_blocks.device / host_pinned / disk` 三个字段传给 selector 时****已经是 tier-exclusive 互斥值****（同一个 block 只出现在它所属最高 tier 的字段里）。因此 selector.rs 里三项直接相加****不会 double-count****。这条去重链路有官方回归测试 `concurrent_tiered_query_does_not_double_count_device_and_lower_tier_overlap` 守护（`lib/llm/src/kv_router/indexer/mod.rs`）。

> 来源：`lib/kv-router/src/scheduling/selector.rs`；`docs/backends/sglang/sglang-hicache.md`（tier-aware routing 公式与 tier priority 承诺）。

---

## 5. KV Events 协议与传输（信息来源之二：worker 事件流）

Indexer 的数据来自哪里？答案是 ****worker 上报的 KV events****——每个 worker 在其内部 KV cache 发生存储 / 驱逐 / 清空时，主动向 event plane 发一条事件，Router 侧的 Indexer 订阅这条事件流并相应更新前缀树。

这一节展开事件的数据结构、传输通道、以及一个关键的横向对比——****Dynamo 的 event 平面本质上是 vLLM ZmqEventPublisher 的超集****（提供了 vLLM ZMQ replay 所缺的 initial state sync 和长时间断线恢复能力）。

### 5.1 Event 结构（Rust struct，逐字段展开）

顶层的传输单元是 `RouterEvent`（`lib/kv-router/src/protocols.rs`）：

pub struct RouterEvent {

    worker_id: WorkerId,

    storage_tier: StorageTier,   // ← medium 字段实际位置

    event: KvCacheEvent,

}

pub struct KvCacheEvent {

    event_id: u64,               // 每 worker 单调递增，用于 gap 检测

    data: KvCacheEventData,

    dp_rank: DpRank,

}

#[serde(tag = ..., rename_all = "snake_case")]

pub enum KvCacheEventData {

    Stored(KvCacheStoreData),

    Removed(KvCacheRemoveData),

    Cleared,

}

pub struct KvCacheStoreData {

    parent_hash: Option<ExternalSequenceBlockHash>,  // 上一 block 的 seq hash

    start_position: Option<u32>,                     // 该 batch 首 block 的绝对位置

    blocks: Vec<KvCacheStoredBlockData>,

}

pub struct KvCacheStoredBlockData {

    block_hash: ExternalSequenceBlockHash,   // 引擎侧自算的 sequence hash

    tokens_hash: LocalBlockHash,             // Router 侧 token bytes 的 XXH3-64

    mm_extra_info: Option<BlockExtraInfo>,

}

pub struct KvCacheRemoveData {

    block_hashes: Vec<ExternalSequenceBlockHash>,

}

****要点澄清****（这里也涉及对旧文档的两处修正）：

1. ﻿`****medium****` ****字段的位置****：不在 `KvCacheEvent` 上，而在****包装层**** `****RouterEvent.storage_tier****` 上。`StorageTier` enum 通过 `to_kv_medium` / `from_kv_medium` 映射到线路上的字符串常量：`"CPU_PINNED"`、`"DISK"`、`"EXTERNAL"`；`Device` 变体不带 medium（None）。SGLang HiCache 发的 `medium=CPU_PINNED` 就是这个通道。
    
2. ****两种 hash 并存****：`KvCacheStoredBlockData` 同时携带 `block_hash: ExternalSequenceBlockHash`（引擎自算的 sequence-level 链式 hash）和 `tokens_hash: LocalBlockHash`（Router 自算的 per-block XXH3-64）。这是为了让 Router 既能与引擎的内部 hash 系统对齐（用 external），又能独立于引擎做 token bytes 层面的匹配（用 local）。
    

> 来源：`lib/kv-router/src/protocols.rs`。

### 5.2 传输通道：ZMQ / NATS JetStream 双通路

****Dynamo 的 event 平面是双通路****，通过 `--event-plane={zmq|nats}` 选择（默认 `zmq`）。两条路径的差异：

|   |   |   |
|---|---|---|
|维度|ZMQ（默认）|NATS JetStream|
|Transport|ZMQ SUB socket，`connect_sub_socket(endpoint, Some(topic))`|`NatsQueue::publish_event(KV_EVENT_SUBJECT, ...)`|
|Wire format|三帧 multipart：`topic` / 8-byte big-endian `engine_seq` / msgpack payload|由`NatsQueue::publish_event` 内部编码（bincode 是明确出现过的候选，源码注释有 "bincode is positional" 提示）|
|Event ID|Router 侧用`AtomicU64::fetch_add(1, SeqCst)` 分配|由 Publisher 端在 JetStream 中分配|
|持久化|无（fire-and-forget）|JetStream stream + object store snapshot（默认 1h 保留）|
|Snapshot restore|无|新 replica 启动自动 restore|
|常量|`pub const KV_EVENT_SUBJECT: &str = "kv-events";`|同 subject|

  

****"KV events 是元信息，不是 tensor"****——两条通路传输的都是"谁存了/删了哪些 block hash"，****不搬 KV tensor 本体****。真正的 KV 数据搬运在 P/D 分离场景由 NIXL 通过 RDMA / NVLink / GDS 直传（详见 §9）。

****JetStream 的 deprecated 状态****：`--router-durable-kv-events` 在 v1.2 起被标记为 deprecated，官方推荐 "The event-plane subscriber in local indexer mode"（也就是默认的 ZMQ + Local Indexer 组合）。

> 来源：`lib/llm/src/kv_router/publisher/zmq_listener.rs`、`lib/llm/src/kv_router/publisher/sinks.rs`。

### 5.3 事件消费者：event_processor

ZMQ / NATS 收到的事件先进 `mpsc::UnboundedSender<Vec<PlacementEvent>>`，由 `lib/llm/src/kv_router/publisher/event_processor.rs` 消费。它做两件事：

1. ****Coalesce / dedup****：`Stored` 事件沿 `parent_hash` 链拼接、`Removed` 合并 `block_hashes`、`Cleared` 立即 flush 并清 dedup filter。Flush 触发点：`dp_rank` 变、`tier` 变、Stored/Removed 混合、parent chain 断、达到 `max_batch_blocks`、超时、cancellation、channel close。
    
2. ****Gap detection****：每个 worker 的 event_id 应该单调 +1；一旦 `incoming_event_id > last_id + 1`，视为丢件，触发 warning `"Input event gap detected: raw events dropped before batching"` 并增加 metric `engines_dropped_events_total`。
    

### 5.4 Gap recovery：Dynamo 相对 vLLM ZMQ 的真正优势

Router 检测到 gap 后不是就此放弃，而是通过一个 ****replay 通道**** 找 worker 侧的 `LocalKvIndexer` 补数据。`LocalKvIndexer.get_events_in_id_range(start_id, end_id)` 有三种响应（`docs/components/router/kv-event-replay-comparison.md`）：

|   |   |
|---|---|
|响应类型|含义|
|`Events`|目标区间在环形 buffer 内命中，binary search 切片返回|
|`TreeDump`|区间太老、或初次同步（`start_id=None`）——****序列化整棵 RadixTree，作为合成的 events 返回****（等价于一份全量 snapshot）|
|`TooNew`|消费者请求的 event_id 超前于 worker 侧的当前进度，无 gap 可补|

  

****与 vLLM**** `****ZmqEventPublisher****` ****的对比****（这是理解 Dynamo 相对 verl `KVCacheStore` 的一个关键差异）：

|   |   |   |
|---|---|---|
|能力|vLLM ZmqEventPublisher|Dynamo Router|
|Ring buffer|`collections.deque[tuple[int, bytes]]`，默认 10,000 条 msgpack batch|同样有环形 buffer|
|Replay socket|ROUTER socket 应答 replay 请求|同样有 replay 通道|
|****Initial state sync****|❌ "consumer that connects after events have already been published starts with an empty view."|✅ TreeDump 响应可以拉全量|
|****Long-outage recovery****|❌|✅ 环形 buffer miss 后 fallback 到 TreeDump|

  

也就是说：****verl**** `****KVCacheStore****` ****用的那种"启动时发一个**** `****b\"replay\"****` ****请求"的机制，Dynamo 天然支持并且做得更完整****——TreeDump 提供全量恢复，Events 提供增量补齐，TooNew 处理消费者超前，三态覆盖了从冷启动到网络抖动的全部情况。这也是为什么 Dynamo 的多 Router 部署可以做到"launch a third router replica even if the first two are down, and it will recover the full prefix state"（`docs/components/router/router-operations`）。

****横向对比小节****：

|   |   |   |
|---|---|---|
|系统|Replay 机制|冷启动行为|
|verl`KVCacheStore`|启动时向 replay socket 发`b"replay"`（超时 5s 降级为纯订阅）|有 replay 通道，但语义比 Dynamo 简单|
|RL-Router`radixPrefixCache`|****无**** replay，本地历史索引|Router 重启后所有历史前缀失效|
|SGLang Gateway`cache_aware`|****无****（本地近似前缀树，请求驱动）|Gateway 重启后 tree 清零，需重新填热|
|****Dynamo****|三态 replay（Events / TreeDump / TooNew） + 可选 JetStream 持久化|有 authoritative RadixTree + 增量与全量恢复|

  

### 5.5 事件应用统计（观测点）

Router 暴露一个 counter 记录事件应用的四类 status × 三类 event_type：`dynamo_component_kv_cache_events_applied`（`lib/llm/src/kv_router/metrics.rs`）：

- ****status****：`ok` / `parent_block_not_found` / `block_not_found` / `invalid_block`﻿
    
- ****event_type****：`stored` / `removed` / `cleared`﻿
    

这个 counter 是排查 "cache 命中率异常低" 的最直接观测点——如果 `parent_block_not_found` 显著高于 `ok`，说明事件流有 gap 或乱序问题（可能是 event_thread 分片错乱、event_plane 拥塞、或 worker 侧 hash 算法不匹配）。

### 5.6 SGLang HiCache 集成：tier-aware 事件的具体来源

Dynamo 想做 tier-aware routing（同时看 device / host / disk 的 overlap）时，需要 worker 侧上报带 tier 信息的事件。SGLang HiCache 集成路径是官方支持最好的一条：

****硬门槛****（`docs/backends/sglang/sglang-hicache.md`）：

> "Tier-aware shared cache routing requires ****SGLang 0.5.11 or later****."

（关联 SGLang PR #22894：`fix(hicache): emit KV events for L2 host cache insertions`。Dynamo 1.3.0 官方 SGLang runtime 镜像已内置 SGLang 0.5.15。）

****六种 tier transition 与事件对应****（HiRadixCache 在每种状态迁移时的行为）：

|   |   |
|---|---|
|Tier transition|事件序列|
|Fresh prefill → GPU|`store(GPU)`|
|GPU → Host DMA 完成|`store(CPU_PINNED)`（****等**** `****finish_event.synchronize()****` ****确认 DMA 落地后才发****）|
|GPU evict 但 Host 还在|`remove(GPU)`|
|Host evict（全 tier 消失）|`remove(CPU_PINNED)`|
|Host → GPU promotion（`load_back`）|`store(GPU)`|
|External → Host prefetch|`store(CPU_PINNED)`|

  

****事件顺序保证****（这一点对正确性至关重要）：

> "`store(new_tier)` is emitted before `remove(old_tier)` so the block is never invisible to the router during a transition."

也就是说，一个 block 从 GPU 迁到 Host 时，先发 `store(CPU_PINNED)`（对应 host tier）、再发 `remove(GPU)`（对应 device tier），保证 Router 视角下这个 block 从未消失过——中间态是"同时在 GPU 和 Host 上"，而不是"两边都没有"。

****Mooncake 版本坑****（`docs/backends/sglang/sglang-hicache.md`）：

> "Mooncake 0.3.10.post2 will crash `MemcpyWorkerPool` when both `--enable-metrics` and `--disable-piecewise-cuda-graph` are enabled; upgrade to 0.5.13+ with Mooncake 0.3.11.post1."

生产部署推荐 SGLang ≥ 0.5.13 + Mooncake ≥ 0.3.11.post1。

****与姊妹方案的对比****：

|   |   |
|---|---|
|方案|Tier 事件源|
|verl`KVCacheStore`|vLLM ZMQ block events（****只有 GPU tier****，无 host/disk 分层）|
|RL-Router`pd_cache_aware`|本地 radix 反推（****完全没有 tier 概念****）|
|SGLang Gateway`cache_aware`|本地近似 tree（同样无 tier）|
|****Dynamo****|HiCache`medium` 字段 / KVBM 分层事件（****真正的 device/host/disk 分层****）|

  

---

## 6. Cost Function 逐项拆解

这是全文最核心的一节。Dynamo 的所有打分决策都归结到 `DefaultWorkerSelector::worker_logit(...)`（`lib/kv-router/src/scheduling/selector.rs`）——本节把这个函数的公式、变量来源、默认值、以及 argmin/softmax 采样的实现细节全部展开。

### 6.1 完整公式

****官方文档版本****（`docs/backends/sglang/sglang-hicache.md`，tier-aware 完整式）：

adjusted_prefill_blocks = max(

    prefill_blocks

    - overlap_score_credit    * device_overlap_blocks       # device credit

    - host_cache_hit_weight   * host_overlap_blocks         # host credit

    - disk_cache_hit_weight   * disk_overlap_blocks         # disk credit

    - shared_cache_multiplier * shared_beyond_blocks,       # shared credit

    0,

)

cost = prefill_load_scale * adjusted_prefill_blocks + decode_blocks

****源码等价实现****（`lib/kv-router/src/scheduling/selector.rs :: worker_logit`）：

let raw_prefill_blocks = raw_prefill_tokens / block_size_f64;

// device credit 应用 decay

let overlap_credit_decay =

    1.0 / (1.0 + weights.overlap_score_credit_decay * normalized_prefill_load);

let effective_overlap_score_credit =

    weights.overlap_score_credit * overlap_credit_decay;

// 四层 credit 加总

let overlap_credit_blocks =

      effective_overlap_score_credit           * device_overlap_blocks as f64

    + self.kv_router_config.host_cache_hit_weight * host_overlap_blocks as f64

    + self.kv_router_config.disk_cache_hit_weight * disk_overlap_blocks as f64

    + shared_overlap_blocks;   // = weights.shared_cache_multiplier * (beyond as f64)

let adjusted_prefill_blocks = (raw_prefill_blocks - overlap_credit_blocks).max(0.0);

let prefill_cost_blocks    = weights.prefill_load_scale * adjusted_prefill_blocks;

let decode_cost_blocks     = worker_load.potential_decode_blocks() as f64;

let logit = prefill_cost_blocks + decode_cost_blocks;

Debug log 里有一句总结原话：`__"prefill_load_scale * adjusted_prefill_blocks + decode_blocks"__`——两套公式完全一致。

> 来源：`lib/kv-router/src/scheduling/selector.rs :: worker_logit`；`docs/backends/sglang/sglang-hicache.md`；`docs/components/router/router-concepts.md`。

### 6.2 变量来源三分类（对齐姊妹文档的钥匙）

姊妹文档 `RL-Router调度策略.md` §3.5 定义了一套"数据来源三分类"作为理解全文打分/判定的钥匙。这套分类同样适用于 Dynamo，且对齐得更清晰：

|   |   |   |   |
|---|---|---|---|
|类别|更新频率|精度|Dynamo 里的对应变量|
|****本地历史索引****|事件驱动（推送）|高（近实时）|`device/host/disk_overlap_blocks`（Indexer 消费 KV events）、`shared_beyond_blocks`（远端查询）|
|****本地计数****（`&mut self` 而非 atomic）|请求粒度（sub-ms）|精确|`active_prefill_tokens`、`potential_decode_blocks`（`ActiveSequences`）、`raw_prefill_blocks`|
|****HTTP 轮询指标****|很少用|—|Dynamo Router 本身不依赖 HTTP polling worker metrics；Planner 侧才 poll（`throughput_metrics_source`）|

  

****Dynamo 与姊妹方案在数据源上的核心差异****：

- verl `KVCAwareBalancer` 有明确的 HTTP polling 通道（5s 拉一次 `vllm:kv_cache_usage_perc` 等），且用 `KVCacheStore` 反向索引维护 cache 状态。
    
- RL-Router `session_aware_v5` 也走 HTTP polling（2s）+ 本地计数。
    
- ****Dynamo Router 决策链路上不查 HTTP**** `****/metrics****`——负载信号全部来自请求生命周期内 Router 自己维护的 `ActiveSequences`，cache 信号全部来自 event plane。这使得 Router 决策路径是纯本地的（sub-ms），且数据实时性比 polling 高。
    

### 6.3 打分函数所有变量的语义与来源

|   |   |   |
|---|---|---|
|变量|语义|来源|
|`raw_prefill_tokens`|请求预期需要 prefill 的 token 数|由`track_prefill_tokens` 开关决定（见 6.5）|
|`raw_prefill_blocks`|`raw_prefill_tokens / block_size`|派生|
|`device_overlap_blocks`|该 worker 的 GPU tier 上匹配的 prefix block 数|Indexer primary（GPU tier）查询|
|`host_overlap_blocks`|该 worker 的 host-pinned tier 上匹配的 prefix block 数|Indexer LowerTierIndexers 的`CPU_PINNED`|
|`disk_overlap_blocks`|该 worker 的 disk tier 上匹配的 prefix block 数|Indexer LowerTierIndexers 的`DISK`|
|`shared_beyond_blocks`|共享池命中中"超出该 worker device overlap 的部分"|SharedKvCache 查询|
|`overlap_score_credit_decay`|device credit 衰减系数|全局 config，****不能 per-request 覆盖****|
|`effective_overlap_score_credit`|`overlap_score_credit * decay_factor`|派生|
|`prefill_load_scale`|prefill 项的权重|全局 config 或 per-request override|
|`overlap_credit_blocks`|四层 credit 加总|派生|
|`adjusted_prefill_blocks`|`max(raw_prefill_blocks - overlap_credit_blocks, 0)`|派生|
|`decode_blocks`|该 worker 的 KV cache memory footprint（含新请求预留）；****详细语义见 §6.11****|`WorkerLoadProjection::potential_decode_blocks() = active_decode_blocks + additional_active_blocks`|
|`logit` / `cost`|最终打分|派生|

  

### 6.4 权重默认值表

来源函数：`lib/kv-router/src/scheduling/config.rs :: KvRouterConfig::default()`。以下所有值都可通过 CLI / env / per-request override 修改（能覆盖范围见后一小节）。

|   |   |   |
|---|---|---|
|字段|默认值|含义|
|`overlap_score_credit`|****1.0****|device credit（`--router-kv-overlap-score-credit` / `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT`）|
|`overlap_score_credit_decay`|****0.0****|device credit 衰减；`1` 表示"1 request 的 excess prefill 让 credit 减半"|
|`host_cache_hit_weight`|****0.75****|host credit（`--router-host-cache-hit-weight`）|
|`disk_cache_hit_weight`|****0.25****|disk credit（`--router-disk-cache-hit-weight`）|
|`shared_cache_multiplier`|****0.0（源码结构体默认）/ 0.5（CLI 打开 shared 时）****|shared pool credit|
|`prefill_load_scale`|****1.0****|prefill 项系数|
|`router_temperature`|****0.0****（deterministic argmin）|非零启用 softmax 采样|
|`use_kv_events`|****true****|关闭后走 approximate mode（自预测 + TTL）|
|`router_ttl_secs`|****120.0****|approximate mode 下的自预测 state TTL|
|`router_track_prefill_tokens`|****true****|关闭后`raw_prefill_tokens = request ISL`|
|`router_track_active_blocks`|****true****|active block 是否记账|
|`router_track_output_blocks`|****false****|output block 是否也计入 decode_blocks|
|`router_assume_kv_reuse`|****true****|假设选中的 worker 会真的复用 cache（写入 Indexer）|
|`router_event_threads`|****4****|Indexer 分片线程数（>1 用 ConcurrentRadixTreeCompressed）|
|`router_queue_policy`|`Fcfs`（enum `#[default]`）|排队策略（详见 §7）|
|`router_queue_threshold`|****None****|无阈值时不排队|
|`router_snapshot_threshold`|****1_000_000****|JetStream snapshot 阈值|
|`router_reset_states`|****false****|JetStream-only；启动时 purge stream + snapshot|
|`router_predicted_ttl_secs`|****None****|Predicted TTL 特性（burst workload）|

  

****关于官方文档里的"最推荐值"****：`docs/components/router/configuration-and-tuning` 有一句 `"Higher values improve Time To First Token (TTFT) at the cost of Inter-Token Latency (ITL)."` ——`overlap_score_credit` 越高越偏 cache affinity，TTFT 更好但可能牺牲 ITL；反之偏负载均衡。

### 6.5 per-request override 的范围

﻿`RouterConfigOverride`（`lib/kv-router/src/scheduling/config.rs`，`RouterConfigOverride` struct）只允许 6 个字段 per-request 覆盖：

|   |   |
|---|---|
|字段|允许覆盖|
|`overlap_score_credit`|✅|
|`prefill_load_scale`|✅|
|`router_temperature`|✅|
|`assume_kv_reuse`|✅|
|`track_prefill_tokens`|✅|
|`shared_cache_multiplier`|✅|
|`overlap_score_credit_decay`|❌****不能覆盖****（只从全局 config 取）|
|`host_cache_hit_weight` / `disk_cache_hit_weight`|❌|

  

decay 与 host/disk 权重不能 per-request 覆盖，是有意为之的——这两个是全局架构决策（每个部署 tier 结构一致），不适合 per-request 变。

### 6.6 Device-credit decay 的语义

﻿`overlap_score_credit_decay > 0` 时启用：

credit_effective = overlap_score_credit / (1 + decay * normalized_excess)

normalized_excess = excess_active_prefill_blocks / request_blocks

含义：****当某个 worker 已经积压了大量 prefill load，即便它的 cache 命中率最高，也不应该继续无脑往那儿打****——因为再打过去，请求会先排队等 prefill 消化完，TTFT 反而更差。decay 因子把 cache-rich but overloaded 的 worker 的 device credit 打折。

官方原话（`docs/components/router/configuration-and-tuning`）：

> "a decay of 1 halves device credit at one request-equivalent of excess prefill load."

****注意****：decay 只作用于 device credit，不作用于 host / disk / shared credit。这是合理的——host/disk 命中的目的不是"用它当 prefill 源"，而是"避免完全冷启动"，所以不需要跟 prefill load 挂钩。

### 6.7 Argmin 与 softmax 采样

Dynamo 用 `logit = cost`（越小越好）而非 verl 的 `score = argmax`。这个方向选择在 softmax 阶段体现为一个巧妙的实现：

****Temperature = 0（默认，deterministic）****：

let mut best_logit = f64::INFINITY;

for (worker, score) in &all_scores {

    if score < best_logit {

        best_logit = *score;

        best_worker = Some(*worker);

        tie_count = 1;

    } else if score == best_logit {

        tie_count += 1;

        // reservoir sampling: 均匀概率替换

        if rng.random_range(0..tie_count) == 0 {

            best_worker = Some(*worker);

        }

    }

}

****Reservoir sampling tie-break****：多个 worker 打平时不是取第一个（会导致所有 tie 请求集中打到同一个 worker），而是每个 tie worker 有 `1/tie_count` 的概率成为当前 winner。这天然处理了"两个 worker 打分完全相同"的情况——常见于 cold start 阶段所有 worker overlap 都是 0 的场景。

****Temperature > 0（softmax 采样）****（`softmax_sample_with_sample`）：

// 1. 找 (min_val, max_val)

// 2. 若 max_val == min_val，均匀采样

// 3. 否则：

let scale = -1.0 / ((max_val - min_val) * temperature);   // 负 scale

let max_scaled = min_val * scale;

// prob_i = ((v_i * scale) - max_scaled).exp()  之后归一化 + inverse-CDF

****关键点****：`scale` 是****负****的，所以 logit 越小的 worker，`v_i * scale` 越大，`.exp()` 后概率越高。这就把 argmin(logit) 通过负 scale 自然翻转成了 argmax(prob)，等价于对 `-logit / ((max-min) * T)` 做标准 softmax。

****归一化步骤中减去**** `****max_scaled****` 是数值稳定性技巧，防止 `.exp()` 溢出（等价于 `log-sum-exp` 减最大值）。

****Temperature 的语义****：

|   |   |
|---|---|
|Temperature|行为|
|0.0（默认）|Deterministic argmin，tie 时 reservoir sampling|
|中等（例如 0.1–1.0）|大概率选低 cost，少量探索|
|较高|更接近均匀分布，退化为随机|

  

> 来源：`lib/kv-router/src/scheduling/selector.rs :: softmax_sample`, `softmax_sample_with_sample`。

### 6.8 track_prefill_tokens 与 raw_prefill_tokens

﻿`track_prefill_tokens=true`（默认）时：

raw_prefill_tokens = active_prefill_tokens + uncached_tokens

raw_prefill_tokens = raw_prefill_tokens.saturating_add(cached_tokens)   // 若 assume_kv_reuse

- ﻿`active_prefill_tokens` = 该 worker 上其他请求当前还没算完的 prefill token 总量。
    
- ﻿`uncached_tokens` = 本次请求扣掉 cache 命中后还需要新算的 tokens。
    
- ﻿`saturating_add(cached_tokens)` = 假设 cache 会被复用，本次请求的 cached 部分也短暂占位。
    

﻿`track_prefill_tokens=false` 时，`raw_prefill_tokens = 请求的 ISL`（完全不考虑 cache）——这是 `--load-aware` preset 用的模式，等价于"关掉 cache 感知，只按预期总 prefill 量 + decode load 打分"。

官方文档补充（`docs/components/router/router-concepts.md`）：

> "the router estimates each candidate worker's uncached prompt work by subtracting its cached prefix tokens from the request's input tokens."  
> "that effective prefill load remains charged at full value until the first output token marks prefill complete."

也就是说：****请求在 prefill 阶段一直按满 prefill load 计****，直到首 token 出来才算 prefill 完成（触发 `mark_prefill_completed`）——这一点由 `ActiveSequences::mark_prefill_completed` 保证。

### 6.9 Expected output length（`nvext.agent_hints.osl`）

Dynamo 有一个"客户端可选传预期输出长度"的机制：请求 payload 里带 `nvext.agent_hints.osl`（Output Sequence Length hint），output blocks 会按进度打折。

官方原话（`docs/components/router/router-concepts.md`）：

> "If the request includes `nvext.agent_hints.osl`, those output blocks receive a fractional weight based on progress toward the expected output length."  
> "Without an expected OSL, tracked output blocks count at full weight until the request finishes."

含义：****长请求接近完成时（已经生成 90% 的 OSL），其占用的 decode block 应该在打分中打折****——因为它很快会释放，把新请求路由过去不会真的排到它后面。这是 Dynamo 独有的一个"未来负载估计"信号，verl / RL-Router / SGLang Gateway 都没有对应机制。

#### 6.9.1 OSL 从 payload 到打分的完整数据流

****先说 pre-condition****：整套 OSL 机制****只有当**** `****router_track_output_blocks=true****`****（默认**** `****false****`****）时才生效****。默认配置下，即便客户端传了 `osl`，也只会存在 `RequestState` 里当作元数据，不影响任何打分决策。

****完整数据流****：

1. Client HTTP 请求携带 nvext:

   { "messages": [...],

     "nvext": { "agent_hints": { "osl": 512 } } }

2. Frontend preprocessor 抽取 OSL:

   lib/llm/src/preprocessor.rs (~1060)

   RoutingHints.expected_output_tokens = nvext.agent_hints.osl

3. Router 派发请求 + 建立 RequestGuard:

   lib/llm/src/kv_router/push_router/request_guard.rs::new

   OutputBlockTracker {

     track_output_blocks = router_track_output_blocks,   // 若 false 后面 add_output_block 完全不调

     isl_tokens          = 请求初始长度,

     block_size,

     expected_output_tokens = osl,                        // 拷贝

     current_total_blocks   = isl_tokens.div_ceil(block_size),

   }

4. 每次 worker 返回一批 tokens 触发 RequestGuard::on_item:

   lib/llm/src/kv_router/push_router/request_guard.rs::on_item

   on_item(item):

     new_tokens         = item.data.token_ids.len()

     cumulative_osl    += new_tokens                       # 已生成 token 数累加

     new_total_blocks   = (isl_tokens + cumulative_osl).div_ceil(block_size)

     # 只在跨过新 block 边界时才走下面这段

     if new_total_blocks <= current_total_blocks:

       return

     current_total_blocks = new_total_blocks

     # 计算 decay_fraction ← 唯一的公式所在

     decay_fraction = expected_output_tokens.map(|expected|

       max(0.0, 1.0 - cumulative_osl / max(expected, 1))

     )

     # 若 osl 未传，decay_fraction = None → 后续按满值 1.0 计

     # 若 cumulative_osl ≥ expected，clamp 到 0.0

     if !track_output_blocks:

       return                                              # 门控点，默认走这里

     scheduler.add_output_block(request_id, decay_fraction)

5. 应用到 block_tracker:

   lib/kv-router/src/sequences/single.rs::add_output_block

   lib/kv-router/src/sequences/block_tracker.rs::append_output + set_unique_suffix_fractional

     block_tracker.output_blocks[random_hash] = 1.0        # 先按满值插入

     if decay_fraction is Some(frac):

       # 把当前请求的"结构独占后缀"权重刷成 frac

       for hash in request.output_hashes:

         output_blocks[hash] = frac

       # 沿 prompt tail 往上，只刷 incoming==1 的边（保留共享 prefix 全权重）

       ...

6. active_blocks 汇总（round 后就是 active_decode_blocks）:

   lib/kv-router/src/sequences/block_tracker.rs::active_blocks

   active_blocks() = round(prompt_total + output_total)

     # prompt_total 与 output_total 都是 f64 加权累加

     # 请求接近 osl 末尾时 output_total 会衰减接近 0

7. 打分公式使用（下次请求路由到该 worker 时）:

   lib/kv-router/src/scheduling/selector.rs::worker_logit

   decode_cost_blocks = potential_decode_blocks() as f64

                      = active_decode_blocks + additional_active_blocks

                      # active_decode_blocks 已经包含 OSL 衰减后的值

   logit = prefill_load_scale * adjusted_prefill_blocks + decode_cost_blocks

#### 6.9.2 核心公式（`request_guard.rs :: OutputBlockTracker::observe`）

let decay_fraction = self

    .expected_output_tokens

    .map(|expected| (1.0 - cumulative_osl as f64 / expected.max(1) as f64).max(0.0));

用数学表达：

decay_fraction = max(0, 1 − cumulative_osl / max(expected_osl, 1))

- ﻿`****expected_osl****` ****未传时****：`decay_fraction = None` → 该 output block 按权重 ****1.0**** 一直计入 `active_decode_blocks`，直到请求 `free()` 才释放。
    
- ****刚生成时****（`cumulative_osl` 很小）：`decay_fraction ≈ 1.0` → 与"无 OSL"接近。
    
- ****生成到 50% OSL 时****：`decay_fraction = 0.5` → 该请求占的 output block 只算一半权重。
    
- ****生成到 100% OSL 或更多时****：`decay_fraction = 0` → 该请求占的 output block 完全不算权重。
    
- ****权重刷新的范围****：****不是只刷新新加的这一个 block****，而是刷新该请求的****所有 output block + 结构独占的 prompt 后缀****（`set_unique_suffix_fractional` 沿 prompt_tail 往上，遇到 `incoming != 1`（即有其他请求共享的边）就停止——保留共享 prefix 的全权重）。
    

#### 6.9.3 触发节奏：****跨 block 边界才触发，不是每 token 一次****

- 每次 worker 返回 stream chunk → `on_item()` 触发 → 累加 `cumulative_osl`；
    
- 但只有当 `new_total_blocks > current_total_blocks`（即累计 tokens 超过当前 block 边界）****才实际调用一次**** `****add_output_block****`；
    
- 调用频率大概是"每 `block_size` 个 output token 触发一次"——`block_size=16` 时约每 16 token 一次。
    

#### 6.9.4 `router_track_output_blocks` 完整语义

****默认**** `****false****` 时的行为（`request_guard.rs:395`）：

if !self.output_blocks.track_output_blocks {

    return;                        // 门控在这里，add_output_block 完全不被调用

}

- output tokens ****完全不进**** `active_decode_blocks`——不是"进但权重 0"，是根本不加进去；
    
- 请求的 `active_decode_blocks` 只包含 ****prompt 侧****（`additional_active_blocks`），并在 `free()` 时释放；
    
- 客户端传 `osl` 也没用，只是当元数据存着。
    

﻿`****true****` ****时****：跨 block 边界触发；有 osl 就衰减，无 osl 就按 1.0 计。

****硬约束****：`router_track_output_blocks=true` 要求 `router_track_active_blocks=true`，否则启动校验直接报错（`config.rs:647`）。

#### 6.9.5 打分公式里 OSL 影响的落点

Output block 通过 `active_decode_blocks` 一条路径影响 `logit`——****没有独立的 output 项****：

logit = prefill_load_scale * adjusted_prefill_blocks

      + potential_decode_blocks()

potential_decode_blocks() = active_decode_blocks + additional_active_blocks

                            ↑

                            这里已经包含 OSL 衰减后的加权 output block

具体地，`active_decode_blocks = round(prompt_total + output_total)`，其中 `output_total` 是 f64 加权累加，OSL 衰减直接影响这个值。也就是说：****一个已经生成 90% OSL 的长请求，占的 output block 在下次决策时几乎不影响**** `****logit****`——虽然物理上它还占着 memory slot，但打分不再"惩罚"这个 worker，因为 Router 相信这个请求快释放了。

OSL ****不影响**** `router_temperature`、`overlap_score_credit_decay`、`prefill_load_scale`、softmax 等其他打分参数——只走 `active_decode_blocks` 这一条路径。

#### 6.9.6 一处需要小心的实现细节

﻿`ActiveSequences::RequestState` 里也存了一份 `expected_output_tokens: Option<u32>`（`single.rs:49`），但 `free()` 里有一行 `let _ = request_state.expected_output_tokens;`——****这份存储在 tracker 里没被读回使用****。

真正驱动 fractional 计算的是 `****RequestGuard::OutputBlockTracker.expected_output_tokens****`（`request_guard.rs:237`）。也就是说 OSL 在系统里有****两份 owner****：

- 数据面（`RequestState`）：只是被动接收，不驱动计算；
    
- 请求生命周期 guard（`RequestGuard`）：真正做 decay 计算的位置。
    

这是 Dynamo 一处架构上"两份状态但只用一份"的设计——理解 OSL 机制时要跟着 `RequestGuard` 那条链路，不是 `RequestState`。

#### 6.9.7 为什么这个机制默认不开？

﻿`router_track_output_blocks=false` 是默认，官方文档没写理由。合理推测：

1. ****依赖客户端合作****：只有客户端主动传 OSL 才有信号；否则 output block 只能按满值兜底（相当于假设"生成到永远"），这个假设本身不见得比"完全不追踪 output"更接近现实。
    
2. ****实现代价****：每 block 边界触发一次 `add_output_block` + `set_unique_suffix_fractional`，后者要沿 prompt trie 往上走——虽然 O(depth) 不重，但增加了 hot path 的开销。
    
3. ****场景独特性****：这个特性对 agentic RL rollout（客户端知道 max_tokens 大概是多少）或 tool-calling（步骤输出长度可预估）特别有用；对通用聊天场景收益有限（用户不会传 OSL）。
    

#### 6.9.8 行业对比：其他框架有没有 OSL-aware scheduling？

Dynamo 的 `nvext.agent_hints.osl` 在业界属于****少见的显式 client-hint 设计****——事实上截至 2026-07-21 的调研快照，它是****唯一在主分支上生产可用的****开源实现。做过一轮跨框架调研，把主流 LLM inference / agentic RL rollout 栈的 OSL-aware 调度支持情况整理如下。

****其他框架现状表（截至 2026-07-21）****：

|   |   |   |
|---|---|---|
|框架|OSL / max_tokens-aware 调度？|关键证据|
|****Dynamo****|✅****主分支生产可用****（client hint）|`nvext.agent_hints.osl`（本节各小节详述）|
|****verl****|⚠️****仅社区讨论，未合入主分支****|见下方"verl 详情"小节|
|SGLang / sgl-router|❌|策略只有`random / round_robin / power_of_two / cache_aware / bucket`，都不看 output length|
|vLLM|❌|相关 issue 只涉及 speculative decoding 内部的自适应，不是 user OSL|
|TensorRT-LLM|❌（未找到证据）|Executor 文档只提`max_tokens` 作为上限，未参与调度|
|Mooncake|❌（论文明确规避）|原文："predicting each request's output length is challenging due to high costs or low accuracy"|
|DistServe|❌|论文未涉及 output length prediction|
|AIBrix|❌|`RoutingContext` 里都没有 `max_tokens` 字段|

  

****verl 详情****：verl 是所有主流开源框架里****唯一为 agentic RL rollout 场景发起过 OSL-aware 调度社区讨论****的项目，但截至 2026-07-21，****相关工作没有一个真正合入主分支****——这个精确状态很关键，之前一些一手材料的表述容易让人误以为 verl 已经落地了，实际未然。

|   |   |   |   |
|---|---|---|---|
|PR|主要思想（大白话解释）|状态|合入主分支？|
|[#2629](https://github.com/volcengine/verl/pull/2629) `even_token` Request Skewness Scheduler|****让每个 DP rank 分到差不多多的"要生成的 token 数"****。RL rollout 一个 batch 里可能有的 prompt 只生成 100 tokens，有的生成 5000 tokens——直接按请求数平均分到不同 DP rank 上，那分到长请求的 rank 会拖住整个 batch（最快和最慢的差距能到 1800 秒以上）。这个 PR 的想法是：****Router 记住每个 prompt 上一次 rollout 实际生成了多少 tokens****，下次分配时把长的和短的搭配起来分到各 rank 上，让每个 rank 的总 token 数尽量相等。****不训预测模型，就查历史查询表****（作者原文明说是为了"engineering simplicity"），冷启动用 offline 估计兜底。分区算法有两种：`even_token`（贪心装箱）和 `even_token_kk`（Karmarkar-Karp 差分）。|****Open****，但 ****stalled****（作者最后活动 2025-08-15 之后近一年无维护者回复）|❌|
|[#2200](https://github.com/volcengine/verl/pull/2200) `reorder_rollout` StreamScheduler|****不预测长度，用"opportunistic streaming"绕开长尾****。思路是：Rollout 时不等所有样本都生成完再算 loss，而是****短样本先"出锅"塞进 batch，长样本被后到的短样本挤到后面****。具体做法是所有请求都发到 engine，但当 batch 被后来的短样本填满后，长样本还没生成完的部分就被****丢弃****（drop）——反正 RL 训练一般是采样一批就够了，不必等长尾。这样避免了"必须等最慢样本生成完"的等待时间。****回避预测，用调度技巧绕开长尾问题****。|****Closed****（2025-10-13，作者自己关闭）|❌|
|[#2981](https://github.com/volcengine/verl/pull/2981) fully async training recipe|****这个 PR 其实和 OSL-aware scheduling 无关****——只是为后续的 partial rollout / async 特性铺路。核心工作是：****把 Trainer（做梯度更新）和 Rollouter（生成样本）解耦成两个独立进程****，中间用 CPU message queue 通信，加 ParameterSynchronizer 做 NCCL 参数同步。合入后 Trainer 和 Rollouter 可以异步工作——Trainer 在训练某个 batch 时，Rollouter 已经在准备下一个 batch 的样本了，不再需要串行等待。****partial rollout / length-aware 调度是留给后续 PR 的 future work，本 PR 不实现****。|****Merged****（2025-10-17）|✅ 但****不实现 OSL-aware****——只做了 Trainer/Rollouter 解耦的基础设施|

  

****三个 PR 反映的社区取舍****：verl 团队思考过 OSL-aware scheduling 的两条不同技术路线——****历史统计****（#2629）和****回避预测****（#2200）——但****两条路线都没有合入主分支****。最终合入的 #2981 只提供了"async 基础设施"，把"是否要在这基础上做 length-aware 调度"作为 future work 留给后续 PR。截至 2026-07-21，****后续 PR 也没有真正落地这类功能****（相关的 #3955、#4023 都是 importance sampling，与 length-aware 调度无关）。

****关键 timeline 与"未合入原因"****：

****PR #2629——维护者集体沉默，非架构否决****

- 提出 `even_token` load-balancing 策略。原文："ensure that the total number of generated tokens handled by each DP instance within a batch is approximately equal"。
    
- 关键设计选择原文："We have opted for a more direct statistical method for its engineering simplicity, ****rather than training a separate prediction model****." 使用"a lookup table that records the actual response length for each request from the previous episode"作为估计依据。
    
- ****未合入的准确原因****是"社区搁置"，****不是任何形式的拒绝****——查证 GitHub API `/pulls/2629/reviews`：
    
    - ****所有 6 条 formal review 都是**** `****state: COMMENTED****`，无 `REQUEST_CHANGES`、无 `APPROVED`；
        
    - 4 位 code owner（`wuxibin89 / PeterSH6 / chenhaiq / zw0610`）****至今全部处于 "Awaiting requested review" 状态****——没有一个 code owner 出具正式 review 结论；
        
    - 讨论区里唯一有架构层面 pushback 语气的是 wuxibin89 一次问题："If a prompt tend to have long response, should we evenly distribute this prompt across DP ranks instead of repeat it in a single DP?"——****作者回答后 wuxibin89 语气接受地回了"OK, so..."，之后不再回复****；
        
    - chenhaiq 明确表达接受方向："@Tyizhanshen This is a good idea! There is one more thing you may need to consider..."（然后再没回复过）；
        
    - eric-haibin-lin 只留了 6 条 code nit（挪 env vars、去 debug print、加 inline 注释、把 `ReqScheduler` 拆独立文件等），****没有一条是架构反对****；作者 2025-08-04 push 修改后 eric-haibin-lin 再没回复；
        
    - 作者 2025-08-15 最后一条 ping "Could a maintainer please take a look? @eric-haibin-lin @wuxibin89 @chenhaiq"——****至今近一年无任何维护者回复****。
        
- ****一处关键澄清****：`#2629` 的代码从一开始就写在 `recipe/req_sched/` 目录下，****社区对 "OSL-aware 应该走 recipe 不进主 trainer" 的政策（见 #2200 那条）在这里已经满足了****，所以不存在"这条应该走 recipe"的策略性拒绝。它就是****没人管****。
    

****PR #2200——维护者明确定调"走 recipe 不进主 trainer"****

- 用 StreamScheduler 做 opportunistic streaming（短样本先出、长样本被 filler 挤后）。
    
- 关闭原因：****作者自己关闭，但****关闭前维护者 chenhaiq 有过明确定调（原文）："as we perviously discussed with haibin, the streaming partial-rollout feature need to be placed ****as a recipe**** to keep the main trainer as simple as possible"——****这是 verl 主线社区对"streaming partial-rollout"特定方向的明确政策****：不进主 trainer，只允许作为 recipe。
    
- ****这条政策只针对 #2200 的具体做法****（streaming + drop long-tail），****不适用于 #2629****（#2629 本身就是 recipe，遵守了这条政策）。
    

****PR #2981——与 OSL-aware 无关，只是基础设施****

- Merged 2025-10-17，但它做的****不是 OSL-aware scheduling****——是"Trainer 和 Rollouter 解耦为两个组件、CPU message queue 通信、ParameterSynchronizer NCCL 同步"的基础设施改造。PR 描述明确写"partial rollout 是 future work"。
    

****综合判断——两个 PR 未合入的真实原因不同****：

|   |   |   |
|---|---|---|
|PR|未合入的真实原因|是否有反对意见|
|#2629|****社区搁置****（无维护者响应），非架构否决|❌ 没有反对；只有作者未回复的 code nit|
|#2200|****社区政策****（streaming partial-rollout 只能走 recipe）|⚠️ 有明确政策定调，但没有"技术反对"|

  

无论哪种原因，最终都是****verl 主分支上没有 OSL-aware scheduling 的实际实现****。但这背后的社区动态并不是"verl 团队认为这个方向不对"——****没有任何证据支持这个说法****。可能的合理解释：verl 项目的 rollout / trainer 架构改动 review 成本高，同期竞争的 PR 众多，简单的"没人跟进"就足够让一个技术上没问题的 PR 无限期 stalled。

****verl 主分支实际存在的调度策略****（今天真实运行的代码，不是 PR 讨论）：

- ****同步路径****：`rollout until batch is complete, then train`——批同步等待，无 request-level 调度；
    
- ****Async server 路径****（`verl/workers/rollout/llm_server.py`）：`GlobalRequestLoadBalancer`——****least in-flight requests**** 负载均衡，按当前在飞请求数最少者选副本，****不看 output length****；
    
- ****fully async 路径****（PR #2981 合入的 `verl/experimental/fully_async_policy/`）：Rollouter one-by-one streaming + `staleness_threshold` 控 freshness，****同样不 length-aware****；
    
- ﻿`****recipe/partial_rollout/****`（submodule）：README 原文自陈 ****"not FCFS and not an output-length-aware reorder scheduler"****，走的是 trajectory-grained pull-pacing + 跨 step interrupt/resume 的路线。
    

****逐一 grep 主分支的关键字命中情况****（都不存在）：

|   |   |
|---|---|
|关键字|主分支命中|
|`even_token` / `RequestScheduler` / `ReqScheduler`|❌|
|`skewness`|❌|
|`StreamScheduler` / `reorder_rollout`|❌|
|`load_table`|❌|

  

所以严格说，****verl 目前没有任何 OSL-aware 或 output-length-aware 的调度逻辑真正在主分支上运行****。这个技术方向在 verl 属于"社区讨论过但被明确拒绝进主 trainer、被推给 recipe"的状态。

****如果 verl 真的合入了 OSL-aware scheduling，会是什么设计路线？****（对照 PR #2629 已经明确说的取舍）：

- 依据：****历史 episode 的实际 response length****（lookup table，非预测模型）
    
- 优势场景：RL rollout（同 prompt 反复采样，历史数据充足）
    
- 冷启动：需 offline `load_table` 估计兜底
    
- 影响调度的方式：Request skewness scheduling（DP rank 间均匀 partition token 数）
    

这与 Dynamo `nvext.agent_hints.osl`（client hint + fractional decay of output block）是两条完全不同的路线。****但目前只有 Dynamo 的路线真正生产可用****。

****学术上的输出长度预测研究****（相对成熟的一条路线，但工业落地为零）：

|   |   |   |   |
|---|---|---|---|
|论文|arXiv|核心思路|效果|
|****Response Length Perception and Sequence Scheduling****（Zheng et al., NeurIPS 2023）|[2305.13144](https://arxiv.org/abs/2305.13144)|用 LLM 自己感知 response length：__"tapping into the potential of LLMs to accurately perceive and predict the response length with minimal overhead"__|86% throughput 提升|
|****S³****（Jin et al., 2023）|[2306.06000](https://arxiv.org/abs/2306.06000)|强调 "designing a system with a priori knowledge of the output sequence can mitigate this problem"，配备 misprediction recovery 机制|6.49× throughput 提升|
|****SSJF****（Qiu et al., 2024）|[2404.08509](https://arxiv.org/abs/2404.08509)|用****小 proxy 模型****预测长度：__"a speculative shortest-job-first scheduler that uses a light proxy model to predict LLM output sequence lengths"__|JCT 降低 30.5-39.6%，throughput 提升 2.2-3.6×|
|****Learning to Rank Scheduling****（Fu et al., 2024）|[2408.15792](https://arxiv.org/abs/2408.15792)|****反直觉****：__"predicting the exact generation length of each request is infeasible, it is possible to predict the relative ranks"__ ——只学 pairwise ranking，规避绝对长度预测|2.8× 延迟改善，6.5× throughput 提升|

  

****关键行业观察****：

1. ****学术与工业之间存在巨大 gap****：SSJF 和 Learning-to-Rank 都是工程可行的方案，但****没有一个主流开源框架真正在主分支上采纳****。可能的原因是：预测模型引入额外部署复杂度、模型可能预测错误导致体验退化、通用 API 服务里预测收益不如"客户端明说"来得直接。verl 社区就明确讨论过并****主动拒绝****了这个方向（保持主 trainer 简洁）。
    
2. ****三种候选路线的落地成熟度****：
    

- - ****Client hint****（Dynamo）→ 主分支生产可用；需要客户端合作，简单可靠但覆盖率低。
        
    - ****历史统计****（verl 讨论中）→ 未合入主分支，仅 recipe 层面探索；需要请求可复现（RL 场景）。
        
    - ****在线预测****（学术界）→ 无工业落地；精度高但落地成本高。
        

1. ****Mooncake 的明确规避**** 提供了一个反面证据：它论文里详细论证了 per-request OSL prediction "high costs or low accuracy, especially under overload conditions"——****Mooncake 团队选择做系统级 aggregate workload prediction，绕开 per-request 预测****。这一定程度上支持了"per-request 预测在生产上并不成熟"这个判断。
    
2. ****Tool-calling / structured output 场景是明显空白****：这类场景输出长度天然可预估（JSON schema、function signature 有强约束），但没有找到任何专门针对此的调度或预测工作。是一个值得关注的机会点。
    

****回到 Dynamo 的定位****：`nvext.agent_hints.osl` 是****所有主流开源栈里唯一在主分支上生产可用、支持 per-request OSL 显式 hint 且直接反映在打分公式的机制****。它选择了"最简单可靠"的 client-hint 路线——不训模型、不维护 lookup table、不做在线预测，把预测责任推给客户端。代价是覆盖率低（客户端要主动传）、默认关闭（生产上大多数用户根本不知道有这个特性）。但对于知道自己 workload 的调用方（如 agentic RL rollout、tool-calling 客户端），这个机制的接入成本几乎为零——一个 JSON 字段而已。

****如果要基于本文做 Dynamo 的实际落地评估****，可以从以下几个角度对比：

- 目标场景是否是 agentic RL rollout 或 tool-calling → 是则 Dynamo OSL hint 值得开启；
    
- 目标场景是否有"同请求多次 rollout"的特性 → 是则可以自建历史统计路线（参考 verl PR #2629 的思路，但注意 verl 自己都还没合入主分支）；
    
- 目标场景是否是通用聊天 API → 客户端不会传 OSL，此机制近似无效。
    

### 6.10 一个具体的打分例子（对齐 verl §7 的展示风格）

#### 6.10.1 请求级别的例子

设有 3 个候选 worker A/B/C，请求 96 tokens、block_size=16、`overlap_score_credit=1.0` / `host_cache_hit_weight=0.75` / `disk_cache_hit_weight=0.25` / `prefill_load_scale=1.0`：

raw_prefill_blocks = 96 / 16 = 6

各 worker 状态：

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|worker|device_overlap|host_overlap|disk_overlap|active_prefill_tokens|decode_blocks|
|A|5|0|0|0|9|
|B|2|3|0|0|4|
|C|0|0|6|0|1|

  

逐个算：

A: overlap_credit_blocks = 1.0*5 + 0.75*0 + 0.25*0 = 5

   adjusted_prefill_blocks = max(6 - 5, 0) = 1

   cost = 1.0 * 1 + 9 = 10

B: overlap_credit_blocks = 1.0*2 + 0.75*3 + 0.25*0 = 4.25

   adjusted_prefill_blocks = max(6 - 4.25, 0) = 1.75

   cost = 1.0 * 1.75 + 4 = 5.75

C: overlap_credit_blocks = 1.0*0 + 0.75*0 + 0.25*6 = 1.5

   adjusted_prefill_blocks = max(6 - 1.5, 0) = 4.5

   cost = 1.0 * 4.5 + 1 = 5.5

****argmin = C****——虽然 A 的 device overlap 最高（5 blocks），但它 decode backlog 也最高（9 blocks）；C 的 overlap 全在 disk（权重仅 0.25，恢复慢），但因为几乎完全空闲，反而是综合最优。B 是 device+host 混合命中的折中选择。

****这个例子说明的两个直觉****：

1. host / disk credit 权重设置（0.75 / 0.25）反映了"读得越慢，credit 越低"的物理事实；
    
2. cost function 不会僵硬地"KV 命中最多的赢"——负载是同权项，能被 cache 收益完全逆转。
    

#### 6.10.2 block级别的例子

以下是更加具体的不同类型的block在最终公式中的总权。实际时并不是每一个block都走一遍判断逻辑其到底是什么类别的，而是Router维护了几个相关变量，加起来就行，这里主要是为了展示细粒度。这些block类型是本文根据最终权重占比的不同而分类出来的，源代码中没有显示的对应类型。

Type-2a分裂成两项的目的是“虽然cache命中了，但是这个block有没有其他已经在worker上的请求正在占用，来避免右项 (memory)的重复计算”，这也是overlap_depth的含义：“”

|   |   |   |   |   |
|---|---|---|---|---|
|Block 类型|描述|左项 (compute)|右项 (memory)|总权重|
|****Type 1a****|其他 live 请求持有 + 该 block 对应的 prefill 尚未算完|+1（通过 `active_prefill_tokens` 混入 `raw_prefill_tokens`）|+1（计入 `active_decode_blocks`）|****+2****|
|****Type 1b****|其他 live 请求持有 + 该 block 已算完（prefill 完成或 decode 中）|0（`mark_prefill_completed` 后 tokens 从 prefill_tracker 移除）|+1（计入 `active_decode_blocks`）|****+1****|
|****Type 2a-trie****|R_new 命中在 device + 该 prefix ****仍被某个 live 请求持有着****|+1 − 1.0×1 = ****0****|被 `overlap_depth` 减掉：****0****|****0****|
|****Type 2a-idle****|R_new 命中在 device + 持有者已 `free()`（trie 未命中）|+1 − 1.0×1 = ****0****|未在 `overlap_depth` 里减：****+1****|****+1****|
|****Type 2b****|R_new 命中在 host 上|+1 − 0.75×1 = ****+0.25****|未在 `overlap_depth` 里减：****+1****|****+1.25****|
|****Type 2c****|R_new 命中在 disk 上|+1 − 0.25×1 = ****+0.75****|未在 `overlap_depth` 里减：****+1****|****+1.75****|
|****Type 2d****|R_new 只在 shared pool 命中|+1 − 0.5×1 = ****+0.5****|未在 `overlap_depth` 里减：****+1****|****+1.5****|
|****Type 3****|R_new 未命中任何 tier|+1|+1|****+2****|

  

### 6.11 `decode_blocks` 精确语义：为什么叫 "decode" 却不只是 decode 阶段的 block

这一小节展开公式右项，因为它是打分公式里****最容易望文生义的变量****。

****源码定义****（`lib/kv-router/src/sequences/prompt_registry.rs :: WorkerLoadProjection`）：

pub struct WorkerLoadProjection {

    pub active_prefill_tokens: usize,

    pub active_decode_blocks: usize,

    /// Request blocks not already shared with active sequences on this worker.

    pub additional_active_blocks: usize,

}

impl WorkerLoadProjection {

    pub fn potential_decode_blocks(self) -> usize {

        self.active_decode_blocks + self.additional_active_blocks

    }

}

打分公式里的 `decode_blocks` = `potential_decode_blocks()` = 两个分量之和。逐个拆开：

#### 6.11.1 分量一：`active_decode_blocks`（现有请求占用）

字段来源：`load.active_blocks`。语义来自 `WorkerLoadSnapshot` 的 docstring 原文：

> "`active_blocks` is the worker's ****unique active decode load**** in blocks."

****关键观察****：这个字段的更新时机在 `lib/kv-router/src/sequences/single.rs` 里明确——

- 请求****加入****时（`add_request_with_prefill_tracking`）就获取 blocks；
    
- 请求 prefill 完成时（`mark_prefill_completed`）****不释放 blocks****——只把请求从 `self.prefill`（prefill-token 追踪结构）里移除，`self.blocks` 完全不动；
    
- 请求真正****结束****时（`free()`）才释放 blocks。
    

也就是说，`active_blocks` 包含****该 worker 上所有活着的请求****（prefill 期的 + decode 期的）在其整个生命周期内占用的 KV block。****它不是"筛出正在 decode 的请求"，而是"所有 live 请求的 block 总占用"****。

****那为什么叫 "decode"？**** 因为 KV cache 的****持续占用****主要发生在 decode 阶段——prefill 完成后 KV block 一直保留直到请求结束，是 decode 阶段真正让 KV cache "满" 的原因。历史命名沿袭下来叫 `active_decode_blocks`，但语义****不是****"筛选阶段"，而是"衡量 KV cache 的 memory footprint"。

#### 6.11.2 分量二：`additional_active_blocks`（新请求预留）——它是怎么算出来的？

字段计算（`lib/kv-router/src/sequences/prompt_registry.rs :: project_worker_loads`）：

additional_active_blocks: query_len.saturating_sub(overlap_depth),

这不是概率意义上的"预测"，是****确定性减法****。两个变量：

- ﻿`****query_len****`：本次新请求对应的 block 总数。请求进入 Router 时 tokenize 已完成，token 序列长度已知，`query_len = token_count / block_size`（默认 `block_size=16`）。****这是事实，不是猜测****——240 tokens 就是 15 blocks，写死的。
    
- ﻿`****overlap_depth****`：本次请求的 prefix 在该 worker 上已经缓存了多少 block。来自 Step 1 里 Indexer 查询的结果（`matched_depth` 参数由 scheduler 从 Indexer 结果里传入 `project_worker_loads`）。****这也是事实****——Router 内部有全局 KV block 索引（§4-5 详述），查一下就有答案。
    

****减法的目的：避免与**** `****active_decode_blocks****` ****double-count****。想想 `decode_blocks = active_decode_blocks + additional_active_blocks`：如果新请求的 prefix 已经在该 worker 上被某个 live 请求持有着（即 `overlap_depth > 0`），那这些 block 已经在 `active_decode_blocks` 里计过一次；如果 `additional_active_blocks` 简单等于 `query_len`，同一批 block 会被算两次。减掉 `overlap_depth` 保证"每个 block 只在 `decode_blocks` 里出现一次"，这个量真实反映"派发后该 worker 上的唯一 block 总数"。

****举例****：一个 15-block 的请求，前 6 blocks 在 worker A 上已经命中（`overlap_depth=6`）：

additional_active_blocks = 15 − 6 = 9

意思是：****如果****把这个请求发给 A，A 需要再分配 9 个 block 的新 KV 空间；前 6 blocks 直接复用已有的，不需要新分配。

****一处确实带"预估"味道的地方****：这个计算****只覆盖 prompt 侧****，不包含 decode 阶段 output tokens 会持续增长带来的 KV 占用。因为 Router 无从知道请求会生成多长——除非客户端主动告知。Dynamo 通过两个可选机制近似 output 侧的未来负担：

- ﻿`****router_track_output_blocks****`（默认 `false`）：启用后，output blocks 也会被计入 `decode_blocks`。因为不知道会生成多久，默认按****满值计****（假设一直在生成，直到请求结束）。
    
- ﻿`****nvext.agent_hints.osl****`（Output Sequence Length hint）：客户端主动告知预期输出长度。Router 会按当前生成进度对 output blocks 做****分数衰减****（比如已生成 80%，output 项的权重打到 20%——很快要释放）。
    

官方原话（`docs/components/router/router-concepts.md`）：

> "If the request includes `nvext.agent_hints.osl`, those output blocks receive a fractional weight based on progress toward the expected output length."  
> "Without an expected OSL, tracked output blocks count at full weight until the request finishes."

****一句话总结****：`additional_active_blocks` 是****从 prompt 长度 + Indexer 命中数直接算出来的确定值****（不是概率预测），它衡量"派过去后该 worker 的 memory 净增量"。默认公式****只覆盖 prompt 侧****；如果想让打分也感知未来 decode 增长，需要开启 `router_track_output_blocks` 并最好带 OSL hint。

#### 6.11.3 `potential_decode_blocks` 的总语义

组合起来：

potential_decode_blocks = 该 worker 现有所有 live 请求占的 blocks

                       + 如果把新请求发过去还要额外占的 blocks

                       = "把新请求发过去之后，该 worker 的 KV cache 总占用"

它衡量的是 ****memory footprint****（内存占用），不是"正在做 decode 的请求数"。

#### 6.11.4 澄清用户可能有的三个疑问

****Q：这是 P/D 分离的产物吗？"decode" 项是不是来自 decode worker？****

不是。打分公式在 ****aggregated 部署****（prefill+decode 同一 worker）下完全一样。P/D 分离下 decode worker 只是通过 `RouterConfigOverride`（`overlap_score_credit=0 / track_prefill_tokens=false`）把左项退化，公式****结构没变****——两项在同一次打分里同时出现，不是"左项来自 P 节点、右项来自 D 节点"。

****Q：可以理解为"当前 worker 的 running 队列里的请求占用的 block 数量"吗？****

方向对但不精确。更准确的说法：

- 不只是"running"，还包括****所有已经加入但未结束的请求****（含正在排队 prefill 的、正在做 prefill 的、正在做 decode 的、已经完成生成但还没释放的）；
    
- ****加上****这次新请求发过去后****将会新占用****的 block（`additional_active_blocks`）。
    

****Q：为什么公式两项分别叫 "prefill" 和 "decode"？打分难道要筛一遍正在 decode 的请求？****

****不筛****。两项名字反映的是****成本维度****，不是请求阶段：

|   |   |   |
|---|---|---|
|项|名字含义|衡量什么|
|`prefill_load_scale * adjusted_prefill_blocks`|****prefill-side cost****|这个新请求还需要做多少****prefill 计算工作****（cache-credit 减完后剩余的）|
|`decode_blocks`|****decode-side cost****|该 worker 的****KV cache 内存有多满****（现有请求占用 + 新请求预留）|

  

一个是****计算成本****（compute），一个是****内存成本****（memory）。KV 存储的持续占用主要来自 decode 阶段所以叫 decode_blocks，KV 计算工作主要来自 prefill 阶段所以叫 prefill_blocks——但两个 metric 在打分时都不做"按阶段筛选"。

#### 6.11.5 一个直觉例子

设 Worker X 上：

- 请求 R1 正在做 prefill（占 20 blocks）
    
- 请求 R2 正在做 decode（占 30 blocks）
    
- 请求 R3 生成完但还没释放（占 15 blocks）
    

新请求 R_new 要 65 blocks，与已有序列 prefix 重合 5 blocks（`overlap_depth=5`，`query_len=65`）。

那么：

active_decode_blocks       = 20 + 30 + 15 = 65        # R1 + R2 + R3 全算，不筛阶段

additional_active_blocks   = 65 - 5 = 60

potential_decode_blocks    = 65 + 60 = 125

decode_blocks (in cost)    = 125

****注意****：R1 明明在做 prefill，它的 20 blocks 也算进了叫 "decode_blocks" 的量里——这就是"命名沿袭 vs 实际语义"造成的困惑来源。

### 6.12 关于打分公式的两个进一步思考

#### 6.12.1 两项的物理意义：compute cost vs memory cost

回到 `cost = w · prefill_blocks + decode_blocks` 这个双项加和的本质：

- ****左项****回答的是："****如果****我把请求发给这个 worker，它还要给我算多少 prefill 工作？" ——包含"新请求本身的 uncached 部分"（`raw_prefill_blocks - overlap_credit`）以及"该 worker 上其他人还没算完的 prefill 排队"（通过 `raw_prefill_tokens = active_prefill_tokens + uncached_tokens` 混入）。****是 compute-side 的净负担****。
    
- ****右项****回答的是："这个 worker 的 KV cache 已经有多满？发过去会不会没地方放？" ——是所有 live 请求（不分阶段）的 memory footprint 之和 + 新请求预留。****是 memory-side 的净占用****。
    

Cost 越小 → compute 剩余工作少 ****且**** memory 剩余空间大 → 综合最优。这是"计算与内存两个维度****同权融合****"的表达，不是"cache 命中 + 负载"的表达（cache 命中只在左项里体现为减项）。

#### 6.12.2 为什么这个公式很聪明：负载不会退化为 prefill vs decode 二选一

一个直觉的替代方案是"分别看 prefill worker 的负载和 decode worker 的负载"，但 Dynamo 没这么做。理由是 aggregated 部署下****同一个 worker 既做 prefill 又做 decode****，两种成本会互相竞争 GPU 资源，把它们放同一个 cost function 里同权相加，是最直接的"综合成本"表达。

而 P/D 分离部署下，decode worker 侧通过 `overlap_score_credit=0 + track_prefill_tokens=false` 让左项退化——`raw_prefill_blocks = 请求 ISL 转 blocks`（不做 cache 减法、不看历史 prefill 排队）——****这时左项含义变成"这个新请求本身有多长（未来会占多少 memory）"****，与右项的"现有 memory 占用"相加，等价于"发过去后该 worker 的总 memory 压力"。同一份公式在两种部署下都合理。

### 6.13 Tier priority 去重的真实实现：`query_lower_tiers` 的 continuation offset

§6.1 的公式里 `overlap_credit = 1.0·device + 0.75·host + 0.25·disk + shared_beyond` 三项直接相加。第一眼看会担心："如果一个 block 在 GPU 和 host 上同时有（HiCache `write_through` 稳态下就是这样），岂不是 device 和 host 各算一次，double-count 收益？"

****这个担心在 Dynamo 里不成立****，因为****去重不发生在 selector.rs 层，而是在更上游的 Indexer 层，通过一个"per-worker continuation offset"机制实现****。真实实现比官方文档 "picks the highest-priority tier" 的措辞更精妙。

#### 6.13.1 关键代码位置

- ﻿`****tier_overlap_blocks****` ****结构定义****：`lib/kv-router/src/scheduling/mod.rs`，三个 `HashMap<WorkerWithDpRank, usize>`：`device` / `host_pinned` / `disk`。
    
- ****Scheduling 层的构造函数****：`lib/kv-router/src/scheduling/overlap.rs :: tier_overlap_blocks_from_tiered_matches` —— ****这个函数本身不去重****，只把上游 `TieredMatchDetails` 的各 tier `hits` 原样拷贝到三个 map。
    
- ****真正的去重发生在****：`lib/kv-router/src/indexer/lower_tier_indexers.rs :: query_lower_tiers` —— 通过 `LowerTierContinuation` 机制实现。
    

#### 6.13.2 Continuation offset 机制（大白话）

想象某个 worker A 的 KV cache 里，一段 prefix 的分布是这样：

block 序号:  0   1   2   3   4   5   6   7   8

在哪一层:  GPU GPU GPU GPU host host disk disk disk

也就是说，前 4 个 block 在 GPU，接着 2 个只在 host，再接着 3 个只在 disk。

****朴素的去重方式****（Dynamo 没这么做）：分别在 GPU/host/disk 集合里查匹配数，然后做集合差集：`host_only = host_matches - device_matches`。这需要维护集合结构、做差集运算。

****Dynamo 实际的做法****（continuation offset）：

Step 1: 在 device tier（GPU）沿 prefix 走，走到 block 3 走不动了（block 4 不在 GPU 上）

        → 记录：device 匹配 4 个 blocks，continuation = (matched=4, last_hash=block3_hash)

Step 2: 拿着 continuation 去 host tier，**从 block 4 开始继续走**（不是从头开始）

        → 走到 block 5 走不动了（block 6 不在 host 上）

        → 记录：host 匹配 2 个 blocks（只算 4-5，不重复算 0-3）

Step 3: 拿着 continuation 去 disk tier，从 block 6 开始继续走

        → 走到 block 8

        → 记录：disk 匹配 3 个 blocks

****结果****：`device=4, host=2, disk=3`，三者****天然互斥****，加起来正好是这段 prefix 的总覆盖（4+2+3=9 blocks）。没有集合差集，也没有逐 block 判定——****只是让每个下层 tier 从上层"结束的位置继续走"****。

﻿`LowerTierContinuation` 结构体携带的就是每个 worker 的 `(matched_count, last_matched_hash)`，作为下一层查询的起点参数。

#### 6.13.3 官方回归测试证据

﻿`lib/llm/src/kv_router/indexer/mod.rs` 有一个专门测试：

concurrent_tiered_query_does_not_double_count_device_and_lower_tier_overlap

测试注释原文：__"when a worker has blocks in both device and lower-tier storage ... Without the fix,__ `__query_lower_tiers__` __would re-query that worker from root in the lower tier, double-counting overlap blocks."__ 断言"同一 worker 的 device=3、host_pinned=0"（这个 worker 在 GPU 已经有 3 个 block 全部命中，host 层就不再重复计入）。

****这个测试的存在说明两件事****：

1. Dynamo 团队意识到 double-count 是个真实存在的风险；
    
2. Continuation offset 机制是****明确为了避免 double-count 而设计的****，不是偶然行为。
    

#### 6.13.4 结论：`selector.rs` 三项相加是安全的

前提是上游 `query_lower_tiers` 正常工作（这条链路有专门的回归测试守护）。

﻿`tier_overlap_blocks.device[w]` / `host_pinned[w]` / `disk[w]` 三个数字是****每个 worker 的 tier-exclusive 块数****——同一个 block ****只会出现在恰好一个字段里****（它所在的最高优先级 tier）。selector.rs 直接把它们乘上各自权重再相加，不会 double-count。

#### 6.13.5 但官方文档的措辞需要修正

官方文档 `docs/backends/sglang/sglang-hicache.md` line 76 那句 __"picks the highest-priority tier when scoring overlap"__ 的措辞实际上****不够精确****。真实语义是：

- ****不是**** "每次打分时从多个 tier 里挑一个最高优先级的算"（听起来像 selector 侧的行为）；
    
- ****而是**** "构造 tier_overlap_blocks 时就把每一块严格归到最高优先级的 tier，selector 拿到的三个 map 已经是互斥的"。
    

语义等价，但用词让人误以为 dedup 逻辑在 scoring 时才发生。实际代码是在 scoring 之前（Indexer 层）就已经做好了。

#### 6.13.6 附加：Response 层有一个不同的"累加视图"（不要混淆）

﻿`OverlapSignals::selected_worker_tiers`（用于 metrics/response 展示）****做了一次累加叠加****：

host_pinned_blocks = device_blocks + host_pinned_extension_blocks

disk_blocks        = host_pinned_blocks + disk_extension_blocks

保证 `gpu_blocks ≤ host_pinned_blocks ≤ disk_blocks`（accumulate 视图）。`build_overlap_scores_response` 里区分 `host_pinned_extension_blocks`（extension = 互斥的原始值）与 `host_pinned_blocks`（累加值）。

****注意****：这个累加只用于展示，`selector.rs :: worker_logit` 拿的是****原始互斥值****（extension），不是累加值。所以 scoring 与 response 两条视图各行其是，不冲突。

### 6.14 `shared_beyond_blocks` 只减 device，与 host/disk 之间存在 double-count 风险

§6.13 讲清楚了 device / host / disk 三个 tier 之间是 tier-exclusive 互斥（Indexer 层通过 continuation offset 保证）。****但 shared credit 与 host / disk credit 之间没有类似机制****，只减 device——这与 tier 内部去重是****两条完全独立的链路****。这一节展开讨论这个不对称设计与它带来的实际问题。

#### 6.14.1 源码事实

﻿`shared_beyond_blocks` 由 `SharedCacheHits::hits_beyond(from_position: u32)` 计算，定义在 `lib/kv-router/src/protocols.rs`：

pub struct SharedCacheHits {

    pub ranges: Vec<Range<u32>>,   // 半开区间 [start, end)，已排序、不重叠

    pub total_hits: u32,

}

pub fn hits_beyond(&self, from_position: u32) -> u32 {

    self.ranges.iter().map(|r| {

        if r.end <= from_position { 0 }

        else if r.start >= from_position { r.end - r.start }

        else { r.end - from_position }

    }).sum()

}

****关键观察****：`hits_beyond` 只按****位置游标****（`from_position: u32`）切割 range，****完全不知道 host / disk 是否有对应命中****。

在 `selector.rs :: worker_logit` 里的调用点：

let beyond = shared_hits.hits_beyond(device_overlap_blocks_u32);

传进去的位置游标是 `****device_overlap_blocks****`，不是 `device + host + disk` 之和。所以：

- ✅ 减了 device 部分（避免与 device credit 重复）；
    
- ❌ ****没减 host 部分****；
    
- ❌ ****没减 disk 部分****。
    

#### 6.14.2 一个具体的 double-count 例子

假设某个请求的 prefix 有 20 个 block，一个候选 worker A 的分布是：

block 位置:  0  1  2  3  4 | 5  6  7 | 8  9  ... 19

Worker A:   device × 5    | host × 3 | (未命中)

Shared pool: (整段 20 blocks 都在 shared pool 里)

计算：

- ﻿`device_overlap_blocks = 5`﻿
    
- ﻿`host_overlap_blocks = 3`（Indexer 已经在 continuation offset 里剔除了 device 命中的 0-4，只算 5-7）
    
- ﻿`disk_overlap_blocks = 0`﻿
    
- ﻿`shared_beyond = hits_beyond(5) = 15`（位置 5-19 的所有 shared 命中，****包含 host 上也有的 5-7****）
    

带入公式：

overlap_credit = 1.0 * 5     # device

               + 0.75 * 3    # host (block 5-7)

               + 0.25 * 0

               + 0.5 * 15    # shared, 包含 block 5-7 也算了一次！

               = 5 + 2.25 + 0 + 7.5

               = 14.75

****block 5-7 三个 block 被算了两次****：一次作为 host credit（`0.75 * 3 = 2.25`），一次作为 shared credit 的一部分（`0.5 * 3 = 1.5`）。这三个 block 单独贡献了 `2.25 + 1.5 = 3.75` credit，如果只按最高 tier 算应该只有 `2.25`（走 host）——****多算了 1.5****。

#### 6.14.3 官方文档与设计意图推测

调研过程中****没有找到****官方文档解释"为什么 shared 只减 device，不减 host/disk"。合理推测有两个方向：

****推测 1：设计者认为 shared 与 host/disk 场景上不会同时出现****

Shared pool 的典型部署（Mooncake 等）是****跨 worker 的全局 KV 池****，通常在多 worker 无法充分复用 prefix 的场景里补足——比如"请求的 prefix 该 worker 上没有，但集群里其他 worker 或全局池有"。而 host/disk 层是****同一个 worker 的分层缓存****（通过 SGLang HiCache）。设计者可能假设：如果一个 block 已经在候选 worker 自己的 host / disk 层有了，那"不需要"再去 shared pool 里检查——因为本地取比远端取快。

但这个假设不严格——****没有代码约束"shared pool 里的 block 不会与某个 worker 的 host tier 重合"****。实际上 Mooncake 这类系统的 block 由所有 worker 共享写入，某个 worker 的 host tier 上的 block 完全可以同时存在于 shared pool 中。

****推测 2：Shared credit 意图是"额外的 network fallback 保险"****

﻿`shared_cache_multiplier` 默认 0.5，权重是四个 tier 里最低的（除了未启用的场景）。设计者可能认为它更多是一种"如果本 worker 什么都没有，还能从 shared pool 兜底取一份"的信号，double-count 那点权重不算严重。但源码没写这个设计意图，纯属推测。

#### 6.14.4 与 §6.13 tier-exclusive 去重的对比

|   |   |   |
|---|---|---|
|维度|Device/Host/Disk 之间（§6.13）|Shared vs Device/Host/Disk（本节）|
|去重机制|Indexer 层`query_lower_tiers` 的 continuation offset|Selector 层`hits_beyond(device_overlap_blocks)`|
|去重范围|三 tier 之间完全互斥|只减 device，不减 host / disk|
|官方回归测试|✅`concurrent_tiered_query_does_not_double_count_device_and_lower_tier_overlap`|未找到|
|是否 double-count|不会|可能会（如上例 block 5-7）|

  

#### 6.14.5 实际影响估计

****默认权重下的影响幅度****：

- 单个 block 若在 host + shared 同时命中：`0.75 + 0.5 = 1.25`（正常应该 `1.0` 走 device 或 `0.75` 走 host）
    
- 单个 block 若在 disk + shared 同时命中：`0.25 + 0.5 = 0.75`（正常应该 `0.5` 走 shared 或 `0.25` 走 disk）
    

****放大到 cost function****：`overlap_credit_blocks` 里被多算的部分会让 `adjusted_prefill_blocks = raw - credit` 偏小（credit 被高估 → 剩余 prefill 被低估 → cost 被低估 → 这个 worker 更可能被选中）。也就是"****同时有 host tier 又在 shared pool 的 worker 会被系统性偏爱****"，即便这种偏爱可能不代表真实的复用优势。

****什么场景会明显****：

- Shared pool 覆盖率高 + HiCache host tier 也在工作 + 请求 prefix 与本 worker 的 host tier 有大量重合（比如 SGLang `write_through` 策略下 host tier 与 GPU 高度同步）
    
- 反之，`--shared-cache-type=none`（默认）或 HiCache 未开启时****完全不发生****——只有一个 tier 有命中就没有 double-count 可言
    

****生产验证方法****：观察 `dynamo_component_router_kv_hit_rate`——理论上限是 100%，如果显著超过（比如 130%+）说明 credit 被系统性高估，可怀疑这条链路。可以临时把 `shared_cache_multiplier` 调到 0 做 A/B 对比。

#### 6.14.6 结论

****这是 Dynamo 源码里一个真实存在的不完美对称****：tier 内部（device/host/disk）通过 continuation offset 做了严格去重，但 shared credit 只对 device 去重，与 host/disk 之间存在 double-count 的可能。这可能是****设计取舍****（shared 只作为兜底信号，权重低到不必严格去重）或****遗漏****，官方文档没有明确解释。

生产部署时如果观察到 shared cache 与 HiCache 同时启用后打分明显偏向"host 命中多的 worker"，可以怀疑这条链路。修复方向如果要提 issue 或补丁：把 `hits_beyond(device_overlap_blocks_u32)` 改成 `hits_beyond((device + host + disk) as u32)`——但这需要证明"host / disk 命中的位置区间"能被表达成 `u32` 位置（当前 `hits_beyond` 只接受位置游标，不接受任意 range 集合，需要扩展一下 API）。

---

## 7. Filter 与 Scoring 的边界

Dynamo Router 的决策是****分两阶段****的：先做"哪些 worker 有资格"（filter）、再对合格候选算分（scoring）。混淆这两个概念会导致调参思路错乱。官方文档明确写道（`docs/components/router/router-filtering.md`）：

> "filters decide whether a worker or DP rank is eligible at all, while scoring ranks the eligible candidates by KV overlap and load."

### 7.1 Filter：硬约束

这些条件失败 → worker ****直接被剔出候选集****，不参与打分：

|   |   |   |
|---|---|---|
|Filter|类型|说明|
|Allowed worker IDs|Routing hint|请求 hint 里可指定"只能路由到这几个 worker"|
|DP-rank bounds|请求约束|`[data_parallel_start_rank, data_parallel_start_rank + data_parallel_size)`|
|Required taints|Topology 硬约束|缺失任一 required taint 的 worker 直接被剔（如 zone/rack）|
|Busy-threshold overload|容量硬约束|`--active-decode-blocks-threshold` / `--active-prefill-tokens-threshold` / `--active-prefill-tokens-threshold-frac`|
|LoRA pinning|Adapter 硬约束|LoRA-pinned worker 若不在原候选集内会被丢弃（不扩大）|

  

****Busy threshold 的运行时调整****：`/busy_threshold` HTTP endpoint 可动态调这三个阈值，不需要重启 Router。

#### 7.1.1 Busy-threshold 过载判定详解：负载看的是 block 与 token，不是 request count

这是 Filter 阶段唯一涉及"动态负载"的判定环节（其他 filter 都是静态属性判断，如 taint / DP-rank）。展开讲，因为 Dynamo 在这里的设计与所有姊妹方案都不同。

****判定逻辑伪代码****：

is_worker_overloaded(worker):

  load = ActiveSequences[worker].projection()

    # → { active_prefill_tokens,     # compute 侧：其他 live 请求还没算完的 prefill token 排队量

    #     active_decode_blocks,       # memory 侧：所有 live 请求持有的 unique KV block 数

    #     ... }

  capacity = worker.runtime_config

    # → { max_active_decode_blocks,   # 从 worker 上报的 KV pool 总容量

    #     max_active_prefill_tokens } # 从 worker 上报的 prefill 引擎容量上限

  # 三个 threshold 参数任一命中即视为过载（OR 关系）

  overloaded = false

  if config.active_decode_blocks_threshold is set:

    # 判定：memory 侧比例是否超阈值

    if (active_decode_blocks / max_active_decode_blocks) > threshold:

      overloaded = true

  if config.active_prefill_tokens_threshold is set:

    # 判定：compute 侧绝对 token 数是否超阈值

    if active_prefill_tokens > threshold:

      overloaded = true

  if config.active_prefill_tokens_threshold_frac is set:

    # 判定：compute 侧比例是否超阈值

    if (active_prefill_tokens / max_active_prefill_tokens) > threshold:

      overloaded = true

  return overloaded

****三个阈值参数详解****：

|   |   |   |   |   |
|---|---|---|---|---|
|参数|单位|变量来源|阈值含义|典型使用场景|
|`--active-decode-blocks-threshold`|比例（0.0-1.0）|`active_decode_blocks / max_active_decode_blocks`|KV pool 使用率超过多少视为满|防止 OOM / preemption；`0.9` 表示 90% KV pool 满就不接客|
|`--active-prefill-tokens-threshold`|绝对 token 数（整数）|`active_prefill_tokens`|prefill 排队 token 总量超过多少视为忙|直接卡住 TTFT 尾延迟；`--router-queue-threshold` 参考的也是这个|
|`--active-prefill-tokens-threshold-frac`|比例（0.0-1.0）|`active_prefill_tokens / max_active_prefill_tokens`|相对形式的 prefill 排队占比阈值|与上一个二选一，比例形式对不同 worker 容量更 robust|

  

三个参数****默认都是**** `****None****`——即 ****Dynamo 默认不开 busy-threshold 过载判定****，任何 worker 都能进入 scoring。运维需要主动配。

****变量来源三分类归位****：

|   |   |   |
|---|---|---|
|变量|来源类别|更新频率|
|`active_decode_blocks`|本地计数（`&mut self` 记账）|sub-ms（请求 admit / free 触发）|
|`active_prefill_tokens`|本地计数（带时间衰减的 token 累加）|sub-ms（请求 admit /`mark_prefill_completed` 触发）|
|`max_active_decode_blocks` / `max_active_prefill_tokens`|Worker 上报的`RuntimeConfig`|只在 worker 注册 / 重配时更新|

  

****关键设计选择：完全不看 in-flight request count****

Dynamo Router 过载判定的所有变量都是 ****block / token 单位****，****没有一个是"在飞请求数"****。这与所有姊妹方案都不同——`RL-Router` 三个策略、`SGLang Gateway`、`verl KVCAwareBalancer` 都以 request count 为主要或次要负载信号：

|   |   |
|---|---|
|方案|过载判定的主要信号|
|****Dynamo****|`active_decode_blocks / capacity` 比例 + `active_prefill_tokens`|
|RL-Router`cache_aware`|`LoadAvailable()`: `requestLoad[tenant] < 512`（****request count****）|
|RL-Router`pd_cache_aware`|`atRequestCapacity`: `ActiveRequests ≥ MaxRequestLoad`（****request count****）|
|RL-Router`session_aware_v5`|`active ≥ MaxSessionLoad`（****request count****）+ `availableBlocks ≤ 128`（block）|
|SGLang Gateway`cache_aware`|双阈值`max-min > 64 AND max > min × 1.5`（****request count**** 差）|
|verl`KVCAwareBalancer`|`load_threshold = 0.9` on Prometheus 综合指标（含 request count）|

  

****为什么这么设计？**** LLM inference 的真实资源约束是****GPU memory（KV cache slot）+ compute FLOPS****，不是"request 数量"。100 tokens 请求与 10000 tokens 请求对 GPU 的实际压力差 100 倍，request count 把它们等同看待偏差很大。Dynamo 的目标场景（长 context + P/D 分离）正是 request count 最失效的场景——请求长度分布长尾，request count 平均下来的负载估计误差最大。

****代价****：想按"该 worker 上活跃请求数超过 N 就不接客"这种直觉配置的运维，Dynamo 的 filter ****配不出来****。必须换成 block 或 token 单位思考。

****运行时热更新****（Dynamo 独有能力）：这三个阈值可以通过 `/busy_threshold` HTTP endpoint 运行时热更新，不需要重启 Router。这在 A/B 试验或应急降级时有用——线上发现 `--active-decode-blocks-threshold=0.8` 太严格导致大量 reject，直接 curl 调到 0.9 秒生效。

****Filter 与 Queue admission 的关系****（澄清防混）：

- Filter 阶段的 busy-threshold（本节）：****worker 直接失去候选资格****，请求 skip 到其他 worker；
    
- Queue admission 阶段的 `--router-queue-threshold`（§7.3）：****所有 worker 都过载时的最后兜底****，请求进 Router pending queue 等一会儿再重打分，不是直接失去资格。
    

两者虽然都涉及"负载太高怎么办"，但作用层次不同：先过滤（filter），再打分（scoring），最后排队（admission）。

### 7.2 Scoring：软信号

以下都是打分公式的一部分，"越差扣得越多但不至于直接排除"：

|   |   |
|---|---|
|Signal|效果|
|Preferred taints|乘性 cost 惩罚（`preferred_taint_multiplier`）|
|`router_temperature`|Softmax 采样温度|
|`overlap_score_credit` / `host_cache_hit_weight` / `disk_cache_hit_weight` / `shared_cache_multiplier`|各 tier 的 cache credit 权重|
|`prefill_load_scale`|prefill 项的整体缩放|

  

### 7.3 Admission / Queue：既不是 filter 也不是 scoring

﻿`--router-queue-threshold` 是****admission backpressure****——当所有合格 worker 都过载（超过阈值），请求不是被拒绝，而是进入 Router 的 pending queue 等一会儿。官方原话（`docs/components/router/router-filtering.md`）：

> "`--router-queue-threshold` is not candidate eligibility. It is admission backpressure."

Queue 内的排序由 `--router-queue-policy` 决定，三种选项：

|   |   |   |
|---|---|---|
|Policy|Key 公式|优化目标|
|`fcfs`（默认）|调整后到达时间：`priority_jump - arrival_offset`|优化 tail TTFT|
|`lcfs`|反向到达时间：`priority_jump + arrival_offset`|逆序处理，实验性|
|`wspt`|`(1 + priority_jump) / isl_tokens`|优化 average TTFT（短请求先）|

  

完整的 queue key 是元组 `(strict_priority, policy_key)`——****strict_priority 高的先出队****，同 strict_priority 内按 policy 排。

### 7.4 Priority 的三层独立机制

（这是一个容易误解的话题，来源：`docs/components/router/priority-scheduling.md`）

Dynamo 的 priority 分****三层****，****每层需要各自的触发条件****，不联动：

|   |   |   |
|---|---|---|
|层次|触发条件|作用|
|Router queue priority|`--router-queue-threshold` 触发（有队列争抢）|队列内排序|
|Engine priority|引擎端开对应 flag（vLLM`--scheduling-policy priority`；SGLang `--enable-priority-scheduling`；TRT-LLM Dynamo 未透传）|引擎内部调度|
|Cache priority|引擎端开 cache 策略（如 SGLang`--radix-eviction-policy priority`）|KV cache 驱逐优先级|

  

两种 priority 字段：

- ﻿`agent_hints.priority`（soft signal）：同时给 router policy scoring 和后端 engine 用。
    
- ﻿`agent_hints.strict_priority`（无符号 tier）：****只用于 router pending queue，不会传给后端 engine****。
    

HTTP header 版本：`x-dynamo-request-priority`、`x-dynamo-request-strict-priority`。负值被 clamp 到 0。

官方文档明确排除的语义（澄清防误解）：

> "Not Kubernetes PriorityClass, not GPU preemption, not admission control, and it does not reserve capacity or preempt admitted work."

也就是说 Dynamo 的 priority ****不是**** Kubernetes PriorityClass、不做 GPU preemption、不做 admission control、也不 reserve 容量。就是队列排序。

### 7.5 `--load-aware` preset：一键关掉 KV 感知

****这是 Dynamo 官方推荐的"绕开 KV Indexer、纯负载路由"入口****（`docs/components/router/configuration-and-tuning`）。开启 `--load-aware` 后：

- 强制 `--router-mode kv`（因为 basic modes 没有 load-aware 组件）；
    
- ﻿`overlap_score_credit = 0.0`（device credit 归零）；
    
- 关闭 `use_kv_events`（不订阅事件）；
    
- 关闭 `durable_kv_events`；
    
- 关闭 `router_assume_kv_reuse`（不写入 Indexer）；
    
- 关闭 `use_remote_indexer` / `serve_indexer`；
    
- ﻿`shared_cache_multiplier = 0.0`（shared credit 也归零）；
    
- ﻿`shared_cache_type = "none"`；
    
- 保留 `host_cache_hit_weight` / `disk_cache_hit_weight`（****注意****：如果 device credit 归零但保留 host/disk，实际上 host/disk 命中信号变相成了唯一的 overlap 信号——但因为 `use_kv_events=false`，Indexer 也不会有 host/disk 事件进来，所以最终等价于 pure load routing）；
    
- 保留 `router_track_active_blocks` + `router_track_prefill_tokens`。
    

****用途****：一份完整的 A/B 对照——把 `--load-aware` 作为 baseline 组、把默认 kv mode 作为实验组，直接对比 cache-aware 相对纯负载的收益。这是官方推荐的评估路径。

---

## 8. 决策分支全景

这一节把前面 3-7 节讨论的所有决策环节合并成一条完整的执行链路，展示"从 Frontend 把请求交给 Router，到 Router 返回一个选中的 worker"之间****每一步在做什么、为什么做、失败时会走哪个分支****。伪代码风格对齐姊妹文档（verl §5 / RL-Router §4.3 §5.3 §6.3），但注释更详细——****即便你直接从这一节开始读也能理解****。

如果你只想知道"Router 是怎么选的"，读完这一节就够了；前面各节是为这里每个 Step 的深度展开。

### 8.1 完整决策树（详细注释版）

flowchart TD

    A[新请求到达 Router] --> B[Step0 compute_block_hash_for_seq 得到 block hashes]

    B --> C[并行查询 Indexer 本地前缀树 与 Shared KV Cache 可选]

    C --> D[Step2 LoRA 约束过滤 只保留有该 adapter 的 worker]

    D --> E[Step3 Eligibility 过滤 剔除过载 缺 taint 非 allowed DP rank 不符]

    E --> F{pinned_worker 指定且不合格?}

    F -- 是 --> F1[ERROR validate_pinned_worker_allowed failed]

    F -- 否 --> G{candidates 为空?}

    G -- 是 且全过载 --> G1[AllEligibleWorkersOverloaded 进入 queue 或拒绝]

    G -- 是 且无注册/被硬约束刷光 --> G2[NoEndpoints]

    G -- 否 --> H[对每个候选组装 overlaps device host disk shared]

    H --> I{启用了 credit decay?}

    I -- 是 --> J[min_active_prefill_tokens = min over candidates]

    I -- 否 --> K[跳过 decay]

    J --> K

    K --> L[逐个候选算 adjusted_prefill_blocks = max raw - 各层 credit, 0]

    L --> M["logit = prefill_load_scale * adjusted_prefill_blocks + decode_blocks"]

    M --> N[应用 preferred_taint_multiplier 软约束打折]

    N --> O{temperature == 0 ?}

    O -- 是 --> P[argmin logits]

    P --> Q{有并列 tie?}

    Q -- 是 --> R[reservoir_sample tie_workers]

    Q -- 否 --> S[winner = argmin]

    O -- 否 --> T[softmax_sample logits temperature]

    R --> S

    S --> U{所有合格 worker 都 > queue_threshold?}

    T --> S

    U -- 是 --> V[enqueue request 按 strict_priority policy_key 排序]

    V --> W[等有 worker 释放后回到打分]

    W --> H

    V --> X{队列满?}

    X -- 是 --> Y[QueueRejected]

    U -- 否 --> Z[assume_kv_reuse 更新 ActiveSequences blocks + prefill_tracker]

    Z --> AA[update_metrics overlap_blocks cached_tokens 等]

    AA --> BB["返回 FindBestMatchOutcome::Routed worker + overlap + cached_tokens"]

acquire_worker(request, tokens, hints):

  # ===================================================================

  # Step 0: 把请求 prompt 变成一串 block hash（cache 命中的判定依据）

  # ===================================================================

  # KV cache 是按 token 组织的，raw text 无法用来匹配缓存。所以第一件事

  # 是 tokenize 后把 token 序列切成 block（默认每 16 tokens 一块），

  # 每个 block 算一个 XXH3-64 hash 作为唯一标识。

  # 这些 hash 就是接下来去 Indexer 查询"谁缓存了这个 block"的 key。

  hashes = compute_block_hash_for_seq(tokens)

  # 特殊情况：如果 block_size 配成 0（一般不会发生，配置错误场景），

  # 就没有 cache 感知，后面所有 overlap 相关的值都是 0。

  if kv_block_size == 0:

    hashes = []

  # ===================================================================

  # Step 1: 并行去两个数据源查"哪些 worker 缓存了这些 block"

  # ===================================================================

  # 这是 Router 决策所需的 cache 侧信息来源。两个数据源相互独立，可以

  # 并行发起，最后汇总结果。这一步不做任何决策，只收集数据。

  parallel {

    # 数据源 1：本地 Indexer（每个 Router 内部维护的全局前缀树）

    # 它订阅所有 worker 发的 KV events（BlockStored/BlockRemoved/Cleared），

    # 实时记录"哪个 block hash 被哪些 worker 缓存在哪一层"。

    # 输出：每个 candidate worker 在 device/host/disk 三个 tier 上分别

    # 命中了多少个 block（这三个数字已经在上游做了 tier-exclusive 去重，

    # 参见 §6.13）。

    indexer_result = query_tiered_matches(hashes)

      # → {worker_id: TierOverlapBlocks {device, host_pinned, disk}}

    # 数据源 2（可选）：跨 worker 共享的 KV 存储池（如 Mooncake）

    # 只在 --shared-cache-type=hicache 时才启用。共享池是所有 worker 都

    # 能访问的全局池，所以命中意味着"任何 worker 都能拿到这个 block"，

    # 与本地 device 命中之间需要显式去重（shared_beyond = 共享池命中 - 该

    # worker 本地 device 命中，避免同一个 block 被 double-count）。

    if shared_cache_enabled:

      shared_result = shared_kv_cache.query(hashes)

        # → {worker_id: shared_beyond_blocks}

      # 如果共享池查询失败：不报错，只 log warning，请求继续用 indexer

      # 的结果打分，等效于关闭了 shared credit 这一项。

    else:

      shared_result = {}

  }

  # ===================================================================

  # Step 2: LoRA 硬约束——请求指定了 LoRA adapter 就只能路由到有该 adapter 的 worker

  # ===================================================================

  # LoRA 是 fine-tune 出来的小 adapter，特定 adapter 只加载在特定 worker 上。

  # 如果请求带 LoRA hint，Router 必须只在"已加载该 adapter 的 worker"里选。

  # 关键约束：这一步只会**缩小**候选集，不会扩大——如果 LoRA-pinned worker

  # 不在初始候选集里（比如那个 worker 已经因为其它原因被剔除了），Router 不

  # 会临时把它加回来，而是直接把这个 worker 从 pinned 名单里删掉。

  candidates = narrow_allowed_by_lora(base_candidates)

  # ===================================================================

  # Step 3: Eligibility 硬约束——把"没资格接客的 worker"全部剔出候选集

  # ===================================================================

  # 这一步是纯 filter，不打分，不排序，只判断"这个 worker 有没有资格进入

  # 后面的打分环节"。任何一条硬约束不满足，worker 直接被踢，不参与打分。

  # 这些约束的具体含义参见 §7.1。

  candidates = candidates

    \ {已过载的 worker}           # busy-threshold overload

                                   # 判定看 block 与 token 单位，**不是** request 数量：

                                   # · --active-decode-blocks-threshold：

                                   #   active_decode_blocks / capacity 超过该比例 → 过载

                                   #   （memory 侧：KV pool 快满了）

                                   # · --active-prefill-tokens-threshold：

                                   #   active_prefill_tokens 超过该绝对值 → 过载

                                   #   （compute 侧：prefill 排队 token 太多）

                                   # · --active-prefill-tokens-threshold-frac：

                                   #   同上但相对比例形式

                                   # 三阈值任一命中即视为过载（OR）；三者默认都是 None（不启用）。

                                   # 详细伪代码与阈值语义见 §7.1.1。

    \ {缺 required_taint 的 worker}  # 拓扑硬约束

                                     # 比如客户端要求 "只能在 us-east 机房"，

                                     # 没有对应 taint 的 worker 全部排除

    \ {不在 allowed_worker_ids 里} # 客户端通过 routing hint 限定的候选集

    \ {DP-rank 不在指定范围内}     # 数据并行的 rank 约束

                                    # [data_parallel_start_rank,

                                    #  data_parallel_start_rank + data_parallel_size)

  # ===================================================================

  # Step 4: 边界情况处理——没有合格候选怎么办

  # ===================================================================

  # 特殊情况 A：客户端指定了 pinned_worker 但那个 worker 不合格

  # （比如它过载了，或者拓扑约束不满足）。这种情况直接报错，因为客户端

  # 明确要求了具体 worker，Router 无权换。

  if pinned_worker specified and pinned_worker not in candidates:

    return ERROR: validate_pinned_worker_allowed failed

  # 特殊情况 B：候选集空

  if candidates == ∅:

    if 存在 worker 但都因为过载被剔除:

      # 服务过热场景。可能进入 Router queue 等等看，也可能直接拒绝。

      # 具体行为由 --router-queue-threshold / --router-queue-policy 决定。

      return AllEligibleWorkersOverloaded  # → QueueRejected 或排队等待

    else:

      # 一个 worker 都没注册，或全被 required_taint 等硬约束刷光了。

      # 这不是负载问题，是配置/环境问题，直接返回错误。

      return NoEndpoints

  # ===================================================================

  # Step 5: 对合格候选逐个打分

  # ===================================================================

  # 到这一步，candidates 里全是"有资格接这个请求"的 worker。

  # 现在要在他们里面按 cost function 排序，cost 越低越优先。

  # cost 由两项组成：prefill-side（compute 成本）+ decode-side（memory 成本），

  # 详细公式与变量含义参见 §6.1 与 §6.11。

  # 5a. 组装本次打分要用的权重（配置读取阶段）

  # 全局配置里读默认权重（overlap_score_credit / host_cache_hit_weight 等），

  # 如果客户端请求带了 RouterConfigOverride，用 override 覆盖对应字段。

  # 支持覆盖的字段清单见 §6.5。

  weights = merge(RouterConfigOverride, global_config)

  # 5b. Decay 预计算（可选，只在启用了 credit decay 时执行）

  # 目的：如果某个"cache 命中最多的 worker"同时已经积压了大量 prefill 工作，

  # 让它的 device credit 打折——避免"命中最高就无脑往那儿打"造成热点。

  # 归一化的分母 min_active_prefill_tokens 反映"当前最闲 worker 的 prefill 排队",

  # 用来判断某个 worker "超出别人多少"，决定 decay 幅度。

  if track_prefill_tokens and overlap_score_credit_decay > 0:

    min_active_prefill_tokens = min over candidates of active_prefill_tokens

  # 5c. 对每个候选 worker 计算 logit（cost 越小越好）

  for worker in candidates:

    # 读取该 worker 的负载信息——注意这里是从 Router 本地维护的

    # ActiveSequences（每个 worker 一份，& mut self 记账）拿的，

    # 不查 HTTP metrics。这是 sub-ms 精度的实时视图。

    load = ActiveSequences[worker].projection()

      # → { active_prefill_tokens,     # 该 worker 上其他请求还没算完的 prefill

      #     active_decode_blocks,       # 现有 live 请求的 memory footprint

      #     additional_active_blocks }  # 新请求发过去后还要额外占的 blocks

    # 组装该 worker 的 overlap 数据（Step 1 查出来的）

    overlaps = {

      device: indexer_result[worker].device,     # GPU 上命中的 block 数

      host:   indexer_result[worker].host_pinned, # host memory 上命中的 block 数

      disk:   indexer_result[worker].disk,        # disk 上命中的 block 数

      shared: shared_result.get(worker) or 0,     # 共享池"额外"命中的 block 数

    }

    # 上面三个 device/host/disk 已经在 Indexer 层做了 tier-exclusive 去重

    # （通过 continuation offset 机制，见 §6.13），所以下面加起来不会 double-count。

    # 核心公式（详见 §6.1）：

    #   adjusted_prefill_blocks =

    #     max(raw_prefill_blocks - device_credit - host_credit - disk_credit - shared_credit, 0)

    #   logit = prefill_load_scale * adjusted_prefill_blocks + decode_blocks

    #

    # 直观理解：

    #   左项 (prefill-side) = 这个新请求发过去后，还要做多少 prefill 计算工作

    #   右项 (decode-side)  = 该 worker 的 KV cache 现在有多满 + 新请求会占多少

    #   logit 越小 = compute 剩余工作少 且 memory 剩余空间大 = 越优先选

    logit = worker_logit(request, load, overlaps, weights, block_size)

    # 5d. 应用 Preferred taint 乘子（软约束加成/惩罚）

    # 比如客户端表达"倾向 GPU 型号 A100 的 worker"，但不是硬性要求。

    # 对满足 preferred taint 的 worker 打分乘以一个小于 1 的因子（cost 打折，

    # 变得更有吸引力）；不满足的乘 1（cost 不变）。

    logits[worker] = logit * preferred_taint_multiplier(worker)

  # ===================================================================

  # Step 6: 从打分结果里选出 winner

  # ===================================================================

  # 有两种选法，取决于 router_temperature 参数（详见 §6.7）。

  if temperature == 0:

    # 确定性模式（默认）：argmin，选 cost 最低的

    winner = argmin(logits)

    # 处理并列：如果多个 worker 打分完全相同（比如 cold start 阶段所有

    # worker 的 overlap 都是 0），不能总是选第一个（否则所有 tie 请求

    # 会集中到同一个 worker），而用 reservoir sampling 让每个 tie worker

    # 有 1/count 的概率成为 winner。

    if 有多个 worker 打成平手:

      winner = reservoir_sample(tie_workers)

  else:

    # 采样模式（temperature > 0）：softmax 概率采样

    # 目的：避免"所有相似请求都涌向同一个 cost 最低的 worker"造成瞬时热点。

    # 实现细节：scale = -1 / ((max_logit - min_logit) * temperature)  ← 负 scale

    # 通过负 scale 把 argmin 翻成 argmax(概率)——低 logit 有更高被选概率，

    # 但不是必选，保留探索空间。inverse-CDF 采样。

    winner = softmax_sample(logits, temperature)

  # ===================================================================

  # Step 7: Admission control——winner 太忙就先排队等等

  # ===================================================================

  # 打分选出 winner 不代表立刻发送。如果所有合格 worker 都比较忙（queue

  # threshold 触发），请求会进入 Router 的 pending queue 等一会儿，等有

  # worker 空闲了再派发。这是 backpressure 机制，不是拒绝。

  # queue 内部按 --router-queue-policy 排序：

  #   fcfs (默认) = 先来先服务，优化 tail TTFT

  #   lcfs        = 后来先服务，实验性

  #   wspt        = 短请求先服务，优化 average TTFT

  # queue key 是 (strict_priority, policy_key) 元组——strict_priority 高的

  # 先出队，同 priority 内按 policy 排（详见 §7.3 §7.4）。

  if 所有合格 worker 都 > queue_threshold:

    enqueue(request, key = (strict_priority, policy_key))

    → 等有 worker 释放负载后再回到 Step 5 重新打分

    → 或队列满时拒绝：QueueRejected

  # ===================================================================

  # Step 8: 更新 Router 内部状态并返回决策

  # ===================================================================

  # 8a. 假设 winner 会真的缓存这次的 prefix（默认 assume_kv_reuse=true）

  # 这一步是把请求的信息"预写入" ActiveSequences——即便 KV events 还没

  # 从 worker 发回来，Router 内部已经知道"我把这个 prefix 派给 winner 了，

  # 后面同 prefix 的请求应该优先给它"。这个乐观假设有 TTL 剪枝兜底，防止

  # 假设错了永远残留。

  if assume_kv_reuse:

    ActiveSequences[winner].add_request(hashes, request_id, ...)

      # → 更新 blocks（占用）+ prefill_tracker（排队 prefill tokens）

  # 8b. 更新观测指标（详见 §11）

  update_metrics(winner, overlap_blocks, cached_tokens, ...)

  # 8c. 返回决策结果给 Frontend

  return FindBestMatchOutcome::Routed {

    worker: winner,                        # 选中的 worker id

    overlap_blocks: {device, host, disk},  # 每层实际命中的 block 数（供调试/observability）

    effective_overlap_blocks: ...,         # 打折后（应用了 credit weight）的有效命中

    cached_tokens: ...,                    # 复用的 token 数（供 client observability）

    routing_hashes: hashes,                # 决策时用的 block hash（供后续 tracking）

  }

### 8.2 分支互斥优先级

从高到低（一次决策链路里只会命中一条）：

1. ﻿`****NoEndpoints****`——没有任何 worker 注册；这是环境/配置问题
    
2. ﻿`****AllEligibleWorkersOverloaded****`（可能引发 `QueueRejected`）——有 worker 但全过载
    
3. ****LoRA / Pinned worker 硬约束失败****——客户端要求的具体 worker 不合格
    
4. ****Eligibility filter 全部刷光****（等价于场景 2）
    
5. ****Scoring 完成 + argmin/softmax 选出 winner****
    
6. ****Queue admission****——winner 太忙时排队等待或最终拒绝
    
7. ﻿`****Routed****`——成功返回选中 worker
    

### 8.3 每一步的源码入口

|   |   |
|---|---|
|Step|入口|
|0. Block hashing|`lib/llm/src/kv_router.rs :: compute_block_hash_for_seq`|
|1a. Indexer 查询|`lib/llm/src/kv_router/route_lookup.rs :: query_tiered_matches` → `lib/kv-router/src/indexer/lower_tier_indexers.rs :: query_lower_tiers`（continuation offset 去重）|
|1b. Shared cache 查询|`lib/llm/src/kv_router.rs :: SharedKvCache` trait|
|2. LoRA filter|`lib/llm/src/kv_router.rs :: narrow_allowed_by_lora`|
|3-4. Eligibility|`lib/kv-router/src/scheduling/selector.rs :: RoutingEligibility`|
|5. Scoring|`lib/kv-router/src/scheduling/selector.rs :: DefaultWorkerSelector::select_worker` / `worker_logit`|
|6. Argmin / softmax|`lib/kv-router/src/scheduling/selector.rs :: softmax_sample`|
|7. Queue admission|`lib/kv-router/src/scheduling/queue_admission/`|
|8. State update|`lib/kv-router/src/sequences/single.rs :: add_request_with_prefill_tracking`|

  

### 8.4 关于本节脉络的三个补充说明

****为什么 Step 1 (数据收集) 早于 Step 3 (硬约束 filter)?****

因为 Indexer / Shared cache 查询不依赖候选集是谁——查询的是"整个集群"每个 worker 分别命中多少 block，即便某个 worker 后面会被硬约束剔除也无所谓。这两步并行发起可以隐藏查询延迟。等到 Step 5 打分时，只对合格候选查表就行，多余的数据被自然丢弃。

****为什么 Step 6 (选 winner) 之后还有 Step 7 (排队)?****

Winner 是"cost 最低的"，但 cost 低不代表"当前有空闲资源接客"。举个极端例子：所有 worker 都过载了，但 A 的 cost 是 100、B 是 120——A 是 winner，但 A 也已经超过 queue_threshold。这种情况下 Router 不该硬塞给 A（会加剧过载），而是让请求进 Router 侧的 pending queue 等有 worker 空闲了再重新打分。

****为什么 Step 8a 要"预写入" ActiveSequences？****

Router 派发请求到 worker → worker 做 prefill → worker 发 `Stored` KV event → Router Indexer 更新——这条链路有毫秒级延迟。如果 Router 派完请求就不管，那紧接着的第二个同 prefix 请求到来时，Indexer 还不知道"第一个请求已经把 prefix 缓存到 winner 上了"，可能会派到别的 worker，浪费 cache 复用机会。

预写入的作用是****用 Router 自己的路由决策作为 cache 状态的乐观假设****——"我既然派给 winner 了，那 winner 就应该有这份 prefix，未来同 prefix 的请求继续偏向 winner"。这个假设可能错（worker 崩了没 prefill 成、或者 cache 被立刻 evict 了），所以有 `router_ttl_secs=120s` 的 TTL 兜底：如果 120s 内没收到 worker 的实际 `Stored` event 确认，这条预写入的假设会过期。

---

## 9. P/D 分离下的路由分工

P/D 分离（disaggregated serving）把 prefill 和 decode 拆到不同的 worker pool，因为两阶段的性能特征不同：prefill 计算密集，decode 内存带宽敏感。Dynamo 的 P/D 路由****并不是"两个独立的 Router 各自决策"****，而是****一个 KvRouter 主体 + 一个 PrefillRouter 前置组件****的架构。

### 9.1 PrefillRouter：Prefill 阶段的前置组件

****源码位置****：`lib/llm/src/kv_router/prefill_router/mod.rs`（及同目录下 `activation.rs`, `admission.rs`, `query.rs`）。

****Doc comment 原文****：__"a forward-only operator that sits between Migration and the decode router."__

****三种运行模式****：

|   |   |   |
|---|---|---|
|模式|触发条件|行为|
|Query-only|客户端只想问"应该 prefill 到哪"|只返 worker id，不派发|
|Pre-routed|外部 EPP 已经决策，通过`x-dynamo-prefill-instance-id` header 传入|Router 只做记录，不重新选|
|Normal|默认|Router 用 KV-aware 打分选 prefill worker|

  

****Prefill worker 的选择流程****：

1. 若 `lifecycle_state() != Active`（如 PrefillRouter 未激活）→ ****bypass****，直接调 `next.generate(...)`。
    
2. ﻿`--router-mode direct` 但未通过 header 传 `prefill_worker_id` → 报错（预期外部 EPP 提供）。
    
3. 克隆请求，`max_tokens := Some(1)`（`original_max_tokens` 备份，供 decode 恢复）。
    
4. ﻿`InnerPrefillRouter::select_and_dispatch_prefill(...)` 走 KV-aware 选择（****用同一个**** `****KvRouter::find_best_match_details_with_policy_class****` ****主体****）。
    
5. ﻿`prepare_prefill_dispatch` 回调里把 `prefill_worker_id` / `prefill_dp_rank` 挂到 `request.routing_mut()`，并算 `bootstrap_info`。
    

### 9.2 Decode 侧：****同一个 KvRouter，credit 强制归零****

****这是本文对旧文档最重要的一处纠正****：旧文档 `Dynamo-KV-Cache-Router-机制详解.md` §9 暗示 "decode 侧选谁继续 decode……更关心 decode worker 的负载、可用 slot、拓扑和 KV transfer 成本"，容易被理解为 decode 走独立的 round-robin / least-loaded 策略。

****实际源码****（`lib/llm/src/kv_router/prefill_router/mod.rs :: build_decode_router_override`）：Decode 侧****仍然走标准的**** `****KvRouter::find_best_match_details_with_policy_class****`，但通过 `RouterConfigOverride` 强制覆盖三个字段：

// build_decode_router_override

overlap_score_credit  = 0.0     // device credit 归零

assume_kv_reuse       = false   // 不写入 Indexer

track_prefill_tokens  = false   // 不看 prompt-side load

Doc comment 原话：__"should not score prompt overlap or account prompt-side load."__

****Decode 打分公式退化****：由于 overlap_credit 全归零、track_prefill_tokens 关闭：

raw_prefill_tokens        = request ISL      # track_prefill_tokens=false

overlap_credit_blocks     = 0

adjusted_prefill_blocks   = raw_prefill_blocks

cost = prefill_load_scale * raw_prefill_blocks + decode_blocks

****关键：这不是 round-robin / least-loaded****。它仍然是 KV-aware Router 的一个 sub-mode，只是把 cache 项归零，让打分退化为"整体请求 workload 估计（prefill 项）+ decode backlog（decode 项）"的组合。相比于 basic mode 的 `least-loaded`，这个模式对****每个 worker 未来的 decode 阻塞情况****估计得更准，因为它加入了 prefill 项——一个 worker 若同时接 prefill 也接 decode（aggregated 部署下），prefill 项能反映这个"还要花时间准备"的负担。

### 9.3 Bootstrap 元数据（KV transfer 通道）

Prefill 完成后，KV cache 要从 prefill worker 传到 decode worker。这个"传输通道的建立元数据"由后端提供，Dynamo 三个 backend 各有名字：

|   |   |   |
|---|---|---|
|Backend|元数据字段|含义|
|SGLang|`bootstrap_info { bootstrap_host, bootstrap_port, bootstrap_room, handoff_id }`|RDMA bootstrap 参数|
|vLLM|`kv_transfer_params`|block IDs + remote worker connection info|
|TRT-LLM|`opaque_state`|序列化的 TRT-LLM 内部 metadata|

  

****SGLang**** `****bootstrap_info****` ****详解****（`lib/llm/src/kv_router/prefill_router/query.rs :: extract_bootstrap_info`）：

- ﻿`bootstrap_host: String`﻿
    
- ﻿`bootstrap_port: u16`﻿
    
- ﻿`bootstrap_room: u64`﻿
    
- ﻿`handoff_id: Some(Uuid::new_v4())`﻿
    

****两种构造路径****：

1. ****从 ModelManager 构造****：`prepare_prefill_dispatch` 里查 disaggregated endpoint → 拿 `bootstrap_host` / `bootstrap_port` / `dp_size` → 随机 `room ∈ 0..=i64::MAX as u64` → `compute_bootstrap_room(dp_rank, dp_size, random_room)` 保证 `room % dp_size == dp_rank`（这样 room 能路由到目标 DP rank）。
    
2. ****从 prefill 响应 JSON 解析****：`extract_bootstrap_info` 从 `disaggregated_params` 里解析 `bootstrap_host` (str), `bootstrap_port` (u16::try_from), `bootstrap_room` (u64)。任一字段缺失或越界 → 返回 None → 走 `Completed` 而非 `Bootstrap` 路径。
    

****两种 PrefillOutcome****：

- ﻿`Bootstrap { bootstrap_info, worker_id }`：后台异步发 prefill，****decode 立即开始****，KV transfer 与 decode 并行。
    
- ﻿`Completed { result, worker_id, worker_link }`：等 prefill stream 全部完成，再把 `PrefillResult` 送给 decode。
    

****关键设计注释****（用于理解"为什么 decode 路由不能被 caller cancellation 阻塞"）：

> "the prefill may have completed and KV transfer is in flight. Blocking decode here orphans the transfer (no receiver) and leaks KV blocks permanently."

也就是说，即便客户端 context 被 kill，decode 路由****仍必须继续****，清理由 decode 端的 `kv_transfer_complete_event` guard 负责。

### 9.4 NIXL 直传：物理传输层

****NIXL****（NVIDIA Inference Xfer Library）是 Dynamo 独立的 KV 传输库（`github.com/ai-dynamo/nixl`），负责把 KV block 从 prefill worker 的 GPU 显存搬到 decode worker 的 GPU 显存。

****关键特性****：

- Point-to-point，通过 plugin architecture 抽象各种传输协议：****UCX（主传输，含 AMD ROCm）****、****GDS/GDS_MT（GPUDirect Storage）****、****GPUNETIO（DOCA）****、****LIBFABRIC****、****POSIX****、****OBJ****、****AZURE_BLOB****、****HF3FS****、****MOONCAKE****、****GUSLI****、****UCCL****。
    
- 官方原话：__"the optimal available transport (NVLink, InfiniBand/UCX, etc.)"__
    
- ****非阻塞****：__"non-blocking, allowing GPU forward passes to continue serving other requests during the transfer."__
    
- ****无独立 "RDMA" plugin****——RDMA/verbs 通过 UCX 的 `--with-verbs` / `--with-cuda` / `--with-gdrcopy` 编译选项支持，或走 LIBFABRIC。
    

****语言绑定****：C++ 核心，Python (pybind11)，Rust (`nixl-sys`)。

### 9.5 Topology-aware KV transfer

跨机房 / 跨 zone 的 KV transfer 可能延迟很高，Dynamo 支持通过 topology-aware 约束把 decode worker 限制在与 prefill 相同的 topology domain 内。

****配置结构****（`docs.nvidia.com/dynamo/kubernetes-deployment/operate/topology-aware-kv-transfer`）：

spec:

  experimental:

    kvTransferPolicy:

      labelKey: <string>          # K8s node label key

      domain:   <string>          # 逻辑域名，regex ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$

      enforcement: required | preferred   # 默认 required

      preferredWeight: 0.0-1.0    # 仅 preferred 时

****两种 enforcement 模式****：

|   |   |
|---|---|
|模式|语义|
|`required`（默认）|decode workers****必须****匹配 prefill 的 domain；不匹配 → router 直接 fail 请求|
|`preferred`|所有 decode workers 都 eligible，通过`preferredWeight` 偏置打分|

  

官方原话：__"required is a decode-routing constraint, not a capacity planner."__——它不做容量规划，只是路由过滤。

****代码层面****：`merge_decode_topology_constraints` 是 ****extend 而非 override**** 语义——不覆盖用户自己设的 taint。

### 9.6 与姊妹方案的 P/D 分离对比

|   |   |   |   |
|---|---|---|---|
|方案|Prefill 路由|Decode 路由|是否共用 Router|
|verl|未明确 P/D 分离（RFC 阶段）|未明确|—|
|RL-Router|`****prefill_policy****` ****/**** `****decode_policy****` ****双策略架构****，Scheduler 顺序调用两次策略|通常配`request_num` 之类简单策略|❌ 分开的 Policy 实例|
|****Dynamo****|PrefillRouter（KV-aware）|KvRouter（credit 归零后的 sub-mode）|✅ 主决策链路共用同一个`find_best_match_details_with_policy_class`|
|SGLang Gateway|`--prefill-policy cache_aware`|`--decode-policy power_of_two` 之类|通过 CLI 分开配置|

  

Dynamo 与其他方案的架构差异：****Dynamo 通过 config override 让"同一个 Router 在 prefill 与 decode 侧做不同决策"****，而不是像 RL-Router / SGLang 那样部署两个独立 Policy 实例。好处是行为一致性有保证、代码路径少一半；代价是 override 逻辑必须严格覆盖所有 KV-aware 相关字段（否则 decode 侧会不小心命中 cache 项）。

---

## 10. 系统级能力

前面 6-9 节讨论的都是"单请求怎么被路由"。这一节讨论 Router 作为一个系统组件在****上下游生命周期****里怎么运转——它怎么发现 worker、怎么在多副本部署下同步状态、怎么冷启动、怎么处理故障。这些能力容易被"专注决策公式"的调研文档忽略，但对生产部署很关键。

### 10.1 Worker Discovery：默认 etcd / K8s CRD 可选

****Discovery backend****（`docs/design-docs/discovery-plane.md`）：

- ****默认 etcd****（bare metal / local 部署）
    
- ****Kubernetes 模式****：需显式 `DYN_DISCOVERY_BACKEND=kubernetes`，用原生 K8s 资源（CRD + EndpointSlice）替代 etcd
    

官方原文：__"The runtime always defaults to etcd. Kubernetes discovery must be explicitly enabled."__

****注意****：****NATS 不是 discovery backend****——它只是 KV events / JetStream 的独立通道。****Ray 也不参与**** Dynamo 的 discovery（这与 verl 基于 Ray actor 的架构完全不同）。

****注册路径****（etcd）：worker 在层级 key 下注册，格式：

/services/{namespace}/{component}/{endpoint}/{instance_id}

示例：/services/vllm-agg/backend/generate/694d98147d54be25

Frontend / Router 通过 watch 该 prefix 发现 worker。K8s 模式下类似——workers 创建 `DynamoWorkerMetadata` CRD + EndpointSlice 表达就绪状态。

****Heartbeat / Liveness****（etcd lease）：

- ****默认 TTL 10 秒****（`docs/design-docs/discovery-plane.md`）
    
- 短 TTL（5s）→ 更快故障探测、更多 keep-alive 流量；长 TTL（30s）→ 开销小、检测慢
    

****Worker 掉线时会发生什么****：

1. keep-alive 停
    
2. lease 过期
    
3. ****"All registered endpoints are automatically deleted"****
    
4. Router 收到 removal event，重新路由到健康 worker
    

****掉线时 KV index 状态怎么处理****（****这是对旧文档的一处纠正****——不是发一堆 RemoveEvent，而是 subtree wholesale purge）：

Router-design 原文：

> "On worker discovery (Added event), the router pulls the worker's entire local indexer state. When a worker is removed, all of its blocks are dropped from the global radix tree — this is the disconnect-side removal behavior (there is no separate RemoveEvent per block on disconnect; the router purges the worker's subtree wholesale)."

也就是说：****worker 掉线时，Router 一次性 purge 该 worker 的整个 subtree，不逐 block 发 RemoveEvent****。这与正常运行时的 block 级驱逐（通过 KVPublisher 的 `Removed` event 配对 `Stored`）是两种机制。

****相关环境变量****：

|   |   |
|---|---|
|变量|默认|
|`DYN_DISCOVERY_BACKEND`|`etcd`（备选 `kubernetes` / `file` / `mem`）|
|`ETCD_ENDPOINTS`|`http://localhost:2379`|
|`ETCD_AUTH_USERNAME` / `PASSWORD` / `CA` / `CLIENT_CERT` / `CLIENT_KEY`|—|
|K8s 侧 operator 注入|`POD_NAME`, `POD_NAMESPACE`, `POD_UID`|

  

### 10.2 Multi-Router mesh sync

Dynamo 支持多 Router replica 部署（`docs/components/router/router-operations`）。多 Router 场景下有两个独立的状态一致性问题：

****问题 1：Prefix cache（Indexer）状态****

- 每个 Router 各自维护一份 Indexer，****通过 event plane 自然收敛****——因为所有 Router 都订阅同一份 KV events。
    
- 官方原话：__"The indexer lives in each router or frontend, and multiple router replicas naturally receive the same prefix-cache updates; they do not need router-to-router synchronization for prefix blocks."__
    
- 冷启动时新 Router 通过 §5.4 的 replay 机制（TreeDump 响应）拉取全量。
    

****问题 2：Active block 状态（负载视图）****

- 默认每个 Router ****只知道自己路由过的请求的 active state****——多 Router 之间的负载视图不同步。
    
- 可通过 `--router-replica-sync` 开启同步，走 ****NATS core****（不是 JetStream）传三种 event：
    

|   |   |   |
|---|---|---|
|Event|何时发|语义|
|`AddRequest`|请求被派发时|通知其他 Router："我给这个 worker 加了负载"|
|`MarkPrefillCompleted`|首 token 出来时|通知其他 Router："这个请求的 prefill 完了，从 prefill_tokens 里剔除"|
|`Free`|请求完成时|通知其他 Router："释放负载"|

  

- 每个 event 带 `router_id`，sender 忽略自己发的 echo，防止死循环。
    

****冷启动阶段各 Router 的 index 会不会不同？****

- Prefix cache：会通过 event plane 自然收敛。
    
- Active block：****新 replica 一定从零开始****。原话：__"If a router replica restarts, it starts with no active-block knowledge."__
    
- 开了 replica sync 后：__"a new router still starts with zero active-block knowledge, but it converges through live request handling and active-sequence events from other replicas."__
    

****一个限制****：__"only one__ `__--serve-indexer__` __replica may exist for a given worker component"__——approximate mode（`--no-router-kv-events`）下的 remote indexer serving 只允许一个 replica，event-driven mode 下无此限制。

### 10.3 Cold start / Replay 机制

这一部分本文 §5.4 已详细讨论过 replay 通道的三态响应（Events / TreeDump / TooNew）。这里补充****完全没有 KV events**** 场景下的行为：

﻿`--no-router-kv-events`（approximate mode）：

- Router ****不订阅任何 KV events****；
    
- Indexer 通过 ****routing decision 自预测**** 缓存状态——每次 Router 派发一个请求，就假设 winner 会真的缓存这个 prefix，直接 apply 一条合成的 `Stored` `KvCacheEvent` 到 trie；
    
- 通过 ****TTL 剪枝****（`router_ttl_secs=120.0` 默认）确保自预测状态不会永远累积。
    

****关键澄清****（对旧文档的又一处纠正）：`--no-router-kv-events` ****不是关掉 Indexer****——Indexer 仍然运行，只是数据源从 "worker 上报事件" 换成了 "Router 自己的路由决策"。这个模式在事件平面不稳定或调试场景有用。

﻿`--router-reset-states`（JetStream-only）：

- 启动某 component 的****第一个 replica****时可用；
    
- 会 purge "the entire stream and radix snapshot"；
    
- 用于清理陈旧状态、重新开始。
    

### 10.4 Request Migration：连接级故障恢复

****Feature Matrix 状态****（v1.2.0）：SGLang ✅、vLLM ✅、TRT-LLM 🚧（multimodal 场景 in-progress）。

****触发条件****（`docs/user-guides/fault-tolerance/request-migration`）：

****只有两类连接级故障****，****不涉及**** worker OOM / preemption：

- ****New Request Migration****：__"Worker is unreachable when creating the initial connection"__（communication system reports chosen worker instance is unavailable）
    
- ****Ongoing Request Migration****：__"Connection lost during active generation after partial responses have been received"__（stream termination before generation completion）
    

****关键澄清****：Migration ****只处理连接层面的故障****（比如 pod restart / network partition），不处理 GPU OOM 或引擎内部 preemption——那些属于引擎侧的错误，不会触发 migration。

****目标 worker 的选择路径****：官方原话是 __"A fresh stream is created with the accumulated request state"__，****但没有明确说走 KV Router 重新选目标****——这属于 v1.2 官方文档未覆盖的开放问题。合理推断是复用同一套 Router 决策（否则新 stream 无 worker 可选），但未找到明确源码或文档证据。

****配置****：

|   |   |   |
|---|---|---|
|Flag / env|默认|说明|
|`--migration-limit` / `DYN_MIGRATION_LIMIT`|`****0****`|****默认关闭 migration****；> 0 才启用（数字表示最大迁移次数）|
|`--migration-max-seq-len` / `DYN_MIGRATION_MAX_SEQ_LEN`|`None`|超过该序列长度就不再迁移；边界值 "exactly at the limit is still migratable"|

  

****已知不支持的场景****：

- ﻿`n > 1` 的 OpenAI-compatible 请求（多样本采样）
    
- Guided decoding / structured output（FSM 状态无法在新 worker 上重建，会导致 __"corrupted output — typically duplicated or nested JSON"__）
    

****Metrics****（frontend `/metrics`）：

- ﻿`dynamo_frontend_model_migration_total{model, migration_type ∈ {new_request, ongoing_request}}`﻿
    
- ﻿`dynamo_frontend_model_migration_max_seq_len_exceeded_total{model}`﻿
    
- ﻿`dynamo_frontend_model_migration_limit{model}`（gauge）
    

### 10.5 Request Cancellation：三层 context 传播

****Feature Matrix****：vLLM ✅、TRT-LLM ✅（有 caveat）、****SGLang 🚧****。

****SGLang 为什么是 🚧****：

> "Cancellation during the remote prefill phase is not supported in disaggregated mode."

也就是说 SGLang 在 disaggregated 部署里目前只支持 ****decode 阶段**** cancel，****remote prefill 阶段的 cancel path 没接通****。从 cancellation 传播机制角度理解——它依赖 `AsyncEngineContext` 的 child linkage 把 frontend 的 cancel 信号传到 worker 的 prefill sub-request context，SGLang 那侧的 prefill sub-context 目前还没接进 Dynamo 的 context-linking 机制。

****TRT-LLM 的 caveat****：__"the TensorRT-LLM engine is temporarily not notified of request cancellations, meaning allocated resources for cancelled requests are not freed."__ ——虽然标 ✅，但 engine 侧其实收不到 cancel 通知，资源直到请求"自然结束"才被释放。

****Cancellation 的整体机制****（跨后端通用）：

- ****Rust trait**** `****AsyncEngineContext****`：方法 `id()`, `is_stopped()`, `is_killed()`, `stopped()`, `killed()`, `stop_generating()`, `stop()`, `kill()`, `link_child()`﻿
    
- ****Python 侧**** `****Context****` ****类****：`async_killed_or_stopped()` 通过 `tokio::select!` 组合两个 Rust async 方法
    
- ****传播链****：Frontend 探测 client disconnect → cancel 自己的 `AsyncEngineContext` → worker runtime 通过 control message 或 TCP 断开检测到 → 触发已 link 的 child context（比如 remote prefill 子请求）
    

> 来源：`docs/fault-tolerance/request-cancellation.md`；`docs/development/backend-guide.md#request-cancellation`。

### 10.6 Multimodal + KV Routing：一个"看起来不支持但其实支持"的场景

****Feature Matrix****（v1.2.0）：Multimodal (Image) SGLang ✅、TRT-LLM ✅、vLLM ✅。但 Feature Matrix 的一个 footnote 说：

> Multimodal + KV-Aware Routing: ****"Not supported."****

这个 footnote 容易被误解为 "Dynamo 完全不支持 multimodal + KV routing"，实际情况更细致（`docs/user-guides/multimodal/multimodal-kv-routing`）：

****TRT-LLM 路径****：使用专门的 ****MM Router Worker****——一个中间组件，pipeline 是：

Download image → Compute mm_hash → Build per-block MM metadata → KvRouter selects best worker

Worker 需 `--publish-events-and-metrics`。

****SGLang 路径****：走 Rust frontend path，不走独立 MM Router Worker，__"engaged automatically when the worker reports backend_framework="sglang""__。

- SGLang 的 cache key 由 token IDs 派生，Dynamo 用一个替换 pad 值把 image hash 注入 tokens：
    

pad_value = MM_PAD_SHIFT_VALUE + (mm_hash % 2^30)

- ****关键 caveat****：__"This requires the sglang fork with the mm_hashes field; without it, MM-aware routing silently degrades to text-prefix-only"__
    

也就是说，SGLang 侧要用一个特定 fork（带 `mm_hashes` 字段）才能真正做到 image-aware KV routing；没有 fork 时会****静默降级****到 text-only overlap（同 image 不同 text 的两个请求会被认为 prefix 完全一致，从而错误路由）。****这是 "Multimodal + KV-Aware Routing 不支持" footnote 的真实含义****——不是 Dynamo 拒绝支持，而是主线 SGLang 缺字段导致降级。

****KV cache vs embedding cache 的区分****（这个 doc 里明确区分）：

> "KV cache is separate from embedding cache (also called encoder cache), which reuses vision encoder outputs."

三个可选优化互相独立：CPU-side LRU embedding cache / 独立 vision encoder worker（可独立扩缩）/ MM-aware KV routing。

### 10.7 Speculative Decoding：Router 层未暴露 draft/target 分流

****Feature Matrix****：TRT-LLM ✅、vLLM ✅、SGLang 🚧。

****SGLang 🚧 原因****：__"code hooks exist (spec_decode_stats in publisher) but no examples or documentation yet."__

****Draft/target 是否共享 Router****：****官方文档未明确说明****。从 Feature Matrix 的语义判断，当前主要覆盖 vLLM/TRT-LLM 引擎内部的 spec decoding，****Router 层未暴露 draft/target 分流概念****。仅在 Planner 侧有一条相关信号——`docs/components/planner/planner-guide` 提到 __"KV hit rate and speculative decode accept length are runtime engine/router signals"__，说明 accept length 会通过 router 侧信号面暴露给 planner 做扩缩决策，但****Router 决策本身不感知****这一点。

****写作建议****：如果读者需要 spec decoding + KV router 组合，这属于 v1.2 未覆盖场景，需要单独验证。

---

## 11. Observability：Router 全部 Prometheus 指标

对齐 verl §14 / RL-Router §9 的完整度，本节列出 Router 相关的所有 Prometheus 指标。这些指标是排障的第一线工具，也是理解 Router 内部行为的入口。

### 11.1 每请求指标（`--router-mode kv` 才 populated）

Label：`router_id`。

|   |   |   |
|---|---|---|
|指标|类型|含义|
|`dynamo_component_router_requests_total`|Counter|累计路由请求数|
|`dynamo_component_router_time_to_first_token_seconds`|Histogram|TTFT|
|`dynamo_component_router_inter_token_latency_seconds`|Histogram|ITL|
|`dynamo_component_router_input_sequence_tokens`|Histogram|ISL 分布|
|`dynamo_component_router_output_sequence_tokens`|Histogram|OSL 分布|
|`****dynamo_component_router_kv_hit_rate****`|Histogram (0.0–1.0)|****预测的 KV cache 命中率****——排障 cache-aware 效果最直接的指标|
|`dynamo_component_router_kv_transfer_estimated_latency_seconds`|Histogram|disaggregated 场景 prefill→first-token 估算|
|`dynamo_component_router_shared_cache_hit_rate`|Histogram|shared pool 命中率|
|`dynamo_component_router_shared_cache_beyond_blocks`|Histogram|shared pool 命中中超出 device overlap 的部分|

  

****注意****：非 KV mode 下这些 histogram __"always zero (registered but never populated)"__——所以在 basic mode（如 round-robin）下看这些指标是永远 0。

### 11.2 Router 路由开销（每请求 ms 级）

Label：`router_id`。这些指标 histogram 的 bucket 是毫秒级。

|   |   |
|---|---|
|指标|阶段|
|`dynamo_router_overhead_block_hashing_ms`|Block hash 计算（compute buckets）|
|`dynamo_router_overhead_seq_hashing_ms`|Sequence hash 计算|
|`dynamo_router_overhead_indexer_find_matches_ms`|****Indexer 查询延迟****——mpsc actor 模型下的 channel roundtrip 时间|
|`dynamo_router_overhead_scheduling_ms`|Scheduler 打分与选择|
|`dynamo_router_overhead_total_ms`|Router 主决策路径总耗时|
|`dynamo_router_overhead_shared_cache_query_ms`|Shared cache 查询延迟|
|`dynamo_router_shared_cache_errors_total`|Shared cache 查询失败次数|

  

### 11.3 Queue 相关指标

Labels：`model`, `worker_type ∈ {prefill, decode}`, `policy_class`。

|   |   |   |
|---|---|---|
|指标|类型|含义|
|`dynamo_frontend_router_queue_pending_requests`|Gauge|队列中等待的请求数|
|`dynamo_frontend_router_queue_pending_isl_tokens`|Gauge|队列中总 ISL|
|`dynamo_frontend_router_queue_pending_cached_tokens`|Gauge|队列中总 cached tokens|
|`****dynamo_frontend_router_queue_backpressure_total****`|Counter (label:`reason`)|****拒绝/等待事件****，reason ∈ `{request_limit, raw_isl_token_limit, cached_token_limit}`|

  

### 11.4 Worker 负载 gauge

Labels：`worker_id`, `dp_rank`, `worker_type`。

|   |   |
|---|---|
|指标|含义|
|`dynamo_frontend_worker_active_decode_blocks`|该 worker 当前 decode 用的 KV blocks|
|`dynamo_frontend_worker_active_prefill_tokens`|该 worker 排队的 prefill tokens|
|`dynamo_frontend_worker_last_time_to_first_token_seconds`|最近 TTFT|
|`dynamo_frontend_worker_last_input_sequence_tokens`|最近 ISL|
|`dynamo_frontend_worker_last_inter_token_latency_seconds`|最近 ITL|

  

### 11.5 KV Event 应用统计

|   |   |
|---|---|
|指标|含义|
|`dynamo_component_kv_cache_events_applied`|四类 status × 三类 event_type（见 §5.5）|
|`dynamo_component_kv_cache_engines_dropped_events_total`|通过 event_id gap 检测出的丢件量|
|`zmq_events_total`|ZMQ ingress 总事件数（labels:`stage`, `event_type`）|
|`zmq_filtered_events_total`|ZMQ 侧被过滤事件（labels:`event_type`, `reason`）|
|`zmq_conversion_issues_total`|ZMQ 侧转换异常|
|`zmq_suspicious_events_total`|ZMQ 侧可疑事件（如 event_id 大幅跳变）|

  

### 11.6 Router 内部 status

|   |   |
|---|---|
|指标|含义|
|`dynamo_component_router_worker_registered`|`1` 表示该 worker/dp_rank 已注册|
|`dynamo_component_router_kv_event_source_mismatch_workers`|应发 KV event 但缺失 indexer query endpoint 的 worker 数|
|`dynamo_component_router_remote_indexer_query_failures_total`|远端 Indexer 查询失败次数|
|`dynamo_component_router_remote_indexer_write_failures_total`|远端 Indexer 写失败|

  

### 11.7 与姊妹方案的观测点对照

|   |   |   |   |
|---|---|---|---|
|语义|verl`KVCAwareBalancer`|RL-Router`session_v5`|Dynamo|
|KV cache 使用率|`vllm:kv_cache_usage_perc` (HTTP polling)|`availableBlocks / totalBlocks` (HTTP polling)|`dynamo_frontend_worker_active_decode_blocks` (Router 本地记账，无 polling)|
|Running 请求数|`vllm:num_requests_running`|`runningCount`|无直接单一 gauge；`active_decode_blocks + active_prefill_tokens` + `active_sequences.len()` 组合|
|Waiting 请求数|`vllm:num_requests_waiting`|`waitingCount`|`dynamo_frontend_router_queue_pending_requests`|
|Cache 命中率|无（可从 hit 分布反推）|无（可从 score breakdown 反推）|`****dynamo_component_router_kv_hit_rate****` ****直接暴露****|
|路由开销|无|无|`****dynamo_router_overhead_*_ms****` ****全套****|

  

****Dynamo 观测面的独特之处****：

1. 直接暴露"预测的 cache 命中率"（`kv_hit_rate`）——这是 verl / RL-Router 都没有直接给的指标。
    
2. 完整的路由开销 histogram（block_hashing / indexer_find_matches / scheduling / total）——可用于诊断 mpsc actor 模型下的 channel 延迟。
    
3. Shared pool 相关的独立指标线。
    
4. Event 层面的丢件、gap、conversion 异常指标——可用于诊断事件平面稳定性。
    

### 11.8 暴露方式

- 默认 Router metrics 走 ****Frontend**** `****/metrics****`****（默认 8000 端口）****
    
- 独立 Router 组件通过 `DYN_SYSTEM_PORT`（默认 `-1`，即 disabled）
    
- Metric family 是****懒注册****的——__"each label set is created the first time it fires, so a freshly-started process shows empty metric families until the first relevant request"__——所以刚启动的 Router 看不到指标是正常的，需要有请求进来才会出现。
    

---

## 12. 横向对比

前面各节在关键节点已经穿插了小段对比。这一节把所有对比整合到一起，给出一份**"读完这一节就能对上号所有姊妹方案"**的横向速查。

### 12.1 与 verl `KVCAwareBalancer`（`#6712 + #6940`）

****共同点****：

- 都是 ****cache-state-aware****（不是 cache-history-aware）——都基于引擎主动上报事件维护"哪个 block hash 被哪个 worker 缓存"的反向索引。
    
- 都权衡 cache 收益与负载成本。
    
- 都用 sticky session bonus + KV overlap credit + load penalty 三项组合决策。
    

****核心差异****：

|   |   |   |
|---|---|---|
|维度|verl`KVCAwareBalancer`|Dynamo KV Router|
|状态更新机制|vLLM ZMQ block events（订阅 + replay socket 超时 5s 降级为纯订阅）+ Prometheus polling（5s）|Event Plane（ZMQ / NATS 双通路）+ Router 本地`ActiveSequences` 记账（sub-ms）|
|Cache 状态感知|只有 GPU tier，`get_tier_prefix_hit_rate` v1 恒返回 0.0，v2 计划接 Mooncake|****真实的四层 tier****（device 1.0 / host 0.75 / disk 0.25 / shared 0.5），源码 `worker_logit` 里直接算|
|Block hash 算法|aibrix chained xxhash（`hash.py`），parent_hash 混入|Per-block XXH3-64（seed=1337 + salt）+ chained SequenceHash 独立|
|负载信号来源|Prometheus polling（kv_usage / running / waiting，5s 延迟）|Router 本地`ActiveSequences`（`&mut self` 无锁记账，sub-ms）|
|Sticky 机制|`STICKY_TOP_SCORE=1e9` 硬优先级 + `load_threshold=0.9` fallback|未内建 sticky（可用`--router-session-affinity-ttl-secs` 外挂）|
|打分方向|`Score = α·S_cache + (1-α)·S_load`（argmax）|`cost = w_prefill·adjusted + decode_blocks`（argmin）|
|P/D 分离|未明确实现（RFC 阶段）|完整（PrefillRouter + credit-override decode）|
|引擎支持|只 vLLM|vLLM / TRT-LLM / SGLang（≥ 0.5.11 tier-aware）|
|Softmax 采样|无|✅`router_temperature > 0` 触发|
|Filter/Scoring 分离|一体式打分|明确分离（`docs/components/router/router-filtering.md`）|
|Priority 支持|无|三层 priority（queue / engine / cache）+ agent_hints|
|冷启动 replay|replay socket +`b"replay"` request（超时 5s 降级）|Events / TreeDump / TooNew 三态响应，覆盖增量与全量|
|合入状态|`#6712 + #6940` 均未合并|生产可用（v1.2.1）|
|实现语言|Python|Rust（核心）+ Python（CLI）|

  

****Dynamo 相对 verl 的关键优势****：

1. ****真实的四层 tier credit****——verl 的 `layer_weights={gpu:0.7,cpu:0.2,ssd:0.1}` 里 cpu/ssd 恒为 0；Dynamo 的 host/disk 是真的能生效（前提是 SGLang HiCache ≥ 0.5.11）。
    
2. ****Filter/Scoring 明确分离****——verl 是一体式打分公式，Dynamo 把硬约束（filter）与软信号（scoring）架构上分开，可维护性更好。
    
3. ****Softmax 采样****——避免 argmin 造成的瞬时热点（verl 只能靠 `load_threshold` 触发 fallback）。
    
4. ****Replay 三态****——覆盖增量补齐 + 全量恢复 + 消费者超前，比 verl 的 "5s 超时降级" 更完整。
    

****verl 相对 Dynamo 的关键优势****：

1. ****Sticky session 一等公民****——verl 明确用 `request_id → replica` 绑定，多轮 Agent 场景直接受益；Dynamo 需要外挂 session affinity 或依赖 cache hit 打分间接实现。
    
2. ****实现更轻量****——纯 Python，rollout 场景改造成本低（尽管 Router 本身是 Ray remote）。
    

### 12.2 与 RL-Router 三策略

RL-Router 的三个策略（`cache_aware` / `pd_cache_aware` / `session_aware_v5`）与 Dynamo 的对应关系：

|   |   |
|---|---|
|RL-Router 策略|与 Dynamo 的对应|
|`cache_aware`|类似`--router-mode kv` 的最基础形态，byte-level 前缀匹配 vs Dynamo 的 token-level block hash|
|`pd_cache_aware`|类似 Dynamo`PrefillRouter` + `KvRouter` 组合的****简化版****，都是 block-hash + 加权打分|
|`session_aware_v5`|类似 Dynamo`--router-session-affinity-ttl-secs` + `ActiveSequences` 组合，但 sticky 是一等公民|

  

****Dynamo 相对 RL-Router 的独有能力****：

1. ****真实的 cache-state-aware****——RL-Router 三策略都是 cache-history-aware（本地 radix 反推），Dynamo 是事件驱动的真实状态。
    
2. ****Tier-aware credit****——RL-Router 的三个策略都没有 tier 概念（全都只看 GPU）。
    
3. ****Per-request override****——`RouterConfigOverride` 允许客户端 per-request 调整 6 个字段；RL-Router 的所有权重都是全局配置或 HTTP payload override。
    
4. ****官方 event 平面****——RL-Router 用 HTTP polling + 本地反推，Dynamo 用事件流。
    
5. ****P/D 独立组件****——Dynamo 有专门的 `PrefillRouter`；RL-Router 通过 `AllocatePD` RPC + 双策略架构外部化。
    
6. ****Softmax 采样****——RL-Router 三策略都是 deterministic argmin。
    

****RL-Router 相对 Dynamo 的独有能力****：

1. ****块紧急态迁移****（`session_aware_v5` 的 `BlockEmergencyThresh=128`）——Dynamo 通过 filter（`--active-decode-blocks-threshold`）来做类似的事，但 filter 是"直接剔除"，不是"迁移"，语义略不同。
    
2. ****HTTP payload 里的**** `****cache_threshold****` ****override****——RL-Router 可 per-request 传阈值；Dynamo 通过 `RouterConfigOverride` 传权重，语义类似但字段不同。
    
3. ****Session affinity warmth****（`effectiveThreshold(requestCount)`）——RL-Router 有专门的"越粘越粘"抗抖动机制；Dynamo 用 Softmax + Preferred taint 达到类似目的但机制不同。
    

### 12.3 与 SGLang Model Gateway `cache_aware`﻿

|   |   |   |
|---|---|---|
|维度|SGLang Gateway`cache_aware`|Dynamo KV Router|
|语言|Rust|Rust|
|命中判定|Gateway 侧近似前缀树（byte 级，请求历史反推）|Router 侧真实 KV block 索引（事件驱动）|
|精度|Text 字符级|Token block 级|
|引擎依赖|引擎无关（不需要事件）|需要 KV event 上报（tier-aware 需 SGLang ≥ 0.5.11）|
|Load 定义|active_requests 原子计数|`ActiveSequences.blocks.potential_decode_blocks()`（`&mut self` 记账）|
|过载保护|双阈值 abs + rel 触发|Busy-threshold filter（`--active-*-threshold`）+ Overload eligibility check|
|Sticky|无显式 sticky|可选`--router-session-affinity-ttl-secs`|
|Multi-pool 隔离|✅`pool::model` key|✅`component/endpoint` 层级|
|P/D 分离|✅`--prefill-policy` / `--decode-policy` 分开配|✅ PrefillRouter + Decode override|
|集群同步|mesh sync（可选）|`--router-replica-sync` NATS core|
|合入状态|SGLang 生态默认策略|Dynamo 生产可用|

  

****Dynamo 相对 SGLang Gateway 的关键差异****：

- Dynamo 是****token/block 级 + 全局事件驱动****，Gateway 是****text 级 + 本地近似****。
    
- Dynamo 需要****KV event 上报基础设施****（vLLM ZMQ / SGLang HiCache）；Gateway 完全解耦。
    
- Dynamo 的 tier-aware 需要 SGLang ≥ 0.5.11；Gateway 无版本要求。
    

****选型建议****：

- 如果 workload 主要是 aggregated 部署 + 短请求 + 无 tier 需求 → SGLang Gateway `cache_aware` 部署更简单。
    
- 如果需要 P/D 分离、tier-aware、或者 workload 是多轮长 context RL rollout → Dynamo KV Router 收益更大。
    

### 12.4 一张八列大表汇总

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|维度|verl`KVCAwareBalancer`|RL-Router`pd_cache_aware`|RL-Router`session_v5`|SGLang Gateway`cache_aware`|****Dynamo KV Router****|
|****信息来源****|vLLM ZMQ + Prom (5s)|本地反推 radix + 本地 counter|HTTP polling (2s) + 本地 counter|本地近似前缀树 + 本地 counter|****Event Plane +**** `****ActiveSequences****` ****记账****|
|****信息类型****|cache-state-aware（GPU only）|cache-history-aware|无 cache 感知（session sticky）|cache-history-aware|****cache-state-aware（4 层 tier）****|
|****Hash 粒度****|vLLM chained xxhash（block）|FNV1a chained（block, size=64）|无（用 sessionID）|Byte-level radix|****XXH3-64 per-block + chained SequenceHash****|
|****打分方向****|argmax（Score 越大越好）|argmin（cost 越小越好）|argmin（复合 score 越小越好）|阈值判断 + argmin|****argmin logit**** + softmax(T) 采样|
|****过载保护****|`load_threshold=0.9`|`isLoadImbalanced` 全局判据|`block_emergency` + `active_load`|双阈值 abs+rel|Filter（busy-threshold）+ Preferred taint 乘子|
|****P/D 分离****|❌|✅（`AllocatePD` RPC + 双策略）|❌|✅（`--prefill-policy` / `--decode-policy`）|****✅****（PrefillRouter + credit override decode）|
|****Sticky****|一等公民（`STICKY_TOP_SCORE`）|无|一等公民（`sessionAssign` + warmth）|无|可选（`--router-session-affinity-ttl-secs`）|
|****引擎支持****|只 vLLM|与后端解耦（默认 vLLM 格式）|与后端解耦|任意|vLLM / TRT-LLM / SGLang|
|****合入状态****|未合并|生产运行|生产运行|生产可用|生产可用（v1.2.1）|

  

---

## 13. 常见误区（含对旧文档结论的修正）

以下 10 条误区涵盖了本文写作过程中发现的、旧文档 `Dynamo-KV-Cache-Router-机制详解.md` 或社区传播中常见的错误理解。

### 误区 1：KV-aware Router 会直接读取 GPU 上的 KV tensor

****不对****。Router 消费的是"哪个 block hash 被哪个 worker 缓存"的****元信息****，不是 KV tensor 本体。真实数据传输由 backend / NIXL / KVBM / HiCache 完成。

### 误区 2：`overlap_score_credit` 是总的 cache credit 权重

****不对****。它****只是 device credit****（GPU tier 的权重）。完整公式有四层独立 credit：

- device credit：`overlap_score_credit`（默认 1.0）
    
- host credit：`host_cache_hit_weight`（默认 0.75）
    
- disk credit：`disk_cache_hit_weight`（默认 0.25）
    
- shared credit：`shared_cache_multiplier`（默认 0.0；CLI 打开 shared 时 0.5）
    

（这也纠正了旧 `Dynamo-KV-Cache-Router-机制详解.md` §6 里 "gpu_credit / host_credit / remote_credit" 三分那种直觉表达——实际是****四层****，且名字与文档参数不一致。）

### 误区 3：argmin 总选最低 cost worker

****不总是****。`--router-temperature=0`（默认）时是 deterministic argmin，但 temperature > 0 时走 softmax 采样——通过 `scale = -1/((max-min)*T)` 把 argmin 翻成 argmax(prob)，低 cost 有更高被选中概率但不是必选。此外 tie 情况用 reservoir sampling 打破。

### 误区 4：Block hash 是 chained rolling hash

****不对****。`LocalBlockHash` 是****每个 block 独立****的 XXH3-64（seed=1337 + namespace/LoRA salt）——相邻 block 的 hash ****没有依赖****。真正的 chained hash 是 `SequenceHash`（`compute_next_sequence_hash(parent_seq_hash, current_block_hash)`）。事件里两者都携带：`tokens_hash: LocalBlockHash` + `block_hash: ExternalSequenceBlockHash`。

（这纠正了旧文档 §6 里 "prefill_blocks / overlap_blocks 都是 chained hash 匹配" 的暗示。）

### 误区 5：P/D 分离下 Decode worker 是 round-robin / least-loaded

****不对****。Decode 侧****仍走**** `****KvRouter::find_best_match_details_with_policy_class****`——是同一个 Router 主体，只是通过 `build_decode_router_override` 强制覆盖 `overlap_score_credit=0 / assume_kv_reuse=false / track_prefill_tokens=false`，让打分公式退化为 `prefill_load_scale * raw_prefill_blocks + decode_blocks`。

（旧文档 §9 讲 "更关心 decode worker 的负载、可用 slot、拓扑" 容易被误读为 "走独立的负载均衡策略"，但源码事实是 credit-override 后的同一个 Router。）

### 误区 6：KVBM 是 HiCache 的进阶 / KVBM 缺失导致 SGLang 无法 tier-aware

****不对****。官方明确写 KVBM 是 HiCache 的 ****alternative****（`docs/backends/sglang/sglang-hicache.md`）——****两条互斥路线****。SGLang 用户走 HiCache + NIXL，Dynamo Router 通过消费 HiCache 的 `medium` 字段做 tier-aware routing。KVBM 对 SGLang 🚧 影响的是"用 Dynamo 自己的块管理器替代 HiCache"这条路径，****不影响**** SGLang HiCache + Dynamo Router 的 tier-aware routing。

（这个误区旧文档 §10 已经修正过，但仍有传播；本文重申。）

### 误区 7：`--no-router-kv-events` 会关掉 Indexer

****不对****。`--no-router-kv-events` ****只是禁用 event 订阅****，Indexer 仍然运行——只是数据源换成"Router 自己的路由决策自预测"，并通过 `router_ttl_secs=120.0` 做 TTL 剪枝。这个模式叫 approximate mode，用于事件平面不稳定或调试。

### 误区 8：Worker 掉线时 Router 会收到大量 RemoveEvent

****不对****。Worker 掉线时（etcd lease 过期），Router 侧不是逐 block 发 RemoveEvent，而是 ****subtree wholesale purge****——一次性把该 worker 在全局 radix tree 上的整个 subtree 丢弃。这与正常运行时的 block 级驱逐（`Removed` event 配对 `Stored`）是两种机制。

### 误区 9：Planner 会替代 Router 做请求调度

****不对****。Planner 是 ****autoscaling controller****（按 TTFT/ITL SLA 扩缩 prefill/decode 副本数），Router 是 ****request scheduler****。两者****并列部署****，Planner 消费 Router 的 metrics 做扩缩决策，****不介入单请求路由****。

### 误区 10：Multimodal + KV routing 在 Dynamo 上完全不支持

****不对****。Feature Matrix 的 footnote 说 "Not supported" 是简化说法，实际上：

- ****TRT-LLM****：走 MM Router Worker，完整支持 image-aware routing。
    
- ****SGLang****：走 Rust frontend + `mm_hashes` 注入 pad tokens；****但需要 sglang fork 带**** `****mm_hashes****` ****字段****，主线 SGLang 会****静默降级****到 text-only overlap（同 image 不同 text 的两个请求被误认为 prefix 一致）。
    

也就是说，"Dynamo 完全不支持" 是错的——正确的表述是 "主线 SGLang 侧字段缺失导致 SGLang backend 上 multimodal KV routing 降级为 text-only；TRT-LLM 完整支持"。

### 误区 11：Dynamo Router 内部有原子 in-flight 计数器

****不对****。Dynamo 全部用 ****actor + mpsc +**** `****&mut self****` ****模型****——`KvIndexer` 单线程持有 `RadixTree`（无锁）、`ActiveSequences` 每 worker 一份 `&mut self` 更新（无锁）、`PrefillLoadTracker` 用普通 `usize` 累加。全仓库只有 `zmq_listener` 里的 `event_id` 分配用 `AtomicU64::fetch_add(1, SeqCst)`。这与 verl / RL-Router 使用 atomic counter 或 map+Mutex 是显著的架构差异。

---

## 14. 调参与排障建议

对齐 verl §14 / RL-Router §9.1 的"现象 → 优先怀疑 → 检查方向"格式。

### 14.1 常见现象排障表

|   |   |   |
|---|---|---|
|现象|优先怀疑|检查方向|
|KV hit rate 低（`dynamo_component_router_kv_hit_rate` 低）|KV events 未生效 / block_size 不匹配 /`overlap_score_credit=0`|查`dynamo_component_kv_cache_events_applied` 中 `status=ok` 占比；对比 `--kv-cache-block-size` 与后端 `--block-size` / `--page-size`；确认 `--router-kv-events` 未被 `--load-aware` 关掉|
|某 worker 特别热 / 尾延迟差|Credit 过高 / temperature=0 造成热点|提高`--router-temperature`；启用 `--router-kv-overlap-score-credit-decay`；降低 `overlap_score_credit`|
|TTFT 没改善（相似请求打散）|Credit 过低 / block_size 太粗|提高`overlap_score_credit`；减小 `--kv-cache-block-size`（需后端配合）；确认 event 平面正常（`engines_dropped_events_total` 是否为 0）|
|ITL 变差|过度追求 prefill 命中，忽略 decode backlog|降低`overlap_score_credit`；提高 `prefill_load_scale`；启用 decay|
|P/D 分离收益不明显|Prompt 不够长 / NIXL 传输慢|检查 prefill 时间；查`dynamo_component_router_kv_transfer_estimated_latency_seconds`；确认 topology-aware 配置|
|Router 决策延迟高（`overhead_total_ms` 高）|Indexer mpsc 拥塞 / 候选 worker 太多|查`overhead_indexer_find_matches_ms` 分位；考虑 `--router-event-threads` 增加；剔除慢 worker|
|SGLang 上 tier-aware routing 无效|SGLang 版本 < 0.5.11 / HiCache 未启用|确认 SGLang ≥ 0.5.11；确认 worker 启动带`--enable-hierarchical-cache`；确认 `medium` 字段在事件里出现|
|Shared cache 命中率恒为 0|`--shared-cache-type=none`|启用`--shared-cache-type=hicache` + Mooncake 配置|
|Multi-Router 部署下负载视图不同步|未开`--router-replica-sync`|开启后重启|
|Queue 一直有请求积压|Busy-threshold 过严 / worker 处理慢|放宽`--active-*-threshold`；查 `dynamo_frontend_router_queue_backpressure_total` 的 reason label|
|Router 重启后 cache hit 骤降|Approximate mode 无 replay / JetStream 未启用|走 event-driven mode（默认）；确认 replay 通道正常；考虑`--router-durable-kv-events`（若接受 deprecated 状态）|

  

### 14.2 调参顺序建议

1. ****确认基础事件链路****：先看 `dynamo_component_kv_cache_events_applied` 是否有 `ok` 事件、`engines_dropped_events_total` 是否为 0。基础不通就调 credit 也没意义。
    
2. ****确认 block_size 对齐****：Router `--kv-cache-block-size` 必须与后端引擎一致。不对齐会导致命中率完全失真。
    
3. ****对比**** `****--load-aware****` ****baseline****：先跑一遍 `--load-aware` 得到无 cache 感知的基线 TTFT / ITL，再开 kv mode 对比收益。这是最干净的评估方法。
    
4. ****调**** `****overlap_score_credit****`：从默认 1.0 开始，命中被低估就上调（1.5-2.0），出现热点就下调（0.5-0.7）。
    
5. ****调**** `****router_temperature****`：热点严重时开 0.1-0.5；对确定性要求高的场景保持 0。
    
6. ****启用 decay****：`--router-kv-overlap-score-credit-decay` 从 0.5 起步，观察 cache-rich but overloaded 场景是否改善。
    
7. ****进阶：tier-aware 与 shared cache****：确认 SGLang ≥ 0.5.11 后启用 HiCache；有 Mooncake 部署再开 shared cache。
    
8. ****最后：P/D 分离****：只有当 prompt 足够长（比如 > 4k tokens）+ TTFT 是主瓶颈时才考虑；配合 NIXL 拓扑感知。
    

### 14.3 何时用什么 preset

|   |   |
|---|---|
|场景|推荐配置|
|RL rollout（多轮 Agent，session_id 稳定）|`--router-mode kv` + `--router-session-affinity-ttl-secs=300` + 默认权重|
|长 context aggregated|`--router-mode kv` + `overlap_score_credit=1.5` + `decay=0.5`|
|长 context P/D 分离|PrefillRouter enabled + NIXL + topology-aware`preferred`|
|多样负载，不确定收益|`--load-aware` 做 baseline，再对比 kv mode|
|事件平面调试|`--no-router-kv-events` + `--router-ttl-secs=60`|
|追求 TTFT（interactive）|高`overlap_score_credit`（1.5-2.0）+ 低 `prefill_load_scale`（0.5-0.8）+ decay 启用|
|追求 throughput|默认权重 +`--router-queue-policy=wspt`|

  

---

## 15. 参考资料与源码索引

### 15.1 官方文档

****Router 相关****：

- ﻿[Router Concepts](https://github.com/ai-dynamo/dynamo/blob/main/docs/components/router/router-concepts.md) — cost function 简版 + basic router modes
    
- ﻿[Router Design](https://github.com/ai-dynamo/dynamo/blob/main/docs/design-docs/router-design.md) — cost function 完整式 + KvIndexer + AddRequest/MarkPrefillCompleted/Free 三个 replica sync event
    
- ﻿[Router Filtering](https://github.com/ai-dynamo/dynamo/blob/main/docs/components/router/router-filtering.md) — filter 与 scoring 的边界
    
- ﻿[Router Configuration and Tuning](https://docs.nvidia.com/dynamo/components/router/configuration-and-tuning) — ****权威默认值来源****
    
- ﻿[Router Operations](https://docs.nvidia.com/dynamo/components/router/router-operations) — multi-router / replay
    
- ﻿[KV Event Replay Comparison](https://github.com/ai-dynamo/dynamo/blob/main/docs/components/router/kv-event-replay-comparison.md) — 三态响应 + vLLM ZmqEventPublisher 对比
    
- ﻿[Router Priority Scheduling](https://github.com/ai-dynamo/dynamo/blob/main/docs/components/router/priority-scheduling.md) — 三层 priority 独立
    

****KVBM / HiCache****：

- ﻿[SGLang HiCache Backend](https://github.com/ai-dynamo/dynamo/blob/main/docs/backends/sglang/sglang-hicache.md) — ****本文 tier-aware 章节的核心来源****
    
- ﻿[KVBM Design](https://github.com/ai-dynamo/dynamo/blob/main/docs/design-docs/kvbm-design.md) — G1/G2/G3/G4 分层 + Connector API
    
- ﻿[KVBM README](https://github.com/ai-dynamo/dynamo/blob/main/docs/components/kvbm/README.md) — 框架支持矩阵
    
- ﻿[vLLM KV Offloading](https://github.com/ai-dynamo/dynamo/blob/main/docs/backends/vllm/vllm-kv-offloading.md) — KVBM / LMCache / FlexKV 三种 backend 并列
    

****P/D 分离 / NIXL****：

- ﻿[Disaggregated Serving Design](https://github.com/ai-dynamo/dynamo/blob/main/docs/design-docs/disaggregated-serving.md) — bootstrap_info / kv_transfer_params / opaque_state
    
- ﻿[Topology-aware KV Transfer](https://docs.nvidia.com/dynamo/kubernetes-deployment/operate/topology-aware-kv-transfer)﻿
    
- ﻿[NIXL repository](https://github.com/ai-dynamo/nixl) — plugin 列表
    

****周边能力****：

- ﻿[Feature Matrix](https://docs.nvidia.com/dynamo/resources/feature-matrix) — v1.2.0 backends 特性对照
    
- ﻿[Request Migration](https://docs.nvidia.com/dynamo/user-guides/fault-tolerance/request-migration)﻿
    
- ﻿[Request Cancellation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fault-tolerance/request-cancellation.md)﻿
    
- ﻿[Multimodal KV Routing](https://docs.nvidia.com/dynamo/user-guides/multimodal/multimodal-kv-routing)﻿
    
- ﻿[Discovery Plane](https://github.com/ai-dynamo/dynamo/blob/main/docs/design-docs/discovery-plane.md)﻿
    
- ﻿[Planner Design](https://github.com/ai-dynamo/dynamo/blob/main/docs/design-docs/planner-design.md)﻿
    
- ﻿[Metrics / Observability](https://github.com/ai-dynamo/dynamo/blob/main/docs/observability/metrics.md)﻿
    

****版本坐标****：

- ﻿[Dynamo v1.2.1 Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.1) — 2026-06-13 稳定版
    

### 15.2 源码索引

|   |   |
|---|---|
|用途|文件|
|KvRouter 顶层|`lib/llm/src/kv_router.rs`|
|Scheduler 薄封装|`lib/llm/src/kv_router/scheduler.rs`|
|LocalScheduler 核心|`lib/kv-router/src/scheduling/local.rs`|
|****打分公式 & softmax****|`lib/kv-router/src/scheduling/selector.rs`|
|Queue admission|`lib/kv-router/src/scheduling/queue_admission/`|
|Overlap refresh|`lib/kv-router/src/scheduling/overlap_refresh.rs`|
|KvRouterConfig 默认|`lib/kv-router/src/scheduling/config.rs`|
|****Block hash & event schema****|`lib/kv-router/src/protocols.rs`|
|KvIndexer actor|`lib/kv-router/src/indexer/kv_indexer.rs`|
|Concurrent radix tree|`lib/kv-router/src/indexer/concurrent_radix_tree_compressed/`|
|Indexer enum wrapper|`lib/llm/src/kv_router/indexer/mod.rs`|
|ZMQ event ingress|`lib/llm/src/kv_router/publisher/zmq_listener.rs`|
|Event coalescer|`lib/llm/src/kv_router/publisher/event_processor.rs`|
|NATS/EventPlane sink|`lib/llm/src/kv_router/publisher/sinks.rs`|
|****ActiveSequences****|`lib/kv-router/src/sequences/single.rs`|
|PrefillLoadTracker|`lib/kv-router/src/sequences/prefill_tracker.rs`|
|BlockTracker|`lib/kv-router/src/sequences/block_tracker.rs`|
|Multi-worker sequences|`lib/kv-router/src/sequences/multi_worker.rs`|
|****PrefillRouter****|`lib/llm/src/kv_router/prefill_router/mod.rs`（+ activation.rs / admission.rs / query.rs）|
|PushRouter wrapper|`lib/llm/src/kv_router/push_router.rs`|
|****KV Router metrics****|`lib/llm/src/kv_router/metrics.rs`|
|CLI: KV router 参数|`components/src/dynamo/common/configuration/groups/kv_router_args.py`|
|CLI: router mode/admission|`components/src/dynamo/common/configuration/groups/router_args.py`|
|CLI: frontend router-adjacent|`components/src/dynamo/frontend/frontend_args.py`|
|Router 组件独立 CLI|`components/src/dynamo/router/args.py`|
|RouterConfig struct|`lib/llm/src/entrypoint.rs`|

  

### 15.3 姊妹调研文档（本仓库）

- ﻿`****dynamo/RL-Router调度策略.md****` — PaddleRL InferRouter 三个策略源码级调研；数据来源三分类的钥匙来自此文档 §3.5，本文借用了这个分类框架。
    
- ﻿`****dynamo/verl KVC Aware Rollout Router 技术调研.md****` — verl `#6712 + #6940` PR 源码级调研；本文 §12.1 的对比基于此文档。
    
- ﻿`****dynamo/Dynamo-KV-Cache-Router-机制详解.md****` — Dynamo KV Router 的旧版调研；本文替代其结论，纠正内容见 §13（尤其 §13 误区 2 / 4 / 5 / 8）。
    

---

__文档止。若源码在后续版本演进导致本文引用失效，以源码为准，本文欢迎补丁。所有引用可通过__ `__文件路径__` __直接跳转到__ `__github.com/ai-dynamo/dynamo__` __对应位置。__