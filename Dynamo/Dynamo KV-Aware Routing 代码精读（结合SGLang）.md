# Dynamo KV-Aware Routing 代码精读（结合 SGLang backend）

> 仓库：`/Users/zhushengguang/CODES/dynamo`
> 文档基准：`docs/components/router/`（router-guide / router-concepts / router-configuration / router-filtering / router-operations / priority-scheduling / deficit-round-robin / kv-event-replay-comparison / multi-dc-kv-routing / topology-aware-kv-transfer 等）
> 行号以当前 checkout 为准，若代码更新以符号名搜索为准。

---

## 0. 全局图景

### 0.1 代码分层

```
components/src/dynamo/frontend/main.py     Python CLI 入口（--router-mode kv）
        │
lib/bindings/python/rust/llm/kv.rs         PyO3 绑定（KvRouter / compute_block_hash_for_seq_py）
        │
lib/llm/src/kv_router.rs + kv_router/      集成层（Rust）：KvRouter/KvPushRouter 门面、
        │                                  事件通路(zmq_listener)、恢复(recovery)、push 转发
lib/kv-router/src/                         核心算法 crate（Rust）：indexer(radix/cuckoo)、
        │                                  scheduling(打分/队列/准入)、sequences(负载追踪)、
        │                                  services(独立服务)、recovery(事件游标)
lib/kv-hashing/                            独立 crate：块哈希计算
        │
components/src/dynamo/sglang/              SGLang backend 集成（Python）：
                                           publisher.py 桥接 SGLang KV events → dynamo
```

### 0.2 一句话架构

Router 通过消费 worker 发布的 KV 事件（`stored`/`removed`），在本地维护一棵全局 radix 树（"谁缓存了哪些前缀块"）；同时用请求生命周期记账维护每个 worker 的活跃负载（prefill token 数、潜在 decode 块数）。路由时对每个候选 `worker × dp_rank` 算一个成本 logit，缓存命中作为 prefill 成本的抵扣项，最后 argmin 或温度采样选择。

### 0.3 三种事件模式（router-concepts.md）

| 模式         | 配置                                                | 前缀状态来源                                     |
| ---------- | ------------------------------------------------- | ------------------------------------------ |
| ZMQ 精确（默认） | backend 开 `--kv-events-config`，event plane 默认 ZMQ | worker 真实 stored/removed 事件                |
| NATS 精确    | `--event-plane nats`                              | 同上，经 NATS 分发（多副本天然共享）                      |
| 近似         | `--no-router-kv-events`                           | router 用自己的路由决策预测 + `--router-ttl-secs` 过期 |

---

## 阶段 1：一条请求的完整路径（主线）

### 1.1 入口

- `components/src/dynamo/frontend/main.py:328`：`--router-mode kv` → `RouterMode.KV`，并从 CLI/env 组装 `KvRouterConfig`（所有 `--router-*` 参数）。
- 环境变量入口：`DYN_ROUTER_MODE=kv`。

### 1.2 KvPushRouter（转发门面）

`lib/llm/src/kv_router/push_router.rs`：

- `KvPushRouter`（106 行）：包 `KvRouter` + 请求转发。
- `generate`（503 行）：tokenize 后的请求进来 → 调 `KvRouter::find_best_match` 拿 `(worker_id, dp_rank, overlap_blocks)` → 把请求 push 给选中的 worker instance → 响应流上挂钩子（首 token 标记 prefill 完成、结束时 free）。
- `query_instance_id` 注解：只查询路由结果不转发（供外部编排用）。

### 1.3 KvRouter（决策核心）

`lib/llm/src/kv_router.rs`：

- `KvRouter`（200 行）：持有 indexer 句柄、`ActiveSequencesMultiWorker`、selector、队列。
- `find_best_match`（771 行）→ `find_best_match_details_with_policy_class_inner`（568 行）：
  1. `compute_block_hash_for_seq(tokens, block_size)` 把 token 序列切块求哈希；
  2. 查 indexer 得到每个 worker 的 overlap（命中块数，分 device/host/disk 层）；
  3. `schedule_request`（687 行）：过滤资格 → 可能入队（背压/DRR）→ selector 打分选 worker；
  4. 把请求登记进 `ActiveSequencesMultiWorker`（占位 active blocks / prefill tokens）。

### 1.4 块哈希

`lib/kv-router/src/protocols.rs`：

- `compute_block_hash_for_seq`（89 行）：按 `block_size` 切块，链式哈希（父块哈希参与子块哈希），与引擎的 prefix cache 块边界对齐——这是路由前缀匹配和引擎实际命中一致的前提。
- `LocalBlockHash`（576 行）：router 自算的哈希。
- `ExternalSequenceBlockHash`（583 行）：引擎事件里自报的哈希。KV 事件建立 external→local 的映射，removed 事件用 external 哈希删除。
- `RouterEvent`（923 行）：事件封装（worker_id + event_id + stored/removed 数据）。
- SGLang 侧对应：page_size 必须等于 router 的 `kv_block_size`（`publisher.py` 里 `kv_block_size=self.server_args.page_size`）。

### 1.5 打分与选择（成本函数精确语义）

`lib/kv-router/src/scheduling/selector.rs`，`DefaultWorkerSelector::worker_logit`（138–266 行），核心逻辑照抄：

```rust
// 1) 原始 prefill 负载（token 记）：worker 当前 active_prefill_tokens
//    + 本请求未命中部分 + 命中部分（保持旧公式的量纲）
raw_prefill_tokens = load.active_prefill_tokens
                   + effective_prefill_tokens(isl, cached)   // = isl - cached，饱和减
                   + cached_tokens;
raw_prefill_blocks = raw_prefill_tokens / block_size;

// 2) 过载衰减（有理衰减，见优化点 2）
excess = (active_prefill_tokens - min_active_prefill_tokens) / block_size;
normalized = excess / request_blocks;
decay = 1 / (1 + overlap_score_credit_decay * normalized);   // decay=0 时恒为 1

// 3) 分层 overlap 抵扣（见优化点 1/3）
overlap_credit_blocks = overlap_score_credit * decay * device_overlap_blocks
                      + host_cache_hit_weight * host_overlap_blocks    // 默认 0.75
                      + disk_cache_hit_weight * disk_overlap_blocks    // 默认 0.25
                      + shared_cache_multiplier * shared_hits_beyond_device;

// 4) 最终成本
logit = prefill_load_scale * (raw_prefill_blocks - overlap_credit_blocks)
      + potential_decode_blocks;   // prefill_load_scale 默认 1.0
```

选择：`softmax_sample`（32–56 行）——`temperature == 0` 直接 `min_by(total_cmp)` 取最小成本；`>0` 时把 logit 取负、缩放到 `[-1/T, 0]` 做数值稳定的 softmax，再按累计概率采样。

选择粒度是 `WorkerWithDpRank`：**直接选到 worker × dp_rank**。对 SGLang DP attention 部署，router 注入 `data_parallel_rank`，SGLang DPC 侧 `maybe_external_dp_rank_routing` 直达指定 rank，DPC 本地负载均衡被旁路（与之前 sgl-model-gateway `--dp-aware` 的机制相同）。

默认值（`lib/kv-router/src/scheduling/config.rs`）：`host_cache_hit_weight=0.75`、`disk_cache_hit_weight=0.25`、`prefill_load_scale=1.0`、`overlap_score_credit_decay=0.0`（关）、`track_prefill_tokens=true`。

---

## 阶段 2：索引器（router 怎么知道谁缓存了什么）

### 2.1 KvIndexer

`lib/kv-router/src/indexer/kv_indexer.rs`：

- `KvIndexer`（228 行）：单线程 tokio actor，串行消费 `RouterEvent`。
- `apply_event`（610 行）：stored → 往 radix 树插入 `(block_hash → worker 集合)`；removed → 删除；带 event_id 连续性校验（乱序/缺口检测）。
- `dump_events`（648 行）：把整棵树导出为合成事件序列（恢复/副本引导用）。
- 查询接口：`find_matches(block_hashes)` → 每个 worker 的最长前缀命中块数（分层：device / host_pinned / disk）。

### 2.2 radix 树三代实现

- `indexer/radix_tree.rs`：基础单线程版。
- `indexer/concurrent_radix_tree.rs`：读写分离并发版（查询不阻塞事件写入）。
- `indexer/concurrent_radix_tree_compressed/`（有 README）：路径压缩版，长公共前缀合并成段，省内存 + 提升缓存局部性。
- `indexer/cuckoo/`：多 DC 用的 Cuckoo filter（见优化点 17）。

### 2.3 近似模式

`indexer/pruning.rs`（`PruneManager`）：`--no-router-kv-events` 时不消费引擎事件，router 在每次路由决策后**把自己刚路由过去的块假装成 stored 插入**，用 `--router-ttl-secs` TTL 过期淘汰。适合事件通路还不可靠的 backend 或开发环境。

### 2.4 事件从哪来

`lib/llm/src/kv_router/publisher/`：

- `zmq_listener.rs`：ZMQ SUB socket，连 worker 的 PUB 端口收事件（msgpack 的 `KVEventBatch`）。
- `event_processor.rs` / `batching.rs` / `dedup.rs`：反序列化、按 event_id 去重、攒批后喂给 KvIndexer。

---

## 阶段 3：负载追踪（成本的另一半）

`lib/kv-router/src/sequences/multi_worker.rs`：

- `ActiveSequencesMultiWorker`（313 行）：每个 `worker × dp_rank` 一份账本。
- `add_request`（743 行）：路由决策时登记——`active_prefill_tokens += uncached_tokens`；`potential_decode_blocks += request_blocks`（含 OSL 提示的分数权重，见优化点 7）。
- 首个响应 token 到达 → 标记 prefill 完成，`active_prefill_tokens` 扣减（或按 AIC 模型时间衰减，见优化点 6）。
- `free`（800 行）：请求结束释放全部占位。
- `replica_free`（1727 行）：处理其他 router 副本同步来的释放事件。
- `scheduling/prefill_load.rs`：`effective_prefill_tokens`（isl−cached 饱和减）与 AIC 衰减模型。
- `sequences/prefill_tracker.rs`：prefill 完成事件的锚点追踪。

关键点：**前缀状态靠事件（最终一致），活跃负载靠请求生命周期记账（router 本地权威）**——两类状态的同步/恢复策略完全不同（见优化点 14）。

---

## 阶段 4：队列与准入

`lib/kv-router/src/scheduling/`：

- `policy.rs`：队列内排序策略。FCFS = 到达时间（priority 作为正向时间提前量，负值截断为 0）；WSPT（86–91 行）= `(1 + priority_jump) / new_tokens`，短请求优先，优化平均 TTFT。
- `queue.rs`：pending 队列 actor；`--router-queue-threshold` 触发时把请求 hold 住（见优化点 8）。
- `queue_admission/`：准入判断（所有 eligible worker 的 active_prefill_tokens 是否都超过 `threshold × max_num_batched_tokens`）。
- `policy_config.rs`：policy-class YAML 解析（policy_family × cache_bucket → 物理队列，quantum/queue_policy/busy threshold per class）。
- `filter.rs`：资格过滤（allow-list、pinned worker/dp_rank、required taints、busy threshold），发生在打分**之前**（见优化点 16）。
- DRR 仲裁：跨 policy-class 的赤字轮转（见优化点 9）。

---

## 阶段 5：SGLang backend 集成

### 5.1 事件发布桥接

`components/src/dynamo/sglang/publisher.py`，`init_kv_event_publish`（280–333 行）：

```python
if self.server_args.kv_events_config:                  # SGLang 原生参数，JSON
    kv_events = json.loads(self.server_args.kv_events_config)
    base_ep = kv_events.get("endpoint")                 # 如 tcp://*:5557
    dp_ranks = get_local_dp_rank_range(self.server_args)  # 本节点的 DP rank 区间
    for dp_rank in dp_ranks:
        # 用 SGLang 自己的 offset_endpoint_port 保证端口对齐：base_port + dp_rank
        zmq_ep = ZmqEventPublisher.offset_endpoint_port(base_ep, dp_rank)
        publisher = KvEventPublisher(          # dynamo.llm 绑定
            endpoint=self.generate_endpoint,
            worker_id=self.kv_worker_id,
            kv_block_size=self.server_args.page_size,   # 块大小对齐！
            zmq_endpoint=zmq_ep, zmq_topic="",
            enable_local_indexer=self.dynamo_args.enable_local_indexer,
            dp_rank=dp_rank,
        )
```

要点：

- SGLang 侧每个 DP attention rank 的 Scheduler 各自绑定 `base_port + attn_dp_rank` 发布事件（`sglang.srt.disaggregation.kv_events.ZmqEventPublisher`）；dynamo.sglang 按同样的端口公式逐个订阅本地 rank。
- 多机：每节点的 dynamo.sglang 只订阅本地 DP rank，跨节点分发交给 event plane（NATS 时）。
- `enable_local_indexer=True` 时 worker 侧还维护一份 `LocalKvIndexer`（buffer + radix 树），供 router 重启后反查恢复（见优化点 13）。

### 5.2 开关推导

`components/src/dynamo/sglang/args.py`（582–594 行）：`use_kv_events = kv_events_config 存在且 publisher != "null"`。不传 `--kv-events-config` 就不发布（与 vLLM 的 `enable_kv_cache_events` 语义对齐）。

### 5.3 启动示例

```bash
# frontend + KV router
python -m dynamo.frontend --router-mode kv --http-port 8000

# SGLang worker（开事件发布）
python -m dynamo.sglang ... \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}'
```

---

## 阶段 6：HA、恢复与扩展（选读）

- 副本同步：`lib/kv-router/src/sequences/replica_sync.rs` + `services/common/replica_sync.rs:228`（`setup_replica_sync`）——active-sequence 生命周期事件经 runtime event plane 发布/订阅。
- 重启恢复：`lib/llm/src/kv_router/indexer/recovery/` + `lib/llm/src/kv_router/worker_query.rs`（`recover_from_worker`）+ `lib/kv-router/src/recovery/cursor.rs`（每 worker 的 `last_recovered_event_id` 游标）。
- 服务化：`lib/kv-router/src/services/{indexer,selection,slot_tracker}`，对应 `standalone-*.md` 三篇。
- 多 DC：`indexer/cuckoo/` + `components/src/dynamo/kv_dc_relay`。

---

## 文档优化点详解（18 条）

### 1. overlap credit 而非独立 overlap score

老式 cache-aware 路由通常是 `score = w1×overlap − w2×load` 两个独立加权项，权重难调（量纲不同）。Dynamo 把缓存命中统一进"成本 = 还要做多少块工作"这一个量纲：命中块直接从 prefill 块数里**抵扣**（`raw_prefill_blocks − overlap_credit_blocks`）。`overlap_score_credit`（`--router-kv-overlap-score-credit`）是每命中一块抵扣多少块成本：credit=1 表示"命中一块 = 少算一块"，credit>1 表示额外奖励缓存亲和（成本可为负，强吸附），credit=0 则完全不建 indexer（纯负载模式）。

### 2. `overlap_score_credit_decay`：防止缓存富集的赢者通吃

纯 credit 模型有一个正反馈：缓存最多的 worker 总赢 → 它更忙、缓存更富集 → 新扩容 worker 永远接不到热点前缀。衰减公式（selector.rs 里）：

```
excess = (该 worker active_prefill_tokens − 全场最小值) / block_size
decay  = 1 / (1 + decay_coef × excess / request_blocks)
effective_credit = overlap_score_credit × decay
```

即：只对**超出负载地板的部分**衰减 device 层 credit（host/disk/shared 不衰减），负载最低的 worker 保留全额 credit。有理函数衰减比硬阈值平滑，用请求自身块数归一化使系数与请求尺寸无关。默认 0（关闭）。

### 3. 分层缓存 credit

命中不是一个数，而是分层的：device（HBM，权重 = credit×decay）、host_pinned（CPU offload，默认 0.75）、disk（NVMe，默认 0.25）、shared（共享缓存池如 HiCache，`shared_cache_multiplier`，只记 device 前缀之外的部分 `hits_beyond`）。反映不同介质取回 KV 的成本差：HBM 命中免算，host/disk 命中要传输但仍比重算便宜。对接 SGLang HiCache 时 `shared_cache_type=hicache`。

### 4. 温度采样

`temperature=0`（默认）：确定性 argmin。`>0`：对负成本做 softmax 采样，成本相近的 worker 按概率分流。作用是打散"同质请求瞬间涌入时全部砸向同一 worker"的羊群效应（路由决策快于负载记账反馈时尤其明显）。温度越高越接近均匀随机。

### 5. prefill/decode 双成本与 `--load-aware` 预设

logit 的两项对应两种资源瓶颈：`prefill_load_scale × adjusted_prefill_blocks`（算力/TTFT 侧）+ `potential_decode_blocks`（显存/并发 decode 侧）。`prefill_load_scale` 调两者相对权重；PD 分离部署里 prefill router 和 decode router 可以配不同值。`--load-aware` 预设 = `overlap_score_credit=0` + 关事件消费：退化为纯负载均衡（等价一个精细版 least-loaded），用于对照实验或缓存无用的负载。

### 6. AIC prefill 负载衰减

静态记账的缺陷：`active_prefill_tokens` 在收到首 token 前一直全额计入，但实际上 prefill 是渐进完成的。`router_prefill_load_model=aic` 用 AIConfigurator 的性能模型预测每个请求的 prefill 时长，按已流逝时间对其贡献做时间衰减（`scheduling/prefill_load.rs`），以"锚点请求"（最近完成 prefill 的请求）校准预测。让高负载下的 prefill 负载视图更接近真实剩余工作量。

### 7. 输出块追踪 + OSL 提示

`potential_decode_blocks` 默认只按输入估计。`--router-track-output-blocks` 让 router 随响应流增长动态增加该请求的 decode 块占用。客户端可传 `nvext.agent_hints.osl`（预期输出长度）：router 据此对"接近完成的请求"给分数权重（快完成的请求对未来负载贡献小），改善长输出场景的 decode 侧均衡。

### 8. router 队列 + 背压（与 busy threshold 的区别）

`--router-queue-threshold`：当**所有** eligible worker 的 `active_prefill_tokens > threshold × max_num_batched_tokens` 时，请求在 router 队列等待，不派发也不拒绝——等容量释放后用**最新**负载/缓存状态再打分（延迟决策换更优决策）。与 busy threshold 的边界（router-filtering.md）：busy threshold 是把单个过载 worker 踢出候选集（过滤），queue threshold 是全员过载时推迟整个决策（背压）。队列内排序 FCFS（尾部 TTFT 稳定）vs WSPT（平均 TTFT 更优、长请求可能饿）。**SGLang 特有坑**：不设 `--max-prefill-tokens` 时 SGLang 上报的 `max_num_batched_tokens` 回退为整个 KV pool 大小，阈值分母被撑大、背压几乎不触发——文档 troubleshooting 里专门点名。

### 9. policy-class 队列 + DRR（deficit-round-robin.md）

按 `policy_family × cache_bucket`（cached/uncached，按 uncached_tokens 分桶）把请求映射到物理队列，每队列独立 queue_policy 和 `quantum`。跨队列用赤字轮转（DRR）仲裁：
- 计费单位是 `scheduling_cost = max(1, uncached_tokens)`（入队时快照，不再重算）；
- 每轮 ring 扫描给类加一个 quantum 的赤字，头请求付得起就派发（同一轮赤字够可以连发多个）；
- 超大请求（cost >> quantum）用**批量虚拟轮**：算出所有可派发类中最小的 `rounds_needed = ceil((cost−deficit)/quantum)`，一次性给所有类加同样轮数的信用——保持权重比例，仲裁复杂度 O(C) 与请求大小无关；
- 阻塞类保留已挣信用但不再累积（防无界突发）；空类赤字清零；
- 队列满时返回 529 限流。
只有默认单类时 DRR 退化为普通单队列。

### 10. 优先级分层（priority-scheduling.md）

三层各管一段，互不传导：
- **router 队列层**：`strict_priority`（硬分层，高 tier 永远先出队）+ `priority`（在 tier 内调整 FCFS/WSPT 的排序 key）。只在队列非空时生效。
- **引擎调度层**：`priority` 透传给 backend（SGLang 需 `--enable-priority-scheduling`；Dynamo 统一"值越大优先级越高"的极性，内部做各引擎极性转换，vLLM 客户端不要自己取负）。`strict_priority` **不**下传引擎。
- **KV 缓存层**：SGLang `--radix-eviction-policy priority`，内存压力下低优先级块先逐出。
优先级不是抢占/容量预留：没有争用（队列空、引擎不排队、内存不紧张）就看不到效果。

### 11. 事件模式三选一 + 近似路由

精确模式（ZMQ 默认 / NATS）：前缀状态是 worker 真实缓存的镜像。近似模式（`--no-router-kv-events`）：indexer 仍在，但状态来自 router 自己的路由决策预测 + `--router-ttl-secs` TTL 过期；引擎侧真实逐出 router 感知不到，可能高估命中。近似模式下 `--serve-indexer` 只允许单副本（预测状态无法多副本收敛）。backend 是否发布与 router 是否消费是两个独立开关。

### 12. predicted-TTL side indexer

精确事件有一个固有窗口：路由决策 → 引擎实际算完并发出第一个 stored 事件之间，兄弟请求（best-of-N、agent fan-out，共享长前缀、几乎同时到达）查 indexer 是查不到刚路由出去的前缀的，会被打散到不同 worker，缓存收益全失。`--router-predicted-ttl-secs` 在主 indexer 旁边挂一个短 TTL 的预测索引：路由决策立即把该请求的块写入预测索引，查询时合并两个索引的结果。真实事件到达后主索引接管，预测项 TTL 过期自清。等于用近似模式的技巧**只**去补精确模式的时间窗。

### 13. 事件回放/恢复对比（kv-event-replay-comparison.md）

两家都是 ZMQ PUB/SUB（有损）+ 单调序号 + 消费端 gap 检测，差异在兜底：

| | vLLM | Dynamo |
|---|---|---|
| worker 侧状态 | 只有 deque 回放缓冲（默认 1 万批，预序列化） | `LocalKvIndexer` = KvIndexer(RadixTree 权威态) + VecDeque 事件环 |
| 缓冲内 gap | ROUTER socket 线性扫描回放 | 二分查找回放（`Events` 响应） |
| gap 超出缓冲 | 消费端只能自己重建（无内建） | **Tree dump 兜底**：全树导出为合成事件（`TreeDump` 响应） |
| 初始同步 | 无（后加入的消费者从空开始） | `start_event_id=None` 即 tree dump 快照 |
| 消费端超前 | — | `TooNew` 错误响应 |

Dynamo 路径：router 每 worker 维护 `last_recovered_event_id`（`worker_query.rs`），发现 gap 或初次发现 worker 时调 `recover_from_worker(worker, dp_rank, start, end)`。代价是 worker 侧要多维护一棵树（radix 树本身对共享前缀有压缩，比存原始事件省）。

### 14. 多副本一致性（router-operations.md）

两类状态两种策略：
- **前缀状态**：不做 router 间同步。每个副本独立消费同一事件流，天然收敛（事件是广播的）。
- **活跃块状态**：默认各副本只看自己路由的请求（local-only）。`--router-replica-sync` 开启后，副本把 add/free 生命周期事件发布到 event plane，其他副本 `replica_free`/replica-add 跟账。新副本启动时活跃状态从零开始，靠请求短生命周期自然收敛（刻意不做持久化）。
- session affinity 同步（`--router-session-affinity-ttl-secs`）是 best-effort：各副本自持 TTL，需要严格粘性时用 ingress 层。

### 15. 独立索引器/选择器/槽位追踪服务

三个可独立部署的服务面（`services/{indexer,selection,slot_tracker}` + `standalone-*.md`）：把 indexer（内存大头）、worker 选择、活跃槽位追踪从 frontend 进程剥出来。Dynamo 原生路径用 `--serve-indexer`（某些 router/frontend 副本对外提供 `kv_indexer_query`）+ `--use-remote-indexer`（消费副本不建本地 overlap 索引、改走 RPC 查询）——N 个 frontend 副本不必各自维护 N 份全量 radix 树。`dynamo.indexer` 则是给非 Dynamo 部署的独立 HTTP+ZMQ 微服务。

### 16. 过滤 vs 背压的边界（router-filtering.md）

打分**前**的硬资格过滤：allowed worker IDs、pinned worker/dp_rank 校验、DP rank 边界（`[start_rank, start_rank + dp_size)`）、required taints、busy threshold 过载剔除（`--active-decode-blocks-threshold` / `--active-prefill-tokens-threshold[-frac]`，可经 `/busy_threshold` HTTP 端点运行时改）。错误分类保真：没有兼容 worker → no endpoint；有兼容但全过载 → overload；pinned 且过载 → `PinnedWorkerOverloaded`（不改路由）。打分**中**的软信号：preferred_taints（乘成本）、温度、credit 权重。打分**后**的背压：queue threshold（推迟，不剔除）。

### 17. 多 DC 路由（multi-dc-kv-routing.md，实验性）

问题：哪个 DC 有最长可复用前缀？把全量事件流跨 WAN 汇聚不现实。方案分两级：
- **DC 内**（Stage 1）：DC Relay（`python -m dynamo.kv_dc_relay --dc-id ...`）消费本 DC 全部 worker 的精确事件，维护精确 ownership + 每个 full hash 的 DC 级引用计数（首个 owner 插入 CKF 指纹、末个 owner 删除，中间只调 refcount——必须留 full hash 才能安全删除）。复用与普通 indexer 相同的 worker-query 恢复框架。
- **跨 DC**（Stage 2）：只发布 Cuckoo filter 投影——barrier 快照 + 带序号的脏桶绝对镜像（幂等）。全局消费者按 DC lane 转置存储（每桶一个原子 u64/lane，最多 16 个 DC 并发查询，弱读契约）。失败在最窄边界恢复（worker gap → Relay 内恢复；lane 失效 → 重发快照）。CKF 有假阳性，容量压力下允许稳定假阴性。当前端到端是进程内 adapter，gRPC 传输和跨 DC 转发是后续工作。

### 18. 拓扑感知 KV 传输（topology-aware-kv-transfer.md）

PD 分离下 prefill→decode 的 KV 传输走 NVLink/RDMA，跨 zone/rack 传输代价高。worker 经 `ModelRuntimeConfig` 发布拓扑（`topology_domains` → 规范化 taint `dynamo.topology/<domain>=<value>`；`kv_transfer_domain` 指定用哪个域约束传输）。prefill router 选定 prefill worker 后，把它的传输域拓扑转成 decode 请求的 `RoutingConstraints`：`required` 模式 → required taint（不匹配的 decode worker 直接不合格，fail-closed：先建约束再派发 prefill）；`preferred` 模式 → preferred taint（全部合格，匹配者成本乘 `preferred_weight` 折扣）。worker 侧靠 `DYN_TOPOLOGY_*` 环境变量 + 挂载文件读拓扑；vLLM/SGLang/TRT-LLM 三 backend 都支持。与优化点 16 的 taint 机制是同一条 `RoutingConstraints` 路径，没有专用 selector。

---

## 动手验证（可选）

```bash
# 1. 起 frontend（KV 路由）
python -m dynamo.frontend --router-mode kv --http-port 8000

# 2. 起两个 SGLang worker（开 KV 事件）
python -m dynamo.sglang --model-path ... \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557"}' \
  --max-prefill-tokens 8192          # 避免优化点 8 的坑

# 3. 发两条共享长前缀的请求，观察是否路由到同一 worker
# 4. 看指标：curl localhost:8000/metrics | grep dynamo_component_kv 
#    及 dynamo_frontend_router_queue_pending_requests（队列背压验证）
# 5. 调试日志：DYN_LOG=dynamo_llm::kv_router=debug 可看到每个 worker 的
#    logit 分解（selector.rs 的 tracing::debug! 输出完整公式各项）
```

## 建议阅读顺序回顾

1. `docs/components/router/router-guide.md` + `router-concepts.md`（30 min）
2. 主线：`push_router.rs` → `kv_router.rs` → `protocols.rs` → `selector.rs`（重点背成本公式）
3. `indexer/kv_indexer.rs` + `radix_tree.rs`（事件→树）
4. `sequences/multi_worker.rs`（负载记账）
5. `scheduling/{policy,queue,filter}.rs`（队列/准入/过滤）
6. `components/src/dynamo/sglang/publisher.py`（SGLang 桥接）
7. 按需：replica_sync / recovery / services / cuckoo
