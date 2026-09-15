# PR 27058：Session Radix Cache 调研报告

**合并时间**：2026-06-23，by @ishandhanani  
**审阅者**：@hzh0425，@stmatengss  
**PR 链接**：https://github.com/sgl-project/sglang/pull/27058  
**关联 RFC**：#27574 (Programmatic KV Cache for Agentic Workloads)  
**整理日期**：2026-07-20

---

## 一、背景：旧方案的致命缺陷

### 1.1 旧机制：`--enable-streaming-session` + lock_ref

PR #27058 之前，SGLang 的 session KV 管理方式是**将 session KV 从 radix cache 中隔离出来，用 `lock_ref` 锁定**，使其不可被 LRU 淘汰。

问题：**在并发负载下，当不可淘汰层（non-evictable tier）填满时，会导致 worker deadlock。**

具体失效路径：
```
多个 session 同时持有 lock_ref
    → non-evictable 内存耗尽
    → 新请求无法分配 KV slot
    → scheduler 无法推进
    → deadlock
```

这是一个系统性的设计缺陷：将 session 语义（"不要淘汰我"）硬编码为内存操作（lock_ref），导致内存管理失去弹性。

### 1.2 新方案的核心思想

PR #27058 提出了根本不同的设计：

> **Session KV 就是普通的可淘汰 radix 条目，session 只是一个"释放分组"的标签。**

- Session 打开 → 无内存操作，只是注册一个 id
- Session 运行中 → KV 照常被 radix cache 管理，照常可被 LRU 淘汰
- Session 关闭 → 精确释放该 session 标记的所有 KV leaf，归还内存

关键转变：**session 语义从"内存锁定"降级为"关闭时批量释放的提示"**，内存池永远不会被 wedge。

---

## 二、架构设计

### 2.1 新增 CLI 标志

```
--enable-session-radix-cache
```

**约束**：必须与 `--radix-eviction-policy priority` 同时使用，否则启动时抛 `ValueError`。

这是一个显式 opt-in 设计——不影响任何现有部署。

### 2.2 核心数据结构

引入 `SessionRadixCacheMixin`（混入类），核心状态：

```python
_session_leaves: defaultdict[str, set[RadixNode]]
# session_id → 该 session 标记过的所有 radix leaf 节点集合

_closed_session_ids: OrderedDict  # tombstone，上限 8192 条
# 已关闭的 session id 集合，防止 "关闭后才完成的请求" 重新泄漏标记
```

每个 `RadixNode` 节点上新增：

```python
node.session_ids: set[str]
# 该节点被哪些 session 标记（支持多 session 共享同一 leaf）
```

为什么是 `set` 而不是 `str`？  
设计讨论中考虑了 scalar `session_id` 字段，但被替换为 `set`——因为**两个 session 可能产生字节完全相同的 KV（内容相同的 token 序列）**，它们会 resolve 到同一个 leaf 节点。如果用 scalar，第一个 `close` 会误清另一个 session 的标记，导致 close 变成 no-op。`set` 解决了这个多 session 共享问题。

### 2.3 Mixin 继承结构

```python
class RadixCache(SessionRadixCacheMixin, KVCacheEventMixin, BasePrefixCache):
    ...
```

MRO 顺序：`SessionRadixCacheMixin` 在 `KVCacheEventMixin` 之前，保证 session 操作的 hook 优先执行。

---

## 三、核心流程分析

### 3.1 请求分类：radix_native_session

Scheduler 新增了一个请求类型判断：

```python
radix_native_session = (
    session_id is not None
    and server_args.enable_session_radix_cache
)
```

**radix-native session 的特殊限制**：不允许使用 `rid/offset/replace/drop_previous_output` 参数——因为这些参数依赖 streaming session 的状态机（维护跨 turn 的 offset），而 radix-native session 每次 turn 必须发完整 context（依赖 radix cache 自动匹配前缀）。

### 3.2 Open Session（几乎是 no-op）

```python
# scheduler.py
def open_session(self, session_id):
    if enable_session_radix_cache:
        tree_cache.register_session(session_id)  # 只清除 tombstone
    else:
        session_controller.open(session_id)      # 旧路径
```

`register_session` 的实际作用只是从 tombstone（`_closed_session_ids`）中清除该 id，允许 id 复用。没有任何内存分配。

### 3.3 Tag Session Leaf（请求完成时打标签）

这是 session radix cache 的核心操作，在两个时机触发：

**请求正常完成（`cache_finished_req`）**：
```python
session_leaf = result.last_device_node  # insert() 返回的最末节点
self._tag_session_leaf(req, radix_key, node=session_leaf)
```

**请求未完成（流式中断，`cache_unfinished_req`）**：
```python
self._tag_session_leaf(req, radix_key, node=new_last_node)
```

`_tag_session_leaf` 内部逻辑：
1. 如果 `node is None`，fallback 到 `match_prefix` 找最长匹配节点
2. 如果 session 已在 tombstone 中（已关闭），**直接 return**，不打标签（防泄漏）
3. 将 `session_id` 加入 `node.session_ids`
4. 将 `node` 加入 `_session_leaves[session_id]`

### 3.4 Delete Leaf Hook（节点被删除时清理）

```python
def _delete_leaf(self, node):
    self._discard_session_leaf(node)  # 先从所有 session 的 leaf set 中移除
    super()._delete_leaf(node)
```

这保证了无论节点因何删除（LRU 淘汰、显式 close、reset），`_session_leaves` 中不会存在悬空引用。

### 3.5 Close Session（精确批量释放）

```python
def release_session(self, session_id) -> int:
    freed = 0
    for node in list(_session_leaves[session_id]):
        # 从该节点向上遍历
        while node is not None:
            node.session_ids.discard(session_id)
            # 满足条件才物理删除：无其他 session 持有 + 无子节点 + 无锁 + 可淘汰叶节点
            if (not node.session_ids
                    and not node.children
                    and not node.lock_ref
                    and node.is_evictable_leaf()):
                parent = node.parent
                physically_delete(node)
                freed += 1
                node = parent
            else:
                break
    _remember_closed_session(session_id)  # 加入 tombstone
    return freed
```

**释放策略的关键设计**：

1. **向上遍历（leaf → root）**：从 leaf 出发往父节点走，形成一条向上的释放链
2. **四个停止条件**：任意一条满足就停止向上释放，保护共享前缀
   - `node.session_ids` 非空（其他 session 也标记了这个节点）
   - `node.children` 非空（还有其他 KV 依赖这个节点）
   - `node.lock_ref`（有正在进行的请求在使用）
   - 不是可淘汰叶节点
3. **O(scan) 复杂度**：没有维护 per-session 的 leaf 索引，close 时扫描 `evictable_leaves`——这是有意设计，换取更简单的并发模型

**Tombstone 机制**（防止迟到请求泄漏）：

```
session close 发生
    → session_id 加入 tombstone
    → 如果此时有请求还在飞行中（in-flight），完成后调用 _tag_session_leaf
    → _tag_session_leaf 检测到 tombstone，直接 return，不打标签
    → 内存不会被泄漏
```

Tombstone 上限 8192 条，使用 `OrderedDict` 实现 LRU 淘汰，防止无限增长。

---

## 四、与 `insert()` 返回值的变化

为了支持 `_tag_session_leaf` 拿到最末节点，`insert()` 的返回值发生了变化：

```python
# before
def insert(self, key) -> int:  # 返回 prefix_len

# after
@dataclass
class InsertResult:
    prefix_len: int
    last_device_node: Any = None  # 新增：最末节点引用

def insert(self, key) -> InsertResult:
```

`_insert_helper` 内部也从返回 `length` 改为返回 `(length, node)` 元组。

这是一个接口变更，影响所有调用 `insert()` 的地方，但范围可控（仅 radix_cache 内部）。

---

## 五、性能测试结果

测试环境：12 个并发 pi-agent × 4 个 worker subagent，GLM-4.7-Flash TP2，2× L40S，pool 大小 143,872 tokens。

| 指标 | 旧方案（OFF） | Session Radix Cache（ON） |
|------|------------|--------------------------|
| Peak pool 占用率 | **100%**（0 空闲 token） | **64%** |
| Peak 驻留缓存 KV | 100% | 53% |
| 强制 LRU 淘汰 token 数 | **284,544** | **0** |
| session close 主动释放次数 | 0 | 82 |

核心结论：**session close 主动释放将 LRU 强制淘汰从 284K tokens 降至 0**——本质是把"被动被踢"变成了"主动归还"，内存压力从系统端转移到了应用端的 session 生命周期管理。

正确性测试（全部通过）：

| 测试场景 | 验证内容 |
|---------|---------|
| Transparency | 带/不带 session tag 的 greedy 输出字节完全一致 |
| Shared-prefix | `close(A)` 只释放 A 的独占链，B 的 KV 和共享前缀保持驻留 |
| Reclamation | 已关闭 session 再发请求，`cached_tokens → 0`（cache miss） |
| Robustness | close 未 open 的 id、double-close、重用同一 id、close 飞行中请求——全部 graceful |
| Oversubscription | 12 个 21K-token session（共 252K > 144K pool）同时 open 不 close——全部服务，无 wedge |

最后一条测试尤为重要：旧方案中 252K > 144K 必然触发 deadlock，新方案通过正常 LRU 淘汰处理了超订。

---

## 六、设计权衡与被否决的方案

### 6.1 Floor Priority（被否决）

早期迭代中考虑给 session KV 设置一个"最低优先级下限"，使其在 LRU 中比普通 KV 更难被淘汰。

**被否决的原因**：session tag 的语义是"关闭时批量释放的分组"，不是"更重要的 KV"。如果用户想要优先级保护，应该通过 RFC #27574 的 Retain/Pin hint 来表达。将优先级和 session tag 耦合会混淆两个独立的语义。

### 6.2 Deferred/Bounded Drain Queue（被否决）

早期考虑在 close 时将释放操作放入一个异步队列，批量处理。

**被否决的原因**：`RadixCache.evict` 本身是同步的，保持一致性更简单。异步队列引入的复杂性（并发安全、队列满处理）不值得。

### 6.3 O(1) Per-session Index vs O(scan)（有意选择 O(scan)）

可以维护一个 `session_id → leaf_nodes` 的索引来实现 O(1) close。

**选择 O(scan) 的原因**：close 是低频操作，scan 的开销可接受；更重要的是，scan-on-close 天然解决了"关闭后才完成的请求"的泄漏问题——关闭后新完成的请求因 tombstone 不会打标签，close 时的 scan 只处理当时已存在的 leaf，两者之间不存在竞态。

---

## 七、与 PR #29436 的关系

这两个 PR 是 RFC #27574 Phase 1 的组合拳：

| PR | 贡献 |
|----|------|
| **#27058**（本 PR）| 实现 session radix cache 的**机制**：tag-on-insert，release-on-close，tombstone 防泄漏 |
| **#29436** | 提供 session 的**身份**：顶层 `session_id` 字段，implicit open，全栈透传 |

执行顺序上，#27058 先合并（6-23），#29436 后合并（6-28）。#29436 合并后，`session_params.id` 的绕弯派生路径被 `session_id` 直接替换，#27058 中 scheduler 里的派生逻辑也随之简化（这正是 #29436 净减 43 行的来源之一）。

---

## 八、总结

PR #27058 的核心贡献是一个设计范式的转变：

> **从「session = 锁定内存」→「session = 可淘汰 KV 上的释放分组标签」**

这个转变带来三个直接收益：
1. **消灭 deadlock**：内存池永远不会被 non-evictable tier 撑满
2. **精确回收**：session close 主动归还内存，不依赖 LRU 被动淘汰
3. **超订安全**：session 总量可以超过内存池，系统通过 LRU 自然降级而非崩溃

代价是：每次请求完成需要打标签（`_tag_session_leaf`），每次 close 需要 O(scan) 遍历——在 session 规模合理的场景下，这个开销是完全可接受的。
