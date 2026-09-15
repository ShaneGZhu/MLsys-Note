# L05 · 同步 vs 异步 actor：饿死实验

> ⭐⭐ **这是全课最重要的一课。** 它解释 Lux 的一个**架构定版**：
> 为什么 `rollout` 必须是**独立的 CPU-only actor**，不能塞进 trainer。

> 🖥️ **不需要 GPU。** 笔记本上也能完整做完 —— 但它解释的是 Lux 最重要的一个架构决策，别因为不用卡就跳过。

## ① 一句话目标

亲眼看到：**一个 actor 里的阻塞调用会饿死它自己的其他所有调用**；而 `max_concurrency` **救不了**。

## ② 先预测

四个 actor，每个都有一个"阻塞 3 秒"的方法和一个 `ping()` 方法。
我们在**不等待**阻塞方法的情况下，立刻测 `ping()` 的延迟：

| 场景 | 阻塞怎么写 | `ping()` 延迟预测 |
| --- | --- | --- |
| **A** 同步 actor | `def blocking(): time.sleep(3)` | ? |
| **B** 异步 actor，**让出** | `async def blocking(): await asyncio.sleep(3)` | ? |
| **C** 异步 actor，**真阻塞** | `async def blocking(): time.sleep(3)` | ? |
| **D** 同 C，但加 `max_concurrency=10` | 同上 | ? |

先写下你的 4 个数字，再跑。

## ③ 完整代码

存成 `lessons/py/05_sync_vs_async.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
"""L05 · 饿死实验：为什么阻塞一个 actor 会拖死它的全部调用。

⚠️ 不需要 GPU —— 笔记本上也能完整跑完。
"""
import asyncio
import time
import ray

BLOCK_SECONDS = 3.0
SETTLE = 0.3          # 让阻塞方法先跑起来，再去 ping


# ── A：同步 actor。Ray 直接在线程里跑，默认 max_concurrency=1 ──
@ray.remote(num_cpus=0)
class SyncActor:
    def blocking(self, s: float) -> str:
        time.sleep(s)                     # 占住这个 actor 的线程
        return "blocking done"

    def ping(self) -> str:
        return "pong"


# ── B：异步 actor，用 await 让出事件循环 ──────────────────────
@ray.remote(num_cpus=0)
class AsyncYielding:
    async def blocking(self, s: float) -> str:
        await asyncio.sleep(s)            # ⭐ 让出控制权，事件循环继续跑别的
        return "blocking done"

    async def ping(self) -> str:
        return "pong"


# ── C：异步 actor，但阻塞是【真的】阻塞（模拟 NCCL 卡在 C 里）──
@ray.remote(num_cpus=0)
class AsyncBlocking:
    async def blocking(self, s: float) -> str:
        time.sleep(s)                     # ⚠️ 不让出 —— 事件循环被按住
        return "blocking done"

    async def ping(self) -> str:
        return "pong"


def measure(actor, label: str) -> None:
    """提交一个长时间阻塞的调用，然后立刻测 ping 的延迟。"""
    ref_block = actor.blocking.remote(BLOCK_SECONDS)   # 不等待，只提交
    time.sleep(SETTLE)                                  # 等它真的跑起来
    t0 = time.time()
    ray.get(actor.ping.remote())
    latency = time.time() - t0
    ray.get(ref_block)                                  # 收尾，确保下次测量干净
    flag = "✅ 立刻返回" if latency < 0.5 else "❌ 被饿死了"
    print(f"  {label:<34} ping 延迟 = {latency:5.2f}s   {flag}")


def main() -> None:
    ray.init()
    print(f"阻塞方法睡 {BLOCK_SECONDS}s；先提交它，再立刻 ping\n")

    measure(SyncActor.remote(), "A 同步 actor")
    measure(AsyncYielding.remote(), "B 异步 + await（让出）")
    measure(AsyncBlocking.remote(), "C 异步 + time.sleep（真阻塞）")
    measure(AsyncBlocking.options(max_concurrency=10).remote(),
            "D 同 C，但 max_concurrency=10")

    print("\n👉 结论：C 和 D 一样地慢 —— max_concurrency 只是让【协程】交错，")
    print("   而一个卡在 C 调用里的协程永远不会 yield。")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/py/05_sync_vs_async.py
```

## ④ 你应该观察到什么

```
阻塞方法睡 3.0s；先提交它，再立刻 ping

  A 同步 actor                       ping 延迟 =  2.70s   ❌ 被饿死了
  B 异步 + await（让出）              ping 延迟 =  0.00s   ✅ 立刻返回
  C 异步 + time.sleep（真阻塞）       ping 延迟 =  2.70s   ❌ 被饿死了
  D 同 C，但 max_concurrency=10      ping 延迟 =  2.70s   ❌ 被饿死了

👉 结论：C 和 D 一样地慢 —— max_concurrency 只是让【协程】交错，
   而一个卡在 C 调用里的协程永远不会 yield。
```

**B 是唯一的例外**，因为 `await` 显式让出了控制权。A/C/D 都是 2.7s（3.0 − 0.3）。

### ⚠️ 如果不符合预期

| 现象 | 说明 |
| --- | --- |
| A 的 ping 很快 | 你把 `SyncActor` 的方法写成了 `async def`。同步 actor 的默认并发度是 **1**，这是饿死的机制 |
| B 的 ping 也慢 | `asyncio.sleep` 写成了 `time.sleep` |
| D 的 ping 变快了 | ⚠️ **那你用的 Ray 版本行为与本文不同**——那是个重要发现，记下来。本文引用的上游结论（NeMo-RL 的 docstring）说的是"救不了" |
| 全都是 0.00s | `SETTLE` 期间阻塞方法已经跑完了？把 `BLOCK_SECONDS` 调大 |

## ⑤ ⭐⭐ 为什么 Lux 关心这个

这条不是"最佳实践"，是 Lux 一个**已定版架构决策的直接依据**。

`Lux/docs/design/control-plane-internals.md` 引了 NeMo-RL `refit_watchdog.py:121-124` 的**逐字原文**：

> Ray runs a SYNC actor method directly in the event loop … a refit that blocks in NCCL
> **starves every other call to the same actor**. `max_concurrency` cannot help;
> it interleaves coroutines, and **a coroutine blocked in C never yields**.

你刚才在 **C 和 D** 两行里亲眼看到了这句话。

### 它定版了 Lux 的什么

**`rollout` 必须是独立的 CPU-only actor，不能放进 trainer。**

设想一下如果放进去会怎样（这是很多框架的默认做法）：

```
TrainActor（一个 actor）
├── 训练方法 add()          ← 里面是 NCCL all-gather，阻塞在 C 里
└── rollout 方法 generate()  ← 想同时发 HTTP 请求
```

`add()` 一跑，`generate()` 就被饿死；权重同步（`refit`）一跑，整个 actor 的其他调用全停。
**训练和生成互相卡死，而且不会有任何报错**——只是变慢，慢到你以为在等网络。

⇒ Lux 的答案：**把它们拆成两个 actor**。这也正是 `02-ray-roles.dot` 里
`RolloutActor`（推理节点 CPU）与 `TrainActor`（训练节点 GPU）分开的原因。

### 顺带：这也是 `RefitAbortWatchdog` 为什么是"线程"而不是"actor"

`Lux/docs/design/control-plane-internals.md` 定版理由之三：

> `WeightSender` / `RefitAbortWatchdog` **不是独立 actor**：看门狗**必须**是进程内线程
> （同一 docstring：「The abort must therefore come from **a thread already inside the process**」）

因为当一个 actor 卡在 NCCL 里时，**从外面调它的方法也会被饿死**——外面的 actor 根本叫不动它。
唯一能在"它卡住时"动手的，是**同一个进程里的另一个线程**。

## ⑥ 一句话总结

> **Actor 是"单线程事件循环"的抽象。** `await` 让出 → 能并发；阻塞在 C 里 → 全停。
> `max_concurrency` 调的是**协程**的并发度，不是**线程**的。
>
> ⇒ **设计编排时，凡是可能长时间阻塞在 C 里的事（NCCL、大文件 IO、同步 HTTP），
> 都该有自己的 actor。**
