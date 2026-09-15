# L01 · Actor 模型：task 与 actor 的区别

> 🖥️ **任意机器即可**（单卡或笔记本都行）。本课与 GPU 无关。

## ① 一句话目标

搞清楚 Ray 的两件东西——**无状态 task** 与**有状态 actor**——以及为什么"有状态"是编排层的全部理由。

## ② 先预测

写下你的答案，再跑代码：

1. `f.remote()` 返回的是什么？（是结果吗？）
2. 如果连续调两次同一个 actor 的方法，第二次能看见第一次留下的东西吗？
3. 普通 task 呢？
4. 下面代码里 `slow.remote()` 和 `fast.remote()` 谁先执行完？

```
A. 顺序执行（先 slow 后 fast）
B. 并发执行（两个一起跑）
C. 不确定
```

## ③ 完整代码

存成 `lessons/01_actor_model.py`：

```python
"""L01 · task 与 actor。"""
import time
import ray


# ── 无状态：task ────────────────────────────────────────────────
@ray.remote
def square(x: int) -> int:
    return x * x


@ray.remote
def slow(seconds: float) -> str:
    time.sleep(seconds)
    return f"slept {seconds}s"


# ── 有状态：actor ───────────────────────────────────────────────
@ray.remote
class Counter:
    def __init__(self):
        self.n = 0

    def inc(self) -> int:
        self.n += 1
        return self.n

    def get(self) -> int:
        return self.n


def main() -> None:
    ray.init()

    # 1. .remote() 返回的是【句柄】，不是结果
    ref = square.remote(7)
    print("1. .remote() 返回:", type(ref).__name__, "->", ref)
    print("   ray.get(ref)   =", ray.get(ref))

    # 2. task 是无状态的：每次调用都是新的一次
    print("2. task 连续两次:", ray.get(square.remote(2)), ray.get(square.remote(2)))
    print("   —— 两次都是 4，task 不记得任何东西")

    # 3. actor 是有状态的
    c = Counter.remote()
    print("3. actor 连续三次 inc:", ray.get(c.inc.remote()),
          ray.get(c.inc.remote()), ray.get(c.inc.remote()))
    print("   —— 1, 2, 3：状态活在 actor 里")

    # 4. ⭐ 关键：actor 的方法调用是【串行】的，task 是【并发】的
    t0 = time.time()
    refs = [slow.remote(1.0) for _ in range(4)]
    ray.get(refs)
    print(f"4a. 4 个 task 各睡 1s，总耗时 {time.time() - t0:.2f}s（并发 -> 约 1s）")

    @ray.remote
    class Sleeper:
        def sleep(self, seconds: float):
            time.sleep(seconds)
            return seconds

    s = Sleeper.remote()
    t0 = time.time()
    ray.get([s.sleep.remote(1.0) for _ in range(4)])
    print(f"4b. 同一个 actor 的 4 次调用各睡 1s，总耗时 {time.time() - t0:.2f}s"
          f"（串行 -> 约 4s）")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/01_actor_model.py
```

## ④ 你应该观察到什么

```
1. .remote() 返回: ObjectRef -> ObjectRef(...)
   ray.get(ref)   = 49
2. task 连续两次: 4 4
3. actor 连续三次 inc: 1 2 3
4a. 4 个 task 各睡 1s，总耗时 ~1.0s（并发 -> 约 1s）
4b. 同一个 actor 的 4 次调用各睡 1s，总耗时 ~4.0s（串行 -> 约 4s）
```

**4a 与 4b 的对比是这一课的全部价值**：

| | 并发度 | 状态 |
| --- | --- | --- |
| task | 多个同时跑 | 无 |
| actor | **一次只跑一个方法** | 有 |

⚠️ **如果 4b 只有 1s** —— 说明这个 actor 被声明成了 async（见 [L05](05-sync-vs-async.md)），或者你机器上 Ray 版本行为不同。先确认 `Sleeper` 的方法是 `def` 不是 `async def`。

### 三个最常见的误解

1. **以为 `.remote()` 会阻塞** —— 它立刻返回 `ObjectRef`，真正的执行在别的进程里。
2. **以为 `ray.get` 是"启动任务"** —— 任务在 `.remote()` 时就提交了，`ray.get` 只是等。
3. **以为 actor 能并发处理自己的方法** —— 默认不能。**这正是 L05 的主题**，也是 Lux 里 `rollout` 必须独立成 actor 的原因。

## ⑤ 为什么 Lux 关心这个

Lux 的 6 种角色**全部**是 actor（`Lux/docs/design/control-plane-internals.md`）：

| 角色 | 为什么必须是 actor 而不是 task |
| --- | --- |
| `LuxDriver` | 要持有"启动到哪一步了"的状态 |
| `TrainActor ×16` | 要持有 FSDP2 的模型分片（**状态极重**） |
| `RolloutActor ×R` | 要持有 HTTP 连接池 |
| `TransferQueueManager` | 要持有队列元数据 |

⇒ **"有状态"就是编排层存在的理由**。如果所有东西都无状态，就不需要编排了。

而 4b 那条"actor 串行"看起来像个限制——**它是 L05 那个致命坑的根源**。
