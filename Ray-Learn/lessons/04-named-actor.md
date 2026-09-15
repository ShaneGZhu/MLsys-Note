# L04 · 具名 actor 与生命周期：死掉以后长什么样

> 🖥️ **任意机器即可**（单卡或笔记本都行）。本课与 GPU 无关。

## ① 一句话目标

学会用**名字**找 actor（而不是靠传句柄），并搞清一个 actor **死掉以后**各种调用分别报什么错——这是"故障可见性"的全部原料。

## ② 先预测

1. 起一个具名 actor，`ray.kill()` 它，然后再调它的方法 —— 报什么？
   - A. 返回 `None`
   - B. 抛 `RayActorError`
   - C. 永远挂住
2. 被 kill 之后，再用 `ray.get_actor("名字")` 还能找到它吗？
3. 驱动进程退出后，它起的普通 actor 还在吗？

## ③ 完整代码

存成 `lessons/04_named_actor.py`：

```python
"""L04 · 具名 actor 与生命周期。"""
import time
import ray


@ray.remote
class Service:
    def __init__(self, tag: str):
        self.tag = tag

    def ping(self) -> str:
        return f"pong from {self.tag}"

    def die(self) -> None:
        """让这个 actor 自己崩掉（模拟训练 rank 挂掉）。"""
        raise RuntimeError(f"{self.tag} crashed on purpose")


def main() -> None:
    ray.init()

    # ── 1. 用名字创建（两个属性：name + namespace）────────────────
    svc = Service.options(name="train_rank0", namespace="lux").remote("r0")
    print("1. 已创建具名 actor: name=train_rank0 namespace=lux")
    print("   ping ->", ray.get(svc.ping.remote()))

    # ── 2. 别的代码不用拿到句柄，也能按名字找到它 ──────────────────
    handle = ray.get_actor("train_rank0", namespace="lux")
    print("2. get_actor 拿到句柄 ->", ray.get(handle.ping.remote()))

    # ⚠️ namespace 不匹配会找不到
    try:
        ray.get_actor("train_rank0", namespace="wrong_ns")
    except ValueError as e:
        print(f"   换一个 namespace 就找不到了: {type(e).__name__}: {e}")

    # ── 3. actor 自己崩了以后，调用方看到什么 ─────────────────────
    print("\n3. 让 actor 自己抛异常（模拟 rank 挂掉）")
    try:
        ray.get(handle.die.remote())
    except ray.exceptions.RayActorError as e:
        print(f"   ✅ 抛的是 RayActorError（不是原异常！）")
        print(f"      {str(e)[:200]}")

    # ── 4. 崩了之后，名字还能用吗？ ───────────────────────────────
    print("\n4. 崩了之后再 get_actor / 再调用")
    found = None
    for attempt in range(5):
        try:
            found = ray.get_actor("train_rank0", namespace="lux")
            print(f"   第 {attempt + 1} 次 get_actor: 还能拿到句柄")
            break
        except ValueError:
            print(f"   第 {attempt + 1} 次 get_actor: ValueError（名字已注销）")
            break
        finally:
            time.sleep(0.2)

    if found is not None:
        try:
            ray.get(found.ping.remote())
            print("   ⚠️ 还能 ping 通 —— 与预期不符，见下方排查")
        except ray.exceptions.RayActorError as e:
            print(f"   ✅ 调用抛 RayActorError: {str(e)[:120]}")

    # ── 5. 显式杀死（正常路径）────────────────────────────────────
    print("\n5. 显式 ray.kill 一个【正常】的 named actor")
    svc2 = Service.options(name="svc2", namespace="lux").remote("s2")
    ray.get(svc2.ping.remote())
    ray.kill(svc2)
    time.sleep(0.5)
    try:
        ray.get_actor("svc2", namespace="lux")
        print("   get_actor 仍能拿到（GCS 注销有延迟）")
    except ValueError as e:
        print(f"   ✅ get_actor: ValueError: {str(e)[:120]}")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/04_named_actor.py
```

## ④ 你应该观察到什么

```
1. 已创建具名 actor: name=train_rank0 namespace=lux
   ping -> pong from r0
2. get_actor 拿到句柄 -> pong from r0
   换一个 namespace 就找不到了: ValueError: ...
3. 让 actor 自己抛异常（模拟 rank 挂掉）
   ✅ 抛的是 RayActorError（不是原异常！）
      ...RuntimeError: r0 crashed on purpose...
4. 崩了之后再 get_actor / 再调用
   第 1 次 get_actor: ValueError（名字已注销）
5. 显式 ray.kill 一个【正常】的 named actor
   ✅ get_actor: ValueError: ...
```

### ⚠️ 本课最可能"不符合预期"的地方

| 现象 | 说明 |
| --- | --- |
| 第 4 步 `get_actor` 还能拿到句柄 | **GCS 注销有延迟**，这是正常的最终一致行为。第 5 步加了 `sleep(0.5)` 就是为它 |
| `die()` 抛的不是 `RayActorError` 而是 `RuntimeError` | Ray 会把 actor 内部的异常**包一层**再抛给调用方；若你看到原始 `RuntimeError`，说明 Ray 版本行为不同——**以你实际看到的为准**，但要知道两种都可能 |
| 第 4 步之后 actor 没死，还能 ping 通 | `die()` 只是让**这一次调用**抛异常，actor 默认**不会**因此死掉。真正让 actor 死的是未捕获异常**导致 actor 退出**——不同 Ray 版本对"方法抛异常是否杀死 actor"的处理需要你自己确认（这正是本课要你观察的点） |

> ⭐ **第 4 步的"以你实际看到的为准"是刻意的**：故障可见性的写法**完全取决于**这个行为，
> 而它随 Ray 版本变。**先测出来，再写代码**——不要照抄别人的 `try/except`。

## ⑤ ⭐ 为什么 Lux 关心这个

### ① 具名 actor 是"分布式注册表"

Lux 的 `LuxDriver` 和 `TransferQueueManager` 都是**具名** actor（`Lux/docs/design/control-plane-internals.md` 角色表）。
具名的意义：**任何 actor 不用拿到句柄就能找到它们**——否则你就得把句柄当参数层层传递，
而 Ray 的句柄在跨进程传递时是可行的、但会让启动顺序变得极脆。

⭐ **TransferQueue 用的就是这个机制**（本轮实测）：

```python
# transfer_queue/interface.py:91-100
ray.get_actor("TransferQueueController", namespace="transfer_queue")
```

⇒ **`tq.init()` 必须在 `ray.init()` 之后**。这就是 Lux 启动顺序里"先起 Ray，再 `tq.init()`"的由来。

### ② "故障可见性"是 Ray 在 Lux 里的一项**明确职责**

`Lux/docs/design/control-plane-internals.md` 逐字：

> 六个仓库里 slime / NeMo-RL / miles 全是**多 actor + torch PG**；**Ray 负责故障可见性**。

这句话的落地就是本课第 3–4 步：**16 个 `TrainActor` 里有一个挂了，你怎么知道？**
- 靠 `ray.get` 抛 `RayActorError`
- 而不是靠它自己上报（它已经死了，上报不了）

⇒ Lux 的 `LuxDriver` 必须**持有所有 TrainActor 的句柄并监视它们**，
而不是假设"没消息就是好消息"。

### ③ `lifetime` 的坑（延伸，两终端实验）

默认 actor 的 lifetime 是**绑定驱动进程**的：驱动退出，actor 就被回收。

```python
# 终端 A
ray start --head
python -c "
import ray; ray.init(address='auto')
@ray.remote
class S:
    def ping(self): return 'pong'
S.options(name='svc', namespace='lux', lifetime='detached').remote()
print('created, exiting driver')
"   # 驱动退出

# 终端 B —— 驱动已退出，actor 还活着
python -c "
import ray; ray.init(address='auto')
h = ray.get_actor('svc', namespace='lux')
print(ray.get(h.ping.remote()))
"
```

⇒ `lifetime='detached'` 让 actor **活过驱动进程**。Lux 是否需要它，取决于"driver 崩了要不要重建整个运行"——
这是一个设计决策，不是默认该开的开关。

## ⑥ 一句话总结

> **具名 actor 给你"按名字找"的能力；而它死掉以后报什么错，决定了你的故障处理能不能写对。**
> 后者必须**在你要用的那个 Ray 版本上实测**——本课第 4 步就是让你亲眼看到它。
