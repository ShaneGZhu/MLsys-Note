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

存成 `lessons/py/04_named_actor.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
"""L04 · 具名 actor 与生命周期。

⚠️ 本脚本【只观察、不假设】：actor 崩了之后调用方到底看到什么异常、
   actor 自己会不会跟着死 —— 这些行为随 Ray 版本变。
   脚本把实际观察到的打印出来，md 里写了两种可能的含义。
"""
import time
import ray


@ray.remote
class Service:
    def __init__(self, tag: str):
        self.tag = tag

    def ping(self) -> str:
        return f"pong from {self.tag}"

    def die(self) -> None:
        """在这个方法里抛异常（模拟某次调用失败，不一定是 actor 崩了）。"""
        raise RuntimeError(f"{self.tag} raised on purpose")


def main() -> None:
    ray.init()

    # ── 1. 用名字创建（name + namespace 两个属性）─────────────────
    svc = Service.options(name="train_rank0", namespace="lux").remote("r0")
    print("1. 已创建具名 actor: name=train_rank0 namespace=lux")
    print("   ping ->", ray.get(svc.ping.remote()))

    # ── 2. 别的代码不用拿句柄，也能按名字找到它 ───────────────────
    handle = ray.get_actor("train_rank0", namespace="lux")
    print("2. get_actor 拿到句柄 ->", ray.get(handle.ping.remote()))

    try:
        ray.get_actor("train_rank0", namespace="wrong_ns")
        print("   ⚠️ 换 namespace 竟然还能找到")
    except ValueError as e:
        print(f"   换 namespace 找不到: {type(e).__name__}: {str(e)[:80]}")

    # ── 3. 方法里抛异常：调用方看到什么？actor 还活着吗？ ─────────
    print("\n3. 调 die()（方法里抛 RuntimeError）")
    try:
        ray.get(handle.die.remote())
        print("   ⚠️ 没有抛异常？")
    except Exception as e:                       # noqa: BLE001 —— 故意宽catch，要打印真实类型
        print(f"   调用方看到: {type(e).__name__}")
        print(f"   {str(e)[:120]}")

    try:
        print("   之后再 ping ->", ray.get(handle.ping.remote()))
        print("   ⇒ actor 【没有】因为方法抛异常而死")
    except Exception as e:                       # noqa: BLE001
        print(f"   之后再 ping 失败: {type(e).__name__} ⇒ actor 死了")

    # ── 4. 显式 ray.kill：这才是真的杀死 ─────────────────────────
    print("\n4. ray.kill 一个正常运行中的具名 actor")
    svc2 = Service.options(name="svc2", namespace="lux").remote("s2")
    ray.get(svc2.ping.remote())
    ray.kill(svc2)

    # ⚠️ GCS 注销有延迟 —— 轮询而不是睡一觉就断言
    gone_at = None
    for attempt in range(20):
        try:
            ray.get_actor("svc2", namespace="lux")
            time.sleep(0.1)
        except ValueError:
            gone_at = attempt
            break
    if gone_at is None:
        print("   1 秒后 get_actor 仍能拿到句柄（GCS 注销延迟，属正常最终一致）")
    else:
        print(f"   ✅ 第 {gone_at + 1} 次轮询时 get_actor 抛 ValueError（名字已注销）")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/py/04_named_actor.py
```

## ④ 你应该观察到什么

```
1. 已创建具名 actor: name=train_rank0 namespace=lux
   ping -> pong from r0
2. get_actor 拿到句柄 -> pong from r0
   换 namespace 找不到: ValueError: ...

3. 调 die()（方法里抛 RuntimeError）
   调用方看到: RayTaskError(RuntimeError)      ← ⚠️ 类型随版本变，见下
   ...
   之后再 ping -> pong from r0
   ⇒ actor 【没有】因为方法抛异常而死

4. ray.kill 一个正常运行中的具名 actor
   ✅ 第 2 次轮询时 get_actor 抛 ValueError（名字已注销）
```

### ⚠️ 本课最可能"不符合预期"的地方

| 现象 | 说明 |
| --- | --- |
| 第 3 步 `die()` 抛出的类型不是 `RayTaskError` | ⚠️ **类型随 Ray 版本变**：可能是 `RayTaskError(RuntimeError)`、`RayActorError`，也可能是原始 `RuntimeError`。**脚本会把真实类型打印出来**——记下你看到的那一个 |
| 第 3 步之后 actor 还能 ping 通 | 说明**方法抛异常没有杀死 actor**（常见默认行为）。但这**不是保证**——正好是本课要你确认的点 |
| 第 4 步轮询 20 次仍能拿到句柄 | **GCS 注销有延迟**，属正常的最终一致。把 `range(20)` 调大或 `sleep` 加长再看 |
| 第 4 步第 1 次轮询就 ValueError | 注销很快，也是正常的——**两种都快/慢都可能**，不要据此下结论 |

> ⭐ **本课刻意不给你"标准答案"**：故障可见性的写法**完全取决于**上面这两个行为，
> 而它们**随 Ray 版本变**。脚本只负责把观察到的打印出来。
> **先在你的版本上测出来，再写 `try/except`**——不要照抄别人的。

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
