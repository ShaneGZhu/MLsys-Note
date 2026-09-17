# L10 · 引擎即 actor：actor 持卡，引擎跑在它的**子进程**里

> 🖥️ **单机 8×H100 即可。** 需要 `--num-gpus` 真的可用（0 卡的机器上 actor 起不来）。

> 📄 **为什么有这一课**：2026-09 设计变更后，**推理节点的 16 张卡也进 Ray 的账本**，
> 于是必须有东西"持有"它们。本课讲那个东西怎么写——以及**为什么引擎必须是子进程而不是
> 直接跑在 actor 里**（答案在 [L05](05-sync-vs-async.md)）。

## ① 一句话目标

写出 `EngineActor`：**持 2 张卡（tp=2）+ 拉起引擎子进程 + 自己始终可响应**，
并验证"引擎崩了 actor 还能重启它"。

## ② 先预测

1. actor 声明 `num_gpus=2`，Ray 把 `CUDA_VISIBLE_DEVICES` 设成什么？
2. actor 用 `subprocess.Popen` 起的子进程，能看到那 2 张卡吗？
3. 引擎是长时间阻塞的。如果它**跑在 actor 进程里**，`ping()` 还能立刻返回吗？
4. 从外面 `kill -9` 掉引擎进程，actor 会死吗？

## ③ 完整代码

存成 `lessons/py/10_engine_actor.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
"""L10 · 引擎即 actor：actor 持卡，引擎跑在它的【子进程】里。

设计变更（2026-09）：推理节点的 16 张卡也进 Ray 的账本 ⇒ **引擎由 Ray 分配卡**，
于是必须有东西"持有"那 2 张卡。做法：

    @ray.remote(num_gpus=2)          ← actor 持有 2 张卡（Ray 设好 CVD）
    class EngineActor:
        def __init__(...):
            subprocess.Popen([...])  ← 引擎是【子进程】，继承 CVD

⭐ 为什么必须是子进程而不是"在 actor 里直接跑引擎"：见 L05 的饿死实验 ——
   一个阻塞在 C 里的调用会饿死同一 actor 的【所有】其他方法。
   引擎是长时间阻塞的，放进 actor 进程 = 这个 actor 再也响应不了任何管理调用。
"""
import os
import subprocess
import sys
import time
import ray

# 一个"假引擎"：它只做两件事 —— 打印自己看到的卡，然后一直活着
FAKE_ENGINE = r"""
import os, sys, time
print(f"[engine pid={os.getpid()}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
time.sleep(600)
"""


@ray.remote(num_gpus=2, num_cpus=2)
class EngineActor:
    """一个 SGLang 引擎的持有者（tp=2 ⇒ 要 2 张卡）。"""

    def __init__(self, idx: int):
        self.idx = idx
        self.cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        self.proc: subprocess.Popen | None = None
        self.log = f"/tmp/fake_engine_{idx}.log"
        self._spawn()

    # ── 引擎生命周期 ────────────────────────────────────────────
    def _spawn(self) -> None:
        # ⭐ 关键：子进程【继承】父进程的 CUDA_VISIBLE_DEVICES
        #    ⇒ Ray 分配的那 2 张卡自动传给引擎，不需要我们自己算
        self.proc = subprocess.Popen(
            [sys.executable, "-c", FAKE_ENGINE],
            env=os.environ.copy(),
            stdout=open(self.log, "w"),
            stderr=subprocess.STDOUT,
        )

    def engine_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def engine_pid(self) -> int:
        return self.proc.pid if self.proc else -1

    def engine_cvd(self) -> str:
        """从子进程自己的日志里读它看到的卡 —— 证明继承生效。"""
        try:
            with open(self.log) as f:
                return f.readline().strip()
        except OSError:
            return "<no log yet>"

    def restart_engine(self) -> str:
        """⭐ 这是"actor 持有引擎"的核心收益：引擎崩了，actor 还活着，能重启它。"""
        if self.engine_alive():
            return "still alive, not restarted"
        self._spawn()
        return f"restarted -> pid {self.proc.pid}"

    # ── 管理方法：它【必须】始终可响应 ───────────────────────────
    def ping(self) -> str:
        return f"pong from engine {self.idx} (cvd={self.cvd})"


def main() -> None:
    ray.init()

    print("=== ① 引擎 actor 拿到 2 张卡，子进程继承 ===")
    eng = EngineActor.remote(0)
    print("  actor 自己的 CVD :", ray.get(eng.ping.remote()))
    time.sleep(1.0)
    print("  子进程看到的     :", ray.get(eng.engine_cvd.remote()))
    print("  engine pid       :", ray.get(eng.engine_pid.remote()))
    print("  ⇒ Ray 给 actor 分 2 张卡，子进程【自动继承】，不用自己算\n")

    print("=== ② ⭐ 引擎在忙时，actor 仍然立刻响应 ===")
    t0 = time.time()
    print("  ping 延迟:", f"{time.time() - t0:.4f}s", "→", ray.get(eng.ping.remote()))
    print("  ⇒ 因为引擎是【另一个进程】。")
    print("     如果引擎跑在 actor 进程里（阻塞在 C 里），这个 ping 会被饿死（L05 的 C/D 两行）\n")

    print("=== ③ ⭐ 引擎崩了，actor 还活着，能重启 ===")
    pid_before = ray.get(eng.engine_pid.remote())
    os.system(f"kill -9 {pid_before}")          # 从外面杀掉引擎进程
    time.sleep(0.5)
    print(f"  kill -9 {pid_before} 之后：")
    print("  engine_alive :", ray.get(eng.engine_alive.remote()))
    print("  actor 还活着吗:", ray.get(eng.ping.remote()), " ← ✅ 是")
    print("  重启引擎     :", ray.get(eng.restart_engine.remote()))
    time.sleep(0.5)
    print("  重启后 CVD   :", ray.get(eng.engine_cvd.remote()))

    print("\n👉 这就是 Lux 需要 EngineActor 的原因：")
    print("   · 它【持有】那 2 张卡 → Ray 的账本因此知道卡被占了（不会重复分配）")
    print("   · 引擎是它的子进程     → 引擎崩了不带走 actor，actor 可以重启引擎并上报")
    print("   · actor 自己始终轻量   → 管理调用（健康/暂停/恢复）不会被引擎阻塞饿死")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/py/10_engine_actor.py
```

## ④ 你应该观察到什么

```
=== ① 引擎 actor 拿到 2 张卡，子进程继承 ===
  actor 自己的 CVD : pong from engine 0 (cvd=0,1)
  子进程看到的     : [engine pid=12345] CUDA_VISIBLE_DEVICES=0,1
  ⇒ Ray 给 actor 分 2 张卡，子进程【自动继承】，不用自己算

=== ② ⭐ 引擎在忙时，actor 仍然立刻响应 ===
  ping 延迟: 0.0003s → pong from engine 0 (cvd=0,1)

=== ③ ⭐ 引擎崩了，actor 还活着，能重启 ===
  kill -9 12345 之后：
  engine_alive : False
  actor 还活着吗: pong from engine 0 (cvd=0,1)  ← ✅ 是
  重启引擎     : restarted -> pid 12399
```

⭐ **第 ① 步的 `cvd=0,1`**：Ray 因为 `num_gpus=2` 设了**两个**卡号，而**子进程直接继承**——
**你不需要自己算 `CUDA_VISIBLE_DEVICES`**。这是"让 Ray 分配卡"最实际的收益。

## ⑤ ⚠️ 反模式：把引擎放进 actor 的进程里

如果你这样写：

```python
@ray.remote(num_gpus=2)
class BadEngineActor:
    def __init__(self):
        self.proc = None

    def start_engine(self):
        # ❌ 在【本 actor 的进程里】同步阻塞地跑引擎
        run_engine_blocking()          # 永不返回

    def ping(self):                     # ❌ 永远等不到
        return "pong"
```

`start_engine()` 一调，这个 actor 就**再也不会响应任何调用**——包括"引擎还活着吗"这种
最需要它的管理调用。

这正是 [L05](05-sync-vs-async.md) 的 C/D 两行：**一个卡在 C 调用里的协程永远不会 yield**，
`max_concurrency` 也救不了。

| | 引擎在 actor 进程里 | ⭐ 引擎是子进程 |
| --- | --- | --- |
| actor 能否响应管理调用 | ❌ 不能（被饿死） | ✅ 能 |
| 引擎崩了 | 可能带走 actor | ✅ actor 检测到并重启 |
| 能否分别暂停/恢复引擎 | ❌ | ✅（发信号给子进程） |
| 卡所有权 | actor 持有 ✓ | actor 持有 ✓（子进程继承 CVD） |

## ⑥ 为什么 Lux 这么设计

新设计下这**三件事**同时成立，缺一不可：

| 要求 | 靠什么满足 |
| --- | --- |
| Ray 的账本必须知道那 16 张推理卡**被占了** | actor 用 `num_gpus=2` **持有**它们 |
| 引擎崩了要能被发现并重启 | 引擎是**子进程** ⇒ actor 活着 ⇒ `poll()` 能检测、能重启 |
| 暂停/恢复/权重更新等管理调用必须**随时可用** | actor 自己**轻量**，重活全在子进程 |

⚠️ 第三条尤其重要：Lux 的权重更新链路要在引擎上做
`pause_generation` / `clear_kv_blocks` / `update_weights_from_disk` / `continue_generation`。
**如果引擎把 actor 占死了，这条链路一步都做不了。**

⇒ 所以：**actor 是"持有者 + 控制口"，引擎是"被持有的重进程"。** 两者必须分开。

## ⑦ 一句话总结

> **`num_gpus=2` 让 actor 在 Ray 的账本里"占住"两张卡；`subprocess.Popen` 让真正干活的引擎
> 跑到另一个进程里。** 前者解决记账，后者解决"actor 不能被饿死"。
>
> ⚠️ 这两件事必须**同时**做到——只做前者会得到一个永远不响应的 actor；
> 只做后者则 Ray 不知道卡被占了。

