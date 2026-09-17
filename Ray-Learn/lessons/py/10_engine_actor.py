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
