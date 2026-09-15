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
