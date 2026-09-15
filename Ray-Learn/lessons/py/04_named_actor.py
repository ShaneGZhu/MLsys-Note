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
