"""L02 Part A2 · ⚠️ actor 超额的后果：永久挂死（不是排队）。

初版这一课用 actor 演示「超额会排队」—— 那是错的，脚本会一直不返回。
原因：actor 在【创建时】就占住 num_gpus，并持有到它【整个生命周期】结束；
不是每次方法调用占一下。所以第 9 个 actor 根本不会被创建。
"""
import os
import time
import ray


@ray.remote(num_gpus=1, num_cpus=1)
class Holder:
    def __init__(self, name: str):
        self.name = name

    def work(self, seconds: float, t0: float) -> tuple[str, str, float]:
        return (self.name,
                os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                round(time.time() - t0, 2))


def main() -> None:
    ray.init()
    print("账本 GPU:", ray.cluster_resources().get("GPU"))

    t0 = time.time()
    n = 9
    holders = [Holder.remote(f"a{i}") for i in range(n)]
    time.sleep(2)                       # 给调度器一点时间
    print(f"已请求 {n} 个 actor；此刻 available GPU = "
          f"{ray.available_resources().get('GPU')}")
    print("⚠️ 没有报错，也没有 warning 说你资源不够\n")

    # ⚠️ 不能假设"前 8 个就是被创建的那些"—— 被创建的是哪 8 个没有保证。
    #    所以用 ray.wait 而不是 ray.get：它把【就绪的】和【一直没就绪的】分开返回，
    #    不会因为其中一个永远不就绪而把整个脚本挂住。
    refs = [h.work.remote(0.1, t0) for h in holders]
    ready, pending = ray.wait(refs, num_returns=n, timeout=5)

    done = sorted(refs.index(r) for r in ready)
    stuck = sorted(refs.index(r) for r in pending)
    print(f"5 秒内返回的 actor: {['a%d' % i for i in done]}")
    print(f"一直没返回的 actor: {['a%d' % i for i in stuck]}")

    if stuck:
        print(f"\n  ✅ a{stuck[0]} 根本没有被【创建】—— 它没有崩溃，")
        print("     它在等一张【永远不会空出来】的卡。")
    else:
        print("\n  ⚠️ 9 个都跑起来了 —— 机器上其实有 ≥9 张卡，先 nvidia-smi -L 确认")

    print("\n⚠️ 关键：actor 在创建时就占住 num_gpus，持有到整个生命周期结束。")
    print("   ⇒ actor 的超额 = 永久挂死；任何 ray.get 在它上面都会一直等下去。")
    print("   ⇒ 这就是为什么不能用 ray.get 去探它 —— 必须用 ray.wait + timeout。")

    ray.shutdown()


if __name__ == "__main__":
    main()
