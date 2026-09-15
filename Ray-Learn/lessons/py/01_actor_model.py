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