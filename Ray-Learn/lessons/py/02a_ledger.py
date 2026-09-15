"""L02 Part A1 · 资源账本：task 会排队（跑完就释放）。

⚠️ 对比 02a_actor_trap.py：把 task 换成 actor，第 9 个就【永远不会跑】。
"""
import os
import time
import ray


# ⭐ 这是 task（函数），不是 actor（类）
@ray.remote(num_gpus=1, num_cpus=1)
def hold(name: str, seconds: float, t0: float) -> tuple[str, str, float]:
    # ⚠️ t0 必须【显式传进来】：remote 函数跑在别的进程里，模块级全局变量
    #    靠 cloudpickle 按值捕获 —— 能work但很容易写错（初版就是这么挂的）。
    #    规则：remote 函数里要用什么，就传什么。
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    time.sleep(seconds)
    return name, gpu, round(time.time() - t0, 2)


def main() -> None:
    ray.init()                       # 单机：自动起本地集群，自动探测到 8 张 GPU

    print("集群资源:", {k: v for k, v in ray.cluster_resources().items()
                        if k in ("GPU", "CPU")})
    print("可用 GPU:", ray.available_resources().get("GPU"))
    print()

    t0 = time.time()
    refs = [hold.remote(f"t{i}", 2.0, t0) for i in range(9)]
    time.sleep(1)
    print("已提交 9 个 task（各要 1 张卡），账上只有 8 张")
    print(f"此刻 available GPU = {ray.available_resources().get('GPU')}")
    print("⚠️ 没有报错。第 9 个被【挂起】了。\n")

    for name, gpu, t in sorted(ray.get(refs), key=lambda r: r[2]):
        mark = "   ← 等的那个" if t > 3 else ""
        print(f"  {name}  CUDA_VISIBLE_DEVICES={gpu:<3} 完成于 {t}s{mark}")

    print("\n👉 前 8 个约 2s，第 9 个约 4s —— 它等别人【跑完并释放】了卡")
    print("👉 全程零异常。**task 的超额不是错误，是排队。**")

    ray.shutdown()


if __name__ == "__main__":
    main()
