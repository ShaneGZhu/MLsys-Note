"""L09 · 编排层的两个"看不见的基础设施"：环境变量注入 + 分布式锁。

对应 slime:
  slime/ray/utils.py   NOSET_VISIBLE_DEVICES_ENV_VARS_LIST / RAY_DEFAULT_ENV_VARS
                       / add_default_ray_env_vars()  (共 75 行)
  slime/ray/utils.py:57 @ray.remote class Lock(RayActor)   (24 行)
"""
import os
import time
import ray

# ── ① 环境变量：照抄 slime 的 utils.py ───────────────────────────
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
]

RAY_DEFAULT_ENV_VARS = {
    # 原文注释：Ray's uvloop integration has caused intermittent async actor issues.
    "RAY_USE_UVLOOP": "0",
    # 阻止 Ray 改写可见设备，让 actor 自己管（配合 ray.get_gpu_ids() 使用，见 L08）
    **{k: "1" for k in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
    # Lux 还要加这条：训练侧必须与 SGLang 一致，否则 NCCL 报错
    "NCCL_CUMEM_ENABLE": "0",
}


def add_default_ray_env_vars(env_vars: dict | None = None) -> dict:
    return RAY_DEFAULT_ENV_VARS | (env_vars or {})


# ── ② 24 行分布式锁（slime 的实现，几乎逐字）────────────────────
@ray.remote
class Lock:
    """非阻塞锁：拿不到就返回 False，调用方自己轮询。

    ⚠️ 故意不是阻塞的 —— 编排层需要"拿不到就去做别的"，而不是卡在这里。
    """

    def __init__(self):
        self._locked = False

    def acquire(self) -> bool:
        if not self._locked:
            self._locked = True
            return True
        return False                       # 调用方轮询重试

    def release(self) -> None:
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False


@ray.remote
class EnvProbe:
    def show(self, keys: list[str]) -> dict:
        return {k: os.environ.get(k, "<unset>") for k in keys}


def main() -> None:
    ray.init()

    # ── ① 环境变量注入：两种方式 ─────────────────────────────────
    print("① 环境变量注入")
    env = add_default_ray_env_vars({"LUX_STAGE": "s0"})
    keys = sorted(env)

    # 方式一：runtime_env（推荐 —— 显式、可读、不污染 driver 进程）
    a1 = EnvProbe.options(runtime_env={"env_vars": env}).remote()
    print("  runtime_env 方式:", ray.get(a1.show.remote(keys)))

    # 方式二：先改 driver 的 os.environ，再起 actor（slime 用的就是这种）
    # ⚠️ 必须在【起 actor 之前】设好 —— 之后设对已存在的 actor 无效
    os.environ.update(env)
    a2 = EnvProbe.remote()
    print("  os.environ 方式:", ray.get(a2.show.remote(keys)))
    print("  ⚠️ 两种都行，但 os.environ 的坑是【顺序】：设晚了 actor 就看不到。\n")

    # ── ② 分布式锁：验证它真的互斥 ───────────────────────────────
    print("② 分布式锁")
    lock = Lock.remote()
    print("  第一次 acquire:", ray.get(lock.acquire.remote()))
    print("  第二次 acquire:", ray.get(lock.acquire.remote()), "  ← 应该 False")
    ray.get(lock.release.remote())
    print("  release 之后:  ", ray.get(lock.acquire.remote()), "  ← 又 True")

    # ⚠️ release 一个没持有的锁会 assert 失败
    ray.get(lock.release.remote())
    try:
        ray.get(lock.release.remote())
        print("  ⚠️ 重复 release 竟然没报错")
    except Exception as e:                      # noqa: BLE001
        print(f"  重复 release -> {type(e).__name__}（assert 生效）")

    print("\n③ 为什么 Lux 需要这个锁")
    print("   权重更新时『独占引擎』只需要一种场景的互斥 —— 24 行够了，不必上 Ray 的锁原语。")

    ray.shutdown()


if __name__ == "__main__":
    main()
