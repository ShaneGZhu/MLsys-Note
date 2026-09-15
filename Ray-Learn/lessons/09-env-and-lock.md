# L09 · 编排层的两个"看不见的基础设施"：环境变量注入 + 分布式锁

> 🖥️ **任意机器即可**（单卡或笔记本都行）。

> 📄 **为什么有这一课**：7 课初版**只在 README 提了一句环境变量**，
> 而 slime 为它写了整个 `utils.py`（75 行）。Lux 的 `config.py` 要照抄这些——**少一条就出事**。

## ① 一句话目标

掌握 slime `utils.py` 的两件事：**环境变量注入**（含三条必须照抄的）与 **24 行分布式锁**。

## ② 先预测

1. 在 actor 启动**之后**改 `os.environ`，那个 actor 能看见吗？
2. `runtime_env={"env_vars": ...}` 和直接改 `os.environ` 有什么区别？
3. `Lock.acquire()` 为什么设计成**非阻塞**的？

## ③ 完整代码

存成 `lessons/py/09_env_and_lock.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
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
```

```bash
uv run python lessons/py/09_env_and_lock.py
```

## ④ 你应该观察到什么

```
① 环境变量注入
  runtime_env 方式: {'LUX_STAGE': 's0', 'NCCL_CUMEM_ENABLE': '0', 'RAY_USE_UVLOOP': '0', ...}
  os.environ 方式:  {...同上...}
  ⚠️ 两种都行，但 os.environ 的坑是【顺序】：设晚了 actor 就看不到。

② 分布式锁
  第一次 acquire: True
  第二次 acquire: False   ← 应该 False
  release 之后:   True   ← 又 True
  重复 release -> RayTaskError(AssertionError)（assert 生效）

③ 为什么 Lux 需要这个锁
   权重更新时『独占引擎』只需要一种场景的互斥 —— 24 行够了，不必上 Ray 的锁原语。
```

### ⚠️ 如果不符合预期

| 现象 | 原因 |
| --- | --- |
| `os.environ` 方式下 actor 看不到变量 | 你在**起 actor 之后**才设的。必须在之前 |
| `runtime_env` 报错 `RuntimeEnvAgent` 超时 | 首次用 runtime_env 时 Ray 要拉起 agent；重试一次通常就好 |
| 重复 release 没抛异常 | `assert` 在 `python -O` 下会被去掉；检查你没开优化模式 |

## ⑤ ⭐ 三条必须照抄的环境变量（`Lux/docs/design/config.py` 要写进去）

slime `utils.py:28-30` 的**逐字注释**：

| 变量 | 值 | 原文注释 / 后果 |
| --- | --- | --- |
| `RAY_USE_UVLOOP` | `0` | *"Ray's uvloop integration has caused **intermittent async actor issues**"* |
| `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` | `1` | 阻止 Ray 改写 `CUDA_VISIBLE_DEVICES`，让 actor 自己管设备（配合 [L08](08-gpu-identity.md)） |
| `NCCL_CUMEM_ENABLE` | `0` | ⭐ **训练侧必须与 SGLang 一致**，否则 NCCL 报错（slime 的原注释：*"because sglang will always set NCCL_CUMEM_ENABLE to 0, we need also set it to 0 to prevent nccl error"*） |

⚠️ 第三条对 Lux **直接相关**——因为我们选的就是 SGLang（Lux 的 ADR-0003，`Lux/docs/decisions/0003-inference-stack-dynamo-sglang.md`）。
**这是一条"两边必须一致"的跨系统不变量**，属于 R 规则那一类。

## ⑥ 为什么锁是 24 行而不是用 Ray 的锁原语

slime 的取舍（`Lux/docs/research/dimensions/orchestration/README.md` §"关键机制六"）：

> **没有用 Ray 的锁原语，24 行自己写**——因为只需要"权重更新时独占引擎"这一种场景。
> 这是"**按需最小实现**"的范例。

而它**故意做成非阻塞**（`acquire()` 返回 `False` 而不是等）：

```
拿不到锁  →  返回 False  →  调用方自己决定"是重试、还是先做别的"
拿不到锁  →  阻塞等待    →  调用方被卡住，而它可能本来有事可做
```

⇒ 这与 [L05](05-sync-vs-async.md) 是同一个道理：**在编排层，"阻塞"是要避免的默认行为**。

## ⑦ Lux 用在哪

| 机制 | Lux 里的用途 |
| --- | --- |
| 环境变量注入 | `config.py` 的启动期；`LuxDriver` 起 actor 之前统一注入 |
| `NCCL_CUMEM_ENABLE=0` | 训练侧与 SGLang 共存的**硬要求**（S0 必验之一） |
| 分布式锁 | 权重更新时独占引擎；`LuxDriver` 的版本屏障 |

## ⑧ 一句话总结

> **编排层有两类"看不见的基础设施"：环境变量（决定进程怎么起）和锁（决定谁能动）。**
> 两者都**不报错**——配错环境变量的症状是"偶发、难复现"；锁用错的症状是"偶发竞态"。
> ⇒ 所以它们要么照抄被验证过的实现，要么别自己发明。
