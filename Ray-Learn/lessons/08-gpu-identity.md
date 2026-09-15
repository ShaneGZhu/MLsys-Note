# L08 · 我在哪张卡上：`ray.get_gpu_ids()` 与 bundle 重排序

> 🖥️ **单机 8×H100 即可。** 本课补的是 [L03](03-placement-group.md) 缺的那一半。

> 📄 **为什么有这一课**：7 课初版**漏了 `ray.get_gpu_ids()`**（slime 编排层用了 3 次），
> 而它正是「rank ↔ GPU 稳定」的**实现手段**——L03 只说了"必须做重排序"，没说怎么做。

## ① 一句话目标

搞清 **Ray 视角的卡号（`ray.get_gpu_ids()`）** 与 **进程视角的卡号（`CUDA_VISIBLE_DEVICES`）** 的关系，
并用它们实现 **bundle 探测 → 重排序**（Lux 的正确性依赖这个）。

## ② 先预测

1. actor 里 `ray.get_gpu_ids()` 和 `os.environ["CUDA_VISIBLE_DEVICES"]` 一致吗？
2. 如果设了 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 呢？
3. 从**外面**（driver）能不能知道 bundle 3 落在哪张物理卡上？

## ③ 完整代码

存成 `lessons/py/08_gpu_identity.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
"""L08 · 我在哪张卡上：ray.get_gpu_ids() vs CUDA_VISIBLE_DEVICES，以及 bundle 重排序。

slime 编排层用了 3 次 ray.get_gpu_ids()，全部与此有关：
  placement_group.py:18  InfoActor.get_ip_and_gpu_id -> (node_ip, ray.get_gpu_ids()[0])
  train_actor.py:22      LOCAL_RANK = resolve_visible_device_id(ray.get_gpu_ids()[0])
"""
import os
import ray

N = 4


@ray.remote(num_gpus=1, num_cpus=1)
class Probe:
    def __init__(self, rank: int):
        self.rank = rank

    def who(self) -> dict:
        return {
            "rank": self.rank,
            "cvd": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "ray_gpu_ids": ray.get_gpu_ids(),          # ⭐ Ray 视角的卡号
            "node_ip": ray.util.get_node_ip_address(),
        }


@ray.remote(num_gpus=1, num_cpus=1)
class InfoActor:
    """slime 的做法：用一次性 actor 探测【这个 bundle 落在哪张卡上】。

    3 行就是全部意义 —— 因为 bundle → 物理 GPU 的映射只有【在那个 bundle 里跑起来】
    才知道，从外面猜不到。
    """

    def get_ip_and_gpu_id(self) -> tuple[str, int]:
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def make_pg():
    from ray.util.placement_group import placement_group
    pg = placement_group([{"GPU": 1, "CPU": 1}] * N, strategy="STRICT_PACK")
    ray.get(pg.ready())
    return pg


def probe_bundles(pg) -> list[tuple[int, str, int]]:
    """⭐ 探测：bundle_index -> (node_ip, gpu_id)。这是重排序的输入。"""
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    out = []
    for i in range(N):
        a = InfoActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i)).remote()
        ip, gpu = ray.get(a.get_ip_and_gpu_id.remote())
        out.append((i, ip, gpu))
    return out


def reorder(bundles: list[tuple[int, str, int]]) -> list[int]:
    """按 (node_ip, gpu_id) 排序，返回新的 bundle 顺序。

    ⇒ 之后只要用 reordered[rank] 当 bundle_index，rank i 就【永远】落在同一张物理卡上。
    """
    return [b[0] for b in sorted(bundles, key=lambda b: (b[1], b[2]))]


def main() -> None:
    ray.init()

    # ── Part A：默认情况下两个机制是一致的 ───────────────────────
    print("[Part A] ray.get_gpu_ids() vs CUDA_VISIBLE_DEVICES")
    pg = make_pg()
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    probes = [Probe.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=i)).remote(i)
        for i in range(N)]
    for d in sorted(ray.get([p.who.remote() for p in probes]), key=lambda d: d["rank"]):
        print(f"  rank={d['rank']}  CVD={d['cvd']:<5} ray_gpu_ids={d['ray_gpu_ids']}  "
              f"node={d['node_ip']}")
    print("  ⇒ 默认 Ray 会把 CVD 设成它分给你的那张卡，所以两者【一致】。")
    print("  ⇒ 但如果你设了 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1，CVD 就是 <unset>，")
    print("     那时【只有 ray.get_gpu_ids() 知道你在哪张卡上】。\n")

    # ── Part B：探测 + 重排序（Lux 的正确性依赖这个）──────────────
    print("[Part B] 探测 bundle → (node_ip, gpu_id)，然后重排序")
    bundles = probe_bundles(pg)
    for i, ip, gpu in bundles:
        print(f"  bundle {i} -> node={ip} gpu={gpu}")
    order = reorder(bundles)
    print(f"\n  原始顺序 {[b[0] for b in bundles]}")
    print(f"  重排序后 {order}   ← 用它当 bundle_index，rank↔物理卡 就固定了")

    # ── Part C：验证重排序确实让 rank↔gpu 稳定 ───────────────────
    rows = []
    for rnd in (1, 2):
        actors = [Probe.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=order[r])).remote(r)
            for r in range(N)]
        got = sorted(ray.get([a.who.remote() for a in actors]), key=lambda d: d["rank"])
        rows.append([(d["rank"], d["ray_gpu_ids"][0]) for d in got])
    print(f"\n  第 1 轮 {rows[0]}")
    print(f"  第 2 轮 {rows[1]}")
    print("  ✅ 完全一致 —— 重排序生效" if rows[0] == rows[1]
          else "  ❌ 不一致，重排序逻辑要检查")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/py/08_gpu_identity.py
```

## ④ 你应该观察到什么

```
[Part A] ray.get_gpu_ids() vs CUDA_VISIBLE_DEVICES
  rank=0  CVD=0     ray_gpu_ids=[0]  node=10.0.0.11
  rank=1  CVD=1     ray_gpu_ids=[1]  node=10.0.0.11
  rank=2  CVD=2     ray_gpu_ids=[2]  node=10.0.0.11
  rank=3  CVD=3     ray_gpu_ids=[3]  node=10.0.0.11
  ⇒ 默认 Ray 会把 CVD 设成它分给你的那张卡，所以两者【一致】。

[Part B] 探测 bundle → (node_ip, gpu_id)，然后重排序
  bundle 0 -> node=10.0.0.11 gpu=0
  bundle 1 -> node=10.0.0.11 gpu=1
  ...

  原始顺序 [0, 1, 2, 3]
  重排序后 [0, 1, 2, 3]   ← 用它当 bundle_index，rank↔物理卡 就固定了

  第 1 轮 [(0, 0), (1, 1), (2, 2), (3, 3)]
  第 2 轮 [(0, 0), (1, 1), (2, 2), (3, 3)]
  ✅ 完全一致 —— 重排序生效
```

> ⚠️ **单机上"原始顺序"和"重排序后"往往相同**——只有一台机器、卡也是 0–3，没什么可排的。
> **多机时它们才会不同**（bundle 落到哪台机器由 Ray 决定）。所以本课的**验证要在 [L07](07-multinode.md) 的双机环境上再做一遍**。

### ⭐ 关键：两个机制什么时候会分叉

| | 谁设的 | 什么时候看不见 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | **Ray 替你设**（默认） | 设了 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 之后，它**不被设置** |
| `ray.get_gpu_ids()` | Ray 的运行时**总是**知道 | **永远可用** |

⇒ **当你想自己管设备（比如要控制内存池、要 CUDA MPS）时，你会关掉 Ray 的 CVD 设置，
那时 `ray.get_gpu_ids()` 是唯一的信息来源。** slime 的 `train_actor.py:22` 就是在做这件事：

```python
def get_local_gpu_id():
    return accelerator.resolve_visible_device_id(ray.get_gpu_ids()[0])
```

### ⭐ `InfoActor` 模式（3 行，但不可替代）

```python
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]
```

**为什么不能从 driver 直接算**：bundle → 物理 GPU 的映射，只有**在那个 bundle 里真的跑起来**才知道。
driver 拿到的 `pg.bundle_specs` 只说"这个 bundle 要 1 张卡"，不说"是哪张"。

⇒ 所以 slime 起一次性 actor 去问，问完就扔。Lux 的 `orchestrate/` 也要这么做。

## ⑤ 为什么 Lux 关心这个

`Lux/docs/research/dimensions/orchestration/README.md` §八「**必须保留的**」第二条：

> **bundle 重排序**（保证 rank ↔ GPU 稳定，**关系到正确性**）

| 不重排序会怎样 | 为什么是正确性问题 |
| --- | --- |
| NCCL 拓扑漂移 | NVLink / PCIe / 跨节点，带宽差一个数量级——同样的代码，性能不可复现 |
| 按 rank 做假设的逻辑失效 | 比如"rank 0 负责发送权重"，而 rank 0 每次落在不同卡上 |
| 复现性丧失 | 两次运行同样的配置，行为不同，而你**没有任何报错** |

⚠️ **这三条都不会报错**——这正是它必须靠机制（重排序）而不是靠"注意一下"来解决的原因。

## ⑥ 一句话总结

> **`ray.get_gpu_ids()` 是"我在哪张卡上"的唯一权威答案**；
> `CUDA_VISIBLE_DEVICES` 只是 Ray 默认替你设的方便，
> **一旦你关掉它（`NOSET_*`），就只有前者可用**。而 bundle→卡的映射**必须靠探测**，猜不出来。
