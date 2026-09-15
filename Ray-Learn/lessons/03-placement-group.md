# L03 · Placement Group：gang scheduling 与 bundle

> 🖥️ **单机 8×H100 即可做完本课。** 跨节点部分要双机起（→ [L07](07-multinode.md)）。

## ① 一句话目标

理解 placement group（PG）解决的两个问题：**要么全给要么不给**（gang scheduling）和**位置确定**（哪个 actor 落在哪个 bundle）。

## ② 先预测

机器有 **8 张 H100**。我们申请一个需要 **12 个 bundle**（各 `{"GPU": 1}`）的 PG。

1. `pg.ready()` 会怎样？
   - A. 立刻完成，Ray 先给 8 个，等剩下的
   - B. 永远不完成（或超时），因为 PG 是 all-or-nothing
   - C. 报错
2. 改成 4 个 bundle 呢？

## ③ 完整代码

存成 `lessons/py/03_placement_group.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
"""L03 · Placement Group：gang scheduling 与 bundle 绑定。"""
import os
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_gpus=1, num_cpus=1)
class TrainActor:
    """假装是一个训练 rank：它只关心【自己落在哪个 bundle / 哪张卡】。"""

    def __init__(self, rank: int):
        self.rank = rank

    def where(self) -> dict:
        ctx = ray.get_runtime_context()
        # ⚠️ get_placement_group_bundle_index() 不是每个 Ray 版本都有 ——
        #    用 getattr 探测，缺了就打 "?"，不要因为这个 AttributeError 让整课跑不起来。
        get_bundle = getattr(ctx, "get_placement_group_bundle_index", None)
        return {
            "rank": self.rank,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            "node_ip": ray.util.get_node_ip_address(),
            "pg_id": str(ctx.get_placement_group_id())[:8],
            "bundle": get_bundle() if callable(get_bundle) else "?",
        }


def make_actor(pg, rank: int):
    return TrainActor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=rank,        # ⭐ 这一行决定「位置」
        )
    ).remote(rank=rank)


def main() -> None:
    ray.init()
    print("账本 GPU 总数:", ray.cluster_resources().get("GPU"))

    # ── 实验 1：要 12 个 GPU bundle，但只有 8 张 ⇒ gang 语义 ──────
    print("\n[实验 1] 申请 12 个 bundle（各 1 GPU），机器只有 8 张")
    pg_big = placement_group([{"GPU": 1, "CPU": 1}] * 12,
                             strategy="STRICT_PACK", name="too_big")
    try:
        ray.get(pg_big.ready(), timeout=8)
        print("  ⚠️ 竟然 ready 了 —— 与预期不符，见 md 的排查表")
    except ray.exceptions.GetTimeoutError:
        print("  ✅ 8 秒内没有 ready：**PG 是 all-or-nothing，不给你一部分**")
    ray.util.remove_placement_group(pg_big)

    # ── 实验 2：要 4 个，能起来 ──────────────────────────────────
    print("\n[实验 2] 申请 4 个 bundle（机器有 8 张）")
    pg = placement_group([{"GPU": 1, "CPU": 1}] * 4,
                         strategy="STRICT_PACK", name="fit")
    ray.get(pg.ready())
    print(f"  ✅ ready。pg={str(pg.id)[:8]}  bundle 数={len(pg.bundle_specs)}")

    # ── 实验 3 / 4：同样的绑定跑两轮，位置必须一致 ───────────────
    rows = []
    for rnd in (3, 4):
        print(f"\n[实验 {rnd}] 4 个 TrainActor 绑到 bundle 0/1/2/3"
              f"{'（再跑一轮，用于比对）' if rnd == 4 else ''}")
        info = ray.get([make_actor(pg, i).where.remote() for i in range(4)])
        info.sort(key=lambda d: d["rank"])
        for d in info:
            print(f"  rank={d['rank']}  bundle={d['bundle']}  gpu={d['gpu']}  "
                  f"node={d['node_ip']}  pg={d['pg_id']}")
        rows.append([(d["rank"], d["gpu"]) for d in info])

    print()
    if rows[0] == rows[1]:
        print("✅ 两轮的 (rank → gpu) 完全一致 —— 绑定是稳定的")
    else:
        print(f"❌ 两轮不一致：\n   第一轮 {rows[0]}\n   第二轮 {rows[1]}")
        print("   ⇒ 必须做 bundle 重排序（见 md §⑤）")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/py/03_placement_group.py
```

## ④ 你应该观察到什么

```
账本 GPU 总数: 8.0

[实验 1] 申请 12 个 bundle（各 1 GPU），机器只有 8 张
  ✅ 8 秒内没有 ready：**PG 是 all-or-nothing，不给你一部分**

[实验 2] 申请 4 个 bundle（机器有 8 张）
  ✅ ready。pg=xxxxxxxx  bundle 数=4

[实验 3] 4 个 TrainActor 绑到 bundle 0/1/2/3
  rank=0  bundle=0  gpu=0  node=10.0.0.11  pg=xxxxxxxx
  rank=1  bundle=1  gpu=1  node=10.0.0.11  pg=xxxxxxxx
  rank=2  bundle=2  gpu=2  node=10.0.0.11  pg=xxxxxxxx
  rank=3  bundle=3  gpu=3  node=10.0.0.11  pg=xxxxxxxx

[实验 4] 4 个 TrainActor 绑到 bundle 0/1/2/3（再跑一轮，用于比对）
  ...（同上，必须逐行相同）

✅ 两轮的 (rank → gpu) 完全一致 —— 绑定是稳定的

⚠️ 若 bundle= 显示为 "?"，说明你的 Ray 版本没有
   get_placement_group_bundle_index()——不影响结论，看 gpu= 那一列即可。
```

### ⚠️ 单机能观察什么、不能观察什么

| | 单机（8 卡） | 双机起 |
| --- | --- | --- |
| gang 语义（all-or-nothing） | ✅ 实验 1 | |
| bundle 绑定生效 | ✅ 实验 3/4（`bundle=` 与 `gpu=` 逐行对应） | |
| **bundle → 物理节点的映射** | ⚠️ `node_ip` 必然全一样（只有一台） | ✅ L07 |
| **STRICT_PACK / STRICT_SPREAD 的差别** | ⚠️ 单节点下无法区分 | ✅ L07 |

⇒ **单机上"分布"这一维学不到**，但"机制"能学全。不要因为实验 3/4 的 `node_ip` 一样就以为绑定没生效——
**绑定生效的证据是 `bundle=` 与 `gpu=` 稳定对应**。

> ⚠️ 若 `ctx.get_placement_group_bundle_index()` 报 `AttributeError`（不同 Ray 版本 API 有差异），
> 去掉这一列即可，**关键看 `gpu=` 是否稳定**。

## ⑤ ⭐ 为什么 Lux 关心这个

### ① gang scheduling 是刚需

Lux 训练侧是 **16 个 `TrainActor`（各 `num_gpus=1`）组成一个 torch 进程组**。
如果 Ray 只给 15 个、剩下 1 个排队，那 15 个会**全部卡在 `init_process_group` 上等**——
**看起来像挂死，实际是在等一个永远不会来的 rank**（L06 会复现这个死锁）。

⇒ **必须用 PG 一次要齐 16 个 bundle。**

### ② bundle 重排序：这是**正确性**问题，不是性能问题

Lux 明确要照搬的机制之一（`Lux/docs/research/dimensions/orchestration/README.md` §八「必须保留的」）。

**问题**：Ray 把 PG 的 bundle 分给节点时，**分配顺序不保证与你的 `bundle_index` 一致**。
实验 4 就是在测这个：**两轮必须逐行相同**。

如果 rank i 每次落在不同的物理 GPU 上，会出两类问题：
- NCCL 拓扑变化（NVLink / PCIe / 跨节点，带宽差一个数量级）
- 任何**按 rank 做假设**的逻辑（如"rank 0 负责发权重"）行为不稳定

**做法**（slime `placement_group.py` 253 行，`actor_group.py:116-129`）：
先探测 bundle → `(node_ip, gpu_id)` 的真实映射，**重排 bundle 顺序**，再按重排后的索引绑定 actor。

```python
actor = TrainActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=reordered_bundle_indices[rank],   # ⭐ 重排后
    )
).remote(world_size, rank, master_addr, master_port)
```

### ③ 一个你会读到的"怪数字"：`num_gpus=0.4`

slime 用 `num_gpus=0.4` 让多个 actor 共享一张卡（**共置**场景，`placement_group.py:155`）。
实际 GPU 归属由 bundle 决定，`0.4` 只是 Ray 的资源记账——**小于 1 才能让多个 actor 落在同一张卡上**。

**Lux 不需要**（2 训 2 推物理分离，`TrainActor` 老实写 `num_gpus=1`），
但你要知道这个技巧，否则读 slime 代码时会以为是笔误。

## ⑥ 下一步

分布那一维（跨节点、STRICT_SPREAD、rank↔GPU 真实稳定性）在 [L07](07-multinode.md)。
