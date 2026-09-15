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
