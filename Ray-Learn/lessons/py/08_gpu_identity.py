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
