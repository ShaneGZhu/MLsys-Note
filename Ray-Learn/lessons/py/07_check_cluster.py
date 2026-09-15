"""L07 · 多机集群断言：节点数 + GPU 账本（+ 可选：rank↔GPU 稳定性）。

用法:
    uv run python 07_check_cluster.py                 # 只查账本
    uv run python 07_check_cluster.py --dump-ranks    # 另起 4 个 actor 打印 rank→(node,gpu)
"""
import os
import sys
import ray

EXPECTED_NODES = 4
EXPECTED_GPU = 16          # ⭐ 2 个训练节点 × 8，不是 32


@ray.remote(num_gpus=1, num_cpus=1)
class Probe:
    def __init__(self, rank: int):
        self.rank = rank

    def where(self) -> str:
        return (f"rank={self.rank} node={ray.util.get_node_ip_address()} "
                f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")


def dump_ranks() -> None:
    """⭐ 用于验证 bundle 重排序是否必要：跑两次，输出必须【完全一致】。"""
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    n = 4
    pg = placement_group([{"GPU": 1, "CPU": 1}] * n, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    actors = [
        Probe.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i)).remote(i)
        for i in range(n)
    ]
    for line in ray.get([a.where.remote() for a in actors]):
        print(line)


def main() -> int:
    ray.init(address="auto")

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"存活节点数: {len(nodes)}")

    total_gpu = 0.0
    for n in sorted(nodes, key=lambda x: x["NodeManagerAddress"]):
        res = n["Resources"]
        gpu = res.get("GPU", 0)
        total_gpu += gpu
        print(f"  {n['NodeManagerAddress']:<16} CPU={res.get('CPU', 0):<8} GPU={gpu}")

    print(f"\nGPU 总账: {ray.cluster_resources().get('GPU', 0)}")
    print(f"CPU 总账: {ray.cluster_resources().get('CPU', 0)}")

    ok = True
    if len(nodes) != EXPECTED_NODES:
        print(f"❌ 节点数 {len(nodes)} != {EXPECTED_NODES}")
        ok = False
    if total_gpu != EXPECTED_GPU:
        hint = ("（很可能是推理节点带着卡 join 了 —— 检查 --num-gpus=0）"
                if total_gpu > EXPECTED_GPU else "")
        print(f"❌ GPU 账本 {total_gpu} != {EXPECTED_GPU}{hint}")
        ok = False
    print("\n✅ 集群账本正确" if ok else "\n❌ 账本不对，先修它再往下走")

    if "--dump-ranks" in sys.argv:
        print("\n--- rank ↔ (node, gpu) ---")
        dump_ranks()

    ray.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
