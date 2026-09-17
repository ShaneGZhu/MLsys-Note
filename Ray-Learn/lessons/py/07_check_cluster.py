"""L07 · 把 4 个节点全部交给 Ray（32 张卡）并验证落点。

⚠️ 2026-09 设计变更：**推理节点的 16 张卡也进 Ray 的账本**（旧设计是 --num-gpus=0 把它们藏起来）。
   风险因此【反过来】：以前怕"Ray 看不见推理卡"，现在怕"Ray 把 TrainActor 放到推理卡上"。
   ⇒ 必须用【自定义资源】给节点打角色，并把角色写进 bundle 规格。

用法:
    uv run python 07_check_cluster.py                 # 查账本 + 节点角色
    uv run python 07_check_cluster.py --check-placement   # 再验两个 PG 的落点
"""
import os
import sys
import ray

EXPECTED_NODES = 4
EXPECTED_GPU_TOTAL = 32          # ⭐ 训练 16 + 推理 16，全部在账本里
TRAIN_NODES = 2
INFER_NODES = 2

TRAIN_ROLE = "train_node"        # 启动时用 --resources='{"train_node": 8}' 声明
INFER_ROLE = "infer_node"


@ray.remote(num_gpus=1, num_cpus=1)
class TrainProbe:
    """假装一个训练 rank。"""

    def __init__(self, rank: int):
        self.rank = rank

    def where(self) -> tuple[int, str, str]:
        return (self.rank, ray.util.get_node_ip_address(),
                os.environ.get("CUDA_VISIBLE_DEVICES", "?"))


@ray.remote(num_gpus=2, num_cpus=2)
class EngineProbe:
    """假装一个 SGLang 引擎：持【2 张卡】（tp=2）。"""

    def __init__(self, idx: int):
        self.idx = idx

    def where(self) -> tuple[int, str, str]:
        return (self.idx, ray.util.get_node_ip_address(),
                os.environ.get("CUDA_VISIBLE_DEVICES", "?"))


def main() -> int:
    ray.init(address="auto")
    ok = True

    # ── ① 账本 ────────────────────────────────────────────────────
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"存活节点数: {len(nodes)}  (期望 {EXPECTED_NODES})\n")

    total_gpu = 0
    train_seen = infer_seen = 0
    for n in sorted(nodes, key=lambda x: x["NodeManagerAddress"]):
        r = n["Resources"]
        gpu = int(r.get("GPU", 0))
        total_gpu += gpu
        role = ("训练" if r.get(TRAIN_ROLE) else
                "推理" if r.get(INFER_ROLE) else
                "⚠️ 无角色资源")
        if r.get(TRAIN_ROLE):
            train_seen += 1
        if r.get(INFER_ROLE):
            infer_seen += 1
        print(f"  {n['NodeManagerAddress']:<16} GPU={gpu:<3} CPU={int(r.get('CPU', 0)):<5} "
              f"{role}")

    print(f"\nGPU 总账: {ray.cluster_resources().get('GPU', 0)}  (期望 {EXPECTED_GPU_TOTAL})")
    print(f"训练节点 {train_seen} 个 / 推理节点 {infer_seen} 个"
          f"  (期望 {TRAIN_NODES} / {INFER_NODES})")

    if len(nodes) != EXPECTED_NODES:
        print(f"❌ 节点数 {len(nodes)} != {EXPECTED_NODES}"); ok = False
    if total_gpu != EXPECTED_GPU_TOTAL:
        print(f"❌ GPU 总账 {total_gpu} != {EXPECTED_GPU_TOTAL}"
              f"{'（推理节点漏了 --num-gpus=8？）' if total_gpu < EXPECTED_GPU_TOTAL else ''}")
        ok = False
    if train_seen != TRAIN_NODES or infer_seen != INFER_NODES:
        print(f"❌ 节点角色不对：说明有节点启动时没写 --resources"
              f"，**落点约束会失效**（Ray 可能把训练放到推理节点）")
        ok = False

    # ── ② 落点：两个 PG，bundle 里带角色资源 ──────────────────────
    if "--check-placement" in sys.argv:
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        print("\n--- 落点验证 ---")
        train_pg = placement_group(
            [{"GPU": 1, "CPU": 1, TRAIN_ROLE: 1}] * 16, strategy="PACK")
        eng_pg = placement_group(
            [{"GPU": 2, "CPU": 2, INFER_ROLE: 1}] * 8, strategy="PACK")
        try:
            ray.get([train_pg.ready(), eng_pg.ready()], timeout=30)
        except ray.exceptions.GetTimeoutError:
            print("❌ PG 没 ready —— 角色资源数量对不上（bundle 规格 vs --resources 的量）")
            ray.shutdown()
            return 1

        def sched(pg, i):
            return PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i)

        trains = [TrainProbe.options(scheduling_strategy=sched(train_pg, i)).remote(i)
                  for i in range(16)]
        engines = [EngineProbe.options(scheduling_strategy=sched(eng_pg, i)).remote(i)
                   for i in range(8)]

        train_ips = {w[1] for w in ray.get([t.where.remote() for t in trains])}
        eng_ips = {w[1] for w in ray.get([e.where.remote() for e in engines])}
        train_gpus = {w[2] for w in ray.get([t.where.remote() for t in trains])}
        eng_gpus = {w[2] for w in ray.get([e.where.remote() for e in engines])}

        print(f"  16 个 TrainActor 落在 {len(train_ips)} 个节点上: {sorted(train_ips)}")
        print(f"   它们的 CVD: {sorted(train_gpus)}")
        print(f"  8 个 EngineActor 落在 {len(eng_ips)} 个节点上: {sorted(eng_ips)}")
        print(f"   它们的 CVD: {sorted(eng_gpus)}   ← 引擎持 2 张卡，CVD 应是【两个号】")

        both = train_ips & eng_ips
        if both:
            print(f"❌ 这些节点上【同时】有训练和引擎: {sorted(both)}")
            print("   ⇒ bundle 里的角色资源没起作用，检查 --resources 的量是否够")
            ok = False
        else:
            print("  ✅ 训练与推理完全分离，没有节点同时承载两者")

    print("\n" + ("✅ 集群与落点都正确" if ok else "❌ 有问题，先修再往下"))
    ray.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
