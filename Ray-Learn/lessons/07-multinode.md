# L07 · 把 4 个节点**全部**交给 Ray（32 张卡）

> 🖥️ **需要 4 台 H100**（2 训 + 2 推）。本课是唯一无法在单机上完成的课。

> ⚠️⚠️ **2026-09 设计变更——本课的前提整个反了。**
> | | 旧设计 | **新设计** |
> | --- | --- | --- |
> | 推理节点的 16 张卡 | **不进** Ray 账本（`--num-gpus=0`） | ⭐ **进**账本（`--num-gpus=8`） |
> | 主要风险 | Ray 看不见推理卡 | ⚠️ **Ray 把 `TrainActor` 放到推理卡上** |
> | 靠什么防 | 藏起来 | ⭐ **用自定义资源给节点打角色 + 写进 bundle 规格** |
>
> ⇒ **"少报卡"这个手段不再适用**。新设计下 Ray 有能力把训练放到任何地方，
> **必须显式约束**。（旧版这一课教的是 `--num-gpus=0`，与现行设计相反，已整体重写。）

## ① 一句话目标

起一个 4 节点 / 32 卡的 Ray 集群，并让 Ray **只把训练放训练节点、只把引擎放推理节点**。

## ② 先预测

Ray 现在看得见全部 32 张卡。

1. 它怎么知道哪 16 张该放训练、哪 16 张该放引擎？
2. 如果什么都不做，会怎样？
3. `--num-gpus=8` 在推理节点上意味着什么？

## ③ 步骤

### 3.1 ⭐ 节点角色 = **自定义资源**（新设计的核心机制）

`ray start --resources` 收一个 JSON。我们用它给节点"贴标签"：

```
--resources='{"train_node": 8}'     # 训练节点：8 个单位（够 8 个 bundle 各要 1）
--resources='{"infer_node": 8}'     # 推理节点
```

⚠️ 为什么不用 node labels：**Ray 2.58.0 的 `ray start` 没有 `--labels` 参数**
（`ray start --help | grep -c label` → **0**）。`.options(label_selector=...)` 语法上接受，
但**没有 CLI 能设置它**。⇒ 现阶段可用且可移植的机制就是 `--resources`。

### 3.2 起 head（训练节点 0）

```bash
ray stop
ray start --head \
  --node-ip-address=<训练节点0 的高速网 IP> \
  --num-gpus=8 --num-cpus=$(nproc) \
  --resources='{"train_node": 8}' \
  --dashboard-host=0.0.0.0
```

### 3.3 训练节点 1

```bash
ray start --address=<训练节点0 的 IP>:6379 \
  --node-ip-address=<训练节点1 的 IP> \
  --num-gpus=8 --num-cpus=$(nproc) \
  --resources='{"train_node": 8}'
```

### 3.4 两个推理节点（⭐ 注意是 `--num-gpus=8` + `infer_node`）

```bash
ray start --address=<训练节点0 的 IP>:6379 \
  --node-ip-address=<本机 IP> \
  --num-gpus=8 --num-cpus=$(nproc) \
  --resources='{"infer_node": 8}'
```

> ⚠️ **`--num-cpus` 取值是个真问题**：推理节点的 CPU 要同时供引擎（tokenize / detokenize）
> 和 Ray actor 用。给 Ray 报满会让两者抢 CPU。具体留多少是 S0 要实测的。

## ④ 断言（这一步不过，后面全是错的）

存成 `lessons/py/07_check_cluster.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
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
```

```bash
uv run python lessons/py/07_check_cluster.py
```

期望输出：

```
存活节点数: 4  (期望 4)

  10.0.0.11        GPU=8   CPU=128  训练
  10.0.0.12        GPU=8   CPU=128  训练
  10.0.0.21        GPU=8   CPU=128  推理
  10.0.0.22        GPU=8   CPU=128  推理

GPU 总账: 32.0  (期望 32)
训练节点 2 个 / 推理节点 2 个  (期望 2 / 2)

✅ 集群与落点都正确
```

## ⑤ 落点验证：两个 PG，bundle 规格里带角色

```bash
uv run python lessons/py/07_check_cluster.py --check-placement
```

脚本会建**两个** PG，**bundle 里带上角色资源**：

```python
train_pg = placement_group([{"GPU": 1, "CPU": 1, "train_node": 1}] * 16)
eng_pg   = placement_group([{"GPU": 2, "CPU": 2, "infer_node": 1}] * 8)
```

⭐ **为什么这样就够了**：PG 的 bundle **只能被满足它的节点满足**。带 `train_node: 1` 的
bundle 落不到推理节点上（那里没有这个资源）——**约束是资源系统强制的，不是靠自觉**。

期望输出：

```
  16 个 TrainActor 落在 2 个节点上: ['10.0.0.11', '10.0.0.12']
   它们的 CVD: ['0', '1', '2', '3', '4', '5', '6', '7']
  8 个 EngineActor 落在 2 个节点上: ['10.0.0.21', '10.0.0.22']
   它们的 CVD: ['0,1', '2,3', '4,5', '6,7']   ← 引擎持 2 张卡，CVD 是【两个号】

  ✅ 训练与推理完全分离，没有节点同时承载两者
```

## ⑥ 排错表

| 现象 | 原因 |
| --- | --- |
| `GPU 总账 = 16` | 推理节点漏了 `--num-gpus=8`（旧设计才是 0，**新设计要 8**） |
| `⚠️ 无角色资源` | 该节点 `ray start` 时没写 `--resources` ⇒ **落点约束整体失效** |
| `PG 没 ready` | bundle 要的角色资源数量 > 节点声明的量（如 16 个 bundle 各要 1，但只声明了 8） |
| **训练和引擎落在同一节点** | ⭐ 最危险：bundle 里没写角色资源，或节点漏了 `--resources`。**这是静默的**——不会报错 |
| worker 连不上 head | 6379 被防火墙挡了，或 `--node-ip-address` 写成了 `127.0.0.1` |
| 多机 gloo/NCCL 超时 | `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` 没设，走了管理网 |

**收工**（每个节点）：`ray stop`

## ⑦ 为什么 Lux 关心这个

### ① 这一课 = Lux S0 的**阶段 0.1**

但**通过条件变了**（因为设计变了）：

| | 旧 | **新** |
| --- | --- | --- |
| 节点数 | 4 | 4 |
| **GPU 总账** | **16**（推理侧不进账本） | ⭐ **32**（全部进账本） |
| 新增断言 | — | ⭐ **没有任何节点同时承载训练与引擎** |

⚠️ **新增的那条才是新设计下真正的风险**。旧设计里"Ray 把训练放到推理节点"是**不可能**的
（它看不见那些卡）；新设计里它**变得可能**，而且**不报错**——你只会看到训练莫名其妙变慢，
或者引擎 OOM。

⇒ **风险从"账本少报了"变成"落点没约束"。两者都是静默的，但后者只能靠显式机制防。**

### ② 这个变更连锁影响的东西（只列，不在本课展开）

| 项 | 变化 |
| --- | --- |
| SGLang 引擎 | 从"Dynamo 自己拉起的普通进程" → **Ray actor 持卡 + 引擎作子进程**（→ [L10](10-engine-actor.md)） |
| ADR-0003 的一条收益 | 「Dynamo 自己拉起」不再是收益，要改写 |
| S0 必验 #17 | 断言从「GPU = 16，不是 32」**反过来**变成「GPU = 32，且落点分离」 |
| `--num-gpus=0` | ❌ **整个手段作废** |

## ⑧ 一句话总结

> **Ray 的账本是声明式的**：你不声明的它当不存在，你声明的它就会往上调度。
> 旧设计用"少报"来防止误调度；**新设计把卡全部交给 Ray，代价是必须自己加约束**
> —— 用**自定义资源写进 bundle 规格**，让资源系统去强制它。

