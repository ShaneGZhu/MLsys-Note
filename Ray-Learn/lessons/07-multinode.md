# L07 · 多机集群：起 Ray，并断言 GPU 账本是 16 不是 32

> ⚠️ **本课必须上真集群**（Lux 目标：4 节点 × 8×H100）。单机无法替代——本课的核心现象
> "跨节点分布"和"GPU 账本"在单节点上根本不会出现。

> 🖥️ **需要 4 台 H100 机器**（2 训 + 2 推）。这是本课程唯一无法在单机上完成的课。

## ① 一句话目标

把 4 个节点组成一个 Ray 集群，并**用一个断言确认账本是对的**：
**GPU 总数必须是 16，不是 32**。

## ② 先预测

Lux 的 4 个节点：**2 个训练节点 + 2 个推理节点**，每个节点 8 张 H100。

问题：`ray status` 里的 `GPU` 总数应该是多少？

- A. 32（4 节点 × 8 卡）
- B. **16**（只有训练节点那 16 张）
- C. 0（Ray 不会自动发现）

> 提示：**上一课（[L02](02-resource-ledger.md)）已经给了答案**——Ray 的资源是**声明式账本**。
> 但这个答案在真集群上的**后果**，只有本课能看到。

## ③ 步骤

### 3.1 先想清楚：为什么推理节点的 GPU 不能入账

| | 谁拉起 | 用哪 8 张卡 | 进 Ray 账本吗 |
| --- | --- | --- | --- |
| **训练节点 ×2** | `TrainActor`（Ray actor） | Ray 分配 | ✅ **进**（`num_gpus=1` × 16） |
| **推理节点 ×2** | **Dynamo 自己拉起 SGLang 引擎** | `CUDA_VISIBLE_DEVICES` 直接占 | ❌ **不进** |

⇒ **如果推理节点带着 8 张卡 join Ray，Ray 会以为自己有 32 张**，
于是**可能把 `TrainActor` 调度到推理节点上**——那张卡上正跑着 SGLang 引擎。

后果不是"慢"：FSDP2 的 16 路分片落到错误的卡上，**NCCL 挂住 / 拓扑错乱**，
而**没有任何东西会提示"GPU 记错了"**。

### 3.2 起集群

在 **head 节点**（建议用训练节点 0）：

```bash
# ⚠️ 显式指定网卡，避免 Ray 挑到管理网
export RAY_GCS_SERVER_PORT=6379

ray start --head \
  --node-ip-address=<训练节点0 的高速网 IP> \
  --num-gpus=8 \
  --num-cpus=$(nproc) \
  --dashboard-host=0.0.0.0
```

在**训练节点 1**：

```bash
ray start --address=<训练节点0 的 IP>:6379 \
  --node-ip-address=<训练节点1 的 IP> \
  --num-gpus=8 \
  --num-cpus=$(nproc)
```

在**两个推理节点**（⭐ 注意 `--num-gpus=0`）：

```bash
ray start --address=<训练节点0 的 IP>:6379 \
  --node-ip-address=<本机 IP> \
  --num-gpus=0 \
  --num-cpus=$(nproc)
```

> ⚠️ **`--num-gpus=0` 是本课的整个要点。** 推理节点的卡确实存在，但**不告诉 Ray**——
> 因为那 8 张卡已经被 Dynamo 拉起的引擎占了。
>
> 📌 `--num-cpus` 取值是个真实的设计问题：推理节点的 CPU 要同时供 SGLang 引擎
> （tokenize / detokenize）和 `RolloutActor` 用。**给 Ray 报满会让两者抢 CPU**。
> 具体留多少，是 S0 要实测的（本课先按 `nproc` 起，观察争抢）。

### 3.3 ⚠️ 网络前提

Ray 跨节点要通端口。**至少要放行**（以 `ray start` 自己打印的为准）：

| 端口 | 用途 |
| --- | --- |
| `6379` | GCS（head 独有，**worker 要连它**） |
| `8265` | dashboard |
| 其余 | `ray start` 会打印实际使用的端口列表 |

另外，**高速网卡必须被显式指定**，否则 Ray 的 CPU 侧通信（GLOO）可能走管理网：

```bash
export GLOO_SOCKET_IFNAME=<高速网卡>     # 与 torch 的 gloo 共用
```

参考 `Lux/docs/runbooks/multi-node-network-check.md`——那里有 IB / RoCE 的识别方法
（`ibdev2netdev`、`NCCL_IB_HCA` 的坑）。

## ④ 断言：这一步不过，后面全是错的

存成 `lessons/py/07_check_cluster.py` —— 📄 **可运行版本就在该文件里，⭐ 以它为准**（本文下面的代码块与它同步维护；改代码请改 `py/`，再回填这里）：

```python
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
```

```bash
uv run python lessons/py/07_check_cluster.py
```

期望输出：

```
存活节点数: 4
  10.0.0.11        CPU=128.0  GPU=8.0
  10.0.0.12        CPU=128.0  GPU=8.0
  10.0.0.21        CPU=128.0  GPU=0.0      ← 推理节点，0 卡
  10.0.0.22        CPU=128.0  GPU=0.0      ← 推理节点，0 卡

GPU 总账: 16.0
CPU 总账: 512.0

✅ 集群账本正确
```

也可以直接用 CLI 看：

```bash
ray status
# Resources 一节里应出现： GPU: 16.0/16.0
```

## ⑤ 跨节点验证：PG 真的把人放到不同节点了吗

这是 [L03](03-placement-group.md) 在单机上**看不到**的部分。把 `TrainActor` 放到
`10.0.0.11` / `10.0.0.12` 两台机器上：

```python
# 4 个 actor，STRICT_SPREAD：强制分到不同节点
pg = placement_group([{"GPU": 1, "CPU": 1}] * 4, strategy="STRICT_SPREAD")
ray.get(pg.ready())

actors = [
    A.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=i)).remote(i)
    for i in range(4)
]
for info in ray.get([a.where.remote() for a in actors]):
    print(info["rank"], info["node_ip"])
```

**应看到 4 个 actor 分布在 2 台训练节点上**（`STRICT_SPREAD` 会尽量打散）。

⚠️ 若 4 个全落在同一台 —— 说明 PG 的 bundle 定义里没写 `GPU`（只有 CPU 时 Ray 可能都塞一处）。
**这正是"bundle 里写什么资源，决定了它能被放在哪"**。

### ⭐ 然后验 bundle 重排序（Lux 的正确性依赖）

跑**两次**同样的绑定，把 `rank → (node_ip, gpu_id)` 记下来：

```bash
uv run python lessons/py/07_check_cluster.py --dump-ranks > run1.txt
uv run python lessons/py/07_check_cluster.py --dump-ranks > run2.txt
diff run1.txt run2.txt && echo "✅ rank↔GPU 映射稳定"
```

**两次必须完全一致**。若不一致，就必须做 slime 那套 **bundle 重排序**
（`third_party/slime/slime/ray/placement_group.py`，253 行）。

## ⑥ 排错表

| 现象 | 原因 |
| --- | --- |
| worker 卡在 `ray start --address` 不动 | head 的 `6379` 被防火墙挡了；或 `--node-ip-address` 写成了 `127.0.0.1` |
| `ray status` 里 GPU = 32 | ⭐ 推理节点忘了 `--num-gpus=0` |
| `ray status` 里 GPU = 0 | 训练节点没写 `--num-gpus=8`（Ray 有时探测不到） |
| 节点的 IP 全是 `127.0.0.1` | `--node-ip-address` 没指定，Ray 挑错了网卡 |
| 多机 gloo 通信超时 | `GLOO_SOCKET_IFNAME` 没设，走了管理网 |
| actor 起不来且不报错 | PG 没 ready（[L03](03-placement-group.md) 的 gang 语义）——先 `ray.get(pg.ready())` |

**收工**（每个节点都要跑）：

```bash
ray stop
```

## ⑦ 为什么 Lux 关心这个

本课的可交付物**就是** Lux S0 的**阶段 0.1**（`Lux/docs/runbooks/s0-verification.md`）：

| 步 | 做什么 | 通过条件 |
| ---: | --- | --- |
| **0.1** | 起 Ray，4 个节点全部 join | `ray status` 看得到 **4 个节点**；且 **`Resources` 里的 GPU 总数必须是 16，不是 32** |

而它**在今天之前是漏掉的**——`s0-verification.md` 原先的阶段 0 里**完全没有"起 Ray"这一步**
（`grep -i ray` 零命中），但阶段 1 之后的每一个验证都依赖 Ray actor。

⇒ **跑通本课 = S0 阶段 0.1 完成**，而且是带着一个**可执行的断言**完成的，
不是"看起来起来了"。

## ⑧ 一句话总结

> **Ray 的资源账本是你声明的，不是它探测的。**
> 在一个"有些卡被别的系统占着"的集群上，**必须主动把那些卡从账本里去掉**（`--num-gpus=0`），
> 否则 Ray 会把训练任务调度到已经跑着推理引擎的卡上——**而且不报错**。
