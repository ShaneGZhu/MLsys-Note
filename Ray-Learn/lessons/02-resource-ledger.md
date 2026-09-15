# L02 · 资源账本：Ray 的"卡"是记账，不是绑卡

> 🖥️ **本课需要真机。** Part A 单机 8×H100 即可；Part B 也只要**一台**机器，
> 但它复现的是 Lux 最危险的静默失效——**别跳过 Part B**。

## ① 一句话目标

理解 Ray 的 `num_gpus` **只是一本账**：超额申请**不报错，只是排队**；
而这本账**只知道你声明过的东西**——机器上真实存在、但已被别人占着的卡，它一无所知。

## ② 先预测

**Part A**：一台 8×H100 的机器，我们起 **8 个 actor**，每个 `num_gpus=1`，再起**第 9 个**。

1. 第 9 个会怎样？（A. 报错 / B. 排队 / C. 挤到某张卡上）
2. `ray.available_resources()["GPU"]` 在 8 个都起来后是多少？

**Part B**：机器上 8 张卡，其中 **4 张（GPU 4–7）被另一个进程占了**（模拟 Dynamo 拉起的 SGLang 引擎）。

3. 如果 `ray start` 时报 **8** 张卡，会怎样？
4. 应该报几？

## ③ Part A · 账本是什么

存成 `lessons/02a_ledger.py`：

```python
"""L02 Part A · 资源账本：超额会排队，不会报错。"""
import time
import ray

T0 = time.time()


@ray.remote(num_gpus=1, num_cpus=1)
class Holder:
    def __init__(self, name: str):
        self.name = name

    def work(self, seconds: float) -> tuple[str, str, float]:
        gpu = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "?")
        time.sleep(seconds)
        return self.name, gpu, round(time.time() - T0, 2)


def main() -> None:
    ray.init()                       # 单机：自动起本地集群，自动探测到 8 张 GPU

    print("集群资源:", {k: v for k, v in ray.cluster_resources().items()
                        if k in ("GPU", "CPU")})
    print("可用 GPU:", ray.available_resources().get("GPU"))
    print()

    # ⭐ 故意超额：8 张卡，起 9 个 actor
    actors = [Holder.remote(f"a{i}") for i in range(9)]
    print("已提交 9 个 actor，账上只有 8 张 GPU")
    time.sleep(1)
    print(f"此刻 available GPU = {ray.available_resources().get('GPU')}")
    print("⚠️ 没有报错。第 9 个被【挂起】了。\n")

    results = ray.get([a.work.remote(2.0) for a in actors])
    for name, gpu, t in sorted(results, key=lambda x: x[2]):
        print(f"  {name}  CUDA_VISIBLE_DEVICES={gpu:<4} 完成于 {t}s")

    print("\n👉 前 8 个在 2s 完成，第 9 个在约 4s —— 它等别人释放了 GPU")
    print("👉 全程零异常。**超额不是错误，是排队。**")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/02a_ledger.py
```

### 期望输出

```
集群资源: {'GPU': 8.0, 'CPU': 128.0}
可用 GPU: 8.0

已提交 9 个 actor，账上只有 8 张 GPU
此刻 available GPU = 0.0
⚠️ 没有报错。第 9 个被【挂起】了。

  a0  CUDA_VISIBLE_DEVICES=3    完成于 2.01s
  a1  CUDA_VISIBLE_DEVICES=1    完成于 2.01s
  ...
  a8  CUDA_VISIBLE_DEVICES=0    完成于 4.03s     ← 等的那个
```

⭐ **注意 `CUDA_VISIBLE_DEVICES` 这一列**：Ray 为每个 actor 分配了**不同的 GPU**，
并替你设好了环境变量。这就是"记账"的实际含义——**它不是一张标签，它真的绑卡**。

> ⚠️ **若不符合预期**
> | 现象 | 原因 |
> | --- | --- |
> | 9 个都在 2s 完成 | 有人在 actor 里自己动了 `CUDA_VISIBLE_DEVICES`，或用了 `num_gpus<1` |
> | 第 9 个永远不完成 | 前面有 actor 没退出（例如异常路径没释放） |
> | `ray.cluster_resources()` 里没有 `GPU` | Ray 没探测到 CUDA。先 `nvidia-smi` 确认，再看 `ray start` 时是否显式写了 `--num-gpus=0` |

---

## ④ Part B · ⭐ 账本 vs 实际占用（本课程最有价值的一节）

> 🖥️ **一台机器就够。** 这一节验的是 Lux 最危险的静默失效。

### 场景

Lux 的 4 个节点里，**2 个推理节点**上有 **Dynamo 自己拉起的 SGLang 引擎**，
它们用 `CUDA_VISIBLE_DEVICES` **直接占住那 8 张卡**——**Ray 完全不知情**。

`Lux/docs/design/control-plane-internals.md` 原文：

> **Ray 只认它自己分配的那 16 张卡。推理侧那 16 张卡在 Ray 的账本之外。**

### 复现（单机版）

我们让一个**外部进程**占住 **GPU 4–7**，充当"引擎"：

**终端 A —— 假冒"引擎"，占住 GPU 4–7**

```bash
python -c "
import torch, time
# 每张卡占 ~70GB，确保后面的人真的抢不到
xs = [torch.zeros(70 * 1024**3 // 4, dtype=torch.uint8, device=f'cuda:{i}')
      for i in range(4, 8)]
print('已占住 cuda:4-7（模拟 Dynamo 拉起的 SGLang 引擎）', flush=True)
time.sleep(3600)
"
```

**终端 B —— 先按「账本 = 8」起 Ray（❌ 错误做法）**

```bash
ray stop
ray start --head --num-gpus=8        # ⚠️ 报了全部 8 张，但 4 张已被占
```

然后在**终端 C** 跑：

```bash
uv run python lessons/02b_occupancy.py --expect-fail
```

`lessons/02b_occupancy.py`：

```python
"""L02 Part B · 账本 vs 实际占用。"""
import os
import sys
import ray
import torch


@ray.remote(num_gpus=1, num_cpus=1)
class TrainActor:
    def __init__(self, rank: int):
        self.rank = rank
        self.dev = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    def try_allocate(self, gib: int = 20) -> str:
        try:
            n = gib * 1024**3 // 4
            self._x = torch.zeros(n, dtype=torch.uint8, device="cuda:0")
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"rank={self.rank} cvd={self.dev} ✅ 占到 {gib}GB（卡共 {total:.0f}GB）"
        except torch.cuda.OutOfMemoryError as e:
            return f"rank={self.rank} cvd={self.dev} ❌ OOM：{str(e)[:80]}"


def main() -> None:
    expect_fail = "--expect-fail" in sys.argv
    ray.init(address="auto")

    print("Ray 认为有:", ray.cluster_resources().get("GPU"), "张 GPU")
    print("物理上由 nvidia-smi 看（Ray 不知道这部分）\n")

    actors = [TrainActor.remote(i) for i in range(8)]
    for line in ray.get([a.try_allocate.remote(20) for a in actors]):
        print(" ", line)

    print()
    if expect_fail:
        print("👉 看 cvd=4/5/6/7 那几个：**Ray 把 actor 放到了已经被引擎占住的卡上**")
        print("   如果它们 OOM，说明 Ray 的账本与物理现实不一致 —— 这正是要避免的。")
        print("   ⚠️ 若没有 OOM（因为 70GB 占用 + 20GB 申请刚好挤下），")
        print("      把占用调到 75GB 或申请调到 40GB 再试 —— 现象要能看见才算数。")
    else:
        print("👉 全部成功 = 账本与物理现实一致。")


if __name__ == "__main__":
    main()
```

**终端 B —— 再按「账本 = 4」起 Ray（✅ 正确做法）**

```bash
ray stop
ray start --head --num-gpus=4        # ⭐ 只报没被占的那 4 张
uv run python lessons/02b_occupancy.py
```

### 期望对比

| `--num-gpus` | Ray 认为有几张 | 结果 |
| --- | --- | --- |
| `8`（❌） | 8 | 8 个 actor 全部启动；落在 `cvd=4..7` 的那几个 **OOM** |
| `4`（✅） | 4 | 只有 4 个 actor 能起来；**没有一个落到被占的卡上** |

### ⚠️ 一个重要的细节：Ray 挑的是**哪** 4 张

`--num-gpus=4` 让 Ray 认为有 4 张，但它**默认取物理索引最小的 4 张**（0–3）。
所以本实验特意让占用落在 **4–7**（高半边）——这样"少报"才等价于"避开"。

**生产上这一点必须显式控制**，不能靠运气。Lux 的对策是**推理节点 `--num-gpus=0`**——
不靠"少报 4 张"，而是**一张都不报**，从根上消除歧义。

> ⚠️ **如果 Ray 挑的卡不是 0–3**（不同版本行为可能不同）：把占用挪到 Ray 实际会挑的那几张上，
> 或者用 `CUDA_VISIBLE_DEVICES=0,1,2,3 ray start ...` 显式限定。
> **这一步的结论是"Ray 挑哪几张"，请以你实测到的为准并记下来。**

---

## ⑤ ⭐ 为什么 Lux 关心这个

### ① 这不是假想问题，是 Lux 的真实拓扑

| 谁 | 多少张卡 | 进 Ray 的账本吗 |
| --- | --- | --- |
| `TrainActor ×16` | 16 | ✅ 进（`num_gpus=1`） |
| `SGLang 引擎 ×8`（各 tp=2） | 16 | ❌ **不进**——Dynamo 拉起，Ray 不知情 |

⇒ **如果两个推理节点 join Ray 时带上了各自的 8 张卡，Ray 会以为自己有 32 张**，
于是**可能把 `TrainActor` 调度到推理节点上**——那张卡上正跑着 SGLang 引擎。

后果**不是"慢"**：FSDP2 的 16 路分片落到错误的卡上，表现为 **NCCL 挂住 / 拓扑错乱**，
而**没有任何东西会提示"GPU 记错了"**。

### ② 所以有一条硬断言

`Lux/docs/runbooks/s0-verification.md` 阶段 0.1 的通过条件：

```bash
ray status      # Resources 里的 GPU 总数必须是 16，不是 32
```

### ③ 一句话总结

> **Ray 的资源是「声明式账本」，不是「运行时探测」。**
> 你没声明的，它当不存在；你**多**声明的，它会真的往上调度。
> **账本错一次，调度就错，而且不报错。**
