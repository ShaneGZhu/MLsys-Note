# L02 · 资源账本：Ray 的"卡"是记账，不是绑卡

> 🖥️ **本课需要真机。** Part A 单机 8×H100 即可；Part B 也只要**一台**机器，
> 但它复现的是 Lux 最危险的静默失效——**别跳过 Part B**。

> ⚠️ **2026-09-15 修正**：本课初版把"超额会排队"写成了对 **actor** 也成立，**那是错的**——
> actor 会**永久挂死**而不是排队（初版代码因此会卡住，如果你跑过初版，它会一直不返回）。
> 现在 Part A 分成 **A1（task：排队）** 与 **A2（actor：挂死）**，这个区别才是本课真正的重点。

## ① 一句话目标

理解 Ray 的 `num_gpus` **只是一本账**；而这本账对**超额**的反应**取决于你用的是 task 还是 actor**——
**task 排队，actor 永久挂死**。两者都不报错。

## ② 先预测

**Part A1**：8 张卡，提交 **9 个 task**（各 `num_gpus=1`，每个跑 2 秒）。

1. 第 9 个 task 什么时候完成？（A. 2s / B. 4s / C. 永远不完成）
2. 全程会报错吗？

**Part A2**：同样 8 张卡，改成创建 **9 个 actor**（各 `num_gpus=1`）。

3. 第 9 个 actor 会怎样？（A. 2s 后跑 / B. 排队等 / C. **根本不创建，一直等下去**）

**Part B**：8 张卡里有 **4 张（GPU 4–7）被另一个进程占了**（模拟 Dynamo 拉起的 SGLang 引擎）。

4. `ray start` 时报 **8** 张会怎样？应该报几？

---

## ③ Part A1 · task：超额会**排队**，最终跑完

存成 `lessons/02a_ledger.py`：

```python
"""L02 Part A · 资源账本：task 会排队。"""
import os
import time
import ray

T0 = time.time()


# ⭐ 注意这里是 task（函数），不是 actor（类）
@ray.remote(num_gpus=1, num_cpus=1)
def hold(name: str, seconds: float) -> tuple[str, str, float]:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    time.sleep(seconds)
    return name, gpu, round(time.time() - T0, 2)


def main() -> None:
    ray.init()                       # 单机：自动起本地集群，自动探测到 8 张 GPU

    print("集群资源:", {k: v for k, v in ray.cluster_resources().items()
                        if k in ("GPU", "CPU")})
    print("可用 GPU:", ray.available_resources().get("GPU"))
    print()

    # 提交 9 个 task，账上只有 8 张卡
    refs = [hold.remote(f"t{i}", 2.0) for i in range(9)]
    time.sleep(1)
    print("已提交 9 个 task（各要 1 张卡），账上只有 8 张")
    print(f"此刻 available GPU = {ray.available_resources().get('GPU')}")
    print("⚠️ 没有报错。第 9 个被【挂起】了。\n")

    for name, gpu, t in sorted(ray.get(refs), key=lambda r: r[2]):
        mark = "   ← 等的那个" if t > 3 else ""
        print(f"  {name}  CUDA_VISIBLE_DEVICES={gpu:<3} 完成于 {t}s{mark}")

    print("\n👉 前 8 个约 2s，第 9 个约 4s —— 它等别人【跑完并释放】了卡")
    print("👉 全程零异常。**task 的超额不是错误，是排队。**")

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

已提交 9 个 task（各要 1 张卡），账上只有 8 张
此刻 available GPU = 0.0
⚠️ 没有报错。第 9 个被【挂起】了。

  t3  CUDA_VISIBLE_DEVICES=3   完成于 2.01s
  t1  CUDA_VISIBLE_DEVICES=1   完成于 2.01s
  ...
  t8  CUDA_VISIBLE_DEVICES=0   完成于 4.03s   ← 等的那个
```

⭐ **注意 `CUDA_VISIBLE_DEVICES` 这一列**：Ray 给每个 task 分了**不同的卡**并替你设好环境变量。
**"记账"不是贴标签，它真的绑卡。**

---

## ④ ⭐ Part A2 · actor：超额会**永久挂死**（这个区别才是重点）

把上面的 `hold` **改成 `Holder` 类**（actor），其余不变，第 9 个就**永远不会跑**。

存成 `lessons/02a_actor_trap.py`：

```python
"""L02 Part A2 · ⚠️ actor 超额的后果：永久挂死。"""
import os
import time
import ray

T0 = time.time()


@ray.remote(num_gpus=1, num_cpus=1)
class Holder:
    def __init__(self, name: str):
        self.name = name

    def work(self, seconds: float) -> tuple[str, str, float]:
        return (self.name,
                os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                round(time.time() - T0, 2))


def main() -> None:
    ray.init()
    print("账本 GPU:", ray.cluster_resources().get("GPU"))

    # 创建 9 个 actor，但只有 8 张卡
    holders = [Holder.remote(f"a{i}") for i in range(9)]
    time.sleep(2)                       # 给调度器一点时间
    print(f"已请求 9 个 actor；此刻 available GPU = "
          f"{ray.available_resources().get('GPU')}")
    print("⚠️ 没有报错，也没有 warning 说你资源不够\n")

    # ── 前 8 个正常 ──────────────────────────────────────────────
    got = ray.get([h.work.remote(0.1) for h in holders[:8]])
    print(f"前 8 个正常返回: {[g[0] for g in got]}")

    # ── 第 9 个：用 timeout 安全地证明"它永远不会来" ─────────────
    print("\n第 9 个 actor 呢？我们给它 5 秒：")
    try:
        r = ray.get(holders[8].work.remote(0.1), timeout=5)
        print(f"  ⚠️ 竟然返回了: {r} —— 与预期不符，见下方排查")
    except ray.exceptions.GetTimeoutError:
        print("  ✅ 5 秒超时。**第 9 个 actor 根本没有被创建** ——")
        print("     它没有崩溃，它在等一张【永远不会空出来】的卡。")

    print("\n⚠️ 关键：actor 在【创建时】就占住 num_gpus，并持有到它【整个生命周期】结束。")
    print("   不是每次方法调用占一下 —— 所以它不会像 task 那样跑完就释放。")
    print("   ⇒ actor 的超额 = 永久挂死；任何 ray.get 在它上面都会一直等下去。")

    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv run python lessons/02a_actor_trap.py
```

### 期望输出

```
账本 GPU: 8.0
已请求 9 个 actor；此刻 available GPU = 0.0
⚠️ 没有报错，也没有 warning 说你资源不够

前 8 个正常返回: ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

第 9 个 actor 呢？我们给它 5 秒：
  ✅ 5 秒超时。**第 9 个 actor 根本没有被创建** ——
     它没有崩溃，它在等一张【永远不会空出来】的卡。

⚠️ 关键：actor 在【创建时】就占住 num_gpus，并持有到它【整个生命周期】结束。
   ⇒ actor 的超额 = 永久挂死；任何 ray.get 在它上面都会一直等下去。
```

### ⭐ task vs actor：一张表记住

| | 什么时候拿资源 | 什么时候释放 | 超额的后果 |
| --- | --- | --- | --- |
| **task**（`@ray.remote` 函数） | 被调度执行时 | **执行完就释放** | **排队**，最终跑完 |
| **actor**（`@ray.remote` 类） | **创建时** | **actor 死亡时** | ⚠️ **永久挂死** |

> ⚠️ **这就是初版这一课错的地方**，也是我写这个 lesson 时踩进去的坑：
> "超额会排队"只对 **task** 成立。**对 actor，超额不是排队，是一个永远不会返回的 `ray.get`。**

### ⚠️ 如果不符合预期

| 现象 | 原因 |
| --- | --- |
| 9 个 task 全在 2s 完成 | 用的不是 `num_gpus`（比如写成了 `num_cpus`，而 CPU 有 128 个） |
| 第 9 个 task 永远不完成 | 有 task 卡住了没释放；或有人在 actor 里占着卡 |
| A2 里第 9 个 actor 竟然跑起来了 | 机器上其实有 ≥9 张卡；先 `nvidia-smi -L \| wc -l` 确认 |
| `cluster_resources()` 没有 `GPU` | Ray 没探测到 CUDA。先 `nvidia-smi`，再看 `ray start` 时是否写了 `--num-gpus=0` |

---

## ⑤ Part B · ⭐ 账本 vs 实际占用（本课程最有价值的一节）

> 🖥️ **一台机器就够。** 验的是 Lux 最危险的静默失效。

### 场景

Lux 的 4 个节点里，**2 个推理节点**上有 **Dynamo 自己拉起的 SGLang 引擎**，
它们用 `CUDA_VISIBLE_DEVICES` **直接占住那 8 张卡**——**Ray 完全不知情**。

`Lux/docs/design/control-plane-internals.md` 原文：

> **Ray 只认它自己分配的那 16 张卡。推理侧那 16 张卡在 Ray 的账本之外。**

### 复现（单机版）

让一个**外部进程**占住 **GPU 4–7**，充当"引擎"。

**终端 A —— 假冒"引擎"**

```bash
python -c "
import torch, time
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

**终端 C**：

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


# ⭐ 用 task 而不是 actor：task 跑完就释放，所以 --num-gpus=4 那一轮不会挂住
@ray.remote(num_gpus=1, num_cpus=1)
def probe(rank: int, gib: int = 20) -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    try:
        _x = torch.zeros(gib * 1024**3 // 4, dtype=torch.uint8, device="cuda:0")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"rank={rank} cvd={cvd} ✅ 占到 {gib}GB（卡共 {total:.0f}GB）"
    except torch.cuda.OutOfMemoryError as e:
        return f"rank={rank} cvd={cvd} ❌ OOM: {str(e)[:60]}"


def main() -> None:
    expect_fail = "--expect-fail" in sys.argv
    ray.init(address="auto")            # 连到 ray start 起的集群

    print("Ray 认为有:", ray.cluster_resources().get("GPU"), "张 GPU")
    print("（物理上有几张由 nvidia-smi 说了算 —— Ray 不知道那部分）\n")

    for line in ray.get([probe.remote(i) for i in range(8)]):
        print(" ", line)

    print()
    if expect_fail:
        print("👉 看 cvd=4/5/6/7 的：**Ray 把 task 放到了已被引擎占住的卡上**")
        print("   它们 OOM 就说明账本与物理现实不一致 —— 这正是要避免的。")
        print("   ⚠️ 若没 OOM（70GB 占用 + 20GB 申请刚好挤下），")
        print("      把占用调到 75GB 或申请调到 40GB 再试 —— 现象要能看见才算数。")
    else:
        print("👉 没有一个落到 4–7 上 = 账本与物理现实一致。")


if __name__ == "__main__":
    main()
```

**然后按「账本 = 4」重来（✅ 正确做法）**

```bash
ray stop
ray start --head --num-gpus=4        # ⭐ 只报没被占的那 4 张
uv run python lessons/02b_occupancy.py
```

### 期望对比

| `--num-gpus` | Ray 认为有几张 | 结果 |
| --- | --- | --- |
| `8`（❌） | 8 | 8 个 task 全部启动；落在 `cvd=4..7` 的 **OOM** |
| `4`（✅） | 4 | 只有 4 个同时跑，其余排队；**没有一个落到 4–7 上** |

### ⚠️ 一个重要细节：Ray 挑的是**哪** 4 张

`--num-gpus=4` 让 Ray 认为有 4 张，但它**默认取物理索引最小的 4 张**（0–3）。
所以本实验特意把占用放在 **4–7（高半边）**——"少报"才等价于"避开"。

**生产上这一点必须显式控制，不能靠运气。** Lux 的对策是推理节点 **`--num-gpus=0`**：
不靠"少报 4 张"，而是**一张都不报**，从根上消除歧义。

> ⚠️ **如果 Ray 挑的不是 0–3**（版本差异）：把占用挪到 Ray 实际会挑的那几张，
> 或用 `CUDA_VISIBLE_DEVICES=0,1,2,3 ray start ...` 显式限定。
> **本步的结论是"Ray 挑哪几张"，请以实测为准并记下来。**

---

## ⑥ ⭐ 为什么 Lux 关心这个

### ① 不是假想问题，是 Lux 的真实拓扑

| 谁 | 多少张卡 | 进 Ray 的账本吗 |
| --- | --- | --- |
| `TrainActor ×16` | 16 | ✅ 进（`num_gpus=1`） |
| `SGLang 引擎 ×8`（各 tp=2） | 16 | ❌ **不进**——Dynamo 拉起，Ray 不知情 |

⇒ **如果两个推理节点 join Ray 时带上各自的 8 张卡，Ray 会以为自己有 32 张**，
可能把 `TrainActor` 调度到推理节点上——那张卡上正跑着 SGLang 引擎。

后果**不是"慢"**：FSDP2 的 16 路分片落到错误的卡上，表现为 **NCCL 挂住 / 拓扑错乱**，
而**没有任何东西提示"GPU 记错了"**。

### ② ⭐⭐ Part A2 那条对 Lux 是硬约束

Lux 起 **16 个 `TrainActor`，每个 `num_gpus=1`**。而 actor **在创建时占卡、持有到死**：

> **16 个 actor 必须正好配 16 张能用的卡。多一个 ⇒ 第 17 个永远不被创建，
> 任何等它的 `ray.get` 永远不返回 —— 而日志里没有一行报错。**

这解释了 Lux 为什么必须：
1. **用 PG 一次要齐 16 个 bundle**（[L03](03-placement-group.md)）——不能"先起 15 个"
2. **`LuxDriver` 必须监视 actor 的创建**，而不是假设 "`ray.get` 会返回"
3. **看门狗必须是进程内线程**（[L05](05-sync-vs-async.md) §⑤）——因为这种挂死**不会自己超时**

### ③ 所以有一条硬断言

`Lux/docs/runbooks/s0-verification.md` 阶段 0.1 的通过条件：

```bash
ray status      # Resources 里的 GPU 总数必须是 16，不是 32
```

### ④ 一句话总结

> **Ray 的资源是「声明式账本」，不是「运行时探测」。**
> 你没声明的，它当不存在；你**多**声明的，它会真的往上调度。
> **而超额的后果取决于 task 还是 actor：task 排队，actor 永久挂死。两者都不报错。**
