# L06 · 手工 rendezvous：让 N 个 actor 组成一个 NCCL 进程组

> 🖥️ **单机 8×H100 可做全部内容。** 跨机（world_size=16 跨 2 台）见 §⑤，需要双机。

## ① 一句话目标

搞懂"16 个互相不认识的 actor 怎么组成一个 torch 进程组"——以及**为什么必须有一个 rank 先起来报地址**。

## ② 先预测

4 个 `TrainActor` 要调 `dist.init_process_group()` 组成进程组。

1. 谁决定 `MASTER_ADDR` / `MASTER_PORT`？
2. 如果按顺序 `for rank: ray.get(worker.setup.remote())`，会怎样？
   - A. 正常，依次加入
   - B. **死锁**——第一个永远等不到第二个
   - C. 报错说 rank 不齐
3. 先把 4 个 `setup()` 全部**提交**（`.remote()`），再一起 `ray.get`，会怎样？

## ③ 完整代码

> 需要 torch：`uv sync --extra dist`

存成 `lessons/06_rendezvous.py`：

```python
"""L06 · 手工 rendezvous + NCCL 进程组（真卡）。"""
import os
import socket
import ray
import torch
import torch.distributed as dist

WORLD_SIZE = 4


def free_port() -> int:
    """问操作系统要一个空闲端口。"""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@ray.remote(num_gpus=1, num_cpus=1)
class TrainActor:
    """一个训练 rank。

    ⚠️ __init__ 里【不】调 init_process_group：那会阻塞到所有 rank 齐为止，
    而 rank 0 还要先回答"我的地址和端口是什么"——顺序错了就死锁。
    """

    def __init__(self, rank: int, world_size: int,
                 master_addr: str | None = None, master_port: int | None = None):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port

    def get_addr_and_port(self) -> tuple[str, int]:
        """⭐ 只有 rank 0 会被调用：报出自己的 IP 与一个空闲端口。"""
        self.master_addr = ray.util.get_node_ip_address()
        self.master_port = free_port()
        return self.master_addr, self.master_port

    def setup(self) -> str:
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        # NCCL 下每个 rank 先认领自己的卡（Ray 已按账本设好 CUDA_VISIBLE_DEVICES）
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        return (f"rank {self.rank} 已加入 world_size={dist.get_world_size()} "
                f"cvd={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"{torch.cuda.get_device_name(0)}")

    def allreduce_and_report(self) -> float:
        """每个 rank 贡献 rank+1，在 GPU 上做 all_reduce。"""
        t = torch.tensor([float(self.rank + 1)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def teardown(self) -> None:
        dist.destroy_process_group()


def main() -> None:
    ray.init()

    # ── 1. rank 0 先起来，报出 rendezvous 点 ─────────────────────
    rank0 = TrainActor.remote(0, WORLD_SIZE)
    master_addr, master_port = ray.get(rank0.get_addr_and_port.remote())
    print(f"1. rank 0 报出 rendezvous 点: {master_addr}:{master_port}")

    # ── 2. 其余 rank 带着这个地址起来 ────────────────────────────
    others = [TrainActor.remote(r, WORLD_SIZE, master_addr, master_port)
              for r in range(1, WORLD_SIZE)]
    workers = [rank0, *others]
    print(f"2. 其余 {WORLD_SIZE - 1} 个 rank 已创建（句柄就绪，进程组尚未建立）")

    # ── 3. ⚠️ 必须【全部提交】再一起 get ─────────────────────────
    print("3. 并发提交 setup()（先全部 .remote()，再一起 ray.get）")
    for msg in ray.get([w.setup.remote() for w in workers]):
        print("   ", msg)

    # ── 4. 真正做一次 GPU 集合通信 ───────────────────────────────
    values = ray.get([w.allreduce_and_report.remote() for w in workers])
    print(f"\n4. 每个 rank 看到的 all_reduce 结果: {values}")
    print(f"   期望 1+2+3+{WORLD_SIZE} = {sum(range(1, WORLD_SIZE + 1))}，实际 {values[0]}")

    ray.get([w.teardown.remote() for w in workers])
    ray.shutdown()


if __name__ == "__main__":
    main()
```

```bash
uv sync --extra dist
uv run python lessons/06_rendezvous.py
```

## ④ 你应该观察到什么

```
1. rank 0 报出 rendezvous 点: 10.0.0.11:53124
2. 其余 3 个 rank 已创建（句柄就绪，进程组尚未建立）
3. 并发提交 setup()（先全部 .remote()，再一起 ray.get）
    rank 0 已加入 world_size=4 cvd=0 NVIDIA H100 80GB HBM3
    rank 1 已加入 world_size=4 cvd=1 NVIDIA H100 80GB HBM3
    rank 2 已加入 world_size=4 cvd=2 NVIDIA H100 80GB HBM3
    rank 3 已加入 world_size=4 cvd=3 NVIDIA H100 80GB HBM3

4. 每个 rank 看到的 all_reduce 结果: [10.0, 10.0, 10.0, 10.0]
   期望 1+2+3+4 = 10，实际 10.0
```

⭐ **注意 `cvd=` 这一列**：4 个 rank 拿到了**不同的卡**——这来自 L02 的账本 + L03 的 bundle。

### ⭐ 现在故意把它写错，看死锁

把第 3 步改成顺序等待：

```python
# ❌ 错误写法
for w in workers:
    print(ray.get(w.setup.remote()))
```

跑起来会**挂住不动**（不是报错）：

```
rank 0 的 init_process_group 在等 rank 1/2/3
   ↓
但驱动阻塞在 ray.get(rank0.setup) 上，还没提交 rank 1 的 setup
   ↓
互相等 —— 死锁
```

⇒ **`init_process_group` 是集合操作，所有 rank 必须同时在跑它。**
Ray 的 `.remote()` 是异步提交、`ray.get` 是同步等待——**把 `ray.get` 放进循环，就把并发变成了串行**。

> ⚠️ 这个错误在 Lux 里会表现为"训练卡在启动阶段、**不报错**"。
> 记住这个症状：**看到"不报错但不动"，先怀疑集合操作没凑齐。**

## ⑤ 验证 NCCL 真的走对了链路

上面只证明了"能通"。**走的是 NVLink 还是 PCIe 还是网络**，要用 `NCCL_DEBUG` 看：

```bash
NCCL_DEBUG=INFO uv run python lessons/06_rendezvous.py 2>&1 \
  | grep -iE "via|NET/|NCCL WARN|NVLS" | head -30
```

关键看 `via` 字段：

| 看到 | 含义 |
| --- | --- |
| `via P2P/IPC`（+ `NVLS`） | ✅ 机内走 **NVLink**（H100 上的期望值） |
| `via P2P/CUMEM` | 走 GPU 直接内存访问，也属机内高速路径 |
| `via NET/IB/...` | 跨节点走 **InfiniBand** ✅ |
| `via NET/Socket` | ⚠️ **退化成 TCP**——多机时通常是 `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` 没配对 |

网络环境按 Lux 的 runbook（`Lux/docs/runbooks/multi-node-network-check.md`）：

```bash
export NCCL_SOCKET_IFNAME=<高速网卡>              # 不设可能挑到管理网/慢网卡
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3   # ⚠️ 不要写 mlx5（会同时命中 RoCE 设备）
```

### 跨机（world_size=16）要加什么

把 `WORLD_SIZE` 改成 16、用 2 台机器，其余不变。前提：

- 2 台机器在**同一个 Ray 集群**里（→ [L07](07-multinode.md)）
- 一个 PG 一次要到 16 个 bundle（→ [L03](03-placement-group.md) 的 gang 语义）
- `MASTER_ADDR` 由 rank 0 报出，**跨机时这个 IP 必须是高速网 IP**

> ⚠️ **跨机 NCCL 最典型的失败是"挂住"**：连上了但握手不了，`init_process_group` 永远不返回。
> Lux 为此专门有 `RefitAbortWatchdog`——因为**这种挂死不会自己超时**，
> 必须有个"进程内的线程"在超时后动手 abort（[L05](05-sync-vs-async.md) §⑤ 解释了为什么必须是线程而不是 actor）。

## ⑥ 为什么 Lux 关心这个

### ① 这就是 slime `actor_group.py:116-129` 的全文

`Lux/docs/research/dimensions/orchestration/README.md` 引的原文：

```python
for rank in range(world_size):
    actor = TrainRayActor.options(..., scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=reordered_bundle_indices[rank],
    )).remote(world_size, rank, master_addr, master_port)
    if rank == 0:
        master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
```

对比你刚写的代码——**结构完全一样**。唯一区别是多了一行 `placement_group_bundle_index`（→ L03）。

⇒ **`RayActor` 那个 10 行基类，唯一用途就是暴露这个地址**（`ray_actor.py` 全文 10 行）。

### ② 为什么"rank 0 先起"是必须的

进程组需要一个**所有 rank 都能连上的汇合点**，而：

- 端口不能写死（会撞车）⇒ 必须**动态取一个空闲端口**
- 地址不能写死（多机时 rank 0 在哪台机器由 Ray 调度决定）⇒ 必须**问 rank 0 自己**

⇒ 于是有了"**rank 0 先起 → 报地址 → 其余加入**"的两段式。
torchrun 把这件事藏在命令行参数里；在 Ray 上你得自己写这 10 行。

### ③ Lux 的具体形状

| 项 | 值 |
| --- | --- |
| world_size | **16**（2 个训练节点 × 8 卡） |
| 每个 rank | 一个 `TrainActor`，`num_gpus=1` |
| 后端 | `nccl` |
| 落点 | 单 PG 的 16 个 bundle，**重排序后**绑定 |
| 谁来做 | `LuxDriver`（`orchestrate/driver.py`） |

## ⑦ 一句话总结

> **手工 rendezvous = 「rank 0 先起 → 报出 IP 与空闲端口 → 其余 rank 带着它加入」+
> 「所有 rank 的 `init_process_group` 必须同时在跑」。**
>
> 第二句是坑：**集合操作不能用顺序 `ray.get` 提交。** 而这个坑的症状是**静默挂死**。
