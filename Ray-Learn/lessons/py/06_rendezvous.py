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
    而 rank 0 还要先回答「我的地址和端口是什么」—— 顺序错了就死锁。
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
    #    ❌ 错误写法：for w in workers: ray.get(w.setup.remote())  -> 死锁
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
