"""L02 Part B · 账本 vs 实际占用。

⚠️ 用 task 而不是 actor：task 跑完就释放，所以 --num-gpus=4 那一轮不会挂住。
⚠️ 字节数：dtype=torch.uint8 时【1 元素 = 1 字节】，所以 gib * 1024**3 就是 gib GiB。
   初版写成 gib * 1024**3 // 4，实际只占了 1/4，实验会「看起来通过」而其实没验到东西。
"""
import os
import sys
import ray
import torch


@ray.remote(num_gpus=1, num_cpus=1)
def probe(rank: int, gib: int = 20) -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    try:
        n_bytes = gib * 1024 ** 3                 # uint8 -> 1 字节/元素
        _x = torch.zeros(n_bytes, dtype=torch.uint8, device="cuda:0")
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"rank={rank} cvd={cvd} ✅ 占到 {gib}GiB（卡共 {total:.0f}GiB）"
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
    else:
        print("👉 没有一个落到 4–7 上 = 账本与物理现实一致。")


if __name__ == "__main__":
    main()
