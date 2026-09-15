# Ray-Learn

> 面向 **Lux 项目的编排层**学 Ray —— 只学那条切片，不是学完整个 Ray。

## 为什么有这个目录

Lux 的编排层（`lux/orchestrate/`）预算 800–1,500 行，参考实现是 slime 的 **1,228 行**（就在 `Lux/third_party/slime/slime/ray/`，可以整个通读）。

但 Ray 的**坑集中在少数几个语义上**，而且 Lux 已经踩过其中几个的文档证据：

| 坑 | 后果 | Lux 里对应 |
| --- | --- | --- |
| `num_gpus` 是**账本**不是绑卡 | 引擎绕过 Ray 抢卡，Ray 会调度到已占用的节点 | S0 #17 |
| **同步 actor 阻塞会饿死同一 actor 的所有调用**；`max_concurrency` 也救不了（"a coroutine blocked in C never yields"） | 训练循环与 rollout 互相卡死 | `rollout` 为什么必须独立成 actor |
| 具名 actor 死了以后**错误长什么样** | 决定故障可见性怎么写 | `LuxDriver` / 故障恢复 |
| placement group 的 **bundle 顺序** | rank ↔ GPU 映射不稳定 ⇒ **正确性问题**，不是性能问题 | "必须保留的机制" |

这 7 课就是照着这张表设计的。

---

## 一、环境（uv，最小可跑通）

### 1. 装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 装完重开 shell，或：
source $HOME/.local/bin/env
uv --version
```

> 不需要 sudo。uv 装在 `~/.local/bin`，自己管 Python 版本（**不会动系统 Python**）。

### 2. 建项目

```bash
mkdir -p ~/Documents/MLsys-Note/Ray-Learn && cd $_
```

`pyproject.toml`（**本目录已放了一份同样的文件，直接 `uv sync` 即可**；以文件为准，下面只是展示内容）：

```toml
[project]
name = "ray-learn"
version = "0.1.0"
description = "Ray lessons scoped to the Lux orchestration layer"
requires-python = ">=3.11"
dependencies = [
    # [default] 带 dashboard —— 学 L02/L03 时"看见"资源与 bundle 比读日志快得多
    "ray[default]>=2.40",
]

[project.optional-dependencies]
# 只有 L06（torch 进程组）需要；前 5 课不用装 torch
dist = ["torch>=2.4"]
```

### 3. 同步环境

```bash
uv sync                      # 只装 ray，够 L01–L05 用
uv sync --extra dist         # 加 torch，给 L06 用
```

`uv sync` 会自己下载合适的 Python、建 `.venv/`、写出 `uv.lock`。**`uv.lock` 要提交**——它是这个学习环境可复现的凭据。

### 4. 验证

```bash
uv run python -c "import ray; ray.init(); print(ray.cluster_resources()); ray.shutdown()"
```

应看到（H100 上 `GPU` 会出现，且是 **8.0**）：

```
{'node:__internal_head__': 1.0, 'GPU': 8.0, 'CPU': 128.0, 'object_store_memory': ..., 'memory': ...}
```

> ⚠️ **若 uv 报 Python 版本不兼容**：Ray 的 wheel 不总是支持最新 Python。按报错把
> `requires-python` 降一档（如 `>=3.11,<3.13`），或直接 `uv python pin 3.12`。uv 会把冲突说清楚，
> 不要靠猜。

### 运行形态：本课程按 **H100 真机**写

| 形态 | 硬件 | 能做的课 |
| --- | --- | --- |
| **A. 单机多卡** | 1 台 × 8×H100 | L01–L06 **全部**，⭐ 且 L02 的**第二部分**是本课程最有价值的一节 |
| **B. 双机** | 2 台 × 8×H100 | + L07 的一半（跨节点放置、NCCL 跨机） |
| **C. 四机** | 4 台 × 8×H100 | + L07 全部（**= Lux S0 阶段 0.1**） |

> ⚠️ **不要跳过 A 直接上 C。** L02 那个"账本 vs 实际占用"的实验，**在一台机器上就能复现**
> Lux 最危险的静默失效；到 4 机上才发现，你会误以为是网络问题。

**单机上不需要 `ray start`**——`ray.init()` 会自动起本地集群并**自动探测到 8 张 GPU**。
⚠️ **只有 L02 Part B 与 L07 需要 `ray start`**，因为它们要控制 Ray **怎么记账**
（`--num-gpus=2` 这种"少报"是 `ray.init()` 做不到的）。

<details>
<summary>📎 想在笔记本（macOS，无 CUDA）上先过一遍概念</summary>

macOS 上 Ray **报告 0 张 GPU**，所有 `num_gpus` 申请都"永远成功"（账本是空的）。
此时把 `num_gpus=1` 换成**自定义资源**：

```python
ray.init(resources={"fake_gpu": 4})
@ray.remote(resources={"fake_gpu": 1})
```

**概念完全一样**——Ray 都只是记账，差别只在真 `num_gpus` 时它额外帮你设 `CUDA_VISIBLE_DEVICES`。

⚠️ 但 **L02 Part B / L06 / L07 在笔记本上没有意义**，不必强求。**L05（饿死实验）在笔记本上同样成立**
——它跟 GPU 无关，是本课程最重要的一课。

</details>

---

## 二、课程地图

| 课 | 学什么 | 关键 Ray API | 解锁 Lux 的哪一块 | 硬件 | 建议 |
| --- | --- | --- | --- | --- | --- |
| [L01](lessons/01-actor-model.md) | task vs actor、ObjectRef、有状态 | `@ray.remote` / `.remote()` / `ray.get` | 一切的基础 | 单卡即可 | 必做 |
| [L02](lessons/02-resource-ledger.md) | **资源账本** + ⭐⭐ **task 排队 vs actor 永久挂死**；Part B：账本 vs 实际占用 | `num_gpus` / `ray start --num-gpus` / `cluster_resources()` | S0 #17（GPU 记账 = 16 不是 32） | **1 台 × 8 卡** | ⭐⭐ 必做（三部分都做） |
| [L03](lessons/03-placement-group.md) | **PG 与 bundle**：gang scheduling、位置确定、重排序 | `placement_group` / `PlacementGroupSchedulingStrategy` | 训练 rank 的稳定性 | 1 台 × 8 卡 | ⭐ 必做 |
| [L04](lessons/04-named-actor.md) | 具名 actor、namespace、**死掉以后什么样** | `name=` / `namespace=` / `get_actor` / `ray.kill` | `LuxDriver`、故障恢复 | 单卡即可 | 必做 |
| [L05](lessons/05-sync-vs-async.md) | **饿死实验**：为什么阻塞一个 actor 会拖死全部 | `max_concurrency` / async actor | ⭐ **`rollout` 为什么必须独立** | **无需 GPU** | ⭐⭐ 最重要 |
| [L06](lessons/06-rendezvous.md) | 手工 rendezvous + **NCCL** 进程组 | `get_node_ip_address` + `init_process_group` + `NCCL_DEBUG` | 16 个 `TrainActor` 组成 PG | 1 台 × 8 卡 | 必做 |
| [L07](lessons/07-multinode.md) | 多机集群：`ray start`、`--num-gpus`、GLOO | `ray start --head/--address` / `ray status` | **S0 阶段 0.1** | **4 台** | 上机时做 |
| [L08](lessons/08-gpu-identity.md) | ⭐ **`ray.get_gpu_ids()`** + `InfoActor` 探测 + **bundle 重排序** | `ray.get_gpu_ids` / `PlacementGroup` 探测 | 「rank↔GPU 稳定」的**实现手段** | 1 台 × 8 卡 | ⭐ 必做（补 L03 的另一半） |
| [L09](lessons/09-env-and-lock.md) | ⭐ **环境变量注入**（3 条必须照抄）+ 24 行分布式锁 | `runtime_env` / `@ray.remote class Lock` | `config.py` 启动期；版本屏障 | 任意机器 | ⭐ 必做 |

**顺序不能换**：L02 是 L03 的前提，L04 是 L05 的前提，L06 是 L07 的前提，**L03 是 L08 的前提**。

> 📌 **L08/L09 是后补的**。初版只有 7 课，按 slime 编排层实际用到的 Ray API 对账后发现
> **漏了 `ray.get_gpu_ids()`（用了 3 次）、环境变量注入（一整个 `utils.py`）、分布式锁**——
> 而这三样恰恰是「rank↔GPU 稳定」和「启动期正确性」的实现手段。补课记录见 commit 历史。

⭐ **如果时间有限，只做三课**：**L02 Part A2 + Part B**（actor 超额永久挂死 + 复现最危险的静默失效）、
**L05**（解释 Lux 一个已定版架构决策）、**L08**（rank↔GPU 稳定的实现手段）。

### 每课的固定结构

```
① 一句话目标
② 先预测 —— 写下你的预期（这一步是学习发生的地方）
③ 完整代码（可直接 uv run）
④ 你应该观察到什么（含"如果不符合预期"的排查表）
⑤ 为什么 Lux 关心这个（指到具体文档/决策）
```

---

## 三、怎么跑一课

```bash
cd ~/Documents/MLsys-Note/Ray-Learn
uv run python lessons/py/02a_ledger.py
```

### ⚠️ 代码有两份，以 `py/` 为准

```
lessons/
├── 01-actor-model.md        ← 讲义：原理 / 预测 / 期望输出 / 为什么 Lux 关心
├── ...
└── py/
    ├── 01_actor_model.py    ← ⭐ 可运行脚本 = 唯一真相源
    ├── 02a_ledger.py
    ├── 02a_actor_trap.py
    └── ...
```

| | 作用 | 改了谁 |
| --- | --- | --- |
| `py/*.py` | **能直接跑的东西**，是本课的真相源 | 改代码**只改这里**，然后把改动回填到 md |
| `*.md` 里的代码块 | 与 `py/` **同步维护**的副本，让讲义能独立阅读 | 不要单独改——那会分叉 |

> ⚠️ **两份代码一定会漂移，这是这个结构的固有代价。**
> 约定是"改 `py/` → 回填 md"，而不是反过来。
> 如果你只想要一份：把 md 里的代码块删成关键片段 + 指向文件（这会更安全，但讲义不能独立读了）。

> ⚠️ **每个脚本结尾都调了 `ray.shutdown()`**。忘记 shutdown 会让下一个脚本连到上一个的集群，
> 出现"资源怎么少了"这类幻觉。

> ⚠️ **脚本可能挂住而不是报错**——这是 Ray 的常态（[L02](lessons/02-resource-ledger.md) A2 就是这个主题）。
> 卡住时 `Ctrl-C`，然后 `ray stop` 清干净再来。**不要把"没输出"当成"没问题"。**

### 实验记录建议

每课末尾留一行自己的结论，例如：

```
L02  实测：第 5 个 actor 排队 8.1s 后拿到资源，没有报错。账本是硬的。
```

**写下数字**——这是从"读过文档"到"知道"的分界线。

---

## 四、学完之后读什么

`Lux/third_party/slime/slime/ray/`——**1,228 行，整个可以读完**。按这个顺序：

| 顺序 | 文件 | 行数 | 读它学什么 | 对应本课 |
| ---: | --- | ---: | --- | --- |
| 1 | `ray_actor.py` | **10** | 编排的最小基类长什么样 | L06 |
| 2 | `utils.py` | 75 | ⭐ 环境变量注入 + **24 行的分布式锁** | L04 |
| 3 | `train_actor.py` | 126 | worker 契约只有 8 个方法 | L06 |
| 4 | `placement_group.py` | 253 | ⭐ 单 PG + offset 切分 + **bundle 重排序** | L03 |
| 5 | `actor_group.py` | 269 | ⭐ 手工 rendezvous 全文 | L06 |
| 6 | `rollout.py` | 495 | 最复杂的 actor，怎么和引擎交互 | L05 |

⚠️ `utils.py` 里有三条**必须照抄否则出事**的环境变量（Lux 的 `config.py` 要用）：

| 变量 | 值 | 原注释 |
| --- | --- | --- |
| `RAY_USE_UVLOOP` | `0` | "Ray's uvloop integration has caused **intermittent async actor issues**" |
| `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` | `1` | 阻止 Ray 改写 `CUDA_VISIBLE_DEVICES` |
| `NCCL_CUMEM_ENABLE` | `0` | 必须与 SGLang 一致，否则 NCCL 报错 |

---

## 五、不要学什么（省时间）

Ray 很大，但 Lux **一行都不用**：

```
❌ Ray Serve          ❌ Ray Tune        ❌ Ray Train
❌ Ray Data           ❌ RLlib           ❌ autoscaling / KubeRay
❌ Ray AIR 的任何部分
```

理由：ADR-0006 的"不做"清单已经把它们排除了。**编排层只做"拉起 + 排序 + 出错能看见"三件事。**

---

## 六、这几课怎么接回 Lux

学完 L01–L06，你就有能力写 `lux/orchestrate/` 的两块：

```
L01 + L04      →  LuxDriver：actor 图 + 启动顺序 + 版本屏障
L02 + L03 + L06 →  16 个 TrainActor 的资源与进程组
L05            →  rollout 与 trainer 的隔离边界（这是 Lux 的定版决策）
L07            →  S0 阶段 0.1（起 Ray），后面一切的先决条件
```

Lux 侧的对应文档：

| 主题 | 文档 |
| --- | --- |
| 6 种 actor 角色表 | `Lux/docs/design/control-plane-internals.md` |
| 编排机制（slime 七条） | `Lux/docs/research/dimensions/orchestration/README.md` |
| 起 Ray 与 GPU 账本断言 | `Lux/docs/runbooks/s0-verification.md` 阶段 0.1 |
| 为什么只有 `orchestrate/` 知道 Ray | `Lux/docs/design/project-layout.md` 约束 2 |
