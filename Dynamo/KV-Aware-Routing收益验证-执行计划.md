# Dynamo KV-Aware Routing 收益验证 — 项目执行计划

## Context（背景与目标）

内部 RL/推理栈使用 SGLang，现有 Infer-Router（SessionAware）。长期问题是"是否用 Dynamo KV-aware Router 替换内部路由层"。本项目是第一阶段：**在受控环境中验证 Dynamo KV-aware Routing 机制本身的可行性与收益上限**，不引入 veRL 和内部栈变量。调研报告（`/Users/zhushengguang/Documents/MLsys-Note/Dynamo/Dynamo KV Cache Router 技术调研.md`）已完成源码级机制分析，本计划负责实证。

**已对齐的关键决策**（访谈确认）：

| 决策项 | 结论 |
|---|---|
| 定位 | 受控环境验证机制本身，为后续对标 Infer-Router 的第二阶段铺路 |
| 部署模式 | **聚合部署（本次不做 PD 分离）**，裸机 + Docker/脚本 |
| 资源 | 8 机 × 8 卡 = 64 卡（H 系） |
| 模型 | GLM-4.5-Air（MoE 106B，FP8） |
| 分布式策略 | 双配置：配置 A（机制验证主力）TP4 × 每机 2 worker = 16 worker；配置 B（生产贴合）每机 1 worker TP8+EP8（可选 `--enable-dp-attention`）= 8 worker；不做跨机 TP/EP |
| KV 层级 | 两阶段：阶段一 GPU-only radix cache；阶段二 HiCache（L2 host，可选 Mooncake 作 L3）+ tier-aware routing |
| 对照组 | kv / round-robin / least-loaded / random（P0）+ kv 参数消融（P1）+ sgl-router cache_aware 外部参照（P2） |
| 压测 | AIPerf 合成多轮会话为主 + 仓库 `agent_benchmark.py` + 内部真实 Agent trace 回放 |
| 会话形态 | 峰值 16-32k tokens，8-15 轮，轮间工具耗时 1s/5s/30s 三档独立扫描 |
| 收益判定 | TTFT p50/p99 + prefix cache 命中率主指标；ITL/吞吐/错误率护栏（±5%）；≥2 个负载点上 kv 相对最强基线 TTFT p50 改善 ≥20% |
| 时间 | ~3 周，1 人主力 |

## 可复用的现成组件（已查证）

- **Worker+Router 启动模式**：`examples/backends/sglang/launch/agg_router.sh` — 每个 worker `python -m dynamo.sglang --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:555X"}' --page-size 16 --enable-metrics`，frontend `python -m dynamo.frontend --router-mode kv`。切对照组只改 `--router-mode {kv|round-robin|least-loaded|random}`。
- **多轮 Agent 压测**：`benchmarks/router/agent_benchmark.py` — 基于 aiperf，并发会话模式，trace 为 Mooncake JSONL 格式（`session_id` / `input_length` / `output_length` / `hash_ids` / `delay`），`--concurrency` 控并发会话数，`--delay` 覆写轮间间隔（即工具执行耗时）。
- **合成数据生成**：`benchmarks/prefix_data_generator/`（`uv pip install -e ./benchmarks`），支持 prefix 长度/树结构参数化；`benchmarks/router/prefix_ratio_benchmark.py` 可做前缀比例扫描的快速对齐（官方在 8×L40S 上有已发布对比结果 `benchmarks/router/results.png` 可参照）。
- **无 GPU 干跑**：`python -m dynamo.mocker` 模拟 worker，可在拿到 GPU 前把整条压测-指标链路调通。
- **控制面**：`docker compose -f dev/docker-compose.yml up -d` 起 etcd + NATS。
- **HiCache 文档**：`docs/backends/sglang/sglang-hicache.md` — tier-aware 路由要求 SGLang ≥ 0.5.11（推荐 ≥ 0.5.13 + Mooncake ≥ 0.3.11.post1，规避 `MemcpyWorkerPool` 崩溃坑）。
- **指标**：frontend `:8000/metrics` 的 `dynamo_component_router_*`（overlap/命中、queue、worker 负载 gauge，详见调研报告 §11）；aiperf 输出 TTFT/ITL/吞吐分位数。

## 实验环境

> 下文所有相对路径均指 Dynamo 仓库（GitHub: <https://github.com/ai-dynamo/dynamo>，本地克隆 `/Users/zhushengguang/CODES/dynamo`）。

- **软件版本钉死**（写入实验记录），各组件来源/出处：

| 组件 | 版本要求 | 来源 / 出处 |
|---|---|---|
| Dynamo SGLang runtime 镜像 | ≥1.3.0（内置 SGLang ≥0.5.15） | GitHub Releases: <https://github.com/ai-dynamo/dynamo/releases>；镜像标签与拉取方式见 `docs/backends/sglang/README.md`；需自建时用 `container/` 下 Dockerfile/构建脚本 |
| SGLang | tier-aware 事件 ≥0.5.11，推荐 ≥0.5.13 | 版本门槛出处：`docs/backends/sglang/sglang-hicache.md`；KV events 机制：`docs/backends/sglang/sglang-reference-guide.md` "Metrics and KV Events" 节 |
| Mooncake（仅阶段二 L3） | ≥0.3.11.post1 | 版本坑（`MemcpyWorkerPool` 崩溃）出处：`docs/backends/sglang/sglang-hicache.md`；项目主页 <https://github.com/kvcache-ai/Mooncake> |
| GLM-4.5-Air FP8 权重 | — | HF: <https://huggingface.co/zai-org/GLM-4.5-Air-FP8>（SGLang 对 GLM-4.5 系列的支持以 sgl-project/sglang 的模型支持列表为准，D3 冒烟即验证） |
| etcd + NATS 控制面 | 随 compose 文件 | `dev/docker-compose.yml`（用法见 `benchmarks/router/README.md` "Setting up etcd and NATS"） |
| AIPerf 压测器 | 最新 pip 版 | <https://github.com/ai-dynamo/aiperf>（`agent_benchmark.py` 内部即调用 aiperf） |
| 数据生成器 | 随仓库 | `benchmarks/prefix_data_generator/`，安装 `uv pip install -e ./benchmarks`（见 `benchmarks/router/README.md` Prerequisites） |
| Router 参数完整参考 | — | `docs/components/router/router-guide.md`、`docs/components/router/router-configuration.md`，或 `python -m dynamo.frontend --help`；打分公式与权重默认值见调研报告 §6 |
| Mooncake 公开 trace（可选对齐用） | — | `wget https://raw.githubusercontent.com/kvcache-ai/Mooncake/d21da178bae8db9651cf18a76824c084145fc725/mooncake_trace.jsonl`（出处：`benchmarks/router/README.md` Step 5） |
- **拓扑（配置 A，机制验证主力）**：每节点 8 卡 → 2 个 TP4 worker；8 节点 = **16 worker**。worker 均为单机内并行，无跨机通信变量；etcd/NATS/frontend 部署在 node0，各 worker 注册到同一控制面（`ETCD_ENDPOINTS` / `NATS_SERVER` 指向 node0）。
- **拓扑（配置 B，生产贴合验证）**：每节点 1 个 worker，TP8 + EP8（MoE 专家并行），可选 `--enable-dp-attention`（attention DP + MoE EP 的典型 MoE 生产配置）→ **8 worker**。仅在配置 A 中差异最大的负载点上重跑 kv vs 最强基线，回答"收益在生产并行策略下是否成立"。注意 DP-attention 下每 DP rank 有独立 KV 事件流（`docs/backends/sglang/sglang-reference-guide.md`），排障面更大。
- **明确不做**：跨机 TP/EP（worker 数掉到 4 以下、路由选择空间过小，且引入 RDMA 网络变量，对路由验证是噪声）。
- **前置验证**：GLM-4.5-Air 在目标 SGLang 版本上单 worker 正确性（对话可用、工具调用模板正确）+ KV events 正常发布（frontend 日志/metrics 中 indexer 有 block 记录）。

## 压测设计（把 KV-aware 收益验证出来的关键）

**流量结构**（合成 trace，Mooncake JSONL 格式，自写生成脚本或扩展 `prefix_data_generator`）：
- 每会话：system+工具定义 2-4k tokens（会话间**不共享**，各会话独立前缀根，模拟不同 agent 实例；另设一组共享 system 前缀的变体）；每轮新增 0.5-2k（工具返回），8-15 轮，峰值 16-32k；输出每轮 100-500 tokens。
- `hash_ids` 按 page-size 对齐生成，轮次间前缀严格嵌套（多轮天然前缀复用）。
- **扫描维度**（默认点 + 单维扫描，控制实验矩阵规模）：
  1. 轮间间隔（工具耗时）：1s / 5s / 30s —— 核心维度，间隔越长 cache 驱逐越频繁，直接检验 router 事件索引的价值与 HiCache 兜底价值；
  2. 并发会话数：从低到高 3-4 档，找到"轻载 / 适载 / 过载"≥2 个负载点；
  3. （P1）上下文规模 8k/32k 两档补充。
- **真实性验证**：导出内部 Agent trace（仅需 token 长度分布 + 轮次 + 间隔，不需要文本内容），转成同格式回放一组。

**每组实验的执行协议**：预热（填 cache）→ 正式测量（固定请求数）→ 采集 aiperf 结果 + `/metrics` 快照 → 重启 worker 清 cache → 下一组。同一 trace + 同一随机种子跨路由模式复用，保证公平。

## 指标与判定

- **主指标**：TTFT p50/p99；prefix cache 命中率（SGLang worker 侧 `cached_tokens` / router 侧 overlap 指标互相印证）。
- **护栏**：ITL/TPOT p50/p99、总吞吐（tokens/s）、错误/超时率——kv 组相对基线劣化不得超过 5%。
- **辅助观测**：router 每请求路由开销（ms 级指标）、各 worker 负载均衡度（吞吐方差）、KV event 应用统计。
- **判定**："有收益" = 在 ≥2 个负载点上，kv 相对最强基线（预计是 least-loaded）TTFT p50 改善 ≥20%，命中率显著提升，护栏不破。同时报告收益随"轮间间隔/并发"的变化曲线，指出收益区间边界。

## 时间计划（3 周）

**Week 1 — 环境与链路打通**
- D1-2：机器/镜像/权重就绪；mocker 干跑打通 trace 生成 → agent_benchmark → 指标采集全链路（无 GPU 依赖，可与资源申请并行）。
  - 控制面：仓库根目录 `docker compose -f dev/docker-compose.yml up -d`（参照 `benchmarks/router/README.md` "Setting up etcd and NATS"）。
  - trace 生成：`uv pip install -e ./benchmarks` 装数据生成器；JSONL 字段规范（`session_id`/`input_length`/`output_length`/`hash_ids`/`delay`）照抄 `benchmarks/router/README.md` "Trace Dataset Format" 节的示例。
  - mocker worker：`python -m dynamo.mocker --model-path <模型> --num-workers 16 --speedup-ratio 10`（用法见 `benchmarks/router/README.md` 的 mocker 小节）；frontend 起 `--router-mode kv`。
  - 压测：`python benchmarks/router/agent_benchmark.py --input-dataset trace.jsonl --concurrency 10 --delay 1000`；连通性用 `benchmarks/router/ping.sh` 验证。
  - 指标链路：`curl :8000/metrics | grep dynamo_component_router` 有值即通。
- D3-4：GLM-4.5-Air 单 worker（TP4）起服 + 正确性冒烟；16 worker（配置 A）+ kv router 跑通，确认 KV events/命中指标非零。
  - 启动脚本以 `examples/backends/sglang/launch/agg_router.sh` 为模板改写：MODEL 换成 GLM-4.5-Air FP8、`--tp 4`、每 worker 独立 `--kv-events-config` ZMQ 端口（5557+i）、独立 `DYN_SYSTEM_PORT`、`CUDA_VISIBLE_DEVICES` 按 4 卡一组划分；保留 `--page-size 16 --enable-metrics --disable-piecewise-cuda-graph`。
  - 跨机：每台机器的 worker 进程设 `ETCD_ENDPOINTS`/`NATS_SERVER` 指向 node0（参照 `benchmarks/router/README.md` Step 2 的 `NATS_SERVER` 用法）。
  - KV events 是否发布成功的判断依据见 `docs/backends/sglang/sglang-reference-guide.md` "KV Events" 节。
- D5：round-robin/least-loaded/random 各冒烟一轮；固化一键切换与结果采集脚本。
  - 切换只改 frontend 的 `--router-mode`（worker 不动）；全部可用模式与参数见 `docs/components/router/router-guide.md` 或 `python -m dynamo.frontend --help`。
  - 结果采集脚本固定产出三件套：aiperf 结果 JSON + `/metrics` 快照 + 启动参数记录（见"验证方式"第 3 条）。
- **里程碑 M1**：四种 router mode 下同一 trace 各完成一次小规模压测，数据管道产出可比对的结果表。

**Week 2 — 阶段一主实验（GPU-only）**
- D6-8：P0 矩阵：4 路由模式 × 3 档轮间间隔 × 3-4 档并发（默认上下文 16-32k）。
  - 轮间间隔用 `agent_benchmark.py --delay {1000|5000|30000}` 覆写，并发用 `--concurrency` 控；每组执行协议见"压测设计"节（预热→测量→采集→重启清 cache）。
  - 每组 ≥300 请求、固定种子重复 2 次；4×3×4 ≈ 48 组，按每组 20-30 分钟排期，优先跑完 kv 与最强基线的全矩阵。
- D9：P1 消融：`--no-kv-events`（approx 模式）、overlap_score_credit 权重、argmin vs softmax 采样温度。
  - `--no-kv-events` 的现成用法即 `agg_router.sh --approx` 分支（frontend 加 `--no-kv-events`，worker 去掉 `--kv-events-config`）。
  - overlap 权重、采样温度等 flag 名与默认值：`docs/components/router/router-guide.md`、`docs/components/router/router-configuration.md`；语义与默认值表见调研报告 §6.4（权重默认值）、§6.7（argmin 与 softmax 采样）。
  - 只在"kv 与基线差异最大的那个负载点"上做消融，控制组数。
- D10：内部真实 trace 回放对照组。
  - 内部 trace（长度/轮次/间隔元数据）转成 `benchmarks/router/README.md` "Trace Dataset Format" 的 JSONL 后直接喂 `agent_benchmark.py`；若需要缩放/加密前缀树，`real_data_benchmark.py` 的合成参数（`--prefix-len-multiplier`/`--prefix-root-multiplier`/`--speedup-ratio` 等）可复用。
- **里程碑 M2**：阶段一收益结论初稿（含判定标准打分）。

**Week 3 — 阶段二 + 生产贴合配置 + 报告**
- D11-12：HiCache 开启（L2 host；时间允许加 Mooncake L3），重跑 P0 中差异最大的负载点，量化 tier-aware 增量收益（重点看 30s 间隔档）。
  - 启用 flags、host 内存配比、Mooncake L3 配置示例、tier 事件行为全部以 `docs/backends/sglang/sglang-hicache.md` 为准（该文档同时是版本门槛的出处）；tier credit 语义见调研报告 §5.6、§6。
  - 验收点：worker 日志出现 `store(CPU_PINNED)` 类 tier 事件、router 指标中出现非 GPU tier 的 overlap。
- D13-14：配置 B（TP8+EP8，可选 DP-attention，8 worker）：在差异最大的 2 个负载点重跑 kv vs 最强基线，验证收益在生产并行下不消失。（原 P2 sgl-router 外部参照与此二选一，优先配置 B——对替换决策信息量更大。）
  - worker 参数改为 `--tp 8 --ep-size 8`（可选 `--enable-dp-attention`），flag 语义以 SGLang server args 文档为准（<https://docs.sglang.ai>）；Dynamo 侧 DP-attention 的多 DP rank KV 事件流说明见 `docs/backends/sglang/sglang-reference-guide.md` "KV Events" 节。
  - 若 DP-attention 事件流异常，按风险表回退到纯 TP8+EP8。
- D15：报告：结论、收益曲线、判定表、对第二阶段（对标 Infer-Router / PD 分离 / veRL 场景）的建议。
  - 结果叙事可对照官方已发布对比（`benchmarks/router/README.md` "Benchmarking Results"，DeepSeek-R1-Distill-8B / 8×L40S / ISL 14000）说明本实验与其差异点（模型 MoE、多机、多轮会话形态）。
- **里程碑 M3**：终版报告评审。

## 风险与困难点

| 风险 | 影响 | 缓解 |
|---|---|---|
| GLM-4.5-Air 在 Dynamo 镜像内的 SGLang 版本上有兼容性问题（MoE/FP8/chat template） | 阻塞主实验 | Week1 D3 就冒烟；备选降级 GLM-4-32B/9B dense（结论外推性下降需注明） |
| 多机 etcd/NATS/ZMQ KV events 跨机连通问题（防火墙、网卡选择） | 环境搭建超期 | mocker 先行验证控制面；worker 事件端口显式指定；预留 D3-4 排障缓冲 |
| KV 收益被"负载不均衡副作用"掩盖：kv 模式高命中但热点 worker 过载，重载点 TTFT 反而劣化 | 结论被误读 | 这本身是有效发现——按负载点分段报告；用 kv 参数消融定位权重敏感性 |
| MoE 模型 prefill 相对便宜（激活参数 12B），TTFT 收益绝对值低于 dense 直觉 | 收益不达 20% 阈值 | 上下文 16-32k 已放大 prefill 成本；如仍不显著，报告中给出收益-上下文长度关系而非单点结论 |
| aiperf 多轮会话模式与自定义 trace 的适配 bug | 压测数据失真 | 用 mocker + 小 trace 先验证"同 session 顺序、delay 生效、命中率符合预期"再上真机 |
| Mooncake/HiCache 版本坑（`MemcpyWorkerPool` 崩溃） | 阶段二受阻 | 钉死 SGLang ≥0.5.13 + Mooncake ≥0.3.11.post1；阶段二本身定位为增量验证，可裁剪不影响主结论 |
| 内部 trace 导出需要审批/脱敏 | P0 之外项延期 | 只导出长度/轮次/间隔元数据；提前发起流程，晚到则挪入第二阶段 |
| 配置 B（DP-attention/EP）多 DP rank KV 事件流的发布/消费异常，或 EP 下性能特征与 TP4 差异大导致两配置结论不一致 | Week 3 排障超期 / 结论需要分层表述 | 配置 B 只跑关键负载点、定位为"生产贴合性验证"而非主结论；若 DP-attention 事件流有问题，退回 TP8+EP8（无 DP-attention）仍可回答主要问题 |
| 16 worker 下单 worker KV 池小（TP4 显存均摊），高并发时驱逐加剧、命中率整体走低 | kv 与基线差距形态变化 | 这是真实权衡（worker 数 vs 单 worker cache 容量），配置 A/B 对比本身即回答此问题；报告中按"每 worker 并发会话数"归一化解读 |

## 验证方式（计划自身的检查点）

1. M1 检查：`curl :8000/metrics | grep dynamo_component_router` 在 kv 模式下命中/overlap 指标随多轮请求单调上升；round-robin 模式下同一 trace 命中率明显更低——证明实验能"分辨"路由策略。
2. 对齐官方数据：用 `prefix_ratio_benchmark.py` 在本环境跑一组 prefix-ratio 扫描，趋势应与 `benchmarks/router/results.png`（kv 优于 round-robin 且随 prefix ratio 增大差距拉大）一致，作为环境正确性的外部锚点。
3. 每组实验产出物固定：aiperf 结果 JSON + metrics 快照 + 启动参数记录，全部入库（git 或表格），保证可复现。
