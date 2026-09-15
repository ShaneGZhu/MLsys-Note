# ThunderAgent 源码 Code Walk Through

> 本文重点是当前仓库 `/Users/zhushengguang/CODES/ThunderAgent` 的源码阅读。
>
> 论文：`ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System`，arXiv:2602.13692。
>
> 原始调研报告：同目录的 `调研报告.md`。
>
> 阅读约定：源码路径使用 `文件:行号`；公式全部使用独立的 `$$ ... $$` 块；“未实现”表示当前核心包中没有对应代码，不代表论文设计没有价值。

## 0. 先给结论：ThunderAgent 到底做了什么

ThunderAgent 不是模型服务本身，也不是 Agent 工具执行器。它位于 Agent client 和 vLLM/SGLang/SkyRL 之间，主要做五件事：

1. 把来自同一个 Agent workflow 的多轮请求识别为同一个 `Program`。
2. 把每个 Program 当前的上下文 token、执行阶段和后端绑定记录下来。
3. 在多个推理后端之间选择容量合适的节点。
4. 当 KV-cache 容量不足时，暂停 Program，放入全局等待队列；容量恢复后再恢复。
5. 透传 OpenAI-compatible 请求，同时解析 streaming response、usage、cached tokens 和 profiling 数据。

用源码模块表示：

```text
ThunderAgent/app.py
    FastAPI route: POST /v1/chat/completions
        |
        v
ThunderAgent/scheduler/router.py
    MultiBackendRouter
        |-- get_or_create_program()
        |-- update_program_before_request()
        |-- proxy_request()
        |-- update_program_after_request()
        |-- scheduler loop
        |
        +--> ThunderAgent/program/state.py
        |       ProgramStatus / ProgramState / Program
        |
        +--> ThunderAgent/backend/state.py
        |       BackendState: tokens, capacity, programs
        |
        +--> ThunderAgent/backend/*_metrics.py
        |       vLLM / SGLang / SkyRL metrics adapter
        |
        +--> ThunderAgent/profile/state.py
                prefill / decode / tool-call / pause timing
```

### 0.1 实现状态

| 标记 | 含义 |
|---|---|
| **核心实现** | `ThunderAgent/` 中有可直接执行的代码 |
| **示例集成** | `examples/` 中有用法，但不是核心包实现 |
| **论文机制，当前未实现** | 论文或报告提出了机制，当前核心包找不到相应模块 |
| **分析公式** | 用来解释设计取舍，源码不一定逐式计算 |

全文按一次真实请求的执行顺序展开。论文公式放在对应代码之后，回答“为什么这样写”，而不是先讲一遍与源码脱节的论文。

## 1. 运行入口：从 CLI 到 FastAPI

### 1.1 `python -m ThunderAgent`

文件：`ThunderAgent/__main__.py:6-32`

CLI 入口使用 `argparse` 创建配置，主要参数包括：

```python
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8300)
parser.add_argument("--backends", default="http://localhost:8000")
parser.add_argument("--router", default="tr", choices=["default", "tr"])
parser.add_argument("--backend-type", choices=["vllm", "sglang", "skyrl"])
```

启动之后，应用会创建 `MultiBackendRouter`，注册 FastAPI 路由，并在应用生命周期中调用 Router 的 `start()` / `stop()`。

### 1.2 两种 Router mode

`ThunderAgent/config.py:6-29`：

```python
@dataclass
class Config:
    backends: List[str] = field(
        default_factory=lambda: ["http://localhost:8000"]
    )
    router_mode: str = "tr"
    backend_type: str = "vllm"
    metrics_enabled: bool = False
    scheduler_interval: float = 5.0
```

- `router_mode="default"`：纯代理，不做容量调度。
- `router_mode="tr"`：默认模式，启用 Program 状态、容量检查、暂停和恢复。

这个开关很重要。后文的 `update_program_before_request()` 不是无条件执行同样的调度逻辑，而是会根据 mode 决定是否做容量相关处理。

## 2. Code Walk Through 总览：一次请求如何走完

假设客户端发送：

```python
client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=messages,
    stream=True,
    extra_body={"program_id": "task-001"},
)
```

进入 `POST /v1/chat/completions` 后，主路径可以压缩成：

```text
app.chat_completions()
  1. request.json()
  2. get_program_id(payload, headers)
  3. router.get_or_create_program(program_id)
  4. profile.on_request_arrive()
  5. router.update_program_before_request(...)
  6. router.proxy_request(...)
       -> forward_streaming_request() / forward_non_streaming_request()
  7. on_usage(...)
  8. router.update_program_after_request(...)
  9. return StreamingResponse / Response
```

后台同时运行另一条路径：

```text
router.start()
  -> _scheduler_loop()
       -> _scheduled_check()
            1. backend.fetch_metrics()
            2. _greedy_resume()
            3. remaining_capacity() < 0
               -> _pause_until_safe(backend)
```

前台请求路径负责“这个请求能不能发”；后台调度路径负责“整体容量不安全时暂停谁、之后恢复谁”。两者通过 `Program`、`BackendState` 和 `global_waiting_queue` 共享状态。

## 3. 第一步：FastAPI 入口和 Program ID

### 3.1 入口函数

文件：`ThunderAgent/app.py:59-117`

核心代码结构如下：

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = get_program_id(payload, request.headers)
    program_state = ta_router.get_or_create_program(program_id)

    if program_state.profile:
        program_state.profile.on_request_arrive()

    # 后面进入请求前调度和后端转发
```

这里已经能看出三个设计决定：

1. ThunderAgent 接收的是普通 JSON HTTP 请求，不要求客户端使用专用 SDK。
2. Program 是由请求中的 ID 关联出来的，不是由模型服务自动推断。
3. Profile 的到达时间在暂停检查之前记录，否则工具等待时间会被漏算。

### 3.2 `get_program_id()` 的优先级

文件：`ThunderAgent/app.py:15-37`

```python
def get_program_id(payload, headers=None) -> str:
    if "program_id" in payload:
        return str(payload["program_id"])

    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "program_id" in extra_body:
        return str(extra_body["program_id"])

    # headers 中查找 X-Session-ID，大小写不敏感
    # 找不到时返回 "default"
```

完整优先级是：

1. `payload["program_id"]`；
2. `payload["extra_body"]["program_id"]`；
3. `X-Session-ID`；
4. `"default"`。

### 3.3 这一步和论文公式的关系

论文公式（1）把 Program 写成：

$$
P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle
$$

这里的 `ID` 并不是抽象概念，而是从 HTTP 请求中真正取出来的字符串。没有这一步，后续所有“跨多轮请求”的调度都退化成请求级代理。

**源码对应：** `app.py:15-37`。

**实现边界：** 如果不同任务都没有传 ID，它们会共享默认值 `default`。这会让多个独立 workflow 共用一个 Program 状态，所以客户端接入时必须稳定提供 session/program ID。

## 4. 第二步：创建和复用 Program

### 4.1 `Program` 数据结构

文件：`ThunderAgent/program/state.py:11-48`

```python
class ProgramStatus(Enum):
    REASONING = "reasoning"
    ACTING = "acting"

class ProgramState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"

@dataclass
class Program:
    program_id: str
    backend_url: Optional[str] = None
    origin_backend: Optional[str] = None
    status: ProgramStatus = ProgramStatus.ACTING
    state: ProgramState = ProgramState.ACTIVE
    context_len: int = 0
    total_tokens: int = 0
    step_count: int = 0
    profile: Optional["ProfileState"] = None
```

这段代码要分成两个轴理解：

- `status` 是执行阶段：当前是模型推理还是工具/等待阶段。
- `state` 是调度生命周期：当前是否被暂停、是否已经结束。

因此 `ACTING + PAUSED` 是合法组合：Program 正处于工具阶段，同时被调度器从后端摘下，进入等待队列。

### 4.2 论文六元组的字段映射

| 论文字段 | 当前字段/逻辑 | 说明 |
|---|---|---|
| `ID` | `program_id` | `get_program_id()` 关联多轮请求 |
| `c` | `context_len`、`total_tokens` | 运行时用 token 近似 KV-cache 占用 |
| `T` | 无 | 当前核心 `Program` 没有工具环境集合 |
| `L` | `backend_url`、`origin_backend` | 当前后端和暂停前后端 |
| `τ` | `ProgramStatus` | Reasoning/Acting |
| `s` | `ProgramState` | Active/Paused/Terminated |

### 4.3 `get_or_create_program()`

文件：`ThunderAgent/scheduler/router.py:305-328`

调用点是 `app.chat_completions()` 中的：

```python
program_state = ta_router.get_or_create_program(program_id)
```

函数的设计要点：

1. 只在 `program_id` 不存在时创建对象。
2. Profile 开启时同时创建 `ProfileState`。
3. 新 Program 不在这里立即决定后端。
4. 后端绑定延迟到 `update_program_before_request()`。

“创建状态”和“分配后端”分开，是为了让后端选择可以看到当前请求的 token 估计、健康状态和已有负载。

## 5. 第三步：请求前调度

### 5.1 `update_program_before_request()` 的职责

文件：`ThunderAgent/scheduler/router.py:371-459`

函数签名：

```python
async def update_program_before_request(
    self,
    program_id: str,
    state: Program,
    payload: Dict[str, Any],
) -> bool:
```

它不是简单的“更新一个计数器”，而是数据面请求进入后端之前的闸门。主要步骤如下：

```text
1. weight sync 未完成？等待 _weight_sync_event
2. step_count += 1
3. 估计当前请求的 prompt/context token
4. Program 已暂停？等待它被恢复
5. 新 Program？选择后端
6. 后端容量不足？等待或触发调度
7. 允许 proxy_request() 继续
```

代码中首先处理权重同步屏障：

```python
if not self._weight_sync_event.is_set():
    await self._weight_sync_event.wait()

state.step_count += 1
```

这保证 RL 权重更新期间，请求不会继续进入可能正在变动的推理后端。

### 5.2 新 Program 的后端选择

相关函数：

- `router.py:244-276`：默认后端选择；
- `router.py:330-363`：带容量和健康状态的选择；
- `router.py:364-370`：按 Program 取回已绑定后端。

总体策略：

- 如果 Program 已绑定后端，优先使用原绑定；
- 如果是新 Program，在健康后端中选择负载较低且可容纳请求的后端；
- 如果没有可用容量，Program 不会无条件发送，而是进入等待/调度路径。

### 5.3 为什么请求前要等，而不是直接返回 503

Agent workflow 的下一轮请求通常依赖上一轮工具结果。如果暂时没有 KV-cache 容量，立即返回错误会把调度问题暴露给上层 Agent；等待队列则允许 ThunderAgent 在后端容量恢复后继续同一个 Program。

当前等待机制涉及：

- `Program.waiting_event`；
- `ProgramState.PAUSED`；
- `global_waiting_queue`；
- `_wait_for_resume()`：`router.py:934-957`。

## 6. 第四步：HTTP 转发和 streaming response

### 6.1 Router 只管理状态，不解析所有 SSE 细节

`MultiBackendRouter.proxy_request()` 位于 `ThunderAgent/scheduler/router.py:1006-1045`，负责选择当前后端并调用 request processor；具体的 HTTP 和 SSE 处理在：

`ThunderAgent/scheduler/vllm_request_processor.py`

这样拆分后：

- Router 关心“发给哪个后端、发完如何更新 Program”；
- request processor 关心“如何把 HTTP 请求转发出去、如何拆 SSE、如何取 usage”。

### 6.2 usage 解析

文件：`vllm_request_processor.py:16-73`

```python
def extract_usage_info(payload):
    if not isinstance(payload, dict):
        return None, None, None, None

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None, None, None

    # 读取 total_tokens / prompt_tokens / completion_tokens
    # 从 prompt_tokens_details 读取 cached_tokens
```

返回值固定为：

```text
(total_tokens, prompt_tokens, completion_tokens, cached_tokens)
```

这四个数分别被后续逻辑用于：

- 更新 `Program.total_tokens`；
- 更新 context/prefill 估计；
- profiling decode/prefill；
- 估计 prefix cache 的共享 token。

### 6.3 streaming 转发

文件：`vllm_request_processor.py:101-221`

```python
async def forward_streaming_request(
    client,
    url,
    payload,
    *,
    on_usage=None,
    on_first_token=None,
    on_token=None,
    on_token_progress=None,
    token_progress_interval=DEFAULT_TOKEN_PROGRESS_INTERVAL,
) -> StreamingResponse:
```

函数在转发 SSE 的同时提供四类回调：

- `on_first_token`：记录 prefill 结束；
- `on_token`：记录 decode 过程；
- `on_token_progress`：增量更新 token；
- `on_usage`：stream 结束时提交最终 usage。

这就是为什么 profiling 不需要修改 vLLM：计时点通过 streaming callback 注入。

### 6.4 这一层的实现边界

核心代码：

- 支持 OpenAI-compatible chat completion；
- 支持 streaming/non-streaming；
- 能读 usage/cached tokens；
- 不实现模型推理；
- 不实现 tokenizer 级 KV block 操作；
- 不实现 Docker 工具调用。

## 7. 第五步：响应后更新 Program

### 7.1 `update_program_after_request()`

文件：`ThunderAgent/scheduler/router.py:460-513`

函数签名：

```python
def update_program_after_request(
    self,
    program_id: str,
    state: Program,
    total_tokens: int,
    prompt_tokens: int = 0,
) -> None:
```

函数文档已经明确写出它的第一个状态变化：

```text
Transitions to ACTING (off GPU, executing tool).
Updates token counts.
If marked for pause, pause immediately.
```

也就是说，模型响应结束后，Program 默认进入 Acting：下一步可能执行工具，也可能等待 Agent client 的下一次请求。下一个请求进入时，才再次进入 Reasoning 过程。

### 7.2 状态变化示意

```text
请求到达前：
    ProgramState.ACTIVE
    ProgramStatus.ACTING 或上一次状态

请求发往模型：
    ProgramStatus.REASONING
    backend_url = 某个后端

模型响应结束：
    ProgramStatus.ACTING
    total_tokens/context_len 更新
    profile.on_request_end(...)

若后台判断容量过载：
    ProgramState.PAUSED
    backend_url 解除
    进入 global_waiting_queue
```

### 7.3 公式如何落到这里

论文把上下文 token `c` 看作 KV-cache 规模：

$$
c_P \approx \text{Program } P \text{ 当前需要保留的上下文 token 数}
$$

源码不直接测量每个 Program 实际占用的 GPU bytes，而是把 vLLM usage 中的 token 统计写入 `total_tokens`，再由 `BackendState` 汇总。因此这里是 **token-level approximation**，不是精确显存测量。

## 8. 第六步：后台调度循环

### 8.1 `_scheduler_loop()`

文件：`ThunderAgent/scheduler/router.py:746-758`

```python
async def _scheduler_loop(self):
    while not self._scheduler_stop:
        try:
            await asyncio.sleep(self._scheduler_interval)
            if self._weight_sync_active:
                continue
            await self._scheduled_check()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Scheduler error: %s", e, exc_info=True)
```

这里有两个值得注意的工程行为：

1. 这是独立后台任务，不需要新请求到达才检查 KV-cache。
2. 权重同步期间跳过调度 tick，避免数据面和权重更新同时改变后端状态。

### 8.2 `_scheduled_check()` 顺序

文件：`router.py:759-771`

```python
async def _scheduled_check(self):
    for backend in self.backends.values():
        await backend.fetch_metrics()

    await self._greedy_resume()

    for backend in self.backends.values():
        if backend.cache_config and backend.remaining_capacity() < 0:
            await self._pause_until_safe(backend)
```

顺序不是论文文字中常见的“先暂停，再恢复”，而是当前实现的：

1. 拉取新指标；
2. 尝试恢复等待中的 Program；
3. 对仍然超容量的后端执行暂停。

**阅读源码时要以这个顺序为准。** 它解释了为什么恢复使用的容量可以是 decay-adjusted capacity，而暂停判断仍然使用原始剩余容量。

## 9. BackendState：如何把 KV-cache 转成容量

### 9.1 BackendState 的职责

文件：`ThunderAgent/backend/state.py:26-88`

```python
class BackendState:
    def __init__(
        self,
        url: str,
        tool_coefficient: float = DEFAULT_TOOL_COEFFICIENT,
        metrics_client: Optional[MetricsClient] = None,
        use_acting_token_decay: bool = False,
    ):
        self.url = url
        self.tool_coefficient = tool_coefficient
        self.use_acting_token_decay = use_acting_token_decay
        self.metrics_client = metrics_client or VLLMMetricsClient(url)
```

一个 `BackendState` 同时保存：

- 后端地址；
- 它当前登记的 Program；
- metrics client；
- cache capacity；
- shared token 估计；
- future paused token；
- Acting token 权重和衰减开关。

### 9.2 Reasoning 和 Acting 的 token 汇总

文件：`backend/state.py:93-107`

```python
@property
def reasoning_program_tokens(self) -> int:
    return sum(
        p.total_tokens
        for p in self._programs.values()
        if p.status == ProgramStatus.REASONING
    )

@property
def acting_program_tokens(self) -> int:
    return sum(
        p.total_tokens
        for p in self._programs.values()
        if p.status == ProgramStatus.ACTING
    )

@property
def active_program_tokens(self) -> int:
    return int(
        self.reasoning_program_tokens
        + self.tool_coefficient * self.acting_program_tokens
    )
```

论文中 `τ=R/A` 是调度优先级的抽象；这里它直接进入容量计算。`acting_token_weight` 越小，Acting Program 在容量模型中的占用越低；`1.0` 则表示与 Reasoning token 等权。

### 9.3 capacity check

文件：`backend/state.py:149-177`

```python
def has_capacity(self, extra_tokens=0, extra_count=0) -> bool:
    if not self.cache_config:
        return True

    tokens = self.active_program_tokens + extra_tokens
    count = self.active_program_count + extra_count
    buffer = count * BUFFER_PER_PROGRAM
    required = tokens - self.shared_tokens + buffer
    return required <= self.cache_config.total_tokens_capacity
```

可以把它写成代码真正使用的近似关系：

$$
\text{required}
= \text{active\_program\_tokens}
- \text{shared\_tokens}
+ \text{active\_program\_count}\times\text{BUFFER\_PER\_PROGRAM}
$$

如果 `required <= total_tokens_capacity`，认为可以容纳；如果没有 `cache_config`，代码保守地返回 `True`，也就是把“没有指标配置”视为“不做容量限制”。

这不是论文完整的 GPU memory model，而是一个可运行的容量代理模型。

### 9.4 capacity 来源

vLLM adapter 中：

文件：`ThunderAgent/backend/vllm_metrics.py:16-25`

```python
@dataclass
class VLLMCacheConfig:
    block_size: int = 0
    num_gpu_blocks: int = 0

    @property
    def total_tokens_capacity(self) -> int:
        return self.block_size * self.num_gpu_blocks
```

对应公式：

$$
C_{total}
= \text{block\_size}
\times \text{num\_gpu\_blocks}
$$

- `block_size`：一个 KV block 可以容纳的 token 数；
- `num_gpu_blocks`：后端可用 GPU block 总数；
- `C_total`：源码统一使用的 token capacity。

### 9.5 shared tokens

`BackendState.update_shared_tokens()` 会调用 metrics client 的 `calculate_shared_tokens()`，使用最新 vLLM prefix cache 信息估计已经共享的 token：

文件：`backend/state.py:129-136`

```python
def update_shared_tokens(self) -> None:
    self.shared_tokens = self.metrics_client.calculate_shared_tokens(
        self.reasoning_program_tokens
    )
```

这对应论文里的 caching cost / prefix reuse 动机，但当前代码只把它作为容量扣减项，没有计算完整的 `Cost_caching`。

## 10. Pause：逐行理解过载处理

### 10.1 触发点

`_scheduled_check()` 发现：

```python
if backend.cache_config and backend.remaining_capacity() < 0:
    await self._pause_until_safe(backend)
```

这里的 `remaining_capacity() < 0` 是实际触发条件。

调研报告使用高水位线表达释放目标：

$$
\Delta C
= \sum_{p\in\mathcal{L}} c_p
- \lambda_{max} C_{total}
$$

但当前代码没有独立的 `lambda_min` / `lambda_max` 配置，也没有把 `ΔC` 作为参数传入 Pause。它是反复调用 `remaining_capacity()`，直到不再为负。

### 10.2 `_pause_until_safe()`

文件：`ThunderAgent/scheduler/router.py:773-805`

源码意图可以直接读成：

```python
while backend.remaining_capacity() < 0:
    acting_programs = self._get_acting_programs_sorted(
        backend.url,
        ascending=True,
    )
    if acting_programs:
        program_id, state = acting_programs[0]
        self._pause_program(program_id, state)
        continue

    reasoning_programs = self._get_reasoning_programs_sorted(
        backend.url,
        ascending=True,
    )
    if reasoning_programs:
        program_id, state = reasoning_programs[0]
        self._pause_program(program_id, state)
        continue

    break
```

核心优先级：

1. 先暂停 Acting；
2. 每一类中按 `total_tokens` 升序，即 shortest first；
3. 每暂停一个就重新计算容量；
4. 没有可暂停 Program 时停止。

### 10.3 为什么 Acting 优先

Acting Program 当前不在执行模型 decode。暂停它通常意味着以后恢复时重新计算上下文，但不会立即打断正在生成 token 的 Reasoning Program。

论文中的成本解释是：如果必须释放 `ΔC`，选择上下文长度 `c_i` 较小的 Program，可以降低重新 prefill 的成本：

$$
\min_{S}\sum_{i\in S} c_i^2
\quad
\text{s.t.}
\quad
\sum_{i\in S} c_i \ge \Delta C
$$

源码没有求解这个优化问题，而是用“阶段优先 + token 升序”实现启发式近似。

## 11. Pause 状态转移和 Global Waiting Queue

### 11.1 `_pause_program()` 改变了什么

`router.py` 中的 Pause 逻辑会完成四类操作：

```text
Program.backend_url -> 解除当前绑定
Program.origin_backend -> 保存暂停前后端
Program.state -> PAUSED
BackendState._programs -> unregister
global_waiting_queue -> add
```

加入队列的元数据由 `_add_to_global_waiting_queue_sync()` 创建：

```python
PausedInfo(
    program_id=program_id,
    total_tokens=state.total_tokens,
    paused_at=time.time(),
    origin_backend=backend.url if backend else None,
    step_count=state.step_count,
)
```

位置：`ThunderAgent/scheduler/router.py:560-571`。

### 11.2 队列为什么是 global

如果每个后端只有自己的本地等待队列，A 节点暂停的 Program 只能回 A 节点；B 节点有空闲容量时也帮不上忙。当前结构：

```python
self.global_waiting_queue: Dict[str, PausedInfo] = {}
```

位置：`router.py:92-96`。

`_get_paused_programs_sorted()` 在 `router.py:578-591` 取出所有仍然存在的 Program，并按 `PausedInfo.total_tokens` 排序。

## 12. Restore：从队列到后端

### 12.1 `_resume_program()`

文件：`ThunderAgent/scheduler/router.py:692-744`

函数接收一个已经从全局池 claim 出来的 `Program`，以及可选的目标后端：

```python
def _resume_program(
    self,
    state: Program,
    target_backend: Optional[BackendState] = None,
) -> None:
    origin_backend = (
        self.backends.get(state.origin_backend)
        if state.origin_backend else None
    )
    backend = target_backend or origin_backend
```

这里的关键不是“恢复到原节点”，而是：

- 目标后端由 BFD placement 指定时，支持跨节点迁移；
- 没有新目标时，回退到 `origin_backend`；
- 恢复后重新注册到目标 BackendState；
- 设置 `backend_url`；
- 设置 `state=ACTIVE`；
- 唤醒等待该 Program 的 event。

对应论文 Restore 状态转移：

$$
P_{Paused}
= \langle ID,c,\mathcal{T},\varnothing,\tau,Paused\rangle
\longrightarrow
P_{Active}
= \langle ID,c,\mathcal{T},\mathcal{L}',\tau,Active\rangle
$$

源码只实现了 `ID/c/τ/state/backend` 这些字段；`T` 仍不存在。

### 12.2 `_greedy_resume()` 的实际算法

文件：`router.py:807-934`。

源码注释直接给出 Best Fit Decreasing（BFD）步骤：

```text
1. 计算所有健康后端的总容量
2. 选择累计 token 不超过总容量的 Program 集合
3. 将候选按 token 降序排列
4. 将最大 Program 放入剩余容量最大的后端
5. 每次放置后重新排序后端
6. 无法继续放置时停止
```

重要实现细节：

- 恢复候选的容量可以使用 `remaining_capacity_with_decay()`；
- 只有健康并且剩余容量超过 `BUFFER_PER_PROGRAM` 的后端参与 placement；
- 具体放置顺序是大 Program 优先，这是 bin packing 的实现选择；
- 这和论文公式中的 Restore score 不是同一个排序过程。

论文报告给出的恢复分数是：

$$
S_{restore}(P)
= \frac{1}{c_P}
+ \mathbb{I}(\tau=\mathbf{R})
$$

它表达“Reasoning 阶段优先，短上下文优先”。当前代码保留了容量和 token 优先思想，但实际跨后端安排采用 BFD。

## 13. 时间衰减：只影响恢复容量

配置入口：`ThunderAgent/config.py:27-29`。

```python
acting_token_weight: float = 1.0
use_acting_token_decay: bool = False
```

`_greedy_resume()` 收集后端容量时：

```python
remaining = (
    backend.remaining_capacity_with_decay()
    if backend.use_acting_token_decay
    else backend.remaining_capacity()
)
```

代码注释说明该衰减用于恢复逻辑。不要把它描述为所有 token 或所有成本项都在衰减：

- Restore 计算可以使用衰减容量；
- Pause 的过载检查使用原始 `remaining_capacity()`；
- 公式中的 `Cost_recompute`、`Cost_unused` 等没有随此开关被计算。

报告中的直观表达是：

$$
C_{acting}(t) \propto 2^{-t}
$$

这里的 `t` 是等待/Acting 时间。它是恢复容量的乐观估计，不是源码里独立记录的完整物理显存曲线。

## 14. vLLM、SGLang、SkyRL：metrics adapter 是怎么接入的

### 14.1 抽象接口

文件：`ThunderAgent/backend/metrics_base.py:13-50`

`MetricsClient` 规定后端适配器需要提供：

- `healthy`；
- `cache_config`；
- `latest_metrics`；
- `metrics_history`；
- `metrics_url`；
- `start_monitoring()` / `stop_monitoring()` / `fetch_metrics()`。

`BackendState` 依赖这个接口，不依赖某一种后端的 Prometheus 或 JSON 格式。

### 14.2 vLLM

文件：`backend/vllm_metrics.py:16-25, 27-190`

vLLM 的静态容量由：

$$
C_{vLLM}
= \text{block\_size}
\times \text{num\_gpu\_blocks}
$$

得到。adapter 还会：

1. 请求 Prometheus 指标；
2. 解析 cache usage；
3. 维护 metrics history；
4. 计算 shared prefix tokens；
5. 提供 `healthy` 给 Router。

### 14.3 SGLang

文件：`backend/sglang_metrics.py:22-150`

SGLang 不直接复用 vLLM 的 `block_size * num_gpu_blocks` 结构，而是通过 server info 获取总 token capacity，再解析 `/metrics` 的运行请求、等待请求和 cache 统计。

### 14.4 SkyRL

文件：`backend/skyrl_metrics.py:1-20, 36-150`

SkyRL endpoint 返回 JSON，并可能聚合多个 vLLM engine。adapter 负责把：

```json
{
  "engines": [
    {
      "kv_cache_usage_pct": 5.9,
      "num_running_reqs": 2,
      "num_waiting_reqs": 0
    }
  ]
}
```

转换成和 `BackendState` 相同的 capacity/health/metrics 接口。

**源码重点：** 后端差异被封装在 adapter；Pause/Restore 不需要知道 Prometheus 文本还是 SkyRL JSON。

## 15. Profiling：源码如何观察一次请求的阶段

文件：`ThunderAgent/profile/state.py:15-280`

`StepMetrics` 记录：

```python
@dataclass
class StepMetrics:
    program_id: str
    step_id: int
    prefill_time: float = 0.0
    decode_time: float = 0.0
    pause_time: float = 0.0
    tool_call_time: float = 0.0
```

事件顺序：

```text
on_request_arrive()
    -> on_request_start()
        -> on_first_token()
            -> on_token()
                -> on_request_end(...)
```

`tool_call_time` 的定义是“上一次响应最后一个 token 到下一次请求到达”的时间。它可以近似观察 Acting 阶段，但不能证明核心包执行了工具。

Profile 输出：

- CSV：`Config.profile_dir`，默认 `/tmp/thunderagent_profiles`；
- `/profiles`：所有 Program；
- `/profiles/{program_id}`：单个 Program；
- `get_averages()`：聚合时延。

### 15.1 和 Cost Model 的关系

论文公式（2）：

$$
\mathrm{Cost}_x
= \int_0^{t_x} M_x(t)\,dt
$$

源码没有直接计算这个积分。它提供的是 `t_x` 的若干观测点，以及 token/cache 近似的 `M_x` 输入。要得到论文级 STP，需要额外把 profiling、后端显存或 KV block 使用率按时间采样后积分。

论文总成本分解：

$$
\mathrm{Cost}_{total}
\approx
\mathrm{Cost}_{decode}
+\mathrm{Cost}_{prefill}
+\mathrm{Cost}_{recompute}
+\mathrm{Cost}_{unused}
+\mathrm{Cost}_{caching}
$$

当前源码没有对应的 `Cost_total` 变量或优化器。这一节必须明确标记为 **分析模型，不是直接运行时代码**。

## 16. 管理 API：从源码看“可操作性”

文件：`ThunderAgent/app.py:119-235`

| 路由 | 代码作用 |
|---|---|
| `GET /programs` | 遍历 `ta_router.programs`，返回 backend、context、tokens、step、status、state |
| `POST /programs/release` | 调用 Router 释放 Program |
| `GET /health` | 返回后端健康和 Program 总量 |
| `GET /profiles` | 返回 profiling 汇总 |
| `GET /profiles/{program_id}` | 返回单个 Program profiling |
| `GET /v1/models` | 把 `/models` 转发给第一个后端 |
| `GET /metrics` | 返回 ThunderAgent 聚合的 backend 和 paused count |
| `POST /weight_sync/begin` | 进入权重同步屏障 |
| `POST /weight_sync/end` | 退出权重同步屏障 |

### 16.1 `/programs` 是最适合 Code Walk Through 的观察点

路由将一个 Program 序列化为：

```python
program_data = {
    "backend": state.backend_url,
    "context_len": state.context_len,
    "total_tokens": state.total_tokens,
    "step_count": state.step_count,
    "status": state.status.value,
    "state": state.state.value,
}
```

因此可以用一个最小实验观察状态机：

1. 第一次请求后 `step_count=1`、`status=acting`；
2. 下一次请求进入模型时，status 临时变为 reasoning；
3. 后端容量不足时，`state=paused`；
4. 恢复后 `state=active`，可能 `backend` 已迁移；
5. release 后进入 terminated/被清理。

## 17. Weight sync：RL 集成为什么需要它

`MultiBackendRouter` 在 `router.py:182-242` 维护权重同步状态；HTTP 路由在 `app.py:217-235`。

运行过程：

```text
POST /weight_sync/begin
    -> 设置 _weight_sync_active
    -> 清理/等待同步条件
    -> 数据面请求在 _weight_sync_event 上等待
    -> scheduler tick 跳过

POST /weight_sync/end
    -> 清除同步状态
    -> set event
    -> 请求和 scheduler 恢复
```

这不是 RL 训练器本身。ThunderAgent 只提供一个“推理路由和后台调度暂停”的协调屏障；reward、optimizer、rollout policy 和 checkpoint 更新都在 `examples/rl_training` 或外部系统中。

## 18. 论文机制中，当前源码没有实现的部分

### 18.1 工具资源集合 `T`

论文公式中的：

$$
P = \langle ID,c,\mathcal{T},\mathcal{L},\tau,s\rangle
$$

当前 `Program` 有 `ID/c/L/τ/s` 的对应字段，但没有 `T`。核心包没有工具环境集合、Docker ID 列表对象或资源句柄表。

### 18.2 生命周期 Hook 和引用计数

论文描述的工具资源流程是：

```text
Program terminated
    -> lifecycle hook
    -> decrement shared environment ref_count
    -> ref_count == 0 ? destroy environment : keep
```

当前代码能找到的是：

- `ProgramState.TERMINATED`；
- `release_program()`；
- `ProfileState.tool_call_time`；
- README 中客户端传 `docker_ids` 的示例。

找不到的是：

- `ref_count`；
- Docker client；
- sandbox manager；
- resource registry；
- environment destroy hook；
- async environment prefetch。

所以 `release_program()` 不能解释成“自动释放 Docker 资源”。它只负责 ThunderAgent 对 Program 和后端登记的清理。

### 18.3 完整高低水位线

报告给出的 Pause 目标：

$$
\Delta C
= \sum_{p\in\mathcal{L}}c_p
- \lambda_{max}C_{total}
$$

Restore 条件：

$$
\sum_{p\in\mathcal{L}}c_p
< \lambda_{min}C_{total}
$$

当前配置中没有 `lambda_min` 和 `lambda_max` 字段；实际触发是 `remaining_capacity() < 0`。因此报告中的高低水位线是论文策略解释，不是当前代码中的两段可配置阈值。

### 18.4 完整 Cost optimizer

论文的短上下文选择问题：

$$
\min_{S}\sum_{i\in S}c_i^2
\quad
\text{s.t.}
\quad
\sum_{i\in S}c_i\ge\Delta C
$$

源码没有求解器，只有：

- Acting 优先；
- token 升序暂停；
- Global queue；
- BFD 恢复。

这是一套启发式 scheduler，不是直接最小化上述目标函数的程序。

## 19. examples：从集成样例到论文实验复现

这一章不能只把 `examples/` 理解为 API 接入示例。仓库中的样例分为三种：直接复现论文图表、围绕论文机制做扩展实验，以及展示 ThunderAgent 可嵌入真实推理或 RL 流水线。判断实验是否成功，不能只看任务能否跑通，还要看吞吐量、KV cache 命中率、扩展效率等系统指标。

### 19.1 总览与复现等级

| Example | 工作负载 | 后端/训练框架 | 复现定位 | 主要验证指标 |
|---|---|---|---|---|
| mini-swe-agent | SWE-bench 软件工程 Agent | vLLM | 论文结果复现 | steps/min |
| OpenHands | SWE-bench 软件工程 Agent | vLLM | 论文相关的端到端复现实验 | steps/min |
| ToolOrchestra | HLE 多 Agent、多工具推理 | vLLM + 外部 API | 论文吞吐图直接复现 | steps/min、KV hit、GPU 利用率 |
| Harbor | 大规模 SWE trajectory 生成 | SGLang | 论文机制的规模化/多节点验证 | 吞吐、KV hit、横向扩展效率 |
| SkyRL | Mini-SWE-Agent 在线 RL rollout | SkyRL + vLLM | 可复现的 RL 对照实验 | rollout tokens/sec |
| slime/tau-bench | 工具调用 Agent 的 GRPO 训练 | slime + SGLang | 可运行的 RL 集成流水线 | reward、训练与 rollout 日志 |

“论文结果复现”表示目录明确提供了论文对应 workload 和绘图或复现实验入口；“规模化/多节点验证”表示验证同一调度机制，但配置和观察不一定逐项对应论文某张图；“集成流水线”只证明接口与生命周期可以贯通，不能单独证明性能优于 baseline。

### 19.2 所有实验共同验证的最小闭环

这些 example 都围绕同一条 Program 生命周期改造上游 Agent：

1. 每个独立任务、trial 或 sample 生成稳定且唯一的 `program_id`；
2. 同一任务的多轮模型调用始终携带该 ID；
3. ThunderAgent 据此复用 Program 状态、后端位置和 token 统计；
4. 任务结束、截断或失败时调用 `POST /programs/release`；
5. `tr` router 才会进一步执行容量检测、Pause、Global Waiting Queue 和跨 backend Restore。

因此，样例首先验证“上游框架能否正确表达 Program 边界”，然后才验证调度收益。若遗漏 release，旧 Program 会留在路由器状态中；若每轮生成不同 ID，ThunderAgent 会把一个多轮 Agent 错拆成多个程序，实验结果也失去意义。

### 19.3 mini-swe-agent：高并发 SWE-bench 推理复现

**实验目的。** `examples/inference/mini-swe-agent` 让大量 mini-swe-agent worker 并发解决 SWE-bench 实例。每个实例包含多轮“模型推理—修改代码—执行 Docker 工具—读取结果”，非常适合制造 Reasoning/Acting 交替和长短不一的工具等待。

**与论文的关系。** README 明确说明 `scripts/reproduce/` 中的脚本用于复现论文报告结果，因此它不是只有功能连通性的 demo。提供两个模型配置：

- `GLM-4.6-FP8`：`scripts/reproduce/reproduce_glm4.6.sh`；
- `Qwen3-235B-A22B`：`scripts/reproduce/reproduce_qwen3_235B.sh`。

参考硬件是 8×H100，任务是 SWE-bench；实际模型由 `src/minisweagent/config/extra/swebench.yaml` 的 `model.model_name` 决定。运行前还要设置模型缓存目录 `HF_HOME`。

**执行链路。** vLLM 承载模型，ThunderAgent 位于 mini-swe-agent 与 vLLM 之间；benchmark worker 调用 OpenAI-compatible API。集成改造位于 `src/minisweagent/run/benchmarks/swebench.py` 和 `src/minisweagent/models/vllm_model.py`：实例 ID 被写入 `extra_body.program_id`，实例结束后向 `/programs/release` 发请求。

**指标与预期结论。** 吞吐定义为稳定服务窗口内完成的 LLM call 数除以窗口时长，单位为 `steps/min`。它主要验证：

- 高并发多轮 SWE Agent 中，Program 级状态能否维持请求亲和性；
- Acting 阶段较长时，调度器能否减少 KV cache thrashing 和重复 prefill；
- Program release 是否能阻止已完成实例持续占用路由状态；
- 相同模型和 worker 数下，`tr` router 是否比普通 request routing 完成更多 step。

**结论边界。** steps/min 是系统吞吐，不等于 SWE-bench resolved rate。要论证调度不损害任务质量，还应同时比较成功率、失败率和相同样本集合；只运行单个 worker 也无法证明 Program-aware scheduler 在内存压力下的优势。

### 19.4 OpenHands：完整软件工程 Agent 的端到端复现

**实验目的。** `examples/inference/OpenHands` 将 OpenHands 的规划、代码编辑、沙箱执行和观察循环接入 ThunderAgent，并在 `princeton-nlp/SWE-bench_Lite` 上运行。相较 mini-swe-agent，它验证的是更重的 Agent runtime 和更复杂的工具生命周期，而不是另一套路由算法。

**复现配置。** 仓库提供 8×H100 上的两个入口：

```bash
bash examples/inference/OpenHands/scripts/reproduce/reproduce_glm4.6.sh
bash examples/inference/OpenHands/scripts/reproduce/reproduce_qwen3_235B.sh
```

对应模型为 GLM-4.6-FP8 和 Qwen3-235B-A22B-Instruct-2507。每个 OpenHands task 使用独立 `program_id`，模型调用完成整个 task 前保持 ID 不变，结束时显式 release。

**指标与预期能力。** README 同样以稳定窗口中的 `steps/min` 衡量吞吐。该实验预期验证：

- ThunderAgent 的 OpenAI-compatible 转发可承载真实 OpenHands 流程；
- 大模型、多轮长上下文和 Docker 工具等待混合时，Acting-aware Pause/Restore 是否提高 GPU 有效工作比例；
- 上游 Agent 异常或结束路径能否正确释放 Program；
- 调度收益是否跨 mini-swe-agent 和 OpenHands 两种软件工程 Agent 保持一致。

**复现限制。** OpenHands、SWE-bench 镜像、模型权重和 Docker 环境都不是 ThunderAgent 核心包的一部分。结果对 sandbox 启动速度、worker 数、镜像缓存和任务抽样敏感；它不能证明 ThunderAgent 实现了 Docker 资源引用计数或论文描述的工具环境回收器。

### 19.5 ToolOrchestra：外部工具延迟下的论文吞吐图复现

**实验目的。** `examples/inference/ToolOrchestra` 在 HLE（Humanity's Last Exam）上运行多 Agent 编排。GPU 0 的本地 Qwen3-8B Orchestrator 经 ThunderAgent/vLLM 生成工具决策，GPU 1 运行 FAISS Retriever；`search`、`enhance_reasoning` 和 `answer` 还会调用外部 expert API。外部 API 的不确定延迟会形成明显 Acting 空窗。

**与论文的关系。** README 明确称以下脚本复现论文的 ToolOrchestra（HLE）Qwen3-8B throughput plot：

```bash
./scripts/reproduce/reproduce_hle_qwen3_8b.sh
./scripts/reproduce/plot_hle_qwen3_8b.py
```

实验在并发度 24、32、40、48 下分别运行三种方法：

| 方法 | Router | 含义 |
|---|---|---|
| baseline | ThunderAgent `default` | 标准 vLLM FCFS/request routing |
| continuum | `default` + vLLM-Continuum | TTL pinning 对照 |
| thunderagent | ThunderAgent `tr` | Program-aware capacity scheduling |

参考硬件是 2×RTX 5090，并依赖 OpenAI、Together AI、Tavily API 和检索索引。默认完整扫描为 12 次运行，每次最长约 2.5 小时。

**输出与指标。** 主指标是 `steps/min`；同时生成 `prefix_cache_timeseries.csv`、`gpu_sm_util_timeseries.csv`、`window_summary.json`、`steps_summary.json` 和 `combined_summary.json`。这些文件允许联合检查 server cache hit ratio、request cached-token ratio、KV cache usage 和 GPU SM utilization。

**预期验证。** 这是观察 `Cost_caching` 与 `Cost_recompute` 权衡最直接的 example：工具等待期间保留所有 KV 会浪费显存，过早驱逐又会增加重算。实验应比较吞吐曲线是否随并发上升而更稳定，并结合 KV hit 与 GPU utilization 判断收益来自缓存/调度，而不是请求失败或外部 API 变快。

**结论边界。** 外部 API 延迟和配额会引入噪声，所以必须使用相同题集、并发度和稳定窗口，并记录失败调用。该实验验证的是本地 Orchestrator 的调度；外部 expert model 本身不经过 ThunderAgent，也不能用它评价不同 expert API 的模型质量。

### 19.6 Harbor：SGLang 上的大规模数据生成与横向扩展

**实验目的。** `examples/datagen/harbor` 使用 Harbor 驱动 OpenHands 批量生成 SWE trajectory，推理后端换成 SGLang。它覆盖单 backend 和多 backend 两种拓扑，重点不是单次任务成功率，而是长时间、高并发生成任务中的稳定吞吐和 Radix cache 保留。

**入口与环境。** 关键入口包括：

- `scripts/run/run-experiment.sh`：单 backend 对照；
- `scripts/run/run-multinode-experiment.sh`：多 backend/多节点实验；
- `scripts/run/launch-sglang.sh`：启动 SGLang；
- `scripts/run/launch-thunderagent.sh`：启动 router；
- `scripts/run/launch-worker.sh`：启动 Harbor worker；
- `scripts/analysis/compute_metrics.py` 和 `compute_metrics_multinode.py`：汇总指标；
- `scripts/analysis/plot_comparison.py`：绘制对比图。

README 给出的典型资源很重：每个 SGLang server 使用 8×H100 80GB，worker 侧建议至少 176 CPU cores、256GB RAM，并为 SWE-bench Docker images 准备约 1.5TB 存储。

**对照和预期观察。** 对照核心是 `default` request routing 与 `tr` Program-aware routing，并进一步比较单节点和双节点。README 报告的目标观察为：

- 高并发时 TR 吞吐提高约 2.08–2.48 倍；
- TR 的 KV hit rate 保持约 94%–97%，default 可从约 97% 降至 36%；
- 双节点 TR 相对单节点 TR 达到约 1.91 倍扩展。

这些数字应当作为特定硬件、模型、并发和数据配置下的复现目标，而不是对任意部署的保证。

**预期验证。** Harbor 重点验证 SGLang metrics adapter、共享 Radix prefix 下的 token capacity 建模、Program affinity、全局等待队列和跨 backend Restore。双节点接近线性扩展还能检验：Program 暂停且旧 KV 已失效后，调度器是否可以安全地把它恢复到其他 backend，从而利用原本闲置的显存。

**结论边界。** Harbor 的 Docker image 管理、benchmark adapters、trajectory 格式和 worker orchestration 来自 Harbor/OpenHands 集成层。实验可观察任务结束后 Program 状态被 release，但不能据此声称核心 `ThunderAgent/` 已实现论文中的容器垃圾回收、工具引用计数或异步环境预热。

### 19.7 SkyRL：训练过程中比较 default 与 TR 的 rollout 吞吐

**实验目的。** `examples/rl_training/SkyRL` 在 Mini-SWE-Agent 环境中 post-train `Qwen/Qwen3-14B`。它把训练和 rollout 放入同一条实验流水线，专门比较调度器对每个 RL step 数据生成速度的影响。

**实验拓扑。** README 的固定复现配置为：

- 1 台 8×H100 80GB 节点；
- 推理侧 DP=4、TP=1，即四个 vLLM engine；
- 训练侧 FSDP2、SP=4，非 colocated，占四张 GPU；
- 数据集 `SumanthRH/SWE-Gym-Subset`；
- `train_batch_size=99`，每 prompt 采样 4 次，共 396 trajectories/step。

对照组为 `dp4_default`；实验组为 `dp4_tr_atw01`，即 `tr` router 且 `acting_token_weight=0.1`。可一次提交两个 SLURM job：

```bash
bash submit_repro_dp4_default_vs_tr_atw01.sh
```

也可分别提交 `sbatch_repro_dp4_default.sh` 和 `sbatch_repro_dp4_tr_atw01.sh`；公共 job body 位于 `scripts/repro/run_dp4_repro_job.sh`。

**指标。** 每个 rollout step 的吞吐计算为：

$$
\mathrm{tokens/sec}
=
\frac{396\times\mathrm{avg\_response\_length}}
{\mathrm{generate\_duration\_seconds}}
$$

README 示例中 default 为 1222.7 tokens/sec，TR 为 1809.6 tokens/sec。这里仍应以完整、配对完成的 step 平均值为结论，不能用一个示例 step 代替统计结果。

**预期验证。** 该实验验证四个推理 engine 上的容量感知调度、Acting token 权重、Pause/Resume/跨 backend transfer，以及训练权重更新期间的同步屏障。相比纯推理实验，它能回答“调度收益是否真正缩短在线 RL 的 rollout 阶段”，而不仅是独立 serving benchmark 是否变快。

**复现限制。** 需要 SLURM、Docker、WANDB key、Hugging Face 访问和特定镜像；集群路径、partition、artifact sync 目录都需修改。tokens/sec 不等于最终 reward 或训练收敛速度，完整结论还应在相同训练步数、seed 和样本下比较 reward/validation 指标。

### 19.8 slime/tau-bench：工具调用 Agent 的 GRPO 训练集成

**实验目的。** `examples/rl_training/slime/tau-bench` 提供 slime 在 tau-bench retail 环境上的可运行 RL pipeline，以 ThunderAgent 替换 slime 原来的 SGLang router。它主要说明 ThunderAgent 可以嵌入 colocated training/rollout 系统，而不是仓库中已经给出一组 default-vs-TR 性能曲线。

**模型和训练配置。** `run_qwen2.5_32B_thunderagent.sh` 使用 Qwen2.5-32B-Instruct，默认 8 GPU，tensor parallel size 为 8，训练算法为 GRPO。关键 rollout 参数为：

- retail 训练任务 `retail_train_tasks.jsonl`；
- `num-rollout=2`；
- `rollout-batch-size=32`；
- 每 prompt 24 samples；
- 最大 response length 2048；
- global batch size 768。

脚本启动 ThunderAgent 时选择 SGLang backend、`tr` router、Acting token decay、profiling 和 metrics，并通过 `--enable-slime-adapter` 接受训练框架动态注册的 rollout engines。每个 `(prompt, sample)` 生成独立 `program_id`，`generate_with_tau.py` 将其加入 `/generate` payload；完成、截断或 abort 时调用 `/programs/release`。

**复现入口和依赖。** README 要求 Docker GPU 环境、Hugging Face 模型权限、slime、tau-bench，以及用于模拟用户的 LiteLLM-compatible API（例如 Gemini）。准备 mock 数据、转换 Qwen checkpoint 后运行：

```bash
bash examples/tau-bench/run_qwen2.5_32B_thunderagent.sh
```

输出包括 slime 训练日志、checkpoint、trajectory JSON、step timing、ThunderAgent profile/metrics；可选打开 tau-bench dev evaluation 和 W&B。

**能验证什么。** 该流水线可验证动态 SGLang engine 注册、`/generate` adapter、Program 生命周期、外部用户模拟导致的 Acting 间隔、RL rollout 与训练 colocate，以及 ThunderAgent 在权重迭代过程中的可用性。

**不能验证什么。** 当前目录没有等价的 baseline 脚本、对照绘图或 README 中声明的吞吐提升数字，所以单独跑通它只能证明集成正确，不能证明 TR 比 slime 原 router 更快。若要形成性能实验，需要固定 seed、任务和 API provider，补充 default/原生 router 对照，并比较 rollout time、tokens/sec、KV hit、reward 和失败率。

### 19.9 如何选择实验，以及如何避免误读结果

- 想复现论文中的 Agent serving 吞吐：优先 mini-swe-agent、OpenHands 或 ToolOrchestra；其中 ToolOrchestra 的对照方法和绘图入口最完整。
- 想验证 SGLang、大规模 trajectory generation 和多节点扩展：选择 Harbor。
- 想验证调度是否加速在线 RL rollout：选择 SkyRL，它已提供配对 default/TR job。
- 想理解训练框架如何接入 Program ID、release 和动态 rollout engines：阅读并运行 slime/tau-bench，但需自行补充性能 baseline。

所有吞吐实验至少应控制模型、数据顺序、并发、生成参数、硬件、运行窗口和失败重试策略；同时报告吞吐与任务质量。KV hit 上升本身不等于吞吐必然上升，吞吐上升也可能来自请求失败或响应变短，因此要联合检查 token 数、成功率、缓存命中和 GPU 利用率。

最后要保持功能归属边界：这些目录证明 ThunderAgent 能调度复杂 Agent workload；Docker sandbox、检索器、benchmark runner、GRPO/FSDP、模型 checkpoint 转换和外部 API 均由上游系统提供，不属于 `ThunderAgent/` 核心调度器实现。

## 20. 最小可执行 Code Walk Through 实验

下面的实验目标不是复现论文全部性能，而是验证源码调用链。

### 实验 A：确认 Program ID 合并多轮请求

1. 启动一个 vLLM backend 和 ThunderAgent。
2. 连续发送两次相同 `extra_body.program_id` 的请求。
3. 访问 `GET /programs`。
4. 检查同一个 Program 的 `step_count` 是否递增、`total_tokens` 是否更新。
5. 换一个 ID 再请求，检查是否产生第二个 Program。

### 实验 B：确认 streaming usage 回调

1. 使用 `stream=true`。
2. 查看 `/profiles/{program_id}`。
3. 检查 prefill、decode、token 数和 cached token。
4. 对比 `stream=false`，确认两条路径最终都调用响应后更新。

### 实验 C：确认 Pause/Restore

1. 配置多个后端。
2. 让多个 Program 累积 token，降低可用 cache capacity。
3. 观察 `/programs` 中 `state=paused` 的 Program。
4. 观察 `/metrics` 中各 backend 的 paused count。
5. 恢复容量后观察 Program 是否重新绑定到某个 backend。
6. 对比 `origin_backend` 和恢复后的 `backend_url`，确认是否发生迁移。

### 实验 D：确认权重同步屏障

1. 调用 `/weight_sync/begin`。
2. 在另一个终端发送 chat completion。
3. 观察请求停在 `_weight_sync_event.wait()`。
4. 调用 `/weight_sync/end`。
5. 检查请求继续执行，scheduler tick 恢复。

### 实验 E：确认无工具资源管理

可以检查源码和运行行为中的边界：

- `Program` 没有工具环境字段；
- release 后核心 Program 状态会清理；
- 但没有核心代码可验证 Docker sandbox 被销毁；
- 因此必须把 Docker 资源清理放到 example/上游集成中单独验证。

## 21. 源码索引：按阅读顺序

建议实际读代码时按下面顺序打开：

1. `ThunderAgent/__main__.py:6-32`：CLI 和应用启动。
2. `ThunderAgent/config.py:6-29`：所有可配置行为。
3. `ThunderAgent/app.py:15-37`：Program ID。
4. `ThunderAgent/app.py:59-117`：chat completion 入口。
5. `ThunderAgent/program/state.py:11-48`：两个 Enum 和 Program。
6. `ThunderAgent/scheduler/router.py:305-370`：创建 Program、选择后端。
7. `ThunderAgent/scheduler/router.py:371-513`：请求前后更新。
8. `ThunderAgent/scheduler/vllm_request_processor.py:16-221`：usage、SSE、转发。
9. `ThunderAgent/backend/state.py:26-226`：容量模型和后端 Program 注册。
10. `ThunderAgent/scheduler/router.py:746-805`：后台检查和 Pause。
11. `ThunderAgent/scheduler/router.py:560-591`：全局等待队列。
12. `ThunderAgent/scheduler/router.py:692-934`：Restore 和 BFD。
13. `ThunderAgent/backend/vllm_metrics.py`、`sglang_metrics.py`、`skyrl_metrics.py`：指标适配。
14. `ThunderAgent/profile/state.py:15-280`：时延观测。
15. `ThunderAgent/app.py:119-235`：管理 API。

## 22. 最后再看论文：公式和源码的正确关系

| 论文公式/机制 | 源码中真正对应的内容 | 结论 |
|---|---|---|
| `P=<ID,c,T,L,τ,s>` | `Program`、ID 解析、BackendState、两个 Enum | 除 `T` 外有主要字段 |
| `Cost_x=∫M_x(t)dt` | Profile 时间点 + token/cache 观测 | 没有运行时积分 |
| 总成本分解 | token capacity、shared tokens、profile | 没有总成本函数 |
| Pause 的 `ΔC` | `remaining_capacity() < 0` | 负剩余容量近似，不是高水位线 |
| `min Σc_i²` | Acting 优先 + token 升序 | 启发式近似 |
| `S_restore` | 队列排序 + BFD placement | 思想相近，算法不完全相同 |
| 时间衰减 | `use_acting_token_decay` + `remaining_capacity_with_decay()` | 只用于恢复容量 |
| Global waiting queue | `global_waiting_queue` + `PausedInfo` | 核心实现 |
| 工具生命周期 Hook | 当前没有工具资源 manager | 核心未实现 |
| 引用计数 | 当前没有 `ref_count` | 核心未实现 |
| 异步工具环境准备 | 当前没有 environment prefetch | 核心未实现 |

## 23. 总结

阅读 ThunderAgent 最有效的方式不是先背论文公式，而是跟着一条请求走：

```text
app.py
  -> program_id
  -> Program
  -> update_program_before_request
  -> BackendState capacity
  -> proxy_request
  -> usage/profile
  -> Program becomes ACTING
  -> scheduler may PAUSE
  -> global queue
  -> BFD RESTORE
```

论文公式解释了为什么要这样做：Program 要跨多轮请求保持身份，短上下文优先可以降低重算成本，阶段状态影响暂停优先级，全局等待队列可以缓解节点不均衡。

源码真正落地的是：Program 状态追踪、OpenAI 代理、后端 capacity adapter、周期调度、Acting 优先暂停、全局等待队列、BFD 恢复、profiling 和 weight-sync。

源码没有直接落地的是：完整 STP 成本计算、可配置高低水位线、工具环境集合、工具资源引用计数、生命周期回收 Hook 和异步工具环境预热。Code Walk Through 必须把这些边界说清楚，这比把论文中的每一项设计都强行映射到一个不存在的函数更准确。
