---
name: nsys-profile-fd
description: Run nsys profiling on FastDeploy inference server with configurable parameters. Automates: start server with nsys launch, wait for port ready, nsys start, run benchmark, nsys stop, save .nsys-rep report. Supports two-round comparison by toggling a config parameter (e.g. use_cudagraph=false,true). Use when profiling FD server performance or comparing two inference configurations.
argument-hint: "[server_script] [bench_script] [report_prefix] [param_key=val1,val2]"
disable-model-invocation: true
allowed-tools: Bash Read Edit
---

# Skill: nsys-profile-fd

对 FastDeploy 推理服务进行 nsys profiling，支持自动对比两种配置（如 cudagraph 开/关）。

## 参数说明

调用格式：`/nsys-profile-fd [$0: server_script] [$1: bench_script] [$2: report_prefix] [$3: param_key=val1,val2]`

- `$0`：服务启动脚本绝对路径（可省略，使用默认值）
- `$1`：benchmark 脚本绝对路径（可省略，使用默认值）
- `$2`：报告文件名前缀（可省略，使用默认值）
- `$3`：对比参数，格式 `key=val1,val2`（可省略，使用默认值）

**不传任何参数**时，使用以下默认值复现 DeepSeek-V3.2 5层 cudagraph 对比实验：

```
server_script = /root/paddlejob/share-storage/gpfs/system-public/zhushengguang/jobspace/scrips/deepseekv32_5layer_EP.sh
bench_script  = /root/paddlejob/share-storage/gpfs/system-public/zhushengguang/jobspace/scrips/benchmark_serving_multinode.sh
report_dir    = /root/paddlejob/workspace/env_run/output/zhushengguang/jobspace/scrips
report_prefix = dsv32_5layer_cudagraph
param_key     = use_cudagraph
param_vals    = false,true
wait_ip       = 10.95.239.139
wait_port     = 8291
nsys_session  = zsg
```

## 调用示例

```bash
# 1. 使用默认值，直接复现本次实验
/nsys-profile-fd

# 2. 自定义报告前缀
/nsys-profile-fd deepseekv32_5layer_EP.sh benchmark_serving_multinode.sh my_exp use_cudagraph=false,true

# 3. 对比其他参数（如 full_cuda_graph）
/nsys-profile-fd deepseekv32_5layer_EP.sh benchmark_serving_multinode.sh dsv32_fullgraph full_cuda_graph=false,true

# 4. 只跑单轮（不做对比）
/nsys-profile-fd deepseekv32_5layer_EP.sh benchmark_serving_multinode.sh single_run use_cudagraph=true
```

## 执行步骤

执行本 Skill 时，按以下流程操作：

### 0. 解析参数

解析 `$ARGUMENTS`：
- 若有位置参数，`$0` = server_script，`$1` = bench_script，`$2` = report_prefix，`$3` = param_key=val1,val2
- 若无参数或部分缺失，使用上述默认值填充
- 从 `$3`（或默认 `use_cudagraph=false,true`）中拆分 param_key、val_list（逗号分隔为多轮）

### 1. 对每轮参数值循环执行以下步骤

设当前轮参数值为 `CURRENT_VAL`，报告名为 `${report_prefix}_${param_key}_${CURRENT_VAL}`。

#### Step 1：修改服务脚本配置

使用辅助脚本将服务脚本中 JSON 配置里对应 key 修改为当前轮的值：

```bash
bash ${CLAUDE_SKILL_DIR}/scripts/patch_config.sh <server_script> <param_key> <CURRENT_VAL>
```

修改后验证（grep 确认值已更新）。

#### Step 2：后台启动服务

```bash
nsys launch --session=<nsys_session> bash <server_script> > /tmp/fd_server_nsys_<CURRENT_VAL>.log 2>&1 &
SERVER_PID=$!
```

#### Step 3：等待服务端口就绪

```bash
for i in $(seq 1 120); do
    if nc -z <wait_ip> <wait_port> 2>/dev/null; then
        echo "✓ 端口就绪 (${i*5}s)"; break
    fi
    echo "等待服务... ${i}/120"; sleep 5
done
```

若 120 次后仍未就绪，打印错误日志后终止本轮。

#### Step 4：启动 nsys 数据采集

```bash
nsys start \
    --output=<report_dir>/<report_name> \
    --session=<nsys_session>
```

验证：
```bash
nsys sessions list  # 确认状态为 Collection
```

#### Step 5：执行 benchmark

```bash
bash <bench_script> 2>&1 | tee /tmp/benchmark_nsys_<CURRENT_VAL>.log
```

#### Step 6：停止 nsys 采集

```bash
nsys stop --session=<nsys_session>
```

等待报告文件生成完毕（检查 `.nsys-rep` 文件是否存在）。

#### Step 7：停止服务

```bash
kill $SERVER_PID 2>/dev/null
pkill -f "multi_api_server" 2>/dev/null
# 等待端口释放
until ! nc -z <wait_ip> <wait_port> 2>/dev/null; do sleep 2; done
echo "✓ 服务已停止，端口已释放"
```

### 2. 所有轮次完成后：输出汇总

打印如下内容：

1. **报告文件列表**：
```
生成的 nsys 报告：
  <report_dir>/<report_name_round1>.nsys-rep
  <report_dir>/<report_name_round2>.nsys-rep
```

2. **benchmark 指标对比表**：从各轮 `/tmp/benchmark_nsys_*.log` 中提取以下指标，以 Markdown 表格输出：
   - Throughput (req/s, tok/s)
   - Mean / Median TTFT (ms)
   - Mean / Median TPOT (ms)
   - Mean / Median ITL (ms)
   - Mean / Median E2EL (ms)

## 支持文件

- `scripts/patch_config.sh`：修改服务脚本中 JSON 配置字段值的辅助脚本
