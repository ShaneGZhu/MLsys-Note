# MySkills — 个人 Claude Code Skills 管理

本目录用于存放个人积累的 Claude Code Skills，方便跨项目复用。

## 目录结构

```
MySkills/
├── README.md                          # 本文件
├── llm_skills_context.md              # Skills 官方文档（中文）
└── .claude/
    └── skills/
        └── nsys-profile-fd/           # nsys profiling 自动化 Skill
            ├── SKILL.md
            └── scripts/
                └── patch_config.sh
```

---

## 如何让 Skills 生效

### 方式一：软链接到项目 `.claude/skills/`（推荐，持久生效）

将 Skill 目录软链接到目标项目的 `.claude/skills/` 下，Claude Code 在该项目中启动时会自动发现并加载。

```bash
# 语法
ln -s <MySkills中skill的绝对路径> <目标项目>/.claude/skills/<skill-name>

# 示例：将 nsys-profile-fd 链接到 FastDeploy 项目
ln -s /root/paddlejob/workspace/env_run/output/zhushengguang/jobspace/notebooks/MySkills/.claude/skills/nsys-profile-fd \
    /root/paddlejob/share-storage/gpfs/system-public/zhushengguang/jobspace/FastDeploy/.claude/skills/nsys-profile-fd
```

**特点：**
- 一次配置，永久生效
- 修改 MySkills 中的 Skill 内容，所有链接项目实时同步
- Skill 归属于项目级别，不影响其他项目

---

### 方式二：启动时 `--add-dir` 加载（灵活，无需改项目结构）

启动 Claude Code 时通过 `--add-dir` 指定本目录，Claude Code 会自动扫描其中的 `.claude/skills/` 并加载所有 Skills。

```bash
# 语法
claude --add-dir <MySkills目录路径>

# 示例
claude --add-dir /root/paddlejob/workspace/env_run/output/zhushengguang/jobspace/notebooks/MySkills
```

**特点：**
- 无需修改任何项目文件
- 每次启动时指定，适合临时使用或在多个不同项目中按需加载
- 支持实时热更新：会话期间修改 Skill 文件无需重启即可生效

---

## Skills 列表

| Skill 名称 | 调用命令 | 说明 |
|-----------|---------|------|
| nsys-profile-fd | `/nsys-profile-fd` | 自动化 FastDeploy 推理服务 nsys profiling，支持双轮参数对比 |

### nsys-profile-fd 快速参考

```bash
# 零参数：复现 DeepSeek-V3.2 5层 cudagraph 对比实验
/nsys-profile-fd

# 自定义参数：指定脚本路径、报告前缀、对比参数
/nsys-profile-fd <server_script> <bench_script> <report_prefix> <param_key=val1,val2>

# 示例：对比 full_cuda_graph 开关
/nsys-profile-fd deepseekv32_5layer_EP.sh benchmark_serving_multinode.sh dsv32_fullgraph full_cuda_graph=false,true
```

默认值：
- 服务脚本：`/root/paddlejob/share-storage/gpfs/system-public/zhushengguang/jobspace/scrips/deepseekv32_5layer_EP.sh`
- Benchmark：`/root/paddlejob/share-storage/gpfs/system-public/zhushengguang/jobspace/scrips/benchmark_serving_multinode.sh`
- 报告目录：`/root/paddlejob/workspace/env_run/output/zhushengguang/jobspace/scrips/`
- 对比参数：`use_cudagraph=false,true`

---

## 添加新 Skill

在 `.claude/skills/` 下创建新目录，包含 `SKILL.md` 即可：

```bash
mkdir -p .claude/skills/<skill-name>
# 编写 .claude/skills/<skill-name>/SKILL.md
# 可选：添加 scripts/ 等支持文件
```

然后在本 README 的 Skills 列表中补充记录。
