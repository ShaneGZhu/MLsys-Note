

# 需要对齐的问题
- [ ] 是否只借鉴/替换路由策略层——用 Dynamo 的 KV-aware router 替换 Infer-Router？
- [ ] 实验环境采用GLM、H卡、只关注调度策略
	- [ ] 明确"有收益"的量化标准，例如：rollout step 端到端耗时 ↓X%、prefix cache 命中率 ↑X%、GPU 利用率 ↑X%、且训练精度/收敛曲线不劣化；
	- [ ] 是否需要配置veRL的环境，来验证实验 or 只根据Dynamo的工具来评估？
- [ ] 内部 RL 栈用的推理引擎是 SGLang，**KVBM在Dynamo中尚未实现**，是否影响实验结论。

# 摸清内部 Infer-Router SessionAware
