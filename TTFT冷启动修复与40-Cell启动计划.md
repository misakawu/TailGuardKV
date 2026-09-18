# TTFT 冷启动修复与 40-Cell 启动计划

## 目标

修复 persistent worker 的首次模型加载被计入 TTFT 的问题。保留旧异常结果作为失效审计证据，但不参与后续聚合；不删除原始样本。

## 实施范围

- 为 persistent worker 和 Qwen2 KV runtime 增加 `warm_profile` RPC：按传入 profile 加载运行时和模型，不生成、不写 session history/KV、不改变 cache state。
- `OnlineQwenSessionBackend` 记录 worker PID/generation、启动和模型加载耗时及预热状态，且仅作为审计元数据。
- 每个 policy 的首个已决 action 在 shadow/真实执行前预热；动态策略后续 profile switch 的释放、加载和重算成本继续计入策略指标。
- 新增 002/003/004 三个 session 的 75-record 诊断门禁、两档紧预算 preflight，以及通过门禁后异步启动的新 40-cell 单 seed 实验。

## 放行条件

三 session 门禁必须无失败、无 worker restart/state loss，且无启动字段混入服务指标；每条 TTFT/recompute 不超过相应 calibration p99 的五倍。两档紧预算 preflight 必须在每个 lossy policy 中出现 policy 筛选或 backend budget event，且全局驻留不超过预算。

正式运行使用 `nohup setsid`，新建运行目录，不覆盖历史结果；运行期间不轮询，待用户通知进程结束后才聚合、作图和验收。
