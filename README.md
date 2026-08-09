# TailGuardKV

## 配置

当前实验只保留三份入口配置。

- [`configs/pilot.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot.yaml)  
  baseline 正式配置。它使用带 `reference` 的 [`data/fixtures/pilot_qa_summary_requests.jsonl`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/data/fixtures/pilot_qa_summary_requests.jsonl)，`data.quality_mode=baseline`，用于生成 `QA + 摘要` 主 smoke 结果。
- [`configs/pilot_50.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_50.yaml)  
  baseline 小规模 smoke 配置。配置结构和 `pilot.yaml` 一致，只是把 `max_requests` 收敛到 50，用于快速检查。
- [`configs/pilot_sharegpt.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_sharegpt.yaml)  
  ShareGPT session/cache 诊断配置。它使用 [`data/fixtures/sharegpt_sessions.json`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/data/fixtures/sharegpt_sessions.json)，`data.quality_mode=session_diagnostic`，只验证多 session、多 turn 下的 replay、resident、budget 和 cache 行为，不参与 baseline 质量比较。

“baseline 和 ShareGPT 分开” 的意思是：baseline 只在带 `reference` 的独立 `QA + 摘要` 请求集上做质量比较；ShareGPT 只做会话和缓存诊断，不再被拿来生成 baseline 质量表。

## 启动

推荐只用这一条命令启动完整实验：

```bash
conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
```

如果要跑小规模 baseline smoke 或 ShareGPT 诊断，只替换 `--config` 为 `configs/pilot_50.yaml` 或 `configs/pilot_sharegpt.yaml`。

## 执行流程

当前实现分三层。

第一层是实验编排层，由 [`run_experiment.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_experiment.py) 的 `pilot-smoke-measured` 入口负责。

- 读取配置。
- 解析输出目录。
- 先调用 profile 构建阶段。
- profile 成功后，再调用 policy replay 阶段。
- 最后汇总 summary 和可视化输出。

第二层是 profile 构建层，由 [`run_util/build_profile_table.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/build_profile_table.py) 负责。

- 读取请求集和 runtime 配置。
- 根据 `quality_mode` 判断当前是 baseline 还是 session 诊断。
- baseline 模式下，先校验请求是否有受支持的 `task`，以及是否带 `reference`。
- 按 profile 和 request chunk 调用 adapter，生成 measured profile rows。
- 给每条 measurement 补充 `task`、`split`、`reference`、`history_turns` 等上下文。
- 计算 `quality_loss`、`quality_score` 和额外 loss 指标。
- 校验 profile 表字段完整性。
- 增量写出 profile CSV 和 session trace。

第三层是 policy replay 层，由 [`run_util/run_policies.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/run_policies.py) 负责。

- 读取 measured profile 表。
- 切分 calibration 和 evaluation 请求。
- 构造 measured replay backend。
- 按 `epsilon`、`delta`、`memory_budget_mib` 生成 policy sweep 组合。
- 对每个 policy 和每个 request 做回放决策。
- 记录动作、fallback、quality、memory、ttft 等结果。
- 写出 policy CSV。
- 汇总 policy summary，返回给实验编排层。
