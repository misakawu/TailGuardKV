# 论文计划未实现事项 TODO

来源：对照 `论文规划/王祯祥_论文工作规划.md` 中 2026-07-14-07-31、2026-08-01-08-15 的要求，并复查 `profile-backend复查.md` 与当前实现。

## P0：先补会卡住验收的缺口

1. 接入真实 vLLM/LMCache smoke trace
   - 当前 `profiles/registry.py` 只注册 `full`、`kivi`、`h2o`。
   - 需要新增最小 vLLM 或 LMCache adapter/backend，至少跑 1 条请求并生成 `measured=True` 记录。
   - 输出里要能看出真实 backend，例如 `extra_backend=vllm` 或 `extra_backend=lmcache`。
   - 失败时不能静默回退到 Transformers。

2. 修正 TTFT 口径
   - 当前 Transformers 和 Qwen2 runtime 都把 `ttft_ms` 写成整次生成总耗时。
   - 要么实现首 token 计时，要么把正式 summary 和论文表述统一改成 `latency_ms`。
   - 加测试挡住“total latency 填进 ttft_ms”的回归。

3. 拆出 KV cache memory
   - 当前 `peak_memory_mib` 是端到端 CUDA peak，不是 KV-only 成本。
   - 增加 `kv_cache_memory_mib` 或 `profile_memory_mib`。
   - full、KIVI、H2O 必须按同一口径统计 KV cache bytes。
   - summary 同时报端到端 peak 和 KV-only memory。

4. 实现并注册 `tailguard`
   - 现在 policy registry 只有 5 个 baseline，没有完整 TailGuardKV 策略。
   - 需要把 QRP、Conformal Guard、STC、exact fallback 接成一个正式 policy。
   - policy summary 要能报告 safe set、fallback ratio、controller overhead、coverage 相关指标。

5. 恢复或重写 `quality_oracle`
   - 计划要求 `quality_oracle` 作为上界，不参与可部署 baseline 排名。
   - 当前 registry 明确拒绝 `quality_oracle`。
   - 需要实现只读真值的 oracle，并在 summary 里标记 `oracle=True`。

## P1：修正会误导实验结论的口径

1. 修正 profile summary 的违例率阈值
   - `MetricCollector.summarize_profiles()` 现在固定用 `epsilon=0.5`。
   - 应改为读取 config 中的 epsilon，或输出 `violation_rate_eps0p05`、`violation_rate_eps0p10`。
   - profile summary 和 policy summary 的 SLO 口径要一致。

2. 明确 quality loss 的主指标
   - 当前输出有 EM/F1/ROUGE-L，但没有 `extra_metric_primary`。
   - `qa_long_context` 现在不会走 QA 的 F1 逻辑，会落到 EM。
   - 需要补任务名映射，并在每条 profile row 里写清 primary metric。

3. 隔离 dry-run 结果
   - dry-run 常数仍在 adapter 里，虽然默认 replay 会拒绝 `measured=False`。
   - dry-run 输出文件名、summary 和 extra 字段都应明确标记 `dry_run=true` 或 `source=synthetic_schema_check`。
   - 正式 profile 输出路径不要使用容易被误读的 `measured` 命名。

4. 文档区分真实模型 runtime 和真实 serving backend
   - 当前 `pilot_50` 能证明 full/KIVI/H2O 跑过真实模型路径。
   - 它还不能证明 vLLM/LMCache backend 已经验收。
   - README、实验总结和 summary 里要写清 backend family，避免把 Transformers/Qwen2 runtime 写成 vLLM/LMCache。

## P2：补齐 08-01-08-15 的 runner 与 baseline 验收

1. 让同一个 runner 跑完整策略集合
   - 当前 `configs/pilot_50.yaml` 只配置 5 个 baseline。
   - 补上 `tailguard` 和 `quality_oracle` 后，`run_experiment.py pilot-smoke-measured` 应能同表输出所有策略。

2. 补 baseline smoke 表
   - 需要一份小规模 smoke 输出，明确包含 5 个 baseline、TailGuardKV、oracle。
   - 表里至少包含 p95 TTFT、memory、quality loss、violation rate、fallback ratio、action distribution。

3. 报告 controller 开销
   - 计划要求拆出 `controller_qrp_ms`、`controller_cg_ms`、`controller_stc_ms`。
   - 当前字段已经在 `PolicyRunRecord` 中预留，但完整 TailGuard 策略还没有接上实测开销。

4. 做一次 H0/H1/H2-lite 汇总检查
   - H0：mean/p95/p99/CVaR/violation 是否能说明 tail-risk。
   - H1：coverage、worst-group、UCB slack、exact fallback ratio 是否达标。
   - H2-lite：同一 SLO 下，TailGuardKV 相比最佳可部署 baseline 是否有 p95 TTFT 或显存收益。

## 已完成但需要继续守住

1. full/KIVI/H2O measured profile 已经能生成。
2. measured replay 默认拒绝 dry-run 表。
3. 5 个 baseline 已能通过同一 policy runner 回放。
4. profile 表已按 chunk 增量写入，并能输出失败 chunk 诊断。

## 完成进度（2026-08-01 P0）

- 已实现 P0 1-5 的代码侧改动：新增 `vllm` adapter/profile `engine_full_lru`，修正 TTFT 不再用总 latency 填充，新增 `kv_cache_memory_mib` schema/validation/summary/replay 字段，注册 `tailguard`，注册 `quality_oracle` 且 summary 标记 `oracle=True`。
- 单元验证命令：
  `python3 -m pytest tests/test_tailguard_core.py tests/test_kivi_cache.py tests/test_h2o_cache.py -q`
  - 异步日志：`out/logs/p0_pytest.nohup.log`
  - 结果：`145 passed in 1.73s`，exit code `0`
- vLLM smoke 命令：
  `python3 -m run_util.build_profile_table --config configs/p0_smoke.yaml --adapters vllm --output out/profile_tables/p0_vllm_smoke_profiles.csv --no-dry-run`
  - 异步日志：`out/logs/p0_vllm_smoke.nohup.log`
  - exit code：`2`
  - 输出 CSV：`out/profile_tables/p0_vllm_smoke_profiles.csv` 未生成
  - 失败诊断：`out/profile_tables/p0_vllm_smoke_profiles_failed_chunks.csv`，1 行失败记录
  - vLLM 行状态：`measured=False`，`ok=False`，`extra_backend=vllm`
  - 失败摘录：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 260.00 MiB. GPU 0 has a total capacity of 10.75 GiB of which 225.62 MiB is free... RuntimeError: Engine core initialization failed.`
- policy replay 命令：
  `python3 -m run_util.run_policies --config configs/p0_smoke.yaml --measurements out/profile_tables/p0_vllm_smoke_profiles.csv --output out/policy_tables/p0_policy_smoke.csv --policies full_lru tailguard quality_oracle --epsilon 0.05 --delta 0.05`
  - 异步日志：`out/logs/p0_policy_smoke.nohup.log`
  - exit code：`2`
  - 输出 CSV：`out/policy_tables/p0_policy_smoke.csv` 未生成
  - 失败原因：上一步 vLLM measured 表未生成，replay 报 `[Errno 2] No such file or directory: 'out/profile_tables/p0_vllm_smoke_profiles.csv'`
