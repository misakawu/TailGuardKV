# Session25 Policy Runtime 最小验证设计

## 目标

在不启动完整 policy cell 的前提下，使用现有 25-session profile 测量与 evaluation split，验证本轮修复后的 online Qwen 运行条件：exact profile 可运行、KIVI 多轮 cache 可复用、profile runtime 切换后可安全重建。

本验证仅用于 `diagnostic_only=true` 的运行门禁，不生成或替代正式 policy 结果。

## 输入与选择

- 配置：`configs/pilot_diagnostic_session25_fourth.yaml`。
- Profile 测量：`out/full_25session_baseline_9_14/full/merged/profile_tables/` 中唯一合并 CSV。
- Fixture：配置中的 125-request fixture。
- 按配置的 `split_seed`、`calibration_fraction` 和 session stratification 复算 evaluation request 集合。
- 从 evaluation split 按 session ID 稳定排序选择三个完整多轮 session；不足三个时 fail closed。

## 验证场景

1. **Exact 场景**：新建 backend，对第一个 evaluation session 的前两个 turn 使用 `full_gpu`，要求所有请求成功、TTFT 有效。
2. **KIVI 场景**：新建 backend，对第二个 evaluation session 的前三个 turn 使用 `kivi_4bit_residual64`，要求所有请求成功、TTFT 有效，第二和第三个 turn 明确报告 cache reuse。
3. **Runtime 切换场景**：新建 backend，执行第三个 session 的 `full_gpu` turn 0，再执行另一个 evaluation session 的 `kivi_4bit_residual64` turn 0，最后回到第三个 session 执行 `full_gpu` turn 1；要求三次执行成功，返回 full 时有 runtime transition/recompute 证据。

每个场景独立创建并关闭 backend，避免前一场景的模型或 CUDA cache 影响后一场景。

## 输出与失败门禁

- 新增脚本：`scripts/validate_session25_policy_runtime.py`。
- 参数：`--config`、`--run-root`、`--output`。
- 输出 JSON 保存输入 provenance、split 请求数、选中 session、逐请求关键字段、各场景状态和总状态。
- 任意 runtime error、OOM、无效 TTFT、KIVI reuse 缺失、runtime transition/recompute 证据缺失均返回非零退出码。
- backend 必须在 `finally` 中关闭；失败记录保留原始错误文本，不转换成成功或零值。

## 启动方式

实现与 CPU 回归测试通过后，使用 `nohup + setsid` 在 `tailguardkv-base` 环境异步启动，日志、PID、退出码和 JSON 报告写入新的独立目录。GPU 实验运行期间不轮询，结束后由用户通知再检查。

## 非目标

- 不运行 240-policy 完整网格。
- 不生成 baseline summary 或图表。
- 不覆盖既有失败日志、CSV、状态文件或 profile 测量。
- 不将验证结果用于正式论文结论。
