# Session25 Policy Attempt Root Design

## Goal

让 session25 policy 网格在不覆盖既有 profile 测量和失败现场的前提下，使用独立的 attempt root 运行完整 240-cell 实验。

## Design

- `--run-root` 是只读 profile 输入根目录，继续提供 `manifest.json`、`supervisor_manifest.json`、`budgets.json` 和合并后的 profile CSV。
- 新增带注释的 `--attempt-root` 参数；提供时，seed 配置、policy CSV、cell 日志、`fourth_grid_status.json` 和按 seed 聚合结果全部写入该目录。
- 未提供 `--attempt-root` 时，保持旧的 `run_root` 输出行为，避免破坏已有调用方。
- attempt root 中已有完整且验收通过的 CSV 可以恢复；已有不完整 CSV 直接 fail closed，禁止覆盖。
- 状态文件记录 `profile_run_root`、`attempt_root`、fixture provenance、总请求数 `125`、calibration 请求数 `60`、evaluation 请求数 `65`，以及每个 cell 的输出、日志、返回码和 split 数。

## Testing

测试通过 monkeypatch 的 policy runner 验证输入输出根目录隔离，同时保留旧 run root 的不完整 CSV，确认它不会阻塞新 attempt。测试还验证状态元数据和每个 policy 的 65 个唯一 evaluation request ID；另测 attempt root 自身的不完整 CSV 会拒绝覆盖。
