# baseline_session 修正方案执行记录（2026-09-01）

## 已完成

- 外部标注工作区新增 hybrid session 构造器：48 个五轮 session、240 行候选，turn 0/1 为 ShareGPT，turn 2/3/4 为 LongBench 注入与固定复述模板。
- 外部 labeler 强制使用 hybrid 候选与 `(request_id, session_id, turn_index)` 精确匹配的实测记录，并要求 KIVI/H2O 主流 profile 在 turn 2–4 完整覆盖。
- 主仓库导入校验和风险 gate 已独立实现；风险统计排除 chat、calibration、非法 provenance 和非有限 loss。
- baseline_session runner 已拆分为 `trace_semantics_gate` 与 `risk_signal_gate`：trace 失败阻断 replay；风险失败保留 replay 并标记 `risk_evidence_insufficient`。

## 验证

- Task 1 外部测试：10 passed。
- Task 2 主仓库测试：30 passed。
- Task 3 session/config 测试：73 passed；完整 session 回归：88 passed。
- 全量 pytest：342 passed，3 failures（两项外部 legacy fixture 测试、一项既有 profile 顺序断言）。

## 阻塞

hybrid 候选真实 profile 测量在 `full_gpu` 的 `hybrid-session-009-turn-2` 处 CUDA OOM：GPU 1 仅剩约 1.32 GiB，尝试分配 1.40 GiB。已写入 2048 行部分测量和 failure chunk 诊断，未据此导出 fixture、导入主仓库或运行正式 policy smoke。继续执行需要先决定显存/分批策略。
