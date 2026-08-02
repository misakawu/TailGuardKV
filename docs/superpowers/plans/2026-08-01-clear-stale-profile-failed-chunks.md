# Clear Stale Profile Failed Chunks Implementation Plan

Goal: 修复成功重跑 profile 后旧 `_failed_chunks.csv` 仍残留、误导用户认为当前输出存在 failed 请求的问题。

Architecture: 在 `build_profile_table()` 开始生成新 profile 主表时，同步清理同 stem 的失败诊断 sidecar；chunk 失败时仍按现有逻辑写出 sidecar。用现有 chunk-failure 测试模式补一个回归测试，证明成功重跑会移除旧诊断文件。

Tech Stack: Python stdlib, `unittest`, existing `run_util.build_profile_table`, existing CSV helpers.

## Tasks

- [ ] Add regression test for stale sidecar cleanup.
- [ ] Clear stale failed-chunks sidecar at run start.
- [ ] Document the cleanup behavior in `README.md`.
- [ ] Run focused verification, inspect diff, and commit.

## Constraints

- 运行所有的脚本或程序时，采用异步方式；进程运行结束或报错时唤醒会话。
- 不改动无关实验输出，不回滚用户已有工作树改动。
- 不改变 profile CSV schema，不改变 chunk 失败时保留已完成主输出的既有行为。
