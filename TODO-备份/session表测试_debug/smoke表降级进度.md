# Smoke 表降级进度与后续交接

更新时间：2026-09-05。本文是后续会话的工作起点，不是实验结论。当前仓库正处于一次较大的历史清理过程中；不要恢复、覆盖或重新加入已删除的旧 `out/` 结果，也不要用它们作为 gate 证据。

## 当前结论

正式 `baseline_quality` 和 `baseline_session` 尚未完成，不能出可比较的 baseline 表。ShareGPT 中不足以组成正式 48-session canonical fixture 的 skeleton session，因此已按 [理想baseline_smoke表说明.md](理想baseline_smoke表说明.md) 末尾的约定降级为 `diagnostic_only`。

这个降级实验只检查 runtime、8 个 profile 的测量覆盖和日志链路。它不满足 48 个 session、canonical bootstrap/replay、512-token final-form preflight、风险分层覆盖等正式条件；任何 policy 表、图或数字都不能用于论文结论，也不能代替 quality/session gate。

## 已完成的工作

1. 已实现 canonical-history 基础合约。
   - [run_util/canonical_history.py](run_util/canonical_history.py) 定义了 schema v1、fixture hash、bootstrap table hash、full-GPU 来源、每 turn history hash 和输出 token 上限 16。
   - [run_util/canonical_session.py](run_util/canonical_session.py) 按全局 arrival 顺序运行 bootstrap，并用 full-GPU 生成的 assistant 文本推进同一 session。`reference` 仍只用于质量评分，不能进入运行 history。
   - [run_util/data_utils.py](run_util/data_utils.py) 会在读取 canonical fixture 时验证上述 metadata，并以固定的 canonical history 替换普通的 reference-derived history。

2. 已在 full、KIVI、H2O runtime 中接入 canonical cache 的 fail-closed 检查。
   - 继续 turn 必须匹配 profile、上一个 turn、history hash 和 prompt 的严格 token 前缀；不匹配返回 `canonical_history_mismatch`。
   - full/H2O 路径已具备使用 suffix token 和已有 cache 的代码，KIVI 保留量化 cache 路径。
   - 这些改动已有 focused unit test 文件，但本次清理和降级后尚未完成一次完整测试验证，不能把它们视为已验收。

3. 已收紧部分外部 fixture 导入接口。
   - [scripts/import_external_fixtures.py](scripts/import_external_fixtures.py) 新增了 `--manifest`，并检查 schema 和 fixture hash。
   - [configs/pilot_external_baseline_session.yaml](configs/pilot_external_baseline_session.yaml) 固定 `max_new_tokens: 16`、`profile_chunk_size: 1`、persistent worker 与双卡策略。

4. 已新增 diagnostic 分批执行器。
   - [scripts/run_diagnostic_session_batches.py](scripts/run_diagnostic_session_batches.py) 现在是串行 supervisor：每个 batch 通过独立子进程运行，逐批持久化状态，即使子进程非零退出也继续后续 batch；只有全部严格覆盖后才原子写入 `merged/`。
   - 每个生成的 fixture/config/manifest 都带 `diagnostic_only: true` 和 `batch_id`。profile coverage 要求每个 fixture request 恰好有 8 个 `ok=true`、`measured=true` 的 profile 行；runtime 的精确 `__pressure_r<round>_c<copy>` request-ID 后缀会被映射回 fixture ID，其他 ID 改写仍拒绝。
   - supervisor focused tests 最近一次结果为 `18 passed`。该测试不代表 GPU runtime、session trace 或正式 baseline 已验收。
   - [scripts/visualize_diagnostic_smoke.py](scripts/visualize_diagnostic_smoke.py) 仅生成诊断完成度图，标题已标明不得作正式比较。

5. 已按要求清理大量旧 fixture、旧 smoke 输出和过时规划文件。工作树中的删除属于当前清理，不要以 `git checkout --`、`git reset --hard` 或宽泛的 `git clean` 恢复。尤其不要对整个 `out/` 做删除：其中有当前运行使用的 batch fixture、config、lock 和日志。

### 分批 supervisor 与合并契约

当前分批执行器由 supervisor 按 manifest 顺序串行运行每个隔离 batch，并在每个 batch 结束后写入状态；所有 batch 都完成后才做最终合并。它会写 `supervisor_manifest.json`，其中记录每批的返回码、输出目录、profile coverage、诊断 gate 状态和是否可合并。

合并只接受严格完整的 profile coverage：每个 fixture `request_id` 必须恰好覆盖 manifest 中的每个 profile，且每行同时满足 `ok=true` 与 `measured=true`。只有全部 batch 通过该校验时才写入 `merged/` 及其 manifest；缺行、重复行、未知 profile、OOM、runner 异常或其他非零退出都会保留 diagnostics/status，并拒绝合并。若非零退出仅由已知 diagnostic trace/risk gate 失败造成、而 profile coverage 仍完整，则只有在同时存在失败的 trace/risk gate artifact，且当前 summary 中唯一的非空 `error` 恰好是以下明确消息之一时，才允许作为 diagnostic-only 结果合并：`session trace semantics gate failed`、`session risk signal gate failed`、`trace semantics gate failed`、`risk signal gate failed`。OOM 或任何其他 summary error 即使存在 gate artifact 也不可合并，并必须保留 `gate_only_failure` 为 false。

因此，运行期间应以顶层 `supervisor_manifest.json` 为准检查所有 batch 状态，再检查 merged profile CSV 的行数；不能依据单个 batch 的 partial CSV 或旧的 `merged/` 目录判断成功。该链路始终保持 `diagnostic_only: true`，其合并结果不得升格为 canonical fixture 或正式 smoke 表。

## 已知失败与当前实验

前两次完整 27-session diagnostic run 均发生 CUDA OOM，且不具备正式解释价值：

- `out/20260905_101723_pilot_diagnostic_session27/`：full-GPU 路径仅部分完成，随后失败。
- `out/20260905_102416_pilot_diagnostic_session27/`：重试仍失败；失败记录显示 GPU 0 仅余约 162 MiB，申请 884 MiB 时触发 `OutOfMemoryError`。

此前分批 run 已结束；不可据此启动第二个 GPU 任务：

- 启动时间：2026-09-05 10:33:31 +0800。
- 顶层 PID：`33034`；锁：`out/locks/diagnostic_session27_gpu01.lock`。
- 日志：`out/logs/diagnostic_session27_batched_20260905_103331.nohup.log`。
- 命令：`scripts/run_diagnostic_session_batches.py --fixture data/fixtures/diagnostic_session27.jsonl --config configs/pilot_diagnostic_session27.yaml --run-root out/diagnostic_session27_batches_20260905_103331 --sessions-per-batch 3`。
- 该进程在 `batch000`（3 个 session、15 条请求）后因 session trace semantics gate 非零退出；当时仍是 supervisor 接入前的 runner，因此没有生成 `supervisor_manifest.json`，也没有生成后续 batch 输出。`batch000` 及其 partial CSV 不得合并为 canonical fixture、正式 fixture 或 smoke 表。

### 最新 supervisor 运行记录

1. `out/diagnostic_session27_supervisor_20260905_114003/`：3 session/batch、共 9 个 batch 已全部执行；顶层状态在 `supervisor_manifest.json`，未生成 `merged/`。
   - `batch000`、`001`、`002`、`004`、`006`、`008` 各得到完整 `120/120` 行，且所有已写 profile 行均为 `ok=true`、`measured=true`；但 trace/risk gate 未形成可比较 policy 结果。
   - `batch003`、`005`、`007` 分别仅得到 `14/120`、`12/120`、`14/120` 行。三次都在 full profile 的最后一轮发生 GPU 1 OOM：`hybrid-session-011-turn-4` 申请 1.31 GiB、`015-turn-4` 申请 1.68 GiB、`023-turn-4` 申请 1.35 GiB。失败时 GPU 1 剩余分别约 640 MiB、1.40 GiB、778 MiB。
   - 总共保留 `760/1080` 条成功测量行；这些行只作 runtime 诊断。该 run 当时尚未修复 pressure-trace request-ID 映射，6 个完整 batch 在 manifest 中也被误记为 unknown/missing coverage；这不改变整体不可合并结论，因为 3 个 batch 存在真实 OOM partial。

2. `out/diagnostic_session27_supervisor_single_20260905_124503/`：在修复 request-ID 映射后，以 1 session/batch 重跑，得到 27 个 `0/40` batch，未发生模型测量也未生成 `merged/`。
   - 直接原因是 session preflight：`baseline_session 要求至少两个交错 session`。单 session 违反 trace 语义最小输入契约，不能用于规避 OOM，也不能作为 profile 或 OOM 证据。

分批降低的是单次数据读取和会话驻留规模，但 `baseline_session` 必须至少保留两个交错 session。下一次运行不能使用 1 session/batch，也不要简单恢复原始三 session 组合；应先实现并测试非均匀的两 session 压力配对分组，再启动新的独立 run root。重点把曾 OOM 的 `011`、`015`、`023` 分别与低驻留 session 配对，避免与高驻留的 `009`、`017`、`021` 同批。保留上述两个 run root、日志和 manifest，不得删行、补行、覆盖或跨 run 合并。

## 尚未实现或尚未验收的正式链路

以下项目是正式实验的阻塞项，按顺序处理。

1. 完成真实 supervisor，而不是只保留校验骨架。
   [scripts/run_canonical_session_supervisor.py](scripts/run_canonical_session_supervisor.py) 目前能验证输入长度、已有 canonical manifest 和 replay 覆盖，但还没有实际调用 persistent full-GPU worker 生成 `bootstrap/full_gpu.csv`、冻结 `canonical_fixture.jsonl`，也没有驱动 7 个 lossy profile 的 replay 或 strict merge。需要实现 `bootstrap -> token recheck -> 7-profile replay -> merge`，任何 `ok/measured=false`、OOM、profile 缺失或 hash 不同都只写 diagnostics 并非零退出。

2. 补齐 bootstrap 的运行期契约。
   full-GPU bootstrap 只能接受 `ok=true` 且 `measured=true` 的输出；输出要经 tokenizer 复核为不超过 16 tokens。失败时不得留下可被 replay 读取的 canonical fixture/manifest。正式 probe 固定 6 sessions x 5 turns = 30 requests，不能由通用 `max_requests` 截断。

3. 为三类 profile 完成真实 GPU 集成验证。
   需要覆盖 full、4 个 KIVI、3 个 H2O 的有效 canonical history suffix reuse，以及错误 hash、错误 turn、错误 profile、非前缀 prompt 的 fail-closed 行为。重点确认 H2O logical sequence state 与 KIVI 完整 cached token IDs 在真实 persistent worker 中没有被重建或静默退回 full prompt。

4. 完成外部标注仓库的 preflight/merge/manifest 工作。
   目标仓库为 `../TailGuardKV-labeling`。候选阶段每段 assistant history 使用 tokenizer-derived、最多 16-token placeholder；bootstrap 后必须用真实 canonical fixture 再检。batch manifest 仍需覆盖 `canonical_history_mode`、schema version、bootstrap table hashes、probe ID、`diagnostic_only`。
   merge 必须拒绝 OOM partial rows、缺任一 8-profile 的行和 hash 不一致的行；正式 importer 必须检查 canonical metadata、manifest hash、48 sessions x 5 turns 的分布和来源唯一性。数据任务保持 LongBench-first、RAGhot QA-only、Summary LongBench-only；XSum 仅用于 quality。

5. 重建并验证正式质量 fixture。
   原 `data/fixtures/baseline_quality_external.jsonl` 与 `baseline_session_external.jsonl` 已删除，不能恢复旧版本。质量 fixture 必须在导入前通过 180-row contract：KIVI/H2O/low-risk 各 60 条、calibration/eval 各 90 条，且 8-profile 测量可完整覆盖。只在 `baseline_quality_signal_gate.passed=true` 后，才允许构造或启动正式 session probe。

6. 完成正式 session fixture 和 gate。
   probe 成功后才可构造 12-session/60-request hybrid batch；最终需累计 48 sessions/240 requests，并使每个 `risk x task x split` 有 4 sessions。导入前运行 `--validate-only`，再启动 session measured smoke。仅当 trace semantics、risk signal 和 quality gate 全部通过时，才输出可比较的 policy 表。

7. 测试与结果验收尚未完成。
   至少需要运行主仓库完整 `pytest -q`、标注仓库完整测试、focused canonical/runtime/supervisor tests、fixture/manifest hash 与分布检查。运行中的 GPU job 结束前，不要同时启动会竞争 GPU 的测试或 smoke。

## 运行约定

所有脚本和实验都要异步启动；进程结束或报错时再唤醒会话，不在运行期间轮询。GPU 任务严格串行，只使用 `CUDA_VISIBLE_DEVICES=0,1`，并以 `nohup + setsid + flock` 启动。当前标准环境为 `tailguardkv-base`，不要依赖 shell 默认的 `python`；persistent profile worker 会按现有 runtime 配置进入其所需环境（当前可见为 `edgekv-h2o`）。

推荐的 diagnostic 启动形式如下。先确认锁未被占用和上一任务已经退出；不要绕过锁。

```bash
mkdir -p out/logs out/locks
nohup setsid flock -n out/locks/diagnostic_session27_gpu01.lock \
  env CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output --cwd "$PWD" -n tailguardkv-base \
  python scripts/run_diagnostic_session_batches.py \
    --fixture data/fixtures/diagnostic_session27.jsonl \
    --config configs/pilot_diagnostic_session27.yaml \
    --run-root out/diagnostic_session27_batches_YYYYMMDD_HHMMSS \
    --sessions-per-batch 3 \
  > out/logs/diagnostic_session27_batched_YYYYMMDD_HHMMSS.nohup.log 2>&1 < /dev/null &
```

记录 shell 返回的 PID、锁路径、输入限制、fixture/manifest hash 和日志路径。正式 quality/session smoke 同样使用 [scripts/run_pilot_measured_async.sh](scripts/run_pilot_measured_async.sh)，并保留 PID 文件；quality 未通过时不得启动 session。

## 下一会话的最短路径

1. 先为 `materialize_session_batches` 增加明确的 session-group 输入，并写测试：每组至少两个完整 session、全局 arrival 顺序保持、不得把 session 截断或重复。不要使用 `sessions_per_batch=1`。
2. 用上次 profile 表的 resident/peak 证据构造两 session 压力配对；`011`、`015`、`023` 不得分别与 `009`、`017`、`021` 同组。新的 fixture/config/manifest 必须使用新 run root，保留 `diagnostic_only: true`。
3. 异步启动后，等待自然结束；读取顶层 `supervisor_manifest.json`。只有每个 batch 的 profile coverage 完整、无 OOM partial，且合并目录存在时，才可把 merged CSV 用作 runtime 诊断覆盖证据。trace/risk gate 失败仍不得生成正式 policy/baseline 结论。
4. diagnostic 链路稳定后，才回到正式 canonical supervisor、外部 fixture、quality gate 与 session gate 工作。不要直接把 diagnostic session27 升格为正式 baseline。
