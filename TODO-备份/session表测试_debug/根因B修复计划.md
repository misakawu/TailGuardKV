# 根因 B 修复计划

## 目标

修复 `baseline_session` 的 calibration / eval split 失衡问题，避免高风险样本在切分时被系统性放进 calibration，导致 eval 侧尾部风险被稀释。

## 方案

1. 保持 `run_util/run_policies.py` 不改，继续只回放 `eval`。
2. 修改 `scripts/generate_pilot_session_trace_requests.py` 为两阶段生成：
   - 先生成全量 profile 实测矩阵。
   - 再按 `(task, length_bucket)` 聚合，用 `max(lossy)` 风险分数做 60/40 分层切分。
3. 新增切分验证门：
   - 每个 `(group, profile)` 输出 cal / eval 的 CDF、KS、p95 差异。
   - 任一组合 `KS > 0.1` 直接失败。
4. 固化一份 `data/golden/split_validation_v1.html`，作为后续 policy sweep 前置门禁。

## 验收

- `baseline_session` 的 eval 里保留非零尾部风险样本。
- `qa` / `summary` 及 KIVI / H2O 主流档位不再稳定全 0。
- split 验证门在失衡时能阻断后续实验。

## 计划执行顺序

1. 先补测试，锁定当前失衡行为。
2. 再改生成脚本与验证逻辑。
3. 最后跑验证并把完成项追加到本文末尾。

## 已完成任务

- 已在 `scripts/generate_pilot_session_trace_requests.py` 增加风险查表、按 `(task, length_bucket)` 分组的 60/40 分层 split，以及 `split_score` 标注。
- 已支持从 measured profile CSV 构建 `max(lossy)` 风险查表；未提供测量表时，脚本会退回 metadata 风险代理分数，保证 fixture 仍可重复生成。
- 已新增 split 验证门，输出每个 `(group, profile)` 的 cal / eval CDF、KS 与 `p95` 差异，并在 `KS > 0.1` 时返回失败。
- 已固化 `data/golden/split_validation_v1.html`，作为 split 校验的落盘报告。
- 已重生成 `data/fixtures/pilot_session_trace_requests.jsonl`，让当前 fixture 跟随新的分层逻辑。
- 已在 `run_util/experiment.py` 和 `configs/pilot_session_trace.yaml` 接入 split validation 强制门禁，正式 `baseline_session` 会在 policy sweep 前先校验 split，再校验 trace quality。
- 已在 `tests/test_pilot_session_trace_dataset.py` 增加两类回归测试：
  - `eval` 侧保留尾部风险样本；
  - `KS` 超阈值时验证门直接失败。
- 已在 `tests/test_session_trace_pressure.py` 增加实验入口回归：
  - split gate 通过时可继续进入 policy sweep；
  - split gate 失败时会阻断后续 policy sweep。

## 验证结果

- `pytest -q tests/test_pilot_session_trace_dataset.py tests/test_session_trace_pressure.py -q`
- `python scripts/generate_pilot_session_trace_requests.py`
- `python -m py_compile scripts/generate_pilot_session_trace_requests.py scripts/validate_trace_quality.py`

## 未实现任务

- 还未按原计划完成“先用现有 `pilot_session_trace_requests.jsonl` 复跑一次，确认 eval 不再出现高风险全落 calibration”的实测复盘记录。现在完成的是脚本级和测试级验证，不是一次完整 baseline 运行复核。
- 还未补充针对真实 `policy_tables/*` 输出的回归校验，因此“lossy policy 不再稳定全 0”的结论目前还没有写入新的实测结果文档。

## 进度记录

### Tuesday, August 11, 2026

- 已完成根因 B 的第一阶段修复：两阶段 split、split gate、HTML 报告、实验入口门禁、以及相关回归测试均已落地。
- 已完成本地验证：
  - `pytest -q tests/test_pilot_session_trace_dataset.py tests/test_session_trace_pressure.py -q`
  - `python -m py_compile scripts/generate_pilot_session_trace_requests.py run_util/experiment.py scripts/validate_trace_quality.py`
  - `python scripts/generate_pilot_session_trace_requests.py --measurements out/20260811_075832_pilot_session_trace/profile_tables/pilot_session_trace_measured_profiles.csv`
- 已跑过第一次真实 `pilot_session_trace` 实验，输出目录为 `out/20260811_075832_pilot_session_trace`。
- 第一次真实实验的结论：
  - profile 测量链路是通的；
  - `split gate` 失败；
  - 失败根因已定位为 `assign_stratified_splits()` 旧逻辑采用“按风险排序后交替分配，再从尾部补齐 60/40”，会把尾部样本整体挪动，导致 `qa/medium` 与 `summary/long` 的若干 profile 出现 `KS > 0.1`。
- 已将 split 分配器改为“按目标 quota 均匀铺开”的 deterministic 分配，并补齐对应回归测试。
- 之后已再次运行真实实验，输出目录为 `out/20260811_093226_pilot_session_trace`。
- 第二次真实实验的结论：
  - `split gate` 已不再是主要失败点；
  - 实验进入到更后面的 `quality gate`；
  - 当前失败原因为 `quality gate` 未通过，`mean_quality_loss > 0.02` 的 profile 数量为 0，主流 KIVI / H2O 档位均未达到阈值。
- 在第二次真实实验中还发现新的性能问题：
  - `profile_chunk_size` 默认仅为 10，导致 480 个请求被切成大量小批次；
  - `profiles.persistent_worker` 为单进程高 CPU 串行执行；
  - GPU 利用率偏低；
  - KIVI / H2O 阶段虽然配置了 GPU2，但由于 persistent worker 生命周期按 adapter 整体复用，设备绑定没有真正切换。
- 性能优化已完成第一步修复：
  - `configs/pilot_session_trace.yaml` 已增加 `profile_chunk_size: 24`；
  - `full` / `kivi` / `h2o` 已补充显式设备绑定配置；
  - `profiles/base.py` 已让 persistent worker 启动时继承 `CUDA_VISIBLE_DEVICES`；
  - `run_util/build_profile_table.py` 已改为在设备绑定变化时重启 persistent worker。
- 已新增并通过一条回归测试，锁定“设备绑定变化时不能复用同一个 persistent worker”的行为：
  - `pytest -q tests/test_tailguard_core.py -q -k 'persistent_worker_when_cuda_binding_changes or chunks_profile_many_and_writes_incrementally'`
- 为了快速验证性能修复，已新增缩小数据量的快速配置 `configs/pilot_session_trace_quick.yaml`。
- 当前后台仍有一条快速真实实验在运行：
  - 启动命令：`python run_experiment.py pilot-smoke-measured --config configs/pilot_session_trace_quick.yaml`
  - 进程 PID：`25651`
  - 当前已验证到的现象：
    - `full_gpu` 阶段能稳定按 `24` 条请求为一个 chunk 推进；
    - GPU0 / GPU1 利用率较之前已有提升；
    - 尚未拿到这条快速实验的最终 summary / gate 结论。

- 这条快速真实实验现已结束，输出目录为 `out/20260811_151358_pilot_session_trace_quick`。
- quick 实验最终结论：
  - 没有进入 split gate / quality gate / policy sweep；
  - 失败发生在 profile 测量阶段；
  - `kivi_2bit_residual32` 在切换到 KIVI profile 时出现 `torch.OutOfMemoryError`；
  - 错误栈显示模型加载仍然发生在 GPU 0，而不是预期的 GPU 2；
  - 这说明“设备绑定变化时重启 persistent worker”的修复还不够，当前 runtime 仍未把 KIVI 单卡实验真正隔离到 GPU 2。
- quick 实验的证据文件：
  - summary: `out/20260811_151358_pilot_session_trace_quick/policy_tables/pilot_session_trace_quick_summary.csv`
  - failed chunks: `out/20260811_151358_pilot_session_trace_quick/profile_tables/pilot_session_trace_quick_profiles_failed_chunks.csv`
- 已完成设备绑定语义的第二轮收敛修复：
  - `profiles/qwen2_kv_runtime.py` 现已在 persistent worker 的 `init` / `run_batch` 返回中显式携带绑定诊断；
  - `profiles/qwen2_runtime_common.py` / `profiles/qwen2_kivi_runtime.py` / `profiles/qwen2_h2o_runtime.py` 现已把 `worker_cuda_visible_devices`、`runtime_cuda_visible_devices`、`runtime_visible_device_count`、`runtime_device_strategy` 写入 measured result；
  - 这一步把“逻辑 device strategy”和“进程实际可见设备集合”拆开记录，不再仅凭 OOM 栈里的 `GPU 0` 文案做物理卡误绑定判断。
- 已按 smoke 交付口径统一配置：
  - `configs/pilot_session_trace_quick.yaml` 已将 `full` / `kivi` / `h2o` 全部切回 `balanced_two_gpu` + `CUDA_VISIBLE_DEVICES=0,1`；
  - `configs/pilot_session_trace.yaml` 已做相同统一，避免当前 11GB 单卡环境继续卡在不稳定的 `single_gpu` measured runtime。
- 已补齐并通过新的绑定回归验证：
  - `tests/test_persistent_qwen2_worker.py` 新增 persistent worker 绑定诊断测试；
  - `tests/test_tailguard_core.py` 新增 measurement extra 绑定字段透传测试；
  - 本地验证已通过：
    - `pytest tests/test_pilot_session_trace_dataset.py tests/test_session_trace_pressure.py tests/test_persistent_qwen2_worker.py tests/test_tailguard_core.py -k 'persistent_worker or binding_diagnostics or split or quality_gate or session_trace'`
    - `python -m py_compile run_util/build_profile_table.py profiles/base.py profiles/qwen2_runtime_common.py profiles/qwen2_kv_runtime.py profiles/qwen2_kivi_runtime.py profiles/qwen2_h2o_runtime.py run_util/experiment.py scripts/generate_pilot_session_trace_requests.py scripts/validate_trace_quality.py`
- 已于 `Tuesday, August 11, 2026` 重跑 quick smoke，输出目录为 `out/20260811_154041_pilot_session_trace_quick`。
- 这次 quick run 的结论：
  - profile 阶段已完整跑通，未再出现 profile load 阶段 OOM；
  - `full_gpu` / `kivi_2bit_residual32` / `h2o_heavy15_recent15` 共 1440 条 measured rows 全部落盘；
  - 新绑定诊断字段已完整写入 CSV：`extra_runtime_cuda_visible_devices`、`extra_runtime_device_strategy`、`extra_runtime_visible_device_count`、`extra_worker_cuda_visible_devices`、`extra_worker_device_strategy` 均覆盖 1440/1440 行；
  - split gate 已通过，并落盘 `out/20260811_154041_pilot_session_trace_quick/profile_tables/pilot_session_trace_quick_split_validation.html`；
  - quick 最终停在 quality gate，而不是设备绑定阶段；
  - 当前阻塞原因已收敛为质量门本身：`mean_quality_loss > 0.02` 的 profile 仍为 0，`kivi_2bit_residual32=0.003237`、`h2o_heavy15_recent15=0.009738`，因此不满足现有 `baseline_session` smoke 的质量阈值。
- 随后已按“补更强 lossy profile，但仅接入 session 轨道”的方向完成配置扩容：
  - `profiles/kivi.py` 新增 `kivi_4bit_residual16`、`kivi_2bit_residual16`；
  - `profiles/h2o.py` 新增 `h2o_heavy05_recent05`、`h2o_heavy08_recent08`；
  - `configs/pilot_session_trace.yaml` 已扩到 12 个 profile；
  - `scripts/generate_pilot_session_trace_requests.py` 的 `MAINSTREAM_KIVI_PROFILES` / `MAINSTREAM_H2O_PROFILES` 已同步纳入新强档位；
  - `configs/pilot.yaml` 维持不变，baseline quality 轨道未跟随扩容。
- 在第一次完整 12 档 run `out/20260811_185519_pilot_session_trace` 中，新的 `kivi_*_residual16` 首次实测暴露出 runtime 兼容性问题：
  - run 在 `full_gpu` 完成后停在 `kivi_4bit_residual16`；
  - 失败表现为 profile 阶段 `AssertionError`，未进入 split gate / quality gate；
  - 根因已定位为 `profiles/qwen2_kivi_runtime.py` 中存在 `self.residual_length % self.group_size == 0` 的断言，而 `residual16` 仍沿用了默认 `kivi_group_size=32`。
- 已完成这条 KIVI residual16 兼容性修复：
  - `kivi_4bit_residual16` 与 `kivi_2bit_residual16` 现已显式写入 `kivi_group_size=16`；
  - 已补并通过回归测试，锁定 `residual16` profile metadata 必须包含 `kivi_group_size=16`。
- 已于 `Tuesday, August 11, 2026` 重新运行完整 session smoke，输出目录为 `out/20260811_190337_pilot_session_trace`。
- 这次完整 12 档 run 的结论：
  - 12 个 profile 共 `5760` 条 measured rows 已全部落盘，说明新增 KIVI / H2O 强档位均已真实跑通；
  - `split gate` 再次失败，因此实验仍未进入 policy sweep；
  - split 失败证据已落盘 `out/20260811_190337_pilot_session_trace/profile_tables/pilot_session_trace_split_validation.html`；
  - 当前可确认的超阈值项至少包括：
    - `qa/medium h2o_heavy05_recent05` 的 `KS=0.1458 > 0.10`；
    - `qa/medium kivi_4bit_residual16` 的 `KS=0.1042 > 0.10`。
- 已对 `out/20260811_190337_pilot_session_trace/profile_tables/pilot_session_trace_measured_profiles.csv` 手动补跑 quality gate 校验，输出为 `out/20260811_190337_pilot_session_trace/profile_tables/pilot_session_trace_quality_gate.json`。
- 这次完整 12 档 run 的质量门结论同样未通过：
  - `covered_tasks` 仍已覆盖 `qa` / `summary`；
  - 但 `mean_quality_loss > 0.02` 的 profile 只有 1 个，即 `h2o_heavy05_recent05=0.026044`；
  - 当前仍缺少满足阈值的 KIVI 主流档位；
  - 本次完整 run 的 KIVI profile 均值最高仍只有 `kivi_2bit_residual32=0.003237`；
  - 因此即使忽略 split gate，当前 measured 结果也还不足以通过现有 quality gate。

### 下一步

- 下一步先不跑 `configs/pilot.yaml`。眼下还没到那一步。
- 先处理 `out/20260811_190337_pilot_session_trace` 这次完整 run 暴露出的 `split gate` 回退问题。这里不能靠猜，得先把原因查清楚。
- 第一件事是回看 `out/20260811_190337_pilot_session_trace/profile_tables/pilot_session_trace_split_validation.html`，重点盯住这两个超阈值项：
  - `qa/medium h2o_heavy05_recent05` 的 `KS=0.1458`
  - `qa/medium kivi_4bit_residual16` 的 `KS=0.1042`
- 要查的不是“它又失败了”，而是这两个 profile 为什么会把 calibration / eval 分布重新拉开。需要顺着数据往前看：
  - 这些请求在 fixture 里的 `risk_family`、`risk_profiles`、`split_score` 是怎么分布的；
  - 新增强档位接入以后，`assign_stratified_splits()` 现在的分层逻辑在哪个环节把尾部样本压偏了；
  - 旧主流档位通过、新强档位回退，差异到底来自 profile 集合变化，还是来自分组粒度本身太粗。
- 这一步收敛后，再做最小改动重生成 `data/fixtures/pilot_session_trace_requests.jsonl`，然后只重跑 `configs/pilot_session_trace.yaml`，先看 split gate 能不能重新稳定通过。
- 只有 split gate 重新站稳了，才讨论 quality gate。现在质量门的问题已经很具体了，不需要再泛泛而谈：
  - H2O 这边已经有一个强档位过线，`h2o_heavy05_recent05=0.026044`；
  - KIVI 这边还是不够，当前最高也只有 `kivi_2bit_residual32=0.003237`。
- 所以 quality gate 后面其实只剩两个方向：
  - 继续补更强的 KIVI lossy 档位；
  - 或者下调 `baseline_session` 当前的 quality gate 期望。
- 在 `split gate` 和 `quality gate` 都过之前，不推进 `python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml`。

## 完整 Session Smoke 之后的任务与 Baseline Smoke 表获取方式

### `configs/pilot_session_trace.yaml` 跑完之后做什么

- 先把这里的判断条件说清楚。
- 只有完整 `baseline_session` measured smoke 同时满足下面两件事，才进入 `configs/pilot.yaml`：
  - `split gate` 通过；
  - `quality gate` 也通过。
- 只过了其中一个都不够。比如现在这轮完整 12 档 run，profile 已经全跑通，但 split gate 和 quality gate 都没收住，所以还不能往下走。
- 如果后面某次完整 run 的 split gate 通过、quality gate 仍失败，那么后续任务就不是做表，而是继续收敛质量门：
  - 要么调整 `baseline_session` 现在的 quality gate 期望；
  - 要么继续补更强的 lossy profile，再重跑完整 session smoke。

### `baseline smoke表` 不是从哪里来

- `baseline smoke表` 不是 `configs/pilot_session_trace.yaml` 直接产出的。
- `configs/pilot_session_trace.yaml` 的任务更像一道前置验收。它先回答一个问题：`baseline_session` 这条 measured smoke 链路现在到底能不能跑通，跑出来的 gate 结果能不能解释。
- 真正要交到论文规划 `2026-08-01` 到 `2026-08-15` 这段工作的 `baseline smoke表`，还是要来自 `configs/pilot.yaml` 的新跑结果。

### 如何获取 `baseline smoke表`

1. 运行：
   - `python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml`
2. 在新的 `out/.../` 运行目录中收集主要产物：
   - `profile_tables/*profiles.csv`
   - `policy_tables/*policy.csv`
   - `policy_tables/*summary.csv`
3. 以 `configs/pilot.yaml` 这次新跑结果为唯一口径，提取 5 个 baseline 的 smoke 结果，整理成论文规划中的 `baseline smoke表`。
4. 更新状态文档：
   - `docs/status/2026-08-15_baseline_smoke_status.md`

### 交付口径要求

- `baseline smoke表` 必须明确写清楚：它基于当前 `pilot main dataset`。
- 不再沿用旧 fixture 的历史结果。
- 如果旧表或旧状态文档还在引用历史 fixture 结果，要明确标注它已经被当前 `configs/pilot.yaml` 的新跑结果替换。
