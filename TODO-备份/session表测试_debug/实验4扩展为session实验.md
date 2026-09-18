# 第四次实验扩展为 25-Session Diagnostic 实验

## 目标与边界

- 延续《第四次小规模实验.md》的预算压力、隐私参数和五条 baseline 比较；其历史参考结果位于 `out-备份/过期结果/second_memory_pressure_20260912/`。旧结果来自 10 请求、两个 session 的 `measured_replay`，仅作异口径诊断参照，不得与新的 online 实测数值合并。
- 本轮输入固定为 `data/fixtures/diagnostic_session27_exclude_batch7.jsonl`：移除不可用 batch7 后实际为 **25 个完整 session、125 个请求、每 session 五个 turn**，不得再记作 27/135。
- 48-session canonical 路线已放弃。本轮始终 `diagnostic_only=true`；不满足 canonical bootstrap/replay、512-token final-form 和正式风险分布门禁。不能作为论文质量保证、正式 baseline 或 TailGuard/Oracle 对比。
- 独立输出到新的 `out/session25_fourth_diagnostic_*` 目录；不得复用或覆盖现有工作树和历史结果。

## 实验设计

- 固定相同的 fixture、到达顺序、八个 profile、五条 baseline 和每个 seed 内的校准划分；跨 turn 保持 backend 会话状态。配置明确 `baseline_session`、`online_qwen`、`session_diagnostic` 和 `diagnostic_only=true`。
- online backend 的 KV 预算口径与第四次实验的 15–96 MiB `measured_replay` 不同：先测量全量 profile，再按 full profile 无淘汰 occupancy 的 p25、p50、p75、p90 推导四档 B。旧 15–96 MiB 和 256 MiB 失败档仅为历史背景，不能直接套在在线运行上，也不能算作本轮失败。
- 保留第四次实验 ε、δ `{0.05, 0.10}` 的 2×2 网格；本轮预先固定 seed 为 `20260906、20260907、20260908`。历史结果只能核实 `20260906`，不得宣称另外两个 seed 与旧实验一致。同一 seed 内各 B/ε/δ/policy 组合复用测量和划分。每 seed 4 B × 4 参数组合 × 5 baseline = 80 cell，总共 **240 cell**，成功 cell 每个有 125 条请求记录。运行前先用两个完整交错 session 检查多 profile 测量；policy 事件链留待运行后审计。
- 如现有运行器仅支持一个 seed 或单一 ε/δ，须先补齐可审计参数传递、分 seed 输出及组合清单；不能用默认参数冒充全网格完成。暂时性失败可有限重试并保留历史，确定性 GPU 故障不得盲目重试。

## 分阶段执行

1. 校验 125 条输入、25 个唯一 session、唯一 ID、每 session turn 0–4 连续，记录输入哈希与来源。原始 `arrival_index` 因移除 batch7 留有 10 个空洞，必须严格递增且唯一；分批后在各批内部重编号、保留相对到达顺序，不改写原 fixture。
2. 校验模型/依赖/设备及诊断配置，先跑两个完整交错 session 的多 profile preflight（`baseline_session` 拒绝仅含一个 session 的输入）；检查 profile 测量覆盖及 resident、TTFT 字段。`--profile-only` 不产生 policy trace，持久/全局 resident 与 backend 事件链须在 policy 运行后审计；preflight 失败则不启动全量任务。
3. 按完整 session 分批落盘 profile 测量，支持续跑，验收 125×8=1000 个有效 request/profile 组合；失败批次保留错误、退出码和完成数，不静默剔除。
4. 推导四档 B，按 seed × B × ε × δ 独立执行 online policy，保存五条 baseline 的逐 turn 记录；全部脚本或程序通过 `nohup` + `setsid` 异步启动，保留 PID、日志和退出码；运行期间不轮询，由用户通知结束后进入下一阶段。
5. 先按 session 汇总 turn，再按 seed × B × ε × δ × baseline 聚合；审计计划/实际 cell、session、request、turn 以及成功/失败/OOM/超时/不完整数。失败 cell 不混入成功指标。
6. Gate 通过后产出诊断 CSV、图表和中文分析报告；未通过时仅产出 `diagnostic_only=true` 问题清单，不伪造正式结论。

首次门禁由于误要求删除批次后的索引连续，已失败并留存 `out/session25_fourth_diagnostic_20260914/preflight.exit` 和 `logs/preflight.launcher.log`。第二次单 session 预检因 `baseline_session` 的至少两个 session 约束失败，保留 `preflight_attempt2.exit` 和 `logs/preflight.attempt2.log`，不修改输入或放宽约束。第三次改用两个交错 session，保留 `preflight_attempt3.exit` 和 `logs/preflight.attempt3.log`。

## 2026-09-14 交接状态

- 已核实第三次预检的 `preflight_attempt3.exit` 为 **0**；`logs/preflight.attempt3.log` 中定向测试 **59 passed**。`preflight_attempt3.json` 记录原 fixture 为 **25 session / 125 request**、10 个到达索引空洞，SHA-256 为 `d89dbf9fc3eadf689ea7bb4ffc83a77ff12a64baec16066d9fd662e41f92451e`。
- `two_session_preflight_attempt3/supervisor_manifest.json` 的 `batch000` 为两个 session、10 请求 × 8 profile = **80/80** 个已验证 profile 对，`mergeable=true`、`errors=[]`、`returncode=0`；其 profile CSV 在 `two_session_preflight_attempt3/batch_outputs/batch000/profile_tables/session25_fourth_profiles_batch000.csv`。此为 `--profile-only`：`trace_csv=null`，不代表 policy 事件链、持久 resident 或全量指标已通过。
- **全量尚未启动**：`full.exit` 尚不存在；下一会话从 `scripts/run_session25_fourth_full.sh` 开始。脚本检查 `preflight_attempt3.exit=0`，依次跑定向测试、12 个批次的 125×8 profile 测量、预算推导与 240 个 policy cell；实验仍须保持 `diagnostic_only=true`。不要复用预检目录作为全量结果，不要删除前两次失败记录。
- 在本工作树根目录异步启动（先检查 `out/session25_fourth_diagnostic_20260914/full.exit` 和 `full.pid`，避免重复启动）：

  ```bash
  run_root="$PWD/out/session25_fourth_diagnostic_20260914"
  mkdir -p "$run_root/logs"
  setsid nohup bash scripts/run_session25_fourth_full.sh > "$run_root/logs/full.launcher.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_root/full.pid"
  ```

- 启动后不轮询；用户通知进程结束后核对 `full.exit`、`logs/full.launcher.log`、`full/supervisor_manifest.json`、profile 1000 对、预算出处和 `full/fourth_grid_status.json` 的 240 cell 清单。按成功/失败分别审计请求与事件语义，再绘图并撰写中文诊断报告；失败或缺项必须留痕，不宣称实验完成。

## 指标与可视化

- 输出质量风险（平均/P95 loss、violation rate）、KV/session/global resident MiB、实际 backend budget hit、策略预过滤率、动作分布、evict/restore/recompute/queue 事件及耗时、mean/P95/P99 TTFT 和完整性计数。
- backend `budget_hit_rate` 不混入 policy filter；不适用字段用空值或 `N/A`。exact profile 也须有 resident 指标。标明 baseline 退化或参数未生效的诊断缺陷，不为了预设排序修改策略。
- 图表覆盖质量/内存/时延、预算与 resident、backend 与 policy 事件、动作分布、turn 进度、profile × B 热力图、Pareto 和失败分布；区分 seed、ε/δ 与样本数，禁止把失败 cell 连成性能曲线。
- 中文报告回答五条 baseline 是否拉开、质量/内存/尾时延取舍、backend 节省还是预过滤、事件造成的 TTFT、P95/P99 风险与失败集中区；必须注明历史结果异口径、本轮仅为 diagnostic。

## 验收条件

- fixture 25/125，profile 1000 对，计划 240 cell 均有成功或失败记录；成功 cell 覆盖全部 25 session / 125 turn。
- 预算和 ε/δ/seed 确实进入执行参数和逐请求记录；成功指标能从原始行重算，跨 turn 事件链自洽，表/图样本数一致。
- 完整性或语义 gate 未通过，不声明实验完成，更不能标为正式 baseline；在独立目录保留退出状态和限制。

## 当前执行进度（更新于 2026-09-14）

### 已完成

- 正式输入已固定为 `data/fixtures/diagnostic_session27_exclude_batch7.jsonl`，共 **25 个 session、125 个请求、每个 session 5 个 turn**。
- 原 batch7 已正式放弃，不恢复、不补跑，也不将其部分产物纳入正式汇总；对应排除 `hybrid-session-014` 和 `hybrid-session-015`。
- session backend 主链路已经具备，包括 persistent Qwen backend、跨 turn runtime state、session KV reuse、restore/recompute、resident KV 和 budget pressure 记录。
- policy 主链路已经具备，包括 turn-level action、observed quality、policy provenance、shadow audit 区分和 policy metrics 聚合。
- Diagnostic Batch Supervisor 的 Task 1 已完成，focused tests 为 6 passed；Task 2 已完成，focused tests 为 11 passed。

### 进行中

- Diagnostic Batch Supervisor 的 Task 3 仍为 `in progress`；现有进度台账记录该任务所在 worktree 有未提交修改和 index lock，需要完成收尾与复核。
- 基于排除 batch7 后 25-session fixture 的正式实验产物正在整理，现有目录不能在完整性验收前直接视为最终结果。

### 尚未确认完成

- 尚未确认排除 batch7 后的 25-session、125-request 全量正式重跑已经完整通过。
- 尚未确认 backend 输出满足 expected/observed request 数、session/turn 连续性、profile 覆盖和无重复记录等完整性门禁。
- 尚未确认全部 policy 已基于本轮正式 backend turn records 完成 replay，并生成 per-turn、per-session 和 per-policy 三层汇总。
- 尚未确认 backend 与 policy 的联合最终汇总、运行 manifest、失败与排除报告已经生成并通过验收。
- “经常不能正常结束”的具体根因仍未由完整失败日志确认；目前不能把排除 batch7 等同于退出问题已经修复，也不能声称修复效果已经验证。

### 下一步执行顺序

1. 完成 Diagnostic Batch Supervisor Task 3，处理残留 index lock 和未提交 worktree，并执行 focused tests。
2. 补齐运行终态和完整性门禁，明确区分 `completed`、`failed`、`timeout`、`oom` 与 `interrupted`，避免部分输出被误判为成功。
3. 使用新的 25-session fixture 输出到全新目录正式重跑，禁止混入原 batch7 或旧实验部分产物。
4. 验收 backend 部分，包括请求完整性、session/turn 连续性、KV restore/recompute、resident/budget pressure 和进程正常退出。
5. 基于本轮 backend turn records 完整运行 policy replay，并生成 per-turn、per-session 和 per-policy 汇总及 provenance。
6. 生成 backend/policy 联合 manifest、最终聚合表、失败与排除报告，完成无残留 worker、工作区状态和复现命令检查。

### 当前结论

本轮实验的 **backend 与 policy 代码主链路基本具备，batch7 排除决策和 25-session 输入已经落地**；当前关键缺口是 Supervisor Task 3 收尾、退出可靠性验证、25-session 正式重跑以及 backend/policy 联合汇总验收。在这些项目全部完成前，本实验仍应标记为 **进行中，尚未形成最终可交付结果**。
