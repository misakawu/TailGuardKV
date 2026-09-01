# TailGuardKV

TailGuardKV 用来验证 KV cache 管理策略在 edge LLM serving 场景下的质量、显存和时延取舍。当前仓库已经不是单纯的论文规划目录，现阶段重点是两条 smoke 轨道：

- `baseline_quality`：对带参考答案的单轮请求做质量比较
- `baseline_session`：对多轮会话做 resident、budget 和 backend pressure 语义检查

`2026-08` 这一轮里，LongBench 和 ShareGPT 的筛选、测量、标注都放在仓库外工作区，主仓库只负责消费外部夹具和跑实验。

## 当前目录

- [`configs/`](./configs/)：实验入口配置
- [`data/fixtures/`](./data/fixtures/)：仓库内固定夹具位置
- [`docs/status/`](./docs/status/)：执行记录和阶段状态
- [`metrics/`](./metrics/)：质量指标计算
- [`profiles/`](./profiles/)：full / KIVI / H2O runtime 适配
- [`run_util/`](./run_util/)：profile 构建、policy replay、汇总逻辑
- [`scripts/`](./scripts/)：导入、生成、启动脚本
- [`tests/`](./tests/)：数据集、配置和回归测试
- [`论文规划/`](./论文规划/)：研究与论文规划文档

## 当前入口配置

目前常用入口不是“三份配置”，而是按用途分开：

- [`configs/pilot.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot.yaml)  
  主 baseline quality 配置，消费仓库内标准质量夹具。
- [`configs/pilot_50.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_50.yaml)  
  小规模 quality 预检配置。
- [`configs/pilot_session_trace.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_session_trace.yaml)  
  session trace 语义检查配置。
- [`configs/pilot_sharegpt.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_sharegpt.yaml)  
  ShareGPT session/cache 诊断配置。
- [`configs/pilot_external_baseline_quality.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_external_baseline_quality.yaml)  
  外部 LongBench 标注夹具的 baseline quality 入口。
- [`configs/pilot_external_baseline_session.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_external_baseline_session.yaml)  
  外部 ShareGPT 标注夹具的 baseline session 入口。

## 外部标注工作区

外部工作区路径：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling`

这个工作区负责四类事情：

- 生成 LongBench 候选
- 生成 ShareGPT session 候选
- 调主仓库 `run_util.build_profile_table` 做 measured profiling
- 导出主仓库可直接消费的 `baseline_quality` / `baseline_session` 夹具

主仓库不再承载 LongBench / ShareGPT 的数据准备逻辑，只保留导入、校验和实验消费入口。

## 当前数据集状态

### LongBench 外部标注集

已完成外部测量和导出，产物在：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_quality_external.jsonl`
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/manifests/baseline_quality_external_manifest.json`

截至 `2026-08-12` 已确认统计：

- 总数 `100`
- 风险分布：`kivi_sensitive=20`、`h2o_sensitive=20`、`low_risk=60`
- 任务分布：`code=42`、`qa=38`、`summary=20`
- split 分布：`calibration=50`、`eval=50`

这里没有继续硬追计划里的 `60 / 60 / 60`。真实测量结果不支持那个目标，当前导出策略已经改成按可用样本收缩敏感组，并用 `tie_sensitive` 回填。

### ShareGPT 外部标注集

当前还在全量测量阶段，产物还没导出完成。

已确认状态如下：

- 候选集：`36` 个 session，`164` 个 turn
- 候选限制：`max_turns=5`、`max_prompt_chars=1024`、`max_effective_prompt_chars=5000`
- 当前测量表：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/sharegpt_session_candidates_profiles.csv`
- 截至 `2026-08-12 15:57`，测量表已写入 `696` 行
- `baseline_session_external.jsonl` 还没有生成

这条线前面已经修过一次 OOM。问题不是单条 prompt，而是多轮 history 累加后的 effective prompt 过长，所以现在按 effective prompt 长度先筛 session，再做全量测量。

## 外部夹具导入

导入脚本是 [`scripts/import_external_fixtures.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/scripts/import_external_fixtures.py)。

只校验 `baseline_quality`：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_quality \
  --input /path/to/baseline_quality.jsonl \
  --validate-only
```

导入 `baseline_quality`：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_quality \
  --input /path/to/baseline_quality.jsonl
```

只校验 `baseline_session`：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_session \
  --input /path/to/baseline_session.jsonl \
  --validate-only
```

导入 `baseline_session`：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_session \
  --input /path/to/baseline_session.jsonl
```

默认导入位置：

- `baseline_quality` -> `data/fixtures/baseline_quality_external.jsonl`
- `baseline_session` -> `data/fixtures/baseline_session_external.jsonl`

当前校验规则包括：

- `baseline_quality` 必须带 `request_id`、`task`、`prompt`、`reference`、`metadata`
- `metadata` 至少带 `source`、`source_dataset`、`split`、`risk_family`
- `baseline_quality` 必须覆盖有效的 `qa / summary / code` 子集，并同时出现 `calibration / eval`
- `baseline_session` 必须带 `session_id`、连续 `turn_index`、全局递增 `arrival_index`
- `baseline_session` 至少包含一个多轮会话和可交错 session
- 两类夹具都要求出现 `kivi_sensitive`、`h2o_sensitive`、`low_risk`

## 运行方式

主 smoke 入口仍然是 `pilot-smoke-measured`：

```bash
conda run -n tailguardkv-base \
  python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
```

跑外部 baseline 夹具时，直接用：

```bash
bash scripts/run_external_baseline_smoke.sh quality
bash scripts/run_external_baseline_smoke.sh session
```

这两个命令分别调用：

- `configs/pilot_external_baseline_quality.yaml`
- `configs/pilot_external_baseline_session.yaml`

## 当前实现边界

当前实现已经支持：

- `baseline_quality` 和 `baseline_session` 双轨实验类型
- 外部夹具导入和 schema 校验
- `code` 任务质量指标支持
- session 轨道的 resident / budget / backend pressure 硬门禁

当前还在推进中的部分：

- ShareGPT 全量测量收尾
- `baseline_session_external.jsonl` 导出
- 两套外部夹具导入主仓库后的正式 smoke 跑表

## 验证

这轮改动里已经补过并跑过的测试包括：

```bash
pytest tests/test_external_fixture_import.py tests/test_pilot_dataset.py tests/test_pilot_session_trace_dataset.py -q
pytest tests/test_external_baseline_configs.py tests/test_code_task_support.py -q
pytest tests/test_longbench_label_strategy.py -q
pytest tests/test_sharegpt_labeling_strategy.py -q
```

执行过程和分钟级进展见：

- [`docs/status/2026-08-12_baseline_smoke_execution_log.md`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/docs/status/2026-08-12_baseline_smoke_execution_log.md)
