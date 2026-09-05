# profile/backend 复查 TODO

来源：复查 `run_profile_test.py --config configs/pilot_50.yaml` 后，对照 `论文规划/王祯祥_论文工作规划.md` 中 0.7.3 的要求：

> 质量 loss、profile latency 和 memory 必须来自实际模型/profile实测，不能使用拍脑袋的解析常数。预实验可以先用 measured-replay backend 做大网格，但必须完成一条 vLLM/LMCache 小 trace 的 smoke test，保证接口能落到真实引擎。

当前 `pilot_50` 结果可以说明 full/KIVI/H2O 跑过真实模型路径，但还不能直接当成 vLLM/LMCache smoke 的验收结果。下面这些问题需要拆开处理。

## 1. 补真实 vLLM/LMCache smoke trace

- 问题：当前 `profiles.registry` 只注册了 `full`、`kivi`、`h2o`。`run_profile_test.py` 没有跑到 vLLM 或 LMCache adapter。
- 位置：
  - `profiles/registry.py`
  - `profiles/full.py`
  - `backends/base.py`
  - `backends/measured_replay.py`
- 现状：`full_gpu` 走 Transformers；KIVI/H2O 走自定义 Qwen2 attention runtime。仓库里有 vLLM/LMCache 环境配置，但没有纳入当前 profile adapter。
- 影响：不满足 0.7.3 里“一条 vLLM/LMCache 小 trace 的 smoke test”的接口验收。
- TODO：
  - 新增最小 `VLLMBackend` 或 `VLLMProfileAdapter`，先只支持 `engine_full_lru` 或等价 full profile。
  - 新增最小 `LMCacheBackend` smoke，至少能启动引擎、跑一条请求、落表。
  - 把 adapter 注册到 `profiles/registry.py`，并加一个单独 smoke config，避免污染当前 KIVI/H2O pilot。
  - 输出字段里明确写 `extra_backend=vllm` 或 `extra_backend=lmcache`。
- 完成标准：
  - `run_profile_test.py` 或独立 smoke runner 能生成至少 1 条 vLLM/LMCache 的 `measured=True` 记录。
  - 失败时不能静默回落到 Transformers。

## 2. 修正 `ttft_ms` 口径

- 问题：现在 `ttft_ms` 实际等于整次生成总时延，不是真正的 first-token latency。
- 位置：
  - `profiles/transformers_runtime.py`
  - `profiles/qwen2_runtime_common.py`
  - `profiles/base.py`
  - `metrics/collector.py`
- 现状：
  - Transformers runtime 返回 `latency_ms=total_ms`、`ttft_ms=total_ms`。
  - Qwen2 KIVI/H2O runtime 也是同样写法。
- 影响：论文里如果报告 p95 TTFT，会把生成 16 token 的总耗时误写成首 token 延迟。
- TODO：
  - 先决定短期口径：如果暂时测不到首 token，就把表头和 summary 改成 `latency_ms`，不要叫 TTFT。
  - 如果继续保留 `ttft_ms`，需要在 runtime 里实现逐 token 或 streamer 计时。
  - 在 summary 里同时保留 `mean_latency_ms` 和 `mean_ttft_ms`，避免混用。
- 完成标准：
  - `ttft_ms` 来自首 token 事件；或者所有报告、summary 和文档都改用 `latency_ms`。
  - 测试覆盖“不能把 total latency 填到 ttft_ms”的情况。

## 3. 拆出 KV cache memory 口径

- 问题：当前 `peak_memory_mib` 是端到端 CUDA peak allocated，不是 profile 的 KV cache 成本。
- 位置：
  - `profiles/transformers_runtime.py`
  - `profiles/qwen2_runtime_common.py`
  - `profiles/kivi_cache.py`
  - `profiles/h2o_cache.py`
  - `metrics/collector.py`
- 现状：memory 来自 `torch.cuda.max_memory_allocated()`。这是真测量，但会被模型权重、临时张量和 allocator 行为盖住。`pilot_50` 里各 profile 的 peak memory 几乎一样，说明这个指标暂时看不出压缩收益。
- 影响：不能用当前 `peak_memory_mib` 证明 KV 压缩节省显存。
- TODO：
  - 增加 `kv_cache_memory_mib` 或 `profile_memory_mib` 字段。
  - KIVI 侧按真实 cache state 统计 full residual、quantized tensor、scale、min 等张量字节数。
  - H2O 侧按保留的 key/value tensor 统计真实物理 cache 字节数。
  - full profile 也统计同口径 KV bytes，作为 baseline。
  - summary 同时报告 `peak_memory_mib` 和 `kv_cache_memory_mib`。
- 完成标准：
  - 同一请求下 full/KIVI/H2O 的 KV memory 有可解释差异。
  - 论文图里明确说明使用的是端到端 peak memory，还是 KV-only memory。

## 4. 把 dry-run 估算常数隔离出正式路径

- 问题：adapter 的 `dry_run` 分支仍有手写比例常数。
- 位置：
  - `profiles/full.py`
  - `profiles/kivi.py`
  - `profiles/h2o.py`
  - `profiles/base.py`
- 现状：这些常数只在 `--dry-run` 下生效，且会写 `measured=False`。当前 `run_profile_test.py` 默认会拒绝非 measured 行，所以这次结果没有被 dry-run 污染。
- 影响：后续如果有人用 `--dry-run` 跑 policy replay，容易把估算表误当实验结果。
- TODO：
  - dry-run 输出文件名和 summary 里加明显标记，例如 `dry_run=true`、`source=synthetic_schema_check`。
  - 正式 config 禁止 `allow_dry_run_replay=True`。
  - README 或实验文档里写清 dry-run 只检查表结构。
  - 可以考虑把 dry-run 常数集中到一个测试 fixture，避免散在 adapter 里。
- 完成标准：
  - `measured=False` 的 profile 表不能进入默认 policy 实验。
  - dry-run 结果不会被默认写到 `out/profile_tables/*measured*` 这类文件名。

## 5. 修正 profile summary 的违例率阈值

- 问题：`MetricCollector.summarize_profiles()` 里 `violation_rate` 固定按 `0.5` 算。
- 位置：
  - `metrics/collector.py`
  - `run_util.build_profile_table`
  - `run_profile_test.py`
- 现状：`configs/pilot_50.yaml` 的 epsilon 是 `0.05` 和 `0.10`，但 profile summary 不读取这些值。
- 影响：summary 里的 profile 违例率不能直接用于 0.7.3 的 SLO 判断。
- TODO：
  - 让 `summarize_profiles()` 接收 epsilon，或输出多个 `violation_rate_eps_*`。
  - `run_profile_test.py` 写 summary 时带上 config 中的 epsilon。
  - 如果只是 profile 表，不做 SLO sweep，就把该列改名为 `violation_rate_eps0p5`，别让读者误解。
- 完成标准：
  - profile summary 的违例率和 config 里的 epsilon 对得上。
  - policy summary 与 profile summary 的 SLO 口径一致。

## 6. 明确 quality loss 的含义

- 问题：quality loss 来自输出文本和 reference/full baseline 的 EM、F1、ROUGE-L 差值，不是人工质量评分，也不是模型内部概率损失。
- 位置：
  - `data_utils.py`
  - `metrics/quality.py`
  - `run_util.build_profile_table`
- 现状：full/exact profile 被置为 `loss=0`；有 reference 时，用 candidate loss 减 baseline loss；没有 reference 时，用 candidate 对 full 输出的文本距离。
- 影响：指标可用，但论文写法必须老实。尤其是 summary 任务用 ROUGE-L，QA 任务目前 `qa_long_context` 不会命中 `select_primary_loss("qa")`，会落到 EM。
- TODO：
  - 检查任务名映射，确认 `qa_long_context` 应该走 F1 还是 EM。
  - 在输出 extra 字段里保留 primary metric 名称，例如 `extra_metric_primary=f1`。
  - 文档中说明 loss 是任务指标差值，不是主观质量。
- 完成标准：
  - 每条 profile row 能看出用了哪个 primary metric。
  - QA/summary 的 loss 口径和论文实验说明一致。

## 7. 把“真实 runtime”和“vLLM/LMCache runtime”在文档里分开

- 问题：当前结果是真实模型执行，但不是 vLLM/LMCache 执行。这个差别容易在实验记录里被写混。
- 位置：
  - `README.md`
  - `README-实验设计总结.md`
  - `run_profile_test.py` 顶部运行说明
  - `out/profile_tables/*_summary.csv`
- TODO：
  - 文档里把 `transformers/qwen2_kv_runtime` 标为当前 smoke 路径。
  - vLLM/LMCache smoke 完成前，不把当前结果写成 vLLM/LMCache 验收。
  - summary payload 增加 backend family 统计，方便一眼看出实际跑了哪些引擎。
- 完成标准：
  - 读 `out/profile_tables/*summary.csv` 或 README 时，不会误以为当前 profile 来自 vLLM/LMCache。
