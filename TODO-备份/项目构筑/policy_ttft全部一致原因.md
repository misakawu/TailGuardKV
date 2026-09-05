# `policy` 段 `ttft` 全部一致的原因总结

## 现象

在 [out/policy_tables/pilot_50_measured_summary.csv](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/.worktrees/pilot-profile-bugfix/out/policy_tables/pilot_50_measured_summary.csv) 中：

- `policy` 段所有策略的 `mean_ttft_ms / p95_ttft_ms / p99_ttft_ms` 基本完全一致。
- 所有 `policy` 行的 `action_distribution` 都是 `{"full_gpu": 25}`。
- `identical_to_full_lru=True`。

因此，`ttft` 一致不是汇总器单独算坏了，而是所有策略最终都回放到了同一批 `full_gpu` 测量行。

---

## 直接原因

### 1. 所有 policy 最终动作都退化成了 `full_gpu`

`run_run_policies.py` 只是把 policy 决定出的 `action.profile` 交给 `MeasuredReplayBackend` 回放；summary 再按这些回放结果聚合。

因此：

- 如果所有策略都选 `full_gpu`
- 那么 `ttft`、`peak_memory_mib`、`quality_loss` 的 policy 聚合值就会高度一致

这解释了为什么 `policy` 段整块重复。

### 2. policy 之所以全退化成 `full_gpu`，是因为 lossy profile 几乎全部被判成不安全或无优势

从 [out/profile_tables/pilot_50_measured_profiles.csv](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/.worktrees/pilot-profile-bugfix/out/profile_tables/pilot_50_measured_profiles.csv) 可以看到：

- `full_gpu/full_cpu/recompute` 的 `quality_loss=0.0`
- KIVI/H2O 各档大面积出现 `quality_score=0.0, quality_loss=1.0`

这导致：

- `static_best/static_safe`：lossy profile 因 `mean_loss > epsilon` 被全部淘汰，退回 exact
- `utility_dynamic`：`1000 * pred_loss` 的惩罚项使 `pred_loss≈1.0` 的 lossy profile 在打分上必输
- `uncalibrated_dynamic`：即使先按 `ttft` 挑到最快 lossy，也会因 `safe=False` 再 fallback 到 exact

所以 policy 层的“全 `full_gpu`”不是汇总器 bug，而是上游 profile 结果和策略逻辑共同作用的结果。

---

## 根本原因

最根本的直接技术原因是：

### 3. `lossy` profiles 与 `exact` profiles 没有走同一条可信推理路径

当前实现中：

- `full_gpu/full_cpu/recompute` 走 `transformers.generate()` 路径
- KIVI/H2O 走 `profiles/qwen2_kv_runtime.py` 里的自定义手写推理路径

这条自定义路径包含：

- 手写 `_greedy_decode()`
- 手写 `_manual_qwen2_forward()`
- 手写 causal mask
- 手写 `position_ids/cache_position`
- 手动替换 Qwen2 attention
- 每步 `argmax` 解码

也就是说，当前 Pilot 不是“同一个模型 + 同一个 runner + 只替换 profile 机制”，而是：

- exact 路径：标准 `transformers.generate()`
- lossy 路径：共享的一条自定义 Qwen2 runtime

这已经违反了论文规划和 README 中“同一 runner / 同一质量链路”的前提。

更关键的是，KIVI 和 H2O 两类不同机制都出现了非常相似的异常输出形态，例如：

- 大量 `!`
- 单词断裂
- 语义明显退化

这说明共同故障点更可能在共享的自定义 runtime，而不是 KIVI 或 H2O 各自算法本身。

因此，**导致 lossy 输出整体异常退化的最核心直接根因，是共享的 `qwen2_kv_runtime.py` 推理路径本身不可靠。**

---

## 设计逻辑级原因

除上述直接技术根因外，还存在若干设计层面的错误或偏差。这些问题即使没有 runtime bug，也会把结果推向“policy 全部回到 full_gpu”。

### 4. 质量定义与论文规划不一致

论文规划第 5.2 节的定义是：

- 先计算任务分数 `Q_i^full` 与 `Q_i^a`
- 再定义损失 `ell_i(a)` 为两者任务分数差的归一化结果

但当前实现并不是这样。

当前 `with_quality()` 的逻辑是：

- 取同 request 的 `full_gpu.output_text` 作为 reference
- 直接把 lossy 输出文本和 full 输出文本做 token/F1/ROUGE-L 比较

这会把“任务质量损失”偷换成“和 full 输出的表面文本相似度损失”。

结果：

- 只要 lossy 输出有 paraphrase、表达方式变化，就可能被高估损失
- 一旦输出里有异常标点，如 `The!`、`baselines!!!!`，当前分词会把它们当成不同 token，直接把损失推到接近 1

因此，当前 `quality_loss=1.0` 既包含“模型真退化”，也包含“度量定义过脆弱”的成分。

### 5. quality metric 的 tokenization 过于脆弱

当前 `metrics/quality.py` 的 `_tokenize()` 只是：

- `lower()`
- `split()`

不会：

- 去除标点
- 规范化 token
- 容忍轻微表面噪声

所以：

- `The` 和 `The!`
- `baselines` 和 `baselines!!!!`

会被当成完全不同 token。

这会把 shared runtime 造成的输出噪声，进一步放大成 `quality_loss=1.0`。

### 6. memory budget 设计不满足“真正触发 profile 选择”的要求

论文规划 0.7.2 要求：

- 使用两个真正触发 profile 选择的紧预算

但当前 `pilot_50.yaml` 配的是：

- `memory_budgets_mib: [6144, 8192]`

而 measured summary 里：

- `full_gpu p95_peak_memory_mib` 约 5070 MiB
- lossy profile 大约 4814-4832 MiB

这意味着：

- `6144` 和 `8192` 都放得下 `full_gpu`
- 预算没有真正迫使策略去选 lossy profile

因此，即使 quality 没问题，控制器也缺少“必须压缩才能满足预算”的外部压力。

这会天然把系统推向 exact profile。

### 7. Pilot 数据覆盖不足，group 基本塌成单任务单长度组

论文要求 Pilot 至少覆盖：

- 2 个任务
- group = 任务类型 × 长度档

但当前 `pilot_50` 结果中，主导样本几乎全是：

- `qa_long_context`
- `length_bucket=long`

这导致：

- predictor 学不到跨任务差异
- conformal guard 学不到跨 group 差异
- safe set 很难形成论文中期望的“部分 lossy safe，部分 unsafe”的结构

最终只会产生一种很粗糙的结果：

- exact 恒 safe
- lossy 大面积 unsafe

### 8. `exact_fallback_ratio` 的实现语义与论文目标不一致

论文里这个指标是为了防止：

- “coverage 只能靠全部 full”

理想含义应更接近：

- unsafe 后被迫回退到 exact 的比例

但当前 summary 实际统计的是：

- 最终动作中 exact profile 的占比

这会把下面几类情况混在一起：

- baseline 本来就固定 exact
- 主动选择 exact
- 真实 unsafe fallback 到 exact

因此，这个指标即使数值恢复，也可能误导论文结论。

---

## 与论文规划中的理想结果相比，当前结果偏离在哪里

论文规划中理想的 Pilot 结果应当满足：

1. safe set 里同时存在 exact 和部分 lossy profile，而不是只剩 exact
2. `exact fallback ratio < 60%`，避免“全部 full”的空保证
3. 至少 2 个有效 cell 中，相比最佳可部署 baseline：
   - `p95 TTFT` 降至少 10%，或
   - 显存降至少 15%
4. 各策略的 `action_distribution` 应明显不同
5. `policy` 段 `ttft`、显存和 fallback 指标应能区分策略优劣

而当前实际情况是：

- safe set 实际上塌成 exact-only
- `policy` 层全部选择 `full_gpu`
- 不同策略在 summary 中几乎没有系统性差异
- 当前结果属于“coverage 只能靠全部 full”的退化形态

因此，当前结果不能解释为：

- KIVI/H2O 本身一定无效
- TailGuard 一定只能全部 fallback
- 论文主张已经被实验否定

更准确的解释是：

- 当前 Pilot 还没有形成一个有效、可比、可信的实验系统

---

## 最终结论

`policy` 段 `ttft` 全部一致，并不是单个汇总 bug，也不是单个 policy bug，而是多层原因叠加：

### 直接技术根因

- KIVI/H2O 共享的自定义 `qwen2_kv_runtime.py` 推理路径不可靠，导致 lossy 输出大面积异常退化

### 设计逻辑级原因

- 用“与 `full_gpu` 输出的文本相似度”替代了论文定义中的“任务质量损失”
- quality metric 的 tokenization 过于脆弱，放大了输出噪声
- memory budget 不紧，没有真正触发 profile 选择
- Pilot 样本覆盖不足，group 基本塌成单任务单长度组
- `exact_fallback_ratio` 的实现语义与论文目标不完全一致

### 结果链条

1. shared lossy runtime 先把 KIVI/H2O 输出跑坏
2. quality metric 再把这些输出系统性打成 `quality_loss≈1.0`
3. policy 层于是把所有 lossy profile 判成 unsafe 或无效益
4. 所有策略最终都回到 `full_gpu`
5. summary 中所有 `policy` 行回放同一批 `full_gpu` 测量值
6. 因而 `ttft`、显存和动作分布整块一致

一句话概括：

**最根本的问题不是 policy 汇总错了，而是当前 Pilot 违反了“同一 runner、同一质量定义、真正受预算约束”的实验前提，导致系统几乎必然退化为全 `full_gpu`。**

## 2026-07-30 之后的已修复项

截至 2026 年 7 月 30 日，下面几项已经落地，不再属于“待确认假设”：

- `pilot_50` / `pilot` 的 smoke 子集不再只按 split 前缀截断，而是按 `split x task x length_bucket` 分层抽样。
- KIVI / H2O measured runner 已切回 `transformers.generate()` 主链；手写 `_greedy_decode()` / `_manual_qwen2_forward()` 不再是默认 measured 路径。
- `quality_loss` 已优先改成“同一 reference 下的任务分数差”；文本相似度只在缺少 reference 时兜底。
- 文本兜底指标已加入大小写、Unicode、空白和标点归一化，尾部标点噪声不会再系统性推高到 `1.0`。
- pilot 预算已从 `[6144, 8192]` 收紧到 `[4900, 5000]`，用于强制制造 exact / lossy 的预算边界。
- `exact_fallback_ratio` 已改成“被迫 fallback 到 exact 的比例”，同时新增 `exact_action_ratio` 保留“最终 exact 动作占比”。

## 剩余风险

仍需继续观察的点主要有两类：

- `run_experiment.py pilot-smoke-measured --config configs/pilot_50.yaml` 的真实 measured smoke 在本机环境里可能长时间阻塞在 profile 子进程 `subprocess.run(...).communicate()`，说明运行链路仍然受外部模型执行时长影响，不能只靠单元测试断言代替。
- 即使语义和预算口径已修正，最终是否真的形成“至少一个 exact 主导 cell + 至少一个 lossy 可行动 cell”，还要看新的 measured profile 表与 policy summary 是否在真实 smoke 上分化出来。
