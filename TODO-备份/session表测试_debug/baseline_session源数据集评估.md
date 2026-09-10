# baseline_session 源数据集评估

## 范围

本文评估的是本轮 `baseline_session` 使用的数据组合：48 个 ShareGPT 两轮骨架，加上 LongBench 的 QA 或 Summary 注入内容。它不是对 ShareGPT 或 LongBench 整体质量的判断，也不把一次失败的候选配对写成数据集本身失效。

实验模型为 Qwen2.5-7B-Instruct。每个候选 session 有五轮：前两轮来自 ShareGPT，后面三轮围绕同一条 LongBench 内容及其参考答案构造。每条请求均在 `full_gpu`、4 个 KIVI profile 和 3 个 H2O profile 下实测。

## 本轮结果

四个 60 请求 batch 均已完成。合并后的测量表有 240 个请求、1,920 条 request/profile 记录。8 个 profile 各有 240 条记录，全部 `ok/measured`，并且都有 `resident_kv_mib_before`、`resident_kv_mib_after` 与 `kv_cumulative_mib`。

这说明本轮的模型运行、双卡绑定、profile adapter 和 session resident 字段采集是可用的。它不能说明正式 `baseline_session` 的 backend gate 已通过，因为最终 fixture 尚未导出，也还没有执行 policy replay 来检查 session reuse、global resident 演化和压力事件。

严格风险标注的实际分布如下。分类使用的是混合 session 的 turn2--turn4 重测结果，而不是 LongBench 单轮旧标签。

| 任务 | KIVI-sensitive | H2O-sensitive | low-risk | tie |
| --- | ---: | ---: | ---: | ---: |
| QA | 0 | 3 | 8 | 13 |
| Summary | 2 | 0 | 18 | 4 |

标注器在 `kivi_sensitive qa hybrid session pool insufficient: 0 < 8` 处停止。它没有生成新的正式 fixture 或 manifest。

## 对照 baseline_session 的要求

当前方案要求最终 fixture 有 48 个五轮 session。三个风险组各 16 个 session；在每个风险组中，QA 和 Summary 各 8 个。每个“风险组 × 任务”再按 session 分成 4 个 calibration 和 4 个 eval session。

敏感组的门槛是：

- KIVI：`kivi_max >= 0.05`，并且 `kivi_max - h2o_max >= 0.02`。
- H2O：`h2o_max >= 0.05`，并且 `h2o_max - kivi_max >= 0.02`。
- 对照组：主流 profile 的损失均不高于 `0.01`。
- 两家族都高、但差值不够的样本是 tie，不能填入任一敏感组。

| 组别与任务 | 需要的 session 数 | 本轮可用数 | 差额 |
| --- | ---: | ---: | ---: |
| KIVI-sensitive / QA | 8 | 0 | 8 |
| KIVI-sensitive / Summary | 8 | 2 | 6 |
| H2O-sensitive / QA | 8 | 3 | 5 |
| H2O-sensitive / Summary | 8 | 0 | 8 |
| low-risk / QA | 8 | 8 | 0 |
| low-risk / Summary | 8 | 18 | 0 |

所以，问题不在低风险对照，也不在 profile 覆盖。问题是敏感组没有同时覆盖两类任务，而且各组样本量远低于 session 级 4/4 split 所需的 8 个。13 个 QA tie 和 4 个 Summary tie 不能作为补数，否则“风险来自 KIVI 还是 H2O”的归因会消失。

## 是否是源数据集无法实现

就当前构造方式而言，答案是“不能实现”。现有 48 个 hybrid candidate 已经完整实测，却没有一条 KIVI-sensitive QA，也没有一条 H2O-sensitive Summary。继续用这 48 条候选无法导出符合契约的 fixture。

但不能据此说 ShareGPT 或 LongBench 无法支持该实验。LongBench 候选文件仍有 500 条内容，其中 QA 和 Summary 各 250 条；已有的 LongBench 单轮测量也覆盖这 500 条内容。当前 hybrid 构造器只是按文件顺序取前 24 条 QA 和前 24 条 Summary，再与 48 个 ShareGPT skeleton 固定配对。它没有使用单轮测量做候选预筛，也没有构造一个比最终 48 条更大的 hybrid 候选池。

现有 ShareGPT hybrid skeleton 文件只有 48 个有效两轮骨架。原始 ShareGPT pilot 文件虽然包含 70 个 session，但按当前“两轮都必须有 prompt 和 reference”的规则，没有额外合格 skeleton。因此，扩大 LongBench 内容时还需要决定候选扩池阶段能否复用 skeleton，或补充新的、满足同一契约的多轮会话来源。

## 与论文规划的关系

《论文规划/王祯祥_论文工作规划.md》把 LongBench 列为 QA、摘要和代码质量工作负载，把 ShareGPT 列为多轮会话与 prefix 复用工作负载。这个组合本身符合论文需要：LongBench 给出可比较的参考答案，ShareGPT 给出 session 结构。

规划同时要求 H0 至少在两类任务中观察到 profile 间的尾部风险差异，并要求 Pilot 的质量损失来自真实 profile 实测。当前结果只证明少数 QA 或 Summary 样本存在家族差异，不能满足 H0 的稳定证据，更不能进入 H1 校准或 H2 的 policy 对比。

规划中列出的另一类工作负载是 RAG：多 chunk、共享知识、冷热变化。它适合作为后续补充，因为共享上下文和重复访问更容易形成 session/prefix 复用与压力事件。不过规划没有给出一个可直接替换 LongBench 或 ShareGPT 的具体 RAG 数据集名称，也没有提供现成的 RAG fixture。不能把“RAG”当作已经准备好的替代数据集。

## 可行路径

首选路径仍是保留 ShareGPT + LongBench 的分工，但重做候选选择：先利用已有 LongBench 单轮实测找出 KIVI 和 H2O 的高风险内容，再构造大于 48 个的 hybrid 候选池并重新测量。最终只从混合测量结果中选出严格满足六个“风险组 × 任务”格子的 48 个 session。单轮结果只能用来提高命中率，不能直接当成最终标签。

如果复用 48 个 skeleton 不被接受，就需要补充满足两轮契约的会话数据。此时可以把 RAG 多轮对话作为新增工作负载，而不是替换所有 ShareGPT session；它需要先定义参考答案、session 边界和 provenance 规则，再进入同样的 profile 重测流程。

本轮不应继续做 fixture 导入、policy smoke 或 policy 优劣结论。风险证据不足时，项目代码规定 policy 状态应为 `risk_evidence_insufficient`；但当前甚至没有合法 fixture，连这一状态的真实 smoke 也还没有生成。

## 结论

本轮源数据组合没有失败在执行层面，失败在风险分层。固定的 48 条 hybrid 配对不足以支撑 `baseline_session` 所需的 KIVI/H2O、QA/Summary 双维风险证据。优先扩大并预筛 LongBench 注入内容；只有在 skeleton 复用不被接受或扩大后仍没有足够敏感内容时，才应引入新的多轮会话或 RAG 数据来源。
