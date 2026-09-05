# TailGuardKV 当前实现对照

这份文档对照 [README-实验设计总结.md](./README-实验设计总结.md) 和当前仓库实现，重点只看实验主路径里的三层：`ProfileAdapter`、`Policy`、`Backend`。结论先写在前面：现在的代码和设计总结大方向是一致的，但实现还偏 Pilot 阶段的统计表驱动版本，接口粒度比总结里画的类图和时序图更收缩，尤其是 `Policy` 和 `Backend` 还没有完全长成“动作级、多对象”的统一执行框架。

## 1. 对照范围

这次对照主要看下面几块：

- `profiles/`
- `policies/`
- `backends/`
- `run_util/core_types.py`
- `run_util/experiment.py`
- `metrics/collector.py`

判断标准不复杂：README 里写的三层边界现在有没有落到代码里；如果落了，落到什么程度；如果没完全落，差在哪一层。

## 2. 整体结论

先说一致的部分。仓库已经把三层骨架搭出来了。

- `profiles/base.py` 里有 `ProfileAdapter` 抽象类。
- `policies/base.py` 里有 `Policy` 抽象类。
- `backends/base.py` 里有 `Backend` 抽象类。

这说明 README 里“先把 profile、policy、backend 三层拆开”的方向不是空想，代码已经按这个方向起步了。`MeasuredReplayBackend` 也确实在维护缓存状态、会话状态和预算压力，不只是简单查表回放。`TailGuardPolicy` 里也能看到 predictor、conformal guard 和安全候选筛选这条主线，说明 TailGuardKV 的控制逻辑不是只停留在文档里。

但再往下看，README 里的描述已经比当前实现走得更远。当前代码更像“能跑通 Pilot 的最小闭环”，而 README 里已经按“后续要稳定扩到主实验”的接口抽象在写。这不是矛盾，更像设计和实现之间天然会出现的时间差。

## 3. ProfileAdapter 层

README 里对这一层的定位是明确的：统一封装不同 profile，产出每请求 × profile 的质量、显存和时延数据，并把这些数据继续喂给上层 policy 和 measured-replay backend。

这一点和代码是基本对得上的。

- `profiles/base.py` 里的 `ProfileAdapter` 已经定义了 `profiles()`、`smoke()`、`profile()` 和 `profile_many()`。
- `profiles/registry.py` 也已经把 `full`、`kivi`、`h2o`、`vllm` 这些 adapter 注册起来了。
- `run_util/core_types.py` 里有 `ProfileSpec` 和 `ProfileMeasurement`，而且 `ProfileMeasurement` 已经带了质量损失、TTFT、峰值显存、KV cache、restore/recompute 等字段。

这说明 README 里说的“动作定义和测量层”现在是有实体的，不是临时拼出来的概念。

不过，README 的类图把这层写得更通用，比如 `load_measurement_table()`、`estimate_cost()` 这种接口，现在并没有出现在 `ProfileAdapter` 基类里。当前实现的重心还是“实际跑 profile 并产出 measurement”，而不是把“运行时成本查询”和“离线表访问”也统一收进同一个抽象。后两件事现在更多还是散在 replay backend 和 policy 的统计逻辑里。

所以这一层的状态可以概括成：**方向已落地，接口还偏实用型，还没有完全收束成 README 里的统一数据服务层。**

## 4. Policy 层

README 里把 `Policy` 写成决策层：读取请求、对象、系统状态，再产出动作。TailGuardKV 这条主线在文字里被拆成 QRP、CG、STC 三段，这一点和当前代码的精神是一致的。

代码里已经有这些东西：

- `policies/base.py` 定义了 `Policy.decide()`。
- `StatsPolicy` 已经把 calibration measurements、predictor、conformal guard、memory budget 这些东西串起来了。
- `policies/tailguard.py` 的 `TailGuardPolicy` 会遍历 lossy profiles，先算预测损失，再算风险上界，然后从安全集合里选一个 TTFT 最优的候选；如果没有安全 lossy profile，就退回 exact fallback。

这部分和 README 最接近。至少从控制逻辑上说，“预测 -> 校准 -> 安全集合 -> fallback”已经在跑。

真正的差距在接口粒度上。

README 里的 `Policy` 已经在按更通用的形式描述了：它接近于一个 `PolicyContext`，里面可能带多个对象，然后输出一组 `Action`。现在代码还不是这样。当前接口是：

- 输入：`request + cache_state + device_state`
- 输出：单个 `Action`

也就是说，现在的 `Policy` 更像“给当前请求挑一个 profile”，而不是“对多个缓存对象做联合动作选择”。这和 README 里的 UML 草图相比，是一个明显收缩版。

另外，README 里把 STC 写得比较像一个独立模块，但当前 `TailGuardPolicy` 里的“控制器”实际上还是很轻的：安全候选主要按预测 TTFT 排序，没有发展成更完整的多动作调度器。换句话说，QRP 和 CG 已经比较清楚，STC 现在还是一个简化版。

所以这一层最合适的判断是：**核心决策链已经存在，但还停在“单请求、单动作、轻量 STC”的 Pilot 形态。**

## 5. Backend 层

这一层是 README 和当前实现差距最大的地方。

README 里把 `Backend` 写成真正的执行层：`Policy` 给它动作，它负责在 replay 或真实系统里执行这些动作，并返回系统后果。类图里甚至已经写成 `ExecutionRequest -> ExecutionResult` 这种形式。

当前代码里的 `Backend` 还没有长到这一步。

- `backends/base.py` 现在只有一个很薄的接口：`run(requests, profiles) -> list[BackendResult]`。
- `MeasuredReplayBackend` 的工作方式也更像“给定请求和目标 profile，查 measurement，再结合 `CacheState` 模拟 restore / recompute / queue / budget pressure”。

这说明现在的 backend 已经不只是读表，它其实有一套 session-aware 的 replay 逻辑，尤其是预算压力和全局 resident KV 的演化已经在模拟了。这部分做得不浅。

但它仍然不是 README 里那种“动作级 backend”。原因很直接：它接收的是 profile 序列，不是显式动作对象；它也没有一个统一的 `ExecutionRequest` 数据结构来描述“从哪个 profile 到哪个 profile、为什么切换、触发了什么恢复或驱逐”。

另一个更实际的偏差是，README 里已经把真实系统 backend 写成 `VLLMBackend` / `LMCacheBackend` 这一层的正式成员，但现在 `backends/` 目录里只有 `base.py` 和 `measured_replay.py`。真实系统相关代码更多还是在 `profiles/` 目录里，以 adapter 形式存在。换句话说，README 里的 backend 分层已经比当前仓库更规整，代码还没有完全收束过去。

所以这层现在最准确的判断是：**measured replay 这一支已经有东西，但“统一动作执行层”还没真正成型，真实 backend 的边界也还没有完全从 profile runtime 里拆出来。**

## 6. 数据结构对照

如果只看 `run_util/core_types.py`，能发现一个很有意思的状态：很多 README 里需要的字段，其实代码已经提前准备了。

例如：

- `ProfileMeasurement` 里已经有 `quality_loss`、`ttft_ms`、`peak_memory_mib`、`kv_incremental_mib`、`restore_ms`、`recompute_ms`。
- `BackendResult` 里已经有 `queue_delay_ms`、`global_resident_kv_mib`、`global_budget_mib`。
- `Action` 和 `ActionDecision` 里已经带了 `pred_loss`、`risk_upper`、`safe`、`fallback_reason`、`controller_overhead_ms` 等字段。
- `CacheState` 里已经有 `session_current_profile`、`session_resident_kv_mib`、`session_offloaded_kv_mib` 这类会话级状态。

这说明实现并不是完全“代码在前、设计在后”。很多为后续扩展准备的数据骨架已经先放好了。

问题不在字段不够，而在这些字段还没有完全组织成 README 里那套更统一的交互模式。简单说，**类型系统已经开始向设计靠拢，接口层还没有完全跟上。**

## 7. 指标层和 Runner

README 里强调要用统一 runner、统一指标和统一输出，这一点在当前实现里是比较扎实的。

- `run_experiment.py` 实际只是入口，主流程在 `run_util/experiment.py`。
- `run_util/experiment.py` 已经把 profile 表构建、policy sweep、summary 聚合和输出路径串起来了。
- `metrics/collector.py` 已经能汇总 profile summary 和 policy run summary，质量、TTFT、显存、fallback、budget hit、restore/recompute、session 分布这些指标都在。

所以从实验框架角度看，最稳定的部分其实是 runner 和 metrics。也正因为这两层比较稳定，README 才能放心把三层架构往前写。底下接口还在调整，但外面那条“统一实验管线”已经基本成立。

## 8. 根因式判断

如果按更系统的方式看，README 和实现之间的差距不是“谁写错了”，根因更像下面这几条：

1. 当前实现优先保证 Pilot 可跑  
   这会自然偏向 measurement table、单请求决策和 replay 仿真，因为这些最容易形成闭环。

2. 真实 backend 还没统一收束  
   真实系统侧的代码还主要挂在 `profiles/`，所以 README 里那种干净的 backend 边界，现在只能部分成立。

3. STC 还在最小版本  
   现在的安全控制已经能用，但还没发展成 README 里那种更完整的动作级调度器。

4. 类型先行，接口后补  
   `core_types.py` 里已经有很多未来需要的字段，但真正把它们串成统一 `ExecutionRequest` / `ExecutionResult` 流程，还差一层接口重构。

这几条放在一起，就能解释为什么 README 看起来比实现“成熟半步”：不是文档夸张，而是实现当前确实停在最小完整原型的位置。

## 9. 最后判断

如果只问一句“README 和当前实现是否一致”，答案不能简单说是或不是。

更准确的说法是：

- 架构方向一致；
- ProfileAdapter 这一层最接近 README；
- Policy 的核心逻辑已经对上，但接口还是简化版；
- Backend 的 measured replay 已经有实体，但统一动作执行层和真实 backend 还没完全按 README 那样落下来；
- runner 和指标层已经足够支撑 Pilot。

所以现在最合适的定位不是“设计和实现脱节”，而是：**README 已经把主实验想要的分层边界讲清楚了，当前代码则实现了其中最关键、最先要跑通的那一段。**

## 10. 2026-08-10 双轨修正

截至 2026-08-10，实验入口又补了一层正式语义边界：

- `baseline_quality` 明确绑定独立 `qa/summary` baseline。
- `baseline_session` 明确绑定 session-aware replay / backend 语义验收。
- 请求、profile 表、policy records 和 summary 都已经按实验类型执行不同校验。

这次修正解决的不是“三层结构是否存在”，而是“同一套 runner 是否会把质量表误读成 backend 表”。现在主 baseline 结果不再承诺 session/backend 语义，而 session 轨道如果缺少 resident/global resident/budget/event 证据会直接返回失败。
