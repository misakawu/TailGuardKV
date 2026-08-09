# 理想 baseline smoke 表说明

这份说明不是在复述某次具体实验结果，而是在定义一张“看起来正常、能解释问题、能区分 baseline”的 smoke 表应该长什么样。

我这里关心两件事：

1. 表里的数字有没有基本语义。
2. 5 条 baseline 曲线有没有拉开，而不是全都挤成一条。

## 先说结论

一张理想的 baseline smoke 表，至少要让人一眼看出下面这些关系：

- `full_lru` 是 exact 参照线，质量最好，内存最高，动作最单一。
- `static_best` 会吃掉一部分 tail 风险，换来更低的平均 TTFT 或更低的平均 KV。
- `static_safe` 比 `static_best` 更保守，质量尾部更稳，必要时会回退 exact。
- `utility_dynamic` 应该最敢选 lossy，平均内存通常最低，但 tail 风险也最大。
- `uncalibrated_dynamic` 介于 `static_best` 和 `utility_dynamic` 之间；它靠点预测选 profile，不保证 tail-SLO。

如果表里看不出这几条，那这张 smoke 表基本就没立住。

## 理想的数值结构

下面这张表不是硬编码阈值，而是一组在多数 smoke cell 里都应该成立的相对关系。`cell` 指一组固定的 `epsilon / delta / memory_budget_mib`。

| policy | p95_quality_loss | violation_rate | mean_kv_cache_memory_mib | p95_ttft_ms | exact_action_ratio | fallback_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `full_lru` | 最低，通常接近 0 | 最低，通常接近 0 | 最高 | 不一定最低，但应稳定 | 1.0 | 0 |
| `static_best` | 低于动态激进策略，高于 `static_safe` | 通常高于 `static_safe` | 低于 `full_lru` | 可能优于 `full_lru`，也可能只是持平 | 0 到 1 之间 | 低到中 |
| `static_safe` | 接近 exact，明显稳于 `static_best` | 理想上不高于 `delta`，至少要低于 `static_best` | 略低于或接近 `full_lru` | 通常不差，但不会像激进策略那样压内存 | 中到高 | 中 |
| `utility_dynamic` | 五者里最高或接近最高 | 五者里最高或接近最高 | 最低 | 不保证最好；如果 restore/recompute 多，反而会更差 | 最低 | 低 |
| `uncalibrated_dynamic` | 高于 `static_safe`，低于或接近 `utility_dynamic` | 高于 `static_safe` | 低于静态 exact 线 | 有机会优于 `full_lru`，但不稳定 | 低到中 | 低到中 |

如果要更直白一点，可以直接看下面这些排序。

### 质量侧的理想排序

- `full_lru ≈ static_safe <= static_best <= uncalibrated_dynamic <= utility_dynamic`
- 这里最关键的是 `static_safe` 和 `static_best` 不能重合。
- `utility_dynamic` 可以差，但必须差得有解释，也就是它确实换到了更低内存或更激进的 lossy 分布。

### 内存侧的理想排序

- `utility_dynamic <= uncalibrated_dynamic <= static_best <= static_safe <= full_lru`
- `static_best` 不该和 `full_lru` 一模一样。
- `uncalibrated_dynamic` 不该静默吞回 exact，不然这个 baseline 就失去意义了。

### 动作分布的理想排序

- `full_lru` 应该几乎只选 exact。
- `static_safe` 可以回退 exact，但要能看出“原本想选 lossy，后来因为 safe 不够而回退”。
- `static_best` 应该固定在某个 lossy 或某个 exact 上；无论哪种，理由要清楚。
- `utility_dynamic` 和 `uncalibrated_dynamic` 应该真的出现 lossy 分布，不然它们就退化了。

## 一个理想 cell 可以长成什么样

拿一个比较典型的 cell 来说，比如：

- `epsilon = 0.05`
- `delta = 0.05`
- `memory_budget_mib = 4900`

如果这组约束下 lossy profile 还算能打，那么一张健康的表，大概会有这种形状：

| policy | mean_quality_loss | p95_quality_loss | violation_rate | mean_kv_cache_memory_mib | p95_ttft_ms | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `full_lru` | `0.00 ~ 0.01` | `0.00 ~ 0.02` | `0.00 ~ 0.02` | 最高 | 中等偏稳 | exact 基线 |
| `static_safe` | `0.00 ~ 0.02` | `0.01 ~ 0.05` | `<= 0.05` 更理想 | 略低于 `full_lru` | 接近 `full_lru` | 保守固定线 |
| `static_best` | `0.02 ~ 0.04` | `0.05 ~ 0.15` | `0.05 ~ 0.25` 都可能出现 | 明显低于 `full_lru` | 可能更好，也可能接近 | 平均优先 |
| `uncalibrated_dynamic` | `0.02 ~ 0.06` | `0.06 ~ 0.18` | 常常高于 `static_safe` | 低于 `static_best` 或接近 | 波动较大 | 点预测动态线 |
| `utility_dynamic` | `0.03 ~ 0.08` | `0.08 ~ 0.20+` | 可能最高 | 最低 | 不一定最好 | 收益导向对照线 |

这里要注意两点：

- `utility_dynamic` 的 TTFT 不一定赢。它省下来的 KV，如果换成了更多 `restore`、`recompute` 或队列开销，`p95_ttft_ms` 反而会更高。
- `static_best` 的平均质量损失可以很好看，但 `p95_quality_loss` 和 `violation_rate` 可能并不好看。这个反差正是 baseline smoke 表应该暴露出来的东西。

## 理想的函数结构

我更看重“表里数字会不会随着约束变化按常识移动”。这部分比某个绝对数值更重要。

### 1. 对 `memory_budget_mib` 的响应

预算从紧到松时，理想上应该看到：

- `budget_hit_rate` 下降。
- `restore_count`、`recompute_count`、`queue_delay_ms` 下降。
- `mean_global_resident_kv_mib` 上升或持平。
- `exact_action_ratio` 上升或者至少不下降太多。
- `p95_ttft_ms` 不会因为预算变松而变差很多。

如果预算放宽后，`budget_hit_rate` 不动，或者 `p95_ttft_ms` 反而明显变差，那就要怀疑 projected memory 或 replay 状态语义没对齐。

### 2. 对 `epsilon` 的响应

`epsilon` 从严到松时，理想上应该看到：

- `lossy_action_ratio` 上升。
- `mean_kv_cache_memory_mib` 下降。
- `mean_quality_loss` 上升。
- `violation_rate` 不一定上升很多，但至少不该反常下降到完全没变化。

如果 `epsilon=0.05` 和 `epsilon=0.10` 下 5 条 baseline 几乎同分布，那通常说明 lossy 候选根本没进到选择逻辑里。

### 3. 对 `delta` 的响应

`delta` 主要影响的是 tail 安全约束，所以最应该动的是：

- `static_safe`
- 任何显式用 conformal 约束 tail 的策略

对于 baseline smoke 来说，理想状态是：

- `static_safe` 在更严格的 `delta` 下更容易回退 exact。
- `utility_dynamic` 基本不该对 `delta` 敏感。
- `uncalibrated_dynamic` 也不该对 `delta` 太敏感，因为它本来就不是 conformal baseline。

### 4. 对 backend 事件的响应

几列事件统计应该能互相对得上：

- `budget_hit_rate` 上去时，`restore_count`、`recompute_count`、`queue_delay_ms` 至少有一项也会上去。
- `mean_cumulative_kv_mib >= mean_resident_kv_mib`。
- 同一会话里，后一 turn 的 `resident_kv_mib_before` 应该能接上前一 turn 的 `resident_kv_mib_after`。
- profile 切换后，如果不是 exact 无状态假象，就应该能看到 `restore` 或 `recompute`。

如果 `budget_hit_rate` 很高，但 `restore/recompute/evict` 全都接近 0，这张表就解释不了 backend 在干什么。

## smoke 表至少要保住的几条硬约束

这几条我会直接拿来判断表有没有坏掉：

- `full_lru` 不能和其余四条 baseline 大面积完全重合。
- `static_best`、`static_safe`、`uncalibrated_dynamic` 不能全部退化成 exact 单一路径。
- `budget_hit_rate` 必须只表示 backend 真的撞预算，不能混入策略预过滤。
- `policy_budget_filter_rate` 要和 `budget_hit_rate` 分开看。
- `action_distribution` 必须能解释 `mean_kv_cache_memory_mib` 和 `violation_rate` 的变化。
- exact profile 也要带上会话 resident 字段，不然 `full_lru` 这条线没有可比性。

## 看表时最有用的几组联读

不要只看单列。baseline smoke 表最有价值的地方，在于几列一起看。

### 质量风险联读

- `mean_quality_loss`
- `p95_quality_loss`
- `violation_rate`

这组主要用来区分：

- 谁只是“平均上看着还行”
- 谁是真的 tail 上稳

### 内存收益联读

- `mean_kv_cache_memory_mib`
- `mean_global_resident_kv_mib`
- `budget_hit_rate`
- `policy_budget_filter_rate`

这组主要看：

- 策略到底是在省真实 backend 内存，还是只是在策略层面收紧候选

### 时延来源联读

- `p95_ttft_ms`
- `restore_time_ms`
- `recompute_time_ms`
- `queue_delay_ms`

这组主要看：

- TTFT 的变化到底来自 profile 变轻，还是来自会话切换和预算压力

## 最后给一个简化判断标准

如果我只用三句话来判断一张 baseline smoke 表是不是“理想”，那就是：

- 五条 baseline 必须真的分开，尤其是 `static_best`、`static_safe`、`utility_dynamic` 不能挤成一团。
- 质量、内存、时延三边的交换关系必须说得通，不能只看到收益，看不到代价。
- 会话驻留、全局驻留、预算命中、`restore/recompute/evict` 这四类 backend 行为，必须都能从表里直接读出来。

满足这三条，这张 smoke 表才有资格拿来给后面的 TailGuard 或 Oracle 做参照。
