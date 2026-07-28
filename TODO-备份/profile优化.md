Profile 层如何选择

profile 层不是 policy 选出来的，而是由配置和 adapter 枚举出来的。

流程是：

1. run_build_profile_table.py 读取 configs/pilot.yaml 里的 profiles.adapters 和
    profiles.names。

2. profiles.registry.build_profile_adapters() 构造 adapter。
3. 每个 adapter 的 profiles() 返回它支持的 profile 列表。
4. run_build_profile_table.py 对每个 request × 每个 profile 跑 measurement。
5. split 由请求数据决定：前 100 条 calibration，后 100 条 eval。

你分析的备份文件是 5baseline_smoke，它对应旧实现/备份 adapter：code-备份/vllm_lru_profile/
vllm_lru.py:17，枚举的是：

engine_full_lru
compress_light
compress_heavy
offload_default
recompute_default

备份 profile 表也确认了：

1000 rows = 5 profiles × 200 requests
每个 profile: calibration 100 + eval 100
全部 measured=True, ok=True

是否有遗漏

按这份 5baseline smoke 自己的配置口径，没有少跑这 5 个 profile。每个 profile 都有 200 行，
calibration/eval 覆盖完整。

但从实验有效性看，有两个重要遗漏：

1. 4 个 lossy profile 不是真实 profile kernel。表里显示：

    engine_full_lru: extra_backend=vllm
    其它 4 个:       extra_backend=synthetic_action_profile

    也就是只有 engine_full_lru 真的跑了 vLLM；compress_light/heavy/offload/recompute 是
    synthetic action profile。

2. lossy profile 的质量没有形成可用梯度。四个 lossy profile 全部：

    quality_loss = 1.0
    violation_rate = 1.0

    在 epsilon=0.05 下，它们全部不可用。这样 static/safe/oracle 类策略只能退回 exact full。

另外，当前工作区最新 configs/pilot.yaml 已经不是 5baseline，而是：

full_gpu
kivi_4bit
kivi_2bit
h2o_heavy_hitter
full_cpu
recompute

所以 out-备份/5baseline_smoke 是旧实验快照，不能直接代表当前代码配置。

为什么 policy 很多数据一致

这是当前 replay 设计的正常结果，不是 CSV 汇总错误。

policy 阶段不重新跑模型，而是：

policy decide -> 得到 action_profile
MeasuredReplayBackend -> 按 (request_id, action_profile) 查 profile 表
MetricCollector -> 对查到的 ttft/loss/memory 做平均和 p95

这份结果里：

full_lru             -> engine_full_lru 100/100
static_best          -> engine_full_lru 100/100
static_safe          -> engine_full_lru 100/100
uncalibrated_dynamic -> engine_full_lru 100/100
quality_oracle       -> engine_full_lru 100/100

所以这 5 个 policy 对每个 request 查到的是同一条 engine_full_lru measurement，TTFT、memory、
quality loss 逐行完全一样，summary 当然也一样。

是否需要优化

需要优化，但不是优化“summary 里重复行”本身。重复行是在当前数据下的合理表现。真正需要优化的是
实验和策略约束：

1. utility_dynamic 应该避免输出 unsafe action。现在它全选 compress_heavy，但 safe_ratio=0、
    violation_rate=1。如果它是可部署策略，应该在 guard 不安全时强制 fallback 到 exact
    profile，而不是只记录 unsafe。

2. profile 层应替换 synthetic lossy profile。否则 compress/offload/recompute 的 TTFT/memory/
    quality 都不是完整真实结果，很难支撑论文结论。

3. 需要增加“中间质量损失”的 profile 或调参。现在只有两档：engine_full_lru loss=0，其它全部
    loss=1。这会让 safe policy 没有选择空间。理想情况应有一些 profile 的 quality_loss <=
    epsilon 或接近阈值，策略才会体现差异。

4. summary 可以加诊断字段，帮助一眼看出退化：

    unique_action_count
    identical_to_full_lru
    unsafe_action_count
    candidate_safe_count

    这不是核心算法优化，但能避免误读“多个 policy 一样”是 bug。

结论：policy 数据一致本身不需要“打散”；它暴露的是当前 5baseline profile 质量全挂、策略只能
fallback 的问题。优先改 profile 真实性和 utility_dynamic 的 unsafe fallback 语义。