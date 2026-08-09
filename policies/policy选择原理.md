full_lru

  - 永远走 exact profile，本轮就是 full_gpu。
  - 它不做 lossy 选择，也不看 epsilon/delta。
  - 这条线基本就是参照线。

  static_best

  - 先看 calibration 表，对每个 lossy profile 算一组统计量。
  - 过滤条件只有两个：mean_loss <= epsilon，以及内存没超过预算。
  - 然后在剩下的 profile 里，选 p95_ttft_ms 最小的那个，固定下来，后面所有请求都
    用它。

  - 它不看 delta，也不是逐请求重选。
  - 代码在 policies/base.py:205 和 policies/static_best.py:20。

  static_safe

  - 和 static_best 几乎一样。
  - 唯一多出来的约束是：calibration 上的 violation_rate <= delta。
  - 也就是说，它还是先选一个固定 profile，只是比 static_best 多一道 tail 过滤。
  - 选完以后，线上请求不会重新挑 profile。
  - 代码在 policies/base.py:205 和 policies/static_safe.py:20。

  这里有个关键点：static_best 和 static_safe 在 decide() 里虽然会计算当前请求的 pred_loss/risk_upper/
  safe，但这些值只被记录下来，没有拿来改动作。所以即使当前请求被判成 safe=False，它们也还是继续用那个固定
  的 lossy profile。这就是为什么你会看到表里写着 calibrated unsafe，动作却没有回退。

  uncalibrated_dynamic

  - 它是逐请求决策，但不是“挑最优”，而是“按 profile 配置顺序找第一个过线的”。
  - 具体做法是：遍历所有 lossy profile，只要某个 profile 的 pred_loss <= epsilon，就立刻选它，后面的不再比
    较。

  - 所以它很依赖 profiles.names 里的顺序。
  - 代码在 policies/uncalibrated_dynamic.py:23。

  utility_dynamic

  - 也是逐请求决策。
  - 它会把所有没超预算的 lossy profile 都算一遍分数：
    score = p95_ttft + 0.05 * memory + 1000 * pred_loss

  - 最后选分数最低的那个。
  - 所以它不是“最激进省内存”，也不是“纯看质量”，而是一个手写加权综合分。
  - 代码在 policies/utility_dynamic.py:28。