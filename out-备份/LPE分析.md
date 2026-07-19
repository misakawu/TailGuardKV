# LPE 结果分析

这次图里有几个现象很集中：TTFT 和 prefill time 几乎贴在一起；LPE 的 TTFT/prefill 最差；0.75 档位的 hit 没有饱和，0.775 之后基本都到顶；在 0.75 这个真正有压力的档位里，LPE 的 hit 夹在 LRU 和 LFU 中间。

下面按这五个问题拆开说。

## 1. TTFT 为什么接近 prefill

代码里的口径很直接。`prefill_ms` 是 `first_token - scheduled`，`ttft_ms` 是 `first_token - arrival` 或 `first_token - queued`。换句话说：

```text
TTFT ~= queue_wait + prefill
```

本轮实验里 queue wait 很小。0.775 以上的 `queue_wait_p95_ms` 只有 0.03 到 0.05 ms 左右，而 `prefill_p95_ms` 是 1100 到 1300 ms 这个量级。0.75 档位稍微特殊一些，queue 均值到了 39 到 45 ms，但和 1200 ms 以上的 prefill p95 比起来，仍然不是主要耗时。

所以图上 TTFT 和 prefill 贴得很近，不是绘图问题，也不是指标重复。它说明这批请求的首 token 等待主要花在 prefill 上，排队和 decode 都不是主项。这个实验的 `max_tokens=16`，decode 本来也很短。

## 2. LPE 为什么会退化

LPE 的打分公式是：

```text
score = p_reuse * c_recomp_ms / logical_size_mb
```

看起来它在同时考虑复用概率、重算代价和 KV 大小。但当前实现里，`c_recomp_ms` 基本按 token 数线性估计，`logical_size_mb` 也按 token 数线性估计。两者一除，长度差异被抵消了很多。最后真正起作用的主要是 `p_reuse`。

这就很尴尬。LPE 原本想保护“复用概率高、重算代价高、占用又值得”的对象，但在这个实现和这批 trace 上，它更像一个带先验的复用概率排序器。长 prefix 的重算代价没有被充分放大。

`p_reuse` 本身也不是免费真相。它由 LRU-K、freq decay 和 prior 混出来，默认权重里 LRU-K 是 0.65，freq decay 是 0.35。新对象证据少，先验影响会偏大；访问次数少时，估计也容易滞后。本轮 trace 里 session 中位轮数只有 3，很多对象还没积累出稳定画像，就已经进入淘汰压力区了。

## 3. 为什么 LPE 的 TTFT/prefill 最差

LPE 差在两处：它可能淘汰了不该淘汰的 prefix，也在热路径上加了额外开销。

先看淘汰。这个 trace 是 round-robin 的多轮 ShareGPT，同一个 session 下一轮复用距离中位数是 5，p95 是 8。也就是说，最强的信号其实很朴素：最近用过的 prefix，大概率马上又会用。LRU 正好吃这个模式。

LPE 没有这么直接。它把 LRU-K、频次衰减和 prior 混在一起，score 相近时还会受到对象级 tie-break 的影响。这样会保护一些“看起来复用概率高”的对象，但这些对象未必就是下一小段时间要用的对象。对 p95 latency 来说，错过几个长 prompt 的 prefix，就足够把尾部拉高。

再看开销。LRU 基本复用 vLLM 原生 free queue 顺序，不需要大规模重排。LPE/LFU 需要维护 rank state，并在 `get_new_blocks()` 前重排 `free_block_queue`。结果里 LPE 每档大约有 1563 次 reorder，处理 2.3 万到 3.6 万个 block；`policy_time_us_avg` 是 80 到 155 us，明显高于 LRU 的 6 到 17 us。

这部分开销单次看不大，但它在 prefill 的 block 分配和缓存路径里反复出现。再叠加错误淘汰带来的重算，LPE 的 TTFT 和 prefill 就会一起变差。

## 4. 为什么 0.75 后 hit 都饱和

这里的 hit rate 不是 trace 侧模拟命中，而是 vLLM native token coverage：`native_hits / native_queries`。也就是 prompt token 中有多少 token 被 prefix cache 覆盖。

我按请求级 CSV 算了一下这个 trace 的一步前缀复用上限：

```text
总 prompt tokens: 661364
同 session 上一轮可作为下一轮 prefix 的 tokens: 404176
一步 prefix 上限: 404176 / 661364 = 0.6111
```

图里 0.775 之后，LRU/LPE/LFU 的 hit rate 基本都在 0.612 附近。这不是三个策略同时变得很强，而是这个 trace 可提供的 native prefix hit 本来就差不多到这里为止。再大的缓存也不能命中不存在的共享前缀。

0.75 档位不一样。它刚好低于保住最近 session prefix 的容量阈值，所以策略开始拉开差距：LRU 约 0.497，LPE 约 0.422，LFU 约 0.379。这个档位才是真正能看出替换策略差异的压力点。

## 5. 为什么 LPE 的 hit 在 LRU 和 LFU 之间

0.75 档位里，LRU 的 hit 最好，LFU 最差，LPE 在中间。这和三者的排序信号一致。

LRU 只看 recency，淘汰最久没用的 block。当前 trace 的复用距离很短，同一 session 下一轮通常在 5 到 8 个请求内回来。这个场景天然偏向 LRU。

LFU 优先看频次。问题是频次在这批 trace 上不够可靠：session 中位只有 3 轮，很多对象频次差异很小；少数旧 session 频次高，也不代表它马上会复用。LFU 可能保住历史上更常见的 prefix，却丢掉马上要用的最近 prefix。

LPE 夹在中间，因为它的 `p_reuse` 本来就是 recency-like 和 frequency-like 的混合。默认权重里 LRU-K 占 0.65，freq decay 占 0.35。所以它比 LFU 更懂“最近访问”这件事，hit 比 LFU 高；但它又不像 LRU 那样干净地按最近访问排序，还会被 prior、score 更新滞后和 tie-break 影响，hit 就追不上 LRU。

从结果看，这个排序很合理：

```text
0.75 LRU hit: 0.497
0.75 LPE hit: 0.422
0.75 LFU hit: 0.379
```

这不是 LPE 完全失效，而是当前 trace 的最优信号太明确了：保住最近一轮 session prefix。LPE 的混合模型在这里不够尖锐，LFU 则偏得更远。

## 小结

这组结果说明，当前实验并没有给 LPE 足够适合它发挥的负载。trace 是短 reuse distance 的多轮会话，LRU 的假设刚好成立；LPE 的代价项又因为实现上的线性抵消，没能真正表达“长 prefix 重算更贵”。0.775 以上 hit 已经碰到 trace 上限，策略差异被盖住。要继续评估 LPE，重点应该放在 0.75 附近，或者构造更长 reuse distance、更混杂对象大小、更明显重算代价差异的 trace。
