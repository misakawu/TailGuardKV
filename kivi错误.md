# KIVI 错误定位

这次 KIVI 失败，不是单纯的环境抖动，也不是某个参数填错了。问题在缓存协议上。表面上看，报错出在 `transformers` 里：

```text
ValueError: not enough values to unpack (expected 2, got 1)
```

栈里最后落到 `DynamicCache.from_legacy_cache()`，也就是它想把 `past_key_values` 当成 Hugging Face 旧版 KV cache 来读：

```python
key_states, value_states = past_key_values[layer_idx]
```

可这次传进去的东西，根本不是它以为的那种结构。

## 先看表现

`out/profile_tables/pilot_50_measured_profiles_failed_chunks.csv` 里 10 条失败样本全部来自 `kivi_4bit_residual32`。`full_gpu`、`full_cpu` 和 `recompute` 都能跑完，只有 KIVI 全挂。这说明问题不在数据集，不在 Qwen2 主体加载，也不在通用生成流程，而是在 KIVI 这条专门改过的 KV cache 路径上。

更具体一点，错误发生在 `model.generate(...)` 的解码阶段，不是模型刚加载就挂，也不是预填充阶段直接报错。也就是说，第一轮 forward 大概率已经走通了，出问题的是带 cache 的后续 token 解码。

## 代码里到底发生了什么

KIVI profile 的入口在 `profiles/qwen2_kv_runtime.py`。`_run_kivi_profile()` 里装配完自定义 attention 之后，直接走：

```python
result = _generate_decode(model, tokenizer, device, payload, torch)
```

而 `_generate_decode()` 里调用的是：

```python
generated = model.generate(...)
```

问题就从这里开始。

### 1. `generate()` 默认在走 `DynamicCache`

当前 `edgekv-kivi` 环境里的 transformers，在支持 cache class 的模型上，会默认创建 `DynamicCache()`。Qwen2 正好就是这一路。

所以第一轮进入 `Qwen2Model.forward()` 时，传下去的 `past_key_values` 不是旧版 tuple，而是一个 `DynamicCache` 对象。

### 2. 本地 KIVI wrapper 不认识这套协议

KIVI attention 的前向里先做了这一步：

```python
past_key_value = _cache_to_legacy(past_key_value, self.layer_idx)
```

这个函数在 `profiles/qwen2_kv_runtime.py` 里的实现很关键。它只做了一件事：如果传进来的是 `DynamicCache`，就从里面拿出当前层的 `key_cache[layer_idx]` 和 `value_cache[layer_idx]`，然后拼成一个普通二元组：

```python
return (key, value)
```

这对标准 attention 是够的，对 KIVI 不够。

因为 KIVI 后面的逻辑，明确假设 `past_key_value` 是一个 9 元组：

```python
key_q, key_full, key_scale, key_mn, value_q, value_full, value_scale, value_mn, _ = past_key_value
```

这 9 个字段里，前 8 个分别是量化后的 key/value、未量化残留块、量化 scale、最小值之类的状态，最后一个是 `kv_seq_len`。这已经不是 Hugging Face 标准 `(key, value)` cache 了，而是一套 KIVI 自己定义的缓存表示。

换句话说，KIVI attention 真正依赖的是“量化缓存 + 残留缓存”的复合状态，但 `_cache_to_legacy()` 只能把输入压成一个 `(key, value)`。从设计上讲，这一步就已经把信息压没了。

### 3. KIVI attention 返回的 `past` 也不是标准 cache

KIVI forward 最后返回：

```python
past = (
    key_q,
    key_full,
    key_scale,
    key_mn,
    value_q,
    value_full,
    value_scale,
    value_mn,
    kv_seq_len,
) if use_cache else None
```

这说明本地实现从头到尾都在使用 KIVI 自己的 cache 格式。

问题在于，这个 `past` 最后不是留在 KIVI 自己的 decode 循环里，而是被 `model.generate()` 当成标准 `past_key_values` 回填到下一轮生成里。到了下一步，transformers 又会把这个值重新解释成“legacy cache”。

### 4. 第二轮解码时，transformers 误解了这个 9 元组

Qwen2Model 的逻辑是：只要 `use_cache=True`，并且传入的 `past_key_values` 不是 `Cache` 实例，就默认把它视作旧版 cache 结构，然后执行：

```python
past_key_values = DynamicCache.from_legacy_cache(past_key_values)
```

`DynamicCache.from_legacy_cache()` 又假设输入长这样：

```python
(
    (layer0_key, layer0_value),
    (layer1_key, layer1_value),
    ...
)
```

也就是“外层按层组织，内层每层两个张量”。

但 KIVI 这边传回去的不是这种结构，而是单层的 9 元组：

```python
(
    key_q,
    key_full,
    key_scale,
    key_mn,
    value_q,
    value_full,
    value_scale,
    value_mn,
    kv_seq_len,
)
```

于是 `from_legacy_cache()` 在遍历 `past_key_values[layer_idx]` 时，拿到的第一个元素其实是 `key_q`。它还以为自己拿到的是 `(key, value)`，于是试图解包：

```python
key_states, value_states = past_key_values[layer_idx]
```

这时候就炸了。因为 `key_q` 是个 tensor，不是二元组，所以报了：

```text
ValueError: not enough values to unpack (expected 2, got 1)
```

这就是为什么错误栈看起来出在 transformers，根因却在我们自己的 KIVI cache 接口设计上。

## 为什么第一轮能跑，第二轮才挂

这个现象一开始看着有点绕，其实不复杂。

第一轮 `generate()` 开始时，transformers 自己提供的是 `DynamicCache()`。虽然本地 KIVI wrapper 对它的适配并不完整，但第一轮还没进入“读取上一轮 KIVI 私有 cache”的阶段，所以还能往前跑。

真正出问题的是第一轮结束以后。KIVI attention 把自己的 9 元组 `past` 返回给 generation loop。到了第二轮，Qwen2Model 再看到这个 tuple，就按 Hugging Face 的旧版 cache 语义去解析。两边说的根本不是一回事，所以这时候才出错。

## 这不是一个小兼容性补丁能糊过去的问题

这次 bug 的核心不在“某个字段漏了”，而在于两套 cache 协议混在一起用了：

- Hugging Face `generate()` 期望的是标准 `Cache`，或者旧版 `(key, value)` 按层组织的 tuple。
- 当前 KIVI 实现维护的是一套专用的 9 元组状态。

这两套东西没有完成协议对接，却共用了同一个 `past_key_values` 通道。

所以问题有两层：

1. `_cache_to_legacy()` 只会把 `DynamicCache` 变成 `(key, value)`，无法恢复 KIVI 真正需要的量化状态。
2. KIVI attention 返回的 9 元组又被直接塞回了 Hugging Face 的 `past_key_values` 流程里，下一轮自然会被误解。

从逻辑上说，这不是“类型不匹配”那么简单，而是“接口语义完全不同”。

## 结论

这次 KIVI 错误的根因可以归结成一句话：

当前实现把 KIVI 私有的 KV cache 结构，当成了 Hugging Face `generate()` 可以原生接收和回传的标准 `past_key_values` 来用；第一轮侥幸跑过，第二轮在 `DynamicCache.from_legacy_cache()` 里被按错误协议解包，最终触发 `not enough values to unpack`。

如果后面要修，方向其实很明确，只能二选一：

- 要么别走 `model.generate()`，改成本地手写 decode 循环，让 KIVI 的 9 元组状态只在自家逻辑里流转。
- 要么正式实现一个能接到 transformers `Cache` 协议上的 KIVI cache 类型，而不是继续把私有结构塞进标准 `past_key_values`。

再往下补零碎判断没有意义。问题已经落到实现边界上了。
