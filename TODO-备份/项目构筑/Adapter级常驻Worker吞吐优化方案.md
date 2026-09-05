# Adapter 级常驻 Worker 吞吐优化方案

## Summary

目标是把 profile 构建阶段从“每个 chunk 都重新 `conda run` + 启动 runtime 子进程”改成“每个 adapter 对应一个常驻 worker，整轮 profile 请求在该 worker 内持续执行”。这次只做 `adapter` 级常驻，不做单一 OS 进程统一，也不做更细的 GPU 级多 worker 调度。

预期收益有三类：

- 去掉每个 chunk 的 conda 启动、Python 解释器启动、模块导入和 runtime 初始化开销。
- 让 `full`、`kivi`、`h2o` 各自的模型与 CUDA 上下文在本 adapter 生命周期内复用。
- 保持现有不同 conda 环境隔离，不强行统一依赖环境。

默认范围只覆盖 measured profile table 构建链路；`run_policies`、summary、plot 阶段不改执行模型。

## Key Changes

### 1. 常驻 worker 架构

- 在主控侧新增一个 adapter worker 管理层，负责：
  - 按 adapter 启动对应 conda 环境中的常驻进程。
  - 通过 stdin/stdout 或本地 JSONL IPC 向 worker 发送批任务。
  - 维护 worker 生命周期、超时、异常退出重启和收尾关闭。
- 每个 adapter 固定一个 worker：
  - `full` -> `tailguardkv-base`
  - `kivi` -> `edgekv-kivi`
  - `h2o` -> `edgekv-h2o`
- worker 启动后常驻，直到该 adapter 的所有 profile 全部跑完才退出。

### 2. 任务与协议

- 新增一个最小 IPC 协议，消息以 JSON 为单位，必须包含：
  - `op`: `init` / `run_batch` / `shutdown`
  - `adapter`
  - `profile`
  - `requests`
  - `runtime_config`
  - `session_runtime_state`
  - `memory_budget_mib`
- `init` 只在 worker 建立后执行一次，用于确认 runtime 配置和环境可用。
- `run_batch` 对应当前一次 chunk 的 measured 执行，返回：
  - `ok`
  - `results`
  - `worker`
  - `session_runtime_state`
  - `fatal_error`（仅在 worker 无法继续服务时返回）
- `shutdown` 用于显式释放模型、cache 和 CUDA 资源，避免主控只靠进程杀掉来清理。

### 3. worker 内部行为

- 每个 worker 内保留 adapter 对应的 runtime 进程内状态：
  - `full` 复用 Qwen2 exact runtime 模型实例。
  - `kivi` 复用 KIVI runtime 模型实例和 session 级 cache 容器。
  - `h2o` 复用 H2O runtime 模型实例和 session 级状态。
- worker 内按 profile 串行处理本 adapter 的请求，不做 profile 间并发。
- 当 profile 切换时：
  - 允许保留同 adapter 共享的模型级初始化。
  - 不要求保留跨 profile 的 session cache 语义。
  - profile 级 session/runtime 状态仍按当前主控逻辑分 profile 隔离。
- OOM、fatal CUDA error、worker 内未捕获异常，都视为该 worker 当前生命周期失效：
  - 当前 batch 返回失败结果。
  - 主控结束该 worker。
  - 同 adapter 后续任务默认不自动重试，直接终止本轮 profile 构建并输出现有诊断文件。

### 4. 主控接入点

- `run_util.build_profile_table` 改为优先走常驻 worker 通道，而不是每次调用 adapter 内部 `subprocess.run(...)`。
- 现有 `profile_chunk_size`、`cuda_visible_devices`、`device_strategy`、`memory_budget_mib`、`session_runtime_state` 继续保留，作为 worker 请求字段透传。
- 现有输出兼容性必须保持不变：
  - profile CSV 字段不变
  - failed chunk sidecar 格式不变
  - chunk 完成日志格式不变
  - summary 输入格式不变
- 保留一个显式回退开关，默认建议：
  - `profile_smoke.use_persistent_workers: true`
  - 设为 `false` 时退回当前一次一子进程模式，便于排障和对照。

### 5. 文件与模块边界

- 在 `profiles/` 下新增专门的 worker server 入口，不把 server 循环直接塞进现有 runtime 文件。
- `profiles/base.py` 保留 payload 构造和结果解析职责，但把“启动一次性子进程”替换为“和常驻 worker 通信”。
- `run_util.build_profile_table` 负责 worker 生命周期和 adapter 维度调度，不负责 runtime 细节。
- 现有 `profiles.qwen2_kv_runtime`、`profiles.transformers_runtime` 中可复用的 batch 执行函数尽量复用，不重写 measured 语义。

## Test Plan

- 单元测试
  - worker 管理层能正确启动、发送 `run_batch`、接收结果、执行 `shutdown`。
  - `use_persistent_workers=true/false` 两条路径都可运行。
  - worker 返回 `fatal_error` 时，主控能停止该 adapter 后续任务并写诊断输出。
  - `session_runtime_state` 在同 profile 多 chunk 之间能持续传递。
- 集成测试
  - `full` adapter measured batch 走常驻 worker，不再按 chunk 触发一次性 `subprocess.run`。
  - `kivi`、`h2o` measured batch 同样走常驻 worker，并保留当前 `memory_budget_mib` 与 session 语义。
  - dry-run 路径保持现状，不强制经过常驻 worker。
- 回归验证
  - `pytest -q` 全量通过。
  - `python -m run_util.build_profile_table --config configs/pilot_throughput_precheck.yaml --dry-run ...` 输出不变。
  - 小规模 measured smoke 至少跑一轮 `pilot_50`，确认结果文件结构与当前链路兼容。
- 性能验收
  - 以 `pilot_throughput_precheck.yaml` 为基线，对比改造前后总耗时。
  - 重点观察每个 adapter 首 chunk 与后续 chunk 的耗时差异；目标是后续 chunk 明显下降。
  - GPU 利用率不低于当前预检方案，且不出现新增的长时间空转。

## Assumptions

- 只做 `adapter` 级常驻 worker，不做跨 adapter 单进程统一。
- 不尝试统一 `full`、`kivi`、`h2o` 的 conda 环境；环境隔离继续存在。
- 不在本阶段引入 profile 间并行或多 worker 负载均衡；先解决反复起进程的问题。
- 当前正在运行的正式实验不作为改造对象；实现应先在新一轮预检或小规模 smoke 上验证。
- 默认采用 `use_persistent_workers: true`，但必须保留回退到旧模式的配置开关，避免新链路出问题时无法继续实验。

## 已完成任务

- 2026-08-09：将 Adapter 级常驻 Worker 吞吐优化方案写入根目录文档，并预留完成记录区。
- 2026-08-09：新增 `profiles/persistent_worker.py` 作为常驻 worker server 入口，采用 JSONL stdin/stdout 协议处理 `init`、`run_batch`、`shutdown`。
- 2026-08-09：在 `profiles/base.py` 中新增常驻 worker client、请求发送与结果解析逻辑，并保留一次一子进程回退路径。
- 2026-08-09：为 `full`、`kivi`、`h2o` adapter 接入 `persistent_worker` 可选通道；measured profile 可走常驻 worker，dry-run 仍走原路径。
- 2026-08-09：在 `profiles/qwen2_kv_runtime.py` 中新增 worker 生命周期接口，支持同一 profile 跨 chunk 复用 runtime，并在 OOM / fatal CUDA error 后返回 `fatal_error` 且释放 runtime。
- 2026-08-09：在 `run_util/build_profile_table.py` 中加入 adapter 级 worker 生命周期管理、`use_persistent_workers` 开关接入，以及 fatal worker 诊断输出逻辑。
- 2026-08-09：在 `run_util/config_loader.py` 与 `configs/pilot_throughput_precheck.yaml` 中接入 `profile_smoke.use_persistent_workers`，默认启用。
- 2026-08-09：新增/更新测试，覆盖 persistent worker 开关、session 状态跨 chunk 传递、fatal 中止诊断、qwen2 runtime 复用与释放。
- 2026-08-09：验证通过 `python -m run_util.build_profile_table --config configs/pilot_throughput_precheck.yaml --dry-run`，dry-run 链路保持可运行并生成预期 chunk 完成日志与汇总输出。
- 2026-08-09：验证通过 `pytest -q`，结果为 `259 passed, 3 subtests passed`。
