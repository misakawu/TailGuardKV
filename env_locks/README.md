# TailGuardKV 环境记录

本目录记录 TailGuardKV 当前采用的隔离环境方案。配置依据为
`/DATACENTER3/zhenxiang.wang/work/环境配置.md` 的“物理隔离，逻辑协同”策略。

## 环境映射

| 角色 | 当前环境 | 状态 | 说明 |
| --- | --- | --- | --- |
| 基础开发环境 | `tailguardkv-base` | 可用 | 从 `edgekv-kivi` 克隆，已补 `scikit-learn`、`lightgbm`、`matplotlib`；用于策略、校准、数据处理和主 runner。 |
| vLLM 引擎环境 | `edgekv-vllm0110` | 可用 | `vllm==0.11.0`，`torch==2.8.0+cu128`；不含 LMCache。 |
| LMCache 引擎环境 | `h3-lmcache-blog` | 可用 | `vllm==0.8.5.post1`，`torch==2.6.0+cu124`，可导入 `lmcache`。 |
| KIVI 压缩环境 | `edgekv-kivi` | 可用 | 已从 `third_party/KIVI` editable 安装；`models`、`quant` 可导入；`kivi_gemv` CUDA 扩展已编译。 |
| H2O/SnapKV 压缩环境 | `edgekv-h2o` | 可用 | SnapKV 已从 `third_party/SnapKV` editable 安装；H2O 通过 `PYTHONPATH=third_party/H2O/h2o_hf` 使用 `utils_hh` 源码入口。 |

## 检查命令

```bash
conda run -n tailguardkv-base python scripts/check_envs.py
```

该脚本只做 import 和版本探测，不加载模型。H2O 的 `lm_eval` 评测 harness
未安装；当前 pilot profile adapter 只依赖 `utils_hh` monkeypatch 入口。

## 本地资源

Pilot 默认模型资产可使用：

```text
/DATACENTER3/zhenxiang.wang/resource/Qwen2.5-7B-Instruct
```

第三方源码已接入本仓库：

```text
third_party/KIVI
third_party/H2O
third_party/SnapKV
```

当前源码版本：

| 组件 | commit |
| --- | --- |
| KIVI | `876b4d2 update readme` |
| H2O | `ac75c2a Merge pull request #34 from foreverpiano/main` |
| SnapKV | `e216ddc Create LICENSE` |

KIVI 编译修复记录：

- `edgekv-kivi` 的 conda GCC/G++ 已降为 12.4，以匹配 CUDA 12.1 编译上限。
- `edgekv-kivi/lib/libcudart.so` 原 symlink 指向缺失的 `libcudart.so.12.1.105`；已备份为 `libcudart.so.broken-12.1.105`，并改指向实际存在的 `libcudart.so.12.9.79`。

## 导出命令

环境稳定后重新导出：

```bash
conda env export -n tailguardkv-base > env_locks/tailguardkv-base.yml
conda env export -n edgekv-vllm0110 > env_locks/edgekv-vllm0110.yml
conda env export -n h3-lmcache-blog > env_locks/h3-lmcache-blog.yml
conda env export -n edgekv-kivi > env_locks/edgekv-kivi.yml
conda env export -n edgekv-h2o > env_locks/edgekv-h2o.yml
```
