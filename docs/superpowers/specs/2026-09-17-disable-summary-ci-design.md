# 汇总图禁用置信区间设计

## 目标

四张 policy 汇总 PNG 只显示 policy 的汇总实线和数据点，不再渲染 bootstrap 置信区间阴影。

## 范围

- 修改 `visual/plot_summary.py` 的图层构造；不改变图表名称、指标、坐标轴、风险标识或输入函数签名。
- 保留汇总 CSV 的 CI 列和 `metrics/bootstrap.py` 的计算，避免改变聚合数据契约。
- 不再从绘图代码解析或使用 CI 列，也不创建 `fill_between` 阴影图层。
- 使用现有 `session27_total_summary.csv` 重生成目标目录中的四张 PNG；不运行 GPU 实验。

## 验证

- 图表测试提供含 CI 字段的 CSV，并断言所有 Matplotlib axes 均没有由 `fill_between` 产生的 collection。
- 既有四张 PNG 的文件名和生成行为保持不变。
