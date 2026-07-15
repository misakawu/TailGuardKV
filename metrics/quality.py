from __future__ import annotations


def normalized_exact_match_loss(candidate: str, reference: str) -> float:
    """最小质量损失函数；真实任务指标后续按数据集替换。"""

    return 0.0 if candidate.strip() == reference.strip() else 1.0
