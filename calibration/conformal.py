from __future__ import annotations


class ConformalGuard:
    """CG 占位实现，后续替换为按 group/profile 的残差分位数。"""

    def __init__(self, epsilon: float, exact_profiles: set[str] | None = None) -> None:
        self.epsilon = epsilon
        self.exact_profiles = exact_profiles or {"full_gpu", "full_cpu", "recompute"}

    def is_safe(self, profile: str, predicted_loss: float, slack: float = 0.0) -> bool:
        if profile in self.exact_profiles:
            return True
        return predicted_loss + slack <= self.epsilon
