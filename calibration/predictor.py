from __future__ import annotations

from core_types import Request


class ConstantRiskPredictor:
    """QRP 占位实现，用来先打通策略和校准接口。"""

    def __init__(self, default_loss: float = 0.0) -> None:
        self.default_loss = default_loss

    def predict_loss(self, request: Request, profile: str) -> float:
        if profile.startswith("full") or profile == "recompute":
            return 0.0
        return self.default_loss
