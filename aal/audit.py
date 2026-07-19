from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt


@dataclass(frozen=True)
class AuditSample:
    request_id: str
    profile: str
    predicted_loss: float
    observed_loss: float
    epsilon: float

    @property
    def violation(self) -> bool:
        return self.observed_loss > self.epsilon


@dataclass
class WilsonDriftDetector:
    epsilon: float
    delta: float
    window_size: int = 100
    release_margin: float = 0.5
    samples: deque[AuditSample] = field(default_factory=deque)
    drift_state: str = "stable"

    def add(self, sample: AuditSample) -> str:
        self.samples.append(sample)
        while len(self.samples) > self.window_size:
            self.samples.popleft()
        upper = self.wilson_upper()
        if upper > self.delta:
            self.drift_state = "drift"
        elif upper <= self.delta * self.release_margin:
            self.drift_state = "stable"
        return self.drift_state

    def wilson_upper(self, z: float = 1.96) -> float:
        n = len(self.samples)
        if n == 0:
            return 0.0
        failures = sum(1 for sample in self.samples if sample.violation)
        phat = failures / n
        denom = 1 + z * z / n
        centre = phat + z * z / (2 * n)
        margin = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
        return (centre + margin) / denom


@dataclass
class AALState:
    epsilon: float
    delta: float
    audit_rate: float = 0.05
    window_size: int = 100
    unsafe_profiles: set[str] = field(default_factory=set)
    detectors: dict[str, WilsonDriftDetector] = field(default_factory=dict)

    def record(self, sample: AuditSample) -> str:
        detector = self.detectors.setdefault(
            sample.profile,
            WilsonDriftDetector(self.epsilon, self.delta, self.window_size),
        )
        state = detector.add(sample)
        if state == "drift":
            self.unsafe_profiles.add(sample.profile)
        elif state == "stable":
            self.unsafe_profiles.discard(sample.profile)
        return state

    def calibration_update(self, profile: str, new_delta: float | None = None) -> None:
        if new_delta is not None:
            self.delta = new_delta
        self.unsafe_profiles.discard(profile)
        self.detectors.pop(profile, None)
