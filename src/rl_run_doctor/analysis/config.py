"""Detector threshold configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Centralized detector thresholds for easy tuning."""

    collapse_drop_ratio: float = 0.30
    plateau_window_fraction: float = 0.30
    plateau_delta_ratio: float = 0.05
    high_variance_cv_threshold: float = 1.0
    low_average_speed_threshold: float = 1.0
    high_crash_rate_threshold: float = 0.10
    low_success_rate_threshold: float = 0.50
    steering_sign_change_rate_threshold: float = 0.35
    steering_delta_threshold: float = 0.50
    train_eval_mismatch_ratio: float = 0.25
    rolling_window: int = 10
    min_points: int = 5

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
