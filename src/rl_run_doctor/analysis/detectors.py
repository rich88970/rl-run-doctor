"""Rule-based RL training issue detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from rl_run_doctor.analysis.config import DetectorConfig
from rl_run_doctor.loaders import LoadedLog


@dataclass(slots=True)
class DiagnosisResult:
    name: str
    status: str
    severity: str
    message: str
    value: float | None = None
    threshold: float | None = None
    details: dict[str, float | str] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisResult:
    results: list[DiagnosisResult]
    warnings: list[str]
    thresholds: dict[str, float | int]

    @property
    def diagnosis_summary(self) -> list[DiagnosisResult]:
        return [result for result in self.results if result.status == "problem"]


def run_diagnosis(log: LoadedLog, config: DetectorConfig | None = None) -> AnalysisResult:
    """Run all detectors and return structured results."""

    detector_config = config or DetectorConfig()
    warnings = list(log.warnings)
    results = [
        _detect_reward_plateau(log.data, detector_config),
        _detect_reward_collapse(log.data, detector_config),
        _detect_high_variance(log.data, detector_config),
        _detect_low_average_speed(log.data, detector_config),
        _detect_high_crash_rate(log.data, detector_config),
        _detect_low_success_rate(log.data, detector_config),
        _detect_steering_oscillation(log.data, detector_config),
        _detect_train_eval_mismatch(log.data, detector_config),
    ]

    for result in results:
        if result.status == "skipped":
            warnings.append(result.message)

    return AnalysisResult(
        results=results,
        warnings=_deduplicate(warnings),
        thresholds=detector_config.to_dict(),
    )


def _detect_reward_plateau(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    reward = _numeric_series(data, "reward")
    skipped = _skip_if_missing_or_short("reward_plateau", "reward", reward, config)
    if skipped is not None:
        return skipped

    window = _window_size(len(reward), config.plateau_window_fraction)
    recent = reward.tail(window)
    scale = max(float(reward.max() - reward.min()), abs(float(reward.mean())), 1.0)
    delta_ratio = abs(float(recent.iloc[-1] - recent.iloc[0])) / scale
    is_problem = delta_ratio <= config.plateau_delta_ratio
    return _result(
        name="reward_plateau",
        is_problem=is_problem,
        severity="warning",
        message=(
            f"Recent reward appears plateaued (delta ratio {delta_ratio:.3f})."
            if is_problem
            else f"Reward is still changing in the recent window (delta ratio {delta_ratio:.3f})."
        ),
        value=delta_ratio,
        threshold=config.plateau_delta_ratio,
    )


def _detect_reward_collapse(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    reward = _numeric_series(data, "reward")
    skipped = _skip_if_missing_or_short("reward_collapse", "reward", reward, config)
    if skipped is not None:
        return skipped

    rolling = reward.rolling(window=min(config.rolling_window, len(reward)), min_periods=1).mean()
    peak = float(rolling.max())
    final = float(rolling.iloc[-1])
    drop_ratio = max(0.0, (peak - final) / max(abs(peak), 1.0))
    is_problem = drop_ratio >= config.collapse_drop_ratio
    return _result(
        name="reward_collapse",
        is_problem=is_problem,
        severity="critical",
        message=(
            f"Reward collapsed from its rolling peak by {drop_ratio:.1%}."
            if is_problem
            else f"No reward collapse detected (drop from peak {drop_ratio:.1%})."
        ),
        value=drop_ratio,
        threshold=config.collapse_drop_ratio,
    )


def _detect_high_variance(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    reward = _numeric_series(data, "reward")
    skipped = _skip_if_missing_or_short("high_variance", "reward", reward, config)
    if skipped is not None:
        return skipped

    mean = abs(float(reward.mean()))
    std = float(reward.std(ddof=0))
    cv = std / mean if mean > 1e-9 else std
    is_problem = cv >= config.high_variance_cv_threshold
    return _result(
        name="high_variance",
        is_problem=is_problem,
        severity="warning",
        message=(
            f"Reward variance is high (coefficient {cv:.3f})."
            if is_problem
            else f"Reward variance is within threshold (coefficient {cv:.3f})."
        ),
        value=cv,
        threshold=config.high_variance_cv_threshold,
    )


def _detect_low_average_speed(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    return _detect_mean_threshold(
        data=data,
        column="speed",
        name="low_average_speed",
        config=config,
        threshold=config.low_average_speed_threshold,
        predicate=lambda value, threshold: value < threshold,
        problem_message=lambda value: f"Average speed is low ({value:.3f}).",
        ok_message=lambda value: f"Average speed is within threshold ({value:.3f}).",
        severity="warning",
    )


def _detect_high_crash_rate(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    return _detect_mean_threshold(
        data=data,
        column="crash",
        name="high_crash_rate",
        config=config,
        threshold=config.high_crash_rate_threshold,
        predicate=lambda value, threshold: value > threshold,
        problem_message=lambda value: f"Crash rate is high ({value:.1%}).",
        ok_message=lambda value: f"Crash rate is within threshold ({value:.1%}).",
        severity="critical",
    )


def _detect_low_success_rate(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    return _detect_mean_threshold(
        data=data,
        column="success",
        name="low_success_rate",
        config=config,
        threshold=config.low_success_rate_threshold,
        predicate=lambda value, threshold: value < threshold,
        problem_message=lambda value: f"Success rate is low ({value:.1%}).",
        ok_message=lambda value: f"Success rate is within threshold ({value:.1%}).",
        severity="warning",
    )


def _detect_steering_oscillation(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    steering = _numeric_series(data, "steering")
    skipped = _skip_if_missing_or_short("steering_oscillation", "steering", steering, config)
    if skipped is not None:
        return skipped

    signs = np.sign(steering.to_numpy(dtype=float))
    non_zero = signs[signs != 0]
    sign_change_rate = 0.0
    if len(non_zero) > 1:
        sign_change_rate = float(np.mean(non_zero[1:] != non_zero[:-1]))
    mean_abs_delta = float(steering.diff().abs().dropna().mean())
    is_problem = (
        sign_change_rate >= config.steering_sign_change_rate_threshold
        or mean_abs_delta >= config.steering_delta_threshold
    )
    return _result(
        name="steering_oscillation",
        is_problem=is_problem,
        severity="warning",
        message=(
            f"Steering appears oscillatory (sign-change rate {sign_change_rate:.1%}, mean delta {mean_abs_delta:.3f})."
            if is_problem
            else f"Steering oscillation is within threshold (sign-change rate {sign_change_rate:.1%})."
        ),
        value=sign_change_rate,
        threshold=config.steering_sign_change_rate_threshold,
        details={"mean_abs_delta": mean_abs_delta},
    )


def _detect_train_eval_mismatch(data: pd.DataFrame, config: DetectorConfig) -> DiagnosisResult:
    train = _numeric_series(data, "train_reward")
    eval_ = _numeric_series(data, "eval_reward")
    if train.empty or eval_.empty:
        return DiagnosisResult(
            name="train_eval_mismatch",
            status="skipped",
            severity="info",
            message="Missing train_reward or eval_reward column; skipped train/eval mismatch detector.",
        )

    train_mean = float(train.mean())
    eval_mean = float(eval_.mean())
    mismatch_ratio = abs(train_mean - eval_mean) / max(abs(train_mean), 1.0)
    is_problem = mismatch_ratio >= config.train_eval_mismatch_ratio
    return _result(
        name="train_eval_mismatch",
        is_problem=is_problem,
        severity="warning",
        message=(
            f"Train/eval reward mismatch is high ({mismatch_ratio:.1%})."
            if is_problem
            else f"Train/eval reward mismatch is within threshold ({mismatch_ratio:.1%})."
        ),
        value=mismatch_ratio,
        threshold=config.train_eval_mismatch_ratio,
        details={"train_mean": train_mean, "eval_mean": eval_mean},
    )


def _detect_mean_threshold(
    data: pd.DataFrame,
    column: str,
    name: str,
    config: DetectorConfig,
    threshold: float,
    predicate: Callable[[float, float], bool],
    problem_message: Callable[[float], str],
    ok_message: Callable[[float], str],
    severity: str,
) -> DiagnosisResult:
    series = _numeric_series(data, column)
    skipped = _skip_if_missing_or_short(name, column, series, config)
    if skipped is not None:
        return skipped
    value = float(series.mean())
    is_problem = predicate(value, threshold)
    return _result(
        name=name,
        is_problem=is_problem,
        severity=severity,
        message=problem_message(value) if is_problem else ok_message(value),
        value=value,
        threshold=threshold,
    )


def _numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(data[column], errors="coerce").dropna()


def _skip_if_missing_or_short(
    detector_name: str, column: str, series: pd.Series, config: DetectorConfig
) -> DiagnosisResult | None:
    if series.empty:
        return DiagnosisResult(
            name=detector_name,
            status="skipped",
            severity="info",
            message=f"Missing {column} column; skipped {detector_name} detector.",
        )
    if len(series) < config.min_points:
        return DiagnosisResult(
            name=detector_name,
            status="skipped",
            severity="info",
            message=f"Not enough {column} points for {detector_name}; need at least {config.min_points}.",
        )
    return None


def _window_size(length: int, fraction: float) -> int:
    return min(length, max(2, int(round(length * fraction))))


def _result(
    name: str,
    is_problem: bool,
    severity: str,
    message: str,
    value: float,
    threshold: float,
    details: dict[str, float | str] | None = None,
) -> DiagnosisResult:
    return DiagnosisResult(
        name=name,
        status="problem" if is_problem else "ok",
        severity=severity if is_problem else "info",
        message=message,
        value=value,
        threshold=threshold,
        details=details or {},
    )


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
