from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl_run_doctor.analysis import DetectorConfig, run_diagnosis
from rl_run_doctor.loaders import LoadedLog


def _log(data: dict[str, list[float]]) -> LoadedLog:
    frame = pd.DataFrame(data)
    if "timestep" not in frame.columns:
        frame["episode_index"] = range(len(frame))
        x_column = "episode_index"
    else:
        x_column = "timestep"
    return LoadedLog(
        path=Path("synthetic.csv"),
        detected_format="csv",
        data=frame,
        original_columns=list(frame.columns),
        normalized_columns=list(frame.columns),
        x_column=x_column,
        warnings=[],
    )


def _result_statuses(results) -> dict[str, str]:
    return {result.name: result.status for result in results}


def test_reward_plateau_detector_flags_flat_recent_rewards() -> None:
    log = _log({"timestep": [0, 1, 2, 3, 4, 5], "reward": [1, 2, 3, 3, 3, 3]})

    analysis = run_diagnosis(log, DetectorConfig(min_points=5))

    assert _result_statuses(analysis.results)["reward_plateau"] == "problem"


def test_reward_collapse_detector_flags_late_drop() -> None:
    log = _log({"timestep": [0, 1, 2, 3, 4, 5], "reward": [10, 20, 50, 80, 20, 10]})

    analysis = run_diagnosis(log, DetectorConfig(rolling_window=1, min_points=5))

    assert _result_statuses(analysis.results)["reward_collapse"] == "problem"


def test_high_variance_detector_flags_noisy_rewards() -> None:
    log = _log({"timestep": [0, 1, 2, 3, 4, 5], "reward": [0, 100, 0, 100, 0, 100]})

    analysis = run_diagnosis(log, DetectorConfig(min_points=5))

    assert _result_statuses(analysis.results)["high_variance"] == "problem"


def test_optional_metric_detectors_flag_problems() -> None:
    log = _log(
        {
            "timestep": [0, 1, 2, 3, 4, 5],
            "reward": [1, 2, 3, 4, 5, 6],
            "speed": [0.1, 0.2, 0.3, 0.2, 0.1, 0.3],
            "crash": [0, 1, 1, 0, 1, 1],
            "success": [0, 0, 0, 1, 0, 0],
            "steering": [-1, 1, -1, 1, -1, 1],
            "train_reward": [100, 100, 100, 100, 100, 100],
            "eval_reward": [40, 40, 40, 40, 40, 40],
        }
    )

    analysis = run_diagnosis(log, DetectorConfig(min_points=5))
    statuses = _result_statuses(analysis.results)

    assert statuses["low_average_speed"] == "problem"
    assert statuses["high_crash_rate"] == "problem"
    assert statuses["low_success_rate"] == "problem"
    assert statuses["steering_oscillation"] == "problem"
    assert statuses["train_eval_mismatch"] == "problem"


def test_missing_optional_columns_warn_instead_of_crashing() -> None:
    log = _log({"timestep": [0, 1, 2, 3, 4], "reward": [1, 2, 3, 4, 5]})

    analysis = run_diagnosis(log, DetectorConfig(min_points=5))
    statuses = _result_statuses(analysis.results)

    assert statuses["low_average_speed"] == "skipped"
    assert statuses["high_crash_rate"] == "skipped"
    assert statuses["low_success_rate"] == "skipped"
    assert statuses["steering_oscillation"] == "skipped"
    assert statuses["train_eval_mismatch"] == "skipped"
    assert any("Missing speed column" in warning for warning in analysis.warnings)


def test_detector_config_thresholds_are_exposed() -> None:
    config = DetectorConfig(collapse_drop_ratio=0.42)
    analysis = run_diagnosis(_log({"timestep": [0, 1, 2, 3, 4], "reward": [1, 2, 3, 4, 5]}), config)

    assert analysis.thresholds["collapse_drop_ratio"] == 0.42
