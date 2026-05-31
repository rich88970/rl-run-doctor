"""Load and normalize reinforcement learning training logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from rl_run_doctor.exceptions import LogLoadError, UnsupportedFormatError


SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson"}


@dataclass(slots=True)
class LoadedLog:
    """A loaded training log with normalized column names."""

    path: Path
    detected_format: str
    data: pd.DataFrame
    original_columns: list[str]
    normalized_columns: list[str]
    x_column: str
    warnings: list[str] = field(default_factory=list)


ALIASES: dict[str, str] = {
    "r": "reward",
    "reward": "reward",
    "episode_reward": "reward",
    "episode_return": "reward",
    "return": "reward",
    "ep_reward": "reward",
    "ep_rew_mean": "reward",
    "mean_reward": "reward",
    "l": "episode_length",
    "length": "episode_length",
    "ep_length": "episode_length",
    "episode_length": "episode_length",
    "episode_len": "episode_length",
    "t": "wall_time",
    "time": "wall_time",
    "wall_time": "wall_time",
    "elapsed_time": "wall_time",
    "timestep": "timestep",
    "timesteps": "timestep",
    "step": "timestep",
    "steps": "timestep",
    "global_step": "timestep",
    "total_timesteps": "timestep",
    "episode": "episode",
    "episode_index": "episode_index",
    "speed": "speed",
    "avg_speed": "speed",
    "average_speed": "speed",
    "mean_speed": "speed",
    "velocity": "speed",
    "crash": "crash",
    "crash_rate": "crash",
    "crashed": "crash",
    "collision": "crash",
    "collision_rate": "crash",
    "collisions": "crash",
    "done_crash": "crash",
    "success": "success",
    "is_success": "success",
    "success_rate": "success",
    "steer": "steering",
    "steering": "steering",
    "steering_angle": "steering",
    "train_reward": "train_reward",
    "training_reward": "train_reward",
    "train_mean_reward": "train_reward",
    "train/mean_reward": "train_reward",
    "eval_reward": "eval_reward",
    "evaluation_reward": "eval_reward",
    "eval_mean_reward": "eval_reward",
    "eval/mean_reward": "eval_reward",
}

NUMERIC_COLUMNS = {
    "reward",
    "episode_length",
    "wall_time",
    "timestep",
    "episode",
    "episode_index",
    "speed",
    "crash",
    "success",
    "steering",
    "train_reward",
    "eval_reward",
}


def load_log(path: str | Path) -> LoadedLog:
    """Load a supported log file and normalize known RL metric columns."""

    log_path = Path(path)
    if not log_path.exists() or not log_path.is_file():
        raise LogLoadError(f"Input file does not exist: {log_path}")

    suffix = log_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported file extension '{suffix}' for {log_path}. Supported extensions: {supported}"
        )

    if log_path.stat().st_size == 0:
        raise LogLoadError(f"Empty log file: {log_path}")

    try:
        if suffix == ".csv":
            data, detected_format = _load_csv(log_path)
        else:
            data, detected_format = _load_json(log_path, suffix)
    except EmptyDataError as exc:
        raise LogLoadError(f"Empty log file: {log_path}") from exc
    except PermissionError as exc:
        raise LogLoadError(f"Unable to read log file: {log_path}") from exc
    except OSError as exc:
        raise LogLoadError(f"Unable to read log file: {log_path}: {exc}") from exc

    if data.empty:
        raise LogLoadError(f"Empty log file: {log_path}")

    original_columns = [str(column) for column in data.columns]
    normalized, normalized_columns, x_column, warnings = normalize_columns(data)

    return LoadedLog(
        path=log_path,
        detected_format=detected_format,
        data=normalized,
        original_columns=original_columns,
        normalized_columns=normalized_columns,
        x_column=x_column,
        warnings=warnings,
    )


def normalize_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str, list[str]]:
    """Normalize recognized column aliases while preserving unknown columns."""

    warnings: list[str] = []
    normalized = data.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    rename_map: dict[str, str] = {}
    mapped_targets: set[str] = set()
    existing_columns = set(normalized.columns)
    for column in normalized.columns:
        target = ALIASES.get(_column_key(column))
        if target is None or target == column:
            continue
        if target in existing_columns or target in mapped_targets:
            warnings.append(
                f"Column '{column}' maps to '{target}', but '{target}' already exists; keeping '{column}'."
            )
            continue
        rename_map[column] = target
        mapped_targets.add(target)

    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    for column in NUMERIC_COLUMNS.intersection(normalized.columns):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    _normalize_rate_column(normalized, "crash")
    _normalize_rate_column(normalized, "success")

    if "timestep" in normalized.columns:
        x_column = "timestep"
    elif "episode_index" in normalized.columns:
        x_column = "episode_index"
        warnings.append("No timestep column found; using existing episode_index as the x-axis.")
    elif "episode" in normalized.columns:
        x_column = "episode"
        warnings.append("No timestep column found; using existing episode as the x-axis.")
    else:
        normalized["episode_index"] = range(len(normalized))
        x_column = "episode_index"
        warnings.append("No timestep column found; using episode index as the x-axis.")

    return normalized, [str(column) for column in normalized.columns], x_column, warnings


def _normalize_rate_column(data: pd.DataFrame, column: str) -> None:
    if column not in data.columns:
        return
    series = pd.to_numeric(data[column], errors="coerce")
    non_null = series.dropna()
    if non_null.empty:
        data[column] = series
        return
    if non_null.max() > 1.0:
        data[column] = series / 100.0
    else:
        data[column] = series


def _load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    metadata_comments = _has_comment_metadata(path)
    data = pd.read_csv(path, comment="#")
    lower_columns = {_column_key(column) for column in data.columns}
    is_monitor = (
        path.name.lower() == "monitor.csv"
        or metadata_comments
        or {"r", "l", "t"}.issubset(lower_columns)
    )
    detected_format = "sb3_monitor_csv" if is_monitor else "csv"
    return data, detected_format


def _load_json(path: Path, suffix: str) -> tuple[pd.DataFrame, str]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise LogLoadError(f"Empty log file: {path}")

    if suffix in {".jsonl", ".ndjson"}:
        return _load_json_lines(text, path), "ndjson"

    try:
        parsed = json.loads(text)
    except JSONDecodeError:
        return _load_json_lines(text, path), "ndjson"

    return _json_object_to_dataframe(parsed, path), "json"


def _load_json_lines(text: str, path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except JSONDecodeError as exc:
            raise LogLoadError(f"Invalid JSON Lines record at {path}:{line_number}") from exc
        if not isinstance(parsed, dict):
            raise LogLoadError(f"JSON Lines records must be objects at {path}:{line_number}")
        records.append(parsed)

    if not records:
        raise LogLoadError(f"Empty log file: {path}")
    return pd.DataFrame.from_records(records)


def _json_object_to_dataframe(parsed: Any, path: Path) -> pd.DataFrame:
    if isinstance(parsed, list):
        if not all(isinstance(row, dict) for row in parsed):
            raise LogLoadError(f"JSON arrays must contain objects: {path}")
        return pd.DataFrame.from_records(parsed)

    if isinstance(parsed, dict):
        if isinstance(parsed.get("records"), list):
            records = parsed["records"]
            if not all(isinstance(row, dict) for row in records):
                raise LogLoadError(f"JSON 'records' must contain objects: {path}")
            return pd.DataFrame.from_records(records)
        if parsed and all(isinstance(value, list) for value in parsed.values()):
            return pd.DataFrame(parsed)
        return pd.DataFrame.from_records([parsed])

    raise LogLoadError(f"Unsupported JSON structure: {path}")


def _has_comment_metadata(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig") as handle:
        for _ in range(5):
            line = handle.readline()
            if not line:
                break
            if line.lstrip().startswith("#"):
                return True
    return False


def _column_key(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")
