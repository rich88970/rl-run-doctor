from __future__ import annotations

from pathlib import Path

import pytest

from rl_run_doctor.exceptions import LogLoadError, UnsupportedFormatError
from rl_run_doctor.loaders import load_log


def test_csv_loader_normalizes_columns(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("steps,episode_reward,avg_speed\n0,1,2.5\n1,2,3.0\n", encoding="utf-8")

    log = load_log(path)

    assert log.detected_format == "csv"
    assert "timestep" in log.normalized_columns
    assert "reward" in log.normalized_columns
    assert "speed" in log.normalized_columns
    assert log.x_column == "timestep"


def test_json_array_loader(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text('[{"timestep": 0, "reward": 1}, {"timestep": 1, "reward": 2}]', encoding="utf-8")

    log = load_log(path)

    assert log.detected_format == "json"
    assert len(log.data) == 2
    assert log.data["reward"].tolist() == [1, 2]


def test_json_single_record_loader(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text('{"timestep": 0, "reward": 10}', encoding="utf-8")

    log = load_log(path)

    assert log.detected_format == "json"
    assert len(log.data) == 1
    assert log.data.loc[0, "reward"] == 10


def test_ndjson_loader(tmp_path: Path) -> None:
    path = tmp_path / "train.ndjson"
    path.write_text('{"timestep": 0, "reward": 1}\n{"timestep": 1, "reward": 2}\n', encoding="utf-8")

    log = load_log(path)

    assert log.detected_format == "ndjson"
    assert len(log.data) == 2
    assert "reward" in log.data.columns


def test_json_extension_can_load_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text('{"timestep": 0, "reward": 1}\n{"timestep": 1, "reward": 2}\n', encoding="utf-8")

    log = load_log(path)

    assert log.detected_format == "ndjson"
    assert len(log.data) == 2


def test_sb3_monitor_loader_skips_comments_and_maps_columns(tmp_path: Path) -> None:
    path = tmp_path / "monitor.csv"
    path.write_text('#{"env_id": "CartPole-v1"}\nr,l,t\n10,100,0.5\n20,120,1.1\n', encoding="utf-8")

    log = load_log(path)

    assert log.detected_format == "sb3_monitor_csv"
    assert "reward" in log.normalized_columns
    assert "episode_length" in log.normalized_columns
    assert "wall_time" in log.normalized_columns
    assert log.x_column == "episode_index"
    assert log.data["reward"].tolist() == [10, 20]


def test_missing_timestep_uses_episode_index(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("reward\n1\n2\n3\n", encoding="utf-8")

    log = load_log(path)

    assert log.x_column == "episode_index"
    assert log.data["episode_index"].tolist() == [0, 1, 2]
    assert any("No timestep column found" in warning for warning in log.warnings)


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LogLoadError):
        load_log(tmp_path / "missing.csv")


def test_loader_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("reward\n1\n", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        load_log(path)


def test_loader_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(LogLoadError):
        load_log(path)


def test_loader_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.ndjson"
    path.write_text('{"reward": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(LogLoadError):
        load_log(path)
