from __future__ import annotations

from pathlib import Path

import pytest

from rl_run_doctor.cli import main


def test_analyze_command_creates_required_outputs(tmp_path: Path) -> None:
    output = tmp_path / "analyze"

    exit_code = main(["analyze", "examples/sample_train.csv", "--output", str(output)])

    assert exit_code == 0
    assert (output / "report.md").is_file()
    assert (output / "report.html").is_file()
    assert (output / "reward_curve.png").is_file()
    assert (output / "rolling_reward.png").is_file()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Detected file format" in report
    assert "collapse_drop_ratio" in report
    assert "Diagnosis Summary" in report


def test_compare_command_creates_required_outputs(tmp_path: Path) -> None:
    output = tmp_path / "compare"

    exit_code = main(
        [
            "compare",
            "examples/sample_train.csv",
            "examples/sample_eval.csv",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (output / "report.md").is_file()
    assert (output / "report.html").is_file()
    assert (output / "compare_reward.png").is_file()


def test_cli_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "rl-doctor" in capsys.readouterr().out


def test_cli_returns_nonzero_for_missing_input(tmp_path: Path) -> None:
    output = tmp_path / "out"

    assert main(["analyze", str(tmp_path / "missing.csv"), "--output", str(output)]) != 0


def test_cli_returns_nonzero_for_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("reward\n1\n", encoding="utf-8")

    assert main(["analyze", str(path), "--output", str(tmp_path / "out")]) != 0


def test_cli_returns_nonzero_for_empty_log(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    assert main(["analyze", str(path), "--output", str(tmp_path / "out")]) != 0


def test_cli_returns_nonzero_for_unreadable_log_content(tmp_path: Path) -> None:
    path = tmp_path / "broken.ndjson"
    path.write_text("{not json}\n", encoding="utf-8")

    assert main(["analyze", str(path), "--output", str(tmp_path / "out")]) != 0
