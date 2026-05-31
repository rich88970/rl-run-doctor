"""Matplotlib plot generation for RL logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from rl_run_doctor.analysis.config import DetectorConfig
from rl_run_doctor.loaders import LoadedLog


@dataclass(frozen=True, slots=True)
class PlotArtifact:
    filename: str
    title: str
    path: Path


def generate_analysis_plots(
    log: LoadedLog, output_dir: str | Path, config: DetectorConfig | None = None
) -> list[PlotArtifact]:
    """Generate all plots supported by the available columns."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    detector_config = config or DetectorConfig()
    plots: list[PlotArtifact] = []

    if "reward" in log.data.columns:
        plots.append(
            _plot_series(
                log=log,
                y_column="reward",
                output_dir=output_path,
                filename="reward_curve.png",
                title="Reward Curve",
                ylabel="Reward",
            )
        )
        reward = pd.to_numeric(log.data["reward"], errors="coerce")
        window = min(detector_config.rolling_window, max(1, len(reward)))
        rolling = reward.rolling(window=window, min_periods=1).mean()
        plots.append(
            _plot_values(
                x=_x_values(log),
                y=rolling,
                output_dir=output_path,
                filename="rolling_reward.png",
                title=f"Rolling Reward (window={window})",
                xlabel=log.x_column,
                ylabel="Rolling reward",
            )
        )

    if "speed" in log.data.columns:
        plots.append(
            _plot_series(
                log=log,
                y_column="speed",
                output_dir=output_path,
                filename="speed_curve.png",
                title="Speed Curve",
                ylabel="Speed",
            )
        )

    if "crash" in log.data.columns or "success" in log.data.columns:
        y_columns = [column for column in ("crash", "success") if column in log.data.columns]
        plots.append(
            _plot_multi_series(
                log=log,
                y_columns=y_columns,
                output_dir=output_path,
                filename="crash_success.png",
                title="Crash and Success Signals",
                ylabel="Rate / indicator",
            )
        )

    if "steering" in log.data.columns:
        plots.append(
            _plot_series(
                log=log,
                y_column="steering",
                output_dir=output_path,
                filename="steering_curve.png",
                title="Steering Curve",
                ylabel="Steering",
            )
        )

    if "train_reward" in log.data.columns and "eval_reward" in log.data.columns:
        plots.append(
            _plot_multi_series(
                log=log,
                y_columns=["train_reward", "eval_reward"],
                output_dir=output_path,
                filename="train_eval_reward.png",
                title="Train vs Eval Reward",
                ylabel="Reward",
            )
        )

    return plots


def generate_compare_plots(
    log_a: LoadedLog, log_b: LoadedLog, output_dir: str | Path
) -> list[PlotArtifact]:
    """Generate comparison plots for two logs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if "reward" not in log_a.data.columns or "reward" not in log_b.data.columns:
        return []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(_x_values(log_a), pd.to_numeric(log_a.data["reward"], errors="coerce"), label=log_a.path.name)
    ax.plot(_x_values(log_b), pd.to_numeric(log_b.data["reward"], errors="coerce"), label=log_b.path.name)
    ax.set_title("Reward Comparison")
    ax.set_xlabel("timestep / episode_index")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return [_save_plot(fig, output_path, "compare_reward.png", "Reward Comparison")]


def _plot_series(
    log: LoadedLog,
    y_column: str,
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
) -> PlotArtifact:
    return _plot_values(
        x=_x_values(log),
        y=pd.to_numeric(log.data[y_column], errors="coerce"),
        output_dir=output_dir,
        filename=filename,
        title=title,
        xlabel=log.x_column,
        ylabel=ylabel,
    )


def _plot_values(
    x: pd.Series,
    y: pd.Series,
    output_dir: Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> PlotArtifact:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_dir, filename, title)


def _plot_multi_series(
    log: LoadedLog,
    y_columns: list[str],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
) -> PlotArtifact:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = _x_values(log)
    for column in y_columns:
        ax.plot(x, pd.to_numeric(log.data[column], errors="coerce"), label=column)
    ax.set_title(title)
    ax.set_xlabel(log.x_column)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_plot(fig, output_dir, filename, title)


def _x_values(log: LoadedLog) -> pd.Series:
    if log.x_column in log.data.columns:
        return pd.to_numeric(log.data[log.x_column], errors="coerce")
    return pd.Series(range(len(log.data)))


def _save_plot(fig: plt.Figure, output_dir: Path, filename: str, title: str) -> PlotArtifact:
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return PlotArtifact(filename=filename, title=title, path=path)
