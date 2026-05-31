"""Markdown and HTML report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from rl_run_doctor.analysis import AnalysisResult
from rl_run_doctor.loaders import LoadedLog
from rl_run_doctor.plots import PlotArtifact


def write_analysis_report(
    log: LoadedLog,
    analysis: AnalysisResult,
    plots: list[PlotArtifact],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write report.md and report.html for one run."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    markdown = _analysis_markdown(log, analysis, plots)
    html = _html_document("RL Run Doctor Report", _analysis_html(log, analysis, plots))
    return _write_reports(output_path, markdown, html)


def write_compare_report(
    log_a: LoadedLog,
    analysis_a: AnalysisResult,
    log_b: LoadedLog,
    analysis_b: AnalysisResult,
    plots: list[PlotArtifact],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write report.md and report.html for a two-run comparison."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    markdown = _compare_markdown(log_a, analysis_a, log_b, analysis_b, plots)
    html = _html_document("RL Run Doctor Compare Report", _compare_html(log_a, analysis_a, log_b, analysis_b, plots))
    return _write_reports(output_path, markdown, html)


def _write_reports(output_path: Path, markdown: str, html: str) -> tuple[Path, Path]:
    markdown_path = output_path / "report.md"
    html_path = output_path / "report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    return markdown_path, html_path


def _analysis_markdown(log: LoadedLog, analysis: AnalysisResult, plots: list[PlotArtifact]) -> str:
    lines = [
        "# RL Run Doctor Report",
        "",
        "## Input",
        f"- Input file: `{log.path}`",
        f"- Detected file format: `{log.detected_format}`",
        f"- Rows: `{len(log.data)}`",
        f"- X-axis column: `{log.x_column}`",
        f"- Normalized columns: `{', '.join(log.normalized_columns)}`",
        "",
        "## Generated Plots",
        *_plot_markdown_lines(plots),
        "",
        "## Detector Thresholds",
        *_threshold_markdown_lines(analysis),
        "",
        "## Diagnosis Summary",
        *_diagnosis_markdown_lines(analysis),
        "",
        "## Warnings",
        *_warning_markdown_lines(analysis.warnings),
        "",
    ]
    return "\n".join(lines)


def _compare_markdown(
    log_a: LoadedLog,
    analysis_a: AnalysisResult,
    log_b: LoadedLog,
    analysis_b: AnalysisResult,
    plots: list[PlotArtifact],
) -> str:
    lines = [
        "# RL Run Doctor Compare Report",
        "",
        "## Inputs",
        *_log_summary_markdown("Run A", log_a),
        *_log_summary_markdown("Run B", log_b),
        "",
        "## Comparison Summary",
        *_comparison_summary_markdown(log_a, log_b),
        "",
        "## Generated Plots",
        *_plot_markdown_lines(plots),
        "",
        "## Detector Thresholds",
        *_threshold_markdown_lines(analysis_a),
        "",
        "## Diagnosis Summary",
        "### Run A",
        *_diagnosis_markdown_lines(analysis_a),
        "",
        "### Run B",
        *_diagnosis_markdown_lines(analysis_b),
        "",
        "## Warnings",
        *_warning_markdown_lines(analysis_a.warnings + analysis_b.warnings),
        "",
    ]
    return "\n".join(lines)


def _analysis_html(log: LoadedLog, analysis: AnalysisResult, plots: list[PlotArtifact]) -> str:
    return "\n".join(
        [
            "<h1>RL Run Doctor Report</h1>",
            "<h2>Input</h2>",
            "<ul>",
            f"<li><strong>Input file:</strong> <code>{escape(str(log.path))}</code></li>",
            f"<li><strong>Detected file format:</strong> <code>{escape(log.detected_format)}</code></li>",
            f"<li><strong>Rows:</strong> {len(log.data)}</li>",
            f"<li><strong>X-axis column:</strong> <code>{escape(log.x_column)}</code></li>",
            f"<li><strong>Normalized columns:</strong> <code>{escape(', '.join(log.normalized_columns))}</code></li>",
            "</ul>",
            _plots_html(plots),
            _thresholds_html(analysis),
            _diagnosis_html(analysis),
            _warnings_html(analysis.warnings),
        ]
    )


def _compare_html(
    log_a: LoadedLog,
    analysis_a: AnalysisResult,
    log_b: LoadedLog,
    analysis_b: AnalysisResult,
    plots: list[PlotArtifact],
) -> str:
    return "\n".join(
        [
            "<h1>RL Run Doctor Compare Report</h1>",
            "<h2>Inputs</h2>",
            _log_summary_html("Run A", log_a),
            _log_summary_html("Run B", log_b),
            _comparison_summary_html(log_a, log_b),
            _plots_html(plots),
            _thresholds_html(analysis_a),
            "<h2>Diagnosis Summary</h2>",
            "<h3>Run A</h3>",
            _diagnosis_list_html(analysis_a),
            "<h3>Run B</h3>",
            _diagnosis_list_html(analysis_b),
            _warnings_html(analysis_a.warnings + analysis_b.warnings),
        ]
    )


def _plot_markdown_lines(plots: list[PlotArtifact]) -> list[str]:
    if not plots:
        return ["- No plots generated."]
    return [f"- `{plot.filename}` - {plot.title}" for plot in plots]


def _threshold_markdown_lines(analysis: AnalysisResult) -> list[str]:
    return [f"- `{name}`: `{value}`" for name, value in analysis.thresholds.items()]


def _diagnosis_markdown_lines(analysis: AnalysisResult) -> list[str]:
    if not analysis.results:
        return ["- No detectors were run."]
    return [
        f"- **{result.name}** [{result.status}/{result.severity}]: {result.message}"
        for result in analysis.results
    ]


def _warning_markdown_lines(warnings: list[str]) -> list[str]:
    unique = _deduplicate(warnings)
    if not unique:
        return ["- None."]
    return [f"- {warning}" for warning in unique]


def _log_summary_markdown(label: str, log: LoadedLog) -> list[str]:
    return [
        f"### {label}",
        f"- Input file: `{log.path}`",
        f"- Detected file format: `{log.detected_format}`",
        f"- Rows: `{len(log.data)}`",
        f"- X-axis column: `{log.x_column}`",
        f"- Normalized columns: `{', '.join(log.normalized_columns)}`",
        "",
    ]


def _comparison_summary_markdown(log_a: LoadedLog, log_b: LoadedLog) -> list[str]:
    rows = [_reward_summary("Run A", log_a), _reward_summary("Run B", log_b)]
    lines = [
        "| Run | File | Final reward | Mean reward | Best reward | Reward std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | `{file}` | {final_reward} | {mean_reward} | {best_reward} | {reward_std} |".format(
                **row
            )
        )
    return lines


def _plots_html(plots: list[PlotArtifact]) -> str:
    if not plots:
        return "<h2>Generated Plots</h2><p>No plots generated.</p>"
    items = []
    for plot in plots:
        filename = escape(plot.filename)
        title = escape(plot.title)
        items.append(f"<li><code>{filename}</code> - {title}<br><img src=\"{filename}\" alt=\"{title}\"></li>")
    return f"<h2>Generated Plots</h2><ul>{''.join(items)}</ul>"


def _thresholds_html(analysis: AnalysisResult) -> str:
    rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td><code>{escape(str(value))}</code></td></tr>"
        for name, value in analysis.thresholds.items()
    )
    return f"<h2>Detector Thresholds</h2><table><tbody>{rows}</tbody></table>"


def _diagnosis_html(analysis: AnalysisResult) -> str:
    return "<h2>Diagnosis Summary</h2>" + _diagnosis_list_html(analysis)


def _diagnosis_list_html(analysis: AnalysisResult) -> str:
    if not analysis.results:
        return "<p>No detectors were run.</p>"
    items = "".join(
        "<li>"
        f"<strong>{escape(result.name)}</strong> "
        f"[{escape(result.status)}/{escape(result.severity)}]: "
        f"{escape(result.message)}"
        "</li>"
        for result in analysis.results
    )
    return f"<ul>{items}</ul>"


def _warnings_html(warnings: list[str]) -> str:
    unique = _deduplicate(warnings)
    if not unique:
        return "<h2>Warnings</h2><p>None.</p>"
    items = "".join(f"<li>{escape(warning)}</li>" for warning in unique)
    return f"<h2>Warnings</h2><ul>{items}</ul>"


def _log_summary_html(label: str, log: LoadedLog) -> str:
    return "\n".join(
        [
            f"<h3>{escape(label)}</h3>",
            "<ul>",
            f"<li><strong>Input file:</strong> <code>{escape(str(log.path))}</code></li>",
            f"<li><strong>Detected file format:</strong> <code>{escape(log.detected_format)}</code></li>",
            f"<li><strong>Rows:</strong> {len(log.data)}</li>",
            f"<li><strong>X-axis column:</strong> <code>{escape(log.x_column)}</code></li>",
            f"<li><strong>Normalized columns:</strong> <code>{escape(', '.join(log.normalized_columns))}</code></li>",
            "</ul>",
        ]
    )


def _comparison_summary_html(log_a: LoadedLog, log_b: LoadedLog) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(row['run'])}</td>"
        f"<td><code>{escape(row['file'])}</code></td>"
        f"<td>{escape(row['final_reward'])}</td>"
        f"<td>{escape(row['mean_reward'])}</td>"
        f"<td>{escape(row['best_reward'])}</td>"
        f"<td>{escape(row['reward_std'])}</td>"
        "</tr>"
        for row in [_reward_summary("Run A", log_a), _reward_summary("Run B", log_b)]
    )
    return (
        "<h2>Comparison Summary</h2>"
        "<table>"
        "<thead><tr><th>Run</th><th>File</th><th>Final reward</th><th>Mean reward</th>"
        "<th>Best reward</th><th>Reward std</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def _reward_summary(label: str, log: LoadedLog) -> dict[str, str]:
    if "reward" not in log.data.columns:
        return {
            "run": label,
            "file": log.path.name,
            "final_reward": "n/a",
            "mean_reward": "n/a",
            "best_reward": "n/a",
            "reward_std": "n/a",
        }

    reward = pd.to_numeric(log.data["reward"], errors="coerce").dropna()
    if reward.empty:
        return {
            "run": label,
            "file": log.path.name,
            "final_reward": "n/a",
            "mean_reward": "n/a",
            "best_reward": "n/a",
            "reward_std": "n/a",
        }

    return {
        "run": label,
        "file": log.path.name,
        "final_reward": _format_number(float(reward.iloc[-1])),
        "mean_reward": _format_number(float(reward.mean())),
        "best_reward": _format_number(float(reward.max())),
        "reward_std": _format_number(float(reward.std(ddof=0))),
    }


def _format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _html_document(title: str, body: str) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            f"<title>{escape(title)}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; line-height: 1.5; margin: 2rem; color: #172026; }",
            "code { background: #f1f3f5; padding: 0.1rem 0.25rem; border-radius: 3px; }",
            "table { border-collapse: collapse; }",
            "td, th { border: 1px solid #d0d7de; padding: 0.35rem 0.5rem; }",
            "img { max-width: 900px; width: 100%; height: auto; border: 1px solid #d0d7de; margin-top: 0.5rem; }",
            "</style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
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
