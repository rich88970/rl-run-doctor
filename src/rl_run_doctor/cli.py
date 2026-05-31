"""Command-line interface for rl-run-doctor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rl_run_doctor import __version__
from rl_run_doctor.analysis import DetectorConfig, run_diagnosis
from rl_run_doctor.dashboard import run_dashboard
from rl_run_doctor.exceptions import RLDoctorError
from rl_run_doctor.loaders import load_log
from rl_run_doctor.plots import generate_analysis_plots, generate_compare_plots
from rl_run_doctor.reports import write_analysis_report, write_compare_report


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "analyze":
            return _run_analyze(args.log_file, args.output)
        if args.command == "compare":
            return _run_compare(args.log_file_a, args.log_file_b, args.output)
        if args.command == "dashboard":
            return run_dashboard()
    except RLDoctorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl-doctor",
        description="Diagnose reinforcement learning training logs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze one training log.")
    analyze.add_argument("log_file", help="Path to a CSV, JSON, JSONL, NDJSON, or monitor.csv log.")
    analyze.add_argument("--output", required=True, help="Directory for plots and reports.")

    compare = subparsers.add_parser("compare", help="Compare two training logs.")
    compare.add_argument("log_file_a", help="Path to the first log file.")
    compare.add_argument("log_file_b", help="Path to the second log file.")
    compare.add_argument("--output", required=True, help="Directory for plots and reports.")

    subparsers.add_parser("dashboard", help="Launch the optional Streamlit dashboard.")
    return parser


def _run_analyze(log_file: str, output: str) -> int:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DetectorConfig()
    log = load_log(log_file)
    analysis = run_diagnosis(log, config)
    plots = generate_analysis_plots(log, output_dir, config)
    report_md, report_html = write_analysis_report(log, analysis, plots, output_dir)

    print(f"Analysis complete: {report_md}")
    print(f"HTML report: {report_html}")
    return 0


def _run_compare(log_file_a: str, log_file_b: str, output: str) -> int:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DetectorConfig()
    log_a = load_log(log_file_a)
    log_b = load_log(log_file_b)
    analysis_a = run_diagnosis(log_a, config)
    analysis_b = run_diagnosis(log_b, config)
    plots = generate_compare_plots(log_a, log_b, output_dir)
    report_md, report_html = write_compare_report(log_a, analysis_a, log_b, analysis_b, plots, output_dir)

    print(f"Comparison complete: {report_md}")
    print(f"HTML report: {report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
