"""Optional Streamlit dashboard."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

from rl_run_doctor.analysis import DetectorConfig, run_diagnosis
from rl_run_doctor.loaders import load_log
from rl_run_doctor.plots import generate_analysis_plots


STREAMLIT_MISSING_MESSAGE = (
    "Streamlit is not installed. Install it with:\n"
    "pip install \"rl-run-doctor[dashboard]\""
)


def run_dashboard() -> int:
    """Launch the Streamlit dashboard if Streamlit is installed."""

    try:
        streamlit_cli = importlib.import_module("streamlit.web.cli")
    except ImportError:
        print(STREAMLIT_MISSING_MESSAGE)
        return 1

    original_argv = sys.argv[:]
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    try:
        streamlit_cli.main()
    finally:
        sys.argv = original_argv
    return 0


def streamlit_main() -> None:
    """Render the dashboard. This function is executed by Streamlit."""

    import streamlit as st

    st.set_page_config(page_title="RL Run Doctor", layout="wide")
    st.title("RL Run Doctor")
    uploaded = st.file_uploader("Upload a CSV, JSON, JSONL, NDJSON, or SB3 monitor.csv log")
    if uploaded is None:
        return

    suffix = Path(uploaded.name).suffix or ".csv"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"uploaded{suffix}"
        temp_path.write_bytes(uploaded.getvalue())
        output_dir = Path(temp_dir) / "report"
        try:
            log = load_log(temp_path)
            analysis = run_diagnosis(log, DetectorConfig())
            plots = generate_analysis_plots(log, output_dir)
        except Exception as exc:  # noqa: BLE001 - Streamlit should show clean runtime errors.
            st.error(str(exc))
            return

        st.subheader("Input")
        st.write(
            {
                "detected_format": log.detected_format,
                "rows": len(log.data),
                "x_column": log.x_column,
                "normalized_columns": log.normalized_columns,
            }
        )

        st.subheader("Diagnosis")
        st.dataframe(
            [
                {
                    "name": result.name,
                    "status": result.status,
                    "severity": result.severity,
                    "message": result.message,
                    "value": result.value,
                    "threshold": result.threshold,
                }
                for result in analysis.results
            ],
            use_container_width=True,
        )

        if analysis.warnings:
            st.subheader("Warnings")
            for warning in analysis.warnings:
                st.warning(warning)

        st.subheader("Plots")
        for plot in plots:
            st.image(str(plot.path), caption=plot.title, use_container_width=True)


if __name__ == "__main__":
    streamlit_main()
