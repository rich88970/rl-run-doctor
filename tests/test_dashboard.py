from __future__ import annotations

import importlib

import pytest

from rl_run_doctor.dashboard.app import STREAMLIT_MISSING_MESSAGE, run_dashboard


def test_dashboard_prints_clean_error_when_streamlit_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "streamlit.web.cli":
            raise ImportError("No module named streamlit")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    exit_code = run_dashboard()

    assert exit_code == 1
    assert capsys.readouterr().out.strip() == STREAMLIT_MISSING_MESSAGE
