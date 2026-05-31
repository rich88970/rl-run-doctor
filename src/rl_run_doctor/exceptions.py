"""Project-specific exceptions."""

from __future__ import annotations


class RLDoctorError(Exception):
    """Base error for expected rl-run-doctor failures."""


class LogLoadError(RLDoctorError):
    """Raised when a log cannot be loaded into a usable dataframe."""


class UnsupportedFormatError(LogLoadError):
    """Raised when a file extension is not supported."""
