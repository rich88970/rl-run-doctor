# Codex Workflow

Use this guide when working on `rl-run-doctor`.

## Setup

Install the project in editable mode with development and dashboard extras:

```bash
python -m pip install -e ".[dev,dashboard]"
```

## Test

Run the test suite:

```bash
python -m pytest
```

## Verify CLI Behavior

After code changes, verify the public CLI:

```bash
rl-doctor --version
rl-doctor analyze examples/sample_train.csv --output out/analyze
rl-doctor compare examples/sample_train.csv examples/sample_eval.csv --output out/compare
```

Preserve the existing public commands:

```bash
rl-doctor analyze <log_file> --output <output_dir>
rl-doctor compare <log_file_a> <log_file_b> --output <output_dir>
rl-doctor dashboard
```

## Project Guidance

- Keep the project lightweight and easy to run locally.
- Do not add network-dependent tests.
- Preserve missing optional column behavior: warn or skip, never crash.
- Update README and tests when user-facing behavior changes.
- Keep detector thresholds centralized in `DetectorConfig`.
- Keep generated reports useful for humans: include inputs, detected format, normalized columns, warnings, plots, thresholds, and diagnosis summaries.
