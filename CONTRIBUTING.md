# Contributing

Thanks for improving `rl-run-doctor`.

## Development Setup

```bash
python -m pip install -e ".[dev,dashboard]"
python -m pytest
```

Before opening a pull request, run:

```bash
rl-doctor --version
rl-doctor analyze examples/sample_train.csv --output out/analyze
rl-doctor compare examples/sample_train.csv examples/sample_eval.csv --output out/compare
```

## Pull Requests

- Keep changes focused and easy to review.
- Add or update tests for behavior changes.
- Update README or examples when user-facing behavior changes.
- Do not add network-dependent tests.

## Code Style

Prefer small, explicit modules over broad abstractions. The package should remain lightweight and easy to run locally.
