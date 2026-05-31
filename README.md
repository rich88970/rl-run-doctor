# rl-run-doctor

[![CI](https://github.com/rich88970/rl-run-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/rich88970/rl-run-doctor/actions/workflows/ci.yml)

`rl-run-doctor` is a lightweight command-line tool for diagnosing reinforcement learning training logs. It reads CSV, JSON, JSON Lines / NDJSON, and Stable-Baselines3 `monitor.csv` files, detects common training problems, generates plots, compares runs, and exports Markdown and HTML reports.

## Quick Start

From a fresh checkout:

```bash
python -m pip install -e ".[dev,dashboard]"
python -m pytest
rl-doctor --help
rl-doctor --version
rl-doctor analyze examples/sample_train.csv --output out/analyze
rl-doctor compare examples/sample_train.csv examples/sample_eval.csv --output out/compare
```

After running those commands, open:

```text
out/analyze/report.md
out/analyze/report.html
out/analyze/reward_curve.png
out/analyze/rolling_reward.png
out/compare/report.md
out/compare/report.html
out/compare/compare_reward.png
```

## Screenshots

These screenshots are generated from the included example logs and are checked in so the links do not break.

![Reward curve](docs/screenshots/analyze/reward_curve.png)

![Rolling reward](docs/screenshots/analyze/rolling_reward.png)

![Reward comparison](docs/screenshots/compare/compare_reward.png)

Regenerate them with:

```bash
rl-doctor analyze examples/sample_train.csv --output docs/screenshots/analyze
rl-doctor compare examples/sample_train.csv examples/sample_eval.csv --output docs/screenshots/compare
```

## Installation

For local development:

```bash
python -m pip install -e ".[dev,dashboard]"
```

For the core CLI without the dashboard dependency:

```bash
python -m pip install -e .
```

The base package does not require Streamlit. Install the dashboard extra only when you want `rl-doctor dashboard`.

## Usage

Analyze one run:

```bash
rl-doctor analyze examples/sample_train.csv --output out/analyze
```

Compare two runs:

```bash
rl-doctor compare examples/sample_train.csv examples/sample_eval.csv --output out/compare
```

Launch the optional dashboard:

```bash
rl-doctor dashboard
```

If Streamlit is not installed, the dashboard command prints:

```text
Streamlit is not installed. Install it with:
pip install "rl-run-doctor[dashboard]"
```

## Using as a Codex Skill

This repository includes a repo-scoped Codex skill at `.agents/skills/rl-run-doctor/SKILL.md`. Codex can use it when working inside this repository to diagnose RL training logs with the `rl-doctor` CLI.

To use the skill across other projects, copy `.agents/skills/rl-run-doctor` to your user-level skills folder:

- Windows: `%USERPROFILE%\.agents\skills\rl-run-doctor`
- macOS/Linux: `$HOME/.agents/skills/rl-run-doctor`

Verify the CLI before using the skill:

```bash
rl-doctor --version
```

Example Codex prompts:

```text
Use the rl-run-doctor skill to analyze logs/PPO_1/monitor.csv and summarize the generated report.
Use the rl-run-doctor skill to compare logs/run_a/monitor.csv and logs/run_b/monitor.csv.
```

## Supported Logs

- CSV files
- JSON arrays
- Single JSON records
- JSON Lines / NDJSON
- Stable-Baselines3 `monitor.csv`

For SB3 monitor files, metadata lines starting with `#` are skipped and columns are normalized as:

- `r` -> `reward`
- `l` -> `episode_length`
- `t` -> `wall_time`

If no timestep column exists, rl-run-doctor uses an existing `episode_index` or `episode` column as the plot x-axis. If neither exists, it creates an episode index.

## Detectors

Detector thresholds are centralized in `DetectorConfig` and are included in every report.

- reward plateau
- reward collapse
- high variance
- low average speed
- high crash rate
- low success rate
- steering oscillation when steering data exists
- train/eval mismatch when both train and eval reward columns exist

Missing optional columns produce warnings and skipped detectors. They do not crash the tool.

## Reports

Each report includes:

- input file path
- detected file format
- normalized columns
- row count
- generated plots
- detector thresholds
- diagnosis summary
- warnings for missing optional columns

## Docker

Build and run:

```bash
docker build -t rl-run-doctor .
docker run --rm -v "$PWD:/work" -w /work rl-run-doctor analyze examples/sample_train.csv --output out/analyze
```

## Contributing

Issues, bug reports, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

MIT
