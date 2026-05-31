---
name: rl-run-doctor
description: Diagnose reinforcement learning training logs, reward curves, Stable-Baselines3 monitor.csv files, crash rates, success rates, steering oscillation, and compare RL training runs using the rl-doctor CLI.
---

Use this skill when the user asks to analyze, diagnose, compare, summarize, or debug reinforcement learning training logs.

## Goal

Use `rl-run-doctor` to inspect RL training logs, generate plots, produce Markdown/HTML reports, and summarize likely training issues in plain language.

## When to use

Use this skill when the task involves:

- reinforcement learning training logs
- Stable-Baselines3 `monitor.csv`
- reward curves
- episode reward / return
- training versus evaluation comparison
- crash rate
- success rate
- speed
- steering
- episode length
- reward collapse
- reward plateau
- unstable training
- high reward variance
- comparing two or more RL runs

Do not use this skill for unrelated general code review unless RL training logs or RL experiment outputs are involved.

## Expected input files

Search for files with extensions:

- `.csv`
- `.json`
- `.jsonl`
- `.ndjson`

Common names and locations:

- `monitor.csv`
- `progress.csv`
- `train.csv`
- `eval.csv`
- `sample_train.csv`
- `sample_eval.csv`
- `logs/`
- `runs/`
- `outputs/`
- `results/`
- `experiments/`

## Setup check

First check whether the CLI is available:

```bash
rl-doctor --version
```

If the command is not available and internet access is allowed, install it from GitHub:

```bash
python -m pip install "git+https://github.com/rich88970/rl-run-doctor.git"
```

If working inside the `rl-run-doctor` repository, install it in editable mode:

```bash
python -m pip install -e ".[dev,dashboard]"
```

## Analyze one log

For a single log file, run:

```bash
rl-doctor analyze <log_file> --output out/rl-doctor-analyze
```

Then inspect:

```text
out/rl-doctor-analyze/report.md
out/rl-doctor-analyze/report.html
```

Also check generated plots such as:

```text
reward_curve.png
rolling_reward.png
speed_curve.png
crash_success.png
steering_curve.png
train_eval_reward.png
```

Only mention plots that actually exist.

## Compare two logs

For two comparable runs, run:

```bash
rl-doctor compare <log_file_a> <log_file_b> --output out/rl-doctor-compare
```

Then inspect:

```text
out/rl-doctor-compare/report.md
out/rl-doctor-compare/report.html
out/rl-doctor-compare/compare_reward.png
```

Summarize:

- final reward
- mean reward
- best reward
- reward standard deviation
- which run appears better
- which run appears more stable
- detector warnings for each run

## Report back to the user

When reporting results, include:

1. Which files were analyzed.
2. Which commands were run.
3. Where the generated reports are located.
4. Main diagnosis.
5. Any missing optional columns.
6. Any limitations.
7. Concrete next steps for RL tuning.

Prefer concise, practical recommendations.

Example summary format:

```text
Analyzed:
- logs/run_a/monitor.csv
- logs/run_b/monitor.csv

Generated:
- out/rl-doctor-compare/report.md
- out/rl-doctor-compare/report.html
- out/rl-doctor-compare/compare_reward.png

Main findings:
- Run B has higher final reward.
- Run A has lower variance.
- Both runs are missing speed and steering columns, so speed and steering diagnostics were skipped.

Suggested next steps:
- Add speed, crash, success, and steering metrics to future logs.
- Re-run rl-doctor after logging those columns.
```
