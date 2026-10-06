# Working rules for coding agents

These apply to any agent working in this repository (Codex, Claude Code, or others).
Work here is regularly handed between agents when one runs out of credits, so every
task must survive the agent stopping mid-way and be easy for the next one to resume.

## Start here

Read [PROGRESS.md](PROGRESS.md) first. It records what has finished, what is running,
and the exact next step.

## Long-running work

- Anything that may take longer than about two minutes (training, benchmark sweeps,
  full evaluations, downloads) runs through `scripts/detach.py`, never in the agent's
  own shell. A detached job keeps running when the agent, terminal, or editor exits:

  ```powershell
  .\.venv\Scripts\python.exe scripts\detach.py --name <job> -- python -m forge <args>
  .\.venv\Scripts\python.exe scripts\detach.py --status
  ```

  Output goes to `logs/<job>.log`; `logs/<job>.json` records the command and exit code.
  `--stop <job>` terminates a job and everything it started.
- Never run two timing measurements at once (benchmarks, training throughput); they
  share the CPU and GPU and corrupt each other's numbers.
- Training stays resumable: a new `--out` per schedule, checkpoints every 100 updates,
  and the identical command plus `--resume` to continue a run that stopped.

## Hand-off

- Update `PROGRESS.md` when a job starts (job name, command, expected duration), when it
  ends (outcome and where results are), and before ending a turn.
- Results that matter go in `results/` (tracked by Git); `checkpoints/`, `data/`, and
  `logs/` are ignored.

## Machines

- **Laptop (Windows, `main` branch)**: `.venv` has Python 3.12 and PyTorch
  2.8.0+cu128 on an RTX 4050 Laptop GPU (6 GB). Training and CUDA benchmarks run here.
  Tests: `.\.venv\Scripts\python.exe -m pytest`; lint: `.\.venv\Scripts\python.exe -m ruff check .`
- **Mac Pro (macOS, `mac` branch)**: CPU benchmarks and data work; no CUDA. Follow
  [runs/MAC.md](runs/MAC.md) and keep `runs/MAC_PROGRESS.md` instead of `PROGRESS.md`.
  Use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe` in the commands above.
