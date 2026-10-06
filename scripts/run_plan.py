"""Run a plan of training runs in order; safe to rerun after any interruption.

    python scripts/run_plan.py runs/plans/lr_sweep.json [--code ../forge-frozen]

Each entry trains `checkpoints/<name>` and then evaluates it on the full
validation split into `results/<plan>/<name>`. Finished entries (an evaluation
exists) are skipped, interrupted ones resume from their last checkpoint, so the
same command continues a plan wherever it stopped. Launch it through
scripts/detach.py so it outlives the session that started it.

`--code` runs training and evaluation from another checkout (a git worktree at a
fixed commit), so editing this one cannot change a plan that is running; results
then record that checkout's commit. Data, checkpoints, and results stay here.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def flags(options):
    args = []
    for key, value in options.items():
        args.append("--" + key.replace("_", "-"))
        if value is not True:
            args.append(str(value))
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan")
    parser.add_argument("--code", help="checkout to run the forge package from")
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    data = str(ROOT / plan["data"])
    common = plan.get("common", {})
    results = ROOT / "results" / plan_path.stem
    code = Path(args.code).resolve() if args.code else ROOT
    env = {**os.environ, "PYTHONPATH": str(code / "src")}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code, capture_output=True, text=True
    ).stdout.strip()
    print(f"plan {plan_path.name}: code {code} at {commit}", flush=True)

    def run(command):
        subprocess.run(command, cwd=code, env=env, check=True)

    for entry in plan["runs"]:
        name = entry["name"]
        out = ROOT / "checkpoints" / name
        evaluation = results / name / "evaluation.json"
        if evaluation.exists():
            print(f"[{name}] done, skipping", flush=True)
            continue
        if not (out / "summary.json").exists():
            options = {**common, **entry.get("train", {})}
            command = [sys.executable, "-m", "forge", "train", "--data", data, "--out", str(out)]
            command += flags(options)
            if (out / "resume.pt").exists():
                command.append("--resume")
            print(f"[{name}] training: {subprocess.list2cmdline(command)}", flush=True)
            run(command)
        command = [sys.executable, "-m", "forge", "evaluate", "--run", str(out), "--data", data]
        command += ["--out", str(results / name)]
        command += flags({k: common[k] for k in ("backend", "device", "attention") if k in common})
        if entry.get("report"):
            command += ["--report", str(ROOT / "docs" / plan_path.stem / name)]
        print(f"[{name}] evaluating", flush=True)
        run(command)
    print("plan complete", flush=True)


if __name__ == "__main__":
    main()
