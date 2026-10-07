"""Run a GRPO plan: train each run, then evaluate it; safe to rerun after any interruption.

    python scripts/run_grpo_plan.py runs/plans/phase4.json [--code ../forge-frozen]

Each entry trains checkpoints/<name> with `forge grpo` (an interrupted run resumes
from its last checkpoint), then evaluates its final policy, and with
"evaluate_snapshots" its intermediate snapshots, on the fixed evaluation prompts:
samples, constraint scores, and fluency under the pretrained model, all written to
results/<plan>/ along with the training log, summary, and logged rollout groups.
Steps whose output already exists are skipped, so the same command continues a plan
wherever it stopped. Launch it through scripts/detach.py.

`--code` runs everything from another checkout (a git worktree at a fixed commit),
so editing this one cannot change a plan that is running.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH_OPTIONS = ("init", "prompts", "exclude", "tokenizer")


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
    results = ROOT / "results" / plan_path.stem
    results.mkdir(parents=True, exist_ok=True)
    code = Path(args.code).resolve() if args.code else ROOT
    env = {**os.environ, "PYTHONPATH": str(code / "src")}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code, capture_output=True, text=True
    ).stdout.strip()
    print(f"plan {plan_path.name}: code {code} at {commit}", flush=True)

    common = {k: str(ROOT / v) if k in PATH_OPTIONS else v for k, v in plan["common"].items()}
    evaluation = plan["evaluation"]
    prompts = str(ROOT / evaluation["prompts"])
    forge = [sys.executable, "-m", "forge"]
    model = ["--backend", "torch", "--device", "cuda", "--attention", "sdpa"]

    def run(command):
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    for entry in plan["runs"]:
        name = entry["name"]
        out = ROOT / "checkpoints" / name
        options = {**common, **entry.get("train", {})}
        if not (out / "summary.json").exists():
            print(f"[{name}] training", flush=True)
            run(forge + ["grpo", "--out", str(out), "--resume"] + flags(options))
        for source, target in (
            ("grpo.jsonl", "training.jsonl"),
            ("summary.json", "summary.json"),
            ("samples.jsonl", "train_samples.jsonl"),
        ):
            shutil.copyfile(out / source, results / f"{name}_{target}")
        policies = [(name, out / "model.npz")]
        if entry.get("evaluate_snapshots"):
            steps = options["steps"]
            policies += [
                (f"{name}_step{p.stem[-4:]}", p)
                for p in sorted(out.glob("model_step*.npz"))
                if int(p.stem[-4:]) < steps  # the last snapshot is model.npz
            ]
        for label, checkpoint in policies:
            samples = results / f"{label}_samples.jsonl"
            if not samples.exists():
                print(f"[{label}] sampling", flush=True)
                run(
                    forge
                    + ["generate-instruct", *model, "--checkpoint", str(checkpoint)]
                    + ["--tokenizer", common["tokenizer"], "--prompts", prompts]
                    + ["--samples", str(evaluation["samples"]), "--out", str(samples)]
                )
            scores = results / f"{label}_eval.json"
            if not scores.exists():
                run(
                    forge
                    + ["instruct-eval", "--samples", str(samples), "--prompts", prompts]
                    + ["--out", str(scores)]
                )
            fluency = results / f"{label}_fluency.json"
            if not fluency.exists():
                run(
                    forge
                    + [
                        "story-loss",
                        *model,
                        "--checkpoint",
                        str(ROOT / evaluation["fluency_model"]),
                    ]
                    + ["--tokenizer", common["tokenizer"], "--samples", str(samples)]
                    + ["--out", str(fluency)]
                )
        print(f"[{name}] done", flush=True)
    print("plan complete", flush=True)


if __name__ == "__main__":
    main()
