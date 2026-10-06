"""Publishable language-model evidence for a finished training run.

Training-time validation scores a small fixed sample for monitoring. Here every byte
of the held-out split is scored exactly once, next to n-gram baselines fit on the
same training bytes, the training curve, and fixed-seed samples.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from forge.model import ByteTokenizer
from forge.training import sha256

# WikiText-style prompts (spaces around punctuation, " = Heading = " titles).
PROMPTS = (
    " = History of the railway = \n The railway",
    "The album was released in",
    "In the early 19th century , the town",
    "The species is found in",
)


def _negative_log_likelihood(model, x, y):
    logits = model.forward(x.astype(np.int64)).astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return -float(np.take_along_axis(log_probs, y.astype(np.int64)[..., None], -1).sum())


def heldout_loss(model, data, context, batch=32):
    """Mean nats/byte over every byte of `data` after the first.

    Non-overlapping windows, so each prediction sees 1 to `context` preceding bytes
    and never more than the training context.
    """
    targets = len(data) - 1
    full = targets // context
    total = 0.0
    for first in range(0, full, batch):
        rows = np.arange(first, min(first + batch, full))[:, None] * context
        indices = rows + np.arange(context)
        total += _negative_log_likelihood(model, data[indices], data[indices + 1])
    tail = targets - full * context
    if tail:
        indices = full * context + np.arange(tail)[None]
        total += _negative_log_likelihood(model, data[indices], data[indices + 1])
    return total / targets


def ngram_baselines(train, heldout, chunk=1 << 26):
    """Add-one smoothed byte unigram and bigram models fit on the training bytes,
    scored on the same held-out targets as `heldout_loss`."""
    unigram = np.bincount(train, minlength=256).astype(np.float64) + 1
    unigram /= unigram.sum()
    pairs = np.zeros(256 * 256)
    for start in range(0, len(train) - 1, chunk):
        stop = min(start + chunk, len(train) - 1)
        codes = np.asarray(train[start:stop], dtype=np.uint16) * 256 + train[start + 1 : stop + 1]
        pairs += np.bincount(codes, minlength=256 * 256)
    bigram = pairs.reshape(256, 256) + 1
    bigram /= bigram.sum(axis=1, keepdims=True)
    previous, target = np.asarray(heldout[:-1]), np.asarray(heldout[1:])
    return {
        "uniform": math.log(256),
        "unigram": float(-np.log(unigram[target]).mean()),
        "bigram": float(-np.log(bigram[previous, target]).mean()),
    }


def training_curve(log_path, bytes_per_update, every=100):
    """Mean training loss per `every` updates, with the monitoring validation loss.

    A resumed run re-logs updates after its last checkpoint; the later row wins.
    """
    by_step = {}
    with Path(log_path).open(encoding="utf-8") as log:
        for line in log:
            row = json.loads(line)
            by_step[row["step"]] = row
    rows = [by_step[step] for step in sorted(by_step)]
    curve = []
    for start in range(0, len(rows), every):
        group = rows[start : start + every]
        last = group[-1]
        point = {
            "step": last["step"],
            "bytes": last["step"] * bytes_per_update,
            "train_loss": float(np.mean([row["loss"] for row in group])),
            "lr": last["lr"],
            "tokens_per_second": last["tokens_per_second"],
        }
        if "val_loss" in last:
            point["monitor_val_loss"] = last["val_loss"]
        curve.append(point)
    return curve


def fixed_samples(model, context, tokens=200, temperature=0.8, top_p=0.95, seed=17):
    """Greedy and nucleus continuations of fixed prompts through the serving engine."""
    from forge.engine import Engine

    # Stay within the trained context; later RoPE positions were never trained.
    tokens = min(tokens, context - max(len(ByteTokenizer.encode(p)) for p in PROMPTS))
    if tokens < 1:
        return {"skipped": f"the {context}-byte training context cannot hold the fixed prompts"}
    engine = Engine(model, max_batch=2 * len(PROMPTS), blocks=256, seed=seed)
    requests = []
    for i, prompt in enumerate(PROMPTS):
        encoded = ByteTokenizer.encode(prompt)
        greedy = engine.submit(f"greedy-{i}", encoded, tokens)
        sampled = engine.submit(
            f"sampled-{i}", encoded, tokens, temperature=temperature, top_p=top_p
        )
        requests.append((prompt, greedy, sampled))
    while engine.busy:
        engine.step()
    return {
        "settings": {
            "tokens": tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "engine_policy": engine.policy,
        },
        "samples": [
            {
                "prompt": prompt,
                "greedy": ByteTokenizer.decode(greedy.generated),
                "sampled": ByteTokenizer.decode(sampled.generated),
            }
            for prompt, greedy, sampled in requests
        ],
    }


def run_settings(run):
    """The batch/context a run trained with, from its summary or resume checkpoint."""
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if "run_config" in summary:
        return summary, summary["run_config"]
    if (run / "run_config.json").exists():
        return summary, json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    import torch

    state = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
    return summary, state["run_config"]


def evaluate_run(model, run, data_dir, out, report_dir=None, batch=32):
    from forge.bench import environment

    run, data_dir, out = Path(run), Path(data_dir), Path(out)
    summary, settings = run_settings(run)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["val_sha256"] != summary["dataset"]["val_sha256"]:
        raise ValueError("this dataset is not the one the run trained on")
    if sha256(data_dir / "val.bin") != manifest["val_sha256"]:
        raise ValueError("validation data checksum mismatch")
    model.weight_origin = "self-trained on the recorded dataset"
    env = environment(model)
    if env["weights_sha256"] != summary["environment"]["weights_sha256"]:
        raise ValueError("checkpoint weights differ from the ones the run summary recorded")
    context = settings["context"]
    train = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
    val = np.memmap(data_dir / "val.bin", dtype=np.uint8, mode="r")

    start = time.perf_counter()
    loss = heldout_loss(model, val, context, batch)
    eval_seconds = time.perf_counter() - start
    baselines = ngram_baselines(train, val)
    curve = training_curve(run / "training.jsonl", settings["batch"] * context)
    samples = fixed_samples(model, context)
    result = {
        "run": run.name,
        "heldout": {
            "split": "validation",
            "bytes_scored": len(val) - 1,
            "method": f"non-overlapping {context}-byte windows; each byte predicted once "
            f"from 1 to {context} preceding bytes",
            "precision": "fp32",
            "loss_nats_per_byte": loss,
            "bits_per_byte": loss / math.log(2),
            "seconds": eval_seconds,
        },
        "monitor_val_loss_at_end": summary["final_val_loss"],
        # gpu_training validates on 8 batches of 2 windows; CPU training on 4 of 2.
        "monitor_sample_bytes": (16 if summary["backend"] == "torch" else 8) * context,
        "baselines_nats_per_byte": baselines,
        "training": {
            "parameters": summary["parameters"],
            "config": summary["config"],
            "run_config": settings,
            "updates": summary["steps"],
            "training_bytes": summary["steps"] * settings["batch"] * context,
            "precision": summary.get("precision", "fp32"),
            "seconds": summary["seconds_this_session"],
            "cuda_peak_allocated_bytes": summary.get("cuda_peak_allocated_bytes"),
            "training_environment": summary["environment"],
        },
        "dataset": manifest,
        "samples": samples,
        "environment": env,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (out / "training_curve.jsonl").open("w", encoding="utf-8") as stream:
        for point in curve:
            stream.write(json.dumps(point) + "\n")
    if report_dir:
        write_report(result, curve, Path(report_dir), out)
    return result


def dataset_name(manifest):
    if "dataset" in manifest:
        return f"{manifest['dataset']} ({manifest['config']})"
    return manifest["source"]


def write_report(result, curve, report_dir, results_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    held, base, train = result["heldout"], result["baselines_nats_per_byte"], result["training"]
    c = train["config"]
    peak = train["cuda_peak_allocated_bytes"]
    link = Path(os.path.relpath(results_dir, report_dir)).as_posix()
    count = train["parameters"]
    size = f"{count / 1e6:.1f}M" if count >= 1e6 else f"{count / 1e3:.0f}K"
    lines = [
        f"# {result['run']}: held-out evaluation",
        "",
        f"A {train['parameters']:,}-parameter byte-level decoder ({c['layers']} layers, "
        f"width {c['dim']}, {c['heads']} heads / {c['kv_heads']} KV heads) trained for "
        f"{train['updates']:,} updates on {train['training_bytes'] / 1e6:.2f}M bytes of "
        f"{dataset_name(result['dataset'])} in "
        + (
            f"{train['seconds'] / 60:.1f} minutes"
            if train["seconds"] >= 120
            else f"{train['seconds']:.0f} seconds"
        )
        + f" ({train['precision']} training"
        + (f", peak CUDA allocation {peak / 2**20:.0f} MiB)." if peak else ")."),
        "",
        "## Held-out loss",
        "",
        f"Scored on all {held['bytes_scored']:,} predicted bytes of the {held['split']} split "
        f"(split: {result['dataset']['split']}; {held['method']}; {held['precision']}).",
        "",
        "| Model | nats/byte | bits/byte |",
        "|---|---:|---:|",
        f"| **Forge {size}** | **{held['loss_nats_per_byte']:.4f}** | "
        f"**{held['bits_per_byte']:.4f}** |",
    ]
    for name, label in (
        ("bigram", "Byte bigram, add-one, fit on the same training bytes"),
        ("unigram", "Byte unigram, add-one, fit on the same training bytes"),
        ("uniform", "Uniform over 256 bytes"),
    ):
        lines.append(f"| {label} | {base[name]:.4f} | {base[name] / math.log(2):.4f} |")
    lines += [
        "",
        f"The training loop's monitoring loss at the last update was "
        f"{result['monitor_val_loss_at_end']:.4f} nats/byte, measured on a fixed "
        f"{result['monitor_sample_bytes']:,}-byte sample; the full-split number above is the "
        "one to cite. Byte-level losses are not comparable to published BPE or word-level "
        "WikiText-103 perplexities.",
        "",
        "## Training curve",
        "",
        "![Loss curve](loss_curve.png)",
        "",
        f"Data: [`training_curve.jsonl`]({link}/training_curve.jsonl) (mean training loss per "
        f"100 updates). Full results: [`evaluation.json`]({link}/evaluation.json).",
        "",
        "## Fixed samples",
        "",
    ]
    samples = result["samples"]
    if "skipped" in samples:
        lines += [f"Skipped: {samples['skipped']}.", ""]
    else:
        settings = samples["settings"]
        lines += [
            f"{settings['tokens']} bytes per prompt through Forge's serving engine "
            f"(`{settings['engine_policy']}` policy): greedy, then temperature "
            f"{settings['temperature']} with top-p {settings['top_p']} "
            f"(seed {settings['seed']}). Not cherry-picked: these prompts and settings are "
            "fixed in `forge/evaluation.py`.",
            "",
        ]
    for sample in samples.get("samples", []):
        lines += [
            f"**Prompt:** `{sample['prompt'].strip().replace(chr(10), '⏎')}`",
            "",
            "Greedy:",
            "",
            "```text",
            sample["prompt"] + sample["greedy"],
            "```",
            "",
            "Sampled:",
            "",
            "```text",
            sample["prompt"] + sample["sampled"],
            "```",
            "",
        ]
    (report_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    millions = [point["bytes"] / 1e6 for point in curve]
    ax.plot(
        millions, [p["train_loss"] for p in curve], lw=1.2, label="training (mean per 100 updates)"
    )
    monitored = [
        (p["bytes"] / 1e6, p["monitor_val_loss"]) for p in curve if "monitor_val_loss" in p
    ]
    if monitored:
        ax.plot(
            *zip(*monitored),
            lw=1,
            alpha=0.7,
            label=f"monitoring validation ({result['monitor_sample_bytes']:,}-byte sample)",
        )
    for name, style in (("bigram", "--"), ("unigram", ":")):
        ax.axhline(base[name], ls=style, color="gray", lw=1, label=f"{name} baseline")
    ax.scatter(
        [millions[-1]],
        [held["loss_nats_per_byte"]],
        marker="*",
        s=140,
        zorder=3,
        color="black",
        label=f"full validation split: {held['loss_nats_per_byte']:.3f}",
    )
    ax.set_ylim(held["loss_nats_per_byte"] - 0.2, base["unigram"] + 0.3)
    ax.set(
        xlabel="Training bytes seen (millions)",
        ylabel="Loss (nats/byte)",
        title=f"Forge · {result['run']} · {dataset_name(result['dataset'])}",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "loss_curve.png", dpi=160)
    plt.close(fig)
