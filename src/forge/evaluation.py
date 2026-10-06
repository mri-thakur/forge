"""Publishable language-model evidence for a finished training run.

Training-time validation scores a small fixed sample for monitoring. Here every
token of the held-out split is scored exactly once, next to n-gram baselines fit on
the same training tokens, the training curve, and fixed-seed samples. Byte-level
runs are the case of 256 tokens, one per byte. Bits per byte divide the total loss
by the split's text bytes, so they compare across tokenizers.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from forge.model import ByteTokenizer
from forge.training import sha256

PROMPTS = {
    # WikiText style: spaces around punctuation, " = Heading = " titles.
    "Salesforce/wikitext": (
        " = History of the railway = \n The railway",
        "The album was released in",
        "In the early 19th century , the town",
        "The species is found in",
    ),
    "roneneldan/TinyStories": (
        "Once upon a time",
        "Lily and Ben went to the park.",
        "The little dog was sad because",
        "One day, a girl named Mia found a",
    ),
}


def _negative_log_likelihood(model, x, y):
    if getattr(model, "backend", None) == "torch":
        import torch

        with torch.inference_mode():
            logits = model.forward_tensor(x.astype(np.int64)).float()
            targets = torch.as_tensor(y.astype(np.int64), device=logits.device).reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets, reduction="sum"
            )
        return float(loss)
    logits = model.forward(x.astype(np.int64)).astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return -float(np.take_along_axis(log_probs, y.astype(np.int64)[..., None], -1).sum())


def heldout_loss(model, data, context, batch=32):
    """Mean nats/token over every token of `data` after the first.

    Non-overlapping windows, so each prediction sees 1 to `context` preceding tokens
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


def ngram_baselines(train, heldout, vocab_size=256, chunk=1 << 24):
    """Add-one smoothed unigram and bigram models fit on the training tokens,
    scored on the same held-out targets as `heldout_loss`."""
    unigram = np.bincount(train, minlength=vocab_size).astype(np.float64) + 1
    unigram /= unigram.sum()
    pairs = np.zeros(vocab_size * vocab_size)
    for start in range(0, len(train) - 1, chunk):
        stop = min(start + chunk, len(train) - 1)
        codes = np.asarray(train[start:stop], dtype=np.int64) * vocab_size
        codes += train[start + 1 : stop + 1]
        pairs += np.bincount(codes, minlength=vocab_size * vocab_size)
    bigram = pairs.reshape(vocab_size, vocab_size) + 1
    bigram /= bigram.sum(axis=1, keepdims=True)
    previous, target = np.asarray(heldout[:-1]), np.asarray(heldout[1:])
    return {
        "uniform": math.log(vocab_size),
        "unigram": float(-np.log(unigram[target]).mean()),
        "bigram": float(-np.log(bigram[previous, target]).mean()),
    }


def training_curve(log_path, tokens_per_update, every=100):
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
            "tokens": last["step"] * tokens_per_update,
            "train_loss": float(np.mean([row["loss"] for row in group])),
            "lr": last["lr"],
            "tokens_per_second": last["tokens_per_second"],
        }
        if "val_loss" in last:
            point["monitor_val_loss"] = last["val_loss"]
        curve.append(point)
    return curve


def fixed_samples(
    model, context, tokenizer, prompts, stop=None, tokens=200, temperature=0.8, top_p=0.95, seed=17
):
    """Greedy and nucleus continuations of fixed prompts through the serving engine,
    cut at the `stop` token when one is given."""
    from forge.engine import Engine

    encoded = [tokenizer.encode(prompt) for prompt in prompts]
    # Stay within the trained context; later RoPE positions were never trained.
    tokens = min(tokens, context - max(map(len, encoded)))
    if tokens < 1:
        return {"skipped": f"the {context}-token training context cannot hold the fixed prompts"}
    engine = Engine(model, max_batch=2 * len(prompts), blocks=256, seed=seed)
    requests = []
    for i, (prompt, ids) in enumerate(zip(prompts, encoded)):
        greedy = engine.submit(f"greedy-{i}", ids, tokens)
        sampled = engine.submit(f"sampled-{i}", ids, tokens, temperature=temperature, top_p=top_p)
        requests.append((prompt, greedy, sampled))
    while engine.busy:
        engine.step()

    def text(generated):
        if stop is not None and stop in generated:
            generated = generated[: generated.index(stop)]
        return tokenizer.decode(generated)

    return {
        "settings": {
            "tokens": tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "engine_policy": engine.policy,
            "stops_at_end_of_text": stop is not None,
        },
        "samples": [
            {"prompt": prompt, "greedy": text(greedy.generated), "sampled": text(sampled.generated)}
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
    dtype = np.dtype(manifest.get("dtype", "uint8"))
    vocab_size = manifest.get("vocab_size", 256)
    train = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
    val = np.memmap(data_dir / "val.bin", dtype=dtype, mode="r")
    if "tokenizer_file" in manifest:
        from forge.bpe import ENDOFTEXT, Tokenizer

        tokenizer = Tokenizer.load(data_dir / manifest["tokenizer_file"])
        stop = tokenizer.special[ENDOFTEXT]
    else:
        tokenizer, stop = ByteTokenizer, None
    prompts = PROMPTS.get(manifest.get("dataset"), PROMPTS["Salesforce/wikitext"])

    start = time.perf_counter()
    loss = heldout_loss(model, val, context, batch)
    eval_seconds = time.perf_counter() - start
    scored = len(val) - 1
    # Story text bytes for token shards; byte shards have one token per byte.
    text_bytes = manifest.get("val_text_bytes", scored)
    baselines = ngram_baselines(train, val, vocab_size)
    tokens_per_update = settings["batch"] * settings.get("accumulate", 1) * context
    curve = training_curve(run / "training.jsonl", tokens_per_update)
    samples = fixed_samples(model, context, tokenizer, prompts, stop)
    default_eval = (8, 2) if summary["backend"] == "torch" else (4, 2)
    result = {
        "run": run.name,
        "heldout": {
            "split": "validation",
            "tokens_scored": scored,
            "text_bytes": text_bytes,
            "method": f"non-overlapping {context}-token windows; each token predicted once "
            f"from 1 to {context} preceding tokens",
            "precision": "fp32",
            "loss_nats_per_token": loss,
            "bits_per_token": loss / math.log(2),
            "bits_per_byte": loss * scored / text_bytes / math.log(2),
            "seconds": eval_seconds,
        },
        "monitor_val_loss_at_end": summary["final_val_loss"],
        "monitor_sample_tokens": settings.get("eval_batches", default_eval[0])
        * settings.get("eval_batch", default_eval[1])
        * context,
        "baselines_nats_per_token": baselines,
        "training": {
            "parameters": summary["parameters"],
            "config": summary["config"],
            "run_config": settings,
            "updates": summary["steps"],
            "training_tokens": summary["steps"] * tokens_per_update,
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
    if "config" in manifest:
        return f"{manifest['dataset']} ({manifest['config']})"
    return manifest.get("dataset", manifest.get("source"))


def write_report(result, curve, report_dir, results_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    held, base = result["heldout"], result["baselines_nats_per_token"]
    train, manifest = result["training"], result["dataset"]
    c = train["config"]
    vocab = c["vocab_size"]
    byte_level = vocab == 256 and "tokenizer_file" not in manifest
    unit = "byte" if byte_level else "token"
    peak = train["cuda_peak_allocated_bytes"]
    link = Path(os.path.relpath(results_dir, report_dir)).as_posix()
    count = train["parameters"]
    size = f"{count / 1e6:.1f}M" if count >= 1e6 else f"{count / 1e3:.0f}K"
    kind = "byte-level" if byte_level else f"{vocab:,}-token BPE"
    lines = [
        f"# {result['run']}: held-out evaluation",
        "",
        f"A {train['parameters']:,}-parameter {kind} decoder ({c['layers']} layers, "
        f"width {c['dim']}, {c['heads']} heads / {c['kv_heads']} KV heads) trained for "
        f"{train['updates']:,} updates on {train['training_tokens'] / 1e6:.2f}M {unit}s of "
        f"{dataset_name(manifest)} in "
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
        f"Scored on all {held['tokens_scored']:,} predicted {unit}s of the {held['split']} "
        f"split (split: {manifest['split']}; {held['method']}; {held['precision']}).",
        "",
        f"| Model | nats/{unit} | bits/byte |",
        "|---|---:|---:|",
        f"| **Forge {size}** | **{held['loss_nats_per_token']:.4f}** | "
        f"**{held['bits_per_byte']:.4f}** |",
    ]
    to_bits_per_byte = held["tokens_scored"] / held["text_bytes"] / math.log(2)
    noun = "Byte" if byte_level else "Token"
    for name, label in (
        ("bigram", f"{noun} bigram, add-one, fit on the same training {unit}s"),
        ("unigram", f"{noun} unigram, add-one, fit on the same training {unit}s"),
        ("uniform", f"Uniform over {vocab:,} {unit}s"),
    ):
        lines.append(f"| {label} | {base[name]:.4f} | {base[name] * to_bits_per_byte:.4f} |")
    comparability = (
        "Byte-level losses are not comparable to published BPE or word-level perplexities."
        if byte_level
        else "Per-token losses depend on the tokenizer; bits per byte compare across "
        "tokenizers on the same text."
    )
    lines += [
        "",
        f"The training loop's monitoring loss at the last update was "
        f"{result['monitor_val_loss_at_end']:.4f} nats/{unit}, measured on a fixed sample of "
        f"{result['monitor_sample_tokens']:,} {unit}s; the full-split number above is the one "
        f"to cite. {comparability}",
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
        stops = ", stopping at end-of-text" if settings.get("stops_at_end_of_text") else ""
        lines += [
            f"Up to {settings['tokens']} {unit}s per prompt through Forge's serving engine "
            f"(`{settings['engine_policy']}` policy{stops}): greedy, then temperature "
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
    millions = [point["tokens"] / 1e6 for point in curve]
    ax.plot(
        millions, [p["train_loss"] for p in curve], lw=1.2, label="training (mean per 100 updates)"
    )
    monitored = [
        (p["tokens"] / 1e6, p["monitor_val_loss"]) for p in curve if "monitor_val_loss" in p
    ]
    if monitored:
        ax.plot(
            *zip(*monitored),
            lw=1,
            alpha=0.7,
            label=f"monitoring validation ({result['monitor_sample_tokens']:,}-{unit} sample)",
        )
    for name, style in (("bigram", "--"), ("unigram", ":")):
        ax.axhline(base[name], ls=style, color="gray", lw=1, label=f"{name} baseline")
    ax.scatter(
        [millions[-1]],
        [held["loss_nats_per_token"]],
        marker="*",
        s=140,
        zorder=3,
        color="black",
        label=f"full validation split: {held['loss_nats_per_token']:.3f}",
    )
    ax.set_ylim(held["loss_nats_per_token"] - 0.2, base["unigram"] + 0.3)
    ax.set(
        xlabel=f"Training {unit}s seen (millions)",
        ylabel=f"Loss (nats/{unit})",
        title=f"Forge · {result['run']} · {dataset_name(manifest)}",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "loss_curve.png", dpi=160)
    plt.close(fig)
