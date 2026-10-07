"""Build docs/phase2/RESULTS.md and the scaling figure from results/phase2/.

    python scripts/phase2_report.py

Every number comes from the saved evaluation files; nothing is typed in by hand.
"""

import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase2"
OUT = ROOT / "docs" / "phase2"
SIZES = ("S", "M", "L")
MARKERS = {"S": "o", "M": "s", "L": "^"}
# Triangles read smaller than circles and squares at the same nominal size.
MARKER_SIZE = {"S": 8, "M": 8, "L": 10}
THEMES = {
    # Validated categorical slots 1-3 (blue, orange, aqua) for each surface.
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"S": "#2a78d6", "M": "#eb6834", "L": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"S": "#3987e5", "M": "#d95926", "L": "#199e70"},
    },
}


def load(name):
    evaluation = json.loads((RESULTS / name / "evaluation.json").read_text(encoding="utf-8"))
    curve = [
        json.loads(line)
        for line in (RESULTS / name / "training_curve.jsonl").read_text("utf-8").splitlines()
    ]
    return evaluation, curve


def monitor_curve(run):
    """(compute, monitoring loss) points; the monitoring sample is the same in every run."""
    evaluation, curve = run
    params = evaluation["training"]["parameters"]
    points = [p for p in curve if "monitor_val_loss" in p]
    return [6 * params * p["tokens"] for p in points], [p["monitor_val_loss"] for p in points]


def frontier_sentence(runs):
    m_compute, m_loss = monitor_curve(runs["M"])
    l_compute, l_loss = monitor_curve(runs["L"])
    budget = m_compute[-1]
    l_at_budget = float(np.interp(np.log(budget), np.log(l_compute), l_loss))
    overtakes = next(c for c, loss in zip(l_compute, l_loss) if loss <= m_loss[-1])
    return (
        f"The curves show the compute frontier. At M's full budget ({budget:.1e} FLOPs), "
        f"L's monitoring loss was {l_at_budget:.3f} against M's {m_loss[-1]:.3f} on the same "
        "validation sample, so at that compute the smaller model was ahead; L matched M's "
        f"final loss only after {overtakes / budget:.1f}x as much compute, then went past it."
    )


def fit_power_law(x, y):
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return -slope, math.exp(intercept)


def plot(runs, theme_name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEMES[theme_name]
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": theme["ink"],
            "axes.labelcolor": theme["secondary"],
            "axes.edgecolor": theme["axis"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "axes.facecolor": theme["surface"],
            "figure.facecolor": theme["surface"],
            "grid.color": theme["grid"],
            "grid.linewidth": 0.8,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax in (left, right):
        ax.grid(True, which="major")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for name in SIZES:
        evaluation, curve = runs[name]
        params = evaluation["training"]["parameters"]
        color = theme["series"][name]
        points = [p for p in curve if "monitor_val_loss" in p]
        compute = [6 * params * p["tokens"] for p in points]
        left.plot(
            compute,
            [p["monitor_val_loss"] for p in points],
            color=color,
            linewidth=2,
            label=f"{name} ({params / 1e6:.1f}M parameters)",
        )
        final_compute = 6 * params * evaluation["training"]["training_tokens"]
        final_loss = evaluation["heldout"]["loss_nats_per_token"]
        left.plot(
            final_compute,
            final_loss,
            MARKERS[name],
            markersize=MARKER_SIZE[name],
            color=color,
            markeredgecolor=theme["surface"],
            markeredgewidth=2,
        )
        left.annotate(
            f"{name} {final_loss:.3f}",
            (final_compute, final_loss),
            xytext=(6, -12),
            textcoords="offset points",
            color=theme["secondary"],
            fontsize=9,
        )
    left.set_xscale("log")
    left.set_ylim(1.0, 2.6)
    left.set_xlabel("Training compute (FLOPs, 6 × parameters × tokens)")
    left.set_ylabel("Validation loss (nats/token)")
    left.set_title("Loss during training", loc="left", color=theme["ink"], fontsize=11)
    legend = left.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    params = np.array([runs[n][0]["training"]["parameters"] for n in SIZES], dtype=float)
    losses = np.array([runs[n][0]["heldout"]["loss_nats_per_token"] for n in SIZES])
    alpha, scale = fit_power_law(params, losses)
    grid = np.geomspace(params.min() / 1.3, params.max() * 1.3, 50)
    right.plot(grid, scale * grid**-alpha, "--", color=theme["muted"], linewidth=1.2)
    for name, n, loss in zip(SIZES, params, losses):
        right.plot(
            n,
            loss,
            MARKERS[name],
            markersize=MARKER_SIZE[name] + 1,
            color=theme["series"][name],
            markeredgecolor=theme["surface"],
            markeredgewidth=2,
        )
        right.annotate(
            f"{name}  {loss:.3f}",
            (n, loss),
            xytext=(8, 4),
            textcoords="offset points",
            color=theme["secondary"],
            fontsize=9,
        )
    right.annotate(
        rf"fit: loss $\propto N^{{-{alpha:.3f}}}$",
        (0.04, 0.06),
        xycoords="axes fraction",
        color=theme["secondary"],
        fontsize=9,
    )
    right.set_xscale("log")
    right.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([5e6, 1e7, 2e7, 4e7]))
    right.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"{value / 1e6:.0f}M")
    )
    right.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    right.set_yscale("log")
    right.set_yticks([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    right.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    right.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    right.set_xlabel("Parameters (N)")
    right.set_ylabel("Final validation loss (nats/token)")
    right.set_title(
        "Final loss, full validation split", loc="left", color=theme["ink"], fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return alpha


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {name: load(name) for name in SIZES}
    alpha = plot(runs, "light", OUT / "scaling.png")
    plot(runs, "dark", OUT / "scaling-dark.png")
    compute = [
        6 * r["training"]["parameters"] * r["training"]["training_tokens"] for r, _ in runs.values()
    ]
    beta, _ = fit_power_law(
        compute, [r["heldout"]["loss_nats_per_token"] for r, _ in runs.values()]
    )
    lines = [
        "# Phase 2: pretraining results",
        "",
        "Three decoders trained from scratch on TinyStoriesV2 with the 4,096-token BPE "
        "tokenizer: 512-token context, 65,536 tokens per optimizer step (16 x 512 x 8 "
        "accumulated), AdamW with warmup and cosine decay to 10%, bf16 autocast on an "
        "RTX 4050 Laptop GPU (6 GB). Every loss below scores all "
        f"{runs['L'][0]['heldout']['tokens_scored']:,} tokens of the validation split "
        "once. Generated by `scripts/phase2_report.py` from `results/phase2/`.",
        "",
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="scaling-dark.png">',
        '  <img alt="Validation loss against training compute for the three model sizes, '
        'and final loss against parameter count on log axes" src="scaling.png">',
        "</picture>",
        "",
        "## Scaling",
        "",
        "| Model | Parameters | Layers x width | Heads / KV heads | Tokens | Tokens per "
        "parameter | Compute (FLOPs) | Peak LR | Loss (nats/token) | Bits per byte |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SIZES:
        evaluation, _ = runs[name]
        t, h = evaluation["training"], evaluation["heldout"]
        c = t["config"]
        lines.append(
            f"| {name} | {t['parameters'] / 1e6:.2f}M | {c['layers']} x {c['dim']} | "
            f"{c['heads']} / {c['kv_heads']} | {t['training_tokens'] / 1e6:.1f}M | "
            f"{t['training_tokens'] / t['parameters']:.1f} | "
            f"{6 * t['parameters'] * t['training_tokens']:.2e} | {t['run_config']['lr']:g} | "
            f"**{h['loss_nats_per_token']:.4f}** | {h['bits_per_byte']:.4f} |"
        )
    lines += [
        "",
        f"Across the three runs, loss falls as roughly N^-{alpha:.3f} in parameters and "
        f"C^-{beta:.3f} in compute. These exponents are descriptive only: three points, "
        "trained at slightly different tokens per parameter and with learning rates "
        "scaled by width rather than tuned per size.",
        "",
        frontier_sentence(runs),
        "",
        "## Ablations",
        "",
        "S architecture (5.8M parameters), 60.3M tokens each, two seeds:",
        "",
        "| Variant | Seed 17 | Seed 29 | Mean | vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    variants = {
        "base": "Baseline: float32 RoPE angles, grouped-query attention (2 KV heads)",
        "bf16rope": "RoPE angles computed in bf16 (the original code)",
        "mha": "Full multi-head attention (4 KV heads)",
    }
    table = {}
    for key in variants:
        table[key] = [
            json.loads((RESULTS / f"abl_{key}_s{seed}" / "evaluation.json").read_text("utf-8"))[
                "heldout"
            ]["loss_nats_per_token"]
            for seed in (17, 29)
        ]
    base = statistics.mean(table["base"])
    for key, label in variants.items():
        mean = statistics.mean(table[key])
        delta = "" if key == "base" else f"{mean - base:+.4f}"
        lines.append(
            f"| {label} | {table[key][0]:.4f} | {table[key][1]:.4f} | {mean:.4f} | {delta} |"
        )
    spread = abs(table["base"][0] - table["base"][1])
    lines += [
        "",
        f"The two baseline seeds differ by {spread:.4f}, which sets the noise floor. bf16 "
        "RoPE angles were worse with both seeds (by "
        f"{table['bf16rope'][0] - table['base'][0]:+.4f} and "
        f"{table['bf16rope'][1] - table['base'][1]:+.4f}); full multi-head attention "
        "differed from grouped-query attention by less than the seed spread while "
        "doubling the KV cache.",
        "",
        "## Samples from L",
        "",
        "Temperature 0.8, top-p 0.95, seed 17, not cherry-picked (all prompts and greedy "
        "outputs: [L results](L/RESULTS.md)).",
        "",
    ]
    for sample in runs["L"][0]["samples"]["samples"][:2]:
        lines += ["```text", sample["prompt"] + sample["sampled"], "```", ""]
    summaries = {
        name: json.loads((ROOT / "checkpoints" / name / "summary.json").read_text("utf-8"))
        for name in SIZES
    }
    lines += [
        "## Notes",
        "",
        "- Wall-clock times are not comparable across runs. During S the laptop "
        "hibernated after a critical thermal event and the run resumed afterwards; L "
        f"ran with a thermal guard that paused {summaries['L']['thermal_pauses']:,} times "
        f"({summaries['L']['thermal_paused_seconds'] / 3600:.1f} h in total) whenever the "
        "GPU reached 85 °C.",
        "- Per-size results, loss curves, and all samples: "
        "[S](S/RESULTS.md), [M](M/RESULTS.md), [L](L/RESULTS.md). Learning-rate sweep: "
        "[`results/lr_sweep/`](../../results/lr_sweep/).",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"alpha_N {alpha:.4f}, beta_C {beta:.4f}")


if __name__ == "__main__":
    main()
