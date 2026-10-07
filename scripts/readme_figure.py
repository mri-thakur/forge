"""Build the README's headline figure (docs/headline.png and its dark variant).

    python scripts/readme_figure.py

Held-out instruction following across the pipeline, read from results/sft/ and
results/phase4/; nothing is typed in by hand.
"""

import json
import statistics
from pathlib import Path

from forge.instruct import load_jsonl, repetition

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
THEMES = {
    # The validated categorical slots of phase2_report.py: blue for the pipeline,
    # orange for the reward-hacking policy; gray for the human-written reference.
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "pipeline": "#2a78d6",
        "hacking": "#eb6834",
        "reference": "#b5b4ad",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "pipeline": "#3987e5",
        "hacking": "#d95926",
        "reference": "#5f5e5a",
    },
}


def rate(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))["satisfied"]["rate"]


def bars():
    loops = statistics.mean(
        repetition(row["story"]) >= 0.10
        for row in load_jsonl(ROOT / "results/phase4/grpo_kl0_samples.jsonl")
    )
    return [
        # (label, value, kind, note)
        ("Pretrained model", rate("results/sft/base_L_eval.json"), "pipeline", None),
        ("+ supervised fine-tuning", rate("results/sft/sft_L_eval.json"), "pipeline", None),
        ("+ RL (GRPO, KL penalty)", rate("results/phase4/grpo_kl0.1_eval.json"), "pipeline", None),
        (
            "RL without KL penalty\n(games the checks)",
            rate("results/phase4/grpo_kl0_eval.json"),
            "hacking",
            f"{loops:.0%} of stories loop",
        ),
        ("Dataset's own stories", rate("results/sft/gold_eval.json"), "reference", "reference"),
    ]


def plot(theme_name, path):
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
            "ytick.color": theme["secondary"],
            "axes.facecolor": theme["surface"],
            "figure.facecolor": theme["surface"],
            "grid.color": theme["grid"],
            "grid.linewidth": 0.8,
        }
    )
    rows = bars()
    positions = [0, 1, 2, 3.4, 4.4]  # a gap separates the pipeline from the contrasts
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for y, (label, value, kind, note) in zip(positions, rows):
        ax.barh(y, 100 * value, height=0.62, color=theme[kind])
        text = f"{100 * value:.1f}%" + (f"  ({note})" if note and kind == "hacking" else "")
        ax.annotate(
            text,
            (100 * value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=theme["secondary"],
            fontsize=9.5,
        )
    ax.set_yticks(positions, [label for label, *_ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Held-out samples meeting every instruction (%), 500 prompts x 4 samples")
    ax.set_title(
        "Following story instructions: required words, a given sentence, dialogue",
        loc="left",
        color=theme["ink"],
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    plot("light", OUT / "headline.png")
    plot("dark", OUT / "headline-dark.png")
    print(json.dumps({label: round(value, 4) for label, value, *_ in bars()}))


if __name__ == "__main__":
    main()
