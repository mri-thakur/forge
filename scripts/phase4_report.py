"""Build docs/grpo/RESULTS.md and its figure from results/phase4/ and results/sft/.

    python scripts/phase4_report.py

Every number comes from the saved result files (and the fixed evaluation prompts in
data/instruct/eval_prompts.jsonl for the constraints and gold stories); nothing is
typed in by hand.
"""

import json
import statistics
import textwrap
from pathlib import Path

import numpy as np

from forge.instruct import WORD, check, forms, load_jsonl, normalize, prompt, repetition

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase4"
SFT = ROOT / "results" / "sft"
OUT = ROOT / "docs" / "grpo"
PROMPTS = ROOT / "data" / "instruct" / "eval_prompts.jsonl"
MAIN = "grpo_kl0.1"
POLICIES = {  # label: (results prefix, training log or None)
    "SFT": (SFT / "sft_L", None),
    "KL 1": (RESULTS / "grpo_kl1", "grpo_kl1"),
    "KL 0.1": (RESULTS / MAIN, MAIN),
    "KL 0": (RESULTS / "grpo_kl0", "grpo_kl0"),
}
SNAPSHOTS = (50, 100, 150)
THEMES = {
    # The validated categorical slots 1-3 (blue, orange, aqua) of phase2_report.py.
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}
MARKERS = ["o", "s", "^"]


def pct(value):
    return f"{100 * value:.1f}%"


def words(text):
    return WORD.findall(normalize(text))


def sentence_starts(sentence, story, n=4):
    """How often the story starts the given sentence (its first n words in order)."""
    head, tokens = words(sentence)[:n], words(story)
    if len(head) < n:
        return None
    return sum(tokens[i : i + n] == head for i in range(len(tokens) - n + 1))


def word_mentions(record, story):
    """Mean number of times each (single-word) required word appears, any form."""
    tokens = words(story)
    counts = [
        sum(token in forms(word) for token in tokens)
        for word in record["words"]
        if WORD.fullmatch(word.lower())
    ]
    return statistics.mean(counts) if counts else None


def bootstrap(records, rows, draws=2000, seed=17):
    """95% interval of the satisfied rate, resampling prompts (their 4 samples
    are not independent)."""
    per_prompt = {}
    for row in rows:
        per_prompt.setdefault(row["id"], []).append(check(records[row["id"]], row["story"]))
    rates = np.array(
        [np.mean([r["satisfied"] for r in results]) for results in per_prompt.values()]
    )
    rng = np.random.default_rng(seed)
    means = rates[rng.integers(0, len(rates), size=(draws, len(rates)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def fluency(prefix):
    if prefix.parent == SFT:
        return json.loads((SFT / "fluency.json").read_text())["files"]["sft_L_samples.jsonl"]
    files = json.loads(Path(f"{prefix}_fluency.json").read_text())["files"]
    return next(iter(files.values()))


def load_policy(prefix, records):
    evaluation = json.loads(Path(f"{prefix}_eval.json").read_text())
    rows = load_jsonl(f"{prefix}_samples.jsonl")
    return summarize(evaluation, rows, fluency(prefix)["nats_per_token"], records)


def summarize(evaluation, rows, nats, records):
    stories = [row["story"] for row in rows]
    starts = [
        c
        for row in rows
        if records[row["id"]]["sentence"]
        for c in [sentence_starts(records[row["id"]]["sentence"], row["story"])]
        if c is not None
    ]
    mentions = [
        m
        for row in rows
        for m in [word_mentions(records[row["id"]], row["story"])]
        if m is not None
    ]
    pc = evaluation["per_constraint"]
    return {
        "satisfied": evaluation["satisfied"]["rate"],
        "reward": evaluation["mean_reward"],
        "word_recall": pc["word_recall"]["rate"],
        "all_words": pc["all_words"]["rate"],
        "sentence": pc["sentence"]["rate"],
        "dialogue": pc["dialogue"]["rate"],
        "finished": evaluation["finished_rate"],
        "fluency": nats,
        "length": statistics.mean(len(words(s)) for s in stories),
        "repetition": statistics.mean(repetition(s) for s in stories),
        "loops": statistics.mean(repetition(s) >= 0.10 for s in stories),
        "sentence_starts": statistics.mean(starts),
        "sentence_restarts": statistics.mean(c >= 2 for c in starts),
        "word_mentions": statistics.mean(mentions),
        "rows": rows,
    }


def training(name):
    log = [json.loads(line) for line in (RESULTS / f"{name}_training.jsonl").open()]
    summary = json.loads((RESULTS / f"{name}_summary.json").read_text())
    first, last = log[:25], log[-25:]
    return {
        "minutes": summary["seconds_this_session"] / 60,
        "pauses": summary["thermal_pauses"],
        "reward_first": statistics.mean(r["reward"] for r in first),
        "reward_last": statistics.mean(r["reward"] for r in last),
        "kl_last": statistics.mean(r["kl"] for r in last),
        "entropy_first": statistics.mean(r["entropy"] for r in first),
        "entropy_last": statistics.mean(r["entropy"] for r in last),
        "run_config": summary["run_config"],
        "pool": summary["prompt_pool"],
    }


def plot(curve, policies, gold, theme_name, path):
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
        ax.grid(True)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    steps = [step for step, _ in curve]
    series = (
        ("satisfied", "All constraints met"),
        ("all_words", "All required words"),
        ("sentence", "Given sentence included"),
    )
    for i, (key, label) in enumerate(series):
        values = [100 * p[key] for _, p in curve]
        left.plot(
            steps,
            values,
            marker=MARKERS[i],
            markersize=8,
            linewidth=2,
            color=theme["series"][i],
            markeredgecolor=theme["surface"],
            markeredgewidth=2,
            label=label,
        )
        left.annotate(
            f"{values[-1]:.1f}%",
            (steps[-1], values[-1]),
            xytext=(8, -3),
            textcoords="offset points",
            color=theme["secondary"],
            fontsize=9,
        )
    left.set_xticks(steps)
    left.set_xlim(-8, steps[-1] + 32)
    left.set_ylim(0, 100)
    left.set_xlabel("GRPO step (0 = the SFT model)")
    left.set_ylabel("Held-out samples (%)")
    left.set_title(
        "Held-out success during RL, KL 0.1", loc="left", color=theme["ink"], fontsize=11
    )
    legend = left.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.33))
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    labels = list(policies)
    x = [policies[label]["fluency"] for label in labels]
    y = [100 * policies[label]["satisfied"] for label in labels]
    color = theme["series"][0]
    right.plot(x, y, color=color, linewidth=2, zorder=2)
    right.plot(
        x,
        y,
        "o",
        markersize=9,
        color=color,
        markeredgecolor=theme["surface"],
        markeredgewidth=2,
        zorder=3,
    )
    offsets = {"SFT": (10, -4), "KL 1": (10, -4), "KL 0.1": (10, -4), "KL 0": (0, 11)}
    names = {"KL 0": "KL 0 (reward hacking)"}
    for label, xi, yi in zip(labels, x, y):
        right.annotate(
            names.get(label, label),
            (xi, yi),
            xytext=offsets[label],
            textcoords="offset points",
            ha="center" if label == "KL 0" else "left",
            color=theme["secondary"],
            fontsize=9,
        )
    right.plot(
        gold["fluency"],
        100 * gold["satisfied"],
        "D",
        markersize=8,
        color=theme["muted"],
        markeredgecolor=theme["surface"],
        markeredgewidth=2,
    )
    right.annotate(
        "Gold stories",
        (gold["fluency"], 100 * gold["satisfied"]),
        xytext=(10, -4),
        textcoords="offset points",
        color=theme["muted"],
        fontsize=9,
    )
    right.set_xlim(0.9, 1.45)
    right.set_ylim(0, 100)
    right.set_xlabel("Story loss under pretrained L (nats/token, lower = more typical)")
    right.set_ylabel("All constraints met (%)")
    right.set_title(
        "The KL penalty trades success against fluency", loc="left", color=theme["ink"], fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def story_block(record, row, max_lines=None):
    lines = []
    for paragraph in row["story"].split("\n"):
        lines += textwrap.wrap(paragraph, 88) or [""]
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"[... {len(lines) - max_lines} more lines like these]"]
    lines = ["```text", *lines]
    result = check(record, row["story"])
    parts = []
    if "words" in result:
        parts.append(f"{sum(result['words']['hits'])} of {len(record['words'])} words")
    if "sentence" in result:
        parts.append("sentence " + ("included" if result["sentence"] else "missing"))
    if "dialogue" in result:
        parts.append("dialogue " + ("present" if result["dialogue"] else "missing"))
    ending = "" if row.get("finished", True) else "; the story never ended"
    check_line = f"Check: {', '.join(parts)}; reward {result['reward']:.2f}{ending}."
    return lines + ["```", "", check_line, ""]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {r["id"]: r for r in load_jsonl(PROMPTS)}
    policies = {label: load_policy(prefix, records) for label, (prefix, _) in POLICIES.items()}
    gold_rows = [{"id": r["id"], "sample": 0, "story": r["story"]} for r in records.values()]
    gold = summarize(
        json.loads((SFT / "gold_eval.json").read_text()),
        gold_rows,
        json.loads((SFT / "fluency.json").read_text())["files"]["eval_prompts.jsonl"][
            "nats_per_token"
        ],
        records,
    )
    intervals = {label: bootstrap(records, p["rows"]) for label, p in policies.items()}
    curve = [(0, policies["SFT"])]
    curve += [(s, load_policy(RESULTS / f"{MAIN}_step{s:04d}", records)) for s in SNAPSHOTS]
    curve.append((200, policies["KL 0.1"]))
    runs = {label: training(name) for label, (_, name) in POLICIES.items() if name}
    plot(curve, policies, gold, "light", OUT / "grpo.png")
    plot(curve, policies, gold, "dark", OUT / "grpo-dark.png")

    sft, main_, none = policies["SFT"], policies["KL 0.1"], policies["KL 0"]
    config = runs["KL 0.1"]["run_config"]
    lr_text = f"{config['lr']:g}".replace("e-0", "e-")
    columns = list(policies) + ["Gold"]
    table = {**policies, "Gold": gold}

    def row(label, key, fmt):
        return f"| {label} | " + " | ".join(fmt(table[c][key]) for c in columns) + " |"

    def interval(label):
        lo, hi = intervals[label]
        return f"{100 * lo:.1f}-{100 * hi:.1f}"

    lines = [
        "# Phase 4: reinforcement learning with verifiable rewards",
        "",
        "GRPO from the phase 3 SFT model, with the rule-based constraint checks as the "
        f"reward. Each of {config['steps']} steps samples {config['group']} stories for each "
        f"of {config['prompts_per_step']} training prompts at temperature 1 through the "
        "batched KV-cache engine. A story's reward is the fraction of its prompt's "
        "constraint types met (0 if it never ends within "
        f"{config['max_new_tokens']} tokens), its advantage is the reward minus its "
        "group's mean over the group's standard deviation, and the loss is the policy "
        "gradient on its tokens plus a penalty on the KL divergence from the frozen SFT "
        "model, computed exactly over the vocabulary at every position "
        f"([`forge/grpo.py`](../../src/forge/grpo.py)). Learning rate {lr_text}, "
        f"{runs['KL 0.1']['minutes']:.0f} minutes per run on the RTX 4050. Prompts come "
        f"from a pool of {runs['KL 0.1']['pool']:,} training records; the evaluation "
        "prompts are never used. Generated by `scripts/phase4_report.py` from "
        "`results/phase4/`.",
        "",
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="grpo-dark.png">',
        '  <img alt="Left: held-out constraint satisfaction at GRPO steps 0 to 200. '
        "Right: constraint satisfaction against story loss under the pretrained model "
        'for SFT and three KL penalty settings" src="grpo.png">',
        "</picture>",
        "",
        "## Held-out results",
        "",
        "The same 500 held-out prompts and settings as phase 3: 4 samples each at "
        "temperature 0.8 and top-p 0.95. KL is the coefficient of the KL penalty; 0.1 is "
        "the main run, 0 and 1 are the ablation.",
        "",
        "| | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
        "| **All constraints met** | "
        + " | ".join(f"**{pct(table[c]['satisfied'])}**" for c in columns)
        + " |",
        "| 95% interval (bootstrap over prompts) | "
        + " | ".join(interval(c) for c in policies)
        + " | |",
        row("Mean reward", "reward", lambda v: f"{v:.3f}"),
        row("Required words found (per word)", "word_recall", pct),
        row("All required words found", "all_words", pct),
        row("Given sentence included", "sentence", pct),
        row("Dialogue when requested", "dialogue", pct),
        row("Story ended", "finished", pct),
        row("Fluency: nats/token under pretrained L", "fluency", lambda v: f"{v:.3f}"),
        row("Mean length (words)", "length", lambda v: f"{v:.0f}"),
        row("Repeated word 4-grams", "repetition", pct),
        "| KL from SFT per token (training, last 25 steps) | 0 | "
        + " | ".join(f"{runs[c]['kl_last']:.4f}" for c in runs)
        + " | |",
        "",
        f"With the KL coefficient at 0.1, GRPO raised the share of held-out samples that "
        f"meet every constraint from {pct(sft['satisfied'])} to {pct(main_['satisfied'])}. "
        f"All required words went from {pct(sft['all_words'])} to "
        f"{pct(main_['all_words'])} and finished stories from {pct(sft['finished'])} to "
        f"{pct(main_['finished'])}, at a small cost in fluency "
        f"({sft['fluency']:.3f} → {main_['fluency']:.3f} nats/token under the pretrained "
        f"model) and none in repetition ({pct(sft['repetition'])} → "
        f"{pct(main_['repetition'])}). The given sentence barely moved "
        f"({pct(sft['sentence'])} → {pct(main_['sentence'])}).",
        "",
        "Held-out results of the main run every 50 steps:",
        "",
        "| Step | " + " | ".join(str(s) for s, _ in curve) + " |",
        "|---|" + "---:|" * len(curve),
        "| All constraints met | " + " | ".join(pct(p["satisfied"]) for _, p in curve) + " |",
        "| All required words | " + " | ".join(pct(p["all_words"]) for _, p in curve) + " |",
        "| Given sentence | " + " | ".join(pct(p["sentence"]) for _, p in curve) + " |",
        "| Fluency (nats/token) | " + " | ".join(f"{p['fluency']:.3f}" for _, p in curve) + " |",
        "",
        f"Most of the gain came in the first {SNAPSHOTS[0]} steps "
        f"({pct(curve[0][1]['satisfied'])} → {pct(curve[1][1]['satisfied'])}).",
        "",
        "## Reward hacking without the KL penalty",
        "",
        f"Without the penalty the reward climbed furthest ({pct(none['satisfied'])} of "
        "samples satisfied), but by gaming the checks, which only test that a word or "
        "sentence appears. Measured on the same samples:",
        "",
        "| | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
        row("Stories looping (≥10% repeated 4-grams)", "loops", pct),
        row("Starts of the given sentence per story", "sentence_starts", lambda v: f"{v:.2f}"),
        row("Stories starting it twice or more", "sentence_restarts", pct),
        row("Mentions per required word", "word_mentions", lambda v: f"{v:.2f}"),
        "",
        "(Sentence rows: prompts with a given sentence; a start is its first four words "
        "in order.) Without the penalty, "
        f"{pct(none['loops'])} of stories loop, against {pct(sft['loops'])} for SFT; "
        f"stories start the given sentence {none['sentence_starts']:.1f} times on average, "
        "retrying near-copies until one matches exactly; and each required word appears "
        f"{none['word_mentions']:.1f} times, against {gold['word_mentions']:.1f} in the "
        "gold stories, sometimes forced into the wrong part of speech. With the penalty "
        "at 0.1 all four stay near the SFT model's levels. At 1 the policy barely moved "
        f"({pct(policies['KL 1']['satisfied'])} satisfied).",
        "",
        "The first evaluation prompt, sample 0 of each policy (not cherry-picked):",
        "",
    ]
    first = next(iter(records.values()))
    lines += ["```text", prompt(first).rstrip(), "```", ""]
    for label in ("SFT", "KL 0.1", "KL 0"):
        sample = next(r for r in policies[label]["rows"] if r["id"] == first["id"])
        lines += [f"**{label}**", ""] + story_block(first, sample)
    with_sentence = [
        (sentence_starts(records[r["id"]]["sentence"], r["story"]) or 0, -i, r)
        for i, r in enumerate(none["rows"])
        if records[r["id"]]["sentence"]
    ]
    _, _, worst = max(with_sentence)
    lines += [
        "Selected to show the sentence failure: the KL 0 sample that starts its given "
        f'sentence most often. The prompt asked for: "{records[worst["id"]]["sentence"]}"',
        "",
        *story_block(records[worst["id"]], worst, max_lines=8),
    ]
    lines += [
        "## Training",
        "",
        "| KL coefficient | Minutes | Thermal pauses | Reward, first → last 25 steps "
        "| KL at the end | Entropy, first → last 25 steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, run in runs.items():
        lines.append(
            f"| {label[3:]} | {run['minutes']:.0f} | {run['pauses']} | "
            f"{run['reward_first']:.3f} → {run['reward_last']:.3f} | {run['kl_last']:.4f} | "
            f"{run['entropy_first']:.3f} → {run['entropy_last']:.3f} |"
        )
    lines += [
        "",
        "Training rewards are on different prompts each step and at temperature 1, so "
        "they are noisier and lower than the held-out numbers above.",
        "",
        "## Limitations",
        "",
        "- One seed per setting. The 95% intervals above cover sampling noise over "
        "prompts, not run-to-run variation of RL itself.",
        "- The checks reward a word's presence, not its use, and the sentence check rewards "
        "a verbatim copy wherever it lands. The KL penalty is what keeps the policy "
        "honest here; a stricter reward (grammatical use, a single copy of the sentence) "
        "would be the next step.",
        "- The given sentence remains the weak point: the main run's stories start it "
        f"{main_['sentence_starts']:.2f} times on average against "
        f"{gold['sentence_starts']:.2f} in the gold stories, so the policy mostly does not "
        "try. The only policy that includes it often does so by brute force.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                label: [round(p["satisfied"], 4), [round(v, 4) for v in intervals[label]]]
                for label, p in policies.items()
            }
        )
    )


if __name__ == "__main__":
    main()
