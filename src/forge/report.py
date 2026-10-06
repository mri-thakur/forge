"""Generate tables/figures from saved results; no hand-entered speedups."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def report(input_path, out):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in data["runs"]:
        groups[(row["arrival_rate"], row["policy"])].append(row)
    lines = [
        "# Measured scheduling results",
        "",
        f"Backend: `{data['environment']['backend']}` / `{data['environment']['device']}`. "
        f"Parameters: {data['environment']['parameters']:,}. Workload: `{data['workload']}`.",
        "",
        "Each cell is the median across saved seeds. TTFT includes queue time. "
        "Throughput and goodput include drain time. Rejected requests remain in the "
        "SLO-fraction denominator. These are in-process results, not HTTP server results.",
        "",
        "| Offered req/s | Policy | Repeats | Output tok/s | p99 TTFT ms | SLO fraction | "
        "Goodput req/s |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (rate, policy), rows in sorted(groups.items()):
        values = [
            float(np.median([row[key] for row in rows]))
            for key in [
                "output_tokens_per_second",
                "ttft_p99_ms",
                "slo_fraction_all_submissions",
                "slo_goodput_requests_per_second",
            ]
        ]
        lines.append(
            f"| {rate:g} | {policy} | {len(rows)} | {values[0]:.1f} | {values[1]:.1f} | "
            f"{values[2]:.1%} | {values[3]:.2f} |"
        )
    lines.extend(
        [
            "",
            "With small traces, p99 is descriptive and statistically uncertain. "
            "Use larger request counts and multiple independent repetitions before "
            "making deployment-capacity or resume claims.",
            "",
        ]
    )
    path = out / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return str(path)
    for key, title, filename in [
        ("output_tokens_per_second", "Output throughput (tokens/s)", "throughput.png"),
        ("ttft_p99_ms", "p99 time to first token (ms)", "ttft.png"),
        ("slo_goodput_requests_per_second", "SLO goodput (requests/s)", "goodput.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        policies = sorted({policy for _, policy in groups})
        for policy in policies:
            rates = sorted(rate for rate, p in groups if p == policy)
            values = [np.median([r[key] for r in groups[(rate, policy)]]) for rate in rates]
            low = [min(r[key] for r in groups[(rate, policy)]) for rate in rates]
            high = [max(r[key] for r in groups[(rate, policy)]) for rate in rates]
            ax.plot(rates, values, marker="o", label=policy)
            ax.fill_between(rates, low, high, alpha=0.12)
        ax.set(
            xlabel="Offered arrival rate (requests/s)",
            ylabel=title,
            title=f"Forge · {data['workload']} · {data['environment']['backend']}",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
    return str(path)
