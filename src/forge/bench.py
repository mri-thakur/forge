"""Open-loop traces and SLO goodput, including queueing, drops, and drain time.

Arrivals are generated independently of service completion. Late submissions
retain their intended arrival timestamps, preventing coordinated omission. This
in-process benchmark includes Python scheduling/sampling but excludes HTTP/SSE.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from forge.engine import Engine, contiguous_generate, reference_generate


def environment(model):
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    source = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    result = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "logical_cpus": os.cpu_count(),
        "cpu": platform.processor(),
        "backend": model.backend,
        "device": str(model.device),
        "parameters": model.parameter_count,
        "model_config": asdict(model.config),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
        "thread_env": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
    }
    weights = hashlib.sha256()
    for key, weight in sorted(model.weights.items()):
        array = weight.detach().cpu().numpy() if model.backend == "torch" else weight
        weights.update(key.encode())
        weights.update(array.tobytes())
    result["weights_sha256"] = weights.hexdigest()
    if model.backend == "torch":
        import torch

        result["torch"] = torch.__version__
        result["attention"] = model.attention
        result["torch_threads"] = torch.get_num_threads()
        if str(model.device).startswith("cuda"):
            result["gpu"] = torch.cuda.get_device_name()
            result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            result["cuda_runtime"] = torch.version.cuda
    return result


def make_trace(count=24, rate=20.0, workload="mixed", seed=17, context=512, vocab=256):
    if count < 1 or rate <= 0 or context < 32:
        raise ValueError("invalid workload parameters")
    rng = np.random.default_rng(seed)
    trace, arrival = [], 0.0
    shared = rng.integers(0, vocab, size=min(48, context // 4)).tolist()
    for i in range(count):
        if workload == "burst":
            arrival = (i // 8) * 8 / rate
        else:
            arrival += float(rng.exponential(1 / rate)) if i else 0
        short = min(32, context // 4)
        prompt_len = (
            short
            if workload == "short"
            else int(rng.choice([short, short * 2, min(256, context // 2)]))
        )
        output_len = int(rng.choice([8, 16, 32]))
        output_len = min(output_len, context - prompt_len)
        prompt = rng.integers(0, vocab, size=prompt_len).tolist()
        if workload == "prefix":
            prompt = shared + prompt[:short]
        if workload == "head_of_line":
            prompt = rng.integers(
                0, vocab, size=min(context - 32, 384) if i % 8 == 0 else short
            ).tolist()
        if len(prompt) + output_len > context:
            prompt = prompt[: context - output_len]
        trace.append(
            {"id": f"r{i:04d}", "arrival": arrival, "prompt": prompt, "max_tokens": output_len}
        )
    return trace


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def summarize(requests, elapsed, ttft_slo, tpot_slo):
    completed = [r for r in requests if r.status == "completed"]
    ttft = [r.emissions[0] - r.arrival for r in completed]
    tpot = [
        (r.emissions[-1] - r.emissions[0]) / (len(r.emissions) - 1) if len(r.emissions) > 1 else 0
        for r in completed
    ]
    good = [
        r
        for r, first, per_token in zip(completed, ttft, tpot)
        if first <= ttft_slo and per_token <= tpot_slo
    ]
    intervals = [b - a for r in completed for a, b in zip(r.emissions, r.emissions[1:])]
    return {
        "submitted": len(requests),
        "completed": len(completed),
        "rejected": sum(r.status == "rejected" for r in requests),
        "cancelled": sum(r.status == "cancelled" for r in requests),
        "elapsed_including_drain_s": elapsed,
        "output_tokens": sum(len(r.generated) for r in completed),
        "output_tokens_per_second": sum(len(r.generated) for r in completed) / elapsed,
        "slo_requests": len(good),
        "slo_fraction_all_submissions": len(good) / len(requests),
        "slo_goodput_requests_per_second": len(good) / elapsed,
        "ttft_p50_ms": percentile(ttft, 50) * 1000 if ttft else None,
        "ttft_p95_ms": percentile(ttft, 95) * 1000 if ttft else None,
        "ttft_p99_ms": percentile(ttft, 99) * 1000 if ttft else None,
        "inter_token_p99_ms": percentile(intervals, 99) * 1000 if intervals else None,
        "tpot_p95_ms": percentile(tpot, 95) * 1000 if tpot else None,
        "e2e_p99_ms": percentile([r.finished - r.arrival for r in completed], 99) * 1000
        if completed
        else None,
        "slo": {"ttft_ms": ttft_slo * 1000, "mean_tpot_ms": tpot_slo * 1000},
        "per_request": [
            {
                "id": r.id,
                "status": r.status,
                "reason": r.reason,
                "prompt_tokens": len(r.prompt),
                "output_tokens": len(r.generated),
                "queue_wait_ms": (r.admitted - r.arrival) * 1000
                if r.admitted is not None
                else None,
                "ttft_ms": (r.emissions[0] - r.arrival) * 1000 if r.emissions else None,
            }
            for r in requests
        ],
    }


def run_trace(model, trace, policy, ttft_slo=0.5, tpot_slo=0.05, **settings):
    engine = Engine(model, policy=policy, record_metrics=True, **settings)
    # Warmup uses a separate engine and does not prime the measured prefix cache.
    warmup = Engine(
        model, blocks=settings.get("blocks", 128), block_size=settings.get("block_size", 16)
    )
    warmup.submit("warmup", [1, 2, 3], 2)
    while warmup.busy:
        warmup.step()
    del warmup
    if model.backend == "torch" and str(model.device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()
    start, cursor = time.perf_counter(), 0
    while cursor < len(trace) or engine.busy:
        now = time.perf_counter()
        while cursor < len(trace) and trace[cursor]["arrival"] <= now - start:
            row = trace[cursor]
            engine.submit(row["id"], row["prompt"], row["max_tokens"], start + row["arrival"])
            cursor += 1
        if engine.busy:
            engine.step()
        elif cursor < len(trace):
            time.sleep(min(0.002, max(0, start + trace[cursor]["arrival"] - now)))
    elapsed = time.perf_counter() - start
    engine.pool.assert_integrity()
    summary = summarize(list(engine.requests.values()), elapsed, ttft_slo, tpot_slo)
    summary.update(
        {
            "policy": policy,
            "settings": settings,
            "cache": engine.pool.snapshot(),
            "iterations": engine.iterations,
            "executed_tokens": engine.executed_tokens,
            "padding_tokens": engine.padding_tokens,
        }
    )
    if engine.cache_samples:
        summary["cache"]["peak_live_tokens"] = max(s["live_tokens"] for s in engine.cache_samples)
        summary["cache"]["peak_reserved_token_slots"] = max(
            s["reserved_token_slots"] for s in engine.cache_samples
        )
        summary["cache"]["median_reserved_slot_utilization"] = float(
            np.median(
                [
                    s["live_tokens"] / s["reserved_token_slots"]
                    for s in engine.cache_samples
                    if s["reserved_token_slots"]
                ]
            )
        )
    if model.backend == "torch" and str(model.device).startswith("cuda"):
        summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    engine.pool.clear_prefixes()
    engine.pool.assert_integrity()
    assert engine.pool.used_blocks == 0
    return summary


def benchmark(
    model,
    out,
    count=24,
    rates=(10, 30),
    workload="mixed",
    seeds=(17, 29, 43),
    policies=("static", "continuous", "chunked"),
    max_batch=8,
    token_budget=64,
    blocks=128,
    block_size=16,
    prefix_entries=0,
    ttft_slo=0.5,
    tpot_slo=0.05,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 1,
        "environment": environment(model),
        "scope": "in-process open-loop; includes scheduler and sampling; excludes HTTP",
        "weights": model.weight_origin,
        "workload": workload,
        "runs": [],
    }
    for rate in rates:
        for seed in seeds:
            trace = make_trace(
                count, rate, workload, seed, model.config.context, model.config.vocab_size
            )
            (out / f"trace_{rate:g}_{seed}.json").write_text(
                json.dumps(trace) + "\n", encoding="utf-8"
            )
            digest = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
            # Rotate policy order to reduce systematic warmup/thermal ordering bias.
            order = list(policies)
            rotation = seeds.index(seed) % len(order)
            order = order[rotation:] + order[:rotation]
            for policy in order:
                row = run_trace(
                    model,
                    trace,
                    policy,
                    ttft_slo,
                    tpot_slo,
                    max_batch=max_batch,
                    token_budget=token_budget,
                    blocks=blocks,
                    block_size=block_size,
                    prefix_entries=prefix_entries,
                    seed=seed,
                )
                row.update({"arrival_rate": rate, "seed": seed, "trace_sha256": digest})
                results["runs"].append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in [
                                "policy",
                                "arrival_rate",
                                "seed",
                                "output_tokens_per_second",
                                "ttft_p99_ms",
                                "slo_fraction_all_submissions",
                            ]
                        }
                    ),
                    flush=True,
                )
                (out / "benchmark.json").write_text(
                    json.dumps(results, indent=2) + "\n", encoding="utf-8"
                )
    return results


def cache_ladder(model, out, repeats=3, prompt_length=64, output_length=24):
    rng = np.random.default_rng(17)
    prompt = rng.integers(0, model.config.vocab_size, size=prompt_length).tolist()
    expected = reference_generate(model, prompt, output_length)
    rows = []
    for name in ("recompute_reference", "contiguous_kv", "paged_kv_single_request"):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            if name == "recompute_reference":
                output = reference_generate(model, prompt, output_length)
            elif name == "contiguous_kv":
                output = contiguous_generate(model, prompt, output_length)
            else:
                engine = Engine(
                    model,
                    max_batch=1,
                    token_budget=prompt_length,
                    blocks=(prompt_length + output_length + 15) // 16,
                )
                request = engine.submit("one", prompt, output_length)
                while engine.busy:
                    engine.step()
                output = request.generated
            times.append(time.perf_counter() - start)
            if output != expected:
                raise AssertionError("greedy tokens differ from reference")
        rows.append(
            {
                "path": name,
                "median_seconds": float(np.median(times)),
                "output_tokens_per_second": output_length / float(np.median(times)),
                "seconds_repeats": times,
            }
        )
    result = {
        "environment": environment(model),
        "prompt_tokens": prompt_length,
        "output_tokens": output_length,
        "equivalent_greedy_output": True,
        "rows": rows,
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
