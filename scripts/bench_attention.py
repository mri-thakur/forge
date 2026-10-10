"""Time one attention layer's forward + backward on the GPU for each way of handling
grouped-query attention, and record which SDPA kernels this PyTorch build provides.

    python scripts/bench_attention.py     # writes results/attention_kernels/attention.json

Shapes are the L model's attention for one training micro-batch (16 x 512 tokens,
8 query heads sharing 4 KV heads of width 64), in bf16.
"""

import json
import platform
import statistics
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "attention_kernels"
BATCH, TIME, HEADS, KV_HEADS, HEAD_DIM = 16, 512, 8, 4, 64
WARMUP, REPEATS = 10, 50
BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "memory_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def tensors():
    generator = torch.Generator(device="cuda").manual_seed(17)

    def make(heads):
        return torch.randn(
            BATCH,
            heads,
            TIME,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )

    return make(HEADS), make(KV_HEADS), make(KV_HEADS)


def gqa_flag(q, k, v):
    """PyTorch's own grouped-query support."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def copied_heads(q, k, v):
    """What Forge does: copy each KV head to its query heads, then plain causal SDPA."""
    repeats = HEADS // KV_HEADS
    k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def gqa_flag_cudnn(q, k, v):
    """The grouped-query flag with the cuDNN kernel forced: it supports GQA, but
    PyTorch's default kernel choice leaves it disabled in this build."""
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return gqa_flag(q, k, v)


def manual(q, k, v):
    """Attention written out: two matmuls and a masked softmax."""
    repeats = HEADS // KV_HEADS
    k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
    scores = (q @ k.transpose(-2, -1)) * HEAD_DIM**-0.5
    mask = torch.ones(TIME, TIME, device="cuda", dtype=torch.bool).tril()
    return scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v


def time_ms(function):
    q, k, v = tensors()
    grad = torch.randn(BATCH, HEADS, TIME, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    def run():
        function(q, k, v).backward(grad)
        q.grad = k.grad = v.grad = None

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(times), "min_ms": min(times), "repeats": REPEATS}


def availability(function):
    """Which SDPA kernels can run this call when forced one at a time."""
    q, k, v = tensors()
    result = {}
    for name, backend in BACKENDS.items():
        try:
            with sdpa_kernel(backend):
                function(q, k, v)
            torch.cuda.synchronize()
            result[name] = "ok"
        except RuntimeError as error:
            result[name] = "unavailable: " + str(error).strip().splitlines()[0][:160]
    return result


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)  # kernel-selection chatter
    variants = {
        "gqa_flag": gqa_flag,
        "gqa_flag_cudnn": gqa_flag_cudnn,
        "copied_heads": copied_heads,
        "manual": manual,
    }
    report = {
        "what": "one attention layer, forward + backward, bf16, CUDA events",
        "shape": {
            "batch": BATCH,
            "time": TIME,
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
        },
        "environment": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "os": platform.platform(),
            "flash_attention_built": torch.backends.cuda.is_flash_attention_available(),
        },
        "timing": {name: time_ms(function) for name, function in variants.items()},
        "kernels": {
            name: availability(function)
            for name, function in variants.items()
            if name in ("gqa_flag", "copied_heads")
        },
    }
    (OUT / "attention.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["median_ms"] for k, v in report["timing"].items()}))
    print(json.dumps(report["kernels"], indent=1))
    print("flash attention built:", report["environment"]["flash_attention_built"])


if __name__ == "__main__":
    main()
