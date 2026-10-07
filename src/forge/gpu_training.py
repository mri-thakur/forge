"""Optional PyTorch training; portable checkpoint export for NumPy inference.

Not included in CPU-only validation when PyTorch is absent. Resume restores
weights, optimizer, training-data RNG, and schedule. No dropout is used; optional
bf16 autocast runs on CUDA only. Cross-device bitwise reproducibility is not promised.
"""

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from forge.model import ModelConfig, NumpyModel
from forge.thermal import ThermalGuard
from forge.torch_backend import TorchModel
from forge.training import sample_batch, sha256


def train_torch(
    data_dir,
    out,
    steps=500,
    dim=128,
    layers=4,
    context=128,
    batch=8,
    seed=17,
    resume=False,
    device="cuda",
    attention="manual",
    precision="fp32",
    heads=4,
    kv_heads=2,
    hidden=None,
    accumulate=1,
    lr=0.002,
    warmup=None,
    min_lr_ratio=0.1,
    eval_batches=8,
    eval_batch=2,
    rope_dtype="float32",
    init=None,
):
    """Train on byte or token shards described by the dataset's manifest.

    Each optimizer step averages `accumulate` micro-batches of `batch` windows, so
    a step sees batch * accumulate * context tokens. The learning rate warms up
    linearly, then follows a cosine down to `min_lr_ratio * lr`. Validation uses a
    fixed sample of `eval_batches * eval_batch` windows.

    `init` starts from a saved model.npz (fine-tuning) instead of random weights;
    the model shape then comes from that file. Datasets with a loss mask (SFT)
    score only masked-in tokens, and their windows start at example boundaries.
    """
    if precision == "bf16" and device != "cuda":
        raise ValueError("bf16 training is currently supported only on CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bf16")
    if accumulate < 1:
        raise ValueError("accumulate must be at least 1")
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for split in ("train", "val"):
        if sha256(data_dir / f"{split}.bin") != manifest[f"{split}_sha256"]:
            raise ValueError(f"{split} checksum mismatch")
    data_hash = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    dtype = np.dtype(manifest.get("dtype", "uint8"))
    vocab_size = manifest.get("vocab_size", 256)
    if manifest.get("loss_mask"):
        from forge.sft import load_sft_arrays

        splits = {name: load_sft_arrays(data_dir, name) for name in ("train", "val")}
    else:
        splits = {
            name: (np.memmap(data_dir / f"{name}.bin", dtype=dtype, mode="r"), None, None)
            for name in ("train", "val")
        }
    rng = np.random.default_rng(seed)
    warmup = min(20, steps // 10) if warmup is None else warmup
    run_config = {
        "batch": batch,
        "context": context,
        "seed": seed,
        "steps": steps,
        "attention": attention,
        "precision": precision,
        "accumulate": accumulate,
        "lr": lr,
        "warmup": warmup,
        "min_lr_ratio": min_lr_ratio,
        "eval_batches": eval_batches,
        "eval_batch": eval_batch,
        "rope_dtype": rope_dtype,
    }
    if init:
        run_config["init"] = {"path": str(init), "sha256": sha256(init)}
    checkpoint = out / "resume.pt"
    if checkpoint.exists() and not resume:
        raise ValueError("run exists; choose another --out or --resume")
    state = torch.load(checkpoint, map_location=device, weights_only=False) if resume else None
    if state and (state["data_hash"] != data_hash or state["total_steps"] != steps):
        raise ValueError("resume dataset/schedule mismatch")
    if state and state["run_config"] != run_config:
        raise ValueError("resume must preserve every training setting in run_config")
    if state:
        reference = NumpyModel(ModelConfig(**state["config"]))
    elif init:
        reference = NumpyModel.load(init)
    else:
        reference = NumpyModel(
            ModelConfig(
                vocab_size=vocab_size,
                dim=dim,
                layers=layers,
                heads=heads,
                kv_heads=kv_heads,
                hidden=hidden or dim * 3,
                context=max(512, context),
                seed=seed,
            )
        )
    config = reference.config
    if config.vocab_size != vocab_size:
        raise ValueError("checkpoint vocabulary does not match the dataset")
    if context > config.context:
        raise ValueError("training context exceeds the model's context")
    if rope_dtype not in ("float32", "bfloat16"):
        raise ValueError("rope_dtype must be float32 or bfloat16")
    model = TorchModel(reference, device, attention, getattr(torch, rope_dtype))
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    def autocast():
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    for key in model.weights:
        if state:
            with torch.no_grad():
                model.weights[key].copy_(state["weights"][key].detach())
        model.weights[key].requires_grad_(True)
    groups = [
        {"params": [w for w in model.weights.values() if w.ndim > 1], "weight_decay": 0.01},
        {"params": [w for w in model.weights.values() if w.ndim == 1], "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=device == "cuda")
    step = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        step = state["step"]

    usable_starts = {
        name: None if starts is None else starts[starts < len(tokens) - context]
        for name, (tokens, _, starts) in splits.items()
    }

    def sample(name, generator, rows):
        tokens, mask, _ = splits[name]
        starts = usable_starts[name]
        if starts is None:
            return (*sample_batch(tokens, generator, rows, context), None)
        begin = starts[generator.integers(0, len(starts), size=rows)]
        indices = begin[:, None] + np.arange(context)
        return (
            tokens[indices].astype(np.int64),
            tokens[indices + 1].astype(np.int64),
            mask[indices + 1],
        )

    def batch_loss(x, y, m=None):
        with autocast():
            logits = model.forward_tensor(x)
            targets = torch.tensor(y, device=device).reshape(-1)
            if m is None:
                return torch.nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size), targets
                )
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size), targets, reduction="none"
            )
            weights = torch.tensor(m, device=device, dtype=losses.dtype).reshape(-1)
            return (losses * weights).sum() / weights.sum().clamp(min=1)

    def validate():
        losses, validation_rng = [], np.random.default_rng(123)
        with torch.no_grad():
            for _ in range(eval_batches):
                losses.append(batch_loss(*sample("val", validation_rng, eval_batch)))
        return float(torch.stack(losses).mean())

    def save():
        temporary = out / "resume.pending.pt"
        torch.save(
            {
                "config": asdict(config),
                "weights": model.weights,
                "optimizer": optimizer.state_dict(),
                "rng": rng.bit_generator.state,
                "step": step,
                "data_hash": data_hash,
                "total_steps": steps,
                "run_config": run_config,
            },
            temporary,
        )
        temporary.replace(checkpoint)

    initial = validate()
    guard = ThermalGuard.for_device(device)
    start, tokens = time.perf_counter(), 0
    try:
        with (out / "training.jsonl").open("a", encoding="utf-8") as log:
            while step < steps:
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for _ in range(accumulate):
                    x, y, m = sample("train", rng, batch)
                    loss = batch_loss(x, y, m) / accumulate
                    loss.backward()
                    total += loss.detach()
                    tokens += x.size
                norm = torch.nn.utils.clip_grad_norm_(list(model.weights.values()), 1.0)
                if not torch.isfinite(total) or not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite training")
                step += 1
                progress = max(0, step - warmup) / max(1, steps - warmup)
                rate = lr * (
                    min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
                )
                if warmup and step <= warmup:
                    rate = lr * step / warmup
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.step()
                paused = guard.paused_seconds if guard else 0.0
                row = {
                    "step": step,
                    "loss": float(total),
                    "gradient_norm": float(norm.item()),
                    "lr": rate,
                    # Excludes thermal pauses, so it measures training speed.
                    "tokens_per_second": tokens / (time.perf_counter() - start - paused),
                }
                if step % 100 == 0 or step == steps:
                    row["val_loss"] = validate()
                    save()
                    print(json.dumps(row), flush=True)
                # The .item() calls above synchronized, so the GPU is idle here.
                if guard and guard.wait_if_hot():
                    row["thermal_paused_seconds_total"] = guard.paused_seconds
                log.write(json.dumps(row) + "\n")
                log.flush()
    except KeyboardInterrupt:
        save()
        raise
    reference = NumpyModel(
        config, {k: v.detach().cpu().numpy().copy() for k, v in model.weights.items()}
    )
    reference.save(out / "model.npz")
    final = validate()
    summary = {
        "backend": "torch",
        "device": device,
        "torch": torch.__version__,
        "precision": precision,
        "config": asdict(config),
        "run_config": run_config,
        "parameters": model.parameter_count,
        "steps": step,
        "tokens_this_session": tokens,
        "seconds_this_session": time.perf_counter() - start,
        "thermal_pauses": guard.pauses if guard else 0,
        "thermal_paused_seconds": guard.paused_seconds if guard else 0.0,
        "initial_val_loss_this_session": initial,
        "final_val_loss": final,
        "val_bits_per_token": final / math.log(2),
        # Byte shards: one token per byte. Token shards: scale by the split's ratio
        # (for masked SFT data, of scored tokens to story bytes).
        "val_bits_per_byte": final
        / math.log(2)
        * manifest.get("val_loss_tokens", manifest.get("val_tokens", 1))
        / manifest.get("val_text_bytes", manifest.get("val_tokens", 1)),
        "dataset": manifest,
    }
    if device == "cuda":
        summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    from forge.bench import environment

    model.weight_origin = "self-trained on the recorded dataset"
    summary["environment"] = environment(model)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
