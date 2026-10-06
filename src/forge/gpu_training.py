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
):
    if precision == "bf16" and device != "cuda":
        raise ValueError("bf16 training is currently supported only on CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bf16")
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for split in ("train", "val"):
        if sha256(data_dir / f"{split}.bin") != manifest[f"{split}_sha256"]:
            raise ValueError(f"{split} checksum mismatch")
    data_hash = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    data = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
    val = np.memmap(data_dir / "val.bin", dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    run_config = {
        "batch": batch,
        "context": context,
        "seed": seed,
        "steps": steps,
        "attention": attention,
        "precision": precision,
    }
    checkpoint = out / "resume.pt"
    if checkpoint.exists() and not resume:
        raise ValueError("run exists; choose another --out or --resume")
    state = torch.load(checkpoint, map_location=device, weights_only=False) if resume else None
    if state and (state["data_hash"] != data_hash or state["total_steps"] != steps):
        raise ValueError("resume dataset/schedule mismatch")
    if state and state["run_config"] != run_config:
        raise ValueError(
            "resume must preserve batch, context, seed, schedule, attention, and precision"
        )
    config = (
        ModelConfig(**state["config"])
        if state
        else ModelConfig(
            dim=dim, layers=layers, hidden=dim * 3, context=max(512, context), seed=seed
        )
    )
    model = TorchModel(NumpyModel(config), device, attention)
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
    optimizer = torch.optim.AdamW(groups, lr=0.002, betas=(0.9, 0.95))
    step = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        step = state["step"]

    def validate():
        losses, validation_rng = [], np.random.default_rng(123)
        with torch.no_grad():
            for _ in range(8):
                x, y = sample_batch(val, validation_rng, 2, context)
                with autocast():
                    logits = model.forward_tensor(x)
                    targets = torch.tensor(y, device=device).reshape(-1)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets)
                losses.append(loss.item())
        return float(np.mean(losses))

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
    start, tokens = time.perf_counter(), 0
    try:
        with (out / "training.jsonl").open("a", encoding="utf-8") as log:
            while step < steps:
                x, y = sample_batch(data, rng, batch, context)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model.forward_tensor(x)
                    targets = torch.tensor(y, device=device).reshape(-1)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(list(model.weights.values()), 1.0)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite training")
                step += 1
                warmup = min(20, steps // 10)
                progress = max(0, step - warmup) / max(1, steps - warmup)
                lr = 0.002 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
                if warmup and step <= warmup:
                    lr = 0.002 * step / warmup
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                tokens += x.size
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "gradient_norm": float(norm.item()),
                    "lr": lr,
                    "tokens_per_second": tokens / (time.perf_counter() - start),
                }
                if step % 100 == 0 or step == steps:
                    row["val_loss"] = validate()
                    save()
                    print(json.dumps(row), flush=True)
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
        "initial_val_loss_this_session": initial,
        "final_val_loss": final,
        "val_bits_per_byte": final / math.log(2),
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
