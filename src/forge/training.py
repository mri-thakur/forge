"""Deterministic CPU training with explicit gradients, AdamW, and exact resume.

The included synthetic corpus is for pipeline validation only. A real-language
quality claim requires a named, licensed corpus and a separately held-out split.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from forge.autograd import Tensor, cross_entropy
from forge.model import ModelConfig, NumpyModel


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(out, source=None, synthetic=False, seed=17):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if synthetic == (source is not None):
        raise ValueError("choose exactly one of --text or --synthetic")
    if synthetic:
        rng = np.random.default_rng(seed)
        # Controlled grammar with variable digits and values; not natural-language evidence.
        records = []
        for i in range(2000):
            value = int(rng.integers(1, 10000))
            state = ["queued", "active", "completed"][i % 3]
            records.append(f'{{"id":{value},"state":"{state}","tokens":{value % 128}}}\n')
        raw = "".join(records).encode()
        origin = "forge synthetic JSON grammar; diagnostic only"
    else:
        raw = Path(source).read_bytes()
        origin = str(Path(source).name)
    if len(raw) < 2048:
        raise ValueError("corpus must contain at least 2048 bytes")
    split = int(len(raw) * 0.9)
    (out / "train.bin").write_bytes(raw[:split])
    (out / "val.bin").write_bytes(raw[split:])
    manifest = {
        "tokenizer": "utf8-bytes-256",
        "source": origin,
        "synthetic": synthetic,
        "seed": seed,
        "split": "chronological 90/10",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "train_tokens": split,
        "val_tokens": len(raw) - split,
        "train_sha256": sha256(out / "train.bin"),
        "val_sha256": sha256(out / "val.bin"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def sample_batch(data, rng, batch, context):
    if len(data) <= context:
        raise ValueError("split must exceed training context")
    starts = rng.integers(0, len(data) - context, size=batch)
    indices = starts[:, None] + np.arange(context)[None, :]
    return data[indices].astype(np.int64), data[indices + 1].astype(np.int64)


def evaluate(model, data, context=32, batches=4, seed=123):
    rng, losses = np.random.default_rng(seed), []
    for _ in range(batches):
        x, y = sample_batch(data, rng, 2, context)
        logits = model.forward(x)
        losses.append(float(cross_entropy(Tensor(logits), y).data))
    return float(np.mean(losses))


class Trainer:
    def __init__(
        self, model, seed=17, learning_rate=0.002, total_steps=200, warmup=10, weight_decay=0.01
    ):
        if total_steps < 1 or learning_rate <= 0 or warmup < 0:
            raise ValueError("invalid training schedule")
        self.model, self.rng = model, np.random.default_rng(seed)
        self.learning_rate, self.total_steps = learning_rate, total_steps
        self.warmup, self.weight_decay, self.step = warmup, weight_decay, 0
        self.m = {k: np.zeros_like(v) for k, v in model.weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in model.weights.items()}

    def update(self, x, y):
        logits, parameters = self.model.forward(x, grad=True)
        loss = cross_entropy(logits, y)
        loss.backward()
        norm = math.sqrt(
            sum(float(np.sum(p.grad.astype(np.float64) ** 2)) for p in parameters.values())
        )
        if not math.isfinite(float(loss.data)) or not math.isfinite(norm):
            raise FloatingPointError("nonfinite loss or gradient")
        clip = min(1.0, 1.0 / (norm + 1e-8))
        self.step += 1
        progress = max(0, self.step - self.warmup) / max(1, self.total_steps - self.warmup)
        lr = self.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress))))
        if self.warmup and self.step <= self.warmup:
            lr = self.learning_rate * self.step / self.warmup
        for key, parameter in parameters.items():
            gradient = (parameter.grad * clip).astype(np.float32)
            self.m[key] = 0.9 * self.m[key] + 0.1 * gradient
            self.v[key] = 0.95 * self.v[key] + 0.05 * gradient**2
            corrected_m = self.m[key] / (1 - 0.9**self.step)
            corrected_v = self.v[key] / (1 - 0.95**self.step)
            weight = self.model.weights[key]
            decay = self.weight_decay if weight.ndim > 1 else 0
            weight -= lr * (corrected_m / (np.sqrt(corrected_v) + 1e-8) + decay * weight)
        return {"step": self.step, "loss": float(loss.data), "gradient_norm": norm, "lr": lr}

    def save(self, path, data_hash):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": asdict(self.model.config),
            "rng": self.rng.bit_generator.state,
            "step": self.step,
            "data_hash": data_hash,
            "learning_rate": self.learning_rate,
            "total_steps": self.total_steps,
            "warmup": self.warmup,
            "weight_decay": self.weight_decay,
        }
        payload = {"metadata": json.dumps(meta)}
        for prefix, values in (("weight", self.model.weights), ("m", self.m), ("v", self.v)):
            payload.update({f"{prefix}:{key}": value for key, value in values.items()})
        # Replace only this run's checkpoint after the new archive is complete.
        temporary = path.with_name(path.stem + ".pending.npz")
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path, data_hash):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["data_hash"] != data_hash:
                raise ValueError("checkpoint data hash does not match dataset")
            weights = {
                k.split(":", 1)[1]: archive[k].copy()
                for k in archive.files
                if k.startswith("weight:")
            }
            model = NumpyModel(ModelConfig(**meta["config"]), weights)
            trainer = cls(
                model,
                learning_rate=meta["learning_rate"],
                total_steps=meta["total_steps"],
                warmup=meta["warmup"],
                weight_decay=meta["weight_decay"],
            )
            trainer.step = meta["step"]
            trainer.rng.bit_generator.state = meta["rng"]
            trainer.m = {k: archive[f"m:{k}"].copy() for k in weights}
            trainer.v = {k: archive[f"v:{k}"].copy() for k in weights}
        return trainer


def train(data_dir, out, steps=100, dim=32, layers=2, context=32, batch=4, seed=17, resume=False):
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if sha256(data_dir / "train.bin") != manifest["train_sha256"]:
        raise ValueError("training data checksum mismatch")
    if sha256(data_dir / "val.bin") != manifest["val_sha256"]:
        raise ValueError("validation data checksum mismatch")
    data_hash = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint8, mode="r")
    checkpoint = out / "resume.npz"
    run_config = {
        "batch": batch,
        "context": context,
        "seed": seed,
        "steps": steps,
        "data_hash": data_hash,
    }
    config_path = out / "run_config.json"
    if resume:
        if json.loads(config_path.read_text()) != run_config:
            raise ValueError("resume must preserve batch, context, seed, steps, and dataset")
        trainer = Trainer.load(checkpoint, data_hash)
        if steps != trainer.total_steps:
            raise ValueError("resume must preserve original --steps learning-rate schedule")
    else:
        if checkpoint.exists():
            raise ValueError("run already exists; choose a new --out or --resume")
        trainer = Trainer(
            NumpyModel(
                ModelConfig(
                    dim=dim, layers=layers, hidden=dim * 3, context=max(512, context), seed=seed
                )
            ),
            seed=seed,
            total_steps=steps,
            warmup=min(10, steps // 10),
        )
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    # Fixed validation samples do not consume training RNG.
    initial_val = evaluate(trainer.model, val_data, context)
    counts = np.bincount(train_data, minlength=256).astype(np.float64) + 1
    unigram = counts / counts.sum()
    baseline = float(-np.log(unigram[val_data]).mean())
    start, tokens = time.perf_counter(), 0
    try:
        with (out / "training.jsonl").open("a", encoding="utf-8") as log:
            while trainer.step < steps:
                x, y = sample_batch(train_data, trainer.rng, batch, context)
                row = trainer.update(x, y)
                tokens += x.size
                row["tokens_per_second"] = tokens / (time.perf_counter() - start)
                if trainer.step % 20 == 0 or trainer.step == steps:
                    row["val_loss_nats_per_byte"] = evaluate(trainer.model, val_data, context)
                    trainer.save(checkpoint, data_hash)
                    print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
    except KeyboardInterrupt:
        trainer.save(checkpoint, data_hash)
        raise
    trainer.model.save(out / "model.npz")
    final_val = evaluate(trainer.model, val_data, context)
    result = {
        "backend": "numpy-cpu-manual-autograd",
        "config": asdict(trainer.model.config),
        "parameters": trainer.model.parameter_count,
        "steps": trainer.step,
        "tokens_this_session": tokens,
        "seconds_this_session": time.perf_counter() - start,
        "initial_val_loss_this_session": initial_val,
        "final_val_loss": final_val,
        "val_bits_per_byte": final_val / math.log(2),
        "unigram_val_loss": baseline,
        "uniform_val_loss": math.log(256),
        "dataset": manifest,
        "quality_claim": "synthetic diagnostic only"
        if manifest["synthetic"]
        else "byte prediction on the named chronological held-out split",
    }
    from forge.bench import environment

    trainer.model.weight_origin = "self-trained on the recorded dataset"
    result["environment"] = environment(trainer.model)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
