"""Reinforcement learning with verifiable rewards: GRPO on story instructions.

Each step takes `prompts` training instructions that have at least one checkable
constraint and samples `group` stories for each from the current policy through the
batched KV-cache engine, at temperature 1 without top-p so that the samples come
from the policy itself. A story's reward is `forge.instruct.check`'s score (the
fraction of constraint types met, required words counted fractionally), or 0 when
the story did not end within `max_new_tokens`, so rambling until the required words
turn up does not pay.

Advantages are group-relative (GRPO; Shao et al., 2024): a story's reward minus its
group's mean, divided by the group's standard deviation. The other stories for the
same prompt are the baseline, so no value network is needed. A group whose stories
all scored the same has zero advantage and contributes only the KL term.

The loss over the sampled story tokens o_it is

    L = -1/N sum_it A_i log pi(o_it) + kl * 1/N sum_it KL(pi || pi_sft)_t

with N the number of story tokens in the step, so each token weighs the same however
long its story is. Each batch of rollouts is used for exactly one optimizer step, so
the policy being updated is the one that sampled them: PPO's probability ratio is 1
(up to numerical precision) and its clipping never triggers, so it is left out. With
a 4,096-token vocabulary the KL to the frozen SFT model is computed exactly over the
vocabulary at every sampled position, rather than estimated from the sampled token.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from forge.bpe import ENDOFTEXT, Tokenizer
from forge.instruct import check, load_jsonl, prompt, verifiable
from forge.model import NumpyModel
from forge.sft import offline_engine, run_to_completion, sample_jsonl
from forge.thermal import ThermalGuard
from forge.torch_backend import TorchModel
from forge.training import sha256


def prompt_pool(path, size, tokenizer, max_prompt_tokens, seed=17, exclude=()):
    """Up to `size` random training records that have a verifiable constraint and a
    prompt of at most `max_prompt_tokens` tokens, in a fixed shuffled order."""
    records = sample_jsonl(path, size, random.Random(seed), set(exclude))
    return [
        r
        for r in records
        if verifiable(r) and len(tokenizer.encode_ordinary(prompt(r))) <= max_prompt_tokens
    ]


def rollouts(model, tokenizer, records, group, max_new_tokens, tag, seed=17, max_batch=64):
    """Sample `group` stories per record at temperature 1, stopping at end-of-text.
    Returns one dict per story with its prompt and generated token ids."""
    end = tokenizer.special[ENDOFTEXT]
    context = model.config.context
    engine = offline_engine(model, max_batch, len(records) * group, seed)
    pending = []
    for record in records:
        ids = tokenizer.encode_ordinary(prompt(record))
        for k in range(group):
            request = engine.submit(
                f"{tag}/{record['id']}#{k}",
                ids,
                min(max_new_tokens, context - len(ids)),
                temperature=1.0,
                stop=end,
            )
            pending.append((record, ids, request))
    run_to_completion(engine, [request for _, _, request in pending])
    samples = []
    for record, ids, request in pending:
        generated = list(request.generated)
        finished = bool(generated) and generated[-1] == end
        story = tokenizer.decode(generated[:-1] if finished else generated).strip()
        samples.append(
            {
                "record": record,
                "prompt": ids,
                "tokens": generated,
                "finished": finished,
                "story": story,
            }
        )
    return samples


def score(sample):
    """The constraint check of one rollout, with the reward set to 0 if it never ended."""
    result = check(sample["record"], sample["story"])
    if not sample["finished"]:
        result["reward"], result["satisfied"] = 0.0, False
    return result


def group_advantages(rewards, group, eps=1e-6):
    """(reward - group mean) / (group std + eps) for consecutive groups of `group`."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group)
    centered = r - r.mean(axis=1, keepdims=True)
    return (centered / (r.std(axis=1, keepdims=True) + eps)).reshape(-1)


def pack(samples, device):
    """Inputs, targets, and a mask selecting the positions that predict story tokens
    (the end-of-text token included), right-padded to the longest sequence. Causal
    attention never lets a real position see the padding."""
    width = max(len(s["prompt"]) + len(s["tokens"]) for s in samples) - 1
    x = np.zeros((len(samples), width), dtype=np.int64)
    y = np.zeros_like(x)
    mask = np.zeros(x.shape, dtype=np.float32)
    for row, s in enumerate(samples):
        sequence = s["prompt"] + s["tokens"]
        n = len(sequence) - 1
        x[row, :n] = sequence[:-1]
        y[row, :n] = sequence[1:]
        mask[row, len(s["prompt"]) - 1 : n] = 1
    return tuple(torch.as_tensor(a, device=device) for a in (x, y, mask))


def token_terms(policy, reference, x, y, autocast=nullcontext):
    """Per position: log-probability of the target under the policy (with gradient),
    exact KL(policy || reference) over the vocabulary, and the policy's entropy."""
    with autocast():
        logits = policy.forward_tensor(x)
    log_probs = logits.float().log_softmax(-1)
    # no_grad, not inference_mode: inference tensors cannot be saved for backward.
    with torch.no_grad(), autocast():
        reference_logits = reference.forward_tensor(x)
    reference_log_probs = reference_logits.float().log_softmax(-1)
    probs = log_probs.exp()
    chosen = log_probs.gather(-1, y[..., None]).squeeze(-1)
    kl = (probs * (log_probs - reference_log_probs)).sum(-1)
    entropy = -(probs * log_probs).sum(-1).detach()
    return chosen, kl, entropy


def accumulate_gradients(
    policy, reference, samples, advantages, kl_coef, micro_batch=16, autocast=nullcontext
):
    """Backpropagate the GRPO loss of one step, in micro-batches of similar length.
    Returns the loss and per-token means of the KL, entropy, and sampled-token
    log-probability."""
    device = policy.device
    lengths = [len(s["prompt"]) + len(s["tokens"]) for s in samples]
    order = sorted(range(len(samples)), key=lengths.__getitem__)
    total = sum(len(s["tokens"]) for s in samples)
    loss_sum, sums = 0.0, {"kl": 0.0, "entropy": 0.0, "log_prob": 0.0}
    for start in range(0, len(order), micro_batch):
        chunk = order[start : start + micro_batch]
        x, y, mask = pack([samples[i] for i in chunk], device)
        advantage = torch.as_tensor(advantages[chunk], device=device, dtype=torch.float32)
        chosen, kl, entropy = token_terms(policy, reference, x, y, autocast)
        penalty = (kl * mask).sum()
        loss = (-(advantage[:, None] * chosen * mask).sum() + kl_coef * penalty) / total
        loss.backward()
        loss_sum += float(loss.detach())
        sums["kl"] += float(penalty.detach())
        sums["entropy"] += float((entropy * mask).sum())
        sums["log_prob"] += float((chosen.detach() * mask).sum())
    return {"loss": loss_sum, **{key: value / total for key, value in sums.items()}}


def _mean(values):
    return sum(values) / len(values) if values else None


def step_metrics(samples, results, advantages, group):
    words = [r["words"]["fraction"] for r in results if "words" in r]
    sentence = [r["sentence"] for r in results if "sentence" in r]
    dialogue = [r["dialogue"] for r in results if "dialogue" in r]
    rewards = np.array([r["reward"] for r in results]).reshape(-1, group)
    return {
        "reward": float(rewards.mean()),
        "satisfied": _mean([r["satisfied"] for r in results]),
        "finished": _mean([s["finished"] for s in samples]),
        "story_tokens": _mean([len(s["tokens"]) for s in samples]),
        "word_recall": _mean(words),
        "sentence": _mean(sentence),
        "dialogue": _mean(dialogue),
        "informative_groups": float((rewards.std(axis=1) > 0).mean()),
        "mean_abs_advantage": float(np.abs(advantages).mean()),
    }


def train_grpo(
    init,
    prompts,
    tokenizer,
    out,
    steps=200,
    prompts_per_step=16,
    group=8,
    lr=3e-5,
    kl=0.1,
    max_new_tokens=384,
    warmup=10,
    micro_batch=16,
    max_batch=128,
    pool_size=20_000,
    exclude=None,
    save_every=25,
    seed=17,
    device="cuda",
    attention="sdpa",
    precision="bf16",
    resume=False,
    snapshot_every=0,
):
    """GRPO from the policy in `init` (a model.npz), which also stays frozen as the KL
    reference. `prompts` is a JSONL of instruction records (the training split);
    records whose id is in the `exclude` JSONL are never used. With `resume`, an
    existing checkpoint in `out` is continued (a fresh run starts if there is none).
    `snapshot_every` > 0 also saves the policy as model_step<N>.npz every that many
    steps, for evaluating how the policy changes during training.
    """
    if precision == "bf16" and device != "cuda":
        raise ValueError("bf16 is supported only on CUDA")
    if group < 2:
        raise ValueError("group-relative advantages need at least 2 samples per prompt")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path, tokenizer = tokenizer, Tokenizer.load(tokenizer)
    run_config = {
        "init": {"path": str(init), "sha256": sha256(init)},
        "prompts": str(prompts),
        "tokenizer_sha256": sha256(tokenizer_path),
        "exclude": str(exclude) if exclude else None,
        "steps": steps,
        "prompts_per_step": prompts_per_step,
        "group": group,
        "lr": lr,
        "kl": kl,
        "max_new_tokens": max_new_tokens,
        "warmup": warmup,
        "micro_batch": micro_batch,
        "pool_size": pool_size,
        "seed": seed,
        "attention": attention,
        "precision": precision,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.0, "clip": 1.0},
    }
    checkpoint = out / "resume.pt"
    if checkpoint.exists() and not resume:
        raise ValueError("run exists; choose another --out or --resume")
    state = (
        torch.load(checkpoint, map_location=device, weights_only=False)
        if resume and checkpoint.exists()
        else None
    )
    if state and state["run_config"] != run_config:
        raise ValueError("resume must preserve every setting in run_config")

    sft = NumpyModel.load(init)
    config = sft.config
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match the tokenizer")
    held_out = {r["id"] for r in load_jsonl(exclude)} if exclude else set()
    started = time.perf_counter()
    pool = prompt_pool(
        prompts, pool_size, tokenizer, config.context - max_new_tokens, seed, held_out
    )
    if len(pool) < prompts_per_step:
        raise ValueError("too few usable prompts")
    pool_ids = hashlib.sha256("\n".join(r["id"] for r in pool).encode()).hexdigest()
    if state and state["pool_sha256"] != pool_ids:
        raise ValueError("the prompt pool changed since the run started")
    print(
        json.dumps({"prompt_pool": len(pool), "seconds": time.perf_counter() - started}),
        flush=True,
    )

    policy = TorchModel(sft, device, attention)
    reference = TorchModel(sft, device, attention)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    for key, weight in policy.weights.items():
        if state:
            with torch.no_grad():
                weight.copy_(state["weights"][key])
        weight.requires_grad_(True)
    weights = list(policy.weights.values())
    optimizer = torch.optim.AdamW(
        weights, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=device == "cuda"
    )
    step = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]

    def autocast():
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def export(path):
        weights = {k: v.detach().cpu().numpy().copy() for k, v in policy.weights.items()}
        NumpyModel(config, weights).save(path)

    def save():
        temporary = out / "resume.pending.pt"
        torch.save(
            {
                "config": asdict(config),
                "weights": policy.weights,
                "optimizer": optimizer.state_dict(),
                "step": step,
                "run_config": run_config,
                "pool_sha256": pool_ids,
            },
            temporary,
        )
        temporary.replace(checkpoint)

    # Prompts are taken in a fixed shuffled order of the pool, so a resumed run
    # sees exactly the prompts the uninterrupted run would have.
    order = np.random.default_rng(seed).permutation(len(pool))
    guard = ThermalGuard.for_device(device)
    start = time.perf_counter()
    try:
        with (
            (out / "grpo.jsonl").open("a", encoding="utf-8") as log,
            (out / "samples.jsonl").open("a", encoding="utf-8") as sample_log,
        ):
            while step < steps:
                chosen = [
                    pool[order[(step * prompts_per_step + j) % len(pool)]]
                    for j in range(prompts_per_step)
                ]
                began = time.perf_counter()
                samples = rollouts(
                    policy, tokenizer, chosen, group, max_new_tokens, f"step{step}", seed, max_batch
                )
                sampled = time.perf_counter()
                results = [score(s) for s in samples]
                advantages = group_advantages([r["reward"] for r in results], group)
                rate = lr * min(1.0, (step + 1) / warmup) if warmup else lr
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                terms = accumulate_gradients(
                    policy, reference, samples, advantages, kl, micro_batch, autocast
                )
                norm = torch.nn.utils.clip_grad_norm_(weights, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                step += 1
                row = {
                    "step": step,
                    **step_metrics(samples, results, advantages, group),
                    **terms,
                    "gradient_norm": float(norm),
                    "lr": rate,
                    "rollout_seconds": sampled - began,
                    "update_seconds": time.perf_counter() - sampled,
                }
                if step % save_every == 0 or step == steps:
                    save()
                    # One whole group, to read for reward hacking as training goes.
                    for s, r, a in list(zip(samples, results, advantages))[:group]:
                        sample_log.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "id": s["record"]["id"],
                                    "reward": r["reward"],
                                    "advantage": float(a),
                                    "finished": s["finished"],
                                    "story": s["story"],
                                }
                            )
                            + "\n"
                        )
                    sample_log.flush()
                    print(json.dumps(row), flush=True)
                if snapshot_every and step % snapshot_every == 0:
                    export(out / f"model_step{step:04d}.npz")
                if guard and guard.wait_if_hot():
                    row["thermal_paused_seconds_total"] = guard.paused_seconds
                log.write(json.dumps(row) + "\n")
                log.flush()
    except KeyboardInterrupt:
        save()
        raise
    export(out / "model.npz")
    summary = {
        "config": asdict(config),
        "run_config": run_config,
        "parameters": policy.parameter_count,
        "steps": step,
        "prompt_pool": len(pool),
        "prompt_pool_sha256": pool_ids,
        "seconds_this_session": time.perf_counter() - start,
        "thermal_pauses": guard.pauses if guard else 0,
        "thermal_paused_seconds": guard.paused_seconds if guard else 0.0,
        "torch": torch.__version__,
        "device": device,
    }
    if device == "cuda":
        summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
