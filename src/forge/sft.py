"""Supervised fine-tuning data and generation for TinyStories-Instruct.

Each example is `prompt(record)` + story + <|endoftext|>. The loss mask is 0 on
prompt tokens and 1 on story and end-of-text tokens, so the model learns to write
the story an instruction asks for rather than to predict instructions. Examples
are packed into one token stream; the example start offsets are saved so that
training rows can begin at an example boundary and always see a whole prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from array import array
from pathlib import Path

import numpy as np

from forge.bpe import ENDOFTEXT, Tokenizer
from forge.instruct import load_jsonl, prompt
from forge.training import sha256


def encode_examples(records, tokenizer):
    tokens, mask, starts, story_bytes = array("H"), array("B"), array("q"), 0
    end = tokenizer.special[ENDOFTEXT]
    for record in records:
        starts.append(len(tokens))
        head = tokenizer.encode_ordinary(prompt(record))
        body = tokenizer.encode_ordinary(record["story"]) + [end]
        tokens.extend(head)
        mask.extend([0] * len(head))
        tokens.extend(body)
        mask.extend([1] * len(body))
        story_bytes += len(record["story"].encode("utf-8"))
    return tokens, mask, starts, story_bytes


def prepare_sft(
    instruct_dir, tokenizer_path, out, train_examples=300_000, val_examples=5_000, seed=17
):
    """Sample training and validation examples and write token, mask, and start files.
    The fixed evaluation prompts are excluded from both."""
    instruct_dir, out = Path(instruct_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    tokenizer = Tokenizer.load(out / "tokenizer.json")
    held_out = {r["id"] for r in load_jsonl(instruct_dir / "eval_prompts.jsonl")}
    rng = random.Random(seed)
    manifest = {
        "dataset": "roneneldan/TinyStoriesInstruct",
        "source_manifest_sha256": sha256(instruct_dir / "manifest.json"),
        "format": "prompt + story + <|endoftext|>; loss on story and end-of-text only",
        "tokenizer_file": "tokenizer.json",
        "tokenizer_sha256": sha256(out / "tokenizer.json"),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "loss_mask": True,
        "seed": seed,
        "excluded_eval_prompts": len(held_out),
    }
    for split, source, count in (
        ("train", "train.jsonl", train_examples),
        ("val", "valid.jsonl", val_examples),
    ):
        records = sample_jsonl(instruct_dir / source, count, rng, held_out)
        tokens, mask, starts, story_bytes = encode_examples(records, tokenizer)
        (out / f"{split}.bin").write_bytes(tokens.tobytes())
        (out / f"{split}_mask.bin").write_bytes(mask.tobytes())
        (out / f"{split}_starts.bin").write_bytes(starts.tobytes())
        manifest.update(
            {
                f"{split}_examples": len(records),
                f"{split}_tokens": len(tokens),
                f"{split}_loss_tokens": int(sum(mask)),
                f"{split}_text_bytes": story_bytes,
                f"{split}_sha256": sha256(out / f"{split}.bin"),
                f"{split}_mask_sha256": sha256(out / f"{split}_mask.bin"),
            }
        )
    canonical = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def sample_jsonl(path, count, rng, exclude=()):
    """`count` random records of a JSONL file, parsing only the chosen lines (the
    training file has 2.4M records). Records whose id is in `exclude` are skipped."""
    with Path(path).open("rb") as stream:
        total = sum(1 for _ in stream)
    wanted = set(rng.sample(range(total), min(total, count + len(exclude))))
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                record = json.loads(line)
                if record["id"] not in exclude:
                    records.append(record)
    rng.shuffle(records)
    return records[:count]


def offline_engine(model, max_batch, requests, seed=17):
    """An engine for batch generation: KV-cache room for `max_batch` full-context
    requests, and none of the serving limits. All requests are queued up front, so a
    serving queue bound (1,024) or queue timeout (60 s) would reject the later ones,
    which then come back empty."""
    from forge.engine import Engine

    return Engine(
        model,
        "continuous",
        max_batch=max_batch,
        blocks=max_batch * (model.config.context // 16 + 1),
        block_size=16,
        seed=seed,
        max_queue=max(1, requests),
        queue_timeout=math.inf,
    )


def run_to_completion(engine, requests):
    while engine.busy:
        engine.step()
    failed = [r.id for r in requests if r.status != "completed"]
    if failed:
        raise RuntimeError(f"{len(failed)} requests did not complete, e.g. {failed[0]}")


def generate(
    model, tokenizer, records, samples=1, temperature=0.8, top_p=0.95, seed=17, max_batch=32
):
    """Sample `samples` stories per record through the batched engine, stopping at
    end-of-text. Returns rows {"id", "sample", "story", "finished"}."""
    end = tokenizer.special[ENDOFTEXT]
    context = model.config.context
    engine = offline_engine(model, max_batch, len(records) * samples, seed)
    requests = []
    for record in records:
        ids = tokenizer.encode_ordinary(prompt(record))
        budget = context - len(ids)
        for k in range(samples):
            request = engine.submit(
                f"{record['id']}#{k}", ids, budget, temperature=temperature, top_p=top_p, stop=end
            )
            requests.append((record["id"], k, request))
    run_to_completion(engine, [request for _, _, request in requests])
    rows = []
    for record_id, k, request in requests:
        generated = request.generated
        finished = bool(generated) and generated[-1] == end
        story = tokenizer.decode(generated[:-1] if finished else generated)
        rows.append({"id": record_id, "sample": k, "story": story.strip(), "finished": finished})
    return rows


def story_loss(model, tokenizer, stories, batch=32):
    """Mean nats per token of `stories` under a torch `model`, each scored on its own
    after <|endoftext|>, the way pretraining saw stories. Under the pretrained model,
    which never saw an instruction, this measures how fluent the story text is. Story
    tokens are scored; the end-of-text token is not, since unfinished stories lack it.
    Stories longer than the context are truncated."""
    import torch

    end = tokenizer.special[ENDOFTEXT]
    context = model.config.context
    sequences = [[end] + tokenizer.encode_ordinary(story)[: context - 1] for story in stories]
    totals, counts = np.zeros(len(sequences)), np.zeros(len(sequences))
    order = sorted(
        (i for i, s in enumerate(sequences) if len(s) > 1), key=lambda i: len(sequences[i])
    )
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        width = max(len(sequences[i]) for i in chunk) - 1
        x = np.zeros((len(chunk), width), dtype=np.int64)
        y, mask = np.zeros_like(x), np.zeros(x.shape, dtype=np.float32)
        for row, i in enumerate(chunk):
            n = len(sequences[i]) - 1
            x[row, :n], y[row, :n], mask[row, :n] = sequences[i][:-1], sequences[i][1:], 1
        with torch.inference_mode():
            log_probs = model.forward_tensor(x).float().log_softmax(-1)
            targets = torch.as_tensor(y, device=log_probs.device)
            nll = -log_probs.gather(-1, targets[..., None]).squeeze(-1)
            nll = (nll * torch.as_tensor(mask, device=log_probs.device)).sum(-1)
        totals[chunk] = nll.cpu().numpy()
        counts[chunk] = mask.sum(-1)
    return {
        "nats_per_token": float(totals.sum() / counts.sum()) if counts.sum() else None,
        "tokens": int(counts.sum()),
        "stories": len(stories),
        "per_story": [float(t / c) if c else None for t, c in zip(totals, counts)],
    }


def load_sft_arrays(data_dir, split):
    data_dir = Path(data_dir)
    tokens = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    mask = np.memmap(data_dir / f"{split}_mask.bin", dtype=np.uint8, mode="r")
    starts = np.fromfile(data_dir / f"{split}_starts.bin", dtype=np.int64)
    return tokens, mask, starts
