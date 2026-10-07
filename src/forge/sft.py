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


def generate(
    model, tokenizer, records, samples=1, temperature=0.8, top_p=0.95, seed=17, max_batch=32
):
    """Sample `samples` stories per record through the batched engine, stopping at
    end-of-text. Returns rows {"id", "sample", "story", "finished"}."""
    from forge.engine import Engine

    end = tokenizer.special[ENDOFTEXT]
    context = model.config.context
    blocks = max_batch * (context // 16 + 1)
    engine = Engine(
        model, "continuous", max_batch=max_batch, blocks=blocks, block_size=16, seed=seed
    )
    requests = []
    for record in records:
        ids = tokenizer.encode_ordinary(prompt(record))
        budget = context - len(ids)
        for k in range(samples):
            request = engine.submit(
                f"{record['id']}#{k}", ids, budget, temperature=temperature, top_p=top_p, stop=end
            )
            requests.append((record["id"], k, request))
    while engine.busy:
        engine.step()
    rows = []
    for record_id, k, request in requests:
        generated = request.generated
        finished = bool(generated) and generated[-1] == end
        story = tokenizer.decode(generated[:-1] if finished else generated)
        rows.append({"id": record_id, "sample": k, "story": story.strip(), "finished": finished})
    return rows


def load_sft_arrays(data_dir, split):
    data_dir = Path(data_dir)
    tokens = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    mask = np.memmap(data_dir / f"{split}_mask.bin", dtype=np.uint8, mode="r")
    starts = np.fromfile(data_dir / f"{split}_starts.bin", dtype=np.int64)
    return tokens, mask, starts
