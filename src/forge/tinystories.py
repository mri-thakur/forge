"""TinyStories: train the BPE tokenizer on the corpus and encode it to token files.

Source files separate stories with <|endoftext|> lines. Each story is stripped of
surrounding whitespace and encoded followed by the <|endoftext|> token, so the model
learns where a story ends. Work is split into byte ranges cut at story boundaries
and spread across worker processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from array import array
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from forge.bpe import ENDOFTEXT, Tokenizer, count_chunks
from forge.training import sha256

REPOSITORY = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SEPARATOR = ENDOFTEXT.encode()


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


def story_ranges(path, parts):
    """Byte ranges covering the file, each ending just after a story separator."""
    size = Path(path).stat().st_size
    cuts = {0, size}
    with Path(path).open("rb") as stream:
        for i in range(1, parts):
            cuts.add(_next_boundary(stream, size * i // parts, size))
    cuts = sorted(cuts)
    return list(zip(cuts, cuts[1:]))


def _next_boundary(stream, start, size):
    stream.seek(start)
    carry, position = b"", start
    while position < size:
        block = stream.read(1 << 20)
        if not block:
            break
        data = carry + block
        found = data.find(SEPARATOR)
        if found >= 0:
            return position - len(carry) + found + len(SEPARATOR)
        carry = data[-(len(SEPARATOR) - 1) :]
        position += len(block)
    return size


def read_stories(path, start, end):
    with Path(path).open("rb") as stream:
        stream.seek(start)
        text = stream.read(end - start).decode("utf-8")
    return [story.strip() for story in text.split(ENDOFTEXT) if story.strip()]


def _count_part(task):
    path, start, end = task
    counts = Counter()
    stories = read_stories(path, start, end)
    for story in stories:
        count_chunks(story, counts=counts)
    return counts, len(stories)


_worker_tokenizers = {}


def _encode_part(task):
    tokenizer_path, path, start, end = task
    if tokenizer_path not in _worker_tokenizers:
        _worker_tokenizers[tokenizer_path] = Tokenizer.load(tokenizer_path)
    tokenizer = _worker_tokenizers[tokenizer_path]
    end_of_text = tokenizer.special[ENDOFTEXT]
    ids, text_bytes = array("H"), 0
    stories = read_stories(path, start, end)
    for story in stories:
        ids.extend(tokenizer.encode_ordinary(story))
        ids.append(end_of_text)
        text_bytes += len(story.encode("utf-8"))
    return ids.tobytes(), len(stories), text_bytes


def _map(function, tasks, workers):
    if workers == 1:
        yield from map(function, tasks)
        return
    with Pool(workers) as pool:
        yield from pool.imap(function, tasks)


def count_corpus(path, workers=None, parts=None):
    workers = workers or default_workers()
    tasks = [(str(path), a, b) for a, b in story_ranges(path, parts or 8 * workers)]
    counts, stories = Counter(), 0
    for part_counts, part_stories in _map(_count_part, tasks, workers):
        counts.update(part_counts)
        stories += part_stories
    return counts, stories


def encode_corpus(tokenizer_path, path, out, workers=None, parts=None):
    """Write `path` as little-endian uint16 tokens to `out`; return counts."""
    workers = workers or default_workers()
    parts = parts or max(8 * workers, math.ceil(Path(path).stat().st_size / (16 << 20)))
    tasks = [(str(tokenizer_path), str(path), a, b) for a, b in story_ranges(path, parts)]
    tokens = stories = text_bytes = 0
    with Path(out).open("wb") as stream:
        for data, part_stories, part_bytes in _map(_encode_part, tasks, workers):
            stream.write(data)
            tokens += len(data) // 2
            stories += part_stories
            text_bytes += part_bytes
    return {"tokens": tokens, "stories": stories, "text_bytes": text_bytes}


def prepare_tinystories(
    train, valid, out, vocab_size=4096, compare_sizes=(2048, 4096, 8192), workers=None
):
    """Train the tokenizer on the training split, compare vocabulary sizes on the
    validation split, and encode both splits with the chosen vocabulary."""
    train, valid, out = Path(train), Path(valid), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or default_workers()
    largest = max(*compare_sizes, vocab_size)

    start = time.perf_counter()
    counts, train_stories = count_corpus(train, workers)
    counted = time.perf_counter()
    full = Tokenizer.train(counts, largest)
    trained = time.perf_counter()

    paths = {}
    for size in sorted({*compare_sizes, vocab_size}):
        paths[size] = out / f"tokenizer_{size}.json"
        full.truncated(size).save(paths[size])

    comparison = compare_tokenizers(valid, {size: Tokenizer.load(p) for size, p in paths.items()})
    train_stats = encode_corpus(paths[vocab_size], train, out / "train.bin", workers)
    val_stats = encode_corpus(paths[vocab_size], valid, out / "val.bin", workers)
    encoded = time.perf_counter()

    manifest = {
        "dataset": REPOSITORY,
        "revision": REVISION,
        "homepage": f"https://huggingface.co/datasets/{REPOSITORY}",
        "licenses": ["CDLA-Sharing-1.0"],
        "synthetic": False,
        "split": "official TinyStoriesV2-GPT4 train/valid files",
        "source_files": [
            {"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in (train, valid)
        ],
        "tokenizer": f"forge-bpe-v1, vocab {vocab_size}, GPT-4 split pattern",
        "tokenizer_file": paths[vocab_size].name,
        "tokenizer_sha256": sha256(paths[vocab_size]),
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "end_of_text_id": vocab_size - 1,
        "distinct_chunks": len(counts),
        "train_stories": train_stats["stories"],
        "val_stories": val_stats["stories"],
        "train_tokens": train_stats["tokens"],
        "val_tokens": val_stats["tokens"],
        "train_text_bytes": train_stats["text_bytes"],
        "val_text_bytes": val_stats["text_bytes"],
        "train_sha256": sha256(out / "train.bin"),
        "val_sha256": sha256(out / "val.bin"),
        "seconds": {
            "count_chunks": counted - start,
            "train_tokenizer": trained - counted,
            "encode": encoded - trained,
        },
        "workers": workers,
    }
    if train_stories != train_stats["stories"]:
        raise RuntimeError("story counts differ between counting and encoding")
    canonical = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, comparison


def compare_tokenizers(valid, tokenizers):
    """Bytes per token on the validation stories (end-of-text tokens excluded), for
    our vocabularies and, when tiktoken is installed, GPT-2's and GPT-4's."""
    stories = read_stories(valid, 0, Path(valid).stat().st_size)
    text_bytes = sum(len(story.encode("utf-8")) for story in stories)
    rows = []
    for size, tokenizer in sorted(tokenizers.items()):
        tokens = sum(len(tokenizer.encode_ordinary(story)) for story in stories)
        rows.append({"tokenizer": f"forge BPE {size}", "vocab_size": size, "tokens": tokens})
    try:
        import tiktoken
    except ImportError:
        tiktoken = None
    if tiktoken:
        for name, label in (("gpt2", "GPT-2"), ("cl100k_base", "GPT-4 (cl100k)")):
            encoding = tiktoken.get_encoding(name)
            tokens = sum(len(encoding.encode_ordinary(story)) for story in stories)
            rows.append({"tokenizer": label, "vocab_size": encoding.n_vocab, "tokens": tokens})
    for row in rows:
        row["bytes_per_token"] = text_bytes / row["tokens"]
    return {"split": "valid", "stories": len(stories), "text_bytes": text_bytes, "rows": rows}
