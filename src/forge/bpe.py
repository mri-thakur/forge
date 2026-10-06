"""Byte-level BPE tokenizer, built from scratch: training, encoding, decoding.

Follows the GPT-2/GPT-4 design from Karpathy's "Let's build the GPT Tokenizer":
text is first split with the GPT-4 regex, so merges never cross the boundaries
between words, numbers, punctuation, and whitespace; each piece's UTF-8 bytes are
then merged greedily, lowest-priority pair first. Special tokens such as
<|endoftext|> are matched before splitting and are never produced by merges.

The same encoder runs OpenAI's published merge tables (`from_tiktoken`), which is
how the tests check it token for token against tiktoken.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import regex

# tiktoken's cl100k_base (GPT-4) split pattern, verbatim.
GPT4_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+"""
    r"""|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
ENDOFTEXT = "<|endoftext|>"


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair`, left to right."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def count_chunks(text, pattern=GPT4_PATTERN, counts=None):
    """Counts of each distinct pre-split chunk, as UTF-8 bytes."""
    counts = Counter() if counts is None else counts
    counts.update(chunk.encode("utf-8") for chunk in regex.findall(pattern, text))
    return counts


def train_merges(chunk_counts, num_merges):
    """Learn up to `num_merges` merges from {chunk bytes: occurrences}.

    The most frequent adjacent pair is merged first; ties go to the smallest pair of
    ids, so training is deterministic. Only words containing the merged pair are
    updated and a lazy max-heap finds the next pair, so the cost grows with the
    number of distinct chunks rather than with the corpus size.
    """
    words = [list(chunk) for chunk in chunk_counts]
    freqs = list(chunk_counts.values())
    pair_counts = defaultdict(int)
    where = defaultdict(set)
    for index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freqs[index]
            where[pair].add(index)
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    for new_id in range(256, 256 + num_merges):
        while heap:
            negative, pair = heapq.heappop(heap)
            if pair_counts.get(pair) == -negative:  # skip stale heap entries
                break
        else:
            break  # every chunk is a single token
        merges.append(pair)
        changed = set()
        for index in where.pop(pair):
            word = words[index]
            merged = merge(word, pair, new_id)
            if len(merged) == len(word):
                continue  # stale: an earlier merge already consumed this pair here
            freq = freqs[index]
            for old in zip(word, word[1:]):
                pair_counts[old] -= freq
                changed.add(old)
            for new in zip(merged, merged[1:]):
                pair_counts[new] += freq
                changed.add(new)
                where[new].add(index)
            words[index] = merged
        for changed_pair in changed:
            count = pair_counts[changed_pair]
            if count > 0:
                heapq.heappush(heap, (-count, changed_pair))
            else:
                del pair_counts[changed_pair]
    return merges


class Tokenizer:
    def __init__(self, merges, special_tokens=(ENDOFTEXT,), pattern=GPT4_PATTERN, byte_ids=None):
        """`merges` maps (left id, right id) to the merged id, which is also its
        priority: lower ids merge first. `byte_ids[b]` is the id of raw byte b."""
        self.pattern = pattern
        self.compiled = regex.compile(pattern)
        self.merges = dict(merges)
        self.byte_ids = list(range(256)) if byte_ids is None else list(byte_ids)
        self.vocab = {token: bytes([b]) for b, token in enumerate(self.byte_ids)}
        for (left, right), token in sorted(self.merges.items(), key=lambda item: item[1]):
            self.vocab[token] = self.vocab[left] + self.vocab[right]
        first_special = max(self.vocab) + 1
        self.special = {name: first_special + i for i, name in enumerate(special_tokens)}
        self.special_pattern = (
            regex.compile("(" + "|".join(regex.escape(s) for s in self.special) + ")")
            if self.special
            else None
        )
        self._cache = {}

    @classmethod
    def train(cls, chunk_counts, vocab_size, special_tokens=(ENDOFTEXT,)):
        merges = train_merges(chunk_counts, vocab_size - 256 - len(special_tokens))
        return cls({pair: 256 + i for i, pair in enumerate(merges)}, special_tokens)

    @classmethod
    def from_tiktoken(cls, encoding):
        """Rebuild an OpenAI encoding's merges from its token ranks."""
        ranks = encoding._mergeable_ranks
        merges = {}
        for token, rank in ranks.items():
            if len(token) > 1:
                left, right = _split_by_rank(ranks, token, rank)
                merges[(ranks[left], ranks[right])] = rank
        byte_ids = [ranks[bytes([b])] for b in range(256)]
        return cls(merges, (), encoding._pat_str, byte_ids)

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special)

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = [self.byte_ids[b] for b in chunk]
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        if len(self._cache) < 1_000_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        """Encode treating special-token strings as ordinary text."""
        ids = []
        for chunk in self.compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text):
        """Encode, mapping each special-token string to its single id."""
        if not self.special_pattern:
            return self.encode_ordinary(text)
        ids = []
        for part in self.special_pattern.split(text):
            if part in self.special:
                ids.append(self.special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        names = {token: name.encode("utf-8") for name, token in self.special.items()}
        data = b"".join(self.vocab[i] if i in self.vocab else names[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        if self.byte_ids != list(range(256)):
            raise ValueError("only tokenizers trained here can be saved")
        merges = sorted(self.merges.items(), key=lambda item: item[1])
        if [token for _, token in merges] != list(range(256, 256 + len(merges))):
            raise ValueError("merge ids must be contiguous from 256")
        payload = {
            "format": "forge-bpe-v1",
            "pattern": self.pattern,
            "merges": [list(pair) for pair, _ in merges],
            "special_tokens": list(self.special),
        }
        Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "forge-bpe-v1":
            raise ValueError("not a forge BPE tokenizer file")
        merges = {tuple(pair): 256 + i for i, pair in enumerate(payload["merges"])}
        return cls(merges, payload["special_tokens"], payload["pattern"])

    def truncated(self, vocab_size):
        """The tokenizer this one was after its first merges; BPE learns merges in
        order, so smaller vocabularies are prefixes of a larger training run."""
        keep = vocab_size - 256 - len(self.special)
        if not 0 <= keep <= len(self.merges):
            raise ValueError("vocab_size outside this tokenizer's range")
        merges = {pair: token for pair, token in self.merges.items() if token < 256 + keep}
        return Tokenizer(merges, tuple(self.special), self.pattern)


def _split_by_rank(ranks, token, max_rank):
    """Replay BPE on `token` using only merges ranked below `max_rank`; for a valid
    merge table this ends in exactly the two parts that formed the token."""
    parts = [bytes([b]) for b in token]
    while True:
        best = None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < max_rank and (best is None or rank < best[0]):
                best = (rank, i)
        if best is None:
            break
        i = best[1]
        parts[i : i + 2] = [parts[i] + parts[i + 1]]
    if len(parts) != 2:
        raise ValueError(f"cannot recover the merge that produced {token!r}")
    return parts
