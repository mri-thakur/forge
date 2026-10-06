from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.bpe import ENDOFTEXT, GPT4_PATTERN, Tokenizer, count_chunks, merge, train_merges

TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside.\n"
    'One day, Lily saw a big dog. "Hello, dog!" she said. The dog wagged its tail.\n'
    "They played all day, and Lily was very happy. 123 + 4567 = 4690. Don't stop!\n"
) * 20


def naive_merges(chunk_counts, num_merges):
    """Recount every pair after every merge, with the same tie-break."""
    words = [(list(chunk), count) for chunk, count in chunk_counts.items()]
    merges = []
    for new_id in range(256, 256 + num_merges):
        stats = Counter()
        for word, count in words:
            for pair in zip(word, word[1:]):
                stats[pair] += count
        if not stats:
            break
        best = min(stats, key=lambda pair: (-stats[pair], pair))
        merges.append(best)
        words = [(merge(word, best, new_id), count) for word, count in words]
    return merges


def test_lecture_example():
    # "Let's build the GPT Tokenizer": three merges turn this into [258, 100, 258, 97, 99].
    tokenizer = Tokenizer.train(count_chunks("aaabdaaabac"), 256 + 3, special_tokens=())
    assert tokenizer.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert tokenizer.decode([258, 100, 258, 97, 99]) == "aaabdaaabac"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_training_matches_naive_recount(seed):
    import random

    rng = random.Random(seed)
    words = ["".join(rng.choice("abcde ") for _ in range(rng.randint(1, 12))) for _ in range(300)]
    counts = count_chunks(" ".join(words) + " éé ✓✓")
    assert train_merges(counts, 60) == naive_merges(counts, 60)


def test_training_stops_when_nothing_is_left_to_merge():
    # The regex splits "ab ab" into "ab" and " ab": merge (a, b), then (" ", ab).
    assert train_merges(count_chunks("ab ab"), 50) == [(97, 98), (32, 256)]
    assert Tokenizer.train(count_chunks("ab ab"), 300).encode("ab ab") == [256, 257]


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_roundtrip_any_text(text):
    tokenizer = _trained()
    assert tokenizer.decode(tokenizer.encode_ordinary(text)) == text


def test_special_tokens_are_single_ids_only_when_requested():
    tokenizer = _trained()
    text = f"The end.{ENDOFTEXT}Once"
    ids = tokenizer.encode(text)
    assert ids.count(tokenizer.special[ENDOFTEXT]) == 1
    assert tokenizer.special[ENDOFTEXT] == tokenizer.vocab_size - 1
    assert tokenizer.special[ENDOFTEXT] not in tokenizer.encode_ordinary(text)
    assert tokenizer.decode(ids) == tokenizer.decode(tokenizer.encode_ordinary(text)) == text


def test_merges_compress_and_vocab_size_is_exact():
    tokenizer = _trained()
    assert tokenizer.vocab_size == 300
    assert len(tokenizer.encode_ordinary(TEXT)) < 0.75 * len(TEXT.encode())  # 43 merges


def test_save_load_and_truncation(tmp_path):
    tokenizer = _trained()
    tokenizer.save(tmp_path / "tok.json")
    loaded = Tokenizer.load(tmp_path / "tok.json")
    assert loaded.encode(TEXT) == tokenizer.encode(TEXT)
    smaller = tokenizer.truncated(280)
    assert smaller.vocab_size == 280
    assert smaller.encode(TEXT) == Tokenizer.train(count_chunks(TEXT), 280).encode(TEXT)


def test_matches_tiktoken_gpt4_token_for_token():
    tiktoken = pytest.importorskip("tiktoken")
    reference = tiktoken.get_encoding("cl100k_base")
    assert reference._pat_str == GPT4_PATTERN
    tokenizer = Tokenizer.from_tiktoken(reference)
    samples = [
        TEXT,
        "Hello world!!! How's it going? I'm fine, you'll see; they've gone.",
        "Numbers 1234567 and 3.14159, emoji 🙂👍🏽, accents naïve café, 漢字とかな, ﷽",
        "    indented\n\n\ttabs\r\nand trailing spaces   ",
        "def f(x):\n    return x**2  # comment\n",
    ]
    for sample in samples:
        assert tokenizer.encode_ordinary(sample) == reference.encode_ordinary(sample)


_cached = {}


def _trained():
    if "tok" not in _cached:
        _cached["tok"] = Tokenizer.train(count_chunks(TEXT), 300)
    return _cached["tok"]
