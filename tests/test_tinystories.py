import random

import numpy as np
import pytest

from forge.bpe import ENDOFTEXT, Tokenizer
from forge.tinystories import prepare_tinystories, read_stories, story_ranges

WORDS = "the a little big dog cat girl boy ran jumped happy sad tree ball park home".split()


def write_corpus(path, count, seed):
    rng = random.Random(seed)
    stories = []
    for _ in range(count):
        sentences = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 9))).capitalize() + "."
            for _ in range(rng.randint(1, 4))
        ]
        if rng.random() < 0.3:
            sentences.append('"Hello, friend!" said Ana. Café ✓.')
        stories.append(" ".join(sentences))
    # Same layout as the source files: separators on their own lines, a leading
    # fragment, and stray blank lines.
    path.write_text("\n" + f"\n{ENDOFTEXT}\n\n".join(stories) + f"\n{ENDOFTEXT}\n", "utf-8")
    return stories


@pytest.mark.parametrize("parts", [1, 2, 3, 7, 50])
def test_ranges_cover_every_story_exactly_once(tmp_path, parts):
    stories = write_corpus(tmp_path / "train.txt", 120, 0)
    ranges = story_ranges(tmp_path / "train.txt", parts)
    assert ranges[0][0] == 0 and ranges[-1][1] == (tmp_path / "train.txt").stat().st_size
    assert all(a < b for a, b in ranges)
    found = [s for a, b in ranges for s in read_stories(tmp_path / "train.txt", a, b)]
    assert found == stories


@pytest.mark.parametrize("workers", [1, 2])
def test_prepare_encodes_every_story_and_decodes_exactly(tmp_path, workers):
    stories = write_corpus(tmp_path / "train.txt", 300, 1)
    valid = write_corpus(tmp_path / "valid.txt", 40, 2)
    manifest, comparison = prepare_tinystories(
        tmp_path / "train.txt",
        tmp_path / "valid.txt",
        tmp_path / "out",
        vocab_size=320,
        compare_sizes=(300, 320),
        workers=workers,
    )
    tokenizer = Tokenizer.load(tmp_path / "out" / manifest["tokenizer_file"])
    assert tokenizer.vocab_size == manifest["vocab_size"] == 320
    for split, expected in (("train", stories), ("val", valid)):
        tokens = np.fromfile(tmp_path / "out" / f"{split}.bin", dtype=np.uint16)
        assert len(tokens) == manifest[f"{split}_tokens"]
        assert tokens[-1] == manifest["end_of_text_id"]
        assert tokenizer.decode(tokens.tolist()) == "".join(s + ENDOFTEXT for s in expected)
    assert manifest["train_stories"] == len(stories)
    sizes = [
        row["vocab_size"] for row in comparison["rows"] if row["tokenizer"].startswith("forge")
    ]
    assert sizes == [300, 320]
    per_token = [row["bytes_per_token"] for row in comparison["rows"][:2]]
    assert per_token[1] > per_token[0] > 1  # more merges, more bytes per token
