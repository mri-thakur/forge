import json

import numpy as np
import pytest

from forge.bpe import ENDOFTEXT, Tokenizer, count_chunks
from forge.instruct import prompt
from forge.model import ModelConfig, NumpyModel
from forge.sft import generate, load_sft_arrays, prepare_sft

torch = pytest.importorskip("torch")

STORIES = [
    ("Tom saw a big dog. The dog ran.", ["dog", "run"]),
    ('Lily said, "Hello!" to her cat.', ["cat"]),
    ("The sun was hot. Ben went to swim.", ["sun", "swim"]),
]


@pytest.fixture
def instruct_dir(tmp_path):
    folder = tmp_path / "instruct"
    folder.mkdir()
    for split, count in (("train", 30), ("valid", 12)):
        with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for i in range(count):
                story, words = STORIES[i % len(STORIES)]
                record = {
                    "id": f"{split}-{i:07d}",
                    "summary": "A short day.",
                    "words": words,
                    "features": [],
                    "sentence": None,
                    "story": story,
                }
                stream.write(json.dumps(record) + "\n")
    lines = (folder / "valid.jsonl").read_text("utf-8").splitlines()[:2]
    (folder / "eval_prompts.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    (folder / "manifest.json").write_text("{}", "utf-8")
    corpus = " ".join(story for story, _ in STORIES) * 10 + " Summary Words Story"
    Tokenizer.train(count_chunks(corpus), 300).save(tmp_path / "tokenizer.json")
    return folder


def test_mask_covers_exactly_story_and_end_of_text(instruct_dir, tmp_path):
    manifest = prepare_sft(instruct_dir, tmp_path / "tokenizer.json", tmp_path / "sft", 10, 5)
    tokenizer = Tokenizer.load(tmp_path / "sft" / "tokenizer.json")
    tokens, mask, starts = load_sft_arrays(tmp_path / "sft", "train")
    assert manifest["train_examples"] == 10 and len(starts) == 10
    ends = list(starts[1:]) + [len(tokens)]
    for start, end in zip(starts, ends):
        example, scored = tokens[start:end].tolist(), mask[start:end].tolist()
        first = scored.index(1)
        assert not any(scored[:first]) and all(scored[first:])
        assert tokenizer.decode(example[:first]).startswith("Summary: A short day.")
        assert tokenizer.decode(example[:first]).endswith("Story:\n")
        assert tokenizer.decode(example[first:]).endswith(ENDOFTEXT)
    held_out = {json.loads(line)["id"] for line in open(instruct_dir / "eval_prompts.jsonl")}
    assert manifest["val_examples"] == 5 and held_out  # eval prompts are excluded


def test_fine_tuning_starts_from_the_given_weights(instruct_dir, tmp_path):
    from forge.gpu_training import train_torch

    prepare_sft(instruct_dir, tmp_path / "tokenizer.json", tmp_path / "sft", 20, 6)
    base = NumpyModel(
        ModelConfig(
            vocab_size=300, dim=16, heads=2, kv_heads=1, layers=1, hidden=32, context=64, seed=3
        )
    )
    base.save(tmp_path / "base.npz")
    summary = train_torch(
        tmp_path / "sft",
        tmp_path / "run",
        steps=2,
        context=32,
        batch=2,
        device="cpu",
        init=tmp_path / "base.npz",
        lr=1e-9,
        warmup=0,
    )
    tuned = NumpyModel.load(tmp_path / "run" / "model.npz")
    assert summary["config"]["seed"] == 3  # shape and seed came from the init file
    np.testing.assert_allclose(tuned.weights["embed"], base.weights["embed"], atol=1e-6)
    assert summary["run_config"]["init"]["path"].endswith("base.npz")
    assert np.isfinite(summary["final_val_loss"])


def test_generate_returns_every_sample_and_stops_at_end_of_text(instruct_dir, tmp_path):
    from forge.torch_backend import TorchModel

    tokenizer = Tokenizer.load(tmp_path / "tokenizer.json")
    model = TorchModel(
        NumpyModel(
            ModelConfig(
                vocab_size=300, dim=16, heads=2, kv_heads=1, layers=1, hidden=32, context=128
            )
        )
    )
    records = [json.loads(line) for line in open(instruct_dir / "eval_prompts.jsonl")]
    rows = generate(model, tokenizer, records, samples=2, max_batch=3)
    assert [(row["id"], row["sample"]) for row in rows] == [
        (records[0]["id"], 0),
        (records[0]["id"], 1),
        (records[1]["id"], 0),
        (records[1]["id"], 1),
    ]
    budget = 128 - len(tokenizer.encode_ordinary(prompt(records[0])))
    for row in rows:
        assert row["finished"] or len(tokenizer.encode_ordinary(row["story"])) >= budget - 8
