import pytest

from forge.bpe import ENDOFTEXT, Tokenizer, count_chunks
from forge.demo import build_record, stream_story, verdict
from forge.model import ModelConfig, NumpyModel

torch = pytest.importorskip("torch")

from forge.torch_backend import TorchModel  # noqa: E402


def test_build_record_parses_the_form():
    record = build_record(
        summary="  A day out. ", words="dog, , run ", features=["Twist", "Dialogue"], sentence=" "
    )
    assert record["summary"] == "A day out."
    assert record["words"] == ["dog", "run"]
    assert record["features"] == ["Dialogue", "Twist"]  # canonical order
    assert record["sentence"] is None


def test_verdict_marks_each_constraint():
    record = build_record(words="dog, cat", features=["Dialogue"], sentence="It was fun.")
    line = verdict(record, 'The dog barked. "Hi!" said Tom. It was fun.')
    assert "✓ dog" in line and "✗ cat" in line
    assert "✓ included" in line and "✓ present" in line and "0.83" in line
    assert "did not finish" in verdict(record, "The dog.", finished=False)
    assert "No checkable instruction" in verdict(build_record(summary="A day."), "Hi.")


def test_stream_story_grows_and_reports_the_ending():
    corpus = "Tom saw a big dog. The dog ran. Lily said hello to her cat." * 5
    tokenizer = Tokenizer.train(count_chunks(corpus + " Summary Words Story"), 300)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=16,
        heads=2,
        kv_heads=1,
        layers=1,
        hidden=32,
        context=64,
    )
    model = TorchModel(NumpyModel(config))
    record = build_record(words="dog")
    steps = list(stream_story(model, tokenizer, record, max_tokens=12))
    assert 1 <= len(steps) <= 12
    assert all(not finished for _, finished in steps[:-1])
    story, finished = steps[-1]
    assert finished or len(steps) == 12
    assert ENDOFTEXT not in story
    again = list(stream_story(model, tokenizer, record, max_tokens=12))
    assert again[-1] == steps[-1]  # the same seed gives the same story
