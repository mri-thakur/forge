import json
import math

import numpy as np
import pytest

from forge.evaluation import evaluate_run, heldout_loss, ngram_baselines, training_curve
from forge.model import NumpyModel
from forge.training import prepare_data, train


class FixedDistribution:
    """Ignores its input: every position predicts the same known distribution."""

    def __init__(self, probabilities, context):
        self.logits = np.log(probabilities)
        self.context = context

    def forward(self, tokens):
        assert tokens.shape[1] <= self.context
        return np.broadcast_to(self.logits, tokens.shape + (256,)).copy()


@pytest.mark.parametrize("length", [66, 100, 129])
def test_heldout_loss_scores_every_byte_once_within_context(length):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=length).astype(np.uint8)
    probabilities = rng.dirichlet(np.ones(256))
    expected = -np.log(probabilities[data[1:]]).mean()
    model = FixedDistribution(probabilities, context=16)
    assert heldout_loss(model, data, context=16, batch=3) == pytest.approx(expected, rel=1e-12)


def test_ngram_baselines_match_direct_counts_across_chunks():
    rng = np.random.default_rng(1)
    train_bytes = rng.integers(0, 4, size=50).astype(np.uint8)
    heldout = rng.integers(0, 4, size=20).astype(np.uint8)
    unigram = np.bincount(train_bytes, minlength=256) + 1.0
    unigram /= unigram.sum()
    bigram = np.ones((256, 256))
    for a, b in zip(train_bytes[:-1], train_bytes[1:]):
        bigram[a, b] += 1
    bigram /= bigram.sum(axis=1, keepdims=True)
    result = ngram_baselines(train_bytes, heldout, chunk=3)
    assert result["uniform"] == pytest.approx(math.log(256))
    assert result["unigram"] == pytest.approx(-np.log(unigram[heldout[1:]]).mean())
    assert result["bigram"] == pytest.approx(-np.log(bigram[heldout[:-1], heldout[1:]]).mean())
    assert ngram_baselines(train_bytes, heldout) == result


def test_training_curve_keeps_last_row_for_steps_replayed_after_resume(tmp_path):
    log = tmp_path / "training.jsonl"
    rows = [{"step": s, "loss": 9.0, "lr": 1, "tokens_per_second": 1} for s in (1, 2, 3, 4)]
    rows += [{"step": s, "loss": 1.0, "lr": 1, "tokens_per_second": 1} for s in (3, 4)]
    rows[-1]["val_loss"] = 0.5
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    curve = training_curve(log, bytes_per_update=10, every=2)
    assert [point["train_loss"] for point in curve] == [9.0, 1.0]
    assert curve[-1] == {
        "step": 4,
        "bytes": 40,
        "train_loss": 1.0,
        "lr": 1,
        "tokens_per_second": 1,
        "monitor_val_loss": 0.5,
    }


def test_evaluate_run_end_to_end_on_cpu(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    prepare_data(data, synthetic=True)
    train(data, run, steps=3, dim=16, layers=1, context=64, batch=2)
    result = evaluate_run(
        NumpyModel.load(run / "model.npz"), run, data, tmp_path / "out", tmp_path / "docs"
    )
    assert result["heldout"]["bytes_scored"] == result["dataset"]["val_tokens"] - 1
    assert math.isfinite(result["heldout"]["loss_nats_per_byte"])
    assert len(result["samples"]["samples"]) == 4
    assert (tmp_path / "out" / "evaluation.json").exists()
    assert "Held-out loss" in (tmp_path / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    # The weights must be the ones the run summary recorded.
    with pytest.raises(ValueError, match="weights differ"):
        evaluate_run(NumpyModel(), run, data, tmp_path / "other")


def test_evaluate_run_skips_samples_when_context_cannot_hold_prompts(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    prepare_data(data, synthetic=True)
    train(data, run, steps=2, dim=16, layers=1, context=32, batch=2)
    result = evaluate_run(
        NumpyModel.load(run / "model.npz"), run, data, tmp_path / "out", tmp_path / "docs"
    )
    assert "32-byte training context" in result["samples"]["skipped"]
    assert "Skipped:" in (tmp_path / "docs" / "RESULTS.md").read_text(encoding="utf-8")
