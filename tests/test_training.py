import json

import numpy as np
import pytest

from forge.model import ModelConfig, NumpyModel
from forge.training import Trainer, prepare_data, sample_batch


def test_checkpoint_resume_reproduces_uninterrupted_training(tmp_path):
    config = ModelConfig(dim=8, heads=2, kv_heads=1, layers=1, hidden=16, vocab_size=8)
    data = np.random.default_rng(4).integers(0, 8, size=200)
    uninterrupted = Trainer(NumpyModel(config), total_steps=6, warmup=1)
    interrupted = Trainer(NumpyModel(config), total_steps=6, warmup=1)
    for _ in range(3):
        uninterrupted.update(*sample_batch(data, uninterrupted.rng, 2, 8))
        interrupted.update(*sample_batch(data, interrupted.rng, 2, 8))
    interrupted.save(tmp_path / "resume.npz", "dataset")
    resumed = Trainer.load(tmp_path / "resume.npz", "dataset")
    for _ in range(3):
        uninterrupted.update(*sample_batch(data, uninterrupted.rng, 2, 8))
        resumed.update(*sample_batch(data, resumed.rng, 2, 8))
    for key in uninterrupted.model.weights:
        np.testing.assert_array_equal(uninterrupted.model.weights[key], resumed.model.weights[key])
        np.testing.assert_array_equal(uninterrupted.m[key], resumed.m[key])
    with pytest.raises(ValueError, match="hash"):
        Trainer.load(tmp_path / "resume.npz", "different dataset")


def test_training_reduces_loss_on_controlled_pattern():
    model = NumpyModel(ModelConfig(dim=8, heads=2, kv_heads=1, layers=1, hidden=16, vocab_size=8))
    trainer = Trainer(model, learning_rate=0.02, total_steps=60, warmup=0)
    x, y = np.tile([1, 2, 3, 4], (2, 1)), np.tile([2, 3, 4, 1], (2, 1))
    losses = [trainer.update(x, y)["loss"] for _ in range(60)]
    assert losses[-1] < 0.2
    np.testing.assert_array_equal(model.forward(x).argmax(-1), y)


def test_model_checkpoint_roundtrip(tmp_path):
    model = NumpyModel(ModelConfig(dim=8, heads=2, kv_heads=1, layers=1, hidden=16))
    model.save(tmp_path / "model.npz")
    loaded = NumpyModel.load(tmp_path / "model.npz")
    tokens = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(model.forward(tokens), loaded.forward(tokens))


def test_prepare_records_hashes_and_labels_synthetic_data(tmp_path):
    manifest = prepare_data(tmp_path, synthetic=True)
    assert manifest["synthetic"]
    assert manifest["train_sha256"] != manifest["val_sha256"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
