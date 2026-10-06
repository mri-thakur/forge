import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.autograd import cross_entropy  # noqa: E402
from forge.gpu_training import train_torch  # noqa: E402
from forge.model import ModelConfig, NumpyModel  # noqa: E402
from forge.torch_backend import TorchModel  # noqa: E402
from forge.training import prepare_data  # noqa: E402


def test_torch_gradients_match_independent_reverse_mode_engine():
    reference = NumpyModel(ModelConfig(dim=8, heads=2, kv_heads=1, hidden=16, layers=1))
    model = TorchModel(reference)
    for weight in model.weights.values():
        weight.requires_grad_(True)
    x, y = np.array([[1, 2, 3]]), np.array([[2, 3, 4]])
    logits, parameters = reference.forward(x, grad=True)
    cross_entropy(logits, y).backward()
    torch.nn.functional.cross_entropy(
        model.forward_tensor(x).reshape(-1, 256), torch.tensor(y).reshape(-1)
    ).backward()
    for key in parameters:
        np.testing.assert_allclose(
            model.weights[key].grad.numpy(), parameters[key].grad, atol=2e-5, rtol=2e-3
        )


def test_torch_training_checkpoint_and_numpy_export(tmp_path):
    data = tmp_path / "data"
    prepare_data(data, synthetic=True)
    summary = train_torch(
        data, tmp_path / "run", steps=2, dim=8, layers=1, context=8, batch=2, device="cpu"
    )
    model = NumpyModel.load(tmp_path / "run" / "model.npz")
    assert summary["steps"] == 2
    assert np.isfinite(model.forward(np.array([[1, 2, 3]]))).all()
    resumed = train_torch(
        data,
        tmp_path / "run",
        steps=2,
        dim=8,
        layers=1,
        context=8,
        batch=2,
        device="cpu",
        resume=True,
    )
    assert resumed["steps"] == 2


def test_gradient_accumulation_matches_one_large_batch(tmp_path):
    data = tmp_path / "data"
    prepare_data(data, synthetic=True)
    common = dict(steps=1, dim=8, layers=1, context=8, device="cpu", warmup=0, min_lr_ratio=1.0)
    train_torch(data, tmp_path / "large", batch=4, accumulate=1, **common)
    train_torch(data, tmp_path / "split", batch=2, accumulate=2, **common)
    large = NumpyModel.load(tmp_path / "large" / "model.npz").weights
    split = NumpyModel.load(tmp_path / "split" / "model.npz").weights
    for key in large:
        np.testing.assert_allclose(split[key], large[key], rtol=1e-5, atol=1e-6)


def test_token_shards_train_with_their_vocabulary(tmp_path):
    from forge.tinystories import prepare_tinystories

    corpus = "".join(
        f"Tom saw a {w} dog. The dog was {w}.\n<|endoftext|>\n" for w in ["big", "red", "sad"] * 40
    )
    (tmp_path / "train.txt").write_text(corpus, "utf-8")
    (tmp_path / "valid.txt").write_text(corpus[:600], "utf-8")
    prepare_tinystories(
        tmp_path / "train.txt", tmp_path / "valid.txt", tmp_path / "data", 270, (270,), workers=1
    )
    summary = train_torch(
        tmp_path / "data",
        tmp_path / "run",
        steps=3,
        dim=8,
        layers=1,
        context=8,
        batch=2,
        device="cpu",
        heads=2,
        kv_heads=2,
        accumulate=2,
    )
    assert summary["config"]["vocab_size"] == 270
    assert summary["config"]["kv_heads"] == 2
    assert NumpyModel.load(tmp_path / "run" / "model.npz").weights["embed"].shape == (270, 8)
    assert np.isfinite(summary["final_val_loss"])
    assert summary["val_bits_per_byte"] < summary["val_bits_per_token"]


def test_bf16_is_rejected_without_cuda(tmp_path):
    data = tmp_path / "data"
    prepare_data(data, synthetic=True)
    with pytest.raises(ValueError, match="only on CUDA"):
        train_torch(
            data,
            tmp_path / "run",
            steps=1,
            dim=8,
            layers=1,
            context=8,
            batch=2,
            device="cpu",
            precision="bf16",
        )
