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
