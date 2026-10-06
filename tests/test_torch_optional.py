import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.cache import BlockPool  # noqa: E402
from forge.model import ModelConfig, NumpyModel  # noqa: E402
from forge.torch_backend import TorchModel  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("attention", ["manual", "sdpa"])
def test_torch_backend_against_independent_numpy_oracle(device, attention):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference = NumpyModel(ModelConfig(dim=16, heads=2, kv_heads=1, layers=2, hidden=32))
    model = TorchModel(reference, device, attention)
    tokens = np.array([[1, 2, 3, 4, 5]])
    np.testing.assert_allclose(
        model.forward(tokens), reference.forward(tokens), atol=2e-5, rtol=2e-4
    )
    pool = BlockPool(model.config, blocks=4, backend="torch", device=device)
    pool.create("a", 16)
    for position in range(5):
        pool.prepare_write("a", [position])
        cached = model.forward(
            tokens[:, position : position + 1], np.array([[position]]), pool, ["a"]
        )
        expected = reference.forward(tokens[:, : position + 1])[:, -1:]
        np.testing.assert_allclose(cached, expected, atol=2e-5, rtol=2e-4)
