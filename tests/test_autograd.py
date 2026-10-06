import numpy as np
import pytest

from forge.autograd import Tensor, cross_entropy, softmax
from forge.model import ModelConfig, NumpyModel


@pytest.mark.parametrize("operation", ["matmul", "rms", "softmax", "embedding", "swiglu"])
def test_vector_jacobian_products_against_finite_difference(operation):
    rng = np.random.default_rng(2)
    values = rng.normal(size=(2, 3))
    constant = Tensor(rng.normal(size=(3, 4)))

    def compute(x):
        if operation == "matmul":
            return ((x @ constant).power(2)).mean()
        if operation == "rms":
            return (x * (x.power(2).mean(axis=-1, keepdims=True) + 1e-6).power(-0.5)).sum()
        if operation == "softmax":
            return (softmax(x, np.array([[True, False, True], [True, True, True]])) * x).sum()
        if operation == "embedding":
            return x[np.array([1, 1, 0])].power(2).sum()
        return (x * x.sigmoid()).mean()

    x = Tensor(values.copy(), True)
    compute(x).backward()
    numerical = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        positive, negative = values.copy(), values.copy()
        positive[index] += 1e-5
        negative[index] -= 1e-5
        numerical[index] = (compute(Tensor(positive)).data - compute(Tensor(negative)).data) / 2e-5
    np.testing.assert_allclose(x.grad, numerical, rtol=1e-5, atol=1e-6)


def test_entire_transformer_gradient_includes_rope_gqa_and_tied_embedding():
    config = ModelConfig(dim=8, heads=2, kv_heads=1, layers=1, hidden=12, vocab_size=8)
    model = NumpyModel(config)
    model.weights = {k: v.astype(np.float64) for k, v in model.weights.items()}
    x, y = np.array([[1, 2, 3]]), np.array([[2, 3, 4]])
    logits, parameters = model.forward(x, grad=True)
    cross_entropy(logits, y).backward()
    for key, index in [
        ("embed", (2, 3)),
        ("layers.0.k", (2, 1)),
        ("layers.0.gate", (1, 3)),
        ("layers.0.q", (0, 2)),
        ("layers.0.attn_norm", (2,)),
    ]:
        weight = model.weights[key]
        original = weight[index]
        weight[index] = original + 1e-5
        positive = cross_entropy(Tensor(model.forward(x)), y).data
        weight[index] = original - 1e-5
        negative = cross_entropy(Tensor(model.forward(x)), y).data
        weight[index] = original
        np.testing.assert_allclose(
            parameters[key].grad[index], (positive - negative) / 2e-5, rtol=2e-3, atol=2e-6
        )
