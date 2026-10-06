import numpy as np
import pytest

from forge.cache import ContiguousKV
from forge.engine import contiguous_generate, reference_generate
from forge.model import ModelConfig, NumpyModel


@pytest.mark.parametrize("length", [1, 7, 17])
def test_contiguous_cache_matches_full_sequence_logits_and_greedy_outputs(length):
    model = NumpyModel(ModelConfig(dim=16, heads=2, kv_heads=1, hidden=32, layers=2))
    tokens = np.random.default_rng(17).integers(0, 256, size=length).tolist()
    cache = ContiguousKV(model.config, length + 4)
    for position in range(length):
        cache.prepare_write([position])
        cached = model.forward(np.array([[tokens[position]]]), np.array([[position]]), cache, ["a"])
        expected = model.forward(np.array([tokens[: position + 1]]))[:, -1:]
        np.testing.assert_allclose(cached, expected, atol=1e-6, rtol=2e-5)
    assert contiguous_generate(model, tokens, 4) == reference_generate(model, tokens, 4)
