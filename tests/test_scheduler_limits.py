import numpy as np
import pytest

from forge.engine import Engine
from forge.model import ModelConfig, NumpyModel


def tiny_model():
    return NumpyModel(ModelConfig(dim=8, heads=2, kv_heads=1, hidden=16, layers=1, context=64))


def test_queue_timeout_is_visible_and_reclaims_no_owned_blocks():
    engine = Engine(tiny_model(), queue_timeout=1)
    request = engine.submit("r", [1, 2], 3, arrival=0)
    engine.step(now=2)
    assert request.status == "rejected"
    assert request.reason == "queue_timeout"
    assert engine.pool.used_blocks == 0


def test_queue_limit_and_context_validation():
    engine = Engine(tiny_model(), max_queue=1)
    engine.submit("r", [1], 2)
    assert engine.submit("s", [1], 2).status == "rejected"
    with pytest.raises(ValueError, match="context"):
        engine.submit("bad", [1] * 63, 2)


def test_prefix_lru_eviction_under_pressure_releases_every_reference():
    engine = Engine(tiny_model(), blocks=4, block_size=4, prefix_entries=1, max_batch=1)
    for i in range(5):
        engine.submit(str(i), [i + 1] * 9, 3)
        while engine.busy:
            engine.step()
            engine.pool.assert_integrity()
    assert len(engine.pool.prefixes) == 1
    engine.pool.clear_prefixes()
    engine.pool.assert_integrity()
    assert engine.pool.used_blocks == 0


def test_sampling_stream_is_independent_of_batch_arrivals():
    model = tiny_model()
    a, b = Engine(model, seed=17), Engine(model, seed=17)
    first = a.submit("same", [1, 2, 3], 8, temperature=0.7)
    second = b.submit("same", [1, 2, 3], 8, temperature=0.7)
    b.submit("other", [4] * 12, 13, temperature=1)
    while a.busy:
        a.step()
    while b.busy:
        b.step()
    np.testing.assert_array_equal(first.generated, second.generated)


def test_shared_prefix_with_different_suffixes_reuses_blocks():
    model = tiny_model()
    engine = Engine(model, block_size=4, prefix_entries=16)
    engine.submit("first", [1, 2, 3, 4] * 3 + [5, 6, 7, 8, 9], 2)
    while engine.busy:
        engine.step()
    request = engine.submit("second", [1, 2, 3, 4] * 3 + [9, 8, 7, 6, 5], 2)
    engine.step()
    assert engine.pool.prefix_hits == 1
    assert engine.pool.handles["second"].reused_tokens == 12
    while engine.busy:
        engine.step()
    from forge.engine import reference_generate

    assert request.generated == reference_generate(model, request.prompt, 2)
    engine.pool.clear_prefixes()
    engine.pool.assert_integrity()
