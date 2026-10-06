import numpy as np
import pytest

from forge.cache import BlockPool, CacheFull
from forge.engine import Engine, reference_generate, sample
from forge.model import ByteTokenizer, ModelConfig, NumpyModel


@pytest.fixture
def model():
    return NumpyModel(
        ModelConfig(dim=16, heads=2, kv_heads=1, layers=2, hidden=32, context=128, vocab_size=32)
    )


@pytest.mark.parametrize("chunk", [1, 3, 8, 16])
def test_logits_match_at_every_cached_step(model, chunk):
    tokens = np.random.default_rng(9).integers(0, 32, size=27)
    pool = BlockPool(model.config, blocks=16, block_size=4)
    pool.create("r", len(tokens))
    for start in range(0, len(tokens), chunk):
        stop = min(len(tokens), start + chunk)
        positions = np.arange(start, stop)[None, :]
        pool.prepare_write("r", range(start, stop))
        cached = model.forward(tokens[None, start:stop], positions, pool, ["r"])
        expected = model.forward(tokens[None, :stop])[:, start:stop]
        np.testing.assert_allclose(cached, expected, rtol=2e-5, atol=1e-6)
    pool.release("r")
    pool.assert_integrity()
    assert pool.used_blocks == 0


@pytest.mark.parametrize("policy", ["static", "continuous", "chunked"])
@pytest.mark.parametrize("budget", [1, 4, 16])
def test_heterogeneous_batches_equal_independent_reference(model, policy, budget):
    engine = Engine(model, policy, max_batch=3, token_budget=budget, blocks=48, block_size=4)
    prompts = [[1, 2, 3], [4] * 11, [3, 5], [7] * 17]
    requests = [engine.submit(str(i), prompt, 5) for i, prompt in enumerate(prompts)]
    for _ in range(200):
        if not engine.busy:
            break
        engine.step()
        engine.pool.assert_integrity()
    assert not engine.busy
    for request in requests:
        assert request.generated == reference_generate(model, request.prompt, 5)
    assert engine.pool.used_blocks == 0


def test_prefix_sharing_reuses_only_full_prompt_blocks(model):
    engine = Engine(model, blocks=48, block_size=4, prefix_entries=3)
    prompt = [1, 2, 3, 4] * 3 + [5]
    first = engine.submit("a", prompt, 4)
    while engine.busy:
        engine.step()
    second = engine.submit("b", prompt, 4)
    while engine.busy:
        engine.step()
    assert first.generated == second.generated == reference_generate(model, prompt, 4)
    assert engine.pool.prefix_hits == 1
    assert second.prefill_cursor == len(prompt)
    engine.pool.clear_prefixes()
    engine.pool.assert_integrity()
    assert engine.pool.used_blocks == 0


def test_copy_on_write_never_changes_source(model):
    pool = BlockPool(model.config, blocks=16, block_size=4)
    pool.create("a", 16)
    pool.prepare_write("a", range(3))
    pool.keys[:, pool.handles["a"].blocks[0]] = 7
    pool.fork("a", "b")
    old = pool.handles["a"].blocks[0]
    pool.prepare_write("b", [3])
    new = pool.handles["b"].blocks[0]
    assert new != old
    np.testing.assert_array_equal(pool.keys[:, new], pool.keys[:, old])
    pool.keys[:, new] = 9
    assert (pool.keys[:, old] == 7).all()
    assert pool.cow_copies == 1
    pool.release("a")
    pool.release("b")
    pool.assert_integrity()


def test_failed_copy_on_write_is_atomic(model):
    pool = BlockPool(model.config, blocks=1, block_size=4)
    pool.create("a", 4)
    pool.fork("a", "b")
    with pytest.raises(CacheFull):
        pool.prepare_write("b", [0])
    pool.assert_integrity()
    assert pool.handles["a"].blocks == pool.handles["b"].blocks


def test_capacity_admission_cancellation_and_eventual_progress(model):
    engine = Engine(model, blocks=4, block_size=4, token_budget=3)
    engine.submit("a", [1] * 8, 8)
    b = engine.submit("b", [2] * 8, 8)
    engine.step()
    assert b.status == "queued"
    engine.cancel("a")
    engine.cancel("a")  # Idempotent cancellation.
    while engine.busy:
        engine.step()
    assert b.status == "completed"
    engine.pool.assert_integrity()
    assert engine.pool.used_blocks == 0


def test_impossible_request_rejected_instead_of_deadlocking(model):
    engine = Engine(model, blocks=1, block_size=4)
    request = engine.submit("a", [1] * 8, 8)
    assert request.status == "rejected" and not engine.busy


def test_randomized_allocator_ownership_stress(model):
    rng = np.random.default_rng(9)
    pool = BlockPool(model.config, blocks=48, block_size=4)
    counter = 0
    for _ in range(400):
        if not pool.handles or rng.random() < 0.6:
            counter += 1
            try:
                if pool.handles and rng.random() < 0.25:
                    source = rng.choice(list(pool.handles))
                    pool.fork(source, str(counter))
                else:
                    pool.create(str(counter), int(rng.integers(1, 33)))
            except CacheFull:
                pass
        else:
            pool.release(rng.choice(list(pool.handles)))
        if pool.handles:
            handle = pool.handles[rng.choice(list(pool.handles))]
            try:
                pool.prepare_write(handle.request_id, [int(rng.integers(handle.capacity))])
            except CacheFull:
                pass
        pool.assert_integrity()
    for request_id in list(pool.handles):
        pool.release(request_id)
    pool.assert_integrity()
    assert pool.used_blocks == 0


@pytest.mark.parametrize("text", ["", "hello", "💻 é नमस्ते", "a\x00b", "print('x')\n"])
def test_byte_tokenizer_roundtrip(text):
    assert ByteTokenizer.decode(ByteTokenizer.encode(text)) == text


def test_sampling_seed_and_nucleus():
    logits = np.array([1.0, 2.0, 3.0])
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    assert [sample(logits, a, 1) for _ in range(40)] == [sample(logits, b, 1) for _ in range(40)]
    assert sample(logits, a, 1, top_k=1) == 2
    assert sample(logits, a, 1, top_p=0.01) == 2


def test_model_causality(model):
    a, b = np.array([[1, 2, 3, 4]]), np.array([[1, 2, 7, 8]])
    np.testing.assert_allclose(model.forward(a)[:, :2], model.forward(b)[:, :2], atol=1e-7)
