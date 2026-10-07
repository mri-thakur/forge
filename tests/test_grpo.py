import json

import numpy as np
import pytest

from forge.bpe import ENDOFTEXT, Tokenizer, count_chunks
from forge.instruct import prompt
from forge.model import ModelConfig, NumpyModel

torch = pytest.importorskip("torch")

from forge import grpo  # noqa: E402
from forge.sft import story_loss  # noqa: E402
from forge.torch_backend import TorchModel  # noqa: E402

STORIES = [
    ("Tom saw a big dog. The dog ran.", ["dog", "run"]),
    ('Lily said, "Hello!" to her cat.', ["cat"]),
    ("The sun was hot. Ben went to swim.", ["sun", "swim"]),
]


@pytest.fixture
def setup(tmp_path):
    with (tmp_path / "train.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(12):
            story, words = STORIES[i % len(STORIES)]
            record = {
                "id": f"train-{i:07d}",
                "summary": "A short day.",
                "words": words,
                "features": ["Dialogue"] if i % 2 else [],
                "sentence": None,
                "story": story,
            }
            stream.write(json.dumps(record) + "\n")
    lines = (tmp_path / "train.jsonl").read_text("utf-8").splitlines()
    (tmp_path / "exclude.jsonl").write_text(lines[0] + "\n", "utf-8")
    corpus = " ".join(story for story, _ in STORIES) * 10 + " Summary Words Story Features"
    tokenizer = Tokenizer.train(count_chunks(corpus), 300)
    tokenizer.save(tmp_path / "tokenizer.json")
    model = NumpyModel(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            dim=16,
            heads=2,
            kv_heads=1,
            layers=1,
            hidden=32,
            context=96,
            seed=3,
        )
    )
    model.save(tmp_path / "sft.npz")
    return tmp_path, tokenizer, model


def test_group_advantages_are_standardized_within_each_group():
    advantages = grpo.group_advantages([1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5], group=4)
    first, second = advantages[:4], advantages[4:]
    assert abs(first.mean()) < 1e-9 and abs(first.std() - 1) < 1e-4
    assert first[0] > 0 and (first[1:] < 0).all()
    np.testing.assert_array_equal(second, 0)  # equal rewards carry no signal


def test_pack_masks_exactly_the_generated_tokens():
    samples = [{"prompt": [5, 6, 7], "tokens": [8, 9]}, {"prompt": [1], "tokens": [2, 3, 4]}]
    x, y, mask = grpo.pack(samples, "cpu")
    assert x.shape == (2, 4)
    for row, sample in enumerate(samples):
        scored = y[row][mask[row].bool()].tolist()
        assert scored == sample["tokens"]
        assert x[row, : len(sample["prompt"])].tolist() == sample["prompt"]


def test_unfinished_stories_earn_nothing():
    record = {"words": ["dog"], "sentence": None, "features": [], "story": ""}
    sample = {"record": record, "story": "The dog ran.", "finished": True}
    assert grpo.score(sample)["reward"] == 1.0
    assert grpo.score({**sample, "finished": False})["reward"] == 0.0


def test_kl_is_exact_and_its_gradient_vanishes_at_the_reference(setup):
    _, _, model = setup
    policy, reference = TorchModel(model), TorchModel(model)
    for weight in policy.weights.values():
        weight.requires_grad_(True)
    samples = [{"prompt": [5, 6, 7], "tokens": [8, 9, 10]}, {"prompt": [1, 2], "tokens": [3]}]
    terms = grpo.accumulate_gradients(policy, reference, samples, np.zeros(2), kl_coef=1.0)
    assert abs(terms["kl"]) < 1e-6
    for weight in policy.weights.values():
        assert weight.grad.abs().max() < 1e-6

    with torch.no_grad():
        policy.weights["layers.0.q"] += 0.5
    x, y, _ = grpo.pack(samples, "cpu")
    _, kl, _ = grpo.token_terms(policy, reference, x, y)
    p = policy.forward_tensor(x).double().log_softmax(-1)
    q = reference.forward_tensor(x).double().log_softmax(-1)
    expected = (p.exp() * (p - q)).sum(-1)
    torch.testing.assert_close(kl.double(), expected, atol=1e-5, rtol=1e-4)


def test_a_policy_gradient_step_raises_advantage_weighted_log_probability(setup):
    _, _, model = setup
    policy, reference = TorchModel(model), TorchModel(model)
    for weight in policy.weights.values():
        weight.requires_grad_(True)
    samples = [{"prompt": [5, 6], "tokens": [8, 9, 10]}, {"prompt": [5, 6], "tokens": [3, 4]}]
    advantages = np.array([1.0, -1.0])

    def objective():
        x, y, mask = grpo.pack(samples, "cpu")
        with torch.no_grad():
            chosen, _, _ = grpo.token_terms(policy, reference, x, y)
        return float((torch.as_tensor(advantages)[:, None] * chosen * mask).sum())

    before = objective()
    grpo.accumulate_gradients(policy, reference, samples, advantages, kl_coef=0.0)
    with torch.no_grad():
        for weight in policy.weights.values():
            weight -= 0.05 * weight.grad
    assert objective() > before


def test_rollouts_follow_the_prompt_and_stop_at_end_of_text(setup):
    tmp_path, tokenizer, model = setup
    records = [json.loads(line) for line in open(tmp_path / "train.jsonl")][:2]
    samples = grpo.rollouts(TorchModel(model), tokenizer, records, 3, 20, "t")
    assert [s["record"]["id"] for s in samples] == [records[0]["id"]] * 3 + [records[1]["id"]] * 3
    end = tokenizer.special[ENDOFTEXT]
    for s in samples:
        assert s["prompt"] == tokenizer.encode_ordinary(prompt(s["record"]))
        assert s["finished"] == (s["tokens"][-1] == end)
        assert len(s["tokens"]) == 20 or s["finished"]
    # Different samples of one prompt differ (temperature 1).
    assert len({tuple(s["tokens"]) for s in samples[:3]}) > 1


def run(tmp_path, out, **kwargs):
    options = dict(
        steps=3,
        prompts_per_step=2,
        group=2,
        lr=1e-2,
        kl=0.1,
        max_new_tokens=16,
        warmup=1,
        micro_batch=3,
        max_batch=4,
        pool_size=100,
        exclude=tmp_path / "exclude.jsonl",
        save_every=1,
        device="cpu",
        attention="manual",
        precision="fp32",
    )
    options.update(kwargs)
    return grpo.train_grpo(
        tmp_path / "sft.npz",
        tmp_path / "train.jsonl",
        tmp_path / "tokenizer.json",
        tmp_path / out,
        **options,
    )


def test_training_logs_every_step_and_never_uses_excluded_prompts(setup, monkeypatch):
    tmp_path, _, _ = setup
    used, original = [], grpo.rollouts

    def spy(model, tokenizer, records, *args, **kwargs):
        used.extend(r["id"] for r in records)
        return original(model, tokenizer, records, *args, **kwargs)

    monkeypatch.setattr(grpo, "rollouts", spy)
    summary = run(tmp_path, "run", snapshot_every=2)
    assert summary["steps"] == 3 and summary["prompt_pool"] == 11
    assert [p.name for p in (tmp_path / "run").glob("model_step*.npz")] == ["model_step0002.npz"]
    assert len(used) == 6 and "train-0000000" not in used
    rows = [json.loads(line) for line in open(tmp_path / "run" / "grpo.jsonl")]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert all(0 <= row["reward"] <= 1 and row["kl"] >= 0 for row in rows)
    assert rows[0]["kl"] < 1e-6  # the first rollouts come from the reference itself
    tuned = NumpyModel.load(tmp_path / "run" / "model.npz").weights["embed"]
    assert not np.allclose(tuned, NumpyModel.load(tmp_path / "sft.npz").weights["embed"])


def test_an_interrupted_run_resumes_to_the_same_weights(setup, monkeypatch):
    tmp_path, _, _ = setup
    run(tmp_path, "straight")
    calls, original = [], grpo.group_advantages

    def interrupt_on_third_step(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(grpo, "group_advantages", interrupt_on_third_step)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "interrupted")
    monkeypatch.setattr(grpo, "group_advantages", original)
    with pytest.raises(ValueError):
        run(tmp_path, "interrupted")  # refuses to overwrite without --resume
    run(tmp_path, "interrupted", resume=True)
    straight = NumpyModel.load(tmp_path / "straight" / "model.npz").weights
    resumed = NumpyModel.load(tmp_path / "interrupted" / "model.npz").weights
    for key in straight:
        np.testing.assert_allclose(resumed[key], straight[key], rtol=0, atol=1e-6)


def test_story_loss_scores_each_story_after_end_of_text(setup):
    _, tokenizer, model = setup
    torch_model = TorchModel(model)
    stories = ["Tom saw a big dog.", "", "The sun was hot."]
    result = story_loss(torch_model, tokenizer, stories, batch=2)
    assert result["per_story"][1] is None and result["stories"] == 3
    end = tokenizer.special[ENDOFTEXT]
    ids = [end] + tokenizer.encode_ordinary(stories[0])
    log_probs = torch_model.forward_tensor(np.array([ids[:-1]])).double().log_softmax(-1)[0]
    expected = -float(log_probs[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].mean())
    assert abs(result["per_story"][0] - expected) < 1e-4
    lengths = [len(tokenizer.encode_ordinary(s)) for s in stories]
    assert result["tokens"] == sum(lengths)
