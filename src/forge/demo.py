"""Live demo logic: build an instruction, stream a story from a model, and show how
the story scores on the constraint checks (the same checks that were the RL reward).
The web interface in demo/app.py is a thin layer over these functions.
"""

from __future__ import annotations

from forge.bpe import ENDOFTEXT
from forge.instruct import check, prompt
from forge.sft import offline_engine

FEATURES = ("Dialogue", "BadEnding", "Twist", "Foreshadowing", "MoralValue", "Conflict")


def build_record(summary="", words="", features=(), sentence=""):
    """An instruction record from form fields; `words` is comma-separated."""
    return {
        "summary": summary.strip() or None,
        "words": [word.strip() for word in words.split(",") if word.strip()],
        "features": [feature for feature in FEATURES if feature in features],
        "sentence": sentence.strip() or None,
        "story": None,
    }


def stream_story(model, tokenizer, record, temperature=0.8, top_p=0.95, seed=17, max_tokens=400):
    """Yield (story so far, finished) after every generated token. The story is
    finished when the model emits end-of-text; it stops early at `max_tokens`."""
    end = tokenizer.special[ENDOFTEXT]
    ids = tokenizer.encode_ordinary(prompt(record))
    budget = min(max_tokens, model.config.context - len(ids))
    if budget < 1:
        raise ValueError("the instruction is too long for the model's context")
    engine = offline_engine(model, 1, 1, seed)
    request = engine.submit("demo", ids, budget, temperature=temperature, top_p=top_p, stop=end)
    while engine.busy:
        engine.step()
        generated = request.generated
        finished = bool(generated) and generated[-1] == end
        yield tokenizer.decode(generated[:-1] if finished else generated).strip(), finished


def verdict(record, story, finished=True):
    """One line of Markdown: each checkable constraint marked ✓ or ✗, and the reward."""
    result = check(record, story)
    parts = []
    if "words" in result:
        marks = ", ".join(
            f"{'✓' if hit else '✗'} {word}"
            for word, hit in zip(record["words"], result["words"]["hits"])
        )
        parts.append(f"**Words:** {marks}")
    if "sentence" in result:
        parts.append(f"**Sentence:** {'✓ included' if result['sentence'] else '✗ missing'}")
    if "dialogue" in result:
        parts.append(f"**Dialogue:** {'✓ present' if result['dialogue'] else '✗ missing'}")
    if not parts:
        return "No checkable instruction: add required words, a sentence, or Dialogue."
    ending = "" if finished else " (the story did not finish)"
    return " · ".join(parts) + f" · **Reward:** {result['reward']:.2f}{ending}"


def load_models(paths, device="cpu"):
    """{label: model.npz path} -> {label: model} for CPU inference."""
    from forge.model import NumpyModel
    from forge.torch_backend import TorchModel

    return {
        label: TorchModel(NumpyModel.load(path), device, "sdpa") for label, path in paths.items()
    }
