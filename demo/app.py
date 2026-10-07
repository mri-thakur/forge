"""Forge demo: one 34M-parameter model at three stages writes a story for the same
instruction, and each story is scored by the checks that served as the RL reward.

    python demo/app.py          # from the repository, models from checkpoints/

As a Hugging Face Space (built by scripts/publish_hf.py) the forge package,
tokenizer, and models sit next to this file.
"""

import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for candidate in (HERE, ROOT / "src"):  # the Space, then the repository
    if (candidate / "forge").is_dir():
        sys.path.insert(0, str(candidate))
        break

from forge.bpe import Tokenizer  # noqa: E402
from forge.demo import FEATURES, build_record, load_models, stream_story, verdict  # noqa: E402

REPOSITORY = "https://github.com/mri-thakur/forge"
STAGES = {  # column title: checkpoint name
    "Pretrained": "L",
    "+ supervised fine-tuning": "sft_L",
    "+ reinforcement learning (GRPO)": "grpo_kl0.1",
}
# Held-out evaluation prompts (TinyStoriesInstruct validation split, CDLA-Sharing-1.0).
EXAMPLES = [
    [
        "seat, wallet, dirty",
        "",
        ["Dialogue", "BadEnding"],
        "Timmy loses his beloved wallet at the park after a dog knocks him off the swing, "
        "and despite looking for it and involving the police, he never finds it.",
    ],
    [
        "accept, pizza, unique",
        'She excitedly asked the pizza, "Will you be my friend?"',
        ["Dialogue"],
        "Sarah befriends a magical pizza that appears in front of her and they become best "
        "friends, doing everything together.",
    ],
    [
        "wash, curtain, clumsy",
        "Tim loved to explore and play.",
        [],
        "A clumsy monkey named Tim rips a curtain while playing, but his mom teaches him that "
        "it's okay to make mistakes and they work together to fix it.",
    ],
    [
        "escape, comb, ordinary",
        "To her surprise, the comb broke the lock and the door opened.",
        ["Dialogue"],
        "An ordinary bunny escapes from a cage using a comb to break the lock.",
    ],
]


def checkpoint(name):
    in_space = HERE / "models" / f"{name}.npz"
    return in_space if in_space.exists() else ROOT / "checkpoints" / name / "model.npz"


tokenizer_path = HERE / "tokenizer.json"
if not tokenizer_path.exists():
    tokenizer_path = ROOT / "data" / "tinystories" / "tokenizer_4096.json"
TOKENIZER = Tokenizer.load(tokenizer_path)
MODELS = load_models({title: checkpoint(name) for title, name in STAGES.items()})


def write(words, sentence, features, summary, temperature, seed):
    """Stream each stage's story in turn, then its check results."""
    record = build_record(summary, words, features, sentence)
    outputs = [""] * (2 * len(MODELS))
    for i, model in enumerate(MODELS.values()):
        story, finished = "", False
        for step, (story, finished) in enumerate(
            stream_story(model, TOKENIZER, record, temperature, seed=int(seed))
        ):
            if step % 3 == 0 or finished:
                outputs[2 * i] = story
                yield outputs
        outputs[2 * i] = story
        outputs[2 * i + 1] = verdict(record, story, finished)
        yield outputs


INTRO = f"""
# Forge: a language model built from scratch

Each column is the same 34M-parameter transformer at a different stage of training:
**pretrained** on TinyStories, **fine-tuned** on instructions, then trained with
**reinforcement learning** (GRPO) to satisfy them. Under each story are the rule-based
checks that served as the RL reward. On 500 held-out instructions, the share of stories
meeting every check rose from 1.8% (pretrained) to 35.4% (fine-tuned) to 56.1% (RL).
The tokenizer, model, training, and inference engine are all implemented from scratch:
[code and results]({REPOSITORY}).

The model only knows children's stories, so keep the words simple. Runs on CPU, so the
three stories take about 15 seconds.
"""

with gr.Blocks(title="Forge: a language model built from scratch") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            words = gr.Textbox(
                label="Required words (comma-separated)", value=EXAMPLES[1][0], max_lines=1
            )
            sentence = gr.Textbox(
                label="A sentence the story must include (optional)",
                value=EXAMPLES[1][1],
                max_lines=1,
            )
            features = gr.CheckboxGroup(
                list(FEATURES),
                value=EXAMPLES[1][2],
                label="Features",
                info="Only Dialogue is checked; the others still steer the story.",
            )
            summary = gr.Textbox(label="Summary (optional)", value=EXAMPLES[1][3], lines=2)
        with gr.Column(scale=1):
            temperature = gr.Slider(0.1, 1.2, value=0.8, step=0.1, label="Temperature")
            seed = gr.Number(value=17, precision=0, label="Seed")
            button = gr.Button("Write the stories", variant="primary")
    cells = []  # story and checks for each stage, in STAGES order
    with gr.Row():
        for title in STAGES:
            with gr.Column():
                gr.Markdown(f"### {title}")
                cells.append(gr.Textbox(label="Story", lines=14, interactive=False))
                cells.append(gr.Markdown())
    gr.Examples(EXAMPLES, inputs=[words, sentence, features, summary])
    button.click(write, [words, sentence, features, summary, temperature, seed], cells)

if __name__ == "__main__":
    demo.launch()
