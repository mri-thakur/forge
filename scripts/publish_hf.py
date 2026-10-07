"""Build, and optionally publish, the Hugging Face Space demo and the model weights.

    python scripts/publish_hf.py                                  # build dist/ only
    hf auth login                                                 # once, with a write token
    python scripts/publish_hf.py --space USER/forge-stories --models USER/forge-tinystories

dist/space/ is a Gradio Space: demo/app.py, the forge package, the tokenizer, and
the pretrained, SFT, and GRPO checkpoints. dist/models/ is a model repository with
the same checkpoints and a model card whose numbers are read from results/.
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
REPOSITORY = "https://github.com/mri-thakur/forge"
CHECKPOINTS = {  # file name: (checkpoint folder, description)
    "L": ("L", "pretrained on TinyStoriesV2 (542.6M tokens)"),
    "sft_L": ("sft_L", "L fine-tuned on 300,000 TinyStories-Instruct examples"),
    "grpo_kl0.1": ("grpo_kl0.1", "SFT model after 200 GRPO steps, KL coefficient 0.1"),
}
TOKENIZER = ROOT / "data" / "tinystories" / "tokenizer_4096.json"
SPACE_CARD = """---
title: Forge Stories
emoji: 📖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: false
short_description: 34M LM from scratch - pretrained vs SFT vs RL
---

A 34M-parameter language model built from scratch, shown at three stages of training:
pretrained, fine-tuned on instructions, and trained with reinforcement learning (GRPO).
Code, training, and results: {repository}
"""
REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.8.0+cpu
numpy>=1.26,<3
regex>=2024.4
"""


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def model_card():
    pretrained = read("results/phase2/L/evaluation.json")
    rates = {
        name: read(path)["satisfied"]["rate"]
        for name, path in (
            ("base", "results/sft/base_L_eval.json"),
            ("sft", "results/sft/sft_L_eval.json"),
            ("rl", "results/phase4/grpo_kl0.1_eval.json"),
        )
    }
    config = pretrained["training"]["config"]
    rows = "\n".join(
        f"| `{name}.npz` | {description} |" for name, (_, description) in CHECKPOINTS.items()
    )
    return f"""---
license: cdla-sharing-1.0
datasets:
- roneneldan/TinyStories
- roneneldan/TinyStoriesInstruct
language:
- en
---

# Forge TinyStories models

A {pretrained["training"]["parameters"] / 1e6:.1f}M-parameter decoder-only transformer
({config["layers"]} layers, width {config["dim"]}, {config["heads"]} heads and
{config["kv_heads"]} KV heads, RoPE, RMSNorm, SwiGLU, tied embeddings, context
{config["context"]}) with its own 4,096-token byte-level BPE tokenizer, built and trained
from scratch on one 6 GB laptop GPU. Code, training logs, and evaluation: {REPOSITORY}

| File | Model |
|---|---|
{rows}
| `tokenizer.json` | the byte-level BPE tokenizer (vocabulary 4,096) |

The pretrained model scores {pretrained["heldout"]["bits_per_byte"]:.4f} bits per byte on
the full TinyStoriesV2 validation split. On 500 held-out TinyStories-Instruct prompts,
the share of sampled stories meeting every checkable instruction (required words, a
given sentence, dialogue) is {rates["base"]:.1%} for the pretrained model,
{rates["sft"]:.1%} after supervised fine-tuning, and {rates["rl"]:.1%} after GRPO.

The weights are NumPy `.npz` files (a JSON `config` plus one array per weight) loaded
by the `forge` package:

```python
from forge.bpe import Tokenizer
from forge.model import NumpyModel
from forge.torch_backend import TorchModel

model = TorchModel(NumpyModel.load("grpo_kl0.1.npz"), "cpu", "sdpa")
tokenizer = Tokenizer.load("tokenizer.json")
```

Trained on TinyStories (Eldan and Li, 2023; CDLA-Sharing-1.0).
"""


def build():
    import gradio

    if DIST.exists():
        shutil.rmtree(DIST)
    space, models = DIST / "space", DIST / "models"
    shutil.copytree(
        ROOT / "src" / "forge", space / "forge", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy(ROOT / "demo" / "app.py", space / "app.py")
    for name, (checkpoint, _) in CHECKPOINTS.items():
        source = ROOT / "checkpoints" / checkpoint / "model.npz"
        for target in (space / "models" / f"{name}.npz", models / f"{name}.npz"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
    for folder in (space, models):
        shutil.copy(TOKENIZER, folder / "tokenizer.json")
    (space / "README.md").write_text(
        SPACE_CARD.format(gradio=gradio.__version__, repository=REPOSITORY), encoding="utf-8"
    )
    (space / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (models / "README.md").write_text(model_card(), encoding="utf-8")
    return space, models


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", help="publish the demo to this Space, e.g. USER/forge-stories")
    parser.add_argument("--models", help="publish the weights to this model repository")
    args = parser.parse_args()
    space, models = build()
    print(f"built {space} and {models}")
    if not (args.space or args.models):
        return
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"publishing as {api.whoami()['name']}")
    if args.models:
        api.create_repo(args.models, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.models, folder_path=models, repo_type="model")
        print(f"weights: https://huggingface.co/{args.models}")
    if args.space:
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(repo_id=args.space, folder_path=space, repo_type="space")
        print(f"demo: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
