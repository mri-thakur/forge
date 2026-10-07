# Plan: a language model built from scratch, end to end, on a laptop

Goal: a portfolio project for ML/DL/AI roles. One model taken through the whole
modern LLM lifecycle, every stage built from scratch and measured:

**own BPE tokenizer → pretraining → supervised fine-tuning → reinforcement learning
with verifiable rewards → live demo**, on a 6 GB RTX 4050 laptop GPU.

Target result (to be measured, not assumed): a model of roughly 15-40M parameters
that writes coherent children's stories, follows instructions such as "use these
words" and "include this sentence", and measurably follows them better after RL
than after supervised fine-tuning alone.

The phases follow Andrej Karpathy's *Neural Networks: Zero to Hero* lectures; phases
3-4 go beyond them (as his nanochat project does). Watch the matching lecture before
each phase and be able to explain every design decision in the code: interviews
will ask.

## Data

[TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023), CDLA-Sharing-1.0,
pinned revisions:

- Pretraining: `roneneldan/TinyStories` @ `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`,
  `TinyStoriesV2-GPT4-train.txt` (2.23 GB) and `-valid.txt`.
- Instructions: `roneneldan/TinyStoriesInstruct` @
  `ee050ed1f8720795be342921335e821856a2b42e`, `TinyStories-Instruct-train.txt`
  (2.66 GB) and `-valid.txt` (27 MB). Records are separated by `<|endoftext|>`; each
  has `Story:` plus some of `Words:`, `Features:` (Dialogue, BadEnding, Twist,
  Foreshadowing, MoralValue, Conflict), `Summary:`, `Random sentence:`, in varying
  order (`Summary:` can follow the story).

Verifiable constraints for rewards: required **words** (allowing inflections), the
**random sentence** included, **dialogue** present when requested. The other
features and the summary are kept in prompts but are not rewarded.

## Phases

| # | Phase | Lecture | Deliverable and metric |
|---|---|---|---|
| 0 | Foundations (done) | micrograd, makemore, Let's build GPT | From-scratch transformer, NumPy autograd engine, CUDA training, held-out evaluation, KV-cache inference engine. WikiText-103 byte model: 1.272 nats/byte. |
| 1 | Tokenizer | Let's build the GPT Tokenizer | Own byte-level BPE (GPT-4-style regex pre-split, `<\|endoftext\|>` special token) trained on TinyStoriesV2; vocab 4096, with 2048/8192 compared. Corpus encoded to uint16 shards. Metric: bytes per token vs the GPT-2 tokenizer. |
| 2 | Pretraining | Let's reproduce GPT-2 | Token-level training with gradient accumulation, SDPA flash attention, bf16, cosine schedule. Three sizes (about 3M/12M/35M) for a scaling curve; ablations at the smallest size, 3 seeds: exact vs bf16 RoPE angles, GQA vs full multi-head attention. Metric: held-out loss (and bits per byte), fixed samples. |
| 3 | Supervised fine-tuning | beyond the playlist | Fine-tune the best model on TinyStories-Instruct, loss on story tokens only. Metric: constraint satisfaction on 500 fixed held-out prompts. |
| 4 | RL with verifiable rewards | beyond the playlist | GRPO with the constraint verifiers as reward and a KL penalty to the SFT model; rollouts through Forge's batched KV-cache engine. Metrics: satisfaction before/after, fluency (held-out loss under the pretrained model), reward-hacking examples, KL-coefficient ablation. |
| 5 | Demo and write-up | | Live demo (in-browser or a hosted Space), README rewritten around the results, plots, resume bullets. |

The serving/scheduling work already in the repo (paged KV cache, batching policies,
profiling) becomes supporting infrastructure: it makes RL rollouts and the demo
fast, and is summarized in an appendix rather than the headline.

## Where it runs

Everything runs on one laptop (Windows, RTX 4050 6 GB, `main` branch). A Mac Pro
was considered for the CPU-side instruction data work but is not used.

## Rules

- Every number in the README or write-up comes from a file in `results/`.
- Fixed seeds and pinned data revisions; held-out sets are never trained on.
- Report what did not work as well as what did.
