# Mac Pro: instruction data, constraint verifiers, evaluation harness

Read [PLAN.md](../PLAN.md) first. The project is a language model built from
scratch end to end (tokenizer → pretraining → SFT → RL → demo). The laptop does the
GPU work (tokenizer, pretraining, later SFT and RL). This Mac builds what phases 3-4
need and the laptop cannot do in parallel: the TinyStories-Instruct data pipeline,
the rule-based constraint verifiers that become the RL reward, and the evaluation
harness. All of it is CPU work and independent of the tokenizer.

Hardware: MacPro7,1, Xeon W-3245 (16 cores / 32 threads), 96 GiB RAM, Radeon Pro
W5500X 8 GB. No CUDA; PyTorch is not needed for this assignment.

## Ground rules

- Work on the `mac` branch; never push to `main`. Before each milestone run
  `git fetch && git merge origin/main` and rerun the tests.
- Keep `runs/MAC_PROGRESS.md` current (create it): what finished, what is running,
  the exact next command. Commit and push it with your work after every milestone
  and whenever a long job starts or ends, so nothing is lost if you stop mid-task.
- Commit messages: no Co-Authored-By trailers and no mention of AI tools.
- New code lives in `src/forge/instruct.py` and `tests/test_instruct.py`, run as
  `python -m forge.instruct <subcommand>`. Do not edit other source files (the laptop
  is changing them); if you find a bug elsewhere, describe it in `MAC_PROGRESS.md`.
- Jobs longer than about two minutes run detached and awake:
  `.venv/bin/python scripts/detach.py --name <job> -- caffeinate -i .venv/bin/python -m forge.instruct <args>`
  (`--status` lists jobs, `--stop <job>` ends one). The launcher's POSIX path is
  untested: check once that a 30-second job keeps running after the agent's terminal
  closes.
- Every number you report comes from a file in `results/instruct/`.

## Milestone 0: environment

```bash
gh auth login                              # GitHub.com, HTTPS, browser; account mri-thakur
gh repo clone mri-thakur/forge ~/forge && cd ~/forge
git switch -c mac && git push -u origin mac
python3 --version                          # needs 3.11 or newer (brew install python@3.12)
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest                 # torch-dependent modules skip; the rest must pass
```

Download the pinned instruction data (2.7 GB) into the Git-ignored `data/raw/`, and
record SHA-256 hashes:

```bash
mkdir -p data/raw && cd data/raw
base=https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/ee050ed1f8720795be342921335e821856a2b42e
curl -L -O "$base/TinyStories-Instruct-valid.txt"
curl -L -O "$base/TinyStories-Instruct-train.txt"
shasum -a 256 TinyStories-Instruct-*.txt
cd ../..
```

## Milestone 1: parse and prepare

Format: records separated by lines equal to `<|endoftext|>`. Each has `Story:` (the
text starts on the next line and runs until the next field line or the record end)
and some of `Summary:`, `Words:` (comma separated), `Features:` (comma separated:
Dialogue, BadEnding, Twist, Foreshadowing, MoralValue, Conflict), and
`Random sentence:`, in varying order; `Summary:` may come after the story. Inspect
the files before trusting this description and record any other variants.

`python -m forge.instruct prepare --train data/raw/TinyStories-Instruct-train.txt --valid data/raw/TinyStories-Instruct-valid.txt --out data/instruct`
writes:

- `train.jsonl`, `valid.jsonl`: one record per line,
  `{"id": "train-0000123", "summary": str|null, "words": [str], "features": [str], "sentence": str|null, "story": str}`.
- `eval_prompts.jsonl`: 500 records drawn from `valid` with seed 17 among those with
  at least one verifiable constraint (words, sentence, or the Dialogue feature).
  This fixed set scores every model in phases 3-4; never train on `valid`.
- `manifest.json`: source revision, file SHA-256s, record counts, malformed or
  skipped records with reasons, license (CDLA-Sharing-1.0).

Also `prompt(record) -> str` returns the canonical prompt, fields in this fixed
order and only when present, ending where the model starts writing:

```text
Summary: ...
Words: a, b, c
Features: Dialogue, Twist
Random sentence: ...
Story:
```

Write field statistics (presence rates, story length distribution) for both splits
to `results/instruct/stats.json`.

## Milestone 2: constraint verifiers

`check(record, story) -> dict` scores a generated story against a record's
constraints:

- **words**: each required word appears as a whole word, case-insensitively,
  allowing regular inflections (jump → jumps, jumped, jumping; happy → happier;
  try → tried). Report per-word hits.
- **sentence**: the random sentence appears after normalizing case, whitespace, and
  quote characters.
- **dialogue** (only when `Dialogue` is a feature): the story contains quoted speech.
- `reward`: the mean over applicable constraint types (words counted as the
  fraction found), in [0, 1]; `satisfied`: every applicable check passes.

Acceptance, measured on the gold stories of `valid` (each scored against its own
record) and written to `results/instruct/gold_check.json`:

- word, sentence, and dialogue checks each pass on at least 97% of gold stories;
  list and explain the failures (they reveal verifier bugs or data noise);
- the same checks against a *different* record's constraints (shuffled pairs) give
  the base rate, which must be far lower; report it.

`tests/test_instruct.py` covers parsing variants (including summary after story),
inflections, normalization, and dialogue detection.

## Milestone 3: evaluation harness

`python -m forge.instruct evaluate --samples <jsonl> --prompts data/instruct/eval_prompts.jsonl --out <json>`,
where each sample line is `{"id": ..., "story": ...}`. Output: satisfaction rate and
mean reward overall and per constraint type, counts, and missing ids. Running it on
the gold stories reproduces the milestone 2 numbers.

Push after each milestone and tell the user it is ready for the laptop to merge.
