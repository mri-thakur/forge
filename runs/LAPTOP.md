# Laptop: fastest route to the first measured release

Known hardware: Intel i5-13420H; NVIDIA RTX 4050 Laptop GPU, 6,141 MiB VRAM.
All development, data preparation, training, and evaluation run on this machine.

## Environment

`.venv` holds PyTorch 2.8.0+cu128 with the serving, plotting, and data extras. The
full test suite, including the CUDA oracle checks and the HTTP server test, passes
on this laptop. To recreate it:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev,serve,plots]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name())"
.\.venv\Scripts\python.exe -m pytest
```

Set laptop power to a consistent plugged-in mode. Record the policy rather than
varying it between runs. For later Triton work, use a verified Linux/WSL2
environment; it is not needed for this first PyTorch implementation.

## Long runs

Start anything long through the detached launcher, so it keeps running if the
terminal, VS Code, or a coding agent exits. Windows power-throttles windowless
background processes (efficiency cores, low clocks, coarse timers), which made a
detached CUDA benchmark 8x slower; the launcher opts every process it starts out of
that. A job started some other way in the background is not protected. Keep the laptop plugged in and stop it
from sleeping; sleep or shutdown still interrupt a run, which then resumes from its
last checkpoint (every 100 updates) by rerunning the same command with `--resume`.

```powershell
.\.venv\Scripts\python.exe scripts\detach.py --name wikitext103_14m -- python -m forge train --backend torch --device cuda --attention sdpa --precision bf16 --data data/wikitext103 --out checkpoints/wikitext103_14m --steps 30000 --dim 384 --layers 8 --context 256 --batch 8
.\.venv\Scripts\python.exe scripts\detach.py --status
.\.venv\Scripts\python.exe -m forge evaluate --backend torch --device cuda --attention sdpa --run checkpoints/wikitext103_14m --data data/wikitext103 --out results/wikitext103_14m --report docs/wikitext103_14m
```

The first command reproduces the published WikiText-103 run; it refuses to start if
that `--out` folder already holds a run.

## Calibration, then training

Start with a short run that fits comfortably. The float32/manual-attention path
is intentionally simple; it has no claimed FlashAttention or compiled speedup.
`--synthetic` is a diagnostic dataset only. For real training, replace it with
`--text path/to/licensed_corpus.txt` and retain the corpus/license provenance.

```powershell
.\.venv\Scripts\python.exe -m forge prepare --synthetic --out data/diagnostic
.\.venv\Scripts\python.exe -m forge train --backend torch --device cuda --data data/diagnostic --out checkpoints/cuda_calibration --steps 100 --dim 128 --layers 4 --context 128 --batch 8
```

Measure tokens/sec and peak memory before choosing model size or a token budget.
The CLI takes **total optimizer steps**, not extra steps. Choose a complete
schedule upfront. Interrupting normally saves a resume checkpoint:

```powershell
.\.venv\Scripts\python.exe -m forge train --backend torch --device cuda --data data/diagnostic --out checkpoints/cuda_calibration --steps 100 --dim 128 --layers 4 --context 128 --batch 8 --resume
```

Expected outputs: `training.jsonl`, `resume.pt`, `model.npz`, `summary.json`.
Actual duration is unknown until calibration. Do not assign a 1.5B-token run on
the basis of VRAM alone. A 1-hour run at measured rate R consumes roughly 3600R
training tokens; budget evaluation/checkpoint overhead separately.

## Serving measurements

Use a separate output folder per workload and configuration. Start with random
weights for hardware calibration, then repeat with the trained checkpoint.

```powershell
.\.venv\Scripts\python.exe -m forge ladder --backend torch --device cuda --dim 256 --layers 6 --out results/cuda_ladder.json --repeats 10
.\.venv\Scripts\python.exe -m forge bench --backend torch --device cuda --dim 256 --layers 6 --requests 128 --rates 5 20 50 --seeds 17 29 43 --workload head_of_line --out results/cuda_calibration
.\.venv\Scripts\python.exe -m forge bench --backend torch --device cuda --checkpoint checkpoints/cuda_calibration/model.npz --requests 1000 --rates 5 20 50 --seeds 101 103 107 --workload mixed --out results/cuda_measured
.\.venv\Scripts\python.exe -m forge report --input results/cuda_measured/benchmark.json --out docs/cuda_measured
```

Expected outputs: `benchmark.json`, exact trace JSONs, per-request metrics,
`RESULTS.md`, throughput/TTFT/goodput plots. Rates are a calibration starting point,
not a promised sustainable load. HTTP serving is a separate, optional path:

```powershell
.\.venv\Scripts\python.exe -m forge serve --backend torch --device cuda --checkpoint checkpoints/cuda_calibration/model.npz
```
