# Progress

Last updated: 2026-10-06 13:55 (Claude Code, continuing from Codex after Codex hit
its usage limit mid-session).

## Running now

Check with `.\.venv\Scripts\python.exe scripts\detach.py --status`.

- `cuda_bench_calibration` (laptop, started 14:05, ~5 min): 128-request `mixed`
  trace at 5/20/50/100 req/s, trained checkpoint, CUDA + SDPA, seed 17, into
  `results/cuda_calibration_14m`. Purpose: find where the GPU stops keeping up, to
  choose the rates for the 1,000-request sweeps. If it stopped, delete that folder
  and rerun the command in `logs/cuda_bench_calibration.json`.

## Done

- CUDA environment in `.venv` (PyTorch 2.8.0+cu128) verified on the RTX 4050:
  63 tests pass, 2 MPS-only tests skip. CUDA matches the NumPy oracle for manual and
  SDPA attention, including through the paged KV cache.
- WikiText-103 imported to `data/wikitext103` with hashes and licensing in
  `manifest.json` (539.1M train bytes, 1.14M validation bytes).
- Training run `checkpoints/wikitext103_14m` finished (after Codex stopped reporting):
  - 14.26M parameters: 8 layers, dim 384, 4 heads / 2 KV heads, FFN 1152, byte vocab.
  - 30,000 updates × batch 8 × context 256 = 61.44M training bytes (11% of the
    corpus); bf16 autocast, SDPA, seed 17. 30.6 min, ~33.5k bytes/s sustained
    (44.2k in the short calibration; the laptop throttles under sustained load).
    Peak CUDA allocation 627 MiB.
  - Command: see "Long runs" in `runs/LAPTOP.md`.
- Full-split evaluation (`forge evaluate`, new in `src/forge/evaluation.py`):
  - **1.2724 nats/byte (1.836 bits/byte)** on all 1,142,171 validation bytes.
    Byte bigram 2.440, unigram 3.184, uniform 5.545. The training loop's 4 KB
    monitoring sample read 1.338 (pessimistic).
  - The loss curve was still falling at the end of the schedule; a longer run
    should improve it.
  - Greedy samples loop ("the second season , and ..."); sampled text is locally
    fluent WikiText style.
  - Outputs: `results/wikitext103_14m/` (evaluation.json, training_curve.jsonl),
    `docs/wikitext103_14m/` (RESULTS.md, loss_curve.png).
- README "Current evidence" and roadmap, and `runs/LAPTOP.md`, updated.
- Agent-independence: `scripts/detach.py` (tested: a job survives its starting
  session's job object being closed), `AGENTS.md` rules, `CLAUDE.md` imports them.
- `gpu_training.py` summaries now include `run_config` (batch/context/etc.).

## Next

1. README roadmap item 1: CUDA serving sweeps with the trained checkpoint, at least
   1,000 requests per setting across load, chunk budget, prefix reuse, and KV
   capacity. Calibrate rates with a short run first (`runs/LAPTOP.md`). Keep prompt
   plus output within the trained 256-byte context if output quality matters.
2. Optional: the Intel Mac Pro as a CPU benchmark/data worker (`runs/MAC.md`).
3. Optional: a longer training run (loss was still falling).

## Notes

- Codex ran as a separate Windows account (`CodexSandboxOffline`), which owns the
  files. Git needed `safe.directory` entries (added globally for `forge` and `eqdata`).
- The repository has no commits yet; everything above is uncommitted.
- The project lives in OneDrive. Large checkpoint writes can collide with sync; if a
  save ever fails with a permission error, pause OneDrive during training.
