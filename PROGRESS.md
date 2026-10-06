# Progress (laptop, `main`)

Last updated: 2026-10-06 15:05. The Mac Pro keeps its own log in
`runs/MAC_PROGRESS.md` on the `mac` branch.

**Direction changed on 2026-10-06** (see [PLAN.md](PLAN.md)): the project is now a
language model built from scratch end to end on TinyStories (tokenizer →
pretraining → SFT → RL → demo), for ML/DL roles. The serving work below becomes
supporting infrastructure; the running sweep finishes but no further serving sweeps
are planned.

## Running now

Check with `.\.venv\Scripts\python.exe scripts\detach.py --status`.

- `cuda_sweep_mixed` (started 14:43, ~45-60 min), one chained job:
  1. 1,000-request `mixed` sweep at 10/20/25/30/35 req/s, seeds 101/103/107, all
     policies, trained checkpoint, CUDA + SDPA → `results/cuda_sweep_mixed`;
  2. `forge report` → `docs/cuda_sweep_mixed`;
  3. `head_of_line` calibration, 128 requests at 5/10/20/40 req/s, seed 17 →
     `results/cuda_calibration_hol`.
  If it stopped: the sweep cannot resume; delete the unfinished output folder and
  rerun the chain recorded in `logs/cuda_sweep_mixed.json` (skip finished parts).
  Low priority after the change of direction; do not rerun it if it is in the way.

## Calibration result (mixed, 128 requests, seed 17)

All policies meet the SLO (500 ms TTFT, 50 ms mean TPOT) for ≥99% of requests up
to 20 req/s; at 40 req/s SLO fraction falls to 35% continuous, 31% chunked, 11%
static. Output throughput plateaus around 370-460 tok/s. Capacity knee lies
between 20 and 40 req/s, hence the sweep rates above.
`results/cuda_calibration_14m`.

## Done

- **Serving-engine overhead cut** (commit d9c6e7a; before = 4eec0f9). The paged
  cache built scatter/gather indices per request per layer with host-to-device
  copies and mask syncs; RoPE tables were recomputed for q and k in every layer.
  Both now happen once per forward. Same trace and commands before/after
  (`results/engine_overhead/`): paged decode 15.98 → 8.79 ms/token; engine
  iteration static/continuous/chunked 19.6/22.8/22.2 → 13.5/15.6/14.9 ms; p99 TTFT
  628/82/191 → 325/35/83 ms (64 requests, 5 req/s, `mixed`, seed 17). Greedy output
  identical. A new test covers reads after copy-on-write; removing the cache
  invalidation makes 5 tests fail.
  - Remaining cost is kernel launches (~360 small kernels per forward): full
    recomputation is 7.81 ms/token, below both KV paths. That justifies the
    roadmap's fused-kernel / CUDA-graph extension later.
  - RoPE tables are now float32 under bf16 autocast. The existing bf16 run trained
    with bf16 angles (up to 0.5 rad off near position 256); fp32 inference uses
    exact ones. Future training runs get exact angles.

- CUDA environment in `.venv` (PyTorch 2.8.0+cu128) verified on the RTX 4050:
  63 tests pass, 2 MPS-only tests skip. CUDA matches the NumPy oracle for manual and
  SDPA attention, including through the paged KV cache.
- WikiText-103 imported to `data/wikitext103` with hashes and licensing in
  `manifest.json` (539.1M train bytes, 1.14M validation bytes).
- Training run `checkpoints/wikitext103_14m` (weights also on the GitHub release
  `wikitext103-14m`):
  - 14.26M parameters: 8 layers, dim 384, 4 heads / 2 KV heads, FFN 1152, byte vocab.
  - 30,000 updates × batch 8 × context 256 = 61.44M training bytes (11% of the
    corpus); bf16 autocast, SDPA, seed 17. 30.6 min, ~33.5k bytes/s sustained
    (44.2k in the short calibration). Peak CUDA allocation 627 MiB.
  - Command: see "Long runs" in `runs/LAPTOP.md`.
- Full-split evaluation (`forge evaluate`): **1.2724 nats/byte (1.836 bits/byte)**
  on all 1,142,171 validation bytes; byte bigram 2.440, unigram 3.184. The loss
  curve was still falling. Outputs in `results/wikitext103_14m/`,
  `docs/wikitext103_14m/`.
- `scripts/detach.py`: jobs survive the starting session (tested by closing a
  kill-on-close job object around it), `--status`, `--stop`.
- **Found and fixed: Windows power throttling made detached jobs 8x slower**
  (196-217 ms vs 26 ms per engine iteration). Windowless background processes get
  EcoQoS (efficiency cores, low clocks, coarse timers), and the exemption is not
  inherited, so the launcher now exempts every descendant. After the fix: 29 ms.
  The first CUDA calibration ran throttled and was deleted.
- GitHub: private repo `mri-thakur/forge`; Mac handoff in `runs/MAC.md`.

## Next

Laptop, following [PLAN.md](PLAN.md):

1. Phase 1, tokenizer: download TinyStoriesV2 (pinned revision) to `data/raw/`;
   implement byte-level BPE (`src/forge/bpe.py`, GPT-4-style regex pre-split,
   `<|endoftext|>` special token) with tests; train at vocab 2048/4096/8192 on a
   sample, compare bytes per token with the GPT-2 tokenizer; encode the corpus to
   uint16 shards.
2. Phase 2, pretraining: token-level training with gradient accumulation; calibrate
   throughput; three sizes plus ablations.
3. Merge the Mac's `mac` branch (instruction data, verifiers, eval harness) before
   phase 3.

## Notes

- Codex ran as a separate Windows account (`CodexSandboxOffline`), which owns the
  files. Git needed `safe.directory` entries (added globally for `forge` and `eqdata`).
- Commits are authored by the user alone: no co-author trailers or AI mentions.
- The project lives in OneDrive. Large checkpoint writes can collide with sync; if a
  save ever fails with a permission error, pause OneDrive during training.
- `results/wikitext103_14m/evaluation.json` records `heldout.seconds` from a
  throttled run (29.9 s); the losses are unaffected.
