# Progress (laptop, `main`)

Last updated: 2026-10-06 16:30. The Mac Pro keeps its own log in
`runs/MAC_PROGRESS.md` on the `mac` branch (Mac offline so far).

**Direction changed on 2026-10-06** (see [PLAN.md](PLAN.md)): the project is now a
language model built from scratch end to end on TinyStories (tokenizer →
pretraining → SFT → RL → demo), for ML/DL roles. The serving work below is
supporting infrastructure.

## Running now

Check with `.\.venv\Scripts\python.exe scripts\detach.py --status`.

- `phase2` (started 16:22; S, ablations, and M done; L resumed at 23:04 from its
  step-100 checkpoint, expected to finish around 05:45 on 10-07):
  `python scripts/run_plan.py runs/plans/phase2.json --code ../forge-frozen`.
  The frozen worktree was moved to fc27cc2 for L, adding the thermal guard
  (`forge/thermal.py`): after each step, at GPU ≥ 85 °C, training sleeps until
  75 °C. On L about 40% of steps pause for ~2.3 s; training-only speed rose to
  ~32k tok/s because the GPU no longer throttles its clocks.
  Runs S (115M tokens), six ablation runs at 60M tokens (exact vs bf16 RoPE
  angles, GQA vs MHA, seeds 17/29), M (315M), L (543M ≈ one pass over the data),
  each evaluated on the full validation split into `results/phase2/<run>`; S, M, L
  also get `docs/phase2/<run>/RESULTS.md`. Code runs from the git worktree
  `../forge-frozen` at commit 0c8adc5, so editing this checkout does not affect it.
  **If it stopped: rerun the same command through `scripts/detach.py`**; finished
  runs are skipped and an interrupted run resumes from its last checkpoint
  (every 100 steps).
- `keepawake_phase2`: holds off idle sleep until `phase2` exits. The user also set
  closing the lid not to sleep the laptop.

## Phase 1 results (tokenizer)

Own byte-level BPE (`forge/bpe.py`; with OpenAI's cl100k merge table it matches
tiktoken token for token), trained on all 2.72M TinyStoriesV2 training stories in
under three minutes. Bytes per token on validation: 3.735 (vocab 2,048), 4.049
(4,096), 4.179 (8,192) vs GPT-2 4.057 (50,257) and GPT-4 4.142 (100,277).
Training split: 542.9M tokens at vocab 4,096. `results/tokenizer/`.

## Phase 2 so far

- Throughput (bf16, 16x512 micro-batches, laptop thermals vary ±30%): S 5.8M
  ≈ 95-130k tok/s, M 15.7M ≈ 46-57k, L 33.6M ≈ 24k (≈6 h per pass over the data).
  This Windows PyTorch build has no FlashAttention; `enable_gqa` falls back to the
  math kernel (24.9 vs 3.2 ms per layer), so KV heads are copied instead.
- LR sweep on S (`results/lr_sweep/`, 24.9M tokens): 1e-3 2.3038, **2e-3 2.2495**,
  4e-3 2.8636, 8e-3 3.2675 nats/token on the full validation split. 4e-3 did not
  diverge; it plateaued early and never caught up.
- S (5.8M, 115.3M tokens): **1.5420 nats/token, 0.5523 bits/byte**; coherent
  samples. Ablations (60.3M tokens, seeds 17/29): baseline 1.7143/1.7304, bf16
  RoPE 1.7323/1.7362 (worse with both seeds), MHA 1.7114/1.7265 (within noise of
  GQA). `results/phase2/`.
- **The laptop hibernated from a critical thermal event at 16:35** during S and
  stayed off until 20:17 (System log, Kernel-Power event 88); the job resumed by
  itself after wake. At 22:15 the GPU ran at 87 °C with thermal slowdown (1,875 of
  3,105 MHz). HP Victus 15; the lid was being kept closed. Watch for repeats: count
  event 88 in the System log.

## Serving calibration (mixed, 128 requests, seed 17)

All policies meet the SLO (500 ms TTFT, 50 ms mean TPOT) for ≥99% of requests up
to 20 req/s; at 40 req/s SLO fraction falls to 35% continuous, 31% chunked, 11%
static. `results/cuda_calibration_14m`; full sweep in `results/cuda_sweep_mixed`.

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

1. When `phase2` finishes: scaling plot (validation loss vs parameters and vs
   compute for S/M/L), ablation table (mean and spread over seeds), read the S/M/L
   samples; write `docs/phase2/` summary.
2. Phase 3 code meanwhile, in this checkout (not `../forge-frozen`): SFT with loss
   on story tokens only, generic prompt/completion JSONL input.
3. Merge the Mac's `mac` branch (instruction data, verifiers, eval harness) before
   running phase 3. If the Mac stays offline, build `forge/instruct.py` here to the
   spec in `runs/MAC.md`.

## Notes

- Codex ran as a separate Windows account (`CodexSandboxOffline`), which owns the
  files. Git needed `safe.directory` entries (added globally for `forge` and `eqdata`).
- Commits are authored by the user alone: no co-author trailers or AI mentions.
- The project lives in OneDrive. Large checkpoint writes can collide with sync; if a
  save ever fails with a permission error, pause OneDrive during training.
- `results/wikitext103_14m/evaluation.json` records `heldout.seconds` from a
  throttled run (29.9 s); the losses are unaffected.
