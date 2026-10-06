# Progress (laptop, `main`)

Last updated: 2026-10-06 14:25. The Mac Pro keeps its own log in
`runs/MAC_PROGRESS.md` on the `mac` branch.

## Running now

Check with `.\.venv\Scripts\python.exe scripts\detach.py --status`.

Nothing.

## In progress: serving-engine overhead (laptop)

Serving on CUDA is dominated by host-side overhead, not GPU work. Profile of the
trained 14M model, `mixed` trace, `continuous` policy, foreground process:
~26 ms per engine iteration (~80 output tok/s at 5 req/s offered). A bare model
forward for 8 decode tokens alone takes 11 ms (kernel-launch bound). Hot spots:

- `BlockPool.write`: per-request, per-layer Python loop with three host-to-device
  index copies and a boolean-mask sync each (~25% of time).
- `TorchModel.rope`: recomputes frequencies and cos/sin for q and k in every layer
  (~20%).
- `BlockPool.read`: rebuilds the same gather indices in every layer.

Plan: vectorize/memoize cache indices once per forward, compute RoPE tables once
per forward, keep outputs identical (NumPy oracle tests), and record before/after
with the same fixed trace. Then calibrate rates and run the 1,000-request sweeps.

## Done

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

1. Finish the engine-overhead work above, with before/after numbers.
2. Calibrate CUDA serving rates (128 requests, a few rates), then README roadmap
   item 1: 1,000-request sweeps across load, chunk budget, prefix reuse, and KV
   capacity.
3. Optional: a longer training run (loss was still falling).

## Notes

- Codex ran as a separate Windows account (`CodexSandboxOffline`), which owns the
  files. Git needed `safe.directory` entries (added globally for `forge` and `eqdata`).
- Commits are authored by the user alone: no co-author trailers or AI mentions.
- The project lives in OneDrive. Large checkpoint writes can collide with sync; if a
  save ever fails with a permission error, pause OneDrive during training.
- `results/wikitext103_14m/evaluation.json` records `heldout.seconds` from a
  throttled run (29.9 s); the losses are unaffected.
