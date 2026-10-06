# Intel Mac Pro: CPU benchmark and data worker

User-reported hardware: MacPro7,1, Xeon W-3245 @ 3.2 GHz, 16 physical cores / 32
threads, 96 GiB system RAM, AMD Radeon Pro W5500X with **8 GB discrete VRAM**.
The 96 GiB is not unified GPU memory and cannot be used as 96 GiB of VRAM. There is
no CUDA, and PyTorch stopped Intel-macOS releases after 2.2, so training stays on the
laptop's RTX 4050. The Mac produces the project's second hardware platform: CPU
serving measurements with the same trained weights and the same request traces.

The laptop is meanwhile running the CUDA serving sweeps. Do not duplicate them.

## Ground rules

- Work on the `mac` branch. Never push to `main`; the laptop merges `mac`.
- Keep `runs/MAC_PROGRESS.md` current (create it): what finished, what is running,
  the exact next command. Commit and push it with results after every milestone, so
  nothing is lost if the agent stops. Read `AGENTS.md` too; its rules apply here.
- Results go in `results/mac_*`, generated tables and plots in `docs/mac_*`.
  Do not edit shared source files except to fix a bug you can demonstrate; keep such
  fixes minimal, add a test, and describe them in `MAC_PROGRESS.md`.
- Anything longer than about two minutes runs detached, under `caffeinate` so the
  Mac cannot sleep, and **one benchmark at a time**. Parallel runs share cores and
  memory bandwidth and invalidate each other's timings:

  ```bash
  .venv/bin/python scripts/detach.py --name <job> -- caffeinate -i .venv/bin/python -m forge <args>
  .venv/bin/python scripts/detach.py --status
  ```

- Disconnect TeamViewer while timing runs; screen capture uses CPU. Record in
  `MAC_PROGRESS.md` anything else that was running.

## Milestone 0: code, environment, weights

```bash
gh auth login                              # GitHub.com, HTTPS, browser; account mri-thakur
gh repo clone mri-thakur/forge ~/forge && cd ~/forge
git switch -c mac && git push -u origin mac
python3 --version                          # needs 3.11 or newer (brew install python@3.12)
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots,serve]'
.venv/bin/python -c "import numpy; numpy.show_config()"   # record which BLAS NumPy uses
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest
```

PyTorch is not installed in this environment, so the torch test modules skip; that is
expected. Everything else must pass.

Trained weights (not in Git) are on the `wikitext103-14m` release:

```bash
mkdir -p checkpoints/wikitext103_14m
gh release download wikitext103-14m -D checkpoints/wikitext103_14m
shasum -a 256 checkpoints/wikitext103_14m/model.npz
# expected 6568f61a82e1c8b7eeb48d98f0e60222b4b4fe2a186a979dee78dc7052013b0e
.venv/bin/python -m forge generate --checkpoint checkpoints/wikitext103_14m/model.npz \
  --prompt "The album was released in" --tokens 60
```

`scripts/detach.py` has only been tested on Windows. Check its POSIX path once:
start a 30-second `sleep` job, confirm `--status` shows it running, close the agent's
terminal, and confirm it still finishes with exit code 0. Then check speed: run the
same short benchmark in the foreground and detached and compare ms per iteration
(`elapsed_including_drain_s / iterations` in `benchmark.json`). On Windows, detached
jobs ran 8x slower until the launcher opted them out of power throttling; if macOS
slows them down too, fix that before timing anything.

```bash
.venv/bin/python -m forge bench --checkpoint checkpoints/wikitext103_14m/model.npz --requests 16 --rates 2 --seeds 17 --policies continuous --out results/tmp/fg
.venv/bin/python scripts/detach.py --name speedcheck -- .venv/bin/python -m forge bench --checkpoint checkpoints/wikitext103_14m/model.npz --requests 16 --rates 2 --seeds 17 --policies continuous --out results/tmp/bg
```

The laptop is optimizing the serving engine on `main`. Before starting each
milestone, run `git fetch && git merge origin/main` so both machines measure the
same code (every result records its source hash), and rerun the tests.

## Milestone 1: useful CPU thread count

Run the same workload at 1, 4, 8, and 16 threads in **separate processes**, first
with a small random model, then with the trained checkpoint. More threads can hurt
small matrix workloads; 32 hardware threads is not a reason to use 32 BLAS workers.

```bash
for n in 1 4 8 16; do
  export OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n VECLIB_MAXIMUM_THREADS=$n
  .venv/bin/python -m forge ladder --dim 128 --layers 4 --repeats 5 --out results/mac_threads_random_${n}.json
  .venv/bin/python -m forge ladder --checkpoint checkpoints/wikitext103_14m/model.npz --repeats 5 --out results/mac_threads_14m_${n}.json
done
```

Pick the thread count with the best contiguous-KV time for the 14M model and use it
for everything after. Outputs record thread settings and source hash.

## Milestone 2: CPU serving sweeps with the trained weights

Same checkpoint, workloads, seeds, and scheduler settings (CLI defaults: max batch
8, token budget 64, 128 blocks of 16) as the laptop's CUDA sweeps, so the two
machines are directly comparable.

1. Calibrate: 128 requests, `--workload mixed`, seed 17, rates `1 2 5 10 20`. Find
   the rate where the SLO fraction collapses. The 14M model on CPU will saturate far
   below the GPU; pick the rates from the measurement, not from the laptop's.
2. Sweep: 1,000 requests at four rates spanning well below to just above that
   capacity, seeds `101 103 107`, all three policies, for `mixed` and then
   `head_of_line`. Estimate each sweep's duration from the calibration first (each
   run lasts at least requests ÷ rate seconds); if one would exceed about 8 hours,
   reduce the number of rates rather than the request count.
3. Report each sweep: `.venv/bin/python -m forge report --input results/<run>/benchmark.json --out docs/<run>`

`benchmark.json` is rewritten after every completed run, but a stopped sweep cannot
resume; rerun it into a fresh folder.

## Optional, only after milestone 2: legacy PyTorch and MPS probe

[PyTorch stopped Intel-macOS binary releases after 2.2](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).
[AMD GPUs on Intel Macs have had MPS support](https://discuss.pytorch.org/t/about-the-mps-category/151972/3),
but Forge has not tested this machine, macOS version, or driver stack. Use a
separate environment so the main one keeps current NumPy:

```bash
python3.11 -m venv .venv-mps-legacy
.venv-mps-legacy/bin/python -m pip install 'numpy==1.26.4' 'torch==2.2.2'
.venv-mps-legacy/bin/python -m pip install -e '.[dev]'
.venv-mps-legacy/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())"
.venv-mps-legacy/bin/python -m pytest tests/test_torch_optional.py
```

If MPS is available and the oracle test passes, run `forge ladder --backend torch
--device mps` and compare with the CPU results. Record the old software stack. Do not
build PyTorch from source.
