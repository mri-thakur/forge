# Experiment contract

The research question is: **at a given latency target and fixed KV capacity, which
scheduling policy admits the most useful work on constrained hardware?**

## Metrics

- TTFT: first emitted token minus the intended arrival time, including queueing.
- TPOT: each request's mean interval after its first token. The SLO uses mean TPOT;
  p99 individual inter-token latency is also reported and is a different metric.
- Throughput: completed output tokens / time from first arrival through drain.
- Goodput: requests meeting both TTFT and TPOT limits / that same elapsed time.
- SLO fraction: successful SLO requests / **all** submissions, including rejection.
- Cache bytes: exact allocated K/V tensor storage. This is not whole-process RSS.
- CUDA peak allocation: reported by PyTorch when using CUDA; excludes driver memory.
- Reservation utilization: live cached tokens / slots reserved for active requests.
  Worst-case continuation reservation and block rounding are both included in slack.

Arrivals are generated before a run. The runner submits all due requests after
every model iteration, retaining their original arrival timestamps. This models an
open-loop client and charges late admission to latency instead of hiding it. It
does not emulate network serialization, HTTP parsing, or SSE buffering.

## Controls

Within each seed/rate, every policy receives the exact same hashed arrival trace,
prompts, output lengths, model weights, pool size, maximum batch size, and sampling
seed. The model has no dropout. Prefix caching starts cold for each policy. Warmup
uses a separate engine. Policy order rotates across seeds. Raw traces and
per-request metrics accompany aggregate results. Source and weight SHA-256 hashes
identify the implementation even before the repository has its first commit.

The full-prefill policies are deliberately unconstrained by the chunked policy's
per-iteration token budget. This is the scheduling treatment, not an accidental
hardware/config difference. Decodes are prioritized by the chunked policy; prefill
selection rotates so one prompt cannot monopolize every available prefill slice.

## Experiments to publish

1. Compare full recomputation, contiguous KV, and paged KV on one request. Attribute
   cache speedups to avoiding recomputation. Report the paged/contiguous overhead.
2. Sweep arrival rates for short, mixed, burst, and long-prompt-interference traces.
3. Repeat prefix-heavy traces with prefix entries 0 and 16. Report hit counts,
   saved prefill execution, allocated bytes, and latency rather than only speedup.
4. Reduce KV blocks until admission queues build. Count timeouts/rejections and
   preserve them in goodput. Requests that can never fit are rejected immediately.
5. Sweep chunk budgets 16/32/64/128 with **separate calibration and evaluation
   seeds**. Publish the complete sweep, including settings that lose to static.
6. Verify CPU/NumPy versus CPU/PyTorch and CUDA before using CUDA results. On the
   Intel Mac, verify the legacy MPS stack independently; do not mix its older
   framework measurements with CUDA while attributing all differences to hardware.

The checked-in CPU runs are small diagnostic experiments on tiny models. They
establish that the measurement machinery works; they are not evidence of
production LLM capacity. A p99 from 128 requests is descriptive. The portfolio
release should use at least 1,000 requests per setting, several repetitions,
sustained load, and report the spread and limits of those measurements.

## What is deliberately measured rather than promised

Paging can lose to contiguous storage for one request. Continuous batching can
lose when padding and Python overhead dominate a tiny model. Chunking can improve
decode responsiveness while worsening a long prompt's TTFT. Byte tokenization
increases sequence length compared with BPE and makes tokenizer comparisons
inappropriate. A negative result is recorded, not removed from the table.

## Training evidence

Training logs specify loss in **nats per byte** and bits per byte; these are not
comparable to BPE perplexities. The synthetic JSON corpus checks implementation and
checkpointing only. It is explicitly labeled in its manifest. For language-model
quality, prepare a licensed corpus with `--text`, retain source/license metadata,
and evaluate a held-out split, unigram baseline, and fixed samples. The current
chronological split does not deduplicate near-identical documents; a larger corpus
requires that additional preprocessing before a generalization claim.

## Foundations

- [PagedAttention paper](https://arxiv.org/abs/2309.06180): block ownership and KV
  management. Forge uses a portable gather path, not vLLM or its fused kernels.
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level
  scheduling. The current scheduler is a small explicit implementation with FIFO
  admission; it does not claim to reproduce Orca's performance.
- [RoFormer](https://arxiv.org/abs/2104.09864) and
  [GQA](https://arxiv.org/abs/2305.13245): positional rotation and shared KV heads.
