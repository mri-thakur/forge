# Decisions you can inspect and reproduce

## Why maintain an independent NumPy model?

It is executable ground truth for cached execution and the optional PyTorch
backend. Agreement between two cache paths in one implementation can preserve a
shared bug. Separate gradients and a finite-difference test provide another check.
Run `pytest tests/test_autograd.py tests/test_contiguous.py tests/test_engine.py`.

## What exactly makes decoding cheaper?

Full recomputation reruns all past tokens at every decode step. KV caching keeps
their projected keys/values. The current token still attends over the cached
history. Paging changes allocation and access, not that attention requirement.
Run `forge ladder` to compare recomputation and the two cached paths separately.

## Why can batching become slower?

The current portable implementation pads each iteration to the largest prefill
chunk in its selected batch. Padding, repeated KV gathers, scheduler work, and
Python overhead can dominate a tiny model. Inspect `padding_tokens` and
`executed_tokens` next to latency. Run the same trace with all three policies.
An explanation must match measured data; a feature name does not guarantee speed.

## What does the allocator prove?

Every live request and prefix entry contributes one reference to each block it
owns. A block is free exactly when its reference count is zero. Randomized tests
compare the allocator counters with counts recomputed from every owner after
each transition. A copy-on-write allocation failure leaves original handles valid.
This is tested, not formally verified for every possible execution.

## How is overload handled?

Admission reserves the maximum requested continuation. Large work waits FIFO if
the current pool cannot fit it. Impossible work is rejected immediately; queue
limits and timeouts make overload visible. This prevents decode deadlock, but
conservative reservation sacrifices utilization. The current engine has no
swap/recompute preemption. That tradeoff belongs in the experiment report.

## Why use goodput?

Output tokens/sec alone can grow while requests spend seconds in a queue. Goodput
counts requests meeting both first-token and mean per-token limits. Rejections
remain in the success-fraction denominator. Intended arrival timestamps are
preserved even when the benchmark submits a request late, so overloaded periods
are not silently omitted from latency.

## Which numbers are ready to defend?

Local CPU equivalence, gradient, ownership, and resume tests; synthetic training
diagnostics; and the explicitly scoped CPU measurements. GPU performance, large
language-model quality, production serving latency, and remote CI status are not
yet measured. The runbooks describe how to acquire those results.
