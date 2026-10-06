"""Iteration-level serving with FIFO admission and decode-first chunked prefill.

Policies are static batching, continuous batching with entire prefills, and
continuous batching with a token budget. All execute the same model/cache. A
request reserves its maximum continuation on admission; insufficient capacity
queues work rather than risking a mid-decode allocation deadlock.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from forge.cache import BlockPool, CacheFull


def sample(logits, rng, temperature=0, top_k=0, top_p=1.0):
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("invalid sampling settings")
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise FloatingPointError("nonfinite logits")
    if temperature == 0:
        return int(logits.argmax())
    scores = logits / temperature
    if top_k:
        threshold = np.partition(scores, -min(top_k, len(scores)))[-min(top_k, len(scores))]
        scores = np.where(scores >= threshold, scores, -np.inf)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    if top_p < 1:
        order = np.argsort(-probs)
        # Keep the first token crossing the threshold as well.
        keep = np.cumsum(probs[order]) - probs[order] < top_p
        probs[order[~keep]] = 0
        probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


@dataclass
class Request:
    id: str
    prompt: list[int]
    max_tokens: int
    arrival: float
    temperature: float = 0
    top_k: int = 0
    top_p: float = 1
    generated: list[int] = field(default_factory=list)
    emissions: list[float] = field(default_factory=list)
    prefill_cursor: int = 0
    admitted: float | None = None
    finished: float | None = None
    status: str = "queued"
    reason: str | None = None
    rng: object = None


class Engine:
    def __init__(
        self,
        model,
        policy="chunked",
        max_batch=8,
        token_budget=64,
        blocks=128,
        block_size=16,
        prefix_entries=0,
        seed=17,
        max_queue=1024,
        queue_timeout=60.0,
        record_metrics=False,
    ):
        if policy not in {"static", "continuous", "chunked"}:
            raise ValueError("unknown scheduling policy")
        if min(max_batch, token_budget, max_queue) < 1 or queue_timeout <= 0:
            raise ValueError("invalid scheduler dimensions")
        self.model, self.policy = model, policy
        self.max_batch, self.token_budget = max_batch, token_budget
        self.max_queue, self.queue_timeout, self.seed = max_queue, queue_timeout, seed
        self.pool = BlockPool(
            model.config, blocks, block_size, prefix_entries, model.backend, model.device
        )
        self.waiting, self.active, self.requests = deque(), [], {}
        self.iterations, self.executed_tokens, self.padding_tokens = 0, 0, 0
        self.queue_peak = 0
        self.cache_samples = [] if record_metrics else None
        self._prefill_rotation = 0

    @property
    def busy(self):
        return bool(self.active or self.waiting)

    def submit(self, request_id, prompt, max_tokens, arrival=None, temperature=0, top_k=0, top_p=1):
        prompt = list(prompt)
        if request_id in self.requests:
            raise ValueError("duplicate request id")
        if (
            not prompt
            or max_tokens < 1
            or any(t < 0 or t >= self.model.config.vocab_size for t in prompt)
        ):
            raise ValueError("invalid prompt or output length")
        if temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid sampling parameters")
        if len(prompt) + max_tokens > self.model.config.context:
            raise ValueError("prompt plus output exceeds context")
        now = time.perf_counter() if arrival is None else arrival
        digest = int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "little")
        request = Request(
            request_id,
            prompt,
            max_tokens,
            now,
            temperature,
            top_k,
            top_p,
            rng=np.random.default_rng(self.seed ^ digest),
        )
        self.requests[request_id] = request
        required = math.ceil((len(prompt) + max_tokens) / self.pool.block_size)
        if required > len(self.pool.refs) or len(self.waiting) >= self.max_queue:
            request.status, request.reason, request.finished = "rejected", "capacity", now
        else:
            self.waiting.append(request_id)
            self.queue_peak = max(self.queue_peak, len(self.waiting))
        return request

    def cancel(self, request_id, now=None):
        request = self.requests[request_id]
        if request.status not in {"queued", "active"}:
            return
        if request_id in self.active:
            self.active.remove(request_id)
            self.pool.release(request_id)
        else:
            self.waiting.remove(request_id)
        request.status = "cancelled"
        request.finished = time.perf_counter() if now is None else now

    def _admit(self, now):
        # Queue timeouts apply even while a static batch is draining.
        while self.waiting and now - self.requests[self.waiting[0]].arrival > self.queue_timeout:
            request = self.requests[self.waiting.popleft()]
            request.status, request.reason, request.finished = "rejected", "queue_timeout", now
        if self.policy == "static" and self.active:
            return
        while self.waiting and len(self.active) < self.max_batch:
            request = self.requests[self.waiting[0]]
            try:
                handle = self.pool.create(
                    request.id, len(request.prompt) + request.max_tokens, request.prompt
                )
            except CacheFull:
                break  # Strict FIFO avoids starving large requests.
            self.waiting.popleft()
            self.active.append(request.id)
            request.prefill_cursor = handle.reused_tokens
            request.admitted, request.status = now, "active"

    def _plan(self):
        planned = []
        prefills = []
        for request_id in self.active:
            request = self.requests[request_id]
            if request.prefill_cursor < len(request.prompt):
                prefills.append(request)
            else:
                position = len(request.prompt) + len(request.generated) - 1
                planned.append((request, [request.generated[-1]], [position], False))
        # Decode is capped by max_batch; remaining budget is shared fairly by
        # rotating the oldest prefill's starting position each iteration.
        budget = max(0, self.token_budget - len(planned))
        if prefills:
            start = self._prefill_rotation % len(prefills)
            prefills = prefills[start:] + prefills[:start]
            self._prefill_rotation += 1
        for request in prefills:
            remaining = len(request.prompt) - request.prefill_cursor
            count = remaining if self.policy != "chunked" else min(remaining, budget)
            if count < 1:
                continue
            cursor = request.prefill_cursor
            planned.append(
                (
                    request,
                    request.prompt[cursor : cursor + count],
                    list(range(cursor, cursor + count)),
                    True,
                )
            )
            budget -= count
        return planned

    def step(self, now=None):
        now = time.perf_counter() if now is None else now
        self._admit(now)
        plan = self._plan()
        if not plan:
            return []
        width = max(len(entry[1]) for entry in plan)
        tokens = np.zeros((len(plan), width), dtype=np.int64)
        positions = np.full(tokens.shape, -1, dtype=np.int64)
        for row, (request, chunk, pos, _) in enumerate(plan):
            tokens[row, : len(chunk)] = chunk
            positions[row, : len(chunk)] = pos
            self.pool.prepare_write(request.id, pos)
        logits = self.model.forward(tokens, positions, self.pool, [p[0].id for p in plan])
        if self.cache_samples is not None:
            snapshot = self.pool.snapshot()
            self.cache_samples.append(
                {k: snapshot[k] for k in ("reserved_token_slots", "live_tokens", "used_blocks")}
            )
        self.iterations += 1
        self.executed_tokens += sum(len(p[1]) for p in plan)
        self.padding_tokens += tokens.size - sum(len(p[1]) for p in plan)
        events = []
        for row, (request, chunk, _, prefill) in enumerate(plan):
            if prefill:
                request.prefill_cursor += len(chunk)
                if request.prefill_cursor < len(request.prompt):
                    continue
                self.pool.remember_prefix(request.id, request.prompt)
            token = sample(
                logits[row, len(chunk) - 1],
                request.rng,
                request.temperature,
                request.top_k,
                request.top_p,
            )
            timestamp = time.perf_counter()
            request.generated.append(token)
            request.emissions.append(timestamp)
            events.append((request.id, token))
            if len(request.generated) == request.max_tokens:
                request.finished, request.status = timestamp, "completed"
                self.active.remove(request.id)
                self.pool.release(request.id)
        return events


def reference_generate(model, prompt, max_tokens):
    tokens, output = list(prompt), []
    for _ in range(max_tokens):
        logits = model.forward(np.asarray([tokens], dtype=np.int64))
        token = int(logits[0, -1].argmax())
        tokens.append(token)
        output.append(token)
    return output


def contiguous_generate(model, prompt, max_tokens):
    from forge.cache import ContiguousKV

    cache = ContiguousKV(model.config, len(prompt) + max_tokens, model.backend, model.device)
    cache.prepare_write(range(len(prompt)))
    logits = model.forward(np.array([prompt]), np.arange(len(prompt))[None], cache, ["one"])
    output = [int(logits[0, -1].argmax())]
    for i in range(1, max_tokens):
        position = len(prompt) + i - 1
        cache.prepare_write([position])
        logits = model.forward(np.array([[output[-1]]]), np.array([[position]]), cache, ["one"])
        output.append(int(logits[0, -1].argmax()))
    return output
