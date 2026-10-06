"""Paged KV ownership, immutable full-prefix sharing, and copy-on-write.

Metadata is backend-independent. Serving reserves each admitted request's full
worst-case continuation, preventing decode deadlock at pool exhaustion. Prefix
reuse is an optional LRU of *completed full prompt blocks*; it consumes real pool
capacity and is evicted before rejecting an allocation.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


class CacheFull(RuntimeError):
    pass


@dataclass
class CacheHandle:
    request_id: str
    blocks: list[int] = field(default_factory=list)
    length: int = 0
    capacity: int = 0
    reused_tokens: int = 0


class BlockPool:
    def __init__(
        self, config, blocks=128, block_size=16, prefix_entries=0, backend="numpy", device="cpu"
    ):
        if blocks < 1 or block_size < 1 or prefix_entries < 0:
            raise ValueError("invalid cache dimensions")
        self.config, self.block_size = config, block_size
        self.refs = [0] * blocks
        self.free = list(range(blocks - 1, -1, -1))
        self.handles = {}
        self.prefixes = OrderedDict()
        self.prefix_entries = prefix_entries
        self.prefix_hits = 0
        self.cow_copies = 0
        self.peak_blocks = 0
        self.backend = backend
        # Every layer of one forward writes and reads the same slots. Indices are
        # built once and reused until block tables or lengths change.
        self._version = 0
        self._write_indices = self._read_indices = (None, None)
        shape = (config.layers, blocks, block_size, config.kv_heads, config.head_dim)
        if backend == "torch":
            import torch

            self.keys = torch.zeros(shape, device=device)
            self.values = torch.zeros(shape, device=device)
        else:
            self.keys = np.zeros(shape, dtype=np.float32)
            self.values = np.zeros(shape, dtype=np.float32)

    @property
    def allocated_bytes(self):
        return 2 * int(np.prod(self.keys.shape)) * 4

    @property
    def used_blocks(self):
        return len(self.refs) - len(self.free)

    def _decref(self, block):
        if self.refs[block] <= 0:
            raise AssertionError("double free")
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self.free.append(block)

    def _evict(self, needed):
        while len(self.free) < needed and self.prefixes:
            _, blocks = self.prefixes.popitem(last=False)
            for block in blocks:
                self._decref(block)

    def _allocate(self, count):
        self._evict(count)
        if len(self.free) < count:
            raise CacheFull(f"need {count} blocks; {len(self.free)} free")
        result = [self.free.pop() for _ in range(count)]
        for block in result:
            self.refs[block] = 1
        self.peak_blocks = max(self.peak_blocks, self.used_blocks)
        return result

    def create(self, request_id, capacity, prompt=None):
        if request_id in self.handles:
            raise ValueError("duplicate request id")
        if capacity < 1 or capacity > self.config.context:
            raise ValueError("invalid request capacity")
        shared, reused = [], 0
        # Always retain at least one prompt token to reconstruct final logits.
        if prompt and self.prefix_entries:
            max_blocks = min((len(prompt) - 1) // self.block_size, capacity // self.block_size)
            for count in range(max_blocks, 0, -1):
                key = tuple(prompt[: count * self.block_size])
                if key in self.prefixes:
                    shared = list(self.prefixes[key])
                    reused = len(shared) * self.block_size
                    self.prefixes.move_to_end(key)
                    break
        # Protect shared references before an allocation can evict their LRU entry.
        for block in shared:
            self.refs[block] += 1
        try:
            blocks = shared + self._allocate(math.ceil(capacity / self.block_size) - len(shared))
        except CacheFull:
            for block in shared:
                self._decref(block)
            raise
        handle = CacheHandle(request_id, blocks, reused, capacity, reused)
        if reused:
            self.prefix_hits += 1
        self.handles[request_id] = handle
        self._version += 1
        return handle

    def fork(self, source, request_id):
        if request_id in self.handles:
            raise ValueError("duplicate request id")
        handle = self.handles[source]
        for block in handle.blocks:
            self.refs[block] += 1
        result = CacheHandle(request_id, list(handle.blocks), handle.length, handle.capacity)
        self.handles[request_id] = result
        self._version += 1
        return result

    def prepare_write(self, request_id, positions):
        """Detach shared writable blocks before any layer writes K/V."""
        handle = self.handles[request_id]
        positions = list(positions)
        if positions and (min(positions) < 0 or max(positions) >= handle.capacity):
            raise ValueError("write outside reserved capacity")
        logical = sorted({position // self.block_size for position in positions})
        shared = [i for i in logical if self.refs[handle.blocks[i]] > 1]
        replacements = self._allocate(len(shared))  # Atomic on insufficient space.
        for i, replacement in zip(shared, replacements):
            original = handle.blocks[i]
            if self.backend == "torch":
                self.keys[:, replacement].copy_(self.keys[:, original])
                self.values[:, replacement].copy_(self.values[:, original])
            else:
                self.keys[:, replacement] = self.keys[:, original]
                self.values[:, replacement] = self.values[:, original]
            self._decref(original)
            handle.blocks[i] = replacement
            self.cow_copies += 1
        if positions:
            handle.length = max(handle.length, max(positions) + 1)
        self._version += 1

    def _to_device(self, array):
        if self.backend == "torch":
            import torch

            return torch.as_tensor(array, device=self.keys.device)
        return array

    def write(self, layer, requests, positions, keys, values):
        key = (self._version, tuple(requests), positions.shape, positions.tobytes())
        if self._write_indices[0] != key:
            rows, columns = np.nonzero(positions >= 0)
            pos = positions[rows, columns]
            tables = [self.handles[r].blocks for r in requests]
            table = np.zeros((len(requests), max(map(len, tables))), dtype=np.int64)
            for row, blocks in enumerate(tables):
                table[row, : len(blocks)] = blocks
            slots = np.stack(
                [rows, columns, table[rows, pos // self.block_size], pos % self.block_size]
            )
            self._write_indices = (key, self._to_device(slots))
        rows, columns, physical, offset = self._write_indices[1]
        self.keys[layer, physical, offset] = keys[rows, columns]
        self.values[layer, physical, offset] = values[rows, columns]

    def read(self, layer, requests):
        key = (self._version, tuple(requests))
        if self._read_indices[0] != key:
            lengths = [self.handles[r].length for r in requests]
            width = max(lengths)
            physical = np.zeros((len(requests), width), dtype=np.int64)
            offsets = np.broadcast_to(np.arange(width) % self.block_size, physical.shape).copy()
            for row, request_id in enumerate(requests):
                handle = self.handles[request_id]
                physical[row, : handle.length] = np.array(handle.blocks)[
                    np.arange(handle.length) // self.block_size
                ]
            slots = self._to_device(np.stack([physical, offsets]))
            self._read_indices = (key, (slots[0], slots[1], lengths))
        physical, offsets, lengths = self._read_indices[1]
        return self.keys[layer, physical, offsets], self.values[layer, physical, offsets], lengths

    def remember_prefix(self, request_id, prompt):
        if not self.prefix_entries:
            return
        handle = self.handles[request_id]
        count = min((len(prompt) - 1) // self.block_size, handle.length // self.block_size)
        if count < 1:
            return
        # Cache each full boundary so identical system prefixes are reusable
        # even when the requests have different user-message suffixes.
        for boundary in range(1, count + 1):
            key = tuple(prompt[: boundary * self.block_size])
            if key in self.prefixes:
                self.prefixes.move_to_end(key)
                continue
            blocks = tuple(handle.blocks[:boundary])
            for block in blocks:
                self.refs[block] += 1
            self.prefixes[key] = blocks
            while len(self.prefixes) > self.prefix_entries:
                _, old = self.prefixes.popitem(last=False)
                for block in old:
                    self._decref(block)

    def release(self, request_id):
        handle = self.handles.pop(request_id)
        for block in handle.blocks:
            self._decref(block)
        self._version += 1

    def clear_prefixes(self):
        for blocks in self.prefixes.values():
            for block in blocks:
                self._decref(block)
        self.prefixes.clear()

    def assert_integrity(self):
        expected = [0] * len(self.refs)
        for handle in self.handles.values():
            assert handle.length <= handle.capacity
            for block in handle.blocks:
                expected[block] += 1
        for blocks in self.prefixes.values():
            for block in blocks:
                expected[block] += 1
        assert expected == self.refs
        assert len(self.free) == len(set(self.free))
        assert set(self.free) == {i for i, ref in enumerate(self.refs) if ref == 0}

    def snapshot(self):
        reserved = sum(len(h.blocks) * self.block_size for h in self.handles.values())
        live = sum(h.length for h in self.handles.values())
        return {
            "pool_bytes": self.allocated_bytes,
            "used_blocks": self.used_blocks,
            "peak_blocks": self.peak_blocks,
            "prefix_hits": self.prefix_hits,
            "cow_copies": self.cow_copies,
            "reserved_token_slots": reserved,
            "live_tokens": live,
            "reserved_slack_slots": reserved - live,
        }


class ContiguousKV:
    """Single-request contiguous KV baseline with an exact reserved capacity."""

    def __init__(self, config, capacity, backend="numpy", device="cpu"):
        if not 0 < capacity <= config.context:
            raise ValueError("invalid contiguous capacity")
        self.capacity, self.length = capacity, 0
        shape = (config.layers, capacity, config.kv_heads, config.head_dim)
        self.backend = backend
        if backend == "torch":
            import torch

            self.keys = torch.zeros(shape, device=device)
            self.values = torch.zeros(shape, device=device)
        else:
            self.keys = np.zeros(shape, dtype=np.float32)
            self.values = np.zeros(shape, dtype=np.float32)

    def prepare_write(self, positions):
        if min(positions) < 0 or max(positions) >= self.capacity:
            raise ValueError("write exceeds contiguous cache")
        self.length = max(self.length, max(positions) + 1)

    def write(self, layer, requests, positions, keys, values):
        if len(requests) != 1:
            raise ValueError("contiguous reference supports a single request")
        pos = positions[0]
        if self.backend == "torch":
            import torch

            pos = torch.as_tensor(pos, device=self.keys.device)
        self.keys[layer, pos] = keys[0]
        self.values[layer, pos] = values[0]

    def read(self, layer, requests):
        return (
            self.keys[layer, : self.length][None],
            self.values[layer, : self.length][None],
            [self.length],
        )
