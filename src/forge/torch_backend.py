"""Optional PyTorch backend. No generate(), attention modules, or serving libraries.

Float32 manual attention mirrors the NumPy oracle. CUDA/MPS timing includes the
logit transfer to CPU for sampling; this is an explicit overhead, not hidden.
"""

import numpy as np
import torch

from forge.model import NumpyModel


class TorchModel:
    backend = "torch"

    def __init__(self, reference: NumpyModel, device="cpu", attention="manual", rope_dtype=None):
        if attention not in {"manual", "sdpa"}:
            raise ValueError("unknown attention implementation")
        self.attention = attention
        # torch.bfloat16 reproduces the original bf16-angle RoPE, for the ablation.
        self.rope_dtype = rope_dtype or torch.float32
        self.config, self.device = reference.config, device
        self.weight_origin = reference.weight_origin
        self.weights = {
            k: torch.as_tensor(v.copy(), device=device) for k, v in reference.weights.items()
        }
        self.parameter_count = reference.parameter_count

    @staticmethod
    def norm(x, weight):
        return x * (x.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt() * weight

    def rope_tables(self, positions):
        # float32 even under bf16 autocast: bf16 angles are off by up to 0.5 rad
        # near position 256. Shared by q and k in every layer of one forward.
        d = self.config.head_dim
        freq = 10000 ** (-torch.arange(0, d, 2, device=self.device, dtype=self.rope_dtype) / d)
        angle = positions[..., None, None] * freq
        return angle.cos(), angle.sin()

    @staticmethod
    def rope(x, cos, sin):
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward_tensor(self, tokens, positions=None, pool=None, requests=None):
        tokens = torch.as_tensor(tokens, device=self.device, dtype=torch.long)
        b, t = tokens.shape
        c, w = self.config, self.weights
        raw_positions = (
            np.broadcast_to(np.arange(t), (b, t)) if positions is None else np.asarray(positions)
        )
        if raw_positions.max() >= c.context:
            raise ValueError("positions exceed context")
        pos = torch.tensor(raw_positions.copy(), device=self.device)
        cos, sin = self.rope_tables(pos)
        # A full sequence without a cache is plain causal attention, which SDPA can
        # run without materializing a mask (and so pick its fastest kernel).
        causal_sdpa = self.attention == "sdpa" and pool is None and positions is None
        mask = None
        x = w["embed"][tokens]
        for layer in range(c.layers):
            p = f"layers.{layer}."
            normalized = self.norm(x, w[p + "attn_norm"])
            q = self.rope(
                (normalized @ w[p + "q"]).reshape(b, t, c.heads, c.head_dim), cos, sin
            ).permute(0, 2, 1, 3)
            k = self.rope((normalized @ w[p + "k"]).reshape(b, t, c.kv_heads, c.head_dim), cos, sin)
            v = (normalized @ w[p + "v"]).reshape(b, t, c.kv_heads, c.head_dim)
            if pool is not None:
                pool.write(layer, requests, raw_positions, k, v)
                k, v, lengths = pool.read(layer, requests)
            # Cached lengths and positions are the same in every layer of one forward.
            if mask is None and pool is not None:
                indices = torch.arange(k.shape[1], device=self.device)[None, None, None, :]
                mask = (indices <= pos[:, None, :, None]) & (
                    indices < torch.tensor(lengths, device=self.device)[:, None, None, None]
                )
            elif mask is None and not causal_sdpa:
                mask = (
                    torch.arange(t, device=self.device)[None, None, None, :]
                    <= pos[:, None, :, None]
                )
            repeats = c.heads // c.kv_heads
            if self.attention == "sdpa":
                # Copy shared KV heads rather than pass enable_gqa: without
                # FlashAttention (absent from Windows builds) enable_gqa falls back to
                # the math kernel, measured 8x slower than this on the RTX 4050.
                k = k.repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
                v = v.repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
                attended = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=causal_sdpa, dropout_p=0.0
                )
            else:
                k = k.repeat_interleave(repeats, dim=2).permute(0, 2, 3, 1)
                v = v.repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
                scores = (q @ k) * c.head_dim**-0.5
                attention = scores.masked_fill(~mask, -1e30).softmax(dim=-1)
                attended = attention @ v
            output = attended.permute(0, 2, 1, 3).reshape(b, t, c.dim)
            x = x + output @ w[p + "o"]
            normalized = self.norm(x, w[p + "ffn_norm"])
            gate = normalized @ w[p + "gate"]
            x = x + (torch.nn.functional.silu(gate) * (normalized @ w[p + "up"])) @ w[p + "down"]
        logits = self.norm(x, w["norm"]) @ w["embed"].T
        return logits

    @torch.inference_mode()
    def forward(self, tokens, positions=None, pool=None, requests=None):
        return self.forward_tensor(tokens, positions, pool, requests).cpu().numpy()
