# coding=utf-8
"""Ground-up MLX Hunyuan dense v1 decoder for HY-MT.

The Hunyuan-specific differences from the shared Qwen text stack are QK norm
after RoPE and the alpha-derived RoPE base. Greedy decode is gated by chrF
against the stock-transformers golden, not by byte parity.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..mlx_common.cache import KVCache, create_additive_causal_mask
from ..mlx_common.dtypes import resolve_dtype
from ..mlx_common.layers import RMSNorm, SwiGLUMLP
from ..mlx_common.weights import load_weights_dict, map_and_load, quantize_predicate
from .config import MLXHunyuanConfig


class HunyuanAttention(nn.Module):
    def __init__(self, cfg: MLXHunyuanConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.scale = self.head_dim**-0.5
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(
            cfg.hidden_size, self.n_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, cfg.hidden_size, bias=bias
        )
        self.use_qk_norm = cfg.use_qk_norm
        if self.use_qk_norm:
            # HunYuan names: query_layernorm / key_layernorm; RMSNorm per head over head_dim.
            self.query_layernorm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_base)

    def __call__(self, x: mx.array, cache: Optional[KVCache]):
        b, length, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(b, length, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(b, length, self.n_kv, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(b, length, self.n_kv, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        offset = cache.offset if cache is not None else 0
        # RoPE first, then QK-norm (HunYuanDenseV1Attention order).
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Pure prefill (offset==0) uses MLX's fused "causal" mask (bit-identical to the
        # additive mask, no per-layer L*L materialization); single-token decode needs none.
        if length == 1:
            mask = None
        elif offset == 0:
            mask = "causal"
        else:
            mask = create_additive_causal_mask(length, offset).astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(out)


class HunyuanDecoderLayer(nn.Module):
    def __init__(self, cfg: MLXHunyuanConfig):
        super().__init__()
        self.self_attn = HunyuanAttention(cfg)
        self.mlp = SwiGLUMLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, cache: Optional[KVCache]):
        x = x + self.self_attn(self.input_layernorm(x), cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class HunyuanModel(nn.Module):
    def __init__(self, cfg: MLXHunyuanConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [HunyuanDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, inputs_embeds: mx.array, caches: Optional[List[KVCache]]):
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, caches[i] if caches is not None else None)
        return self.norm(h)

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in range(len(self.layers))]


class MLXHunyuanForCausalLM(nn.Module):
    def __init__(self, config: MLXHunyuanConfig, tie_lm_head: bool = True):
        super().__init__()
        self.config = config
        self.tie_lm_head = tie_lm_head
        self.compute_dtype = mx.float32
        self.model = HunyuanModel(config)
        if not tie_lm_head:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _logits(self, hidden: mx.array) -> mx.array:
        if self.tie_lm_head:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        repetition_penalty: float = 1.0,
    ) -> List[int]:
        """Greedy decode for a single sequence (B=1). Returns generated ids (no eos).

        Applies a repetition penalty over the full sequence (prompt + generated),
        matching transformers' RepetitionPenaltyLogitsProcessor, so output tracks
        the HY-MT reference (do_sample=False, repetition_penalty=1.05).

        Pipelined greedy loop (mlx-lm style): step i+1's forward is enqueued
        on-device before token i is synced with ``.item()``, which overlaps GPU
        compute with the host read. The hidden gap is a fixed ~1.5 ms/token, so
        on HY-MT's large decode (measured ~1% on Hy-MT2-1.8B-4bit on M1) the
        win is small; the loop is kept pipelined for parity with the ASR loop
        since it is never slower (the old fresh-thread async_eval failure no
        longer reproduces on current MLX, and by decode time the worker thread
        has already run synchronous evals). Greedy output is identical to the
        synchronous loop -- same ops, same order.
        The repetition penalty is kept on-device as a vocab-sized ``seen_mask``
        updated by scatter from the on-device token id, matching transformers'
        RepetitionPenaltyLogitsProcessor over prompt + generated; the scatter for
        token i is enqueued before step i+1's logits read it, so the device-side
        dependency order is unchanged.
        """
        eos = set(int(x) for x in eos_token_ids)
        penalty = float(repetition_penalty)
        use_penalty = penalty != 1.0
        caches = self.model.make_cache()

        seen_mask = None
        if use_penalty:
            seen_mask = mx.zeros((self.config.vocab_size,), dtype=mx.float32)
            seen_mask[input_ids.reshape(-1)] = 1.0  # seed with prompt tokens

        def next_token(h_last: mx.array) -> mx.array:
            row = self._logits(h_last)[0]  # (vocab,)
            if use_penalty:
                penalized = mx.where(row > 0, row / penalty, row * penalty)
                row = mx.where(seen_mask > 0, penalized, row)
            return mx.argmax(row, axis=-1)  # scalar, kept on-device

        embeds = self.model.embed_tokens(input_ids)
        hidden = self.model(embeds, caches)

        y = next_token(hidden[:, -1, :])
        mx.async_eval(y)
        generated: List[int] = []
        for _ in range(int(max_new_tokens)):
            if use_penalty:
                seen_mask[y.reshape(1)] = (
                    1.0  # fold the candidate into the penalty context
                )
            hidden = self.model(self.model.embed_tokens(y.reshape(1, 1)), caches)
            y_next = next_token(hidden[:, -1, :])
            mx.async_eval(y_next)
            tok = int(y.item())
            if tok in eos:
                break
            generated.append(tok)
            y = y_next
        return generated


def load_mlx_hunyuan(
    model_dir: str, dtype: str = "bfloat16"
) -> Tuple[MLXHunyuanForCausalLM, MLXHunyuanConfig]:
    config = MLXHunyuanConfig.from_pretrained(model_dir)
    compute_dtype = resolve_dtype(dtype)

    weights = load_weights_dict(model_dir)
    has_lm_head = any(k.endswith("lm_head.weight") for k in weights)
    model = MLXHunyuanForCausalLM(config, tie_lm_head=not has_lm_head)
    model.compute_dtype = compute_dtype

    if config.quantization:
        nn.quantize(
            model,
            group_size=int(config.quantization["group_size"]),
            bits=int(config.quantization["bits"]),
            class_predicate=quantize_predicate(weights),
        )

    map_and_load(weights, model, compute_dtype)
    model.eval()
    mx.eval(model.parameters())
    return model, config
