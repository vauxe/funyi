# coding=utf-8
"""Top-level MLX Qwen3-ASR model: audio merge, prefill, greedy decode."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..mlx_common.dtypes import resolve_dtype
from ..mlx_common.weights import load_weights_dict, map_and_load, quantize_predicate
from .audio_encoder import AudioEncoder
from .config import MLXQwen3ASRConfig
from .text_decoder import TextModel

__all__ = ["MLXQwen3ASRForConditionalGeneration", "load_mlx_qwen3_asr", "resolve_dtype"]


class MLXQwen3ASRForConditionalGeneration(nn.Module):
    def __init__(self, config: MLXQwen3ASRConfig, tie_lm_head: bool = False):
        super().__init__()
        self.config = config
        self.tie_lm_head = False if config.is_forced_aligner else bool(tie_lm_head)
        self.compute_dtype = mx.float32
        self.audio_tower = AudioEncoder(config.audio)
        self.model = TextModel(config.text)
        # Forced-aligner head: a classify_num-way timestamp classifier (always its own
        # weight, never tied). Otherwise the standard vocab LM head, which is tied when
        # the checkpoint stores no lm_head.weight (logits via embed_tokens.as_linear,
        # works for both nn.Embedding and the quantized QuantizedEmbedding).
        out_features = (
            config.classify_num if config.is_forced_aligner else config.text.vocab_size
        )
        if config.is_forced_aligner or not tie_lm_head:
            self.lm_head = nn.Linear(
                config.text.hidden_size, int(out_features), bias=False
            )

    def _logits(self, hidden: mx.array) -> mx.array:
        if self.tie_lm_head:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    # --- multimodal pieces -------------------------------------------------
    def get_audio_features(
        self, input_features: mx.array, feature_lengths: Sequence[int]
    ) -> mx.array:
        feats = []
        for i, flen in enumerate(feature_lengths):
            flen = int(flen)
            feats.append(self.audio_tower(input_features[i][:, :flen], flen))
        return mx.concatenate(feats, axis=0)

    def _merge_audio(
        self, input_ids: mx.array, inputs_embeds: mx.array, audio_features: mx.array
    ) -> mx.array:
        """Replace audio-placeholder rows (B=1) with audio_features (assumes one contiguous run)."""
        ids = np.asarray(input_ids[0]).reshape(-1)
        positions = np.flatnonzero(ids == self.config.audio_token_id)
        n_audio = int(audio_features.shape[0])
        assert positions.size == n_audio, (
            f"audio placeholder count {positions.size} != audio feature count {n_audio}"
        )
        if n_audio == 0:
            return inputs_embeds
        start = int(positions[0])
        contiguous = bool(np.all(positions == np.arange(start, start + n_audio)))
        assert contiguous, "non-contiguous audio placeholders not supported"
        audio = audio_features.astype(inputs_embeds.dtype)[None]  # (1, N, H)
        return mx.concatenate(
            [inputs_embeds[:, :start], audio, inputs_embeds[:, start + n_audio :]],
            axis=1,
        )

    # --- generation --------------------------------------------------------
    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        input_features: Optional[mx.array] = None,
        feature_lengths: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Greedy decode for a single sequence (B=1). Returns generated token ids (no eos)."""
        eos = set(int(x) for x in eos_token_ids)
        caches = self.model.make_cache()

        embeds = self.model.embed_tokens(input_ids)
        if input_features is not None and feature_lengths is not None:
            audio_features = self.get_audio_features(input_features, feature_lengths)
            embeds = self._merge_audio(input_ids, embeds, audio_features)

        hidden = self.model(embeds, caches)  # prefill; rope offsets come from the cache

        # Synchronous greedy loop: read each token back with .item() before stepping.
        # We deliberately avoid mx.async_eval -- the realtime service drives the backend
        # from a dedicated worker thread, and async_eval as the first op on a fresh thread
        # has no initialized GPU stream. The async pipeline was ~7% and decode is
        # compute-bound, so the synchronous loop is the robust choice (greedy output identical).
        generated: List[int] = []
        for _ in range(int(max_new_tokens)):
            tok = int(mx.argmax(self._logits(hidden[:, -1, :]), axis=-1).item())
            if tok in eos:
                break
            generated.append(tok)
            step_embeds = self.model.embed_tokens(
                mx.array([[tok]], dtype=input_ids.dtype)
            )
            hidden = self.model(step_embeds, caches)
        return generated

    def align_logits(
        self,
        input_ids: mx.array,
        input_features: mx.array,
        feature_lengths: Sequence[int],
    ) -> mx.array:
        """Forced-aligner forward: one causal prefill, full-sequence logits (1, L, classify_num).

        No decode loop and no KV cache -- alignment reads the logits at the
        <timestamp> positions of a single forward (mirrors the torch aligner's
        ``self.model.thinker(**inputs).logits``).
        """
        embeds = self.model.embed_tokens(input_ids)
        audio_features = self.get_audio_features(input_features, feature_lengths)
        embeds = self._merge_audio(input_ids, embeds, audio_features)
        hidden = self.model(embeds, None)
        return self._logits(hidden)


def load_mlx_qwen3_asr(
    model_dir: str, dtype: str = "float16"
) -> Tuple[MLXQwen3ASRForConditionalGeneration, MLXQwen3ASRConfig]:
    config = MLXQwen3ASRConfig.from_pretrained(model_dir)
    compute_dtype = resolve_dtype(dtype)

    weights = load_weights_dict(model_dir)
    has_lm_head = any(k.endswith("lm_head.weight") for k in weights)
    model = MLXQwen3ASRForConditionalGeneration(config, tie_lm_head=not has_lm_head)
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
