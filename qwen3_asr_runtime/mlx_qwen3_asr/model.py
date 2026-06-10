# coding=utf-8
"""Top-level MLX Qwen3-ASR model: audio merge, prefill, greedy decode."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..mlx_common.cache import KVCache
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
        y = mx.argmax(self._logits(hidden[:, -1, :]), axis=-1)
        return self._greedy_loop(y, caches, eos, int(max_new_tokens))

    def _greedy_loop(
        self,
        y: mx.array,
        caches: List[KVCache],
        eos: set,
        max_new_tokens: int,
    ) -> List[int]:
        """Pipelined greedy loop: ``y`` (shape (1,)) is the first candidate token.

        Step i+1's forward is enqueued on-device before token i is synced with
        ``.item()``, so the GPU works through the next step while the host reads
        the current token (mlx-lm style). The hidden host-sync gap is a fixed
        ~1.5 ms/token, so the relative gain depends on per-token GPU time:
        measured on M1 0.6B, ~11% with 4-bit weights (~11 ms/tok) and ~2% with
        bf16 (~29 ms/tok). Greedy output is identical to a synchronous loop --
        same ops in the same order. When EOS lands, the one already-enqueued
        extra step is discarded; caches are per-call so the stale extra cache
        entry is never observed.
        """
        mx.async_eval(y)
        generated: List[int] = []
        for _ in range(max_new_tokens):
            hidden = self.model(self.model.embed_tokens(y.reshape(1, 1)), caches)
            y_next = mx.argmax(self._logits(hidden[:, -1, :]), axis=-1)
            mx.async_eval(y_next)
            tok = int(y.item())
            if tok in eos:
                break
            generated.append(tok)
            y = y_next
        return generated

    def generate_with_draft(
        self,
        input_ids: mx.array,
        draft_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        input_features: Optional[mx.array] = None,
        feature_lengths: Optional[Sequence[int]] = None,
        stats: Optional[Dict[str, int]] = None,
    ) -> List[int]:
        """Greedy decode with speculative verification of ``draft_ids`` (B=1).

        Mirrors ``spec_decode.spec_decode_generate``: one prefill covers
        ``prompt + draft``, the argmax at each draft position is compared with
        the draft token, the accepted prefix costs no decode steps, and the
        verifier's argmax at the first mismatch (or past a fully-accepted
        draft) is a free next token. The KV cache is then cropped to the
        accepted prefix and plain greedy decode continues. Same caveat as the
        CUDA path: prefill-path KV differs from decode-path KV by half-precision
        epsilon, so output is not byte-identical to a plain re-decode; gate
        presets with the streaming CER sweep.
        """
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return []
        draft = [int(x) for x in draft_ids][:max_new_tokens]
        if not draft:
            raise ValueError("generate_with_draft requires a non-empty draft.")
        eos = set(int(x) for x in eos_token_ids)
        k = len(draft)

        prompt_len = int(input_ids.shape[1])
        ext_ids = mx.concatenate(
            [input_ids, mx.array([draft], dtype=input_ids.dtype)], axis=1
        )
        caches = self.model.make_cache()
        embeds = self.model.embed_tokens(ext_ids)
        if input_features is not None and feature_lengths is not None:
            audio_features = self.get_audio_features(input_features, feature_lengths)
            embeds = self._merge_audio(ext_ids, embeds, audio_features)
        hidden = self.model(embeds, caches)

        # k+1 next-token predictions at positions prompt_len-1 .. prompt_len+k-1:
        # candidates for (draft[0], ..., draft[k-1], token-after-draft).
        preds_dev = mx.argmax(self._logits(hidden[:, prompt_len - 1 :, :]), axis=-1)
        preds = np.asarray(preds_dev[0]).astype(np.int64).tolist()

        accepted = 0
        for j in range(k):
            if preds[j] != draft[j]:
                break
            accepted += 1
        if stats is not None:
            stats["draft_tokens"] = k
            stats["accepted_tokens"] = accepted

        generated = draft[:accepted]
        if len(generated) >= max_new_tokens:
            return generated
        if preds[accepted] in eos:
            # Matches the CUDA path's pre-crop return and skips the wasted
            # forward _greedy_loop would otherwise enqueue for the EOS token.
            return generated

        # Rewind to just after the accepted prefix; the bonus/correction token
        # preds[accepted] has not been fed through the model yet -- it becomes
        # the greedy loop's first candidate.
        for cache in caches:
            cache.crop(prompt_len + accepted)
        bonus = mx.array([preds[accepted]], dtype=input_ids.dtype)
        generated.extend(
            self._greedy_loop(bonus, caches, eos, max_new_tokens - len(generated))
        )
        return generated

    def align_logits(
        self,
        input_ids: mx.array,
        input_features: mx.array,
        feature_lengths: Sequence[int],
        positions: Optional[Sequence[int]] = None,
    ) -> mx.array:
        """Forced-aligner forward: one causal prefill, no decode loop, no KV cache
        (mirrors the torch aligner's ``self.model.thinker(**inputs).logits``).

        Alignment only reads the logits at the <timestamp> positions, so when
        ``positions`` is given the classify head runs on just those rows and the
        result is (1, P, classify_num); otherwise full-sequence (1, L, classify_num).
        """
        embeds = self.model.embed_tokens(input_ids)
        audio_features = self.get_audio_features(input_features, feature_lengths)
        embeds = self._merge_audio(input_ids, embeds, audio_features)
        hidden = self.model(embeds, None)
        if positions is not None:
            hidden = hidden[:, mx.array(list(positions), dtype=mx.int32), :]
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
