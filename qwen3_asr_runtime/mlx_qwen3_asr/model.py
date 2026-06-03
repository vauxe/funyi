# coding=utf-8
"""Top-level MLX Qwen3-ASR model: audio merge, prefill, greedy decode."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .audio_encoder import AudioEncoder
from .config import MLXQwen3ASRConfig
from .text_decoder import TextModel
from .weights import load_weights_dict, map_and_load, quantize_predicate

_DTYPES = {
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
    "float16": mx.float16,
    "fp16": mx.float16,
    "half": mx.float16,
    "float32": mx.float32,
    "fp32": mx.float32,
    "float": mx.float32,
}


def resolve_dtype(name: str) -> mx.Dtype:
    key = str(name).lower().replace("torch.", "").strip()
    if key not in _DTYPES:
        raise ValueError(f"unsupported dtype: {name}")
    return _DTYPES[key]


class MLXQwen3ASRForConditionalGeneration(nn.Module):
    def __init__(self, config: MLXQwen3ASRConfig, tie_lm_head: bool = False):
        super().__init__()
        self.config = config
        self.tie_lm_head = tie_lm_head
        self.compute_dtype = mx.float32
        self.audio_tower = AudioEncoder(config.audio)
        self.model = TextModel(config.text)
        # When the checkpoint ties the LM head (no stored lm_head.weight), logits are
        # computed from the input embedding via embed_tokens.as_linear (works for both
        # nn.Embedding and the quantized QuantizedEmbedding).
        if not tie_lm_head:
            self.lm_head = nn.Linear(config.text.hidden_size, config.text.vocab_size, bias=False)

    def _logits(self, hidden: mx.array) -> mx.array:
        if self.tie_lm_head:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    # --- multimodal pieces -------------------------------------------------
    def get_audio_features(self, input_features: mx.array, feature_lengths: Sequence[int]) -> mx.array:
        feats = []
        for i, flen in enumerate(feature_lengths):
            flen = int(flen)
            feats.append(self.audio_tower(input_features[i][:, :flen], flen))
        return mx.concatenate(feats, axis=0)

    def _merge_audio(self, input_ids: mx.array, inputs_embeds: mx.array, audio_features: mx.array) -> mx.array:
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
            [inputs_embeds[:, :start], audio, inputs_embeds[:, start + n_audio :]], axis=1
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
        y = mx.argmax(self._logits(hidden[:, -1, :]), axis=-1)  # (1,) kept on-device
        mx.async_eval(y)

        # Async-eval decode loop (mlx-lm style): feed the previous token as an mx.array so
        # step n+1's graph is built before step n is read back, overlapping GPU work with the
        # Python loop and hiding per-token sync latency. Greedy result is unchanged.
        generated: List[int] = []
        for _ in range(int(max_new_tokens)):
            step_embeds = self.model.embed_tokens(y.reshape(1, 1))
            hidden = self.model(step_embeds, caches)
            y_next = mx.argmax(self._logits(hidden[:, -1, :]), axis=-1)
            mx.async_eval(y_next)
            tok = int(y.item())
            if tok in eos:
                break
            generated.append(tok)
            y = y_next
        return generated


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
