# coding=utf-8
"""MLX (Apple Silicon) backend for the Qwen3 forced aligner.

Only the model forward is MLX-specific. Windowing and timestamp text
post-processing are shared with the torch aligner through ``ForcedAlignerCommon``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from transformers import AutoConfig, AutoProcessor

from .forced_aligner import (
    FORCED_ALIGNER_SUPPORTED_LANGUAGES,
    ForcedAlignResult,
    ForcedAlignerCommon,
    Qwen3ForceAlignTextProcessor,
)
from .hf_qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor
from .mlx_common.hub import resolve_model_dir
from .utils import ensure_list

# CUDA/optimization kwargs the torch aligner understands but MLX does not.
_DROPPED_KWARGS = {
    "device_map",
    "attn_implementation",
    "fused_rmsnorm",
    "fused_linears",
    "quantized_linears",
}


class MLXForcedAlignerBackend(ForcedAlignerCommon):
    def __init__(
        self,
        model: Any,
        processor: Any,
        config: Any,
        aligner_processor: Optional[Qwen3ForceAlignTextProcessor] = None,
    ) -> None:
        import mlx.core as mx
        from .mlx_common.dtypes import resolve_dtype

        self._mx = mx
        self.model = model
        self.processor = processor
        self.config = config
        self.aligner_processor = aligner_processor or Qwen3ForceAlignTextProcessor()
        self.device = "mlx"
        if config.timestamp_token_id is None or config.timestamp_segment_time is None:
            raise ValueError(
                "MLXForcedAlignerBackend requires a forced-aligner checkpoint "
                "(config.timestamp_token_id / timestamp_segment_time / classify_num)."
            )
        self.timestamp_token_id = int(config.timestamp_token_id)
        self.timestamp_segment_time = float(config.timestamp_segment_time)
        self._compute = getattr(model, "compute_dtype", None) or resolve_dtype(
            "bfloat16"
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> "MLXForcedAlignerBackend":
        from .mlx_qwen3_asr import load_mlx_qwen3_asr

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)

        dtype_name = str(kwargs.pop("dtype", None) or "bfloat16")
        local_files_only = bool(kwargs.pop("local_files_only", True))
        for key in list(kwargs):
            if key in _DROPPED_KWARGS:
                kwargs.pop(key)

        model_dir = resolve_model_dir(
            pretrained_model_name_or_path, local_files_only=local_files_only
        )
        model, config = load_mlx_qwen3_asr(
            model_dir, dtype=dtype_name
        )  # sets model.compute_dtype
        processor = AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
        return cls(model=model, processor=processor, config=config)

    def get_supported_languages(self) -> Optional[List[str]]:
        return sorted({str(x).lower() for x in FORCED_ALIGNER_SUPPORTED_LANGUAGES})

    def _align_normalized(
        self,
        audios: List[np.ndarray],
        *,
        text: Union[str, List[str]],
        language: Union[str, List[str]],
    ) -> List[ForcedAlignResult]:
        mx = self._mx
        texts = ensure_list(text)
        languages = ensure_list(language)
        if len(languages) == 1 and len(audios) > 1:
            languages = languages * len(audios)
        if not (len(audios) == len(texts) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: audio={len(audios)}, text={len(texts)}, language={len(languages)}"
            )

        results: List[ForcedAlignResult] = []
        for wav, t, lang in zip(audios, texts, languages):
            word_list, aligner_input_text = self.aligner_processor.encode_timestamp(
                t, lang
            )
            wav = np.asarray(wav, dtype=np.float32)
            inputs = self.processor(
                text=[aligner_input_text],
                audio=[wav],
                return_tensors="np",
                padding=True,
            )
            input_ids_np = np.asarray(inputs["input_ids"]).astype(np.int64)
            feats = mx.array(
                np.asarray(inputs["input_features"], dtype=np.float32)
            ).astype(self._compute)
            flen = [
                int(np.asarray(inputs["feature_attention_mask"]).sum(-1).reshape(-1)[0])
            ]
            input_ids = mx.array(input_ids_np.astype(np.int32))

            ts_positions = np.flatnonzero(input_ids_np[0] == self.timestamp_token_id)
            if ts_positions.size == 0:
                results.append(ForcedAlignResult(items=[]))
                continue
            # The classify head runs only on the <timestamp> rows: (1, P, classify_num).
            logits = self.model.align_logits(
                input_ids, feats, flen, positions=ts_positions.tolist()
            )
            masked_output_id = np.asarray(mx.argmax(logits[0], axis=-1))
            timestamp_ms = (
                masked_output_id.astype(np.float64) * self.timestamp_segment_time
            )

            timestamp_output = self.aligner_processor.parse_timestamp(
                word_list, timestamp_ms
            )
            for it in timestamp_output:
                it["start_time"] = round(it["start_time"] / 1000.0, 3)
                it["end_time"] = round(it["end_time"] / 1000.0, 3)
            results.append(self._to_structured_items(timestamp_output))
        return results


__all__ = ["MLXForcedAlignerBackend"]
