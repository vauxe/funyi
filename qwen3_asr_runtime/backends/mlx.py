# coding=utf-8
"""MLX (Apple Silicon) backend for the Qwen3-ASR runtime.

Reuses the CPU-only HF processor/tokenizer for text and mel-feature work, and
runs the model forward on MLX (Metal). It implements only the required
ASRRuntimeBackend methods; CUDA-only options (cuda_graph, flashinfer, fused
kernels, W8A16, speculative decode) do not apply and are silently dropped so
existing call sites that pass them keep working. Streaming works through the
standard infer_with_prompts prefix re-feed -- no token-level prefix API needed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from transformers import AutoConfig, AutoProcessor

from ..hf_qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor
from .base import ASRRuntimeBackend

# CUDA/optimization kwargs that the transformers backend understands but MLX does not.
_DROPPED_KWARGS = {
    "cuda_graph",
    "cuda_graph_len_bucket",
    "flashinfer",
    "fused_rmsnorm",
    "fused_linears",
    "quantized_linears",
    "quantized_linear_components",
    "attn_implementation",
    "device_map",
}


def _dtype_name(dtype: Any) -> str:
    if dtype is None:
        return "bfloat16"
    if isinstance(dtype, str):
        return dtype
    return str(dtype)  # e.g. torch.bfloat16 -> "torch.bfloat16"; resolve_dtype strips the prefix


class MLXASRBackend(ASRRuntimeBackend):
    name = "mlx"

    def __init__(self, model: Any, processor: Any, config: Any, dtype_name: str):
        import mlx.core as mx  # local import: MLX is only present on Apple Silicon
        from ..mlx_common.dtypes import resolve_dtype

        self._mx = mx
        self.model = model
        self.processor = processor
        self.config = config
        self.device = "mlx"
        self.dtype = dtype_name
        self._compute = resolve_dtype(dtype_name)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "MLXASRBackend":
        from ..mlx_qwen3_asr import load_mlx_qwen3_asr

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)

        from ..mlx_common.hub import resolve_model_dir

        dtype_name = _dtype_name(kwargs.pop("dtype", None))
        local_files_only = bool(kwargs.pop("local_files_only", False))
        for key in list(kwargs):
            if key in _DROPPED_KWARGS:
                kwargs.pop(key)
        # Any remaining kwargs are tolerated but unused here.

        # Resolve an HF id to its local snapshot dir (the MLX loader needs config.json +
        # safetensors on disk); a local path is returned unchanged.
        model_dir = resolve_model_dir(pretrained_model_name_or_path, local_files_only=local_files_only)
        model, config = load_mlx_qwen3_asr(model_dir, dtype=dtype_name)
        processor = AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
        return cls(model=model, processor=processor, config=config, dtype_name=dtype_name)

    def eval(self) -> None:
        return None

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        return self.processor.apply_chat_template(
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )

    def encode_text(self, text: str) -> List[int]:
        return list(self.processor.tokenizer.encode(text))

    def decode_text(self, token_ids: Sequence[int]) -> str:
        return self.processor.tokenizer.decode(list(token_ids))

    def infer_with_prompts(
        self,
        prompts: List[str],
        wavs: List[np.ndarray],
        *,
        max_inference_batch_size: int,
        max_new_tokens: int,
    ) -> List[str]:
        if not prompts:
            return []
        mx = self._mx
        outs: List[str] = []
        for prompt, wav in zip(prompts, wavs):
            wav = np.asarray(wav, dtype=np.float32)
            inputs = self.processor(text=[prompt], audio=[wav], return_tensors="np", padding=True)
            input_ids = mx.array(np.asarray(inputs["input_ids"]).astype(np.int32))
            feats = mx.array(np.asarray(inputs["input_features"], dtype=np.float32)).astype(self._compute)
            flen = [int(np.asarray(inputs["feature_attention_mask"]).sum(-1).reshape(-1)[0])]
            gen_ids = self.model.generate(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                eos_token_ids=self.config.eos_token_ids,
                input_features=feats,
                feature_lengths=flen,
            )
            outs.append(
                self.processor.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return outs
