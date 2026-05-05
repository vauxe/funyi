# coding=utf-8
from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..hf_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from ..decode_runtime import CudaGraphDecoder
from ..flashinfer_attention import register_flashinfer
from ..fused_linears import patch_model_fused_linears
from ..fused_rmsnorm import patch_model_rmsnorms
from ..quant_linears import patch_model_quantized_linears
from ..spec_decode import spec_decode_generate
from ..utils import chunk_list
from .base import ASRRuntimeBackend

AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)


class TransformersASRBackend(ASRRuntimeBackend):
    name = "transformers"

    def __init__(self, model: Any, processor: Any):
        self.model = model
        self.processor = processor

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        self.dtype = getattr(model, "dtype", torch.float32)
        self._cuda_graph_decoder: Optional[CudaGraphDecoder] = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "TransformersASRBackend":
        if "attn_implementation" not in kwargs:
            default_attn = cls._default_attn_implementation(kwargs.get("device_map"))
            if default_attn is not None:
                kwargs["attn_implementation"] = default_attn

        cuda_graph = bool(kwargs.pop("cuda_graph", False))
        cuda_graph_len_bucket = int(kwargs.pop("cuda_graph_len_bucket", 1))
        use_flashinfer = bool(kwargs.pop("flashinfer", False))
        use_fused_rmsnorm = bool(kwargs.pop("fused_rmsnorm", False))
        use_fused_linears = bool(kwargs.pop("fused_linears", False))
        use_quantized_linears = bool(kwargs.pop("quantized_linears", False))
        if "quantized_linear_components" in kwargs:
            raise RuntimeError("quantized_linear_components was removed; W8A16 now always uses qkv and gate_up.")
        if use_quantized_linears and not use_fused_linears:
            raise RuntimeError("quantized_linears=True requires fused_linears=True")

        if use_flashinfer:
            # Register the kernel under the key the model config will later pick up.
            if register_flashinfer("flashinfer"):
                # Override the requested attn_implementation so thinker's sub-configs
                # route through our flashinfer dispatcher. We still need to propagate
                # the key down because of the sub_configs bug noted below.
                kwargs["attn_implementation"] = "flashinfer"
            else:
                raise RuntimeError("flashinfer is not installed; install dependencies with `uv sync --python 3.12`")

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # NOTE: attn_implementation does NOT propagate from the top config down
        # to thinker / text_config / audio_config. The upstream wrapper has the
        # same behavior, so ASR runs actually execute with whatever the
        # sub-configs default to (typically sdpa) regardless of the top flag.
        # Keeping this "bug" deliberately so that runtime output matches the
        # local runtime-default golden. See notes in AGENTS.md before changing.
        if use_flashinfer:
            # For flashinfer we *do* want propagation (the whole point is to use
            # the custom attention kernel). Set it explicitly on all sub-configs.
            for cfg in (
                getattr(model.config, "thinker_config", None),
                getattr(getattr(model, "thinker", None), "config", None),
                getattr(getattr(getattr(model, "thinker", None), "config", None), "text_config", None),
                getattr(getattr(getattr(model, "thinker", None), "config", None), "audio_config", None),
            ):
                if cfg is not None:
                    cfg._attn_implementation = "flashinfer"
        if use_fused_rmsnorm:
            patched = patch_model_rmsnorms(model)
            if patched == 0:
                raise RuntimeError("fused_rmsnorm=True but no RMSNorm modules found")
        if use_fused_linears:
            summary = patch_model_fused_linears(model)
            if summary["attn_fused"] == 0 and summary["mlp_fused"] == 0:
                raise RuntimeError("fused_linears=True but no layers were fused")
        if use_quantized_linears:
            summary = patch_model_quantized_linears(model)
            if summary["qkv"] == 0 and summary["gate_up"] == 0:
                raise RuntimeError("quantized_linears=True but no linears were quantized")
        cls._prepare_inference_only_model(model)
        if use_fused_linears or use_quantized_linears:
            cls._release_patch_time_cuda_cache(model)
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            fix_mistral_regex=True,
        )
        backend = cls(model=model, processor=processor)
        if cuda_graph:
            backend.enable_cuda_graph(graph_len_bucket=cuda_graph_len_bucket)
        return backend

    def enable_cuda_graph(self, *, graph_len_bucket: int = 1) -> None:
        if self._cuda_graph_decoder is None:
            self._cuda_graph_decoder = CudaGraphDecoder(self.model.thinker, graph_len_bucket=graph_len_bucket)

    def reset_decode_runtime(self) -> None:
        if self._cuda_graph_decoder is not None:
            self._cuda_graph_decoder.reset_runtime()

    def eval(self) -> None:
        if hasattr(self.model, "eval"):
            self.model.eval()

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

    @torch.inference_mode()
    def infer_streaming_with_draft(
        self,
        prompt: str,
        wav: np.ndarray,
        draft_ids: Sequence[int],
        *,
        max_new_tokens: int,
        stats: dict[str, int] | None = None,
    ) -> str:
        """Speculative streaming decode. Returns decoded text after the prompt.

        ``draft_ids`` must be the tokenization of the text that would be added
        onto ``prompt`` as a rollback prefix. Encoder is run once for
        ``prompt + draft``; accepted draft tokens skip the per-token decode
        loop. When ``cuda_graph`` is also enabled the tail decode runs through
        the captured CUDA graph via
        :meth:`CudaGraphDecoder.generate_with_draft`. Not byte-identical under
        bf16; see ``spec_decode.py`` docstring.
        """
        inputs = self.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inputs = self._move_inputs(inputs)
        prompt_len = int(inputs["input_ids"].shape[1])
        draft = list(draft_ids)
        if self._cuda_graph_decoder is not None:
            sequences = self._cuda_graph_decoder.generate_with_draft(
                input_ids=inputs["input_ids"],
                input_features=inputs.get("input_features"),
                attention_mask=inputs["attention_mask"],
                feature_attention_mask=inputs.get("feature_attention_mask"),
                draft_ids=draft,
                max_new_tokens=int(max_new_tokens),
                stats=stats,
            )
        else:
            sequences = spec_decode_generate(
                self.model.thinker,
                input_ids=inputs["input_ids"],
                input_features=inputs.get("input_features"),
                attention_mask=inputs["attention_mask"],
                feature_attention_mask=inputs.get("feature_attention_mask"),
                draft_ids=draft,
                max_new_tokens=int(max_new_tokens),
                stats=stats,
            )
        decoded = self.processor.batch_decode(
            sequences[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    @torch.inference_mode()
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

        outs: List[str] = []
        batch_size = max_inference_batch_size if max_inference_batch_size >= 0 else len(prompts)

        for prompt_batch, wav_batch in zip(chunk_list(prompts, batch_size), chunk_list(wavs, batch_size)):
            inputs = self.processor(text=prompt_batch, audio=wav_batch, return_tensors="pt", padding=True)
            inputs = self._move_inputs(inputs)
            prompt_len = int(inputs["input_ids"].shape[1])
            if self._cuda_graph_decoder is not None and prompt_batch and len(prompt_batch) == 1:
                sequences = self._cuda_graph_decoder.generate(
                    input_ids=inputs["input_ids"],
                    input_features=inputs.get("input_features"),
                    attention_mask=inputs["attention_mask"],
                    feature_attention_mask=inputs.get("feature_attention_mask"),
                    max_new_tokens=max_new_tokens,
                )
            else:
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, logits_to_keep=1)
                sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            decoded = self.processor.batch_decode(
                sequences[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(list(decoded))
        return outs

    def _move_inputs(self, inputs: Any) -> Any:
        if not hasattr(inputs, "items"):
            if self.device is not None:
                inputs = inputs.to(self.device)
            if self.dtype is not None:
                inputs = inputs.to(self.dtype)
            return inputs

        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                continue
            kwargs: Dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device
            if self.dtype is not None and torch.is_floating_point(value):
                kwargs["dtype"] = self.dtype
            if kwargs:
                inputs[key] = value.to(**kwargs)
        return inputs

    @staticmethod
    def _default_attn_implementation(device_map: Any = None) -> Optional[str]:
        if find_spec("flash_attn") is None and find_spec("flash_attn_2_cuda") is None:
            return None
        if torch.cuda.is_available():
            return "flash_attention_2"
        if device_map is not None and "cuda" in str(device_map).lower():
            return "flash_attention_2"
        return None

    @staticmethod
    def _prepare_inference_only_model(model: Any) -> None:
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)

    @staticmethod
    def _release_patch_time_cuda_cache(model: Any) -> None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
