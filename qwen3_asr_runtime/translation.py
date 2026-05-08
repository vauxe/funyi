# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Iterable, Optional

import torch

DEFAULT_HYMT_MODEL = "tencent/HY-MT1.5-1.8B"
DEFAULT_HYMT_MAX_NEW_TOKENS = 512
DEFAULT_HYMT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_HYMT_DECODE_BACKEND = "fixed_mask"
_HYMT_PROMPT_TEMPLATE = (
    "Translate the following segment into {target_language}, keeping the original format, "
    "without additional explanation.\n\n{source_text}"
)


@dataclass(frozen=True)
class HYMTGenerationConfig:
    max_new_tokens: int = DEFAULT_HYMT_MAX_NEW_TOKENS
    top_k: int = 20
    top_p: float = 0.6
    repetition_penalty: float = 1.05
    temperature: float = 0.7
    do_sample: bool = True
    extra_generate_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"logits_to_keep": 1, "cache_implementation": "static"}
    )


@dataclass(frozen=True)
class HYMTTranslationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    encode_wall_sec: float
    generate_wall_sec: float
    decode_wall_sec: float
    total_wall_sec: float


class HYMTTranslator:
    """Small synchronous transformers adapter for Tencent Hunyuan HY-MT.

    This class intentionally depends only on `transformers`, not on the HY-MT
    repository runtime. It is meant to be loaded once at service startup.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_HYMT_MODEL,
        *,
        device: str = "cuda:0",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str | None = DEFAULT_HYMT_ATTN_IMPLEMENTATION,
        decode_backend: str = DEFAULT_HYMT_DECODE_BACKEND,
        generation_config: HYMTGenerationConfig | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device or "cuda:0")
        self.dtype = _resolve_dtype(dtype, device=self.device)
        self.attn_implementation = _resolve_attn_implementation(attn_implementation)
        self.decode_backend = _resolve_decode_backend(decode_backend)
        self.generation_config = generation_config or HYMTGenerationConfig()

        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be provided together")
        if model is None or tokenizer is None:
            tokenizer, model = self._load_model(
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                attn_implementation=self.attn_implementation,
            )

        self.tokenizer = tokenizer
        self.model = model.eval()
        _disable_noop_hymt_dynamic_rope(self.model)
        self.input_device = _infer_input_device(self.model)

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        return self.profile_translate(
            text,
            target_language=target_language,
            source_language=source_language,
            max_new_tokens=max_new_tokens,
        ).text

    def warmup(
        self,
        texts: Iterable[str],
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
        sync_cuda: bool = False,
    ) -> list[HYMTTranslationResult]:
        results: list[HYMTTranslationResult] = []
        for text in texts:
            if not str(text or "").strip():
                continue
            results.append(
                self.profile_translate(
                    str(text),
                    target_language=target_language,
                    source_language=source_language,
                    max_new_tokens=max_new_tokens,
                    sync_cuda=sync_cuda,
                )
            )
        return results

    def profile_translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
        sync_cuda: bool = False,
    ) -> HYMTTranslationResult:
        total_started = time.perf_counter()
        source_text = str(text or "").strip()
        if not source_text:
            return HYMTTranslationResult(
                text="",
                prompt_tokens=0,
                generated_tokens=0,
                encode_wall_sec=0.0,
                generate_wall_sec=0.0,
                decode_wall_sec=0.0,
                total_wall_sec=0.0,
            )
        target = str(target_language or "").strip()
        if not target:
            raise ValueError("target_language must not be empty")

        prompt = build_hymt_prompt(
            source_text,
            target_language=target,
            source_language=source_language,
        )
        encode_started = time.perf_counter()
        input_ids = self._encode_prompt(prompt)
        self._sync_if_needed(sync_cuda)
        encode_wall_sec = time.perf_counter() - encode_started

        generate_started = time.perf_counter()
        generated_ids = self._generate(input_ids, max_new_tokens=max_new_tokens)
        self._sync_if_needed(sync_cuda)
        generate_wall_sec = time.perf_counter() - generate_started

        decode_started = time.perf_counter()
        generated_list = generated_ids.tolist()
        output = self.tokenizer.decode(generated_list, skip_special_tokens=True).strip()
        decode_wall_sec = time.perf_counter() - decode_started
        return HYMTTranslationResult(
            text=output,
            prompt_tokens=int(input_ids.shape[-1]),
            generated_tokens=len(generated_list),
            encode_wall_sec=encode_wall_sec,
            generate_wall_sec=generate_wall_sec,
            decode_wall_sec=decode_wall_sec,
            total_wall_sec=time.perf_counter() - total_started,
        )

    def _load_model(
        self,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        attn_implementation: str | None,
    ) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = _resolve_model_path(self.model_path, local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        elif self.device.startswith("cuda"):
            kwargs["device_map"] = {"": self.device}

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if self.device != "auto" and not self.device.startswith("cuda"):
            model = model.to(torch.device(self.device))
        return tokenizer, model

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        return self._tokenize_prompt(prompt).to(self.input_device)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids

    @torch.inference_mode()
    def _generate(self, input_ids: torch.Tensor, *, max_new_tokens: int | None) -> torch.Tensor:
        generate_kwargs = self._build_generate_kwargs(max_new_tokens=max_new_tokens)
        if self.decode_backend == "fixed_mask" and self._can_use_fixed_mask_decode(generate_kwargs):
            generate_kwargs["custom_generate"] = _hymt_fixed_mask_generate
        outputs = self.model.generate(
            input_ids=input_ids,
            **generate_kwargs,
        )
        return outputs[0, input_ids.shape[-1] :].detach().cpu()

    def _can_use_fixed_mask_decode(self, generate_kwargs: dict[str, Any]) -> bool:
        return (
            self.input_device.type == "cuda"
            and bool(generate_kwargs.get("use_cache", True))
            and generate_kwargs.get("cache_implementation") == "static"
        )

    def _build_generate_kwargs(self, *, max_new_tokens: int | None) -> dict[str, Any]:
        config = self.generation_config
        kwargs: dict[str, Any] = {
            "do_sample": bool(config.do_sample),
            "max_new_tokens": int(max_new_tokens if max_new_tokens is not None else config.max_new_tokens),
            "repetition_penalty": float(config.repetition_penalty),
            "use_cache": True,
        }
        if config.do_sample:
            kwargs["top_k"] = int(config.top_k)
            kwargs["top_p"] = float(config.top_p)
            kwargs["temperature"] = float(config.temperature)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id
        kwargs.update(config.extra_generate_kwargs)
        return kwargs

    def _sync_if_needed(self, sync_cuda: bool) -> None:
        if sync_cuda and self.input_device.type == "cuda":
            torch.cuda.synchronize(self.input_device)


def build_hymt_prompt(text: str, *, target_language: str, source_language: str = "") -> str:
    source_text = str(text or "").strip()
    target = str(target_language or "").strip()
    if not target:
        raise ValueError("target_language must not be empty")
    return _HYMT_PROMPT_TEMPLATE.format(target_language=target, source_text=source_text)


def _resolve_model_path(model_path: str, *, local_files_only: bool) -> str:
    path = str(model_path)
    if not local_files_only or Path(path).exists():
        return path

    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=path, local_files_only=True))


def _resolve_dtype(dtype: str | torch.dtype | None, *, device: str) -> Optional[torch.dtype]:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        uses_cuda = str(device).startswith("cuda") or (str(device) == "auto" and torch.cuda.is_available())
        return torch.bfloat16 if uses_cuda else None
    name = str(dtype).strip().lower()
    if name in {"", "auto"}:
        return None
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return table[name]


def _resolve_attn_implementation(attn_implementation: str | None) -> str | None:
    if attn_implementation is None:
        return None
    value = str(attn_implementation).strip()
    if value.lower() in {"", "none", "auto", "default"}:
        return None
    return value


def _resolve_decode_backend(decode_backend: str | None) -> str:
    value = str(decode_backend or DEFAULT_HYMT_DECODE_BACKEND).strip().lower().replace("-", "_")
    if value in {"", "default", "fixed", "fixed_mask"}:
        return "fixed_mask"
    if value in {"generate", "hf", "hf_generate"}:
        return "generate"
    raise ValueError(f"Unsupported HY-MT decode_backend: {decode_backend}")


def _disable_noop_hymt_dynamic_rope(model: Any) -> bool:
    """Skip HY-MT's dynamic RoPE update when it is provably a no-op.

    HY-MT ships `max_position_embeddings=262144`. Realtime subtitle segments
    stay far below that, so the dynamic update branch only performs a tensor
    max plus host-side guards on every decode forward without changing
    `inv_freq`.
    """
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "hunyuan_v1_dense":
        return False
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None or getattr(rotary, "rope_type", None) != "dynamic":
        return False

    max_cached = _as_int_or_none(getattr(rotary, "max_seq_len_cached", None))
    original_max = _as_int_or_none(getattr(rotary, "original_max_seq_len", None))
    max_positions = _as_int_or_none(getattr(getattr(rotary, "config", None), "max_position_embeddings", None))
    if max_cached is None or original_max is None or max_positions is None:
        return False
    if max_cached != original_max or max_cached != max_positions:
        return False

    rotary._hymt_original_rope_type = rotary.rope_type
    rotary.rope_type = "default"
    return True


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hymt_fixed_mask_generate(
    model: Any,
    input_ids: torch.LongTensor,
    logits_processor: Any,
    stopping_criteria: Any,
    generation_config: Any,
    synced_gpus: bool = False,
    streamer: Any | None = None,
    **model_kwargs: Any,
) -> torch.LongTensor:
    """Sampling loop for static-cache, single-user HY-MT subtitle translation.

    It keeps transformers' generation setup, logits processors, stopping
    criteria, and sampling ops, but bypasses per-token input/attention-mask
    concatenation by using fixed-shape token and mask buffers.
    """
    del streamer
    batch_size, prompt_len = input_ids.shape[:2]
    max_length = int(generation_config.max_length)
    sequences = torch.empty((batch_size, max_length), dtype=input_ids.dtype, device=input_ids.device)
    sequences[:, :prompt_len] = input_ids

    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=input_ids.device)
    source_attention_mask = model_kwargs.get("attention_mask")
    if source_attention_mask is None:
        attention_mask[:, :prompt_len] = True
    else:
        mask_len = int(source_attention_mask.shape[1])
        attention_mask[:, :mask_len] = source_attention_mask.to(device=input_ids.device, dtype=torch.bool)
    past_key_values = model_kwargs.get("past_key_values")
    cache_max_length = _cache_max_length(past_key_values) or max_length
    static_attention_masks = _build_static_sdpa_attention_masks(
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        cache_max_length=cache_max_length,
        prompt_len=prompt_len,
        source_attention_mask=source_attention_mask,
        device=input_ids.device,
    )

    cache_positions = torch.arange(max_length, device=input_ids.device, dtype=torch.long)
    use_cache = bool(model_kwargs.get("use_cache", True))
    logits_to_keep = model_kwargs.get("logits_to_keep", 1)
    pad_token_id = generation_config._pad_token_tensor
    do_sample = bool(generation_config.do_sample)
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

    compile_kwargs = dict(model_kwargs)
    compile_kwargs["cache_position"] = cache_positions[prompt_len - 1 : prompt_len]
    model_forward = model.__call__
    if model._valid_auto_compile_criteria(compile_kwargs, generation_config):
        model_forward = model.get_compiled_call(generation_config.compile_config)

    cur_len = int(prompt_len)
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    is_prefill = True
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        current_input_ids = sequences[:, :cur_len]
        if is_prefill:
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=_attention_mask_for_step(
                    static_attention_masks, attention_mask, cur_len, prefill=True
                ),
                past_key_values=past_key_values,
                cache_position=cache_positions[:cur_len],
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            is_prefill = False
        else:
            outputs = model_forward(
                input_ids=sequences[:, cur_len - 1 : cur_len],
                attention_mask=_attention_mask_for_step(
                    static_attention_masks, attention_mask, cur_len, prefill=False
                ),
                past_key_values=past_key_values,
                cache_position=cache_positions[cur_len - 1 : cur_len],
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                return_dict=True,
            )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(current_input_ids, next_token_logits)
        if do_sample:
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        if cur_len >= max_length:
            break
        sequences[:, cur_len] = next_tokens
        if static_attention_masks is None:
            attention_mask[:, cur_len] = True
        cur_len += 1
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(sequences[:, :cur_len], None)
        this_peer_finished = unfinished_sequences.max() == 0
        del outputs

    return sequences[:, :cur_len]


def _build_static_sdpa_attention_masks(
    *,
    model: Any,
    batch_size: int,
    max_length: int,
    cache_max_length: int,
    prompt_len: int,
    source_attention_mask: Any | None,
    device: torch.device,
) -> torch.Tensor | None:
    if batch_size != 1 or getattr(getattr(model, "config", None), "_attn_implementation", None) != "sdpa":
        return None
    if source_attention_mask is not None:
        if getattr(source_attention_mask, "ndim", None) != 2 or int(source_attention_mask.shape[1]) != prompt_len:
            return None
        prompt_mask = source_attention_mask.to(device=device, dtype=torch.bool)
        if not bool(prompt_mask.all().item()):
            return None

    query_positions = torch.arange(max_length, device=device)
    key_positions = torch.arange(cache_max_length, device=device)
    return key_positions.view(1, 1, 1, cache_max_length) <= query_positions.view(1, 1, max_length, 1)


def _attention_mask_for_step(
    static_attention_masks: torch.Tensor | None,
    attention_mask: torch.Tensor,
    cur_len: int,
    *,
    prefill: bool,
) -> torch.Tensor:
    if static_attention_masks is None:
        return attention_mask
    if prefill:
        return static_attention_masks[:, :, :cur_len, :]
    return static_attention_masks[:, :, cur_len - 1 : cur_len, :]


def _cache_max_length(past_key_values: Any | None) -> int | None:
    if past_key_values is None or not hasattr(past_key_values, "get_max_cache_shape"):
        return None
    try:
        value = past_key_values.get_max_cache_shape()
    except TypeError:
        return None
    if value is None:
        return None
    return int(value)


def _infer_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    for parameter in model.parameters():
        if not getattr(parameter, "is_meta", False):
            return parameter.device
    return torch.device("cpu")


__all__ = [
    "DEFAULT_HYMT_ATTN_IMPLEMENTATION",
    "DEFAULT_HYMT_DECODE_BACKEND",
    "DEFAULT_HYMT_MODEL",
    "DEFAULT_HYMT_MAX_NEW_TOKENS",
    "HYMTGenerationConfig",
    "HYMTTranslationResult",
    "HYMTTranslator",
    "build_hymt_prompt",
]
