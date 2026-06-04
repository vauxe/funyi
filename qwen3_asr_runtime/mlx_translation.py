# coding=utf-8
"""MLX (Apple Silicon) adapter for Tencent Hunyuan HY-MT translation.

The public methods mirror ``translation.HYMTTranslator`` so the realtime
translation actor can drive torch and MLX translators the same way. CUDA-only
kwargs are accepted and ignored on this path.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, List

import numpy as np

from transformers import AutoTokenizer

from .mlx_common.hub import resolve_model_dir, snapshot_commit
from .translation import (
    HYMTGenerationConfig,
    HYMTTranslationResult,
    build_hymt_prompt,
)

DEFAULT_MLX_HYMT_MODEL = "mlx-community/Hy-MT2-1.8B-4bit"


class MLXHYMTTranslator:
    """MLX HY-MT translator with the same call surface as ``HYMTTranslator``."""

    def __init__(
        self,
        model_path: str = DEFAULT_MLX_HYMT_MODEL,
        *,
        dtype: str | None = "bfloat16",
        local_files_only: bool = True,
        model_revision: str | None = None,
        generation_config: HYMTGenerationConfig | None = None,
        **_ignored: Any,
    ) -> None:
        from .mlx_hunyuan import load_mlx_hunyuan

        self.model_path = str(model_path)
        self.dtype = "bfloat16" if dtype in (None, "auto") else str(dtype)
        self.generation_config = generation_config or HYMTGenerationConfig()
        if self.generation_config.do_sample:
            raise NotImplementedError(
                "MLXHYMTTranslator supports greedy decoding only (do_sample=False); "
                "sampling is not implemented on the MLX backend."
            )

        resolved = resolve_model_dir(self.model_path, local_files_only=local_files_only, revision=model_revision)
        self.tokenizer = _load_tokenizer(resolved, local_files_only=local_files_only)
        self.model, self.config = load_mlx_hunyuan(resolved, dtype=self.dtype)
        self.resolved_model_commit = snapshot_commit(resolved)

    # --- public API ---------------------------------------------------------
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

    def translate_batch(
        self,
        texts: Iterable[str],
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
    ) -> List[str]:
        return [
            self.translate(
                text,
                target_language=target_language,
                source_language=source_language,
                max_new_tokens=max_new_tokens,
            )
            for text in texts
        ]

    def warmup(
        self,
        texts: Iterable[str],
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
        sync_cuda: bool = False,  # accepted, ignored (no CUDA on MLX)
        batch_size: int = 1,  # accepted, ignored (MLX runs batch-1)
    ) -> List[HYMTTranslationResult]:
        return [
            self.profile_translate(
                text,
                target_language=target_language,
                source_language=source_language,
                max_new_tokens=max_new_tokens,
            )
            for text in texts
            if str(text or "").strip()
        ]

    def profile_translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str = "",
        max_new_tokens: int | None = None,
        sync_cuda: bool = False,  # accepted, ignored
    ) -> HYMTTranslationResult:
        import mlx.core as mx

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

        prompt = build_hymt_prompt(source_text, target_language=target, source_language=source_language)

        encode_started = time.perf_counter()
        input_np = self._encode_prompt(prompt)
        input_ids = mx.array(input_np)
        encode_wall_sec = time.perf_counter() - encode_started

        generate_started = time.perf_counter()
        generated = self.model.generate(
            input_ids,
            max_new_tokens=int(
                max_new_tokens if max_new_tokens is not None else self.generation_config.max_new_tokens
            ),
            eos_token_ids=self.config.eos_token_ids,
            repetition_penalty=float(self.generation_config.repetition_penalty),
        )
        generate_wall_sec = time.perf_counter() - generate_started

        decode_started = time.perf_counter()
        output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        decode_wall_sec = time.perf_counter() - decode_started
        return HYMTTranslationResult(
            text=output,
            prompt_tokens=int(input_np.shape[-1]),
            generated_tokens=len(generated),
            encode_wall_sec=encode_wall_sec,
            generate_wall_sec=generate_wall_sec,
            decode_wall_sec=decode_wall_sec,
            total_wall_sec=time.perf_counter() - total_started,
        )

    # --- helpers ------------------------------------------------------------
    def _encode_prompt(self, prompt: str) -> np.ndarray:
        messages = [{"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="np",
        )
        arr = np.asarray(ids, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr


def _load_tokenizer(resolved: str, *, local_files_only: bool) -> Any:
    """Load the HY-MT tokenizer, tolerating mlx-community's re-serialized config.

    mlx-community checkpoints write ``tokenizer_class: TokenizersBackend`` (not a
    registered transformers class) but still ship a ``tokenizer.json`` and a
    ``chat_template.jinja``. Fall back to loading the fast tokenizer directly from
    those, wiring the special tokens and chat template by hand.
    """
    try:
        return AutoTokenizer.from_pretrained(
            resolved, local_files_only=local_files_only, fix_mistral_regex=True
        )
    except (ValueError, KeyError, OSError):
        import json
        from pathlib import Path

        from transformers import PreTrainedTokenizerFast

        d = Path(resolved)
        cfg_path = d / "tokenizer_config.json"
        tc = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        special = {k: tc[k] for k in ("bos_token", "eos_token", "pad_token", "unk_token") if tc.get(k)}
        tok = PreTrainedTokenizerFast(tokenizer_file=str(d / "tokenizer.json"), **special)
        jinja = d / "chat_template.jinja"
        if jinja.exists():
            tok.chat_template = jinja.read_text(encoding="utf-8")
        return tok


__all__ = ["DEFAULT_MLX_HYMT_MODEL", "MLXHYMTTranslator"]
