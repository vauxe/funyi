# coding=utf-8
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import numpy as np


class ASRRuntimeBackend(ABC):
    """
    Runtime interface used by the runtime ASR wrapper.

    The wrapper owns upstream-compatible transcription and streaming logic.
    Backends only provide prompt formatting, tokenizer helpers, and batch
    audio-to-text generation.
    """

    name: str
    model: Any
    processor: Any
    device: Any
    dtype: Any

    @abstractmethod
    def eval(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode_text(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    def reset_decode_runtime(self) -> None:
        return None

    def infer_streaming_with_draft(
        self,
        prompt: str,
        wav: "np.ndarray",
        draft_ids: Sequence[int],
        *,
        max_new_tokens: int,
        stats: dict[str, int] | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def infer_with_prompts(
        self,
        prompts: List[str],
        wavs: List[np.ndarray],
        *,
        max_inference_batch_size: int,
        max_new_tokens: int,
    ) -> List[str]:
        raise NotImplementedError
