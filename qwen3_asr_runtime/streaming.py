# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


_LEADING_TAIL_BOUNDARY_CHARS = " \t\r\n，,。.!！?？、；;：:"


def _clean_tail_text(text: str) -> str:
    return str(text or "").strip().lstrip(_LEADING_TAIL_BOUNDARY_CHARS).strip()


def _remove_text_prefix(text: str, prefix: str) -> str:
    full_text = str(text or "").strip()
    prefix_text = str(prefix or "").strip()
    if prefix_text and full_text.startswith(prefix_text):
        return full_text[len(prefix_text) :].strip()
    return full_text


def _normalized_text(text: str) -> str:
    normalized, _ = _normalized_with_end_offsets(text)
    return normalized


def _normalized_with_end_offsets(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(str(text or "")):
        if char.isspace() or char in _LEADING_TAIL_BOUNDARY_CHARS:
            continue
        chars.append(char.lower() if char.isascii() else char)
        offsets.append(index + 1)
    return "".join(chars), offsets


@dataclass(frozen=True)
class RecognitionTail:
    text: str
    aligned: bool


@dataclass(frozen=True)
class RecognitionFrame:
    """One finite-window ASR result with prompt text separated from evidence."""

    window_start_sample: int
    audio_end_sample: int
    full_text: str
    language: str
    decoded_text: str = ""
    generated_text: str = ""


class TailSelector:
    """Select the replaceable transcript tail from one recognition frame."""

    @classmethod
    def select(
        cls,
        frame: RecognitionFrame,
        *,
        stable_text_prefix: str,
        stable_end_sample: int,
        previous_partial_text: str = "",
    ) -> RecognitionTail:
        stable = _clean_tail_text(stable_text_prefix)
        previous_partial = _clean_tail_text(previous_partial_text)
        full = _clean_tail_text(frame.full_text)
        generated = _clean_tail_text(frame.generated_text)
        decoded = _clean_tail_text(frame.decoded_text)
        candidate_after_window = decoded or generated or full

        if not full and not candidate_after_window:
            return RecognitionTail(text="", aligned=True)

        if not stable:
            return RecognitionTail(text=full or candidate_after_window, aligned=True)

        if full.startswith(stable):
            return RecognitionTail(
                text=_clean_tail_text(full[len(stable) :]),
                aligned=True,
            )

        if decoded and decoded.startswith(stable):
            return RecognitionTail(
                text=_clean_tail_text(decoded[len(stable) :]),
                aligned=True,
            )

        if full and stable.startswith(full):
            return RecognitionTail(text="", aligned=True)

        if decoded and stable.startswith(decoded):
            return RecognitionTail(text="", aligned=True)

        window_start = int(frame.window_start_sample)
        window_overlaps_stable_cursor = window_start < int(stable_end_sample)
        partial_is_current_tail = bool(previous_partial) and (
            candidate_after_window.startswith(previous_partial)
            or previous_partial.startswith(candidate_after_window)
        )
        if partial_is_current_tail:
            return RecognitionTail(
                text=candidate_after_window,
                aligned=True,
            )
        if window_overlaps_stable_cursor:
            overlap = cls._stable_suffix_decoded_prefix_overlap(
                stable, candidate_after_window
            )
        else:
            overlap = 0
        if overlap > 0:
            return RecognitionTail(
                text=_clean_tail_text(candidate_after_window[overlap:]),
                aligned=True,
            )

        if window_start >= int(stable_end_sample):
            tail = candidate_after_window
            if cls._looks_like_stable_context_echo(stable, tail):
                return RecognitionTail(text=tail, aligned=False)
            return RecognitionTail(text=tail, aligned=True)

        return RecognitionTail(text=full or candidate_after_window, aligned=False)

    @staticmethod
    def _looks_like_stable_context_echo(stable: str, tail: str) -> bool:
        stable_norm = _normalized_text(stable)
        tail_norm = _normalized_text(tail)
        max_len = min(len(stable_norm), len(tail_norm))
        index = 0
        while index < max_len and stable_norm[index] == tail_norm[index]:
            index += 1
        return index >= 8

    @staticmethod
    def _stable_suffix_decoded_prefix_overlap(stable: str, decoded: str) -> int:
        stable_norm, _ = _normalized_with_end_offsets(stable)
        decoded_norm, decoded_offsets = _normalized_with_end_offsets(decoded)
        max_len = min(len(stable_norm), len(decoded_norm))
        for overlap in range(max_len, 3, -1):
            if stable_norm[-overlap:] == decoded_norm[:overlap]:
                return decoded_offsets[overlap - 1]
        return 0


@dataclass
class ASRStreamingState:
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_samples: int
    max_window_samples: Optional[int]
    max_prefix_tokens: Optional[int]

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray
    audio_seen_samples: int
    audio_trim_cursor: int

    prompt_raw: str
    force_language: Optional[str]

    language: str
    text: str
    carried_text_prefix: str
    partial_text: str
    _raw_decoded: str

    spec_decode: bool = False
    spec_decode_stats: dict[str, int] = field(default_factory=dict)
    recognition_frame: RecognitionFrame | None = None

    @property
    def audio_dropped_samples(self) -> int:
        return int(self.audio_trim_cursor)

    @audio_dropped_samples.setter
    def audio_dropped_samples(self, value: int) -> None:
        self.audio_trim_cursor = int(value)

    @property
    def committed_text(self) -> str:
        """Backward-compatible alias for carried model-prefix text.

        This is not user-visible committed transcript history.
        """
        return self.carried_text_prefix

    @committed_text.setter
    def committed_text(self, value: str) -> None:
        self.carried_text_prefix = str(value or "")


@dataclass
class StreamingPrefixPlan:
    prefix: str
    draft_ids: list[int]
    trimmed: bool = False


class RollingWindowTrimPolicy:
    """Keep a bounded model audio window and advance the audio trim cursor."""

    def apply(self, state: ASRStreamingState) -> None:
        if state.max_window_samples is None:
            return
        overflow = int(state.audio_accum.shape[0]) - int(state.max_window_samples)
        if overflow <= 0:
            return
        state.audio_trim_cursor += overflow
        state.audio_accum = state.audio_accum[overflow:].copy()


@dataclass
class StableTextUpdate:
    stable_text: str
    partial_text: str
    stable_end_sample: int | None


@dataclass
class TextStabilizer:
    previous_tail_text: str = ""
    previous_tail_end_sample: int | None = None

    def observe(
        self, tail_text: str, *, end_sample: int, can_commit: bool
    ) -> StableTextUpdate:
        tail = self.clean_tail_text(tail_text)
        if not can_commit:
            self.set_tail(tail, end_sample=end_sample)
            return StableTextUpdate(
                stable_text="", partial_text=tail, stable_end_sample=None
            )

        stable = self.repeated_tail_prefix(self.previous_tail_text, tail)
        stable_end_sample = self.previous_tail_end_sample if stable else None
        if stable and stable_end_sample is not None:
            partial = self.remove_text_prefix(tail, stable)
            self.set_tail(partial, end_sample=end_sample)
            return StableTextUpdate(
                stable_text=stable,
                partial_text=partial,
                stable_end_sample=stable_end_sample,
            )

        self.set_tail(tail, end_sample=end_sample)
        return StableTextUpdate(
            stable_text="", partial_text=tail, stable_end_sample=None
        )

    def finalize(self, tail_text: str, *, end_sample: int) -> StableTextUpdate:
        stable = self.clean_tail_text(tail_text)
        self.set_tail("", end_sample=None)
        return StableTextUpdate(
            stable_text=stable,
            partial_text="",
            stable_end_sample=end_sample if stable else None,
        )

    def set_tail(self, text: str, *, end_sample: int | None) -> None:
        tail = self.clean_tail_text(text)
        self.previous_tail_text = tail
        self.previous_tail_end_sample = (
            int(end_sample) if tail and end_sample is not None else None
        )

    @staticmethod
    def clean_tail_text(text: str) -> str:
        return _clean_tail_text(text)

    @staticmethod
    def is_tail_update(previous: str, current: str) -> bool:
        previous_text = TextStabilizer.clean_tail_text(previous)
        current_text = TextStabilizer.clean_tail_text(current)
        if not previous_text or not current_text:
            return False
        return current_text.startswith(previous_text) or previous_text.startswith(
            current_text
        )

    @staticmethod
    def repeated_tail_prefix(previous: str, current: str) -> str:
        previous_text = str(previous or "").strip()
        current_text = str(current or "").strip()
        max_len = min(len(previous_text), len(current_text))
        index = 0
        while index < max_len and previous_text[index] == current_text[index]:
            index += 1
        next_char = current_text[index : index + 1]
        return TextStabilizer.trim_stable_prefix_to_boundary(
            current_text[:index], next_char
        )

    @staticmethod
    def trim_stable_prefix_to_boundary(prefix: str, next_char: str) -> str:
        raw_prefix = str(prefix or "")
        next_text = str(next_char or "")
        right_stripped = raw_prefix.rstrip()
        stable_prefix = right_stripped.strip()
        if not stable_prefix or not next_text:
            return stable_prefix
        if len(right_stripped) < len(raw_prefix):
            return stable_prefix
        last_char = stable_prefix[-1]
        if not (
            last_char.isascii()
            and next_text[0].isascii()
            and last_char.isalnum()
            and next_text[0].isalnum()
        ):
            return stable_prefix
        for index in range(len(stable_prefix) - 1, -1, -1):
            if not stable_prefix[index].isalnum():
                return stable_prefix[: index + 1].strip()
        return ""

    @staticmethod
    def remove_text_prefix(text: str, prefix: str) -> str:
        return _remove_text_prefix(text, prefix)


__all__ = [
    "ASRStreamingState",
    "RecognitionFrame",
    "RecognitionTail",
    "RollingWindowTrimPolicy",
    "StableTextUpdate",
    "StreamingPrefixPlan",
    "TailSelector",
    "TextStabilizer",
]
