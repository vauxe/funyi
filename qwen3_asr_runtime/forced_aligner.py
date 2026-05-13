# coding=utf-8
from __future__ import annotations

import os
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import nagisa
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from .hf_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from .utils import (
    AudioLike,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    SAMPLE_RATE,
    ensure_list,
    normalize_audios,
    normalize_language_name,
)


FORCED_ALIGNER_SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "Cantonese",
    "English",
    "German",
    "Spanish",
    "French",
    "Italian",
    "Portuguese",
    "Russian",
    "Korean",
    "Japanese",
]


@dataclass(frozen=True)
class ForcedAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ForcedAlignSentence:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ForcedAlignTextSegment:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ForcedAlignResult:
    items: List[ForcedAlignItem]

    def __iter__(self) -> Iterable[ForcedAlignItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]


@dataclass(frozen=True)
class _TranscriptWindow:
    audio: np.ndarray
    offset: float
    text: str
    entries: List[tuple[int, ForcedAlignTextSegment, int]]


class Qwen3ForceAlignTextProcessor:
    def __init__(self) -> None:
        ko_dict_path = os.path.join(os.path.dirname(__file__), "assets", "korean_dict_jieba.dict")
        ko_scores: dict[str, float] = {}
        with open(ko_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ko_scores[line.split()[0]] = 1.0
        self.ko_score = ko_scores
        self.ko_tokenizer: Any = None

    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )

    def tokenize_chinese_mixed(self, text: str) -> List[str]:
        tokens: List[str] = []
        current_latin: List[str] = []

        def flush_latin() -> None:
            nonlocal current_latin
            if current_latin:
                cleaned = self.clean_token("".join(current_latin))
                if cleaned:
                    tokens.append(cleaned)
                current_latin = []

        for ch in text:
            if self.is_cjk_char(ch):
                flush_latin()
                tokens.append(ch)
            elif self.is_kept_char(ch):
                current_latin.append(ch)
            else:
                flush_latin()

        flush_latin()
        return tokens

    def tokenize_japanese(self, text: str) -> List[str]:
        return [cleaned for word in nagisa.tagging(text).words if (cleaned := self.clean_token(word))]

    def tokenize_korean(self, ko_tokenizer: Any, text: str) -> List[str]:
        return [cleaned for word in ko_tokenizer.tokenize(text) if (cleaned := self.clean_token(word))]

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens: List[str] = []
        buf: List[str] = []

        def flush_buf() -> None:
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def fix_timestamp(self, data: Any) -> List[int]:
        data = data.tolist() if hasattr(data, "tolist") else list(data)
        n = len(data)
        if n == 0:
            raise ValueError("max() arg is an empty sequence")

        parent = [-1] * n
        values = sorted(set(data))
        tree: List[tuple[int, int]] = [(0, -1)] * (len(values) + 1)
        max_length = 0
        max_idx = -1

        for i, value in enumerate(data):
            rank = bisect_right(values, value)
            best_len, best_idx = self._fenwick_query(tree, rank)
            length = best_len + 1
            parent[i] = best_idx
            self._fenwick_update(tree, rank, (length, i))
            if length > max_length:
                max_length = length
                max_idx = i

        lis_indices: List[int] = []
        while max_idx != -1:
            lis_indices.append(max_idx)
            max_idx = parent[max_idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for idx in lis_indices:
            is_normal[idx] = True

        result = data.copy()
        i = 0
        while i < n:
            if is_normal[i]:
                i += 1
                continue

            j = i
            while j < n and not is_normal[j]:
                j += 1

            anomaly_count = j - i
            left_val = None
            for k in range(i - 1, -1, -1):
                if is_normal[k]:
                    left_val = result[k]
                    break

            right_val = None
            for k in range(j, n):
                if is_normal[k]:
                    right_val = result[k]
                    break

            if anomaly_count <= 2:
                for k in range(i, j):
                    if left_val is None:
                        result[k] = right_val
                    elif right_val is None:
                        result[k] = left_val
                    else:
                        result[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
            elif left_val is not None and right_val is not None:
                step = (right_val - left_val) / (anomaly_count + 1)
                for k in range(i, j):
                    result[k] = left_val + step * (k - i + 1)
            elif left_val is not None:
                for k in range(i, j):
                    result[k] = left_val
            elif right_val is not None:
                for k in range(i, j):
                    result[k] = right_val

            i = j

        return [int(res) for res in result]

    @classmethod
    def _fenwick_query(cls, tree: List[tuple[int, int]], idx: int) -> tuple[int, int]:
        best = (0, -1)
        while idx > 0:
            best = cls._better_lis_candidate(best, tree[idx])
            idx -= idx & -idx
        return best

    @classmethod
    def _fenwick_update(cls, tree: List[tuple[int, int]], idx: int, candidate: tuple[int, int]) -> None:
        while idx < len(tree):
            tree[idx] = cls._better_lis_candidate(tree[idx], candidate)
            idx += idx & -idx

    @staticmethod
    def _better_lis_candidate(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        if left[1] == -1:
            return right
        if right[1] == -1:
            return left
        return left if left[1] <= right[1] else right

    def encode_timestamp(self, text: str, language: str) -> tuple[List[str], str]:
        language = language.lower()
        if language == "japanese":
            word_list = self.tokenize_japanese(text)
        elif language == "korean":
            if self.ko_tokenizer is None:
                from soynlp.tokenizer import LTokenizer

                self.ko_tokenizer = LTokenizer(scores=self.ko_score)
            word_list = self.tokenize_korean(self.ko_tokenizer, text)
        else:
            word_list = self.tokenize_space_lang(text)

        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text
        return word_list, input_text

    def parse_timestamp(self, word_list: Sequence[str], timestamp: Any) -> List[Dict[str, Any]]:
        timestamp_fixed = self.fix_timestamp(timestamp)
        timestamp_output = []
        for i, word in enumerate(word_list):
            timestamp_output.append(
                {
                    "text": word,
                    "start_time": timestamp_fixed[i * 2],
                    "end_time": timestamp_fixed[i * 2 + 1],
                }
            )
        return timestamp_output


class Qwen3ForcedAlignerBackend:
    def __init__(
        self,
        model: Qwen3ASRForConditionalGeneration,
        processor: Qwen3ASRProcessor,
        aligner_processor: Optional[Qwen3ForceAlignTextProcessor] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.aligner_processor = aligner_processor or Qwen3ForceAlignTextProcessor()

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self.timestamp_token_id = int(model.config.timestamp_token_id)
        self.timestamp_segment_time = float(model.config.timestamp_segment_time)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "Qwen3ForcedAlignerBackend":
        register_qwen3_asr_auto_classes()
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3ASRForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3ASRForConditionalGeneration."
            )

        processor_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "local_files_only", "revision", "token", "trust_remote_code")
            if key in kwargs
        }
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            fix_mistral_regex=True,
            **processor_kwargs,
        )
        return cls(model=model, processor=processor, aligner_processor=Qwen3ForceAlignTextProcessor())

    def _to_structured_items(self, timestamp_output: List[Dict[str, Any]]) -> ForcedAlignResult:
        items: List[ForcedAlignItem] = []
        for it in timestamp_output:
            items.append(
                ForcedAlignItem(
                    text=str(it.get("text", "")),
                    start_time=float(it.get("start_time", 0)),
                    end_time=float(it.get("end_time", 0)),
                )
            )
        return ForcedAlignResult(items=items)

    @torch.inference_mode()
    def align(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        text: Union[str, List[str]],
        language: Union[str, List[str]],
    ) -> List[ForcedAlignResult]:
        return self._align_normalized(normalize_audios(audio), text=text, language=language)

    @torch.inference_mode()
    def _align_normalized(
        self,
        audios: List[np.ndarray],
        *,
        text: Union[str, List[str]],
        language: Union[str, List[str]],
    ) -> List[ForcedAlignResult]:
        texts = ensure_list(text)
        languages = ensure_list(language)

        if len(languages) == 1 and len(audios) > 1:
            languages = languages * len(audios)

        if not (len(audios) == len(texts) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: audio={len(audios)}, text={len(texts)}, language={len(languages)}"
            )

        word_lists = []
        aligner_input_texts = []
        for t, lang in zip(texts, languages):
            word_list, aligner_input_text = self.aligner_processor.encode_timestamp(t, lang)
            word_lists.append(word_list)
            aligner_input_texts.append(aligner_input_text)

        inputs = self.processor(
            text=aligner_input_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        self._drop_single_full_attention_mask(inputs)
        self._drop_single_full_feature_mask(inputs)
        inputs = self._move_inputs_like_official(inputs)

        input_ids = inputs["input_ids"]
        timestamp_mask = input_ids == self.timestamp_token_id
        logits = self.model.thinker(**inputs).logits

        results: List[ForcedAlignResult] = []
        for row, word_list in enumerate(word_lists):
            masked_output_id = logits[row, timestamp_mask[row], :].argmax(dim=-1)
            timestamp_ms = (masked_output_id * self.timestamp_segment_time).to("cpu").numpy()
            timestamp_output = self.aligner_processor.parse_timestamp(word_list, timestamp_ms)
            for it in timestamp_output:
                it["start_time"] = round(it["start_time"] / 1000.0, 3)
                it["end_time"] = round(it["end_time"] / 1000.0, 3)
            results.append(self._to_structured_items(timestamp_output))

        return results

    def align_sentence(
        self,
        audio: AudioLike,
        text: str,
        language: str,
    ) -> Optional[ForcedAlignSentence]:
        result = self.align(audio=audio, text=text, language=language)[0]
        if not result.items:
            return None
        return ForcedAlignSentence(
            text=text,
            start_time=result.items[0].start_time,
            end_time=result.items[-1].end_time,
        )

    def align_transcript_segments(
        self,
        audio: AudioLike,
        segments: Sequence[ForcedAlignTextSegment],
        language: str,
        *,
        window_sec: float = 90.0,
        pad_sec: float = 0.0,
    ) -> List[Optional[ForcedAlignSentence]]:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        if pad_sec < 0:
            raise ValueError("pad_sec must be non-negative")
        if window_sec + 2 * pad_sec > MAX_FORCE_ALIGN_INPUT_SECONDS:
            raise ValueError(
                f"window_sec + 2 * pad_sec must be <= {MAX_FORCE_ALIGN_INPUT_SECONDS}"
            )

        items = list(segments)
        if not items:
            return []

        normalized = normalize_audios(audio)
        if len(normalized) != 1:
            raise ValueError("audio must be a single continuous input")
        wav = normalized[0]
        audio_duration = len(wav) / SAMPLE_RATE
        outputs: List[Optional[ForcedAlignSentence]] = [None] * len(items)
        windows: List[_TranscriptWindow] = []
        for group in self._iter_transcript_windows(items, window_sec=window_sec):
            start = max(0.0, group[0][1].start_time - pad_sec)
            end = min(audio_duration, group[-1][1].end_time + pad_sec)
            if end <= start:
                continue
            crop = wav[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
            if crop.size == 0:
                continue
            windows.append(
                _TranscriptWindow(
                    audio=crop,
                    offset=start,
                    text=" ".join(segment.text for _, segment in group),
                    entries=[
                        (idx, segment, len(self.aligner_processor.encode_timestamp(segment.text, language)[0]))
                        for idx, segment in group
                    ],
                )
            )

        for window in windows:
            result = self._align_normalized([window.audio], text=window.text, language=language)[0]
            cursor = 0
            for original_idx, segment, token_count in window.entries:
                if token_count > 0 and cursor + token_count <= len(result.items):
                    outputs[original_idx] = ForcedAlignSentence(
                        text=segment.text,
                        start_time=round(window.offset + result.items[cursor].start_time, 3),
                        end_time=round(window.offset + result.items[cursor + token_count - 1].end_time, 3),
                    )
                cursor += token_count
        return outputs

    def get_supported_languages(self) -> Optional[List[str]]:
        fn = getattr(self.model, "get_support_languages", None)
        if not callable(fn):
            return None
        langs = fn()
        if langs is None:
            return None
        return sorted({str(x).lower() for x in langs})

    def _move_inputs_like_official(self, inputs: Any) -> Any:
        device = getattr(self.model, "device", self.device)
        dtype = getattr(self.model, "dtype", None)
        if hasattr(inputs, "to"):
            if dtype is not None:
                return inputs.to(device=device, dtype=dtype)
            return inputs.to(device=device)

        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                continue
            kwargs: dict[str, Any] = {"device": device}
            if dtype is not None and torch.is_floating_point(value):
                kwargs["dtype"] = dtype
            inputs[key] = value.to(**kwargs)
        return inputs

    @staticmethod
    def _drop_single_full_attention_mask(inputs: Any) -> None:
        attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
        if (
            torch.is_tensor(attention_mask)
            and attention_mask.shape[0] == 1
            and bool(attention_mask.all().item())
        ):
            inputs["attention_mask"] = None

    @staticmethod
    def _drop_single_full_feature_mask(inputs: Any) -> None:
        feature_attention_mask = inputs.get("feature_attention_mask") if hasattr(inputs, "get") else None
        if (
            torch.is_tensor(feature_attention_mask)
            and feature_attention_mask.shape[0] == 1
            and bool(feature_attention_mask.all().item())
        ):
            inputs["audio_feature_lengths"] = feature_attention_mask.sum(dim=1)
            inputs["feature_attention_mask"] = None

    @staticmethod
    def _iter_transcript_windows(
        segments: Sequence[ForcedAlignTextSegment],
        *,
        window_sec: float,
    ) -> Iterable[List[tuple[int, ForcedAlignTextSegment]]]:
        ordered = sorted(enumerate(segments), key=lambda item: (item[1].start_time, item[1].end_time))
        group: List[tuple[int, ForcedAlignTextSegment]] = []
        group_start = 0.0
        for item in ordered:
            segment = item[1]
            if not group:
                group = [item]
                group_start = segment.start_time
                continue
            if segment.end_time - group_start > window_sec:
                yield group
                group = [item]
                group_start = segment.start_time
            else:
                group.append(item)
        if group:
            yield group


def normalize_forced_align_language(language: str) -> str:
    language = normalize_language_name(language)
    if language not in FORCED_ALIGNER_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported forced-align language: {language}. "
            f"Supported: {FORCED_ALIGNER_SUPPORTED_LANGUAGES}"
        )
    return language


def register_qwen3_asr_auto_classes() -> None:
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)


__all__ = [
    "FORCED_ALIGNER_SUPPORTED_LANGUAGES",
    "ForcedAlignItem",
    "ForcedAlignResult",
    "ForcedAlignSentence",
    "ForcedAlignTextSegment",
    "Qwen3ForceAlignTextProcessor",
    "Qwen3ForcedAlignerBackend",
    "normalize_forced_align_language",
    "register_qwen3_asr_auto_classes",
]
