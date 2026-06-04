# coding=utf-8
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import audioread.exceptions
import librosa
import numpy as np
import soundfile as sf

from .forced_aligner import normalize_forced_align_language
from .transcription_document import TranscriptDocument, TranscriptSegment
from .utils import (
    MIN_ASR_INPUT_SECONDS,
    SAMPLE_RATE,
    float_range_normalize,
    merge_languages,
    normalize_audios,
    split_audio_into_chunks,
)

_DEFAULT_OFFLINE_CHUNK_SEC = 60.0
_DEFAULT_TIMESTAMP_TIMEOUT_SEC = 30.0


class OfflineTranscriptionInputError(ValueError):
    """Raised for client-supplied audio that cannot be decoded."""


@dataclass(frozen=True)
class OfflineTranscriptionOptions:
    language: str | None = None
    context: str = ""
    target_language: str | None = None
    timestamps: bool = True


@dataclass(frozen=True)
class OfflineTranscriptionStreamEvent:
    kind: str
    segment: TranscriptSegment | None = None
    document: TranscriptDocument | None = None


async def transcribe_file(
    model: Any,
    audio_source: Any,
    *,
    options: OfflineTranscriptionOptions | None = None,
    timestamp_actor: Any | None = None,
    translation_actor: Any | None = None,
    translation_max_new_tokens: int | None = None,
    asr_executor: Executor | None = None,
    chunk_sec: float = _DEFAULT_OFFLINE_CHUNK_SEC,
    timestamp_timeout_sec: float = _DEFAULT_TIMESTAMP_TIMEOUT_SEC,
) -> TranscriptDocument:
    opts = options or OfflineTranscriptionOptions()

    segments: list[TranscriptSegment] = []
    languages: list[str] = []
    total_samples = 0
    for chunk, offset_sec, chunk_samples in _iter_audio_chunks(audio_source, chunk_sec=float(chunk_sec)):
        start_ms = int(round(float(offset_sec) * 1000))
        end_ms = start_ms + int(round(1000 * int(chunk_samples) / SAMPLE_RATE))
        total_samples = max(total_samples, int(round(offset_sec * SAMPLE_RATE)) + int(chunk_samples))

        result = await _transcribe_chunk(
            model,
            chunk,
            context=opts.context,
            language=opts.language,
            asr_executor=asr_executor,
        )
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            continue
        language = str(getattr(result, "language", "") or opts.language or "")
        if language:
            languages.append(language)
        segment = TranscriptSegment(
            id=f"seg_{len(segments) + 1:06d}",
            index=len(segments) + 1,
            start_ms=start_ms,
            end_ms=max(start_ms, end_ms),
            text=text,
            language=language,
            timing_status="estimated",
        )
        if opts.timestamps and timestamp_actor is not None:
            segment = await _align_segment(
                segment,
                audio=chunk,
                base_ms=start_ms,
                timestamp_actor=timestamp_actor,
                timeout_sec=float(timestamp_timeout_sec),
            )
        segments.append(segment)

    document_language = merge_languages(languages)
    if opts.target_language:
        if translation_actor is None:
            raise ValueError("Translation requested but no translation model is loaded.")
        segments = await _translate_segments(
            segments,
            translation_actor=translation_actor,
            target_language=opts.target_language,
            source_language=document_language,
            max_new_tokens=translation_max_new_tokens,
        )

    return TranscriptDocument(
        duration_ms=int(round(1000 * total_samples / SAMPLE_RATE)),
        language=document_language,
        segments=segments,
    )


async def stream_transcribe_file(
    model: Any,
    audio_source: Any,
    *,
    options: OfflineTranscriptionOptions | None = None,
    timestamp_actor: Any | None = None,
    asr_executor: Executor | None = None,
    chunk_sec: float = _DEFAULT_OFFLINE_CHUNK_SEC,
    timestamp_timeout_sec: float = _DEFAULT_TIMESTAMP_TIMEOUT_SEC,
) -> AsyncIterator[OfflineTranscriptionStreamEvent]:
    opts = options or OfflineTranscriptionOptions()
    if opts.target_language:
        raise ValueError(
            "stream_transcribe_file does not support translation; use a service-layer translation side track."
        )

    segments: list[TranscriptSegment] = []
    languages: list[str] = []
    total_samples = 0
    for chunk, offset_sec, chunk_samples in _iter_audio_chunks(audio_source, chunk_sec=float(chunk_sec)):
        start_ms = int(round(float(offset_sec) * 1000))
        end_ms = start_ms + int(round(1000 * int(chunk_samples) / SAMPLE_RATE))
        total_samples = max(total_samples, int(round(offset_sec * SAMPLE_RATE)) + int(chunk_samples))

        result = await _transcribe_chunk(
            model,
            chunk,
            context=opts.context,
            language=opts.language,
            asr_executor=asr_executor,
        )
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            continue
        language = str(getattr(result, "language", "") or opts.language or "")
        if language:
            languages.append(language)
        segment = TranscriptSegment(
            id=f"seg_{len(segments) + 1:06d}",
            index=len(segments) + 1,
            start_ms=start_ms,
            end_ms=max(start_ms, end_ms),
            text=text,
            language=language,
            timing_status="estimated",
        )
        if opts.timestamps and timestamp_actor is not None:
            segment = await _align_segment(
                segment,
                audio=chunk,
                base_ms=start_ms,
                timestamp_actor=timestamp_actor,
                timeout_sec=float(timestamp_timeout_sec),
            )
        segments.append(segment)
        yield OfflineTranscriptionStreamEvent(kind="segment", segment=segment)

    yield OfflineTranscriptionStreamEvent(
        kind="complete",
        document=TranscriptDocument(
            duration_ms=int(round(1000 * total_samples / SAMPLE_RATE)),
            language=merge_languages(languages),
            segments=segments,
        ),
    )


def _iter_audio_chunks(audio_source: Any, *, chunk_sec: float) -> Iterator[tuple[np.ndarray, float, int]]:
    path = _local_file_path(audio_source)
    if path is not None:
        yield from _iter_file_audio_chunks(path, chunk_sec=chunk_sec)
        return

    wav = normalize_audios(audio_source)[0]
    total_samples = int(wav.shape[0])
    for chunk, offset_sec in split_audio_into_chunks(wav, SAMPLE_RATE, max_chunk_sec=chunk_sec):
        offset_samples = int(round(offset_sec * SAMPLE_RATE))
        chunk_samples = min(int(chunk.shape[0]), max(0, total_samples - offset_samples))
        yield chunk, offset_sec, chunk_samples


def _local_file_path(audio_source: Any) -> Path | None:
    if not isinstance(audio_source, (str, Path)):
        return None
    try:
        path = Path(audio_source)
        return path if path.is_file() else None
    except OSError:
        return None


def _iter_file_audio_chunks(path: Path, *, chunk_sec: float) -> Iterator[tuple[np.ndarray, float, int]]:
    try:
        sample_rate = int(librosa.get_samplerate(str(path)))
        native_chunk_samples = max(1, int(round(chunk_sec * sample_rate)))
        offset_samples = 0
        for block in librosa.stream(
            str(path),
            block_length=1,
            frame_length=native_chunk_samples,
            hop_length=native_chunk_samples,
            mono=True,
            dtype=np.float32,
        ):
            audio = np.asarray(block, dtype=np.float32)
            if sample_rate != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE).astype(np.float32)
            audio = float_range_normalize(audio)
            original_samples = int(audio.shape[0])
            yield _pad_short_chunk(audio), offset_samples / float(SAMPLE_RATE), original_samples
            offset_samples += original_samples
    except (audioread.exceptions.DecodeError, OSError, RuntimeError, sf.SoundFileError, ValueError) as exc:
        raise OfflineTranscriptionInputError(f"Unsupported or unreadable audio file: {path.name}") from exc


def _pad_short_chunk(chunk: np.ndarray) -> np.ndarray:
    min_samples = int(MIN_ASR_INPUT_SECONDS * SAMPLE_RATE)
    if int(chunk.shape[0]) >= min_samples:
        return chunk
    return np.pad(chunk, (0, min_samples - int(chunk.shape[0])), mode="constant", constant_values=0.0).astype(
        np.float32
    )


async def _align_segment(
    segment: TranscriptSegment,
    *,
    audio: np.ndarray,
    base_ms: int,
    timestamp_actor: Any,
    timeout_sec: float,
) -> TranscriptSegment:
    language = _forced_align_language(segment.language)
    if language is None:
        return segment
    start_sec, end_sec, error = await timestamp_actor.align_segment(
        audio,
        text=segment.text,
        language=language,
        timeout_sec=timeout_sec,
    )
    if error is not None or start_sec is None or end_sec is None:
        return replace(segment, timing_status=error or "failed")
    return replace(
        segment,
        start_ms=base_ms + int(round(start_sec * 1000)),
        end_ms=base_ms + int(round(end_sec * 1000)),
        timing_status="aligned",
    )


async def _transcribe_chunk(
    model: Any,
    chunk: np.ndarray,
    *,
    context: str,
    language: str | None,
    asr_executor: Executor | None,
) -> Any:
    call = partial(
        model.transcribe,
        audio=(chunk, SAMPLE_RATE),
        context=context,
        language=language,
        return_time_stamps=False,
    )
    if asr_executor is None:
        return call()[0]
    loop = asyncio.get_running_loop()
    return (await loop.run_in_executor(asr_executor, call))[0]


async def _translate_segments(
    segments: list[TranscriptSegment],
    *,
    translation_actor: Any,
    target_language: str,
    source_language: str,
    max_new_tokens: int | None,
) -> list[TranscriptSegment]:
    if not segments:
        return segments
    texts = [segment.text for segment in segments]
    outputs = await translation_actor.translate_batch(
        texts,
        target_language=target_language,
        source_language=source_language,
        max_new_tokens=max_new_tokens,
        timeout_sec=None,
    )
    translated: list[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        text, error = outputs[index] if index < len(outputs) else (None, "missing translation output")
        translated.append(segment if error is not None or not text else replace(segment, translation=text))
    return translated


def _forced_align_language(language: str) -> str | None:
    try:
        return normalize_forced_align_language(language)
    except ValueError:
        return None


__all__ = [
    "OfflineTranscriptionInputError",
    "OfflineTranscriptionOptions",
    "OfflineTranscriptionStreamEvent",
    "stream_transcribe_file",
    "transcribe_file",
]
