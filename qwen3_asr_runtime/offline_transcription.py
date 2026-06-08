# coding=utf-8
from __future__ import annotations

import asyncio
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator, Sequence
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
from .offline_units import (
    SourceUnit,
    SourceUnitBuilder,
    TimedToken,
    estimated_timed_tokens_from_text,
    layout_source_cues,
    timed_tokens_from_aligned_items,
)
from .transcription_document import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptTranslationUnit,
)
from .utils import (
    MIN_ASR_INPUT_SECONDS,
    SAMPLE_RATE,
    float_range_normalize,
    merge_languages,
    normalize_audios,
)

_DEFAULT_OFFLINE_CHUNK_SEC = 120.0
_DEFAULT_TIMESTAMP_TIMEOUT_SEC = 30.0
_BOUNDARY_HOLD_MS = 3_000
_MAX_REFEED_SEC = 30.0
_REFEED_MIN_CHUNK_SEC = 30.0
_MIN_TRANSCRIPT_SEGMENT_MS = 80


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
    translation_unit: "OfflineTranslationUnit | None" = None


@dataclass(frozen=True)
class OfflineTranslationUnit:
    source_text: str
    source_language: str
    source_segment_ids: tuple[str, ...]
    source_segment_indices: tuple[int, ...]
    anchor_segment_list_index: int


@dataclass(frozen=True)
class _SourceUnitBatch:
    units: tuple[SourceUnit, ...]
    language: str
    total_samples: int


@dataclass(frozen=True)
class _DecodeWindow:
    audio: np.ndarray
    base_ms: int
    duration_ms: int


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
    translation_units: list[OfflineTranslationUnit] = []
    document_translation_units: list[TranscriptTranslationUnit] = []
    languages: list[str] = []
    total_samples = 0
    async for batch in _iter_source_unit_batches(
        model,
        audio_source,
        options=opts,
        timestamp_actor=timestamp_actor,
        asr_executor=asr_executor,
        chunk_sec=chunk_sec,
        timestamp_timeout_sec=timestamp_timeout_sec,
    ):
        total_samples = max(total_samples, batch.total_samples)
        if batch.language:
            languages.append(batch.language)
        _append_source_units(segments, batch.units, translation_units=translation_units)

    document_language = merge_languages(languages)
    if opts.target_language:
        if translation_actor is None:
            raise ValueError(
                "Translation requested but no translation model is loaded."
            )
        segments, document_translation_units = await _translate_units(
            segments,
            translation_units,
            translation_actor=translation_actor,
            target_language=opts.target_language,
            source_language=document_language,
            max_new_tokens=translation_max_new_tokens,
        )

    return TranscriptDocument(
        duration_ms=int(round(1000 * total_samples / SAMPLE_RATE)),
        language=document_language,
        segments=segments,
        translation_units=document_translation_units,
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
    async for batch in _iter_source_unit_batches(
        model,
        audio_source,
        options=opts,
        timestamp_actor=timestamp_actor,
        asr_executor=asr_executor,
        chunk_sec=chunk_sec,
        timestamp_timeout_sec=timestamp_timeout_sec,
    ):
        total_samples = max(total_samples, batch.total_samples)
        if batch.language:
            languages.append(batch.language)
        for unit in batch.units:
            for event in _append_source_unit_events(segments, unit):
                yield event

    document_language = merge_languages(languages)

    yield OfflineTranscriptionStreamEvent(
        kind="complete",
        document=TranscriptDocument(
            duration_ms=int(round(1000 * total_samples / SAMPLE_RATE)),
            language=document_language,
            segments=segments,
        ),
    )


async def _iter_source_unit_batches(
    model: Any,
    audio_source: Any,
    *,
    options: OfflineTranscriptionOptions,
    timestamp_actor: Any | None,
    asr_executor: Executor | None,
    chunk_sec: float,
    timestamp_timeout_sec: float,
) -> AsyncIterator[_SourceUnitBatch]:
    unit_builder = SourceUnitBuilder()
    provisional_unit: SourceUnit | None = None
    audio_tail = np.empty(0, dtype=np.float32)
    refeed_enabled = _validate_chunk_sec(chunk_sec) >= _REFEED_MIN_CHUNK_SEC
    max_refeed_samples = (
        int(round(_MAX_REFEED_SEC * SAMPLE_RATE)) if refeed_enabled else 0
    )
    total_samples = 0
    for (
        chunk,
        offset_sec,
        chunk_samples,
        is_final_window,
        source_total_samples,
    ) in _iter_source_audio_chunks(
        audio_source,
        chunk_sec=chunk_sec,
    ):
        total_samples = max(total_samples, source_total_samples)
        main_audio = np.asarray(chunk[: int(chunk_samples)], dtype=np.float32)
        start_ms = int(round(float(offset_sec) * 1000))
        duration_ms = _samples_to_ms(int(chunk_samples))
        end_ms = start_ms + duration_ms
        decode_window = _make_decode_window(
            main_audio,
            main_start_ms=start_ms,
            provisional_unit=provisional_unit,
            audio_tail=audio_tail,
            max_refeed_samples=max_refeed_samples,
        )
        tokens, language, timing_status = await _transcribe_decode_window(
            model,
            decode_window,
            main_audio=main_audio,
            main_start_ms=start_ms,
            main_duration_ms=duration_ms,
            context=options.context,
            language=options.language,
            timestamp_actor=timestamp_actor if options.timestamps else None,
            asr_executor=asr_executor,
            timeout_sec=float(timestamp_timeout_sec),
        )
        units = _add_window_tokens(
            unit_builder,
            tokens,
            language=language,
            timing_status=timing_status,
            previous_provisional=provisional_unit,
            main_start_ms=start_ms,
            replace_provisional=_should_replace_provisional(
                provisional_unit,
                tokens,
                timing_status=timing_status,
                decode_start_ms=decode_window.base_ms,
                main_start_ms=start_ms,
            ),
        )
        provisional_unit = None
        if refeed_enabled and not is_final_window and timing_status == "aligned":
            provisional_unit = _take_boundary_provisional_unit(
                unit_builder,
                units,
                boundary_ms=end_ms,
                hold_ms=_BOUNDARY_HOLD_MS,
                max_refeed_ms=_samples_to_ms(max_refeed_samples),
            )
        if is_final_window:
            units.extend(unit_builder.flush())
        audio_tail = _append_audio_tail(
            audio_tail, main_audio, max_samples=max_refeed_samples
        )
        yield _SourceUnitBatch(tuple(units), language, source_total_samples)

    flush_language = ""
    if provisional_unit is not None:
        flush_language = provisional_unit.language
        unit_builder.add_tokens(
            provisional_unit.tokens,
            language=provisional_unit.language,
            timing_status=provisional_unit.timing_status,
        )
    flush_units = unit_builder.flush()
    if flush_units:
        yield _SourceUnitBatch(tuple(flush_units), flush_language, total_samples)


def _iter_source_audio_chunks(
    audio_source: Any,
    *,
    chunk_sec: float,
) -> Iterator[tuple[np.ndarray, float, int, bool, int]]:
    chunk_sec = _validate_chunk_sec(chunk_sec)
    path = _local_file_path(audio_source)
    if path is None:
        wav = np.asarray(normalize_audios(audio_source)[0], dtype=np.float32)
        yield from _iter_pcm_window_chunks(
            iter((wav,)), chunk_sec=chunk_sec, total_samples=int(wav.shape[0])
        )
        return
    if _soundfile_readable(path):
        yield from _iter_file_audio_chunks(path, chunk_sec=chunk_sec)
        return
    # Video containers and other media libsndfile cannot open (mp4, mkv, mov,
    # webm, m4a, ...) are decoded to 16 kHz mono by ffmpeg and streamed straight
    # from its stdout, so no temporary file or whole-file buffer is needed.
    yield from _iter_pcm_window_chunks(
        _iter_ffmpeg_pcm_blocks(path), chunk_sec=chunk_sec
    )


def _iter_pcm_window_chunks(
    blocks: Iterator[np.ndarray],
    *,
    chunk_sec: float,
    total_samples: int | None = None,
) -> Iterator[tuple[np.ndarray, float, int, bool, int]]:
    """Slice a stream of contiguous 16 kHz mono float32 blocks into fixed windows.

    An in-memory array (one block) and a live ffmpeg pipe (many blocks) share
    this, so video decoding needs neither a temporary file nor a whole-file
    buffer. `total_samples` is reported verbatim when known (arrays); for a pipe
    it grows to the true total on the final window, which is what the duration
    accumulator consumes.
    """
    chunk_samples = _chunk_sample_count(chunk_sec)
    buffer = np.empty(0, dtype=np.float32)
    offset_samples = 0
    blocks_done = False
    while True:
        while not blocks_done and buffer.shape[0] <= chunk_samples:
            block = next(blocks, None)
            if block is None:
                blocks_done = True
                break
            block = np.asarray(block, dtype=np.float32)
            if block.shape[0]:
                buffer = (
                    block if buffer.shape[0] == 0 else np.concatenate((buffer, block))
                )
        if buffer.shape[0] == 0:
            break
        is_final_window = blocks_done and buffer.shape[0] <= chunk_samples
        actual_samples = (
            buffer.shape[0] if is_final_window else min(chunk_samples, buffer.shape[0])
        )
        window = buffer[:actual_samples].astype(np.float32, copy=False)
        emitted_total = (
            total_samples
            if total_samples is not None
            else offset_samples + actual_samples
        )
        yield (
            _pad_short_chunk(window),
            offset_samples / float(SAMPLE_RATE),
            actual_samples,
            is_final_window,
            emitted_total,
        )
        if is_final_window:
            break
        buffer = buffer[chunk_samples:]
        offset_samples += chunk_samples


def _iter_file_audio_chunks(
    path: Path,
    *,
    chunk_sec: float,
) -> Iterator[tuple[np.ndarray, float, int, bool, int]]:
    chunk_samples = _chunk_sample_count(chunk_sec)
    try:
        total_samples = _audio_duration_samples(path)
        if total_samples <= 0:
            return
        sample_rate = int(librosa.get_samplerate(str(path)))
        native_frame_samples = max(
            1, int(round(chunk_samples * sample_rate / SAMPLE_RATE))
        )
        offset_samples = 0
        for block in librosa.stream(
            str(path),
            block_length=1,
            frame_length=native_frame_samples,
            hop_length=native_frame_samples,
            mono=True,
            dtype=np.float32,
        ):
            remaining_samples = max(0, total_samples - offset_samples)
            actual_samples = min(chunk_samples, remaining_samples)
            if actual_samples <= 0:
                break
            audio = np.asarray(block, dtype=np.float32)
            if sample_rate != SAMPLE_RATE:
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE
                ).astype(np.float32)
            audio = _fit_audio_samples(float_range_normalize(audio), actual_samples)
            is_final_window = offset_samples + actual_samples >= total_samples
            yield (
                _pad_short_chunk(audio),
                offset_samples / float(SAMPLE_RATE),
                actual_samples,
                is_final_window,
                total_samples,
            )
            if is_final_window:
                break
            offset_samples += chunk_samples
    except (
        audioread.exceptions.DecodeError,
        OSError,
        RuntimeError,
        sf.SoundFileError,
        ValueError,
        EOFError,
    ) as exc:
        raise OfflineTranscriptionInputError(
            f"Unsupported or unreadable media file: {path.name}"
        ) from exc


def _soundfile_readable(path: Path) -> bool:
    try:
        sf.info(str(path))
    except (sf.SoundFileError, RuntimeError, OSError):
        return False
    return True


def _iter_ffmpeg_pcm_blocks(
    path: Path, *, block_sec: float = 30.0
) -> Iterator[np.ndarray]:
    """Stream media decoded to 16 kHz mono float32 directly from ffmpeg's stdout."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise OfflineTranscriptionInputError(
            "ffmpeg is required to decode this media file but was not found on PATH."
        )
    block_bytes = (
        max(1, int(round(block_sec * SAMPLE_RATE))) * 4
    )  # float32 little-endian samples
    try:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "f32le",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise OfflineTranscriptionInputError(
            f"Unsupported or unreadable media file: {path.name}"
        ) from exc
    stdout = process.stdout
    assert stdout is not None
    try:
        carry = b""
        while True:
            data = stdout.read(block_bytes)
            if not data:
                break
            carry += data
            usable = len(carry) - (len(carry) % 4)
            if usable:
                yield np.frombuffer(carry[:usable], dtype="<f4").astype(np.float32)
                carry = carry[usable:]
        if process.wait() != 0:
            raise OfflineTranscriptionInputError(
                f"Unsupported or unreadable media file: {path.name}"
            )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        stdout.close()


def _chunk_sample_count(chunk_sec: float) -> int:
    chunk_sec = _validate_chunk_sec(chunk_sec)
    return max(1, int(round(chunk_sec * SAMPLE_RATE)))


def _audio_duration_samples(path: Path) -> int:
    try:
        info = sf.info(str(path))
        return max(
            0, int(round(float(info.frames) * SAMPLE_RATE / float(info.samplerate)))
        )
    except sf.SoundFileError:
        duration_sec = librosa.get_duration(path=str(path))
        return max(0, int(round(float(duration_sec) * SAMPLE_RATE)))


def _fit_audio_samples(audio: np.ndarray, samples: int) -> np.ndarray:
    target_samples = max(0, int(samples))
    if int(audio.shape[0]) >= target_samples:
        return audio[:target_samples].astype(np.float32, copy=False)
    return np.pad(
        audio,
        (0, target_samples - int(audio.shape[0])),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32)


def _validate_chunk_sec(chunk_sec: float) -> float:
    try:
        value = float(chunk_sec)
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_sec must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("chunk_sec must be a finite positive number")
    return value


def _local_file_path(audio_source: Any) -> Path | None:
    if not isinstance(audio_source, (str, Path)):
        return None
    try:
        path = Path(audio_source)
        return path if path.is_file() else None
    except OSError:
        return None


def _pad_short_chunk(chunk: np.ndarray) -> np.ndarray:
    min_samples = int(MIN_ASR_INPUT_SECONDS * SAMPLE_RATE)
    if int(chunk.shape[0]) >= min_samples:
        return chunk
    return np.pad(
        chunk,
        (0, min_samples - int(chunk.shape[0])),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32)


def _samples_to_ms(samples: int) -> int:
    return int(round(1000 * int(samples) / SAMPLE_RATE))


def _make_decode_window(
    main_audio: np.ndarray,
    *,
    main_start_ms: int,
    provisional_unit: SourceUnit | None,
    audio_tail: np.ndarray,
    max_refeed_samples: int,
) -> _DecodeWindow:
    main_samples = int(main_audio.shape[0])
    if provisional_unit is None or provisional_unit.timing_status != "aligned":
        return _DecodeWindow(
            audio=_pad_short_chunk(main_audio),
            base_ms=int(main_start_ms),
            duration_ms=_samples_to_ms(main_samples),
        )

    provisional_start_ms = int(provisional_unit.tokens[0].start_ms)
    needed_samples = max(
        0, int(round((int(main_start_ms) - provisional_start_ms) * SAMPLE_RATE / 1000))
    )
    if (
        needed_samples <= 0
        or needed_samples > max(0, int(max_refeed_samples))
        or needed_samples > int(audio_tail.shape[0])
    ):
        return _DecodeWindow(
            audio=_pad_short_chunk(main_audio),
            base_ms=int(main_start_ms),
            duration_ms=_samples_to_ms(main_samples),
        )

    refeed_audio = audio_tail[-needed_samples:]
    decode_audio = np.concatenate((refeed_audio, main_audio)).astype(
        np.float32, copy=False
    )
    decode_start_ms = int(main_start_ms) - _samples_to_ms(needed_samples)
    return _DecodeWindow(
        audio=_pad_short_chunk(decode_audio),
        base_ms=decode_start_ms,
        duration_ms=_samples_to_ms(needed_samples + main_samples),
    )


async def _transcribe_decode_window(
    model: Any,
    decode_window: _DecodeWindow,
    *,
    main_audio: np.ndarray,
    main_start_ms: int,
    main_duration_ms: int,
    context: str,
    language: str | None,
    timestamp_actor: Any | None,
    asr_executor: Executor | None,
    timeout_sec: float,
) -> tuple[list[TimedToken], str, str]:
    is_refeed_window = int(decode_window.base_ms) < int(main_start_ms)
    text, detected_language = await _transcribe_text(
        model,
        decode_window.audio,
        context=context,
        language=language,
        asr_executor=asr_executor,
    )
    align_audio = decode_window.audio
    align_base_ms = int(decode_window.base_ms)
    align_duration_ms = int(decode_window.duration_ms)
    used_main_fallback = False
    if not text and is_refeed_window:
        used_main_fallback = True
        align_audio = _pad_short_chunk(main_audio)
        align_base_ms = int(main_start_ms)
        align_duration_ms = int(main_duration_ms)
        text, fallback_language = await _transcribe_text(
            model,
            align_audio,
            context=context,
            language=language,
            asr_executor=asr_executor,
        )
        detected_language = fallback_language or detected_language
    if not text:
        return [], detected_language, "estimated"

    tokens = await _aligned_timed_tokens(
        text,
        audio=align_audio,
        language=detected_language,
        base_ms=align_base_ms,
        duration_ms=align_duration_ms,
        timestamp_actor=timestamp_actor,
        timeout_sec=timeout_sec,
    )
    if tokens:
        return tokens, detected_language, "aligned"

    if not used_main_fallback and is_refeed_window:
        text, fallback_language = await _transcribe_text(
            model,
            _pad_short_chunk(main_audio),
            context=context,
            language=language,
            asr_executor=asr_executor,
        )
        detected_language = fallback_language or detected_language
    tokens = estimated_timed_tokens_from_text(
        text, base_ms=int(main_start_ms), duration_ms=int(main_duration_ms)
    )
    return tokens, detected_language, "estimated"


async def _transcribe_text(
    model: Any,
    audio: np.ndarray,
    *,
    context: str,
    language: str | None,
    asr_executor: Executor | None,
) -> tuple[str, str]:
    result = await _transcribe_chunk(
        model,
        audio,
        context=context,
        language=language,
        asr_executor=asr_executor,
    )
    return str(getattr(result, "text", "") or "").strip(), str(
        getattr(result, "language", "") or language or ""
    )


def _add_window_tokens(
    unit_builder: SourceUnitBuilder,
    tokens: Sequence[TimedToken],
    *,
    language: str,
    timing_status: str,
    previous_provisional: SourceUnit | None,
    main_start_ms: int,
    replace_provisional: bool,
) -> list[SourceUnit]:
    units: list[SourceUnit] = []
    if previous_provisional is not None and not replace_provisional:
        units.extend(
            unit_builder.add_tokens(
                previous_provisional.tokens,
                language=previous_provisional.language,
                timing_status=previous_provisional.timing_status,
            )
        )
        if timing_status == "aligned":
            tokens = [
                token for token in tokens if int(token.start_ms) >= int(main_start_ms)
            ]
    units.extend(
        unit_builder.add_tokens(tokens, language=language, timing_status=timing_status)
    )
    return units


def _should_replace_provisional(
    provisional: SourceUnit | None,
    tokens: Sequence[TimedToken],
    *,
    timing_status: str,
    decode_start_ms: int,
    main_start_ms: int,
) -> bool:
    return (
        provisional is not None
        and timing_status == "aligned"
        and int(decode_start_ms) < int(main_start_ms)
        and _tokens_cover_provisional(provisional, tokens)
    )


def _tokens_cover_provisional(
    provisional: SourceUnit, tokens: Sequence[TimedToken]
) -> bool:
    provisional_key = _text_key(provisional.text)
    if not provisional_key:
        return False
    return _text_key("".join(str(token.text or "") for token in tokens)).startswith(
        provisional_key
    )


def _text_key(text: str) -> str:
    return "".join(char.casefold() for char in str(text or "") if char.isalnum())


def _take_boundary_provisional_unit(
    unit_builder: SourceUnitBuilder,
    units: list[SourceUnit],
    *,
    boundary_ms: int,
    hold_ms: int,
    max_refeed_ms: int,
) -> SourceUnit | None:
    pending = unit_builder.pending_tokens
    if (
        pending
        and _tokens_near_boundary(pending, boundary_ms=boundary_ms, hold_ms=hold_ms)
        and _tokens_fit_refeed_window(
            pending, boundary_ms=boundary_ms, max_refeed_ms=max_refeed_ms
        )
    ):
        return unit_builder.take_pending_unit()
    if (
        units
        and _tokens_near_boundary(
            units[-1].tokens, boundary_ms=boundary_ms, hold_ms=hold_ms
        )
        and _tokens_fit_refeed_window(
            units[-1].tokens, boundary_ms=boundary_ms, max_refeed_ms=max_refeed_ms
        )
    ):
        return units.pop()
    return None


def _tokens_fit_refeed_window(
    tokens: Sequence[TimedToken], *, boundary_ms: int, max_refeed_ms: int
) -> bool:
    if not tokens:
        return False
    return int(tokens[0].start_ms) >= int(boundary_ms) - max(1, int(max_refeed_ms))


def _tokens_near_boundary(
    tokens: Sequence[TimedToken], *, boundary_ms: int, hold_ms: int
) -> bool:
    if not tokens:
        return False
    return int(tokens[-1].end_ms) >= int(boundary_ms) - max(1, int(hold_ms))


def _append_audio_tail(
    tail: np.ndarray, audio: np.ndarray, *, max_samples: int
) -> np.ndarray:
    max_samples = max(0, int(max_samples))
    if max_samples <= 0:
        return np.empty(0, dtype=np.float32)
    combined = audio if int(tail.shape[0]) == 0 else np.concatenate((tail, audio))
    return combined[-max_samples:].astype(np.float32, copy=False)


async def _aligned_timed_tokens(
    text: str,
    *,
    audio: np.ndarray,
    language: str,
    base_ms: int,
    duration_ms: int,
    timestamp_actor: Any | None,
    timeout_sec: float,
) -> list[TimedToken]:
    """Return forced-aligned timed tokens, or an empty list to fall back to estimated timing."""
    align_language = (
        _forced_align_language(language) if timestamp_actor is not None else None
    )
    if align_language is None or not hasattr(timestamp_actor, "align_items"):
        return []
    try:
        result, error = await timestamp_actor.align_items(
            audio,
            text=text,
            language=align_language,
            timeout_sec=timeout_sec,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return []
    if error is not None or result is None:
        return []
    items = list(getattr(result, "items", []) or [])
    try:
        return timed_tokens_from_aligned_items(
            text, items, base_ms=int(base_ms), duration_ms=int(duration_ms)
        )
    except (TypeError, ValueError, OverflowError):
        return []


def _append_source_unit(
    segments: list[TranscriptSegment],
    unit: SourceUnit,
    *,
    translation_units: list[OfflineTranslationUnit] | None = None,
) -> tuple[list[TranscriptSegment], OfflineTranslationUnit | None]:
    unit_segments, unit_translation = _segments_for_source_unit(
        unit,
        next_index=len(segments) + 1,
        segment_list_start=len(segments),
    )
    if not unit_segments:
        return [], None
    unit_segments = _fit_segments_after_previous(
        unit_segments, previous_end_ms=segments[-1].end_ms if segments else None
    )
    segments.extend(unit_segments)
    if translation_units is not None:
        translation_units.append(unit_translation)
    return unit_segments, unit_translation


def _fit_segments_after_previous(
    unit_segments: Sequence[TranscriptSegment],
    *,
    previous_end_ms: int | None,
) -> list[TranscriptSegment]:
    fitted: list[TranscriptSegment] = []
    cursor = None if previous_end_ms is None else int(previous_end_ms)
    for segment in unit_segments:
        start_ms = segment.start_ms
        end_ms = segment.end_ms
        if start_ms is None or end_ms is None:
            fitted.append(segment)
            continue
        start = int(start_ms)
        end = max(start, int(end_ms))
        duration = max(_MIN_TRANSCRIPT_SEGMENT_MS, end - start)
        if cursor is not None and start < cursor:
            start = cursor
            end = start + duration
        elif end <= start:
            end = start + _MIN_TRANSCRIPT_SEGMENT_MS
        fitted_segment = replace(segment, start_ms=start, end_ms=end)
        fitted.append(fitted_segment)
        cursor = (
            int(fitted_segment.end_ms) if fitted_segment.end_ms is not None else cursor
        )
    return fitted


def _append_source_units(
    segments: list[TranscriptSegment],
    units: Sequence[SourceUnit],
    *,
    translation_units: list[OfflineTranslationUnit],
) -> None:
    for unit in units:
        _append_source_unit(segments, unit, translation_units=translation_units)


def _append_source_unit_events(
    segments: list[TranscriptSegment],
    unit: SourceUnit,
) -> list[OfflineTranscriptionStreamEvent]:
    unit_segments, unit_translation = _append_source_unit(segments, unit)
    events = [
        OfflineTranscriptionStreamEvent(kind="segment", segment=segment)
        for segment in unit_segments
    ]
    if unit_translation is not None:
        events.append(
            OfflineTranscriptionStreamEvent(
                kind="translation_unit", translation_unit=unit_translation
            )
        )
    return events


def _segments_for_source_unit(
    unit: SourceUnit,
    *,
    next_index: int,
    segment_list_start: int,
) -> tuple[list[TranscriptSegment], OfflineTranslationUnit]:
    cues = layout_source_cues(unit)
    segments: list[TranscriptSegment] = []
    for offset, cue in enumerate(cues):
        index = int(next_index) + offset
        segments.append(
            TranscriptSegment(
                id=f"seg_{index:06d}",
                index=index,
                start_ms=int(cue.start_ms),
                end_ms=max(int(cue.start_ms), int(cue.end_ms)),
                text=cue.text,
                language=unit.language,
                timing_status=unit.timing_status,
            )
        )
    translation = OfflineTranslationUnit(
        source_text=unit.text,
        source_language=unit.language,
        source_segment_ids=tuple(segment.id for segment in segments),
        source_segment_indices=tuple(int(segment.index) for segment in segments),
        anchor_segment_list_index=int(segment_list_start) + len(segments) - 1,
    )
    return segments, translation


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


def apply_unit_translation(
    segments: list[TranscriptSegment],
    unit: OfflineTranslationUnit,
    *,
    target_language: str,
    text: str | None,
    error: str | None = None,
) -> TranscriptTranslationUnit | None:
    """Write a source unit's translation onto its anchor segment in place.

    Returns a TranscriptTranslationUnit when grouped coverage must survive a
    final document rebuild. Successful single-cue and failed single-cue results
    fit on the anchor segment; grouped success/status needs a document unit so
    UI/export/SRT can keep the same projected coverage after finalization.
    """
    anchor_index = int(unit.anchor_segment_list_index)
    if not 0 <= anchor_index < len(segments):
        return None
    if not text or error is not None:
        segments[anchor_index] = replace(
            segments[anchor_index],
            translation=None,
            translation_status=str(error or "failed"),
            translation_message="translation failed",
        )
        if len(unit.source_segment_ids) <= 1 and len(unit.source_segment_indices) <= 1:
            return None
        return TranscriptTranslationUnit(
            text="",
            target_language=target_language,
            source_segment_ids=unit.source_segment_ids,
            source_segment_indices=unit.source_segment_indices,
            translation_status=str(error or "failed"),
            translation_message="translation failed",
        )
    segments[anchor_index] = replace(segments[anchor_index], translation=str(text))
    if len(unit.source_segment_ids) <= 1 and len(unit.source_segment_indices) <= 1:
        return None
    return TranscriptTranslationUnit(
        text=str(text),
        target_language=target_language,
        source_segment_ids=unit.source_segment_ids,
        source_segment_indices=unit.source_segment_indices,
    )


async def _translate_units(
    segments: list[TranscriptSegment],
    translation_units: list[OfflineTranslationUnit],
    *,
    translation_actor: Any,
    target_language: str,
    source_language: str,
    max_new_tokens: int | None,
) -> tuple[list[TranscriptSegment], list[TranscriptTranslationUnit]]:
    if not translation_units:
        return segments, []
    translated = list(segments)
    document_units: list[TranscriptTranslationUnit] = []
    units_by_language: dict[str, list[OfflineTranslationUnit]] = {}
    for unit in translation_units:
        if not 0 <= int(unit.anchor_segment_list_index) < len(translated):
            continue
        language = str(unit.source_language or source_language or "")
        units_by_language.setdefault(language, []).append(unit)

    for language, language_units in units_by_language.items():
        outputs = await translation_actor.translate_batch(
            [unit.source_text for unit in language_units],
            target_language=target_language,
            source_language=language,
            max_new_tokens=max_new_tokens,
            timeout_sec=None,
        )
        for offset, unit in enumerate(language_units):
            text, error = (
                outputs[offset]
                if offset < len(outputs)
                else (None, "missing translation output")
            )
            document_unit = apply_unit_translation(
                translated,
                unit,
                target_language=target_language,
                text=text,
                error=error,
            )
            if document_unit is not None:
                document_units.append(document_unit)
    document_units.sort(key=lambda unit: min(unit.source_segment_indices, default=0))
    return translated, document_units


def _forced_align_language(language: str) -> str | None:
    try:
        return normalize_forced_align_language(language)
    except ValueError:
        return None


__all__ = [
    "OfflineTranscriptionInputError",
    "OfflineTranscriptionOptions",
    "OfflineTranscriptionStreamEvent",
    "OfflineTranslationUnit",
    "apply_unit_translation",
    "stream_transcribe_file",
    "transcribe_file",
]
