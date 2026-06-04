# coding=utf-8
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import qwen3_asr_runtime.offline_transcription as offline_transcription
from qwen3_asr_runtime.offline_transcription import (
    OfflineTranscriptionOptions,
    stream_transcribe_file,
    transcribe_file,
)
from qwen3_asr_runtime.utils import SAMPLE_RATE


class FakeOfflineModel:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[dict[str, object]] = []

    def transcribe(self, *, audio: object, context: str, language: str | None, return_time_stamps: bool) -> list[object]:
        wav, sample_rate = audio  # type: ignore[misc]
        self.calls.append(
            {
                "samples": int(wav.shape[0]),
                "sample_rate": sample_rate,
                "context": context,
                "language": language,
                "return_time_stamps": return_time_stamps,
            }
        )
        text = self.texts.pop(0) if self.texts else ""
        return [SimpleNamespace(language=language or "Chinese", text=text)]


class FakeTimestampActor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def align_segment(
        self,
        audio: np.ndarray,
        *,
        text: str,
        language: str,
        timeout_sec: float | None,
    ) -> tuple[float | None, float | None, str | None]:
        self.calls.append(
            {
                "samples": int(audio.shape[0]),
                "text": text,
                "language": language,
                "timeout_sec": timeout_sec,
            }
        )
        return 0.1, 0.8, None


class FakeTranslationActor:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    async def translate_batch(
        self,
        texts: list[str],
        *,
        target_language: str,
        source_language: str,
        max_new_tokens: int | None,
        timeout_sec: float | None,
    ) -> list[tuple[str | None, str | None]]:
        self.calls.append(
            {
                "texts": texts,
                "target_language": target_language,
                "source_language": source_language,
                "max_new_tokens": max_new_tokens,
                "timeout_sec": timeout_sec,
            }
        )
        return [(self.outputs.pop(0), None) for _ in texts]


@pytest.mark.asyncio
async def test_transcribe_file_returns_document_with_alignment_and_translation() -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    wav[int(SAMPLE_RATE * 0.95) : int(SAMPLE_RATE * 1.05)] = 0.0
    model = FakeOfflineModel(["第一句", "第二句"])
    timestamps = FakeTimestampActor()
    translation = FakeTranslationActor(["first", "second"])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(
            language="Chinese",
            context="术语",
            target_language="English",
        ),
        timestamp_actor=timestamps,
        translation_actor=translation,
        translation_max_new_tokens=128,
        chunk_sec=1.0,
    )

    assert document.duration_ms == 2000
    assert document.language == "Chinese"
    assert [(segment.text, segment.translation) for segment in document.segments] == [
        ("第一句", "first"),
        ("第二句", "second"),
    ]
    assert [(segment.start_ms, segment.end_ms, segment.timing_status) for segment in document.segments] == [
        (100, 800, "aligned"),
        (1050, 1750, "aligned"),
    ]
    assert model.calls[0]["context"] == "术语"
    assert len(timestamps.calls) == 2
    assert translation.calls == [
        {
            "texts": ["第一句", "第二句"],
            "target_language": "English",
            "source_language": "Chinese",
            "max_new_tokens": 128,
            "timeout_sec": None,
        }
    ]


@pytest.mark.asyncio
async def test_transcribe_file_streams_local_file_path(monkeypatch, tmp_path) -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, wav, SAMPLE_RATE)
    model = FakeOfflineModel(["第一句", "第二句"])
    timestamps = FakeTimestampActor()

    def fail_normalize_audios(_audio: object) -> list[np.ndarray]:
        raise AssertionError("file path transcription should not load the full audio at once")

    monkeypatch.setattr(offline_transcription, "normalize_audios", fail_normalize_audios)

    document = await transcribe_file(
        model,
        str(audio_path),
        options=OfflineTranscriptionOptions(language="Chinese"),
        timestamp_actor=timestamps,
        chunk_sec=1.0,
    )

    assert document.duration_ms == 2000
    assert [segment.text for segment in document.segments] == ["第一句", "第二句"]
    assert [(segment.start_ms, segment.end_ms, segment.timing_status) for segment in document.segments] == [
        (100, 800, "aligned"),
        (1100, 1800, "aligned"),
    ]


@pytest.mark.asyncio
async def test_stream_transcribe_file_yields_segments_before_final_document() -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["第一句", "第二句"])
    timestamps = FakeTimestampActor()

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            (wav, SAMPLE_RATE),
            options=OfflineTranscriptionOptions(
                language="Chinese",
            ),
            timestamp_actor=timestamps,
            chunk_sec=1.0,
        )
    ]

    assert [event.kind for event in events] == ["segment", "segment", "complete"]
    assert [event.segment.text for event in events if event.segment is not None] == ["第一句", "第二句"]
    assert [event.segment.translation for event in events if event.segment is not None] == [None, None]
    assert events[-1].document is not None
    assert events[-1].document.text == "第一句第二句"
    assert [segment.translation for segment in events[-1].document.segments] == [None, None]


@pytest.mark.asyncio
async def test_stream_transcribe_file_rejects_translation_options() -> None:
    wav = np.ones((int(SAMPLE_RATE * 1.0),), dtype=np.float32) * 0.1
    events = stream_transcribe_file(
        FakeOfflineModel(["第一句"]),
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese", target_language="English"),
        chunk_sec=1.0,
    )

    with pytest.raises(ValueError, match="does not support translation"):
        await anext(events)
