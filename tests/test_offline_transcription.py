# coding=utf-8
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import qwen3_asr_runtime.offline_transcription as offline_transcription
from qwen3_asr_runtime.offline_units import (
    SourceUnit,
    SourceUnitBuilder,
    TimedToken,
    estimated_timed_tokens_from_text,
    layout_source_cues,
    timed_tokens_from_aligned_items,
)
from qwen3_asr_runtime.offline_transcription import (
    OfflineTranscriptionOptions,
    stream_transcribe_file,
    transcribe_file,
)
from qwen3_asr_runtime.transcription_document import (
    TranscriptSegment,
    TranscriptTranslationUnit,
)
from qwen3_asr_runtime.utils import SAMPLE_RATE


class FakeOfflineModel:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        *,
        audio: object,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> list[object]:
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


class FakeLanguageResultModel:
    def __init__(self, results: list[tuple[str, str]]) -> None:
        self.results = list(results)

    def transcribe(
        self,
        *,
        audio: object,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> list[object]:
        del audio, context, language, return_time_stamps
        if not self.results:
            return [SimpleNamespace(language="", text="")]
        text, detected_language = self.results.pop(0)
        return [SimpleNamespace(language=detected_language, text=text)]


class FakeItemTimestampActor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def align_items(
        self,
        audio: np.ndarray,
        *,
        text: str,
        language: str,
        timeout_sec: float | None,
    ) -> tuple[object | None, str | None]:
        self.calls.append(
            {
                "samples": int(audio.shape[0]),
                "text": text,
                "language": language,
                "timeout_sec": timeout_sec,
            }
        )
        return SimpleNamespace(items=_aligned_items_for_text(text, audio)), None


class FakeMixedTimestampActor:
    def __init__(
        self,
        *,
        fail_item_calls: set[int] | None = None,
        invalid_item_calls: set[int] | None = None,
    ) -> None:
        self.fail_item_calls = fail_item_calls or set()
        self.invalid_item_calls = invalid_item_calls or set()
        self.calls: list[dict[str, object]] = []

    async def align_items(
        self,
        audio: np.ndarray,
        *,
        text: str,
        language: str,
        timeout_sec: float | None,
    ) -> tuple[object | None, str | None]:
        call_index = len(self.calls) + 1
        self.calls.append(
            {
                "samples": int(audio.shape[0]),
                "text": text,
                "language": language,
                "timeout_sec": timeout_sec,
            }
        )
        if call_index in self.fail_item_calls:
            return None, "failed"
        if call_index in self.invalid_item_calls:
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        text=text[:1], start_time=float("nan"), end_time=1.0
                    ),
                    SimpleNamespace(text=text[1:2], start_time=0.1, end_time=0.2),
                ]
            ), None
        return SimpleNamespace(items=_aligned_items_for_text(text, audio)), None


def _aligned_items_for_text(text: str, audio: np.ndarray) -> list[object]:
    kept_chars = [ch for ch in text if ch not in "，。！？,.!? "]
    step = float(audio.shape[0]) / float(SAMPLE_RATE) / max(1, len(kept_chars))
    items = []
    clock = 0.0
    for ch in text:
        if ch in "，。！？,.!? ":
            continue
        items.append(SimpleNamespace(text=ch, start_time=clock, end_time=clock + step))
        clock += step
    return items


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


def test_timed_tokens_from_aligned_items_rejects_non_monotonic_times() -> None:
    tokens = timed_tokens_from_aligned_items(
        "ab",
        [
            SimpleNamespace(text="a", start_time=0.5, end_time=1.0),
            SimpleNamespace(text="b", start_time=0.8, end_time=1.2),
        ],
        duration_ms=2000,
    )

    assert tokens == []


def test_layout_source_cues_expands_zero_duration_cues() -> None:
    unit = SourceUnit(
        text="对。",
        language="Chinese",
        timing_status="aligned",
        tokens=(TimedToken("对。", 1000, 1000),),
    )

    assert [
        (cue.text, cue.start_ms, cue.end_ms) for cue in layout_source_cues(unit)
    ] == [("对。", 920, 1000)]


def test_append_source_unit_moves_short_cue_after_existing_segment() -> None:
    segments = [
        TranscriptSegment(
            id="seg_000001",
            index=1,
            start_ms=1000,
            end_ms=2000,
            text="前一句。",
            language="Chinese",
            timing_status="aligned",
        )
    ]
    unit = SourceUnit(
        text="对。",
        language="Chinese",
        timing_status="aligned",
        tokens=(TimedToken("对。", 2000, 2000),),
    )

    appended, _ = offline_transcription._append_source_unit(segments, unit)

    assert [
        (segment.text, segment.start_ms, segment.end_ms) for segment in appended
    ] == [("对。", 2000, 2080)]
    assert [
        (segment.text, segment.start_ms, segment.end_ms) for segment in segments
    ] == [
        ("前一句。", 1000, 2000),
        ("对。", 2000, 2080),
    ]


def test_source_unit_builder_preserves_ascii_word_boundary_across_chunks() -> None:
    builder = SourceUnitBuilder()

    assert builder.add_tokens([TimedToken("hello", 0, 500)], language="English") == []
    units = builder.add_tokens([TimedToken("world.", 500, 1000)], language="English")

    assert [(unit.text, unit.language) for unit in units] == [
        ("hello world.", "English")
    ]
    assert [cue.text for cue in layout_source_cues(units[0])] == ["hello world."]
    assert builder.flush() == []


def test_source_unit_builder_keeps_true_repeated_text_across_batches() -> None:
    builder = SourceUnitBuilder()

    assert builder.add_tokens([TimedToken("谢谢", 0, 500)], language="Chinese") == []
    assert builder.add_tokens([TimedToken("谢谢", 700, 1200)], language="Chinese") == []

    assert [unit.text for unit in builder.flush()] == ["谢谢谢谢"]


def test_source_unit_builder_uses_ascii_period_as_sentence_boundary() -> None:
    builder = SourceUnitBuilder()
    tokens = estimated_timed_tokens_from_text(
        "Hello world. Next sentence.", base_ms=0, duration_ms=4000
    )

    units = builder.add_tokens(tokens, language="English") + builder.flush()

    assert [unit.text for unit in units] == ["Hello world.", "Next sentence."]


def test_source_unit_builder_uses_attached_alpha_period_as_sentence_boundary() -> None:
    builder = SourceUnitBuilder()
    tokens = estimated_timed_tokens_from_text(
        "Hello.Next sentence.", base_ms=0, duration_ms=3000
    )

    units = builder.add_tokens(tokens, language="English") + builder.flush()

    assert [unit.text for unit in units] == ["Hello.", "Next sentence."]


def test_source_unit_builder_uses_aligned_ascii_period_as_sentence_boundary() -> None:
    tokens = timed_tokens_from_aligned_items(
        "Hello world. Next sentence.",
        [
            SimpleNamespace(text="Hello", start_time=0.0, end_time=0.5),
            SimpleNamespace(text="world", start_time=0.5, end_time=1.0),
            SimpleNamespace(text="Next", start_time=1.0, end_time=1.5),
            SimpleNamespace(text="sentence", start_time=1.5, end_time=2.0),
        ],
        base_ms=0,
        duration_ms=4000,
    )
    builder = SourceUnitBuilder()

    units = builder.add_tokens(tokens, language="English") + builder.flush()

    assert [unit.text for unit in units] == ["Hello world.", "Next sentence."]


def test_source_unit_builder_keeps_attached_decimal_period_inside_sentence() -> None:
    builder = SourceUnitBuilder()
    tokens = estimated_timed_tokens_from_text(
        "Pi is 3.14. Next sentence.", base_ms=0, duration_ms=4000
    )

    units = builder.add_tokens(tokens, language="English") + builder.flush()

    assert [unit.text for unit in units] == ["Pi is 3.14.", "Next sentence."]


def test_source_unit_builder_keeps_aligned_decimal_period_inside_sentence() -> None:
    tokens = timed_tokens_from_aligned_items(
        "Pi is 3.14. Next sentence.",
        [
            SimpleNamespace(text="Pi", start_time=0.0, end_time=0.3),
            SimpleNamespace(text="is", start_time=0.3, end_time=0.6),
            SimpleNamespace(text="3", start_time=0.6, end_time=0.9),
            SimpleNamespace(text="14", start_time=0.9, end_time=1.2),
            SimpleNamespace(text="Next", start_time=1.2, end_time=1.5),
            SimpleNamespace(text="sentence", start_time=1.5, end_time=1.8),
        ],
        base_ms=0,
        duration_ms=4000,
    )
    builder = SourceUnitBuilder()

    units = builder.add_tokens(tokens, language="English") + builder.flush()

    assert [unit.text for unit in units] == ["Pi is 3.14.", "Next sentence."]


def test_source_unit_builder_keeps_long_sentence_as_one_translation_unit() -> None:
    text = "今天讨论字幕显示问题，并且保持翻译输入完整，还要避免把一句普通话切成半截，只有真正太长的时候才拆开。"
    tokens = [
        TimedToken(char, index * 220, (index + 1) * 220)
        for index, char in enumerate(text)
    ]
    builder = SourceUnitBuilder()

    units = builder.add_tokens(tokens, language="Chinese") + builder.flush()

    assert [unit.text for unit in units] == [text]
    assert [cue.text for cue in layout_source_cues(units[0])] == [
        "今天讨论字幕显示问题，",
        "并且保持翻译输入完整，",
        "还要避免把一句普通话切成半截，",
        "只有真正太长的时候才拆开。",
    ]


def test_layout_estimated_cues_are_bounded_and_sequential_without_chunk_gaps() -> None:
    text = (
        "今天讨论字幕显示问题，并且保持翻译输入完整，还要避免一个短展示块覆盖整段音频。"
    )
    tokens = estimated_timed_tokens_from_text(text, base_ms=0, duration_ms=60_000)
    unit = SourceUnit(
        text=text, language="Chinese", timing_status="estimated", tokens=tuple(tokens)
    )

    cues = layout_source_cues(unit, max_cue_ms=6000, max_cue_width=24)

    assert len(cues) > 1
    assert all(cue.end_ms - cue.start_ms <= 6000 for cue in cues)
    assert [cue.start_ms for cue in cues[1:]] == [cue.end_ms for cue in cues[:-1]]


def test_source_unit_builder_closes_estimated_long_unpunctuated_unit_before_final_flush() -> (
    None
):
    builder = SourceUnitBuilder()
    text = " ".join(f"word{index}" for index in range(1, 22))
    tokens = estimated_timed_tokens_from_text(text, base_ms=0, duration_ms=21_000)

    units = builder.add_tokens(tokens, language="English", timing_status="estimated")

    assert len(units) == 1
    assert units[0].timing_status == "estimated"
    assert units[0].text.startswith("word1 word2")
    assert "word21" not in units[0].text
    assert builder.flush()[0].text.endswith("word21")


def test_source_unit_builder_bounds_oversized_sentence_before_late_punctuation() -> (
    None
):
    builder = SourceUnitBuilder(max_unit_ms=3000)
    tokens = estimated_timed_tokens_from_text(
        "one two three four five.", base_ms=0, duration_ms=5000
    )

    units = builder.add_tokens(tokens, language="English")

    assert len(units) == 2
    assert [unit.text for unit in units] != ["one two three four five."]
    assert units[-1].text.endswith("five.")
    assert builder.flush() == []


@pytest.mark.parametrize("chunk_sec", [0.0, -1.0, float("nan"), float("inf"), "bad"])
def test_iter_source_audio_chunks_rejects_invalid_chunk_sec_before_chunking(
    chunk_sec: object,
) -> None:
    with pytest.raises(ValueError, match="chunk_sec must be a finite positive number"):
        list(
            offline_transcription._iter_source_audio_chunks(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                chunk_sec=chunk_sec,
            )
        )


def test_iter_source_audio_chunks_uses_fixed_windows(tmp_path) -> None:
    wav = np.ones((int(SAMPLE_RATE * 125.0),), dtype=np.float32) * 0.1
    wav[int(SAMPLE_RATE * 61.0) : int(SAMPLE_RATE * 61.5)] = 0.0
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, wav, SAMPLE_RATE)

    chunks = list(
        offline_transcription._iter_source_audio_chunks(str(audio_path), chunk_sec=60.0)
    )

    assert [chunk[2] for chunk in chunks] == [
        60 * SAMPLE_RATE,
        60 * SAMPLE_RATE,
        5 * SAMPLE_RATE,
    ]


def test_iter_source_audio_chunks_streams_unreadable_media_through_ffmpeg(
    monkeypatch, tmp_path
) -> None:
    # A file libsndfile cannot open (stand-in for a video container).
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"\x00\x01\x02 not real media bytes")
    decoded = np.ones((int(SAMPLE_RATE * 1.5),), dtype=np.float32) * 0.1

    def fake_blocks(path, **_kwargs):
        assert str(path) == str(media_path)
        # Split into two blocks so the windower exercises its buffering.
        yield decoded[:SAMPLE_RATE]
        yield decoded[SAMPLE_RATE:]

    monkeypatch.setattr(offline_transcription, "_iter_ffmpeg_pcm_blocks", fake_blocks)

    chunks = list(
        offline_transcription._iter_source_audio_chunks(str(media_path), chunk_sec=1.0)
    )

    assert [chunk[2] for chunk in chunks] == [SAMPLE_RATE, int(SAMPLE_RATE * 0.5)]
    assert chunks[-1][3] is True
    assert chunks[-1][4] == int(
        SAMPLE_RATE * 1.5
    )  # cumulative total reaches the true length at EOF


def test_take_boundary_provisional_unit_requires_pending_boundary_nearness() -> None:
    builder = SourceUnitBuilder()
    builder.add_tokens(
        [
            TimedToken("还", 0, 500),
            TimedToken("没", 500, 1000),
        ],
        language="Chinese",
        timing_status="aligned",
    )

    provisional = offline_transcription._take_boundary_provisional_unit(
        builder,
        [],
        boundary_ms=30_000,
        hold_ms=3_000,
        max_refeed_ms=30_000,
    )

    assert provisional is None
    assert "".join(token.text for token in builder.pending_tokens) == "还没"


def test_iter_ffmpeg_pcm_blocks_requires_ffmpeg_on_path(monkeypatch, tmp_path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"not real media")
    monkeypatch.setattr(offline_transcription.shutil, "which", lambda _name: None)

    with pytest.raises(
        offline_transcription.OfflineTranscriptionInputError, match="ffmpeg is required"
    ):
        list(offline_transcription._iter_ffmpeg_pcm_blocks(media_path))


def test_iter_ffmpeg_pcm_blocks_rejects_undecodable_media(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for this test")
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"\x00\x01\x02 not real media bytes")

    with pytest.raises(
        offline_transcription.OfflineTranscriptionInputError, match="media file"
    ):
        list(offline_transcription._iter_ffmpeg_pcm_blocks(media_path))


@pytest.mark.asyncio
async def test_stream_transcribe_file_decodes_real_video_with_ffmpeg(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for this test")
    video_path = tmp_path / "clip.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=128x96:rate=15:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-shortest",
                "-c:v",
                "mpeg4",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(video_path),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg cannot encode the test video in this environment")

    model = FakeOfflineModel(["你好。", "世界。"])
    events = [
        event
        async for event in stream_transcribe_file(
            model,
            str(video_path),
            options=OfflineTranscriptionOptions(language="Chinese"),
            chunk_sec=1.0,
        )
    ]

    segments = [
        event.segment
        for event in events
        if event.kind == "segment" and event.segment is not None
    ]
    assert segments, "expected the video audio track to decode into transcript segments"
    assert any(event.kind == "complete" for event in events)
    assert model.calls and all(
        call["sample_rate"] == SAMPLE_RATE for call in model.calls
    )


@pytest.mark.asyncio
async def test_transcribe_file_redecodes_provisional_tail_before_emitting() -> None:
    wav = np.ones((int(SAMPLE_RATE * 61.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(
        ["你知道，对一个十八岁。", "你知道，对一个十八岁的小男生，目标有了。", ""]
    )

    class TailTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del audio, language, timeout_sec
            clock = 29.2 if text == "你知道，对一个十八岁。" else 1.0
            items = []
            for char in text:
                if char in "，。！？,.!? ":
                    continue
                items.append(
                    SimpleNamespace(text=char, start_time=clock, end_time=clock + 0.05)
                )
                clock += 0.05
            return SimpleNamespace(items=items), None

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese"),
        timestamp_actor=TailTimestampActor(),
        chunk_sec=30.0,
    )

    assert [segment.text for segment in document.segments] == [
        "你知道，对一个十八岁的小男生，目标有了。"
    ]
    assert len(model.calls) == 3


@pytest.mark.asyncio
async def test_transcribe_file_backfills_aligned_provisional_tail_when_next_window_does_not_replace_it() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 61.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["边界句。", "后续句。", ""])

    class BoundaryTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del audio, language, timeout_sec
            clock = 29.2 if text == "边界句。" else 1.5
            items = []
            for char in text:
                if char in "，。！？,.!? ":
                    continue
                items.append(
                    SimpleNamespace(text=char, start_time=clock, end_time=clock + 0.05)
                )
                clock += 0.05
            return SimpleNamespace(items=items), None

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese"),
        timestamp_actor=BoundaryTimestampActor(),
        chunk_sec=30.0,
    )

    assert [segment.text for segment in document.segments] == ["边界句。", "后续句。"]


@pytest.mark.asyncio
async def test_transcribe_file_retries_main_window_when_refeed_decode_is_empty() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 61.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["边界句。", "", "当前句。", ""])

    class BoundaryTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del audio, language, timeout_sec
            clock = 29.2 if text == "边界句。" else 0.5
            items = []
            for char in text:
                if char in "，。！？,.!? ":
                    continue
                items.append(
                    SimpleNamespace(text=char, start_time=clock, end_time=clock + 0.05)
                )
                clock += 0.05
            return SimpleNamespace(items=items), None

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese"),
        timestamp_actor=BoundaryTimestampActor(),
        chunk_sec=30.0,
    )

    assert [segment.text for segment in document.segments] == ["边界句。", "当前句。"]


@pytest.mark.asyncio
async def test_transcribe_file_replaces_provisional_pending_prefix() -> None:
    wav = np.ones((int(SAMPLE_RATE * 58.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["今天讨论字幕显", "今天讨论字幕显示问题。"])

    class OverlapTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del audio, language, timeout_sec
            clock = 28.5 if text == "今天讨论字幕显" else 0.0
            items = []
            for char in text:
                if char in "，。！？,.!? ":
                    continue
                items.append(
                    SimpleNamespace(text=char, start_time=clock, end_time=clock + 0.2)
                )
                clock += 0.2
            return SimpleNamespace(items=items), None

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese"),
        timestamp_actor=OverlapTimestampActor(),
        chunk_sec=30.0,
    )

    assert document.text == "今天讨论字幕显示问题。"
    assert [segment.text for segment in document.segments] == ["今天讨论字幕显示问题。"]


@pytest.mark.asyncio
async def test_transcribe_file_keeps_estimated_prefix_text_without_timestamps() -> None:
    wav = np.ones((int(SAMPLE_RATE * 57.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["今天讨论字幕显", "今天讨论字幕显示问题。"])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese", timestamps=False),
        chunk_sec=30.0,
    )

    assert document.text == "今天讨论字幕显今天讨论字幕显示问题。"
    assert [segment.text for segment in document.segments] == [
        "今天讨论字幕显今天讨论字幕显示问题。"
    ]
    assert [segment.timing_status for segment in document.segments] == ["estimated"]


@pytest.mark.asyncio
async def test_transcribe_file_keeps_specific_estimated_repeat_after_previous_unit_flushed() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 61.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["前段结束。重叠句子。", "重叠句子。后续内容。"])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese", timestamps=False),
        chunk_sec=60.0,
    )

    assert document.text == "前段结束。重叠句子。重叠句子。后续内容。"
    assert [segment.text for segment in document.segments] == [
        "前段结束。",
        "重叠句子。",
        "重叠句子。",
        "后续内容。",
    ]
    assert [segment.timing_status for segment in document.segments] == [
        "estimated",
        "estimated",
        "estimated",
        "estimated",
    ]


@pytest.mark.asyncio
async def test_transcribe_file_keeps_short_true_repeated_estimated_prefix() -> None:
    wav = np.ones((int(SAMPLE_RATE * 61.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["谢谢", "谢谢大家。"])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese", timestamps=False),
        chunk_sec=60.0,
    )

    assert document.text == "谢谢谢谢大家。"
    assert [segment.text for segment in document.segments] == ["谢谢谢谢大家。"]
    assert [segment.timing_status for segment in document.segments] == ["estimated"]


@pytest.mark.asyncio
async def test_transcribe_file_bounds_estimated_short_cue_duration_without_timestamps() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 120.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["短句。"])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese", timestamps=False),
        chunk_sec=120.0,
    )

    assert [segment.text for segment in document.segments] == ["短句。"]
    assert [segment.timing_status for segment in document.segments] == ["estimated"]
    assert (
        max(segment.end_ms - segment.start_ms for segment in document.segments) <= 6000
    )


@pytest.mark.asyncio
async def test_transcribe_file_keeps_estimated_tail_text_not_repeated_by_next_window() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 61.0),), dtype=np.float32) * 0.1
    text = "一二三四五六七八九十十一十二十三十四十五十六十七十八十九二十。"
    model = FakeOfflineModel([text, ""])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="Chinese", timestamps=False),
        chunk_sec=60.0,
    )

    assert document.text == text
    assert document.segments
    assert all(segment.timing_status == "estimated" for segment in document.segments)


@pytest.mark.asyncio
async def test_transcribe_file_does_not_redecode_final_window_trailing_silence() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 1.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["hello.", "tail hallucination."])

    class ShortTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del audio, language, timeout_sec
            return SimpleNamespace(
                items=[
                    SimpleNamespace(text=text.rstrip("."), start_time=0.0, end_time=0.2)
                ]
            ), None

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(language="English"),
        timestamp_actor=ShortTimestampActor(),
        chunk_sec=30.0,
    )

    assert [segment.text for segment in document.segments] == ["hello."]
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_transcribe_file_returns_document_with_alignment_and_translation() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    wav[int(SAMPLE_RATE * 0.95) : int(SAMPLE_RATE * 1.05)] = 0.0
    model = FakeOfflineModel(["第一句。", "第二句。"])
    timestamps = FakeItemTimestampActor()
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
        ("第一句。", "first"),
        ("第二句。", "second"),
    ]
    assert [segment.timing_status for segment in document.segments] == [
        "aligned",
        "aligned",
    ]
    assert all(
        segment.start_ms is not None
        and segment.end_ms is not None
        and 0 <= segment.start_ms < segment.end_ms <= document.duration_ms
        for segment in document.segments
    )
    assert model.calls[0]["context"] == "术语"
    assert len(timestamps.calls) == 2
    assert translation.calls == [
        {
            "texts": ["第一句。", "第二句。"],
            "target_language": "English",
            "source_language": "Chinese",
            "max_new_tokens": 128,
            "timeout_sec": None,
        }
    ]


@pytest.mark.asyncio
async def test_transcribe_file_streams_local_file_without_full_decode(
    monkeypatch, tmp_path
) -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    wav[int(SAMPLE_RATE * 0.95) : int(SAMPLE_RATE * 1.05)] = 0.0
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, wav, SAMPLE_RATE)
    model = FakeOfflineModel(["第一句。", "第二句。"])
    timestamps = FakeItemTimestampActor()

    def fail_normalize_audios(_audio: object) -> list[np.ndarray]:
        raise AssertionError(
            "local file transcription should not decode the whole file before chunking"
        )

    monkeypatch.setattr(
        offline_transcription, "normalize_audios", fail_normalize_audios
    )
    monkeypatch.setattr(
        offline_transcription.librosa,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "local file transcription should stream windows instead of offset-loading each chunk"
            )
        ),
    )

    document = await transcribe_file(
        model,
        str(audio_path),
        options=OfflineTranscriptionOptions(language="Chinese"),
        timestamp_actor=timestamps,
        chunk_sec=1.0,
    )

    assert document.duration_ms == 2000
    assert [segment.text for segment in document.segments] == ["第一句。", "第二句。"]
    assert [segment.timing_status for segment in document.segments] == [
        "aligned",
        "aligned",
    ]
    assert all(
        segment.start_ms is not None
        and segment.end_ms is not None
        and 0 <= segment.start_ms < segment.end_ms <= document.duration_ms
        for segment in document.segments
    )


@pytest.mark.asyncio
async def test_transcribe_file_translates_aligned_and_estimated_units() -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["第一句。", "第二句"])
    timestamps = FakeMixedTimestampActor(invalid_item_calls={2})
    translation = FakeTranslationActor(["first sentence", "second sentence"])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(
            language="Chinese", target_language="English"
        ),
        timestamp_actor=timestamps,
        translation_actor=translation,
        chunk_sec=1.0,
    )

    assert [(segment.text, segment.translation) for segment in document.segments] == [
        ("第一句。", "first sentence"),
        ("第二句", "second sentence"),
    ]
    assert translation.calls == [
        {
            "texts": ["第一句。", "第二句"],
            "target_language": "English",
            "source_language": "Chinese",
            "max_new_tokens": None,
            "timeout_sec": None,
        }
    ]


@pytest.mark.asyncio
async def test_translate_units_records_failures_on_anchor_segments() -> None:
    segments = [
        TranscriptSegment(
            id="seg_000001",
            index=1,
            start_ms=0,
            end_ms=1000,
            text="第一句",
            language="Chinese",
        )
    ]
    translation_units = [
        offline_transcription.OfflineTranslationUnit(
            source_text="第一句",
            source_language="Chinese",
            source_segment_ids=("seg_000001",),
            source_segment_indices=(1,),
            anchor_segment_list_index=0,
        )
    ]

    class FailingTranslationActor:
        async def translate_batch(
            self,
            texts: list[str],
            *,
            target_language: str,
            source_language: str,
            max_new_tokens: int | None,
            timeout_sec: float | None,
        ) -> list[tuple[str | None, str | None]]:
            del texts, target_language, source_language, max_new_tokens, timeout_sec
            return [(None, "timeout")]

    translated, translated_units = await offline_transcription._translate_units(
        segments,
        translation_units,
        translation_actor=FailingTranslationActor(),
        target_language="English",
        source_language="Chinese",
        max_new_tokens=None,
    )

    assert translated_units == []
    assert translated[0].translation is None
    assert translated[0].translation_status == "timeout"
    assert translated[0].translation_message == "translation failed"


@pytest.mark.asyncio
async def test_translate_units_returns_document_units_in_source_order_for_mixed_languages() -> (
    None
):
    segments = [
        TranscriptSegment(
            id=f"seg_{index:06d}",
            index=index,
            start_ms=(index - 1) * 1000,
            end_ms=index * 1000,
            text=f"segment {index}",
            language="English" if index in {1, 2, 5, 6} else "Chinese",
        )
        for index in range(1, 7)
    ]
    translation_units = [
        offline_transcription.OfflineTranslationUnit(
            source_text="first English unit",
            source_language="English",
            source_segment_ids=("seg_000001", "seg_000002"),
            source_segment_indices=(1, 2),
            anchor_segment_list_index=1,
        ),
        offline_transcription.OfflineTranslationUnit(
            source_text="中文单元",
            source_language="Chinese",
            source_segment_ids=("seg_000003", "seg_000004"),
            source_segment_indices=(3, 4),
            anchor_segment_list_index=3,
        ),
        offline_transcription.OfflineTranslationUnit(
            source_text="second English unit",
            source_language="English",
            source_segment_ids=("seg_000005", "seg_000006"),
            source_segment_indices=(5, 6),
            anchor_segment_list_index=5,
        ),
    ]

    _, translated_units = await offline_transcription._translate_units(
        segments,
        translation_units,
        translation_actor=FakeTranslationActor(["first", "second", "middle"]),
        target_language="French",
        source_language="",
        max_new_tokens=None,
    )

    assert [unit.text for unit in translated_units] == ["first", "middle", "second"]
    assert [unit.source_segment_indices for unit in translated_units] == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]


@pytest.mark.asyncio
async def test_transcribe_file_uses_unit_language_for_mixed_language_translation() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    model = FakeLanguageResultModel([("hello.", "English"), ("第二句。", "Chinese")])
    timestamps = FakeMixedTimestampActor()
    translation = FakeTranslationActor(["bonjour", "second sentence"])

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(target_language="French"),
        timestamp_actor=timestamps,
        translation_actor=translation,
        chunk_sec=1.0,
    )

    assert [
        (segment.text, segment.language, segment.translation)
        for segment in document.segments
    ] == [
        ("hello.", "English", "bonjour"),
        ("第二句。", "Chinese", "second sentence"),
    ]
    assert [
        (call["texts"], call["source_language"], call["target_language"])
        for call in translation.calls
    ] == [
        (["hello."], "English", "French"),
        (["第二句。"], "Chinese", "French"),
    ]


@pytest.mark.asyncio
async def test_transcribe_file_exposes_grouped_translation_units_in_document() -> None:
    wav = np.ones((int(SAMPLE_RATE * 9.0),), dtype=np.float32) * 0.1
    text = "今天讨论字幕显示问题，并且保持翻译输入完整，还要避免把一句普通话切成半截，只有真正太长的时候才拆开。"
    model = FakeOfflineModel([text])
    timestamps = FakeItemTimestampActor()
    translation = FakeTranslationActor(
        ["We discuss subtitle display while preserving translation context."]
    )

    document = await transcribe_file(
        model,
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(
            language="Chinese", target_language="English"
        ),
        timestamp_actor=timestamps,
        translation_actor=translation,
        chunk_sec=9.0,
    )

    assert len(document.segments) > 1
    assert document.text == text
    assert [segment.translation for segment in document.segments[:-1]] == [None] * (
        len(document.segments) - 1
    )
    assert (
        document.segments[-1].translation
        == "We discuss subtitle display while preserving translation context."
    )
    assert document.translation_units == [
        TranscriptTranslationUnit(
            text="We discuss subtitle display while preserving translation context.",
            target_language="English",
            source_segment_ids=tuple(segment.id for segment in document.segments),
            source_segment_indices=tuple(
                segment.index for segment in document.segments
            ),
        )
    ]
    assert document.to_payload()["translationUnits"] == [
        {
            "text": "We discuss subtitle display while preserving translation context.",
            "targetLanguage": "English",
            "sourceSegmentIds": [segment.id for segment in document.segments],
            "sourceSegmentIndices": [segment.index for segment in document.segments],
        }
    ]


@pytest.mark.asyncio
async def test_stream_transcribe_file_yields_segments_before_final_document() -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["第一句。", "第二句。"])
    timestamps = FakeItemTimestampActor()

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

    assert [event.kind for event in events] == [
        "segment",
        "translation_unit",
        "segment",
        "translation_unit",
        "complete",
    ]
    assert [event.segment.text for event in events if event.segment is not None] == [
        "第一句。",
        "第二句。",
    ]
    assert [
        event.segment.translation for event in events if event.segment is not None
    ] == [None, None]
    assert events[-1].document is not None
    assert events[-1].document.text == "第一句。第二句。"
    assert [segment.translation for segment in events[-1].document.segments] == [
        None,
        None,
    ]


@pytest.mark.asyncio
async def test_stream_transcribe_file_falls_back_when_item_timing_is_invalid() -> None:
    wav = np.ones((int(SAMPLE_RATE * 1.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["第一句"])
    timestamps = FakeMixedTimestampActor(invalid_item_calls={1})

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            (wav, SAMPLE_RATE),
            options=OfflineTranscriptionOptions(language="Chinese"),
            timestamp_actor=timestamps,
            chunk_sec=1.0,
        )
    ]

    segments = [event.segment for event in events if event.kind == "segment"]
    assert [event.kind for event in events] == [
        "segment",
        "translation_unit",
        "complete",
    ]
    assert [
        (segment.text, segment.start_ms, segment.end_ms, segment.timing_status)
        for segment in segments
        if segment
    ] == [("第一句", 0, 1000, "estimated")]


@pytest.mark.asyncio
async def test_stream_transcribe_file_keeps_complete_estimated_sentence_when_alignment_is_invalid() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 4.0),), dtype=np.float32) * 0.1
    text = "今天讨论字幕显示问题，并且保持翻译输入完整。"
    model = FakeOfflineModel([text])

    class ItemOnlyTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del audio, text, language, timeout_sec
            return SimpleNamespace(
                items=[
                    SimpleNamespace(text="今", start_time=float("nan"), end_time=1.0),
                    SimpleNamespace(text="天", start_time=0.1, end_time=0.2),
                ]
            ), None

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            (wav, SAMPLE_RATE),
            options=OfflineTranscriptionOptions(language="Chinese"),
            timestamp_actor=ItemOnlyTimestampActor(),
            chunk_sec=4.0,
        )
    ]

    segments = [event.segment for event in events if event.kind == "segment"]
    assert [event.kind for event in events] == [
        "segment",
        "translation_unit",
        "complete",
    ]
    assert [
        (segment.text, segment.start_ms, segment.end_ms, segment.timing_status)
        for segment in segments
        if segment
    ] == [
        ("今天讨论字幕显示问题，并且保持翻译输入完整。", 0, 4000, "estimated"),
    ]


@pytest.mark.asyncio
async def test_stream_transcribe_file_flushes_pending_unit_with_original_language_on_fallback() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    model = FakeLanguageResultModel([("hello", "English"), ("第二句", "Chinese")])

    class LanguageSwitchTimestampActor(FakeMixedTimestampActor):
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            if language == "English":
                self.calls.append(
                    {"samples": int(audio.shape[0]), "text": text, "language": language}
                )
                return SimpleNamespace(
                    items=[SimpleNamespace(text=text, start_time=0.0, end_time=0.5)]
                ), None
            return await super().align_items(
                audio, text=text, language=language, timeout_sec=timeout_sec
            )

    timestamps = LanguageSwitchTimestampActor(invalid_item_calls={2})

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            (wav, SAMPLE_RATE),
            options=OfflineTranscriptionOptions(),
            timestamp_actor=timestamps,
            chunk_sec=1.0,
        )
    ]

    segments = [
        event.segment
        for event in events
        if event.kind == "segment" and event.segment is not None
    ]
    assert [
        (segment.text, segment.language, segment.timing_status) for segment in segments
    ] == [
        ("hello", "English", "aligned"),
        ("第二句", "Chinese", "estimated"),
    ]


@pytest.mark.asyncio
async def test_stream_transcribe_file_splits_long_sentence_into_subtitle_cues_and_translation_unit() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 9.0),), dtype=np.float32) * 0.1
    text = "今天讨论字幕显示问题，并且保持翻译输入完整，还要避免把一句普通话切成半截，只有真正太长的时候才拆开。"
    model = FakeOfflineModel([text])
    timestamps = FakeItemTimestampActor()

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            (wav, SAMPLE_RATE),
            options=OfflineTranscriptionOptions(language="Chinese"),
            timestamp_actor=timestamps,
            chunk_sec=9.0,
        )
    ]

    segments = [event.segment for event in events if event.kind == "segment"]
    translation_units = [
        event.translation_unit for event in events if event.kind == "translation_unit"
    ]

    assert [event.kind for event in events] == [
        "segment",
        "segment",
        "segment",
        "segment",
        "translation_unit",
        "complete",
    ]
    assert [segment.text for segment in segments if segment is not None] == [
        "今天讨论字幕显示问题，",
        "并且保持翻译输入完整，",
        "还要避免把一句普通话切成半截，",
        "只有真正太长的时候才拆开。",
    ]
    assert [segment.timing_status for segment in segments if segment is not None] == [
        "aligned"
    ] * 4
    assert (
        max(
            int((segment.end_ms or 0) - (segment.start_ms or 0))
            for segment in segments
            if segment is not None
        )
        <= 6000
    )
    assert len(translation_units) == 1
    assert translation_units[0] is not None
    assert translation_units[0].source_text == text
    assert translation_units[0].source_segment_ids == tuple(
        segment.id for segment in segments if segment is not None
    )
    assert translation_units[0].source_segment_indices == tuple(
        segment.index for segment in segments if segment is not None
    )
    assert events[-1].document is not None
    assert [segment.text for segment in events[-1].document.segments] == [
        "今天讨论字幕显示问题，",
        "并且保持翻译输入完整，",
        "还要避免把一句普通话切成半截，",
        "只有真正太长的时候才拆开。",
    ]


@pytest.mark.asyncio
async def test_stream_transcribe_file_flushes_unpunctuated_unit_at_final() -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["hello", ""])

    class EarlyItemTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del audio, language, timeout_sec
            return SimpleNamespace(
                items=[SimpleNamespace(text=text, start_time=0.0, end_time=0.1)]
            ), None

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            (wav, SAMPLE_RATE),
            options=OfflineTranscriptionOptions(language="English"),
            timestamp_actor=EarlyItemTimestampActor(),
            chunk_sec=1.0,
        )
    ]

    assert [event.kind for event in events] == [
        "segment",
        "translation_unit",
        "complete",
    ]
    assert events[0].segment is not None
    assert events[0].segment.text == "hello"


@pytest.mark.asyncio
async def test_stream_transcribe_file_keeps_unclosed_source_unit_across_chunks(
    tmp_path,
) -> None:
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    wav[int(SAMPLE_RATE * 0.95) : int(SAMPLE_RATE * 1.05)] = 0.0
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, wav, SAMPLE_RATE)
    model = FakeOfflineModel(["今天讨论字幕", "显示问题。"])
    timestamps = FakeItemTimestampActor()

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            str(audio_path),
            options=OfflineTranscriptionOptions(language="Chinese"),
            timestamp_actor=timestamps,
            chunk_sec=1.0,
        )
    ]

    segments = [event.segment for event in events if event.kind == "segment"]
    translation_units = [
        event.translation_unit for event in events if event.kind == "translation_unit"
    ]

    assert [segment.text for segment in segments if segment is not None] == [
        "今天讨论字幕显示问题。"
    ]
    assert len(segments) == 1
    assert segments[0] is not None
    assert segments[0].start_ms == 0
    assert segments[0].end_ms is not None and segments[0].end_ms > 1000
    assert len(translation_units) == 1
    assert translation_units[0] is not None
    assert translation_units[0].source_text == "今天讨论字幕显示问题。"


@pytest.mark.asyncio
async def test_stream_transcribe_file_keeps_unclosed_source_unit_across_timing_fallback() -> (
    None
):
    wav = np.ones((int(SAMPLE_RATE * 2.0),), dtype=np.float32) * 0.1
    model = FakeOfflineModel(["今天讨论字幕", "显示问题。"])
    timestamps = FakeMixedTimestampActor(fail_item_calls={2})

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            (wav, SAMPLE_RATE),
            options=OfflineTranscriptionOptions(language="Chinese"),
            timestamp_actor=timestamps,
            chunk_sec=1.0,
        )
    ]

    segments = [
        event.segment
        for event in events
        if event.kind == "segment" and event.segment is not None
    ]
    translation_units = [
        event.translation_unit for event in events if event.kind == "translation_unit"
    ]

    assert [(segment.text, segment.timing_status) for segment in segments] == [
        ("今天讨论字幕显示问题。", "estimated")
    ]
    assert len(translation_units) == 1
    assert translation_units[0] is not None
    assert translation_units[0].source_text == "今天讨论字幕显示问题。"


@pytest.mark.asyncio
async def test_stream_transcribe_file_clamps_item_timestamps_to_real_file_duration(
    tmp_path,
) -> None:
    wav = np.ones((int(SAMPLE_RATE * 0.2),), dtype=np.float32) * 0.1
    audio_path = tmp_path / "short.wav"
    sf.write(audio_path, wav, SAMPLE_RATE)
    model = FakeOfflineModel(["hello."])

    class PaddedDurationTimestampActor:
        async def align_items(
            self,
            audio: np.ndarray,
            *,
            text: str,
            language: str,
            timeout_sec: float | None,
        ) -> tuple[object | None, str | None]:
            del text, language, timeout_sec
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        text="hello",
                        start_time=0.0,
                        end_time=float(audio.shape[0]) / float(SAMPLE_RATE),
                    )
                ]
            ), None

    events = [
        event
        async for event in stream_transcribe_file(
            model,
            str(audio_path),
            options=OfflineTranscriptionOptions(language="English"),
            timestamp_actor=PaddedDurationTimestampActor(),
            chunk_sec=1.0,
        )
    ]

    segments = [event.segment for event in events if event.segment is not None]
    assert events[-1].document is not None
    assert events[-1].document.duration_ms == 200
    assert [
        (segment.text, segment.start_ms, segment.end_ms) for segment in segments
    ] == [("hello.", 0, 200)]


@pytest.mark.asyncio
async def test_stream_transcribe_file_rejects_translation_options() -> None:
    wav = np.ones((int(SAMPLE_RATE * 1.0),), dtype=np.float32) * 0.1
    events = stream_transcribe_file(
        FakeOfflineModel(["第一句"]),
        (wav, SAMPLE_RATE),
        options=OfflineTranscriptionOptions(
            language="Chinese", target_language="English"
        ),
        chunk_sec=1.0,
    )

    with pytest.raises(ValueError, match="does not support translation"):
        await anext(events)
