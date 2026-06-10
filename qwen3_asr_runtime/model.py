# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .backends import ASRRuntimeBackend, TransformersASRBackend
from .streaming import (
    ASRStreamingState,
    RecognitionFrame,
    RollingWindowTrimPolicy,
    StreamingPrefixPlan,
)
from .utils import (
    MAX_ASR_INPUT_SECONDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)

_ASR_TEXT_TAG = "<asr_text>"
_LIVE_DEFAULT_MAX_PREFIX_TOKENS = 192
_TRIM_POLICY = RollingWindowTrimPolicy()


@dataclass
class ASRTranscription:
    language: str
    text: str
    time_stamps: Optional[Any] = None


class Qwen3ASRModel:
    """
    Qwen3-ASR runtime transformers inference wrapper.

    This implementation intentionally does not depend on upstream package code.
    It keeps the public inference behavior aligned with the upstream
    transformers backend, including the vLLM-style streaming state machine.
    """

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        backend_runtime: Optional[ASRRuntimeBackend] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 512,
    ):
        if backend_runtime is None:
            if model is None or processor is None:
                raise ValueError(
                    "Either backend_runtime or both model and processor must be provided."
                )
            backend_runtime = TransformersASRBackend(model=model, processor=processor)

        self.backend_runtime = backend_runtime
        self.backend = backend_runtime.name
        self.model = backend_runtime.model
        self.processor = backend_runtime.processor
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = max_new_tokens

        self.device = backend_runtime.device
        self.dtype = backend_runtime.dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 512,
        backend: str = "transformers",
        **kwargs,
    ) -> "Qwen3ASRModel":
        backend_name = str(backend).lower().strip()
        if backend_name == "transformers":
            backend_runtime = TransformersASRBackend.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        elif backend_name == "mlx":
            # Lazy import: MLX is only available on Apple Silicon, so importing it
            # must not be required on CUDA-only hosts.
            from .backends.mlx import MLXASRBackend

            backend_runtime = MLXASRBackend.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return cls(
            backend_runtime=backend_runtime,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    def get_supported_languages(self) -> List[str]:
        return list(SUPPORTED_LANGUAGES)

    @classmethod
    def low_latency_preset_kwargs(cls) -> Dict[str, Any]:
        """Streaming kwargs for single-user low-latency live captioning.

        Pair with ``from_pretrained(cuda_graph=True, flashinfer=True,
        fused_rmsnorm=True, fused_linears=True)`` on the loading side. See
        ``@docs/streaming_runtime.md`` "Low-Latency Single-User Preset" and
        the local streaming CER gate.
        """
        return {
            "chunk_size_sec": 0.5,
            "unfixed_chunk_num": 4,
            "unfixed_token_num": 5,
            "max_window_sec": 20.0,
            "max_prefix_tokens": 64,
            "spec_decode": True,
        }

    @classmethod
    def mlx_streaming_preset_kwargs(cls) -> Dict[str, Any]:
        """Streaming kwargs tuned for the MLX (Apple Silicon) backend.

        The MLX backend has no cuda_graph/flashinfer, so every step re-encodes
        the audio window and re-prefills the decoder; a shorter window and a
        larger chunk keep the step under the chunk budget on typical speech
        (measured M1 / 0.6B-4bit: ~0.2 s fixed mel+encoder+prefill cost plus
        ~8 suffix tokens decoded per step). The MLX backend does implement
        ``infer_streaming_with_draft``, but ``spec_decode`` stays off by
        default: with the ~5-token rollback drafts the saving is bounded by
        accepted-tokens x per-token decode (~25 ms/step on M1 0.6B-4bit, inside
        step-to-step noise) and the output drifts from the plain path by
        half-precision epsilon, so enabling it needs the streaming CER sweep
        first. Decode-heavier configs (bf16 weights, ~3x the per-token decode
        cost) should save roughly 70-90 ms/step by the same arithmetic, but
        confirming that end-to-end needs thermally stable hardware -- on a
        fanless M1, sustained-load throttling swamps differences of this size.
        """
        return {
            "chunk_size_sec": 1.0,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 5,
            "max_window_sec": 8.0,
            "max_prefix_tokens": 48,
            "spec_decode": False,
        }

    def eval(self) -> "Qwen3ASRModel":
        self.backend_runtime.eval()
        return self

    @torch.inference_mode()
    def prewarm_realtime_cuda_graph(
        self,
        *,
        context: str = "",
        language: Optional[str] = "Chinese",
        max_window_sec: float = 20.0,
        max_prefix_tokens: int = 64,
    ) -> bool:
        prewarm = getattr(self.backend_runtime, "prewarm_cuda_graph", None)
        if prewarm is None:
            return False
        force_language = None
        if language is not None and str(language).strip() != "":
            force_language = normalize_language_name(str(language))
            validate_language(force_language)
        prompt = self._build_text_prompt(context=context, force_language=force_language)
        prompt += self._build_prewarm_prefix(max_prefix_tokens)
        sample_count = max(1, int(round(float(max_window_sec) * SAMPLE_RATE)))
        wav = np.zeros((sample_count,), dtype=np.float32)
        return bool(prewarm(prompt=prompt, wav=wav, max_new_tokens=self.max_new_tokens))

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        if return_time_stamps:
            raise ValueError(
                "This runtime transformers implementation does not include Qwen3-ForcedAligner. "
                "Use return_time_stamps=False."
            )

        wavs = normalize_audios(audio)
        sample_count = len(wavs)

        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and sample_count > 1:
            ctxs = ctxs * sample_count
        if len(ctxs) != sample_count:
            raise ValueError(
                f"Batch size mismatch: audio={sample_count}, context={len(ctxs)}"
            )

        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * sample_count
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and sample_count > 1:
                langs_in = langs_in * sample_count
            if len(langs_in) != sample_count:
                raise ValueError(
                    f"Batch size mismatch: audio={sample_count}, language={len(langs_in)}"
                )

        langs_norm: List[Optional[str]] = []
        for item in langs_in:
            if item is None or str(item).strip() == "":
                langs_norm.append(None)
                continue
            normalized = normalize_language_name(str(item))
            validate_language(normalized)
            langs_norm.append(normalized)

        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=MAX_ASR_INPUT_SECONDS,
            )
            for j, (chunk_wav, offset_sec) in enumerate(parts):
                chunks.append(
                    AudioChunk(
                        orig_index=i,
                        chunk_index=j,
                        wav=chunk_wav,
                        sr=SAMPLE_RATE,
                        offset_sec=offset_sec,
                    )
                )

        chunk_ctx = [ctxs[c.orig_index] for c in chunks]
        chunk_lang = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs = [c.wav for c in chunks]
        raw_outputs = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)

        out_langs: List[List[str]] = [[] for _ in range(sample_count)]
        out_texts: List[List[str]] = [[] for _ in range(sample_count)]
        for chunk, raw_text, forced_lang in zip(chunks, raw_outputs, chunk_lang):
            lang, text = parse_asr_output(raw_text, user_language=forced_lang)
            out_langs[chunk.orig_index].append(lang)
            out_texts[chunk.orig_index].append(text)

        results: List[ASRTranscription] = []
        for i in range(sample_count):
            merged_text = "".join([text for text in out_texts[i] if text is not None])
            merged_language = merge_languages(out_langs[i])
            results.append(
                ASRTranscription(
                    language=merged_language,
                    text=merged_text,
                    time_stamps=None,
                )
            )
        return results

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
        max_window_sec: Optional[float] = None,
        max_prefix_tokens: Optional[int] = None,
        spec_decode: bool = False,
    ) -> ASRStreamingState:
        if chunk_size_sec is None or float(chunk_size_sec) <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

        force_language = None
        if language is not None and str(language).strip() != "":
            force_language = normalize_language_name(str(language))
            validate_language(force_language)

        chunk_size_samples = max(1, int(round(float(chunk_size_sec) * SAMPLE_RATE)))
        max_window_samples = None
        if max_window_sec is not None:
            max_window_sec = float(max_window_sec)
            if max_window_sec <= 0:
                raise ValueError(f"max_window_sec must be > 0, got: {max_window_sec}")
            if max_window_sec < float(chunk_size_sec):
                raise ValueError(
                    f"max_window_sec must be >= chunk_size_sec when set, got "
                    f"max_window_sec={max_window_sec}, chunk_size_sec={chunk_size_sec}"
                )
            max_window_samples = max(1, int(round(max_window_sec * SAMPLE_RATE)))
            if max_prefix_tokens is None:
                max_prefix_tokens = _LIVE_DEFAULT_MAX_PREFIX_TOKENS
        if max_prefix_tokens is not None and int(max_prefix_tokens) <= 0:
            raise ValueError(
                f"max_prefix_tokens must be > 0 when set, got: {max_prefix_tokens}"
            )

        self.backend_runtime.reset_decode_runtime()
        prompt_raw = self._build_text_prompt(
            context=context, force_language=force_language
        )

        return ASRStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_samples=int(chunk_size_samples),
            max_window_samples=max_window_samples,
            max_prefix_tokens=int(max_prefix_tokens)
            if max_prefix_tokens is not None
            else None,
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            audio_seen_samples=0,
            audio_trim_cursor=0,
            prompt_raw=prompt_raw,
            force_language=force_language,
            language="",
            text="",
            carried_text_prefix="",
            partial_text="",
            _raw_decoded="",
            spec_decode=bool(spec_decode),
        )

    @torch.inference_mode()
    def streaming_transcribe(
        self, pcm16k: np.ndarray, state: ASRStreamingState
    ) -> ASRStreamingState:
        if state is None:
            raise ValueError(
                "state must not be None. Call init_streaming_state() first."
            )
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        audio = self._normalize_stream_pcm(pcm16k)
        if audio.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, audio], axis=0)

        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]
            self._append_streaming_audio(state, chunk)
            self._run_streaming_decode_step(state)
            state.chunk_id += 1

        return state

    @torch.inference_mode()
    def finish_streaming_transcribe(
        self, state: ASRStreamingState
    ) -> ASRStreamingState:
        if state is None:
            raise ValueError("state must not be None.")
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)
        self._append_streaming_audio(state, tail)
        prefix = self._build_finish_streaming_prefix(state)
        prompt = state.prompt_raw + prefix
        generated = self._infer_with_prompts([prompt], [state.audio_accum])[0]
        raw_decoded = prefix + generated
        self._set_streaming_decoded(state, raw_decoded, prompt_prefix=prefix)
        state.chunk_id += 1
        return state

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        messages = self._build_messages(context=context, audio_payload="")
        base = self.backend_runtime.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    def _build_prewarm_prefix(self, max_prefix_tokens: int) -> str:
        token_budget = max(0, int(max_prefix_tokens))
        if token_budget <= 0:
            return ""
        seed = "测试实时字幕翻译。"
        text = seed
        while len(self.backend_runtime.encode_text(text)) < token_budget:
            text += seed
        token_ids = self.backend_runtime.encode_text(text)[:token_budget]
        return self.backend_runtime.decode_text(token_ids)

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        prompts = [
            self._build_text_prompt(context=c, force_language=fl)
            for c, fl in zip(contexts, languages)
        ]
        return self._infer_with_prompts(prompts, wavs)

    def _infer_with_prompts(
        self, prompts: List[str], wavs: List[np.ndarray]
    ) -> List[str]:
        return self.backend_runtime.infer_with_prompts(
            prompts,
            wavs,
            max_inference_batch_size=self.max_inference_batch_size,
            max_new_tokens=self.max_new_tokens,
        )

    def _run_streaming_decode_step(self, state: ASRStreamingState) -> None:
        plan = self._build_streaming_prefix_plan(state)
        prefix = plan.prefix
        prompt = state.prompt_raw + prefix
        stats = state.spec_decode_stats

        generated: Optional[str] = None
        draft_ids = plan.draft_ids
        if state.spec_decode and draft_ids:
            stats["spec_attempt_steps"] = stats.get("spec_attempt_steps", 0) + 1
            if plan.trimmed:
                stats["spec_trimmed_attempt_steps"] = (
                    stats.get("spec_trimmed_attempt_steps", 0) + 1
                )
            try:
                spec_stats: Dict[str, int] = {}
                generated = self.backend_runtime.infer_streaming_with_draft(
                    prompt=prompt,
                    wav=state.audio_accum,
                    draft_ids=draft_ids,
                    max_new_tokens=self.max_new_tokens,
                    stats=spec_stats,
                )
                verified = int(spec_stats.get("draft_tokens", 0))
                accepted = int(spec_stats.get("accepted_tokens", 0))
                stats["spec_verified_draft_tokens"] = (
                    stats.get("spec_verified_draft_tokens", 0) + verified
                )
                stats["spec_accepted_tokens"] = (
                    stats.get("spec_accepted_tokens", 0) + accepted
                )
            except NotImplementedError:
                generated = None
        elif state.spec_decode:
            stats["spec_no_draft_steps"] = stats.get("spec_no_draft_steps", 0) + 1
        if generated is None:
            generated = self._infer_with_prompts([prompt], [state.audio_accum])[0]
        raw_decoded = prefix + generated
        self._set_streaming_decoded(state, raw_decoded, prompt_prefix=prefix)

    def _build_streaming_prefix_plan(
        self, state: ASRStreamingState
    ) -> StreamingPrefixPlan:
        if state.chunk_id < state.unfixed_chunk_num:
            return StreamingPrefixPlan(prefix="", draft_ids=[])

        cur_ids = self.backend_runtime.encode_text(state._raw_decoded)
        rollback = max(0, int(state.unfixed_token_num))
        while True:
            end_idx = max(0, len(cur_ids) - rollback)
            prefix = (
                self.backend_runtime.decode_text(cur_ids[:end_idx])
                if end_idx > 0
                else ""
            )
            if "\ufffd" not in prefix or end_idx == 0:
                prefix, trimmed = self._limit_streaming_prefix(state, prefix)
                return StreamingPrefixPlan(
                    prefix=prefix,
                    draft_ids=list(cur_ids[end_idx:]),
                    trimmed=trimmed,
                )
            rollback += 1

    def _build_finish_streaming_prefix(self, state: ASRStreamingState) -> str:
        return self._build_streaming_prefix_plan(state).prefix

    def _append_streaming_audio(
        self, state: ASRStreamingState, chunk: np.ndarray
    ) -> None:
        if chunk.shape[0] == 0:
            return
        state.audio_seen_samples += int(chunk.shape[0])
        state.audio_accum = self._append_audio(state.audio_accum, chunk)
        _TRIM_POLICY.apply(state)

    def _set_streaming_decoded(
        self, state: ASRStreamingState, raw_decoded: str, *, prompt_prefix: str
    ) -> None:
        state._raw_decoded = raw_decoded
        language, decoded_text = parse_asr_output(
            raw_decoded, user_language=state.force_language
        )
        _, prompt_text_prefix = parse_asr_output(
            prompt_prefix, user_language=state.force_language
        )
        generated_text = self._strip_prompt_text_prefix(
            decoded_text, prompt_text_prefix
        )
        full_text = state.carried_text_prefix + decoded_text
        state.language = language
        state.partial_text = decoded_text
        state.text = full_text
        state.recognition_frame = RecognitionFrame(
            window_start_sample=state.audio_trim_cursor,
            audio_end_sample=state.audio_seen_samples,
            full_text=full_text,
            language=language,
            decoded_text=decoded_text,
            generated_text=generated_text,
        )

    @staticmethod
    def _strip_prompt_text_prefix(decoded_text: str, prompt_text_prefix: str) -> str:
        decoded = str(decoded_text or "").strip()
        prefix = str(prompt_text_prefix or "").strip()
        if prefix and decoded.startswith(prefix):
            return decoded[len(prefix) :].strip()
        return decoded

    def _limit_streaming_prefix(
        self, state: ASRStreamingState, prefix: str
    ) -> tuple[str, bool]:
        if state.max_prefix_tokens is None or not prefix:
            return prefix, False

        header, text = self._split_raw_output_text(state, prefix)
        text_ids = self.backend_runtime.encode_text(text)
        if len(text_ids) <= state.max_prefix_tokens:
            return prefix, False

        split = len(text_ids) - int(state.max_prefix_tokens)
        dropped = self.backend_runtime.decode_text(text_ids[:split])
        kept = self.backend_runtime.decode_text(text_ids[split:])
        if dropped:
            state.carried_text_prefix += dropped
        return header + kept, True

    @staticmethod
    def _split_raw_output_text(state: ASRStreamingState, raw: str) -> tuple[str, str]:
        if state.force_language:
            return "", raw
        marker_idx = raw.find(_ASR_TEXT_TAG)
        if marker_idx < 0:
            return "", raw
        split_idx = marker_idx + len(_ASR_TEXT_TAG)
        return raw[:split_idx], raw[split_idx:]

    @staticmethod
    def _normalize_stream_pcm(pcm16k: np.ndarray) -> np.ndarray:
        audio = np.asarray(pcm16k)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32, copy=False)

    @staticmethod
    def _append_audio(current: np.ndarray, chunk: np.ndarray) -> np.ndarray:
        if current.shape[0] == 0:
            return chunk
        return np.concatenate([current, chunk], axis=0)


__all__ = ["ASRStreamingState", "ASRTranscription", "Qwen3ASRModel"]
