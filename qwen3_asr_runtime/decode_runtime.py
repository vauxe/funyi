# coding=utf-8
"""
Custom decode loop with CUDA graph capture for Qwen3-ASR thinker.

Replaces transformers `generate` for ASR decoding. Prefill still runs through
the normal HF forward (variable length, once per transcribe). Decode steps run
against fixed-shape buffers and a StaticCache, and a single decode step is
captured into a CUDA graph that is replayed for subsequent tokens.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import torch
from transformers.cache_utils import DynamicCache, StaticCache

from .cuda_serialization import CUDA_GRAPH_CAPTURE_LOCK


@dataclass
class DecodeBuffers:
    input_ids: torch.Tensor          # [B, 1]
    attention_mask: torch.Tensor     # [B, max_len]
    cache_position: torch.Tensor     # [1]
    position_ids: torch.Tensor       # [3, B, 1] multimodal rope positions


def _normalize_eos_ids(eos: Any) -> List[int]:
    if eos is None:
        return []
    if isinstance(eos, (list, tuple, set)):
        return [int(x) for x in eos]
    return [int(eos)]


# Qwen3-ASR uses these as the default EOS token ids when the model's
# generation_config is missing.
_UPSTREAM_DEFAULT_EOS_TOKEN_IDS = [151645, 151643]
ProfileCallback = Callable[[str, float], None]


def _resolve_eos_token_ids(thinker: Any, explicit: Any = None) -> List[int]:
    if explicit is not None:
        return _normalize_eos_ids(explicit)
    gen_cfg = getattr(thinker, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is not None:
        return _normalize_eos_ids(gen_cfg.eos_token_id)
    cfg_eos = getattr(thinker.config, "eos_token_id", None)
    if cfg_eos is not None:
        return _normalize_eos_ids(cfg_eos)
    return list(_UPSTREAM_DEFAULT_EOS_TOKEN_IDS)


class _ProfileSection:
    def __init__(self, owner: Any, name: str) -> None:
        self.owner = owner
        self.name = name
        self.start: float | None = None

    def __enter__(self) -> None:
        self.start = self.owner._profile_start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.owner._profile_stop(self.name, self.start)
        return False


class CudaGraphCaptureRequired(RuntimeError):
    """Raised when a frozen CUDA graph decoder would need a larger capture."""


class CudaGraphDecoder:
    """
    Reusable decode runtime. One instance per thinker; it re-allocates buffers
    on the first call that needs more room and keeps the CUDA graph across
    subsequent calls as long as max_len does not grow past the captured value.
    """

    def __init__(self, thinker: Any, *, graph_len_bucket: int = 1) -> None:
        self.thinker = thinker
        self.device = next(thinker.parameters()).device
        self.dtype = next(thinker.parameters()).dtype
        self.graph_len_bucket = max(1, int(graph_len_bucket))
        text_config = thinker.config.text_config
        self.text_config = text_config
        self.num_layers = int(text_config.num_hidden_layers)
        self.num_kv_heads = int(text_config.num_key_value_heads)
        self.head_dim = int(text_config.head_dim)

        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_max_len: int = 0
        self._buffers: DecodeBuffers | None = None
        self._logits_buf: torch.Tensor | None = None
        self._static_cache: StaticCache | None = None
        self._profile_callback: ProfileCallback | None = None
        self._runtime_capture_enabled = True

    def set_profile_callback(self, callback: ProfileCallback | None) -> None:
        """Install an optional timing callback used only by profiling tools."""
        self._profile_callback = callback

    def reset_graph(self) -> None:
        graph = self._graph
        if graph is not None:
            graph.reset()
        self._graph = None
        self._graph_max_len = 0

    def reset_runtime(self) -> None:
        if not self._runtime_capture_enabled and self._graph is not None:
            return
        self.reset_graph()
        self._buffers = None
        self._logits_buf = None
        self._static_cache = None

    def freeze_runtime_capture(self) -> None:
        if self._graph is None:
            raise RuntimeError("cannot freeze CUDA graph capture before a graph is captured")
        self._runtime_capture_enabled = False

    @property
    def runtime_capture_enabled(self) -> bool:
        return self._runtime_capture_enabled

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        eos_token_id: Any = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("CudaGraphDecoder currently supports batch size 1.")
        if not self.device.type == "cuda":
            raise RuntimeError("CudaGraphDecoder requires a CUDA device.")

        eos_set = set(_resolve_eos_token_ids(self.thinker, eos_token_id))

        batch = 1
        prompt_len = int(input_ids.shape[1])
        max_len = prompt_len + int(max_new_tokens)

        # Reset rope_deltas so prefill recomputes it.
        self.thinker.rope_deltas = None

        # --- Prefill (variable-length forward through the HF path) ---
        # Use DynamicCache for prefill so attention memory scales with prompt_len
        # only, not max_cache_len. We transfer to StaticCache before decode.
        prefill_cache = DynamicCache()
        cache_position = torch.arange(prompt_len, device=self.device)

        prefill_out = self.thinker(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            past_key_values=prefill_cache,
            cache_position=cache_position,
            logits_to_keep=1,
            use_cache=True,
            return_dict=True,
        )
        static_cache = self._ensure_static_cache(max_len=max_len, batch=batch)
        self._copy_dynamic_to_static(prefill_cache, static_cache, prompt_len)
        del prefill_cache
        next_token_id = int(prefill_out.logits[:, -1, :].argmax(dim=-1).item())
        generated: List[int] = []
        if next_token_id in eos_set:
            return self._build_output(input_ids, generated)
        generated.append(next_token_id)
        if len(generated) >= max_new_tokens:
            return self._build_output(input_ids, generated)

        rope_deltas = self.thinker.rope_deltas
        if rope_deltas is None:
            raise RuntimeError("thinker.rope_deltas missing after prefill; cannot run decode loop.")

        return self._decode_after_prefill(
            input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            prefix_len=prompt_len,
            next_input_id=next_token_id,
            generated=generated,
            static_cache=static_cache,
            rope_deltas=rope_deltas,
            max_len=max_len,
            eos_set=eos_set,
            max_new_tokens=max_new_tokens,
        )

    @torch.inference_mode()
    def generate_with_draft(
        self,
        *,
        input_ids: torch.Tensor,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
        draft_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_id: Any = None,
        stats: dict[str, int] | None = None,
    ) -> torch.Tensor:
        """Graph-aware counterpart of ``generate()`` with speculative draft.

        Verifier prefill runs once over ``prompt + draft`` with a DynamicCache;
        accepted draft tokens cost only the extra prefill positions. The cache
        is cropped to ``prompt_len + accepted`` and copied into the StaticCache,
        then the tail decode loop runs against the captured CUDA graph just as
        in the plain ``generate()`` path. See ``spec_decode.py`` for the
        non-graph variant and the bf16-non-determinism caveat.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("CudaGraphDecoder currently supports batch size 1.")
        if self.device.type != "cuda":
            raise RuntimeError("CudaGraphDecoder requires a CUDA device.")
        prompt_len = int(input_ids.shape[1])
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return self._build_output(input_ids, [])

        draft = [int(x) for x in draft_ids][:max_new_tokens]
        if not draft:
            raise ValueError("generate_with_draft requires a non-empty draft; use generate() instead.")

        eos_set = set(_resolve_eos_token_ids(self.thinker, eos_token_id))

        K = len(draft)
        max_len = prompt_len + max_new_tokens

        self.thinker.rope_deltas = None

        draft_t = torch.tensor([draft], dtype=input_ids.dtype, device=self.device)
        ext_input_ids = torch.cat([input_ids, draft_t], dim=1)
        ext_attention_mask = torch.cat(
            [attention_mask,
             torch.ones((1, K), dtype=attention_mask.dtype, device=self.device)],
            dim=1,
        )
        ext_len = prompt_len + K

        prefill_cache = DynamicCache()
        cache_position = torch.arange(ext_len, device=self.device)
        prefill_out = self.thinker(
            input_ids=ext_input_ids,
            input_features=input_features,
            attention_mask=ext_attention_mask,
            feature_attention_mask=feature_attention_mask,
            past_key_values=prefill_cache,
            cache_position=cache_position,
            logits_to_keep=K + 1,
            use_cache=True,
            return_dict=True,
        )
        verify_logits = prefill_out.logits[0, :, :]                         # [K+1, V]
        preds = verify_logits.argmax(dim=-1).tolist()                       # [K+1]

        accepted = 0
        for j in range(K):
            if preds[j] == draft[j]:
                accepted += 1
            else:
                break

        if stats is not None:
            stats["draft_tokens"] = K
            stats["accepted_tokens"] = accepted

        generated: List[int] = list(draft[:accepted])
        if len(generated) >= max_new_tokens:
            return self._build_output(input_ids, generated)
        next_id = int(preds[accepted])
        if next_id in eos_set:
            return self._build_output(input_ids, generated)
        generated.append(next_id)
        if len(generated) >= max_new_tokens:
            return self._build_output(input_ids, generated)

        rope_deltas = self.thinker.rope_deltas
        if rope_deltas is None:
            raise RuntimeError("thinker.rope_deltas missing after prefill; cannot run decode loop.")

        effective_len = prompt_len + accepted
        prefill_cache.crop(effective_len)

        static_cache = self._ensure_static_cache(max_len=max_len, batch=1)
        self._copy_dynamic_to_static(prefill_cache, static_cache, effective_len)
        del prefill_cache

        return self._decode_after_prefill(
            input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            prefix_len=effective_len,
            next_input_id=next_id,
            generated=generated,
            static_cache=static_cache,
            rope_deltas=rope_deltas,
            max_len=max_len,
            eos_set=eos_set,
            max_new_tokens=max_new_tokens,
        )

    def _decode_after_prefill(
        self,
        *,
        input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prefix_len: int,
        next_input_id: int,
        generated: List[int],
        static_cache: StaticCache,
        rope_deltas: torch.Tensor,
        max_len: int,
        eos_set: set,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Run the warmup + graph-capture + decode loop from a prefilled state.

        ``prefix_len`` is the number of positions already in ``static_cache``
        (equal to ``prompt_len`` for the plain path, ``prompt_len + accepted``
        for the spec path). ``next_input_id`` is fed at slot ``prefix_len``.
        ``generated`` must already include ``next_input_id`` as its last entry.
        """
        batch = 1
        prompt_len = int(prompt_attention_mask.shape[1])
        buffers = self._ensure_decode_buffers(max_len=max_len, batch=batch, ref_ids=input_ids)
        graph_len = int(buffers.attention_mask.shape[1])
        buffers.attention_mask[:, :prompt_len] = prompt_attention_mask
        buffers.attention_mask[:, prompt_len:].zero_()
        # Mark accepted-draft positions (if any) as valid cache slots.
        if prefix_len > prompt_len:
            buffers.attention_mask[0, prompt_len:prefix_len] = 1

        current_len = prefix_len
        buffers.input_ids[0, 0] = int(next_input_id)
        current_len += 1
        buffers.attention_mask[0, current_len - 1] = 1
        buffers.cache_position[0] = current_len - 1
        self._update_position_ids(buffers, rope_deltas)

        warmup_logits = self._decode_forward(
            buffers,
            static_cache,
            attention_mask_slice=buffers.attention_mask[:, :current_len],
        )
        if self._logits_buf is None:
            self._logits_buf = torch.empty_like(warmup_logits)
        self._logits_buf.copy_(warmup_logits)
        next_token_t = self._logits_buf[:, -1, :].argmax(dim=-1).view(1, 1)
        buffers.input_ids.copy_(next_token_t)

        first_id = int(next_token_t.item())
        if first_id in eos_set:
            return self._build_output(input_ids, generated)
        generated.append(first_id)
        if len(generated) >= max_new_tokens:
            return self._build_output(input_ids, generated)

        if self._graph is None or self._graph_max_len != graph_len:
            if not self._runtime_capture_enabled:
                raise CudaGraphCaptureRequired(
                    f"frozen CUDA graph is not available for graph_len={graph_len}; "
                    f"captured graph_len={self._graph_max_len}"
                )
            if self._profile_callback is None:
                self._capture_graph(
                    buffers=buffers,
                    static_cache=static_cache,
                    next_tok_t=next_token_t,
                    current_len=current_len,
                    rope_deltas=rope_deltas,
                )
            else:
                with self._profile_section("cuda_graph.capture_total"):
                    self._capture_graph(
                        buffers=buffers,
                        static_cache=static_cache,
                        next_tok_t=next_token_t,
                        current_len=current_len,
                        rope_deltas=rope_deltas,
                    )

        while len(generated) < max_new_tokens:
            current_len += 1
            buffers.attention_mask[0, current_len - 1] = 1
            buffers.cache_position[0] = current_len - 1
            self._update_position_ids(buffers, rope_deltas)
            assert self._graph is not None
            if self._profile_callback is None:
                self._graph.replay()
            else:
                with self._profile_section("cuda_graph.replay"):
                    self._graph.replay()
            next_token_t = self._logits_buf[:, -1, :].argmax(dim=-1).view(1, 1)
            buffers.input_ids.copy_(next_token_t)
            tok_id = int(next_token_t.item())
            if tok_id in eos_set:
                break
            generated.append(tok_id)

        return self._build_output(input_ids, generated)

    # ---------- internals ----------

    def _copy_dynamic_to_static(self, dyn: DynamicCache, static: StaticCache, prompt_len: int) -> None:
        if self._profile_callback is None:
            self._copy_dynamic_to_static_impl(dyn, static, prompt_len)
            return
        with self._profile_section("cuda_graph.cache_copy_dynamic_to_static"):
            self._copy_dynamic_to_static_impl(dyn, static, prompt_len)

    def _copy_dynamic_to_static_impl(self, dyn: DynamicCache, static: StaticCache, prompt_len: int) -> None:
        for layer_idx in range(self.num_layers):
            dyn_layer = dyn.layers[layer_idx]
            k, v = dyn_layer.keys, dyn_layer.values
            static_layer = static.layers[layer_idx]
            if not static_layer.is_initialized:
                static_layer.lazy_initialization(k)
            static_layer.keys[:, :, :prompt_len, :].copy_(k[:, :, :prompt_len, :])
            static_layer.values[:, :, :prompt_len, :].copy_(v[:, :, :prompt_len, :])

    def _profile_sync(self) -> None:
        if self._profile_callback is None or not torch.cuda.is_available():
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        torch.cuda.synchronize()

    def _profile_section(self, name: str) -> _ProfileSection:
        return _ProfileSection(self, name)

    def _profile_start(self) -> float | None:
        if self._profile_callback is None:
            return None
        self._profile_sync()
        return time.perf_counter()

    def _profile_stop(self, name: str, start: float | None) -> None:
        callback = self._profile_callback
        if callback is None or start is None:
            return
        self._profile_sync()
        callback(name, time.perf_counter() - start)

    def _ensure_static_cache(self, *, max_len: int, batch: int) -> StaticCache:
        max_len = self._reserve_len(max_len)
        cache = self._static_cache
        if cache is None or cache.max_cache_len < max_len or cache.max_batch_size < batch:
            if not self._runtime_capture_enabled:
                cached_len = 0 if cache is None else int(cache.max_cache_len)
                raise CudaGraphCaptureRequired(
                    f"frozen CUDA graph cache is too small: requested max_len={max_len}, "
                    f"cached max_len={cached_len}"
                )
            # Release the graph private pool before allocating a larger cache.
            self.reset_graph()
            self._static_cache = StaticCache(
                config=self.text_config,
                max_batch_size=batch,
                max_cache_len=max_len,
                device=self.device,
                dtype=self.dtype,
            )
            return self._static_cache
        cache.reset()
        return cache

    def _ensure_decode_buffers(self, *, max_len: int, batch: int, ref_ids: torch.Tensor) -> DecodeBuffers:
        max_len = self._reserve_len(max_len)
        buffers = self._buffers
        if (
            buffers is None
            or buffers.attention_mask.shape[0] < batch
            or buffers.attention_mask.shape[1] < max_len
        ):
            if not self._runtime_capture_enabled:
                cached_len = 0 if buffers is None else int(buffers.attention_mask.shape[1])
                raise CudaGraphCaptureRequired(
                    f"frozen CUDA graph buffers are too small: requested max_len={max_len}, "
                    f"cached max_len={cached_len}"
                )
            self.reset_graph()
            self._buffers = DecodeBuffers(
                input_ids=torch.zeros(batch, 1, dtype=ref_ids.dtype, device=self.device),
                attention_mask=torch.zeros(batch, max_len, dtype=torch.long, device=self.device),
                cache_position=torch.zeros(1, dtype=torch.long, device=self.device),
                position_ids=torch.zeros(3, batch, 1, dtype=torch.long, device=self.device),
            )
            return self._buffers
        return buffers

    def _reserve_len(self, max_len: int) -> int:
        bucket = self.graph_len_bucket
        if bucket <= 1:
            return int(max_len)
        max_len = int(max_len)
        return ((max_len + bucket - 1) // bucket) * bucket

    def _update_position_ids(self, buffers: DecodeBuffers, rope_deltas: torch.Tensor) -> None:
        # Match thinker.forward's decode branch: position = cache_position[0] + rope_deltas
        # with seq_length=1 so every cell of [3,1,1] holds the same scalar.
        delta = buffers.cache_position[0] + rope_deltas.view(-1)[0]
        buffers.position_ids.copy_(delta.to(buffers.position_ids.dtype).expand_as(buffers.position_ids))

    def _decode_forward(
        self,
        buffers: DecodeBuffers,
        static_cache: StaticCache,
        *,
        attention_mask_slice: torch.Tensor,
    ) -> torch.Tensor:
        # Call thinker (not thinker.model) so audio-merge / rope branches run
        # exactly as they would during HF generate. Passing position_ids ourselves
        # also avoids the ``cache_position[0] == 0`` host-side guard in
        # thinker.forward, which would otherwise break graph capture.
        out = self.thinker(
            input_ids=buffers.input_ids,
            attention_mask=attention_mask_slice,
            position_ids=buffers.position_ids,
            past_key_values=static_cache,
            cache_position=buffers.cache_position,
            use_cache=True,
            return_dict=True,
        )
        return out.logits

    def _capture_graph(
        self,
        *,
        buffers: DecodeBuffers,
        static_cache: StaticCache,
        next_tok_t: torch.Tensor,
        current_len: int,
        rope_deltas: torch.Tensor,
    ) -> None:
        with CUDA_GRAPH_CAPTURE_LOCK:
            # Graph capture locks in buffers.attention_mask's full max_len shape.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                buffers.input_ids.copy_(next_tok_t.view(1, 1))
                buffers.cache_position[0] = current_len
                buffers.attention_mask[0, current_len] = 1
                self._update_position_ids(buffers, rope_deltas)
                logits = self._decode_forward(
                    buffers,
                    static_cache,
                    attention_mask_slice=buffers.attention_mask,
                )
                self._logits_buf.copy_(logits)
            torch.cuda.current_stream().wait_stream(side)

            self.reset_graph()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits = self._decode_forward(
                    buffers,
                    static_cache,
                    attention_mask_slice=buffers.attention_mask,
                )
                self._logits_buf.copy_(logits)
            self._graph = graph
            self._graph_max_len = int(buffers.attention_mask.shape[1])

    def _build_output(self, input_ids: torch.Tensor, generated: Sequence[int]) -> torch.Tensor:
        prompt_len = input_ids.shape[1]
        total = prompt_len + len(generated)
        out = torch.empty(1, total, dtype=input_ids.dtype, device=self.device)
        out[:, :prompt_len] = input_ids
        if generated:
            out[0, prompt_len:] = torch.tensor(list(generated), dtype=input_ids.dtype, device=self.device)
        return out


__all__ = ["CudaGraphCaptureRequired", "CudaGraphDecoder"]
