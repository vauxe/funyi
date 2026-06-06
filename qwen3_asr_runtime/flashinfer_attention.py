# coding=utf-8
"""
FlashInfer attention integration for transformers.

Registers a "flashinfer" entry in transformers.ALL_ATTENTION_FUNCTIONS that
dispatches single-query decode (batch=1, seq_q=1) to
``flashinfer.single_decode_with_kv_cache``. Anything else - multi-query
prefill, the audio-encoder's ragged attention with an explicit mask, batched
decode - falls back to SDPA.

We tried routing prefill through ``single_prefill_with_kv_cache`` as well;
one-layer output matched SDPA within BF16 noise, but the multi-layer
accumulated behavior on real Qwen3-ASR inputs diverged dramatically (CER
jumped from ~10% to >370%). That path is intentionally disabled.
"""

from __future__ import annotations

import math
import os
import shutil
from typing import Any, Optional, Tuple

import torch


_FLASHINFER_AVAILABLE: Optional[bool] = None
_NINJA_READY: Optional[bool] = None


def _ensure_ninja_on_path() -> bool:
    """Make the Python ninja wheel visible to FlashInfer's subprocess JIT.

    FlashInfer invokes the ``ninja`` executable by name during first-use JIT.
    When callers run ``.venv/bin/python`` directly, the package can be installed
    while ``.venv/bin`` is absent from PATH. The ninja wheel exposes that bin dir
    through ``ninja.BIN_DIR``; adding it here keeps runtime setup local to the
    optional FlashInfer path.
    """
    global _NINJA_READY
    if _NINJA_READY is not None:
        return _NINJA_READY

    if shutil.which("ninja") is not None:
        _NINJA_READY = True
        return True

    try:
        import ninja
    except Exception:
        _NINJA_READY = False
        return False

    bin_dir = getattr(ninja, "BIN_DIR", None)
    if not bin_dir:
        _NINJA_READY = False
        return False

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(bin_dir) + (os.pathsep + old_path if old_path else "")
    _NINJA_READY = shutil.which("ninja") is not None
    return _NINJA_READY


def _have_flashinfer() -> bool:
    global _FLASHINFER_AVAILABLE
    if _FLASHINFER_AVAILABLE is None:
        try:
            import flashinfer  # noqa: F401

            _FLASHINFER_AVAILABLE = True
        except Exception:
            _FLASHINFER_AVAILABLE = False
    return _FLASHINFER_AVAILABLE


def _sdpa_fallback(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float,
    scaling: Optional[float],
    is_causal: Optional[bool],
    **kwargs: Any,
):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **kwargs,
    )


def flashinfer_attention_forward(
    module: Any,
    query: torch.Tensor,  # [B, Hq, Lq, D]
    key: torch.Tensor,  # [B, Hkv, Lkv, D]
    value: torch.Tensor,  # [B, Hkv, Lkv, D]
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    # Only try flashinfer for the single-request decode path: B=1, Lq=1.
    B, Hq, Lq, D = query.shape
    Hkv, Lkv = key.shape[1], key.shape[2]

    if (
        not _have_flashinfer()
        or B != 1
        or Lq != 1
        or dropout != 0.0
        or query.device.type != "cuda"
    ):
        return _sdpa_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout,
            scaling,
            is_causal,
            **kwargs,
        )

    # StaticCache stores k/v as HND: [num_kv_heads, kv_len, head_dim].
    # Pass that layout directly so decode avoids a per-layer transpose+copy.
    import flashinfer

    q = query.squeeze(0).squeeze(1).contiguous()  # [Hq, D]
    k = key.squeeze(0)  # [Hkv, Lkv, D]
    v = value.squeeze(0)  # [Hkv, Lkv, D]
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # Resolve the scale explicitly instead of relying on FlashInfer's internal default
    # matching SDPA's default; the two paths must agree even if a library default changes.
    sm_scale = float(scaling) if scaling is not None else (1.0 / math.sqrt(D))

    # window_left: not set here (full attention)
    out = flashinfer.single_decode_with_kv_cache(
        q,
        k,
        v,
        kv_layout="HND",
        sm_scale=sm_scale,
        use_tensor_cores=False,
    )
    # out shape: [qo_len=1-impled? or Hq, D] -> single_decode returns [Hq, D]
    # We need [B, Hq, Lq, D] to match the upstream attention contract.
    out = out.view(1, 1, Hq, D).transpose(1, 2).contiguous()  # [B=1, Hq, Lq=1, D]
    return out, None


def register_flashinfer(name: str = "flashinfer") -> bool:
    """Register flashinfer attention under the given key in transformers.
    Returns True on success, False if flashinfer isn't available.

    Uses AttentionInterface.register() (classmethod, _global_mapping) so the
    registration is visible to every AttentionInterface instance created by
    transformers internals - the __setitem__ path only writes to _local_mapping
    and does not propagate.
    """
    if not _have_flashinfer():
        return False
    if not _ensure_ninja_on_path():
        raise RuntimeError(
            "flashinfer=True requires the `ninja` executable; install dependencies with `uv sync --python 3.12`."
        )
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(name, flashinfer_attention_forward)
    return True


__all__ = ["flashinfer_attention_forward", "register_flashinfer"]
