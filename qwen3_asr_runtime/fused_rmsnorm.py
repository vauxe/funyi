# coding=utf-8
"""
Optional drop-in replacement for Qwen3ASR's custom RMSNorm modules using
torch.nn.functional.rms_norm. NOT bit-equivalent to the hand-rolled
fp32-upcast path; we trade a small numerical delta for 8-kernel -> 1-kernel
reduction.

Enable with TransformersASRBackend.from_pretrained(..., fused_rmsnorm=True).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(
        hidden_states,
        normalized_shape=self.weight.shape,
        weight=self.weight,
        eps=self.variance_epsilon,
    )


def patch_model_rmsnorms(model) -> int:
    """Replace every Qwen3ASR*RMSNorm.forward with fused_rmsnorm_forward.
    Returns the number of modules patched.
    """
    count = 0
    for m in model.modules():
        cls_name = type(m).__name__
        # Match by class name substring so we catch both Qwen3ASRTextRMSNorm
        # (audio encoder) and Qwen3ASRThinkerTextRMSNorm (text decoder) without
        # importing the model file here.
        if "RMSNorm" in cls_name and hasattr(m, "weight") and hasattr(m, "variance_epsilon"):
            m.forward = fused_rmsnorm_forward.__get__(m, type(m))
            count += 1
    return count


__all__ = ["fused_rmsnorm_forward", "patch_model_rmsnorms"]
