# coding=utf-8
"""Opt-in W8A16 linears for the fused text decoder."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _w8a16_gemv_kernel(
        x,
        wq,
        scales,
        y,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            xv = tl.load(x + k, mask=k < K, other=0.0).to(tl.float32)
            wv = tl.load(
                wq + offs_n[:, None] * K + k[None, :],
                mask=(offs_n[:, None] < N) & (k[None, :] < K),
                other=0,
            ).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)
        scale = tl.load(scales + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        tl.store(y + offs_n, acc * scale, mask=offs_n < N)

    @triton.jit
    def _w8a16_gemm_kernel(
        x,
        wq,
        scales,
        y,
        M,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            xv = tl.load(
                x + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(
                wq + offs_n[None, :] * K + k[:, None],
                mask=(offs_n[None, :] < N) & (k[:, None] < K),
                other=0,
            ).to(tl.float32)
            # ieee (not tf32): keeps the GEMM path's accumulation precision consistent with
            # the fp32 GEMV path so the two W8A16 paths agree. GEMM is off the single-token
            # decode hot path. NOTE: W8A16 is a CER-gated optimized path — re-run the W8A16
            # CER gate before release after this change.
            acc += tl.dot(xv, wv, input_precision="ieee")
        scale = tl.load(scales + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        tl.store(
            y + offs_m[:, None] * N + offs_n[None, :],
            acc * scale[None, :],
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

else:
    _w8a16_gemv_kernel = None
    _w8a16_gemm_kernel = None


def _require_triton() -> None:
    if triton is None or _w8a16_gemv_kernel is None or _w8a16_gemm_kernel is None:
        raise RuntimeError("quantized_linears=True requires Triton")


def _choose_blocks(out_features: int) -> tuple[int, int]:
    if out_features >= 8192:
        return 32, 1024
    if out_features >= 4096:
        return 16, 256
    return 32, 1024


def _w8a16_gemv(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    _require_triton()
    block_n, block_k = _choose_blocks(int(weight_q.shape[0]))
    y = torch.empty((weight_q.shape[0],), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(weight_q.shape[0], block_n),)
    _w8a16_gemv_kernel[grid](
        x.view(-1),
        weight_q,
        scales,
        y,
        weight_q.shape[1],
        weight_q.shape[0],
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return y.view(1, 1, -1)


def _w8a16_gemm(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    _require_triton()
    out_shape = (*x.shape[:-1], int(weight_q.shape[0]))
    in_features = int(x.shape[-1])
    m = int(x.numel() // in_features)
    n = int(weight_q.shape[0])
    block_m = 16 if m >= 16 else 8
    block_n = 32 if n >= 8192 else 64
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w8a16_gemm_kernel[grid](
        x.view(m, in_features),
        weight_q,
        scales,
        y,
        m,
        in_features,
        n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
    )
    return y.view(*out_shape)


class W8A16Linear(nn.Module):
    """Weight-only int8 GEMV/GEMM for fused decoder projections."""

    def __init__(self, linear: nn.Linear, *, prefill_gemm: str = "triton"):
        super().__init__()
        _require_triton()
        if linear.weight.device.type != "cuda":
            raise RuntimeError("W8A16Linear requires CUDA weights")
        if linear.weight.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("W8A16Linear requires float16 or bfloat16 weights")
        if linear.bias is not None:
            raise RuntimeError("W8A16Linear only supports bias=False linears")
        if prefill_gemm not in ("triton", "cublas"):
            raise ValueError(f"prefill_gemm must be 'triton' or 'cublas', got {prefill_gemm!r}")
        # Multi-token (prefill) GEMM path. 'triton' is the fp32-ieee Triton kernel
        # (kept for the ASR offline path, numerically matched to the GEMV). 'cublas'
        # dequantizes to the activation dtype and uses cuBLAS BF16 — faster for
        # decode-bound models whose prefill is on the hot path (e.g. HY-MT
        # translation: ~3x faster prefill than the Triton kernel, turning W8A16
        # net-positive there). Single-token decode uses the int8 GEMV either way.
        self._prefill_gemm = prefill_gemm
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.input_dtype = linear.weight.dtype
        with torch.no_grad():
            weight = linear.weight.detach()
            scales = (weight.abs().amax(dim=1).float() / 127.0).clamp_min(1e-8)
            weight_q = torch.round(weight.float() / scales[:, None]).clamp(-128, 127).to(torch.int8)
        self.register_buffer("weight_q", weight_q.contiguous(), persistent=False)
        self.register_buffer("scales", scales.contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.device.type == "cuda"
            and x.dim() == 3
            and x.shape[0] == 1
            and x.shape[1] == 1
            and x.shape[2] == self.in_features
            and x.is_contiguous()
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            return _w8a16_gemv(x, self.weight_q, self.scales)
        if (
            x.device.type == "cuda"
            and x.dim() >= 2
            and x.shape[-1] == self.in_features
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            if self._prefill_gemm == "cublas":
                weight = self.weight_q.to(x.dtype) * self.scales.to(x.dtype).unsqueeze(1)
                return torch.nn.functional.linear(x, weight)
            return _w8a16_gemm(x.contiguous(), self.weight_q, self.scales)
        raise RuntimeError(
            "W8A16Linear received an unsupported input "
            f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}."
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def patch_model_quantized_linears(
    model: Any,
    *,
    warmup: bool = True,
) -> dict[str, int]:
    """Patch fused text projections with W8A16 wrappers.

    Requires ``patch_model_fused_linears`` to have created ``qkv_proj`` and
    ``gate_up_proj``. The 2048-output ``o_proj`` and ``down_proj`` stay BF16
    because local GEMV probes show they are faster with cuBLAS/CUTLASS. qkv and
    gate_up do not retain BF16 fallback weights. lm_head stays BF16 because it
    directly controls greedy token selection and W8A16 can flip low-margin
    argmax decisions.
    """
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        raise RuntimeError("model has no `thinker` attribute; cannot quantize")

    counts = {"qkv": 0, "gate_up": 0}
    for layer in thinker.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        if not hasattr(attn, "qkv_proj") or not hasattr(mlp, "gate_up_proj"):
            raise RuntimeError("quantized_linears=True requires fused_linears=True")
        if not isinstance(attn.qkv_proj, W8A16Linear):
            attn.qkv_proj = W8A16Linear(attn.qkv_proj)
            counts["qkv"] += 1
        if not isinstance(mlp.gate_up_proj, W8A16Linear):
            mlp.gate_up_proj = W8A16Linear(mlp.gate_up_proj)
            counts["gate_up"] += 1

    if warmup:
        warmup_quantized_linears(model)
    return counts


def patch_linears_w8a16(
    model: Any,
    *,
    suffixes: tuple[str, ...] = ("gate_proj", "up_proj"),
    prefill_gemm: str = "cublas",
    warmup: bool = True,
) -> int:
    """Replace bias-free CUDA fp16/bf16 ``nn.Linear`` whose qualified name ends in
    one of ``suffixes`` with :class:`W8A16Linear`. Architecture-generic (used for
    the HY-MT translation decoder; out=6144 gate/up have full GEMV occupancy and
    are the only HY-MT linears that speed up decode). Returns the count patched.
    """
    count = 0
    for name, mod in list(model.named_modules()):
        if (
            isinstance(mod, nn.Linear)
            and mod.bias is None
            and mod.weight.device.type == "cuda"
            and mod.weight.dtype in (torch.bfloat16, torch.float16)
            and name.endswith(tuple(suffixes))
        ):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], W8A16Linear(mod, prefill_gemm=prefill_gemm))
            count += 1
    if warmup and count:
        warmup_quantized_linears(model)
    return count


def warmup_quantized_linears(model: Any) -> None:
    """Compile each unique Triton specialization before CUDA graph capture."""
    modules = [module for module in model.modules() if isinstance(module, W8A16Linear)]
    seen: set[tuple[int, int, torch.dtype, torch.device]] = set()
    with torch.no_grad():
        for module in modules:
            key = (
                module.in_features,
                module.out_features,
                module.input_dtype,
                module.weight_q.device,
            )
            if key in seen:
                continue
            seen.add(key)
            for seq_len in (1, 2, 16):
                x = torch.zeros(
                    (1, seq_len, module.in_features),
                    device=module.weight_q.device,
                    dtype=module.input_dtype,
                )
                module(x)
        if modules and torch.cuda.is_available():
            torch.cuda.synchronize(modules[0].weight_q.device)


__all__ = [
    "W8A16Linear",
    "patch_linears_w8a16",
    "patch_model_quantized_linears",
    "warmup_quantized_linears",
]
