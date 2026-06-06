# coding=utf-8
"""Load safetensors into an MLX module tree with a fail-loud name map.

Handles two checkpoint flavours:
  * official HF fp16/bf16 weights (``thinker.*`` prefix, torch Conv2d layout
    ``(out,in,kH,kW)``, separate ``lm_head.weight``);
  * pre-quantized MLX checkpoints (e.g. mlx-community/*-4bit): no ``thinker.``
    prefix, conv already in MLX ``(out,kH,kW,in)`` layout, ``model.*`` linears +
    embedding stored as packed uint32 + scales/biases, ``lm_head`` tied (absent).

The conv layout and prefix are auto-detected against the model's own parameter
shapes, so the same loader serves ASR, the forced aligner, and HY-MT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

_HF_PREFIX = "thinker."
_INT_DTYPES = {
    mx.uint8,
    mx.uint16,
    mx.uint32,
    mx.uint64,
    mx.int8,
    mx.int16,
    mx.int32,
    mx.int64,
}


def _find_safetensors(model_dir: Path) -> list[Path]:
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        raw = json.loads(index.read_text(encoding="utf-8"))
        weight_map = raw.get("weight_map") or {}
        shards = [
            model_dir / name
            for name in sorted(set(str(name) for name in weight_map.values()))
        ]
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors found in {model_dir}")
    missing = [path.name for path in shards if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint index references missing shard(s): {missing[:5]}"
        )
    return shards


def load_weights_dict(model_dir: str) -> Dict[str, mx.array]:
    weights: Dict[str, mx.array] = {}
    for shard in _find_safetensors(Path(model_dir)):
        for key, value in mx.load(str(shard)).items():
            if key in weights:
                raise KeyError(f"duplicate tensor {key!r} while loading {shard.name}")
            weights[key] = value
    return weights


def quantize_predicate(weights: Dict[str, mx.array]):
    """nn.quantize class_predicate: quantize exactly the modules stored quantized."""
    stripped = {
        k[len(_HF_PREFIX) :] if k.startswith(_HF_PREFIX) else k for k in weights
    }

    def predicate(path: str, module: nn.Module) -> bool:
        return hasattr(module, "to_quantized") and f"{path}.scales" in stripped

    return predicate


def map_and_load(
    weights: Dict[str, mx.array], model: nn.Module, compute_dtype: mx.Dtype
) -> nn.Module:
    model_params = dict(tree_flatten(model.parameters()))
    model_keys = set(model_params)

    mapped: Dict[str, mx.array] = {}
    for key, value in weights.items():
        nk = key[len(_HF_PREFIX) :] if key.startswith(_HF_PREFIX) else key
        target = model_params.get(nk)
        # Conv2d layout: torch (out,in,kH,kW) -> MLX (out,kH,kW,in). Pre-converted
        # checkpoints already match; only transpose when the shape disagrees and the
        # transpose makes it agree.
        if (
            target is not None
            and value.ndim == 4
            and tuple(value.shape) != tuple(target.shape)
        ):
            t = mx.transpose(value, (0, 2, 3, 1))
            if tuple(t.shape) == tuple(target.shape):
                value = t
        # Cast float tensors to the compute dtype; leave packed-quantized (integer) weights.
        if value.dtype not in _INT_DTYPES:
            value = value.astype(compute_dtype)
        mapped[nk] = value

    ckpt_keys = set(mapped)
    missing = model_keys - ckpt_keys
    extra = ckpt_keys - model_keys
    if missing:
        raise KeyError(
            f"model params not provided by checkpoint: {sorted(missing)[:10]}"
        )
    if extra:
        raise KeyError(
            f"checkpoint tensors not consumed by model: {sorted(extra)[:10]}"
        )
    for k in model_keys:
        want = tuple(model_params[k].shape)
        got = tuple(mapped[k].shape)
        if want != got:
            raise ValueError(
                f"shape mismatch for {k}: model {want} vs checkpoint {got}"
            )

    model.load_weights(list(mapped.items()))
    return model
