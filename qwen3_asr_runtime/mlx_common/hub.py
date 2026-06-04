# coding=utf-8
"""Hugging Face checkpoint resolution shared by the MLX adapters.

The MLX model loaders need a local directory (config.json + safetensors), so a
bare HF id is resolved to its cached snapshot dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_model_dir(model_path: str, *, local_files_only: bool = True, revision: Optional[str] = None) -> str:
    """Return a local checkpoint directory for a path or HF id."""
    if Path(model_path).exists():
        return str(model_path)
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=str(model_path), revision=revision, local_files_only=local_files_only))


def snapshot_commit(model_path: str) -> Optional[str]:
    """Extract the commit hash from a resolved ``.../snapshots/<hash>/...`` path."""
    parts = Path(str(model_path)).parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[index + 1] or None
    return None
