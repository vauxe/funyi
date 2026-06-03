# coding=utf-8
"""MLX audio encoder for Qwen3-ASR (mirrors Qwen3ASRAudioEncoder).

Reproduces conv downsampling, sinusoidal positional embedding, the per-chunk
packing, the windowed ``cu_seqlens``, and the block-diagonal (non-causal)
attention. The chunk-structure arithmetic is done in Python ints (deterministic,
exactly matching ``_get_feat_extract_output_lengths``) to avoid device syncs.
"""
from __future__ import annotations

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import MLXAudioConfig

NEG_INF = -1e9


def feat_extract_output_length(input_length: int) -> int:
    """Python-int copy of _get_feat_extract_output_lengths for a single length."""
    leave = input_length % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_length // 100) * 13


def _sinusoids(length: int, channels: int, max_timescale: int = 10000) -> mx.array:
    assert channels % 2 == 0
    log_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_increment * mx.arange(channels // 2).astype(mx.float32))
    scaled = mx.arange(length).astype(mx.float32).reshape(length, 1) * inv_timescales.reshape(1, -1)
    return mx.concatenate([mx.sin(scaled), mx.cos(scaled)], axis=1)  # (length, channels)


class _Buffer:
    """Plain holder so a non-learnable array is not tracked as a Module parameter."""

    def __init__(self, value: mx.array):
        self.value = value


class AudioAttention(nn.Module):
    def __init__(self, cfg: MLXAudioConfig):
        super().__init__()
        self.embed_dim = cfg.d_model
        self.num_heads = cfg.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array, mask: mx.array) -> mx.array:
        seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        q = mx.expand_dims(q.transpose(1, 0, 2), 0)  # (1, nh, S, hd)
        k = mx.expand_dims(k.transpose(1, 0, 2), 0)
        v = mx.expand_dims(v.transpose(1, 0, 2), 0)

        # Windowed (block-diagonal) attention per the model design: each cu_seqlens block
        # attends only within itself, matching the FA2 varlen path the model is served with
        # and the block mask built by the upstream `_prepare_attention_mask`. NOTE: the
        # vendored transformers eager/sdpa fallback passes NO mask to its layers (full
        # attention), so a CPU eager reference only matches single-block (short) audio;
        # tools/parity_mlx_vs_hf.py patches the reference to window for a true comparison.
        scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * self.scaling + mask
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = mx.matmul(weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(seq_len, -1)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    def __init__(self, cfg: MLXAudioConfig):
        super().__init__()
        self.self_attn = AudioAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def __call__(self, hidden_states: mx.array, mask: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(nn.gelu(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states


class AudioEncoder(nn.Module):
    def __init__(self, cfg: MLXAudioConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.n_window = cfg.n_window
        self.n_window_infer = cfg.n_window_infer
        self.conv_chunksize = cfg.conv_chunksize
        self.conv2d1 = nn.Conv2d(1, cfg.downsample_hidden_size, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(cfg.downsample_hidden_size, cfg.downsample_hidden_size, 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(cfg.downsample_hidden_size, cfg.downsample_hidden_size, 3, stride=2, padding=1)
        freq_after = (((cfg.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(cfg.downsample_hidden_size * freq_after, d, bias=False)
        self.layers = [AudioEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        self.ln_post = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, d)
        self.proj2 = nn.Linear(d, cfg.output_dim)
        # Non-learnable sinusoidal table (persistent=False upstream; not in safetensors).
        self._pos = _Buffer(_sinusoids(cfg.max_source_positions, d))

    def _pack_single_audio(self, input_features: mx.array, feature_len: int):
        chunk_width = self.n_window * 2
        chunk_count = max(1, (feature_len + chunk_width - 1) // chunk_width)
        tail_len = feature_len - (chunk_count - 1) * chunk_width
        max_chunk_len = chunk_width if chunk_count > 1 else tail_len
        mel_bins = input_features.shape[0]

        flattened = mx.zeros((chunk_count * max_chunk_len, mel_bins), dtype=input_features.dtype)
        valid = mx.swapaxes(input_features[:, :feature_len], 0, 1)  # (feature_len, mel)
        flattened[:feature_len] = valid
        padded = flattened.reshape(chunk_count, max_chunk_len, mel_bins)
        padded = mx.swapaxes(padded, 1, 2)  # (chunk_count, mel, max_chunk_len)

        chunk_lengths = [chunk_width] * chunk_count
        chunk_lengths[-1] = tail_len
        feat_after = [feat_extract_output_length(cl) for cl in chunk_lengths]
        return padded, chunk_lengths, feat_after

    def __call__(self, input_features: mx.array, feature_len: int) -> mx.array:
        """input_features: (num_mel_bins, feature_len). Returns (N_audio_tokens, output_dim)."""
        padded, chunk_lengths, feat_after = self._pack_single_audio(input_features, feature_len)
        chunk_count = padded.shape[0]
        max_after = max(feat_after)

        # NHWC for MLX conv: (chunk_count, mel, max_chunk_len, 1)
        x = mx.expand_dims(padded, -1)
        embeds = []
        for s in range(0, chunk_count, self.conv_chunksize):
            chunk = x[s : s + self.conv_chunksize]
            chunk = nn.gelu(self.conv2d1(chunk))
            chunk = nn.gelu(self.conv2d2(chunk))
            chunk = nn.gelu(self.conv2d3(chunk))
            embeds.append(chunk)
        embed = mx.concatenate(embeds, axis=0)  # (b, f, t, c)  NHWC
        b, f, t, c = embed.shape
        embed = embed.transpose(0, 2, 3, 1).reshape(b, t, c * f)  # (b, t, c*f), c-major
        embed = self.conv_out(embed)  # (b, t, d_model)

        pos = self._pos.value[: embed.shape[1]].astype(embed.dtype)
        embed = embed + mx.expand_dims(pos, 0)

        # Gather valid positions: first feat_after[i] rows of chunk i (row-major).
        flat = embed.reshape(b * t, -1)
        idx: List[int] = []
        for i in range(chunk_count):
            base = i * t
            idx.extend(range(base, base + feat_after[i]))
        hidden_states = flat[mx.array(idx, dtype=mx.int32)]  # (N, d_model)

        # Windowed cu_seqlens (token blocks the encoder attends within).
        window_aftercnn = max_after * (self.n_window_infer // (self.n_window * 2))
        aftercnn_total = feat_extract_output_length(feature_len)
        block_lens: List[int] = []
        block_lens += [window_aftercnn] * (aftercnn_total // window_aftercnn)
        remainder = aftercnn_total % window_aftercnn
        if remainder != 0:
            block_lens.append(remainder)
        mask = self._block_mask(block_lens, hidden_states.shape[0])

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = nn.gelu(self.proj1(hidden_states))
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    @staticmethod
    def _block_mask(block_lens: List[int], seq_len: int) -> mx.array:
        """Additive (1, 1, S, S) mask: 0 within each cu_seqlens block, NEG_INF across blocks."""
        block_ids: List[int] = []
        for b, length in enumerate(block_lens):
            block_ids.extend([b] * length)
        assert len(block_ids) == seq_len, f"block coverage {len(block_ids)} != seq_len {seq_len}"
        ids = mx.array(block_ids, dtype=mx.int32)
        same = mx.expand_dims(ids, 1) == mx.expand_dims(ids, 0)  # (S, S)
        mask = mx.where(same, mx.array(0.0, dtype=mx.float32), mx.array(NEG_INF, dtype=mx.float32))
        return mask.reshape(1, 1, seq_len, seq_len)
