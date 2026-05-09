# Streaming Runtime

Use when changing `streaming_transcribe`, bounded live captions, or streaming
gates. WebSocket/VAD behavior is in `@docs/realtime_asr_service.md`.

## Defaults

Library streaming must stay upstream-compatible:

- `chunk_size_sec=2.0`
- `max_window_sec=None`
- `spec_decode=False`
- optimized load flags off
- each step re-feeds the accumulated audio

This path is hash-regressed by `local_goldens/streaming_regression.json`.
Optimized paths are CER-gated, not hash-gated.

## Live Modes

Setting `max_window_sec` enables bounded live semantics:

- audio context is capped;
- old text may move to `committed_text`;
- `partial_text` remains mutable;
- `state.text = committed_text + partial_text`;
- omitted `max_prefix_tokens` becomes `192`.

| Mode | Window | Role |
|---|---:|---|
| live20 | 20s | local low-latency service |
| live30 | 30s | generic bounded-live baseline |
| live45 | 45s | stricter quality mode |

Keep `abs-delta` drift separate from `worse-delta` quality regression.

## Current Decisions

- live30 is the generic start point.
- live20 is the service preset: lower first-text/update latency, higher total
  wall and more drift.
- `spec_decode + cuda_graph` is the current best live30 speed/latency path, but
  not byte-identical under bf16.
- W8A16 means qkv/gate_up only.
- service graph bucket is `cuda_graph_len_bucket=64`; library/tool defaults stay
  `1`.

## Service Preset

Load-time flags:

```python
Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    cuda_graph=True,
    flashinfer=True,
    fused_rmsnorm=True,
    fused_linears=True,
    quantized_linears=True,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
```

Streaming kwargs:

```python
Qwen3ASRModel.low_latency_preset_kwargs()
```

returns `chunk_size_sec=0.5`, `unfixed_chunk_num=4`, `unfixed_token_num=5`,
`max_window_sec=20.0`, `max_prefix_tokens=64`, and `spec_decode=True`.

## Spec Decode

`spec_decode=True` verifies rollback draft tokens with a prefill over
`prompt + rollback_draft`. Accepted draft tokens skip decode steps. Under bf16,
prefill-path KV can drift from decode-path KV, so gate with streaming CER.

Implementation invariant: prefix trimming may commit old text, but must preserve
the rolled-back token suffix as `draft_ids`.

## Do Not Reopen Without New Evidence

- `chunk_size_sec=3`, `unfixed_token_num=3`, or `max_new_tokens=64`
- live30 `max_prefix_tokens=64` outside the service profile
- graph buckets `32` or `256`
- audio-prefix KV reuse or frozen-block audio/KV reuse
- persistent reserved StaticCache/graph across steps without allocator control
