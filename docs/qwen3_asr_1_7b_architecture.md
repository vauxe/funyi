# Qwen3-ASR 1.7B Architecture

Runtime-shape facts only.

## Flow

Qwen3-ASR is not classic encoder-decoder ASR:

```text
processor -> audio tower -> multimodal merge -> decoder prefill -> decode
```

Steps:

1. processor builds `input_features` and expands `<audio>`;
2. audio tower emits `audio_features`;
3. decoder-only text model builds token embeddings;
4. audio placeholder embeddings are replaced with `audio_features`;
5. autoregressive generation continues.

There is no text-to-audio cross-attention block.

## Key Sizes

Text: hidden `2048`, intermediate `6144`, layers `28`, heads `16`, KV heads `8`,
head dim `128`, vocab `151936`, max positions `65536`.

Audio: mel bins `128`, hidden `1024`, layers `24`, heads `16`, FFN `4096`,
output `2048`, `n_window_infer=800`.

Audio output dim matches text hidden size, so features can be inserted directly
into the text embedding stream.

## Runtime Boundaries

- The processor is part of the model contract; it computes feature lengths and
  expands the audio placeholder.
- `input_features` is a safer first runtime boundary than raw waveform.
- `Qwen3ASRAudioEncoder` uses Conv2d downsampling, positional embeddings, a
  24-layer encoder, and projection `1024 -> 2048`.
- audio tower sequence handling is ragged/windowed; do not replace with dense
  attention unless parity is proved.
- prefill sees audio, performs multimodal merge, and initializes `rope_deltas`.
- decode sees only new tokens, KV cache, and reused `rope_deltas`.
- ignoring `rope_deltas` will drift.

## Long Audio

Long-audio support is wrapper-level: split to `MAX_ASR_INPUT_SECONDS`,
transcribe chunks, concatenate text, merge language. Current parity is not one
giant core-model forward.
