# Qwen3-ASR 1.7B Architecture

## Purpose

This note records only the model facts that matter for standalone runtime work.

## High-level structure

`Qwen/Qwen3-ASR-1.7B` is not a classic seq2seq ASR model.

Practical structure:

1. processor builds `input_features` and expands the `<audio>` placeholder
2. audio tower converts `input_features` to `audio_features`
3. decoder-only text model builds text embeddings
4. multimodal merge replaces audio placeholder embeddings with `audio_features`
5. autoregressive generation runs on the merged sequence

So the right mental model is:

- `speech`
- `merge`
- `decoder prefill`
- `decoder decode`

not “encoder + decoder cross-attention”.

## Real 1.7B config

### Text

- `hidden_size = 2048`
- `intermediate_size = 6144`
- `num_hidden_layers = 28`
- `num_attention_heads = 16`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `vocab_size = 151936`
- `max_position_embeddings = 65536`

### Audio

- `num_mel_bins = 128`
- `d_model = 1024`
- `encoder_layers = 24`
- `encoder_attention_heads = 16`
- `encoder_ffn_dim = 4096`
- `output_dim = 2048`
- `n_window_infer = 800`

Important point:

- audio tower output is `2048`, which matches the text hidden size

That is why audio features can be inserted directly into the text embedding stream.

## Processor contract

The processor is part of the model contract, not a thin helper.

It:

- runs the Whisper-style feature extractor
- derives audio-feature lengths
- expands the `<audio>` placeholder so placeholder count matches audio feature length

Implication:

- the safest component boundary starts from `input_features`, not raw waveform

## Audio tower

The audio tower is `Qwen3ASRAudioEncoder`.

Structure:

1. three `Conv2d` downsampling layers
2. projection into audio hidden size
3. positional embedding
4. 24-layer Transformer encoder
5. projection from `1024` to `2048`

Important runtime facts:

- it chunks long mel features internally
- it uses ragged attention-style sequence handling
- `get_audio_features()` avoids batch inference for precision reasons

Implication:

- this is not a good candidate for a first monolithic runtime boundary

## Text model

The text backbone is `Qwen3ASRThinkerTextModel`.

It is a decoder-only Transformer with:

- 28 blocks
- RMSNorm
- GQA (`16` attention heads, `8` KV heads)
- gated SiLU MLP
- RoPE and KV cache for generation

## Multimodal merge

There is no text-to-audio cross-attention block.

Instead:

1. text embeddings are built from `input_ids`
2. `audio_features` are computed
3. positions where `input_ids == audio_token_id` are found
4. those embedding slots are replaced with `audio_features`

This is the key runtime decomposition fact.

## Prefill vs decode

Generation has two different phases.

### Prefill

- audio is still present
- multimodal merge happens
- RoPE-related state is initialized

### Decode

- no new audio enters
- generation continues with new token input and KV cache only

Implication:

- split `prefill` and `decode`

## RoPE state

The model tracks `rope_deltas` during multimodal prefill and reuses them during decode.

Implication:

- any runtime path that ignores `rope_deltas` will drift from the reference implementation

## Long-audio behavior in this repo

Long-audio support is wrapper-level, not a single giant core-model forward.

The runtime wrapper:

1. splits input audio into chunks up to `MAX_ASR_INPUT_SECONDS`
2. transcribes each chunk
3. concatenates text
4. merges language results

Implication:

- current long-audio parity is wrapper-level offline parity

## Runtime Decomposition Takeaway

The right component decomposition is:

1. `speech`
2. `merge`
3. `prefill`
4. `decode`

And the right first input boundary is:

- `input_features`

not raw waveform.
