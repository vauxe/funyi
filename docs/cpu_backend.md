# CPU Backend

The CPU backend runs the Torch/transformers path on plain CPU, with no NVIDIA
GPU and no Apple Silicon MLX. It is an opt-in fallback for machines that have
neither: GPU-less Windows or Linux hosts, and Intel (non-arm64) Macs, where the
MLX path does not apply.

It is supported but slow, with no realtime guarantee. The ASR model, the forced
aligner, and HY-MT translation all run in fp32 on CPU. Treat it as an offline /
small-model path, not a live-captions path.

## Enabling It

The Torch/transformers backends require CUDA by default. Opt into CPU with
`--allow-cpu`, or `FUNYI_ALLOW_CPU=1` for the launcher scripts:

```bash
FUNYI_ALLOW_CPU=1 ./scripts/start_backend.sh
```

On native Windows:

```powershell
$env:FUNYI_ALLOW_CPU = "1"; .\scripts\start_backend.ps1
```

`--allow-cpu` alone is enough: with no per-component device flags, the ASR model,
the forced aligner, and translation all resolve to the CPU device in fp32. You do
not need to set `--device-map`, `--timestamp-device-map`, or `--translation-device`
individually.

On Apple Silicon, `--backend auto` selects MLX, not CPU. On an Intel Mac, MLX is
unavailable, so the transformers path with `--allow-cpu` is the CPU route.

## What Runs On CPU

The CPU profile loads every model in fp32 (bf16/fp16 kernels are
slow-to-unsupported on CPU) and disables the CUDA-only acceleration:

- CUDA graph, FlashInfer, and the fused kernels are off.
- `--w8a16` and `--translation-w8a16` are ignored on CPU with a warning, because
  W8A16 is a CUDA Triton path.
- The forced-aligner `fused_rmsnorm` timestamp speedup is off; it is a
  CUDA-profile default.

FireRed Stream-VAD already runs on CPU via `onnxruntime`, unchanged.

## Recommended Configuration

CPU latency scales with model size, and the 1.7B ASR plus 1.8B translation are
heavy. For anything interactive, prefer the smaller ASR model and turn
translation off:

```bash
FUNYI_ALLOW_CPU=1 \
FUNYI_ASR_MODEL=Qwen/Qwen3-ASR-0.6B \
FUNYI_TRANSLATION_MODEL= \
./scripts/start_backend.sh
```

For file transcription, choose `File` as the audio source in the desktop client;
it uses the same backend and tolerates slower-than-realtime decoding.

## Limitations

- Not realtime. Live streaming on CPU lags well behind audio, especially with the
  1.7B ASR model or translation enabled.
- No CUDA-only optimizations, so per-token decode is the unaccelerated fp32 path.

## Validation

The CPU path is held to the same quality posture as the CUDA and MLX backends:
CER vs the official-code golden and punctuation-stripped CER vs allowed SRT for
ASR, and chrF2 vs the stock-model golden for translation, never byte parity. The
public smoke (`compileall` and `pytest`) needs no private audio and covers the
CPU device-resolution and flag-defaulting logic. The CER / chrF2 sweep tools in
`@docs/validation_and_regression.md` currently target CUDA; running them against
the CPU device is a follow-up.
