# coding=utf-8
"""Parity + CER check for the MLX backend against the official-code reference.

Two reference modes (per AGENTS.md, quality is gated by CER vs an official-code
golden, never by byte equality):

  * live HF (default): runs the unmodified upstream `transformers` Qwen3-ASR
    model on CPU on the SAME weights, and compares the MLX backend against it.
    Also reports numerical diffs (audio features, prefill argmax) which, in
    float32, localize any forward bug to a single component.
  * --reference-json: compares MLX output against stored reference texts. Use this for the formal
    offline gate without needing torch at run time.

Examples:
    uv run python tools/parity_mlx_vs_hf.py --model Qwen/Qwen3-ASR-0.6B \
        --wav local_data/e2e_en_espeak_20260522T201810.wav --seconds 6 --dtype float32
    uv run python tools/parity_mlx_vs_hf.py --model <path> --dtype bfloat16 --max-cer 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime_helpers import _cer  # noqa: E402

SAMPLE_RATE = 16000


def _load_clip(path: str, seconds: float) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if seconds and seconds > 0:
        wav = wav[: int(seconds * SAMPLE_RATE)]
    return np.asarray(wav, dtype=np.float32)


PROMPT_MESSAGES = [
    {"role": "system", "content": ""},
    {"role": "user", "content": [{"type": "audio", "audio": ""}]},
]


def _build_processor(model: str):
    from transformers import AutoConfig, AutoProcessor

    from qwen3_asr_runtime.hf_qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)
    return AutoProcessor.from_pretrained(model, fix_mistral_regex=True)


def _patch_hf_audio_windowing() -> None:
    """Make the HF reference apply cu_seqlens windowing in its audio encoder.

    The vendored transformers forward passes no mask to the audio layers, so the
    eager fallback does FULL attention — whereas the model is served with FA2 varlen
    windowing (each cu_seqlens block attends only within itself). Without this patch
    the reference only matches single-block (short) audio. We patch the module-level
    eager_attention_forward to build the block-diagonal mask from cu_seq_lens_q when no
    mask is supplied; the text path always passes a causal mask, so it is unaffected.
    """
    import torch

    import qwen3_asr_runtime.hf_qwen3_asr.modeling_qwen3_asr as M

    orig = M.eager_attention_forward

    def windowed(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        cu = kwargs.get("cu_seq_lens_q")
        if attention_mask is None and cu is not None:
            s = query.shape[-2]
            m = torch.full(
                (1, 1, s, s),
                torch.finfo(query.dtype).min,
                dtype=query.dtype,
                device=query.device,
            )
            cu_list = cu.tolist() if hasattr(cu, "tolist") else list(cu)
            for i in range(1, len(cu_list)):
                a, b = int(cu_list[i - 1]), int(cu_list[i])
                m[..., a:b, a:b] = 0
            attention_mask = m
        return orig(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )

    M.eager_attention_forward = windowed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, help="HF id or local path of Qwen3-ASR-0.6B"
    )
    ap.add_argument(
        "--wav", action="append", default=None, help="audio file(s); repeatable"
    )
    ap.add_argument(
        "--seconds",
        type=float,
        default=6.0,
        help="trim each clip to N seconds (0 = full)",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        help="MLX compute dtype (float32 for strict parity)",
    )
    ap.add_argument("--hf-dtype", default="float32", help="HF reference dtype")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--max-cer",
        type=float,
        default=0.02,
        help="gate: mean CER vs reference must be <=",
    )
    ap.add_argument(
        "--reference-json",
        default=None,
        help="JSON {wav_basename: text} to gate against instead of live HF",
    )
    args = ap.parse_args()

    import mlx.core as mx

    from qwen3_asr_runtime.mlx_qwen3_asr import load_mlx_qwen3_asr

    wavs = args.wav or ["local_data/e2e_en_espeak_20260522T201810.wav"]
    processor = _build_processor(args.model)
    prompt = processor.apply_chat_template(
        PROMPT_MESSAGES, add_generation_prompt=True, tokenize=False
    )

    ml, cfg = load_mlx_qwen3_asr(args.model, dtype=args.dtype)
    eos = cfg.eos_token_ids

    reference = None
    hf = None
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text())
    else:
        import torch

        from transformers import AutoModel
        from qwen3_asr_runtime.hf_qwen3_asr import Qwen3ASRForConditionalGeneration  # noqa: F401

        _patch_hf_audio_windowing()  # reference must window like the served model, not the eager fallback
        hf = Qwen3ASRForConditionalGeneration.from_pretrained(
            args.model, dtype=getattr(torch, args.hf_dtype), attn_implementation="eager"
        ).eval()

    cers, token_matches, audio_diffs = [], [], []
    for wav_path in wavs:
        wav = _load_clip(wav_path, args.seconds)
        inputs = processor(
            text=[prompt], audio=[wav], return_tensors="np", padding=True
        )
        ids = np.asarray(inputs["input_ids"]).astype(np.int32)
        feats = np.asarray(inputs["input_features"], dtype=np.float32)
        flen = [
            int(np.asarray(inputs["feature_attention_mask"]).sum(-1).reshape(-1)[0])
        ]

        ml_gen = ml.generate(
            mx.array(ids),
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos,
            input_features=mx.array(feats).astype(ml.compute_dtype),
            feature_lengths=flen,
        )
        ml_text = processor.tokenizer.decode(
            ml_gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        name = Path(wav_path).name
        if reference is not None:
            ref_text = reference.get(name, "")
            token_match = None
            audio_diff = None
        else:
            import torch

            with torch.no_grad():
                hf_af = (
                    hf.thinker.get_audio_features(
                        torch.tensor(feats),
                        feature_attention_mask=torch.tensor(
                            np.asarray(inputs["feature_attention_mask"])
                        ),
                    )
                    .float()
                    .numpy()
                )
                ml_af = np.asarray(
                    ml.get_audio_features(mx.array(feats), flen).astype(mx.float32)
                )
                audio_diff = float(np.abs(hf_af - ml_af).max())
                res = hf.generate(
                    input_ids=torch.tensor(ids.astype(np.int64)),
                    input_features=torch.tensor(feats),
                    attention_mask=torch.tensor(np.asarray(inputs["attention_mask"])),
                    feature_attention_mask=torch.tensor(
                        np.asarray(inputs["feature_attention_mask"])
                    ),
                    max_new_tokens=args.max_new_tokens,
                )
            seq = res.sequences if hasattr(res, "sequences") else res
            hf_gen = [t for t in seq[0, ids.shape[1] :].tolist() if t not in set(eos)]
            ref_text = processor.tokenizer.decode(
                hf_gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            token_match = sum(1 for a, b in zip(hf_gen, ml_gen) if a == b) / max(
                1, max(len(hf_gen), len(ml_gen))
            )
            audio_diffs.append(audio_diff)
            token_matches.append(token_match)

        cer = _cer(ml_text, ref_text)
        cers.append(cer)
        print(f"\n=== {name} (dur~{args.seconds}s) ===")
        if audio_diff is not None:
            print(
                f"  audio-feat max|diff| = {audio_diff:.2e}   token-match = {token_match:.3f}"
            )
        print(f"  CER vs reference     = {cer:.4f}")
        print(f"  MLX : {ml_text!r}")
        print(f"  REF : {ref_text!r}")

    mean_cer = sum(cers) / len(cers)
    print(f"\nmean CER = {mean_cer:.4f} over {len(cers)} clip(s) (dtype={args.dtype})")
    if audio_diffs:
        print(
            f"max audio-feat diff = {max(audio_diffs):.2e}  min token-match = {min(token_matches):.3f}"
        )
    ok = mean_cer <= args.max_cer
    print("GATE:", "PASS" if ok else "FAIL", f"(threshold {args.max_cer})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
