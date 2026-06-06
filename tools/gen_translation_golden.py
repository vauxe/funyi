# coding=utf-8
"""Generate the official-code translation golden with the UNMODIFIED stock model.

Mirrors the ASR approach: the ASR quality gate compares funyi against a golden
produced by upstream code, not against funyi gating itself. This does the same
for HY-MT translation -- it runs the stock `transformers` model with plain
`generate()` and NONE of funyi's runtime (no `fixed_mask` custom decode, no
static cache, no `logits_to_keep`, no dynamic-rope surgery, no fused kernels, no
W8A16). The funyi runtime is then gated against this golden via
`tools/gate_translation.py --quality-baseline-json <golden>`.

Only the I/O contract is shared with funyi: the prompt (`build_hymt_prompt`) and
the chat-template tokenization, so the comparison isolates the model RUNTIME, not
the prompt. Quality flags and chrF are computed with the exact gate functions so
"new errors" against the golden are apples-to-apples.

The golden is audio-free public-text-derived output; it lives in `local_goldens/`
(git-ignored) per project convention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse only pure contracts: prompt builder, and the gate's quality + chrF logic,
# so the golden's errors/warnings/reference_similarity match what the gate computes.
from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_MODEL,
    _normalize_model_revision,
    _resolve_model_path,
    _resolved_commit,
    _revision_kwargs,
    build_hymt_prompt,
)
from tools.gate_translation import (  # noqa: E402
    _REFERENCE_METRIC,
    _evaluate_quality,
    _load_cases,
    _reference_similarity,
    _select_cases,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the stock-model HY-MT translation golden."
    )
    parser.add_argument(
        "--dataset", required=True, help="JSONL case file (gate schema)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Golden JSON path (usable as --quality-baseline-json).",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids or groups. Default: all.",
    )
    parser.add_argument("--model", default=DEFAULT_HYMT_MODEL)
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Pin the model to an immutable commit for a reproducible golden (recorded in the payload).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Stock attention impl for the reference. 'sdpa' is the transformers default; "
        "'eager' is the most conservative reference.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model download (default local-only).",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = _select_cases(_load_cases(Path(args.dataset)), args.cases)
    if not cases:
        raise ValueError("no golden cases selected")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    local_only = not args.allow_download
    model_revision = _normalize_model_revision(args.model_revision)
    model_path = _resolve_model_path(
        args.model, local_files_only=local_only, model_revision=model_revision
    )
    revision_kwargs = _revision_kwargs(model_path, model_revision)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_only,
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=True,
        **revision_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=local_only,
        trust_remote_code=args.trust_remote_code,
        **revision_kwargs,
    )
    model.to(torch.device(args.device))
    model.eval()
    load_wall_sec = time.perf_counter() - load_started

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        output, generated_tokens = _translate_stock(
            model,
            tokenizer,
            str(case["text"]),
            target_language=str(case["target_language"]),
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        issues = _evaluate_quality(
            case,
            output,
            generated_tokens=generated_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        chrf = _reference_similarity(case, output)
        rows.append(
            {
                "id": case["id"],
                "group": case.get("group", ""),
                "source_language": case.get("source_language", ""),
                "target_language": case["target_language"],
                "output": output,
                "generated_tokens": generated_tokens,
                "errors": [
                    issue.__dict__ for issue in issues if issue.severity == "error"
                ],
                "warnings": [
                    issue.__dict__ for issue in issues if issue.severity == "warning"
                ],
                **({"reference_similarity": chrf} if chrf is not None else {}),
            }
        )
    wall_sec = time.perf_counter() - started

    import transformers

    payload = {
        "generator": "stock_transformers",
        "reference_metric": _REFERENCE_METRIC,
        "model": args.model,
        "model_revision": model_revision,
        "resolved_model_commit": _resolved_commit(model, model_path),
        "local_files_only": local_only,
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        # decode_backend lets the gate's run_config_diff surface the intended,
        # expected difference (stock generate vs funyi's fixed_mask) as a warning.
        "decode_backend": "stock_generate",
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "generation": {
            "do_sample": False,
            "repetition_penalty": args.repetition_penalty,
            "extra_generate_kwargs": {},
        },
        "decode": "greedy",
        # Greedy bf16 argmax is not bit-reproducible across these versions; record
        # them so a golden/runtime environment mismatch is auditable.
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "case_count": len(rows),
        "load_wall_sec": round(load_wall_sec, 3),
        "generate_wall_sec": round(wall_sec, 3),
        "cases": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    error_cases = sum(1 for row in rows if row["errors"])
    print(
        f"wrote {len(rows)} golden cases to {out} | gross-error cases {error_cases} | generate {wall_sec:.1f}s",
        file=sys.stderr,
    )


@torch.inference_mode()
def _translate_stock(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    target_language: str,
    device: str,
    max_new_tokens: int,
    repetition_penalty: float,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[str, int]:
    prompt = build_hymt_prompt(text, target_language=target_language)
    # Identical tokenization to funyi's _tokenize_prompt, so the prompt is the same.
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(torch.device(device))
    generate_kwargs: dict[str, Any] = {
        "do_sample": False,  # greedy: argmax, fully deterministic, no seed
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "use_cache": True,
    }
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id
    outputs = model.generate(input_ids=input_ids, **generate_kwargs)
    new_tokens = outputs[0, input_ids.shape[-1] :].tolist()
    # Mirror funyi's runtime exactly (translation.py): decode the full new-token
    # slice with skip_special_tokens (which already drops pad/eos); do NOT
    # pre-strip pad. generated_tokens is the full slice length, so the gate's
    # max_new_tokens_hit check is computed the same way on both sides.
    output = str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()
    return output, len(new_tokens)


if __name__ == "__main__":
    main()
