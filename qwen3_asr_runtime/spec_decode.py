# coding=utf-8
"""Speculative verification for streaming decode.

Streaming re-runs the model every step with a small rollback prefix. Most of
the rolled-back tokens get re-generated identically, so instead of *re-running
decode* for them, we append them to the prompt as a draft, do one prefill
that covers ``prompt + draft`` positions, and compare each draft position's
argmax against the draft token. Accepted tokens cost nothing beyond the extra
prefill positions; the first rejected position gives us one "free" correct
token from the verifier's logits, and regular decode continues from there.

Not byte-identical under bf16. Prefill-path KV (from the one-shot forward over
``prompt + draft``) differs from decode-path KV (from feeding ``draft`` one
token at a time) by bf16 ε. That drift propagates through the text decoder's
28 layers and can flip argmax at low-margin positions -- observed as the
occasional homophone or punctuation swap vs a plain decode. Validate quality
with a local CER sweep before enabling new presets by default.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
from transformers.cache_utils import DynamicCache

from .decode_runtime import _resolve_eos_token_ids


@torch.inference_mode()
def spec_decode_generate(
    thinker: Any,
    *,
    input_ids: torch.Tensor,            # [1, prompt_len] (prompt only)
    input_features: Optional[torch.Tensor],
    attention_mask: torch.Tensor,       # [1, prompt_len]
    feature_attention_mask: Optional[torch.Tensor],
    draft_ids: Sequence[int],           # token ids to verify; must be non-empty
    max_new_tokens: int,
    eos_token_id: Any = None,
    stats: Optional[dict[str, int]] = None,
) -> torch.Tensor:
    """Greedy decode with speculative verification of ``draft_ids``.

    Returns ``[1, prompt_len + n_generated]`` ids. The ``draft_ids`` are NOT
    included in the returned sequence unless they were "generated" by the
    verifier (i.e. accepted).

    ``draft_ids`` must be non-empty. Empty-draft callers should route to the
    backend's ordinary generate path instead so that numerical behaviour stays
    consistent with non-spec streaming steps.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("spec_decode_generate currently supports batch size 1.")
    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return _build_output(input_ids, [])
    draft = [int(x) for x in draft_ids][:max_new_tokens]
    K = len(draft)
    if K == 0:
        raise ValueError("spec_decode_generate requires a non-empty draft.")
    eos_set = set(_resolve_eos_token_ids(thinker, eos_token_id))

    # Reset rope_deltas so prefill recomputes them.
    thinker.rope_deltas = None

    # Extend input_ids and attention_mask with draft tokens so prefill covers
    # prompt + draft with the usual causal mask.
    draft_t = torch.tensor([draft], dtype=input_ids.dtype, device=device)
    ext_input_ids = torch.cat([input_ids, draft_t], dim=1)
    ext_attention_mask = torch.cat(
        [attention_mask,
         torch.ones((1, K), dtype=attention_mask.dtype, device=device)],
        dim=1,
    )
    ext_len = prompt_len + K

    cache = DynamicCache()
    cache_position = torch.arange(ext_len, device=device)
    out = thinker(
        input_ids=ext_input_ids,
        input_features=input_features,
        attention_mask=ext_attention_mask,
        feature_attention_mask=feature_attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        logits_to_keep=K + 1,
        use_cache=True,
        return_dict=True,
    )
    # logits shape [1, K+1, V]
    # Positions of interest: [prompt_len - 1 .. ext_len - 1]. These give the
    # model's argmax for "next token at position prompt_len..ext_len" i.e. the
    # K+1 candidates over (draft[0], draft[1], ..., draft[K-1], next-after-K).
    verify_logits = out.logits[0, :, :]                         # [K+1, V]
    verify_argmax = verify_logits.argmax(dim=-1)                # [K+1]
    preds = verify_argmax.tolist()

    # Compare preds[0..K-1] against draft[0..K-1].
    accepted = 0
    for j in range(K):
        if preds[j] == draft[j]:
            accepted += 1
        else:
            break

    if stats is not None:
        stats["draft_tokens"] = K
        stats["accepted_tokens"] = accepted

    # After acceptance: generated so far = draft[:accepted]. Next token is
    # preds[accepted] (the verifier's choice at the first rejected position,
    # or -- if all accepted -- the bonus token beyond the draft).
    generated: List[int] = list(draft[:accepted])
    if len(generated) >= max_new_tokens:
        return _build_output(input_ids, generated)
    next_id = int(preds[accepted])

    # Check EOS / budget before proceeding.
    if next_id in eos_set:
        return _build_output(input_ids, generated)
    generated.append(next_id)
    if len(generated) >= max_new_tokens:
        return _build_output(input_ids, generated)

    # Rope deltas produced during prefill above. Use them for the decode loop.
    rope_deltas = thinker.rope_deltas
    if rope_deltas is None:
        raise RuntimeError("thinker.rope_deltas missing after prefill; cannot continue decode.")

    # Crop cache to the point right after the accepted prefix. That position
    # is (prompt_len + accepted). The bonus/reject token (next_id) has not
    # been put through the model yet -- it'll be the next decode input.
    effective_len = prompt_len + accepted
    cache.crop(effective_len)

    # Build running attention_mask that matches the kept cache length plus
    # however many decode tokens we've generated. For the decode loop we grow
    # it one slot per step.
    running_mask = torch.ones((1, effective_len), dtype=attention_mask.dtype, device=device)

    # Decode loop -- same structure as a plain HF causal LM generate.
    cur_len = effective_len  # cache length seen so far (not counting next_id)
    next_input = torch.tensor([[next_id]], dtype=input_ids.dtype, device=device)
    while len(generated) < max_new_tokens:
        cur_len += 1
        running_mask = torch.cat(
            [running_mask, torch.ones((1, 1), dtype=running_mask.dtype, device=device)],
            dim=1,
        )
        pos_id_scalar = int(cur_len - 1) + int(rope_deltas.view(-1)[0].item())
        pos_ids = torch.full((3, 1, 1), pos_id_scalar, dtype=torch.long, device=device)
        cp = torch.tensor([cur_len - 1], dtype=torch.long, device=device)
        step = thinker(
            input_ids=next_input,
            attention_mask=running_mask,
            position_ids=pos_ids,
            past_key_values=cache,
            cache_position=cp,
            use_cache=True,
            return_dict=True,
        )
        tok = int(step.logits[0, -1, :].argmax(dim=-1).item())
        if tok in eos_set:
            break
        generated.append(tok)
        next_input = torch.tensor([[tok]], dtype=input_ids.dtype, device=device)
    return _build_output(input_ids, generated)


def _build_output(input_ids: torch.Tensor, generated: Sequence[int]) -> torch.Tensor:
    prompt_len = input_ids.shape[1]
    total = prompt_len + len(generated)
    out = torch.empty(1, total, dtype=input_ids.dtype, device=input_ids.device)
    out[:, :prompt_len] = input_ids
    if generated:
        out[0, prompt_len:] = torch.tensor(list(generated), dtype=input_ids.dtype, device=input_ids.device)
    return out


__all__ = ["spec_decode_generate"]
