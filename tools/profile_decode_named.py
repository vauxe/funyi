# coding=utf-8
"""
Profile decode step with record_function spans on named sub-blocks of the
text decoder so we know which block (rotary, rmsnorm, attention, mlp,
lm_head, cache.update) dominates.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_asr_runtime import Qwen3ASRModel
from transformers.cache_utils import DynamicCache


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--fused-rmsnorm", action="store_true")
    parser.add_argument("--fused-linears", action="store_true")
    parser.add_argument("--quantized-linears", action="store_true")
    parser.add_argument("--audio", required=True, help="Local 16 kHz audio file used to build the profile prompt.")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    cli = parser.parse_args()
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        max_new_tokens=64,
    )
    if cli.flashinfer:
        kwargs["flashinfer"] = True
    if cli.fused_rmsnorm:
        kwargs["fused_rmsnorm"] = True
    if cli.fused_linears:
        kwargs["fused_linears"] = True
    if cli.quantized_linears:
        kwargs["quantized_linears"] = True
    model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", **kwargs)
    model.eval()
    hf = model.backend_runtime.model
    thinker = hf.thinker

    # Wrap known submodules with record_function spans.
    def wrap_call(obj, name):
        orig = obj.forward
        def patched(*args, **kwargs):
            with record_function(name):
                return orig(*args, **kwargs)
        obj.forward = patched

    wrap_call(thinker.model.rotary_emb, "rotary_emb")
    wrap_call(thinker.lm_head, "lm_head")
    for i, layer in enumerate(thinker.model.layers):
        wrap_call(layer.input_layernorm, "rmsnorm")
        wrap_call(layer.post_attention_layernorm, "rmsnorm")
        wrap_call(layer.self_attn.q_norm, "qk_norm")
        wrap_call(layer.self_attn.k_norm, "qk_norm")
        if hasattr(layer.self_attn, "qkv_proj"):
            wrap_call(layer.self_attn.qkv_proj, "qkv_proj")
        else:
            wrap_call(layer.self_attn.q_proj, "q_proj")
            wrap_call(layer.self_attn.k_proj, "k_proj")
            wrap_call(layer.self_attn.v_proj, "v_proj")
        wrap_call(layer.self_attn.o_proj, "o_proj")
        wrap_call(layer.self_attn, "self_attn")
        if hasattr(layer.mlp, "gate_up_proj"):
            wrap_call(layer.mlp.gate_up_proj, "gate_up_proj")
        else:
            wrap_call(layer.mlp.gate_proj, "gate_proj")
            wrap_call(layer.mlp.up_proj, "up_proj")
        wrap_call(layer.mlp.down_proj, "down_proj")
        wrap_call(layer.mlp, "mlp")

    processor = model.backend_runtime.processor
    sample_rate = 16000
    wav, sr = sf.read(
        cli.audio,
        start=int(round(cli.start_sec * sample_rate)),
        frames=int(round(cli.duration_sec * sample_rate)),
        dtype="float32",
    )
    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz audio, got {sr} from {cli.audio}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    prompt = model._build_text_prompt(context="", force_language=None)
    inputs = processor(text=[prompt], audio=[wav.astype(np.float32)],
                       return_tensors="pt", padding=True).to("cuda:0").to(torch.bfloat16)
    prompt_len = int(inputs["input_ids"].shape[1])

    thinker.rope_deltas = None
    with torch.inference_mode():
        cache = DynamicCache()
        out = thinker(
            input_ids=inputs["input_ids"],
            input_features=inputs.get("input_features"),
            attention_mask=inputs["attention_mask"],
            feature_attention_mask=inputs.get("feature_attention_mask"),
            past_key_values=cache,
            cache_position=torch.arange(prompt_len, device="cuda:0"),
            use_cache=True,
            return_dict=True,
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
        attention_mask = inputs["attention_mask"]
        rope_deltas = thinker.rope_deltas

        for _ in range(3):  # warmup
            cur_len = int(attention_mask.shape[1])
            cp = torch.tensor([cur_len], device="cuda:0", dtype=torch.long)
            pd = torch.arange(1, device="cuda:0").view(1, -1).add(
                cp[0] + rope_deltas.view(-1)[0]).unsqueeze(0).expand(3, 1, 1)
            out = thinker(input_ids=next_tok, attention_mask=attention_mask, position_ids=pd,
                          past_key_values=cache, cache_position=cp, use_cache=True, return_dict=True)
            next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
        torch.cuda.synchronize()

        N = 20
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(N):
                cur_len = int(attention_mask.shape[1])
                cp = torch.tensor([cur_len], device="cuda:0", dtype=torch.long)
                pd = torch.arange(1, device="cuda:0").view(1, -1).add(
                    cp[0] + rope_deltas.view(-1)[0]).unsqueeze(0).expand(3, 1, 1)
                with record_function("STEP"):
                    out = thinker(input_ids=next_tok, attention_mask=attention_mask, position_ids=pd,
                                  past_key_values=cache, cache_position=cp, use_cache=True, return_dict=True)
                    next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
            torch.cuda.synchronize()

    # Filter to our named regions + STEP
    events = prof.key_averages()
    # Also dump the top aten ops globally to account for the 'missing' 40%.
    print("=== TOP ATEN OPS BY CUDA TIME ===")
    all_rows = sorted(events, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))
    for e in all_rows[:25]:
        cu = getattr(e, "self_device_time_total", 0) or getattr(e, "cuda_time_total", 0)
        print(f"  {e.key[:40]:40s}  cuda_ms={cu/1000:.2f}  count={e.count}")
    print()
    wanted = {"STEP", "rotary_emb", "lm_head", "rmsnorm", "qk_norm", "self_attn", "mlp",
              "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    rows = [e for e in events if e.key in wanted]
    def cuda_us(e):
        # Works on both FunctionEventAvg and FunctionEvent
        for attr in ("self_device_time_total", "device_time_total", "self_cuda_time_total", "cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return v
        return 0.0

    rows.sort(key=lambda e: -cuda_us(e))
    print(f"{'region':12s} {'cuda_ms_total':>14s} {'cuda_ms_avg':>12s} {'cpu_ms_total':>13s} {'count':>6s}")
    for e in rows:
        cu = cuda_us(e)
        cp = getattr(e, "cpu_time_total", 0.0) or getattr(e, "self_cpu_time_total", 0.0)
        print(f"{e.key:12s} {cu/1000:>14.2f} {cu/max(1,e.count)/1000:>12.3f} "
              f"{cp/1000:>13.2f} {e.count:>6d}")


if __name__ == "__main__":
    main()
