# coding=utf-8
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from qwen3_asr_runtime.model import Qwen3ASRModel
from qwen3_asr_runtime.spec_decode import spec_decode_generate


class FakeBackend:
    name = "fake"
    model = None
    processor = None
    device = None
    dtype = None

    def __init__(self) -> None:
        self.draft_calls: list[dict[str, object]] = []
        self.prompt_calls: list[str] = []

    def eval(self) -> None:
        return None

    def reset_decode_runtime(self) -> None:
        return None

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool) -> str:
        del messages, add_generation_prompt, tokenize
        return "prompt:"

    def encode_text(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode_text(self, token_ids) -> str:
        return "".join("G" if int(tok) == 9001 else chr(int(tok)) for tok in token_ids)

    def infer_streaming_with_draft(self, prompt: str, wav: np.ndarray, draft_ids, *, max_new_tokens: int, stats=None) -> str:
        del wav, max_new_tokens
        self.draft_calls.append({"prompt": prompt, "draft_ids": list(draft_ids)})
        if stats is not None:
            stats["draft_tokens"] = len(list(draft_ids))
            stats["accepted_tokens"] = 1
        return "G"

    def infer_with_prompts(self, prompts: list[str], wavs: list[np.ndarray], *, max_inference_batch_size: int, max_new_tokens: int) -> list[str]:
        del wavs, max_inference_batch_size, max_new_tokens
        self.prompt_calls.extend(prompts)
        return ["F" for _ in prompts]


class StreamingSpecDraftTest(unittest.TestCase):
    def test_trimmed_prefix_still_reuses_rollback_token_ids(self) -> None:
        backend = FakeBackend()
        model = Qwen3ASRModel(backend_runtime=backend, max_new_tokens=8)
        state = model.init_streaming_state(
            language="Chinese",
            unfixed_chunk_num=0,
            unfixed_token_num=2,
            chunk_size_sec=1.0,
            max_prefix_tokens=3,
            spec_decode=True,
        )
        state._raw_decoded = "abcdef"

        model._run_streaming_decode_step(state)

        self.assertEqual(len(backend.draft_calls), 1)
        self.assertEqual(backend.draft_calls[0]["prompt"], state.prompt_raw + "bcd")
        self.assertEqual(backend.draft_calls[0]["draft_ids"], [ord("e"), ord("f")])
        self.assertEqual(backend.prompt_calls, [])
        self.assertEqual(state.committed_text, "a")
        self.assertEqual(state._raw_decoded, "bcdG")
        self.assertEqual(state.text, "abcdG")
        self.assertEqual(state.spec_decode_stats["spec_attempt_steps"], 1)
        self.assertEqual(state.spec_decode_stats["spec_trimmed_attempt_steps"], 1)
        self.assertEqual(state.spec_decode_stats["spec_accepted_tokens"], 1)


class FakeSpecThinker:
    generation_config = SimpleNamespace(eos_token_id=[])
    config = SimpleNamespace(eos_token_id=[])

    def __init__(self, preds: list[int]) -> None:
        self.preds = preds
        self.rope_deltas = None

    def __call__(self, *args, **kwargs):
        del args
        keep = int(kwargs["logits_to_keep"])
        vocab = max(self.preds[:keep]) + 1
        logits = torch.full((1, keep, vocab), -1000.0)
        for idx, tok in enumerate(self.preds[:keep]):
            logits[0, idx, tok] = 0.0
        return SimpleNamespace(logits=logits)


class SpecDecodeBudgetTest(unittest.TestCase):
    def test_draft_is_clipped_to_max_new_tokens_before_acceptance(self) -> None:
        thinker = FakeSpecThinker(preds=[3, 4, 99])
        out = spec_decode_generate(
            thinker,
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            input_features=None,
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            feature_attention_mask=None,
            draft_ids=[3, 4, 5],
            max_new_tokens=2,
        )

        self.assertEqual(out.tolist(), [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
