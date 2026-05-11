# coding=utf-8
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch
from transformers.generation.stopping_criteria import EosTokenCriteria, MaxLengthCriteria

from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_ATTN_IMPLEMENTATION,
    DEFAULT_HYMT_DECODE_BACKEND,
    DEFAULT_HYMT_MAX_NEW_TOKENS,
    HYMTGenerationConfig,
    HYMTTranslator,
    _attention_mask_for_step,
    _build_static_sdpa_attention_masks,
    build_hymt_prompt,
    _fast_stop_eos_token_ids,
    _resolve_model_path,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.decoded_ids: list[int] = []
        self.template_calls = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> torch.Tensor:
        self.template_calls += 1
        self.messages = list(messages)
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.return_tensors = return_tensors
        return torch.tensor([[10, 11, 12]], dtype=torch.long)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.decoded_ids = [int(item) for item in token_ids]
        self.skip_special_tokens = skip_special_tokens
        return " translated text "


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.generate_kwargs = dict(kwargs)
        suffix = torch.tensor([[21, 22]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class FakeHunyuanModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="hunyuan_v1_dense")
        self.model = SimpleNamespace(
            rotary_emb=SimpleNamespace(
                rope_type="dynamic",
                max_seq_len_cached=262144,
                original_max_seq_len=262144,
                config=SimpleNamespace(max_position_embeddings=262144),
            )
        )


class TranslationPromptTest(unittest.TestCase):
    def test_prompt_uses_one_template_for_chinese_source_text(self) -> None:
        prompt = build_hymt_prompt("今天天气很好。", target_language="English")

        self.assertIn("Translate the following segment into English", prompt)
        self.assertIn("keeping the original format", prompt)
        self.assertIn("今天天气很好。", prompt)

    def test_prompt_uses_one_template_for_non_chinese_pair(self) -> None:
        prompt = build_hymt_prompt("It is on the house.", target_language="German", source_language="English")

        self.assertIn("Translate the following segment into German", prompt)
        self.assertIn("without additional explanation", prompt)
        self.assertIn("It is on the house.", prompt)

    def test_prompt_does_not_branch_by_source_language(self) -> None:
        prompt_en = build_hymt_prompt("hello", target_language="Japanese", source_language="English")
        prompt_zh = build_hymt_prompt("hello", target_language="Japanese", source_language="Chinese")

        self.assertEqual(prompt_en, prompt_zh)


class HYMTTranslatorTest(unittest.TestCase):
    def test_default_attention_implementation_is_sdpa(self) -> None:
        self.assertEqual(DEFAULT_HYMT_ATTN_IMPLEMENTATION, "sdpa")
        self.assertEqual(DEFAULT_HYMT_DECODE_BACKEND, "fixed_mask")

    def test_default_max_new_tokens_matches_asr_default(self) -> None:
        self.assertEqual(DEFAULT_HYMT_MAX_NEW_TOKENS, 512)
        self.assertEqual(HYMTGenerationConfig().max_new_tokens, DEFAULT_HYMT_MAX_NEW_TOKENS)

    def test_model_load_uses_default_attention_implementation(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as from_tokenizer, mock.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as from_model, mock.patch(
            "qwen3_asr_runtime.translation._resolve_model_path",
            return_value="fake-model",
        ):
            HYMTTranslator("fake-model", device="cpu")

        from_tokenizer.assert_called_once()
        from_model.assert_called_once()
        self.assertEqual(from_model.call_args.kwargs["attn_implementation"], DEFAULT_HYMT_ATTN_IMPLEMENTATION)

    def test_translate_decodes_only_generated_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            generation_config=HYMTGenerationConfig(
                max_new_tokens=5,
                do_sample=False,
            ),
            model=model,
            tokenizer=tokenizer,
        )

        text = translator.translate("It is on the house.", target_language="Chinese", max_new_tokens=7)
        result = translator.profile_translate("It is on the house.", target_language="Chinese", max_new_tokens=7)

        self.assertEqual(text, "translated text")
        self.assertEqual(result.text, "translated text")
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.generated_tokens, 2)
        self.assertEqual(tokenizer.decoded_ids, [21, 22])
        self.assertFalse(tokenizer.add_generation_prompt)
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 7)
        self.assertFalse(model.generate_kwargs["do_sample"])
        self.assertEqual(model.generate_kwargs["cache_implementation"], "static")
        self.assertEqual(model.generate_kwargs["logits_to_keep"], 1)
        self.assertNotIn("custom_generate", model.generate_kwargs)
        self.assertNotIn("temperature", model.generate_kwargs)

    def test_generate_backend_can_use_hf_generate(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            decode_backend="generate",
            generation_config=HYMTGenerationConfig(max_new_tokens=5),
            model=model,
            tokenizer=tokenizer,
        )

        translator.translate("hello", target_language="Chinese")

        self.assertEqual(translator.decode_backend, "generate")
        self.assertNotIn("custom_generate", model.generate_kwargs)

    def test_load_disables_hymt_noop_dynamic_rope_update(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeHunyuanModel()
        rotary = model.model.rotary_emb

        HYMTTranslator("fake-model", device="cpu", model=model, tokenizer=tokenizer)

        self.assertEqual(rotary.rope_type, "default")
        self.assertEqual(rotary._hymt_original_rope_type, "dynamic")

    def test_warmup_skips_empty_texts(self) -> None:
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            generation_config=HYMTGenerationConfig(),
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
        )

        results = translator.warmup(["", " first ", "second"], target_language="Chinese", max_new_tokens=4)

        self.assertEqual(len(results), 2)
        self.assertEqual([result.text for result in results], ["translated text", "translated text"])


class HYMTFixedMaskDecodeTest(unittest.TestCase):
    def test_static_sdpa_attention_masks_are_causal_4d_views(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))

        masks = _build_static_sdpa_attention_masks(
            model=model,
            batch_size=1,
            max_length=4,
            cache_max_length=5,
            prompt_len=2,
            source_attention_mask=None,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(masks)
        assert masks is not None
        self.assertEqual(tuple(masks.shape), (1, 1, 4, 5))
        self.assertEqual(
            masks[0, 0].tolist(),
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
            ],
        )
        self.assertEqual(
            tuple(_attention_mask_for_step(masks, torch.empty(1, 4), 2, prefill=True).shape),
            (1, 1, 2, 5),
        )
        self.assertEqual(
            tuple(_attention_mask_for_step(masks, torch.empty(1, 4), 3, prefill=False).shape),
            (1, 1, 1, 5),
        )

    def test_static_sdpa_attention_masks_fall_back_for_padded_prompts(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))

        masks = _build_static_sdpa_attention_masks(
            model=model,
            batch_size=1,
            max_length=4,
            cache_max_length=4,
            prompt_len=3,
            source_attention_mask=torch.tensor([[True, False, True]]),
            device=torch.device("cpu"),
        )

        self.assertIsNone(masks)

    def test_fast_stop_eos_ids_accepts_single_eos_and_max_length_only(self) -> None:
        criteria = [
            MaxLengthCriteria(max_length=12),
            EosTokenCriteria(eos_token_id=torch.tensor([7])),
        ]

        self.assertEqual(_fast_stop_eos_token_ids(criteria), frozenset({7}))

    def test_fast_stop_eos_ids_rejects_multi_eos_or_unknown_criteria(self) -> None:
        self.assertIsNone(_fast_stop_eos_token_ids([EosTokenCriteria(eos_token_id=torch.tensor([7, 8]))]))
        self.assertIsNone(_fast_stop_eos_token_ids([SimpleNamespace()]))


class HYMTModelPathTest(unittest.TestCase):
    def test_local_files_keeps_existing_path(self) -> None:
        with TemporaryDirectory() as directory:
            self.assertEqual(_resolve_model_path(directory, local_files_only=True), directory)

    def test_local_files_resolves_model_id_to_snapshot(self) -> None:
        with mock.patch("huggingface_hub.snapshot_download", return_value=Path("/cache/snapshot")) as download:
            self.assertEqual(_resolve_model_path("org/model", local_files_only=True), "/cache/snapshot")

        download.assert_called_once_with(repo_id="org/model", local_files_only=True)


if __name__ == "__main__":
    unittest.main()
