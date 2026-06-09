# coding=utf-8
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)

import qwen3_asr_runtime.translation as translation_module
from qwen3_asr_runtime.translation import (
    DEFAULT_HYMT_ATTN_IMPLEMENTATION,
    DEFAULT_HYMT_DECODE_BACKEND,
    DEFAULT_HYMT_FUSED_RMSNORM,
    DEFAULT_HYMT_MAX_NEW_TOKENS,
    DEFAULT_HYMT_MODEL,
    DEFAULT_HYMT_W8A16,
    HYMTGenerationConfig,
    HYMTTranslator,
    _attention_mask_for_step,
    _build_static_sdpa_attention_masks,
    build_hymt_prompt,
    _fast_stop_eos_token_ids,
    _model_commit_hash,
    _normalize_model_revision,
    _resolve_model_path,
    _snapshot_commit_from_path,
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
        outputs = {
            (21, 22): " translated text ",
            (31, 32): " second translated ",
        }
        return outputs.get(tuple(self.decoded_ids), " translated text ")


class VariableLengthTokenizer(FakeTokenizer):
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
        content = str(messages[0]["content"])
        if "second" in content:
            return torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        return torch.tensor([[10, 11]], dtype=torch.long)


class ShortBatchDecodeTokenizer(FakeTokenizer):
    def batch_decode(
        self, rows: list[list[int]], *, skip_special_tokens: bool
    ) -> list[str]:
        del rows, skip_special_tokens
        return ["only one output"]


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generate_kwargs: dict[str, object] = {}
        self.input_ids: torch.Tensor | None = None

    def generate(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.generate_kwargs = dict(kwargs)
        self.input_ids = input_ids.detach().cpu()
        suffix = torch.tensor(
            [[21, 22], [31, 32]], dtype=torch.long, device=input_ids.device
        )[: input_ids.shape[0]]
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


class TestTranslationPrompt:
    def test_prompt_uses_one_template_for_chinese_source_text(self) -> None:
        prompt = build_hymt_prompt("今天天气很好。", target_language="English")

        assert "Translate the following segment into English" in prompt
        assert "keeping the original format" in prompt
        assert "今天天气很好。" in prompt

    def test_prompt_uses_one_template_for_non_chinese_pair(self) -> None:
        prompt = build_hymt_prompt(
            "It is on the house.", target_language="German", source_language="English"
        )

        assert "Translate the following segment into German" in prompt
        assert "without additional explanation" in prompt
        assert "It is on the house." in prompt

    def test_prompt_does_not_branch_by_source_language(self) -> None:
        prompt_en = build_hymt_prompt(
            "hello", target_language="Japanese", source_language="English"
        )
        prompt_zh = build_hymt_prompt(
            "hello", target_language="Japanese", source_language="Chinese"
        )

        assert prompt_en == prompt_zh


class TestHYMTTranslator:
    def test_default_model_is_hymt2_with_optimized_decode(self) -> None:
        assert DEFAULT_HYMT_MODEL == "tencent/Hy-MT2-1.8B"
        assert DEFAULT_HYMT_DECODE_BACKEND == "fixed_mask"
        assert DEFAULT_HYMT_W8A16
        assert DEFAULT_HYMT_FUSED_RMSNORM

    def test_injected_model_uses_default_optimization_profile(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        with (
            mock.patch(
                "qwen3_asr_runtime.quant_linears.patch_linears_w8a16",
                return_value=2,
            ) as patch_w8a16,
            mock.patch(
                "qwen3_asr_runtime.fused_rmsnorm.patch_model_rmsnorms",
                return_value=3,
            ) as patch_rmsnorm,
        ):
            translator = HYMTTranslator(
                "fake-model", device="cpu", model=model, tokenizer=tokenizer
            )

        assert translator.decode_backend == DEFAULT_HYMT_DECODE_BACKEND
        assert translator.w8a16 is DEFAULT_HYMT_W8A16
        assert translator.fused_rmsnorm is DEFAULT_HYMT_FUSED_RMSNORM
        patch_w8a16.assert_called_once_with(
            model, suffixes=("gate_proj", "up_proj"), prefill_gemm="cublas"
        )
        patch_rmsnorm.assert_called_once()
        assert translator.w8a16_patch_count == 2
        assert translator.fused_rmsnorm_patch_count == 3

    def test_pinned_transformers_has_native_hymt2_model_mapping(self) -> None:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        )

        assert CONFIG_MAPPING_NAMES["hunyuan_v1_dense"] == "HunYuanDenseV1Config"
        assert (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["hunyuan_v1_dense"]
            == "HunYuanDenseV1ForCausalLM"
        )

    def test_default_attention_implementation_is_sdpa(self) -> None:
        assert DEFAULT_HYMT_ATTN_IMPLEMENTATION == "sdpa"

    def test_default_max_new_tokens_matches_asr_default(self) -> None:
        assert DEFAULT_HYMT_MAX_NEW_TOKENS == 512
        assert HYMTGenerationConfig().max_new_tokens == DEFAULT_HYMT_MAX_NEW_TOKENS

    def test_default_generation_is_greedy(self) -> None:
        assert not HYMTGenerationConfig().do_sample

    def test_model_load_uses_default_attention_implementation(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        with (
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as from_tokenizer,
            mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as from_model,
            mock.patch(
                "qwen3_asr_runtime.translation._resolve_model_path",
                return_value="fake-model",
            ),
        ):
            HYMTTranslator("fake-model", device="cpu", w8a16=False, fused_rmsnorm=False)

        from_tokenizer.assert_called_once()
        from_model.assert_called_once()
        assert from_tokenizer.call_args.kwargs["trust_remote_code"] is False
        assert from_tokenizer.call_args.kwargs["fix_mistral_regex"] is True
        assert from_model.call_args.kwargs["trust_remote_code"] is False
        assert (
            from_model.call_args.kwargs["attn_implementation"]
            == DEFAULT_HYMT_ATTN_IMPLEMENTATION
        )

    def test_model_load_forwards_revision_and_records_commit(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        model.config = SimpleNamespace(_commit_hash="commit456")  # type: ignore[attr-defined]
        with (
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as from_tokenizer,
            mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as from_model,
            mock.patch(
                "qwen3_asr_runtime.translation._resolve_model_path",
                return_value="org/model",
            ),
        ):
            translator = HYMTTranslator(
                "org/model",
                device="cpu",
                model_revision=" abc123 ",
                local_files_only=False,
                w8a16=False,
                fused_rmsnorm=False,
            )

        assert translator.model_revision == "abc123"
        assert translator.resolved_model_commit == "commit456"
        assert from_tokenizer.call_args.kwargs["revision"] == "abc123"
        assert from_model.call_args.kwargs["revision"] == "abc123"

    def test_translate_decodes_only_generated_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            w8a16=False,
            fused_rmsnorm=False,
            generation_config=HYMTGenerationConfig(
                max_new_tokens=5,
                do_sample=False,
            ),
            model=model,
            tokenizer=tokenizer,
        )

        text = translator.translate(
            "It is on the house.", target_language="Chinese", max_new_tokens=7
        )
        result = translator.profile_translate(
            "It is on the house.", target_language="Chinese", max_new_tokens=7
        )

        assert text == "translated text"
        assert result.text == "translated text"
        assert result.prompt_tokens == 3
        assert result.generated_tokens == 2
        assert tokenizer.decoded_ids == [21, 22]
        assert not tokenizer.add_generation_prompt
        assert model.generate_kwargs["max_new_tokens"] == 7
        assert not model.generate_kwargs["do_sample"]
        assert model.generate_kwargs["cache_implementation"] == "static"
        assert model.generate_kwargs["logits_to_keep"] == 1
        assert "custom_generate" not in model.generate_kwargs
        assert model.generate_kwargs["top_k"] == 50
        assert model.generate_kwargs["top_p"] == 1.0
        assert model.generate_kwargs["temperature"] == 1.0

    def test_generate_backend_can_use_hf_generate(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            decode_backend="generate",
            w8a16=False,
            fused_rmsnorm=False,
            generation_config=HYMTGenerationConfig(max_new_tokens=5),
            model=model,
            tokenizer=tokenizer,
        )
        translator.translate("hello", target_language="Chinese")

        assert translator.decode_backend == "generate"
        assert "custom_generate" not in model.generate_kwargs

    def test_fixed_mask_backend_can_be_enabled(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            decode_backend="fixed_mask",
            w8a16=False,
            fused_rmsnorm=False,
            generation_config=HYMTGenerationConfig(max_new_tokens=5),
            model=model,
            tokenizer=tokenizer,
        )
        translator._can_use_fixed_mask_decode = lambda kwargs: True  # type: ignore[method-assign]

        translator.translate("hello", target_language="Chinese")

        assert translator.decode_backend == "fixed_mask"
        assert (
            model.generate_kwargs["custom_generate"]
            is translation_module._hymt_fixed_mask_generate
        )

    def test_load_disables_hymt_noop_dynamic_rope_update(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeHunyuanModel()
        rotary = model.model.rotary_emb

        HYMTTranslator(
            "fake-model",
            device="cpu",
            w8a16=False,
            fused_rmsnorm=False,
            model=model,
            tokenizer=tokenizer,
        )

        assert rotary.rope_type == "default"
        assert rotary._hymt_original_rope_type == "dynamic"

    def test_warmup_skips_empty_texts(self) -> None:
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            w8a16=False,
            fused_rmsnorm=False,
            generation_config=HYMTGenerationConfig(),
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
        )

        results = translator.warmup(
            ["", " first ", "second"], target_language="Chinese", max_new_tokens=4
        )

        assert len(results) == 2
        assert [result.text for result in results] == [
            "translated text",
            "translated text",
        ]

    def test_translate_batch_preserves_order_and_left_pads_prompts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenizer = VariableLengthTokenizer()
        clock = iter([0.0, 1.0, 1.1, 2.0, 2.25, 3.0, 3.05, 3.2])
        monkeypatch.setattr(
            translation_module.time, "perf_counter", lambda: next(clock)
        )
        model = FakeModel()
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            w8a16=False,
            fused_rmsnorm=False,
            generation_config=HYMTGenerationConfig(),
            model=model,
            tokenizer=tokenizer,
        )

        results = translator.profile_translate_batch(
            ["first", "", "second"],
            target_language="Chinese",
            max_new_tokens=4,
        )

        assert [result.text for result in results] == [
            "translated text",
            "",
            "second translated",
        ]
        assert [result.prompt_tokens for result in results] == [2, 0, 4]
        assert [result.generated_tokens for result in results] == [2, 0, 2]
        assert results[0].generate_wall_sec == 0.25
        assert results[0].generate_wall_sec == results[2].generate_wall_sec
        assert results[0].total_wall_sec == 3.2
        assert tokenizer.template_calls == 2
        assert model.input_ids is not None
        assert model.input_ids.tolist() == [[0, 0, 10, 11], [10, 11, 12, 13]]
        attention_mask = model.generate_kwargs["attention_mask"]
        assert isinstance(attention_mask, torch.Tensor)
        assert attention_mask.cpu().tolist() == [
            [False, False, True, True],
            [True, True, True, True],
        ]

    def test_translate_batch_rejects_decoder_output_count_mismatch(self) -> None:
        translator = HYMTTranslator(
            "fake-model",
            device="cpu",
            w8a16=False,
            fused_rmsnorm=False,
            generation_config=HYMTGenerationConfig(),
            model=FakeModel(),
            tokenizer=ShortBatchDecodeTokenizer(),
        )

        with pytest.raises(RuntimeError):
            translator.profile_translate_batch(
                ["first", "second"], target_language="Chinese"
            )


class TestHYMTFixedMaskDecode:
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

        assert masks is not None
        assert tuple(masks.shape) == (1, 1, 4, 5)
        assert masks[0, 0].tolist() == [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
        assert tuple(
            _attention_mask_for_step(masks, torch.empty(1, 4), 2, prefill=True).shape
        ) == (1, 1, 2, 5)
        assert tuple(
            _attention_mask_for_step(masks, torch.empty(1, 4), 3, prefill=False).shape
        ) == (1, 1, 1, 5)

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

        assert masks is None

    def test_fast_stop_eos_ids_accepts_single_eos_and_max_length_only(self) -> None:
        criteria = [
            MaxLengthCriteria(max_length=12),
            EosTokenCriteria(eos_token_id=torch.tensor([7])),
        ]

        assert _fast_stop_eos_token_ids(criteria) == frozenset({7})

    def test_fast_stop_eos_ids_rejects_multi_eos_or_unknown_criteria(self) -> None:
        assert (
            _fast_stop_eos_token_ids(
                [EosTokenCriteria(eos_token_id=torch.tensor([7, 8]))]
            )
            is None
        )
        assert _fast_stop_eos_token_ids([SimpleNamespace()]) is None


class TestHYMTModelPath:
    def test_empty_model_revision_is_none(self) -> None:
        assert _normalize_model_revision("  ") is None

    def test_local_files_keeps_existing_path(self, tmp_path: Path) -> None:
        model_path = str(tmp_path)

        assert _resolve_model_path(model_path, local_files_only=True) == model_path

    def test_model_revision_is_rejected_for_existing_local_path(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(
            ValueError, match="model_revision only applies to Hugging Face model ids"
        ):
            _resolve_model_path(
                str(tmp_path), local_files_only=True, model_revision="abc123"
            )

    def test_local_files_resolves_model_id_to_snapshot(self) -> None:
        snapshot_path = Path("/cache/snapshot")
        with mock.patch(
            "huggingface_hub.snapshot_download", return_value=snapshot_path
        ) as download:
            assert (
                _resolve_model_path("org/model", local_files_only=True)
                == str(snapshot_path)
            )

        download.assert_called_once_with(
            repo_id="org/model", revision=None, local_files_only=True
        )

    def test_local_files_resolves_model_id_to_pinned_snapshot(self) -> None:
        snapshot_path = Path("/cache/snapshot")
        with mock.patch(
            "huggingface_hub.snapshot_download", return_value=snapshot_path
        ) as download:
            assert (
                _resolve_model_path(
                    "org/model", local_files_only=True, model_revision="abc123"
                )
                == str(snapshot_path)
            )

        download.assert_called_once_with(
            repo_id="org/model", revision="abc123", local_files_only=True
        )

    def test_extracts_commit_from_model_or_snapshot_path(self) -> None:
        assert (
            _model_commit_hash(
                SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123"))
            )
            == "abc123"
        )
        assert (
            _snapshot_commit_from_path("/cache/models--org--model/snapshots/def456")
            == "def456"
        )
