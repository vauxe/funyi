from types import SimpleNamespace
from unittest.mock import patch

import torch

from qwen3_asr_runtime.backends.transformers import TransformersASRBackend
from qwen3_asr_runtime.decode_runtime import CudaGraphCaptureRequired
from qwen3_asr_runtime.hf_qwen3_asr.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen3_asr_runtime.model import Qwen3ASRModel


def _fake_model() -> SimpleNamespace:
    text_config = SimpleNamespace(_attn_implementation="sdpa")
    audio_config = SimpleNamespace(_attn_implementation="sdpa")
    thinker_config = SimpleNamespace(
        _attn_implementation="sdpa",
        text_config=text_config,
        audio_config=audio_config,
    )
    thinker = SimpleNamespace(
        config=thinker_config,
        model=SimpleNamespace(config=text_config),
        audio_tower=SimpleNamespace(config=audio_config),
    )
    return SimpleNamespace(
        config=SimpleNamespace(thinker_config=thinker_config),
        thinker=thinker,
        device=torch.device("cpu"),
        dtype=torch.float32,
        eval=lambda: None,
        requires_grad_=lambda _: None,
    )


class _FakeGenerateProcessor:
    def __init__(self, *, pad_token_id: int | None) -> None:
        self.tokenizer = SimpleNamespace(pad_token_id=pad_token_id)

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        del messages, add_generation_prompt, tokenize
        return "prompt:"

    def __call__(
        self,
        *,
        text: list[str],
        audio: list[object],
        return_tensors: str,
        padding: bool,
    ) -> dict:
        del text, audio, return_tensors, padding
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def batch_decode(
        self,
        sequences: torch.Tensor,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        del sequences, skip_special_tokens, clean_up_tokenization_spaces
        return ["decoded"]


class _RecordingGenerateModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.generate_kwargs: dict | None = None

    def generate(self, **kwargs: object) -> SimpleNamespace:
        self.generate_kwargs = dict(kwargs)
        return SimpleNamespace(sequences=torch.tensor([[1, 2, 3]], dtype=torch.long))


class _FailingCudaGraphDecoder:
    def generate(self, **kwargs: object) -> torch.Tensor:
        del kwargs
        raise CudaGraphCaptureRequired()


class _RecordingThinker:
    def __init__(self) -> None:
        self.generate_kwargs: dict | None = None

    def generate(self, **kwargs: object) -> SimpleNamespace:
        self.generate_kwargs = dict(kwargs)
        return SimpleNamespace(sequences=torch.tensor([[1, 2]], dtype=torch.long))


class TestTransformersBackendAttention:
    def test_flashinfer_routes_thinker_configs_through_flashinfer_dispatcher(self) -> None:
        model = _fake_model()

        with (
            patch("qwen3_asr_runtime.backends.transformers.register_flashinfer", return_value=True),
            patch.object(TransformersASRBackend, "_default_attn_implementation", return_value="flash_attention_2"),
            patch(
                "qwen3_asr_runtime.backends.transformers.AutoModel.from_pretrained",
                return_value=model,
            ) as load_model,
            patch("qwen3_asr_runtime.backends.transformers.AutoProcessor.from_pretrained", return_value=object()),
        ):
            TransformersASRBackend.from_pretrained("dummy-model", flashinfer=True)

        assert load_model.call_args.kwargs['attn_implementation'] == 'flashinfer'
        assert model.thinker.config._attn_implementation == 'flashinfer'
        assert model.thinker.model.config._attn_implementation == 'flashinfer'
        assert model.thinker.audio_tower.config._attn_implementation == 'flashinfer'

    def test_flashinfer_overrides_explicit_attention_backend(self) -> None:
        model = _fake_model()

        with (
            patch("qwen3_asr_runtime.backends.transformers.register_flashinfer", return_value=True),
            patch(
                "qwen3_asr_runtime.backends.transformers.AutoModel.from_pretrained",
                return_value=model,
            ) as load_model,
            patch("qwen3_asr_runtime.backends.transformers.AutoProcessor.from_pretrained", return_value=object()),
        ):
            TransformersASRBackend.from_pretrained(
                "dummy-model",
                flashinfer=True,
                attn_implementation="flash_attention_2",
            )

        assert load_model.call_args.kwargs['attn_implementation'] == 'flashinfer'
        assert model.thinker.model.config._attn_implementation == 'flashinfer'
        assert model.thinker.audio_tower.config._attn_implementation == 'flashinfer'


class TestTransformersBackendGeneration:
    def test_infer_with_prompts_passes_tokenizer_pad_token_to_hf_generate(self) -> None:
        model = _RecordingGenerateModel()
        backend = TransformersASRBackend(model=model, processor=_FakeGenerateProcessor(pad_token_id=4321))

        assert backend.infer_with_prompts(
            ["prompt"],
            [object()],
            max_inference_batch_size=1,
            max_new_tokens=4,
        ) == ["decoded"]
        assert model.generate_kwargs is not None
        assert model.generate_kwargs["pad_token_id"] == 4321
        assert model.generate_kwargs["max_new_tokens"] == 4
        assert model.generate_kwargs["logits_to_keep"] == 1

    def test_cuda_graph_capture_fallback_passes_pad_token_to_hf_generate(self) -> None:
        model = _RecordingGenerateModel()
        backend = TransformersASRBackend(model=model, processor=_FakeGenerateProcessor(pad_token_id=3141))
        backend._cuda_graph_decoder = _FailingCudaGraphDecoder()

        assert backend.infer_with_prompts(
            ["prompt"],
            [object()],
            max_inference_batch_size=1,
            max_new_tokens=9,
        ) == ["decoded"]
        assert model.generate_kwargs is not None
        assert model.generate_kwargs["pad_token_id"] == 3141
        assert model.generate_kwargs["max_new_tokens"] == 9
        assert model.generate_kwargs["logits_to_keep"] == 1

    def test_resolves_pad_token_from_generation_config_when_tokenizer_has_none(self) -> None:
        model = SimpleNamespace(generation_config=SimpleNamespace(pad_token_id=9876))
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None))

        assert TransformersASRBackend._resolve_generation_token_kwargs(model, processor) == {
            "pad_token_id": 9876,
        }

    def test_resolves_pad_token_from_first_eos_when_no_pad_token_exists(self) -> None:
        model = SimpleNamespace(generation_config=SimpleNamespace(pad_token_id=None, eos_token_id=[4444, 5555]))
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=None))

        assert TransformersASRBackend._resolve_generation_token_kwargs(model, processor) == {
            "pad_token_id": 4444,
        }

    def test_resolves_pad_token_from_upstream_eos_default_when_no_token_config_exists(self) -> None:
        model = SimpleNamespace()
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=None))

        assert TransformersASRBackend._resolve_generation_token_kwargs(model, processor) == {
            "pad_token_id": 151645,
        }

    def test_infer_with_prompts_resolves_pad_token_at_generate_time(self) -> None:
        model = _RecordingGenerateModel()
        model.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=[4444, 5555])
        processor = _FakeGenerateProcessor(pad_token_id=None)
        backend = TransformersASRBackend(model=model, processor=processor)

        processor.tokenizer.pad_token_id = 2222

        assert backend.infer_with_prompts(
            ["prompt"],
            [object()],
            max_inference_batch_size=1,
            max_new_tokens=4,
        ) == ["decoded"]
        assert model.generate_kwargs is not None
        assert model.generate_kwargs["pad_token_id"] == 2222

    def test_infer_with_prompts_prefers_model_config_pad_before_thinker_pad(self) -> None:
        model = _RecordingGenerateModel()
        model.generation_config = SimpleNamespace(pad_token_id=None)
        model.config = SimpleNamespace(pad_token_id=1111)
        model.thinker = SimpleNamespace(
            generation_config=SimpleNamespace(pad_token_id=2222),
            config=SimpleNamespace(pad_token_id=3333),
        )
        backend = TransformersASRBackend(model=model, processor=_FakeGenerateProcessor(pad_token_id=None))

        assert backend.infer_with_prompts(
            ["prompt"],
            [object()],
            max_inference_batch_size=1,
            max_new_tokens=4,
        ) == ["decoded"]
        assert model.generate_kwargs is not None
        assert model.generate_kwargs["pad_token_id"] == 1111

    def test_resolves_pad_token_from_model_config_eos_before_thinker_eos(self) -> None:
        model = SimpleNamespace(
            generation_config=SimpleNamespace(pad_token_id=None, eos_token_id=None),
            config=SimpleNamespace(pad_token_id=None, eos_token_id=[1111, 1212]),
            thinker=SimpleNamespace(
                generation_config=SimpleNamespace(pad_token_id=None, eos_token_id=[2222, 2323]),
                config=SimpleNamespace(pad_token_id=None, eos_token_id=[3333, 3434]),
            ),
        )
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=None))

        assert TransformersASRBackend._resolve_generation_token_kwargs(model, processor) == {
            "pad_token_id": 1111,
        }

    def test_streaming_transcribe_passes_pad_token_to_hf_generate(self) -> None:
        backend_model = _RecordingGenerateModel()
        model = Qwen3ASRModel(model=backend_model, processor=_FakeGenerateProcessor(pad_token_id=1357))
        state = model.init_streaming_state(chunk_size_sec=1.0)

        model.streaming_transcribe(torch.ones(16_000).numpy(), state)

        assert backend_model.generate_kwargs is not None
        assert backend_model.generate_kwargs["pad_token_id"] == 1357

    def test_qwen3_asr_generate_forwards_generation_config_pad_token(self) -> None:
        thinker = _RecordingThinker()
        model = SimpleNamespace(
            thinker=thinker,
            generation_config=SimpleNamespace(pad_token_id=2468),
            config=SimpleNamespace(pad_token_id=None),
        )

        Qwen3ASRForConditionalGeneration.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
        )

        assert thinker.generate_kwargs is not None
        assert thinker.generate_kwargs["pad_token_id"] == 2468

    def test_qwen3_asr_generate_uses_call_generation_config_pad_before_eos_fallback(self) -> None:
        thinker = _RecordingThinker()
        model = SimpleNamespace(
            thinker=thinker,
            generation_config=SimpleNamespace(pad_token_id=None),
            config=SimpleNamespace(pad_token_id=None),
        )

        Qwen3ASRForConditionalGeneration.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            eos_token_id=[13579, 24680],
            generation_config=SimpleNamespace(pad_token_id=7777),
        )

        assert thinker.generate_kwargs is not None
        assert thinker.generate_kwargs["pad_token_id"] == 7777

    def test_qwen3_asr_generate_uses_call_generation_config_eos_for_pad_fallback(self) -> None:
        thinker = _RecordingThinker()
        model = SimpleNamespace(
            thinker=thinker,
            generation_config=SimpleNamespace(pad_token_id=None, eos_token_id=None),
            config=SimpleNamespace(pad_token_id=None, eos_token_id=None),
        )

        Qwen3ASRForConditionalGeneration.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            generation_config=SimpleNamespace(pad_token_id=None, eos_token_id=[7777, 8888]),
        )

        assert thinker.generate_kwargs is not None
        assert thinker.generate_kwargs["pad_token_id"] == 7777
        assert thinker.generate_kwargs["eos_token_id"] == [7777, 8888]

    def test_qwen3_asr_generate_skips_empty_eos_list_for_pad_fallback(self) -> None:
        thinker = _RecordingThinker()
        model = SimpleNamespace(
            thinker=thinker,
            generation_config=SimpleNamespace(pad_token_id=None, eos_token_id=[]),
            config=SimpleNamespace(pad_token_id=None, eos_token_id=None),
        )

        Qwen3ASRForConditionalGeneration.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
        )

        assert thinker.generate_kwargs is not None
        assert thinker.generate_kwargs["pad_token_id"] == 151645
        assert thinker.generate_kwargs["eos_token_id"] == [151645, 151643]

    def test_qwen3_asr_generate_falls_back_to_first_eos_token_as_pad_token(self) -> None:
        thinker = _RecordingThinker()
        model = SimpleNamespace(
            thinker=thinker,
            generation_config=SimpleNamespace(pad_token_id=None),
            config=SimpleNamespace(pad_token_id=None),
        )

        Qwen3ASRForConditionalGeneration.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            eos_token_id=[13579, 24680],
        )

        assert thinker.generate_kwargs is not None
        assert thinker.generate_kwargs["pad_token_id"] == 13579
