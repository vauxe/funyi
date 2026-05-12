import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from qwen3_asr_runtime.backends.transformers import TransformersASRBackend


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


class TransformersBackendAttentionTest(unittest.TestCase):
    def test_flashinfer_routes_thinker_configs_through_flashinfer_dispatcher(self) -> None:
        model = _fake_model()

        with (
            patch("qwen3_asr_runtime.backends.transformers.register_flashinfer", return_value=True),
            patch.object(TransformersASRBackend, "_default_attn_implementation", return_value="flash_attention_2"),
            patch("qwen3_asr_runtime.backends.transformers.AutoModel.from_pretrained", return_value=model) as load_model,
            patch("qwen3_asr_runtime.backends.transformers.AutoProcessor.from_pretrained", return_value=object()),
        ):
            TransformersASRBackend.from_pretrained("dummy-model", flashinfer=True)

        self.assertEqual(load_model.call_args.kwargs["attn_implementation"], "flashinfer")
        self.assertEqual(model.thinker.config._attn_implementation, "flashinfer")
        self.assertEqual(model.thinker.model.config._attn_implementation, "flashinfer")
        self.assertEqual(model.thinker.audio_tower.config._attn_implementation, "flashinfer")

    def test_flashinfer_overrides_explicit_attention_backend(self) -> None:
        model = _fake_model()

        with (
            patch("qwen3_asr_runtime.backends.transformers.register_flashinfer", return_value=True),
            patch("qwen3_asr_runtime.backends.transformers.AutoModel.from_pretrained", return_value=model) as load_model,
            patch("qwen3_asr_runtime.backends.transformers.AutoProcessor.from_pretrained", return_value=object()),
        ):
            TransformersASRBackend.from_pretrained(
                "dummy-model",
                flashinfer=True,
                attn_implementation="flash_attention_2",
            )

        self.assertEqual(load_model.call_args.kwargs["attn_implementation"], "flashinfer")
        self.assertEqual(model.thinker.model.config._attn_implementation, "flashinfer")
        self.assertEqual(model.thinker.audio_tower.config._attn_implementation, "flashinfer")


if __name__ == "__main__":
    unittest.main()
