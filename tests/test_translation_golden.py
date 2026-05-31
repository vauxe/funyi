# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import tools.gen_translation_golden as golden_module


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> torch.Tensor:
        del messages, tokenize, add_generation_prompt, return_tensors
        return torch.tensor([[10, 11]], dtype=torch.long)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        # Only the generated suffix should reach decode; the prompt tokens [10, 11]
        # must have been trimmed by _translate_stock. A regression that decodes the
        # full sequence is surfaced instead of silently passing.
        if list(token_ids) != [21, 22]:
            return f"UNTRIMMED:{list(token_ids)}"
        return "你好"


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(_commit_hash="commit456")

    def to(self, device: torch.device) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        return self

    def generate(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        suffix = torch.tensor([[21, 22]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def test_main_writes_pinned_stock_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "cases.jsonl"
    output = tmp_path / "golden.json"
    dataset.write_text(
        json.dumps({"id": "case-1", "text": "hello", "target_language": "Chinese"}) + "\n",
        encoding="utf-8",
    )
    calls: dict[str, tuple[str, dict[str, object]]] = {}

    def fake_tokenizer_from_pretrained(model_path: str, **kwargs: object) -> FakeTokenizer:
        calls["tokenizer"] = (model_path, kwargs)
        return FakeTokenizer()

    def fake_model_from_pretrained(model_path: str, **kwargs: object) -> FakeModel:
        calls["model"] = (model_path, kwargs)
        return FakeModel()

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer_from_pretrained))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fake_model_from_pretrained),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gen_translation_golden.py",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--model",
            "org/model",
            "--model-revision",
            "abc123",
            "--allow-download",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ],
    )

    golden_module.main()
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert calls["tokenizer"][0] == "org/model"
    assert calls["tokenizer"][1]["revision"] == "abc123"
    assert calls["model"][1]["revision"] == "abc123"
    assert payload["model_revision"] == "abc123"
    assert payload["resolved_model_commit"] == "commit456"
    assert payload["generation"] == {
        "do_sample": False,
        "repetition_penalty": 1.05,
        "extra_generate_kwargs": {},
    }
    assert payload["cases"][0]["output"] == "你好"
