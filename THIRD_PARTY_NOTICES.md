# Third-Party Notices

## Qwen3-ASR / Hugging Face Transformers Integration Code

The items listed below are copied from or derived from Qwen3-ASR / Hugging Face
Transformers integration code and assets. The source files retain their upstream
copyright headers; `modeling_qwen3_asr.py` has local Funyi runtime-integration
changes and carries a modification notice in its file header.

- `qwen3_asr_runtime/hf_qwen3_asr/__init__.py`
- `qwen3_asr_runtime/hf_qwen3_asr/configuration_qwen3_asr.py`
- `qwen3_asr_runtime/hf_qwen3_asr/modeling_qwen3_asr.py`
- `qwen3_asr_runtime/hf_qwen3_asr/processing_qwen3_asr.py`
- `qwen3_asr_runtime/utils.py`
- `qwen3_asr_runtime/assets/korean_dict_jieba.dict` (jieba-format Korean
  dictionary data, copied verbatim from the Qwen3-ASR `qwen_asr` inference
  assets; used by the force-aligner text processor)

These items are licensed under the Apache License, Version 2.0. See
`LICENSES/Apache-2.0.txt`.

Upstream source:

- QwenLM/Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- Model repository: https://huggingface.co/Qwen/Qwen3-ASR-1.7B

## FireRed Stream-VAD Postprocessor

`qwen3_asr_runtime/firered_stream_vad_postprocessor.py` is vendored verbatim
from FireRedVAD and retains its upstream copyright header (only a NOTICE block
is added). Copyright 2026 Xiaohongshu (Author: Kaituo Xu, Wenpeng Li, Kai Huang,
Kun Liu).

This file is licensed under the Apache License, Version 2.0. See
`LICENSES/Apache-2.0.txt`.

Upstream source:

- FireRedTeam/FireRedVAD: https://github.com/FireRedTeam/FireRedVAD
  (`fireredvad/core/stream_vad_postprocessor.py`,
  commit c30ec49e8cc69642b0ee65362eba11b9d11c6e54)
