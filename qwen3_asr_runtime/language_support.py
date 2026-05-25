# coding=utf-8
from __future__ import annotations

# Qwen3-ASR model-card language list for Qwen3-ASR-1.7B and Qwen3-ASR-0.6B.
# The model card separately lists Chinese dialect coverage; those dialect names
# are not language prompt values.
QWEN3_ASR_MODEL_CARD_LANGUAGES: tuple[str, ...] = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Hungarian",
    "Macedonian",
    "Romanian",
)

QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES: tuple[str, ...] = (
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)

# HY-MT1.5-1.8B model-card language list. Realtime translation targets must
# come from this list.
HYMT_MODEL_CARD_LANGUAGES: tuple[str, ...] = (
    "Chinese",
    "English",
    "French",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Turkish",
    "Russian",
    "Arabic",
    "Korean",
    "Thai",
    "Italian",
    "German",
    "Vietnamese",
    "Malay",
    "Indonesian",
    "Filipino",
    "Hindi",
    "Traditional Chinese",
    "Polish",
    "Czech",
    "Dutch",
    "Khmer",
    "Burmese",
    "Persian",
    "Gujarati",
    "Urdu",
    "Telugu",
    "Marathi",
    "Hebrew",
    "Bengali",
    "Tamil",
    "Ukrainian",
    "Tibetan",
    "Kazakh",
    "Mongolian",
    "Uyghur",
    "Cantonese",
)


__all__ = [
    "HYMT_MODEL_CARD_LANGUAGES",
    "QWEN3_ASR_MODEL_CARD_LANGUAGES",
    "QWEN3_FORCED_ALIGNER_MODEL_CARD_LANGUAGES",
]
