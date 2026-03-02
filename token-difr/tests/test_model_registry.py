"""Unit tests for model registry resolution helpers."""

from token_difr.model_registry import get_fireworks_name, resolve_hf_name


def test_resolve_hf_name_case_insensitive() -> None:
    assert resolve_hf_name("qwen/qwen3-8b") == "Qwen/Qwen3-8B"


def test_resolve_hf_name_openrouter_alias() -> None:
    assert resolve_hf_name("qwen/qwen3-235b-a22b-2507") == "Qwen/Qwen3-235B-A22B-Instruct-2507"


def test_resolve_hf_name_openrouter_alias_deepseek() -> None:
    assert resolve_hf_name("deepseek/deepseek-v3.2") == "deepseek-ai/DeepSeek-V3.2"


def test_get_fireworks_name_from_alias() -> None:
    assert get_fireworks_name("qwen/qwen3-8b") == "accounts/fireworks/models/qwen3-8b"


def test_get_fireworks_name_from_alias_minimax() -> None:
    assert get_fireworks_name("minimax/minimax-m2.5") == "accounts/fireworks/models/minimax-m2p5"
