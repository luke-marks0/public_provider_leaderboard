"""Tests for splitting Modal routing query params from OpenAI client base URLs."""

from __future__ import annotations

import audit
import token_difr.audit as package_audit


def test_split_openai_base_url_and_query_script_module() -> None:
    base_url = "https://example.modal.run/v1?model_name=Qwen/Qwen3-8B&trust_remote_code=false"
    clean_base_url, query = audit._split_openai_base_url_and_query(base_url)

    assert clean_base_url == "https://example.modal.run/v1"
    assert query == {
        "model_name": "Qwen/Qwen3-8B",
        "trust_remote_code": "false",
    }


def test_split_openai_base_url_and_query_package_module() -> None:
    base_url = "https://example.modal.run/v1?model_name=openai/gpt-oss-120b&max_model_len=8192"
    clean_base_url, query = package_audit._split_openai_base_url_and_query(base_url)

    assert clean_base_url == "https://example.modal.run/v1"
    assert query == {
        "model_name": "openai/gpt-oss-120b",
        "max_model_len": "8192",
    }


def test_create_async_openai_client_passes_default_query_script(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_async_openai(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(audit, "AsyncOpenAI", fake_async_openai)
    _ = audit._create_async_openai_client(
        api_key="token",
        base_url="https://example.modal.run/v1?trust_remote_code=false",
    )

    assert captured["api_key"] == "token"
    assert captured["base_url"] == "https://example.modal.run/v1"
    assert captured["default_query"] == {"trust_remote_code": "false"}


def test_create_async_openai_client_passes_default_query_package(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_async_openai(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(package_audit, "AsyncOpenAI", fake_async_openai)
    _ = package_audit._create_async_openai_client(
        api_key="token",
        base_url="https://example.modal.run/v1?model_name=Qwen/Qwen3-8B",
    )

    assert captured["api_key"] == "token"
    assert captured["base_url"] == "https://example.modal.run/v1"
    assert captured["default_query"] == {"model_name": "Qwen/Qwen3-8B"}
