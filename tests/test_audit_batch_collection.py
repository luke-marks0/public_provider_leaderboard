"""Unit tests for batch collection before verification in audit.py."""

from __future__ import annotations

from pathlib import Path

import audit
from token_difr.audit import AuditResult, TokenSequence


def test_main_collects_all_provider_tokens_before_verifying(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")
    monkeypatch.setattr(audit, "N_PROMPTS", 1)
    monkeypatch.setattr(audit, "MAX_TOKENS", 1)
    monkeypatch.setattr(audit, "list_openrouter_providers", lambda model: ["provider-a", "provider-b"])
    monkeypatch.setattr(audit, "construct_prompts", lambda **kwargs: [[{"role": "user", "content": "hi"}]])
    monkeypatch.setattr(audit, "get_fireworks_name", lambda model: "accounts/fireworks/models/test-model")

    events: list[str] = []

    def fake_collect(*args, **kwargs):  # type: ignore[no-untyped-def]
        provider = kwargs["provider"]
        events.append(f"collect:{provider}")
        return ([TokenSequence(prompt_token_ids=[1], output_token_ids=[2])], 10)

    def fake_verify(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append(f"verify:{kwargs['model']}:{kwargs.get('fireworks_verification_model', 'serverless')}")
        return AuditResult(
            exact_match_rate=1.0,
            avg_prob=1.0,
            avg_margin=0.0,
            avg_logit_rank=0.0,
            avg_gumbel_rank=0.0,
            infinite_margin_rate=0.0,
            total_tokens=1,
            n_sequences=1,
        )

    monkeypatch.setattr(audit, "collect_provider_sequences", fake_collect)
    monkeypatch.setattr(audit, "verify_provider_sequences", fake_verify)

    audit.main(
        models=["Qwen/Qwen3-8B"],
        use_reference_tokens=False,
        verification_backend="fireworks",
    )

    assert events == [
        "collect:provider-a",
        "collect:provider-b",
        "verify:Qwen/Qwen3-8B:serverless",
        "verify:Qwen/Qwen3-8B:serverless",
    ]
