"""Unit tests for Fireworks verification target overrides in audit_provider."""

from token_difr.audit import (
    AuditResult,
    TokenSequence,
    _resolve_fireworks_verification_model,
    audit_provider,
)


def test_resolve_fireworks_verification_model_uses_registry_by_default() -> None:
    assert _resolve_fireworks_verification_model("Qwen/Qwen3-8B") == "accounts/fireworks/models/qwen3-8b"


def test_resolve_fireworks_verification_model_prefers_override() -> None:
    override = "accounts/test-account/deployments/test-deployment"
    assert _resolve_fireworks_verification_model("nonexistent/model", override) == override


def test_audit_provider_forwards_fireworks_verification_model(monkeypatch) -> None:
    capture: dict[str, object] = {}

    async def fake_audit_provider_async(**kwargs) -> AuditResult:
        capture.update(kwargs)
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

    monkeypatch.setattr("token_difr.audit._audit_provider_async", fake_audit_provider_async)

    override = "accounts/test-account/deployments/test-deployment"
    result = audit_provider(
        conversations=[],
        model="Qwen/Qwen3-8B",
        fireworks_verification_model=override,
    )

    assert capture["fireworks_verification_model"] == override
    assert result.exact_match_rate == 1.0


def test_audit_provider_forwards_modal_backend_overrides(monkeypatch) -> None:
    capture: dict[str, object] = {}

    async def fake_audit_provider_async(**kwargs) -> AuditResult:
        capture.update(kwargs)
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

    monkeypatch.setattr("token_difr.audit._audit_provider_async", fake_audit_provider_async)

    result = audit_provider(
        conversations=[],
        model="Qwen/Qwen3-8B",
        verification_backend="modal",
        verification_model="Qwen/Qwen3-8B",
        verification_base_url="https://example.modal.run/v1",
    )

    assert capture["verification_backend"] == "modal"
    assert capture["verification_model"] == "Qwen/Qwen3-8B"
    assert capture["verification_base_url"] == "https://example.modal.run/v1"
    assert result.exact_match_rate == 1.0


def test_audit_provider_runs_collection_before_verification(monkeypatch) -> None:
    call_order: list[str] = []

    async def fake_collect(**kwargs):  # type: ignore[no-untyped-def]
        call_order.append("collect")
        assert kwargs["provider"] == "test-provider"
        return ([TokenSequence(prompt_token_ids=[1], output_token_ids=[2])], 42)

    async def fake_verify(**kwargs):  # type: ignore[no-untyped-def]
        call_order.append("verify")
        assert kwargs["vocab_size"] == 42
        assert len(kwargs["sequences"]) == 1
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

    monkeypatch.setattr("token_difr.audit._collect_provider_sequences_async", fake_collect)
    monkeypatch.setattr("token_difr.audit._verify_provider_sequences_async", fake_verify)

    result = audit_provider(
        conversations=[],
        model="Qwen/Qwen3-8B",
        provider="test-provider",
    )

    assert call_order == ["collect", "verify"]
    assert result.total_tokens == 1
