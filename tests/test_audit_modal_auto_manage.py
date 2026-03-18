"""Unit tests for auto-managed Modal verification behavior in audit.py."""

from __future__ import annotations

from pathlib import Path

import audit
from token_difr.audit import AuditResult


def test_load_modal_profile_uses_model_config_file() -> None:
    profile = audit._load_modal_profile("openai/gpt-oss-120b")
    assert profile["model"] == "openai/gpt-oss-120b"
    assert profile["modal_gpu"] == "H100"
    assert profile["modal_max_containers"] == 1
    assert profile["tensor_parallel_size"] == 1
    assert profile["max_model_len"] == 2048
    assert profile["max_num_seqs"] == 16
    config_path = profile.get("config_path")
    assert isinstance(config_path, str)
    assert config_path.endswith("gpt-oss-120b.json")


def test_load_modal_profile_preserves_smaller_context_limits() -> None:
    profile = audit._load_modal_profile("moonshotai/Kimi-K2.5")
    assert profile["max_model_len"] == 2048


def test_build_modal_start_command_forces_zero_min_containers() -> None:
    profile = {
        "model": "openai/gpt-oss-120b",
        "served_model_name": "",
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "trust_remote_code": False,
        "modal_gpu": "H100",
        "modal_max_containers": 1,
        "modal_scaledown_window_seconds": 60,
    }

    command = audit._build_modal_start_command(
        server_name="audit-gpt-oss-120b",
        app_name="token-difr-vllm",
        class_name="VllmServer",
        profile=profile,
        deploy=False,
    )

    idx = command.index("--min-containers")
    assert command[idx + 1] == "0"
    max_idx = command.index("--max-containers")
    assert command[max_idx + 1] == "1"
    assert "--no-deploy" in command
    assert "--no-trust-remote-code" in command


def test_main_modal_auto_managed_scales_each_model_to_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, "N_PROMPTS", 1)
    monkeypatch.setattr(audit, "MAX_TOKENS", 1)
    monkeypatch.setattr(audit, "list_openrouter_providers", lambda model: ["provider-a"])
    monkeypatch.setattr(audit, "construct_prompts", lambda **kwargs: [[{"role": "user", "content": "hi"}]])

    started: list[str] = []
    stopped: list[str] = []

    def fake_start(*, hf_model: str, app_name: str, class_name: str, deploy: bool):  # type: ignore[no-untyped-def]
        name = f"server-{hf_model.split('/')[-1]}"
        started.append(name)
        return (
            name,
            "https://example.modal.run/v1",
            {"modal_gpu": "H100", "tensor_parallel_size": 1},
        )

    def fake_stop(name: str) -> None:
        stopped.append(name)

    monkeypatch.setattr(audit, "_start_modal_verification_server_for_model", fake_start)
    monkeypatch.setattr(audit, "_stop_modal_verification_server_by_name", fake_stop)

    fake_result = AuditResult(
        exact_match_rate=1.0,
        avg_prob=1.0,
        avg_margin=0.0,
        avg_logit_rank=0.0,
        avg_gumbel_rank=0.0,
        infinite_margin_rate=0.0,
        total_tokens=1,
        n_sequences=1,
    )
    monkeypatch.setattr(audit, "audit_provider", lambda *args, **kwargs: fake_result)

    audit._main_modal(
        models=["openai/gpt-oss-120b", "Qwen/Qwen3-8B"],
        use_reference_tokens=False,
        modal_verification_base_url=None,
        modal_verification_model=None,
        modal_stop_after_verification=True,
        modal_app_name="token-difr-vllm",
        modal_class_name="VllmServer",
        modal_deploy_before_start=False,
    )

    assert len(started) == 2
    assert stopped == started
