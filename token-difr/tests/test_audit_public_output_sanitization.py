"""Unit tests for public output sanitization in audit.py."""

import json
from pathlib import Path

import audit


def test_save_results_omits_sensitive_fields(tmp_path: Path) -> None:
    results = {
        "model": "Qwen/Qwen3-8B",
        "parameters": {
            "n_prompts": 100,
            "fireworks_on_demand_deployment": "accounts/test/deployments/private-deploy",
            "fireworks_serverless_model": "accounts/fireworks/models/qwen3-8b",
            "fireworks_base_model_for_deployment": "accounts/fireworks/models/qwen3-8b",
            "fireworks_deployment_created_for_audit": False,
        },
        "providers": {
            "provider-a": {
                "exact_match_rate": 1.0,
                "fireworks_verification_target": "accounts/test/deployments/private-deploy",
            }
        },
    }
    output_file = tmp_path / "audit_results.json"

    audit.save_results(results, str(output_file))
    payload = json.loads(output_file.read_text(encoding="utf-8"))

    assert "fireworks_on_demand_deployment" not in payload["parameters"]
    assert "fireworks_serverless_model" not in payload["parameters"]
    assert "fireworks_base_model_for_deployment" not in payload["parameters"]
    assert "fireworks_verification_target" not in payload["providers"]["provider-a"]

    # The in-memory result object should remain unchanged.
    assert "fireworks_on_demand_deployment" in results["parameters"]
    assert "fireworks_verification_target" in results["providers"]["provider-a"]


def test_save_results_redacts_identifier_substrings_in_error_fields(tmp_path: Path) -> None:
    results = {
        "providers": {
            "provider-a": {
                "error": (
                    "failed org_abcd123 cmpl-xyz987 "
                    "accounts/my-account/deployments/private-deploy and retried"
                ),
                "serverless_error": "org_foo and cmpl-bar and accounts/a/deployments/b",
                "note": "org_should_not_change",
            }
        }
    }
    output_file = tmp_path / "audit_results.json"

    audit.save_results(results, str(output_file))
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    provider_payload = payload["providers"]["provider-a"]

    assert provider_payload["error"] == (
        "failed redacted redacted accounts/redacted/deployments/redacted and retried"
    )
    assert provider_payload["serverless_error"] == (
        "redacted and redacted and accounts/redacted/deployments/redacted"
    )
    assert provider_payload["note"] == "org_should_not_change"
