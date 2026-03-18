"""Unit tests for modal teardown behavior in audit.py."""

import json
import subprocess
from pathlib import Path

import audit


def test_find_modal_server_name_by_base_url_matches_canonical_url(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "servers.json"
    state_payload = {
        "local_servers": {},
        "modal_servers": {
            "qwen3-8b": {
                "base_url": "https://example.modal.run/v1?b=2&a=1",
            }
        },
    }
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    monkeypatch.setattr(audit, "STATE_FILE", state_path)

    name = audit._find_modal_server_name_by_base_url("https://example.modal.run/v1?a=1&b=2")
    assert name == "qwen3-8b"


def test_find_modal_server_name_by_base_url_ambiguous_path_returns_none(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "servers.json"
    state_payload = {
        "local_servers": {},
        "modal_servers": {
            "server-a": {"base_url": "https://example.modal.run/v1?model_name=A"},
            "server-b": {"base_url": "https://example.modal.run/v1?model_name=B"},
        },
    }
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    monkeypatch.setattr(audit, "STATE_FILE", state_path)

    name = audit._find_modal_server_name_by_base_url("https://example.modal.run/v1")
    assert name is None


def test_stop_modal_verification_server_invokes_serve_stop(monkeypatch) -> None:
    monkeypatch.setattr(audit, "_find_modal_server_name_by_base_url", lambda base_url: "modal-qwen")

    captured: dict[str, object] = {}

    def fake_run(command, text, capture_output):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["text"] = text
        captured["capture_output"] = capture_output
        return subprocess.CompletedProcess(command, 0, stdout="stopped\n", stderr="")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    audit._stop_modal_verification_server("https://example.modal.run")

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == audit.sys.executable
    assert command[1].endswith("serve.py")
    assert command[2:] == ["modal", "stop", "--name", "modal-qwen"]
    assert captured["text"] is True
    assert captured["capture_output"] is True


def test_stop_modal_verification_server_skips_when_untracked(monkeypatch) -> None:
    monkeypatch.setattr(audit, "_find_modal_server_name_by_base_url", lambda base_url: None)

    called = {"value": False}

    def fake_run(command, text, capture_output):  # type: ignore[no-untyped-def]
        called["value"] = True
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    audit._stop_modal_verification_server("https://example.modal.run")
    assert called["value"] is False
