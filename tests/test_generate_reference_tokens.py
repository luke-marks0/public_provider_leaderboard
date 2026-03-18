"""Unit tests for canonical prompt selection in generate_reference_tokens.py."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import generate_reference_tokens as grt


def test_load_conversations_prefers_canonical_prompt_file(monkeypatch, tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical_prompts.json"
    canonical_payload = {
        "conversations": [
            [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "one"}],
            [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "two"}],
        ]
    }
    canonical_path.write_text(json.dumps(canonical_payload), encoding="utf-8")

    discovered_path = tmp_path / "Qwen_Qwen3-8B.json"
    discovered_payload = {
        "conversations": [
            [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "other"}]
        ]
    }
    discovered_path.write_text(json.dumps(discovered_payload), encoding="utf-8")

    monkeypatch.setattr(grt, "CANONICAL_PROMPTS_FILE", canonical_path)
    args = Namespace(prompts_file=None, n_prompts=2, system_prompt="You are a helpful assistant.")

    conversations = grt._load_conversations(
        args,
        model_name="Qwen/Qwen3-8B",
        output_dir=tmp_path,
    )

    assert conversations == canonical_payload["conversations"]


def test_canonical_prompt_file_matches_existing_shared_reference_bundle() -> None:
    root = Path(__file__).resolve().parents[1] / "reference_tokens"
    canonical = json.loads((root / "canonical_prompts.json").read_text(encoding="utf-8"))["conversations"]
    shared = json.loads((root / "Qwen_Qwen3-8B.json").read_text(encoding="utf-8"))["conversations"]

    assert canonical == shared
