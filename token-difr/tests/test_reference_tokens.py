"""Unit tests for reference-token helpers."""

import json

from token_difr.reference_tokens import (
    load_conversations_from_json,
    normalize_reference_filename,
)


def test_normalize_reference_filename_uses_canonical_model_name() -> None:
    assert normalize_reference_filename("qwen/qwen3-8b") == "Qwen_Qwen3-8B.json"


def test_load_conversations_from_json_accepts_wrapped_payload(tmp_path) -> None:
    payload = {
        "conversations": [
            [{"role": "user", "content": "hello"}],
            [{"role": "system", "content": "hi"}, {"role": "user", "content": "there"}],
        ]
    }
    source = tmp_path / "prompts.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    conversations = load_conversations_from_json(source)
    assert len(conversations) == 2
    assert conversations[0][0]["role"] == "user"
