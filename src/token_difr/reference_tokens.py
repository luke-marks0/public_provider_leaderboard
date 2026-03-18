"""Reference-token bundle generation utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from token_difr.common import render_conversation_for_tokenization
from token_difr.model_registry import resolve_hf_name
from token_difr.openai_generation import generate_completion_tokens

_TOKENIZER_FILE_CANDIDATES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_token_matrix_hash(prompt_token_ids_list: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(prompt_token_ids_list, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_tokenizer_metadata(model_name: str, revision: str | None) -> dict[str, Any]:
    resolved_revision = (revision or "").strip()
    files: list[dict[str, str]] = []
    try:
        from huggingface_hub import HfApi, hf_hub_download

        if not resolved_revision:
            info = HfApi().model_info(model_name)
            if isinstance(info.sha, str):
                resolved_revision = info.sha

        for file_name in _TOKENIZER_FILE_CANDIDATES:
            try:
                local_path = Path(
                    hf_hub_download(
                        repo_id=model_name,
                        filename=file_name,
                        revision=resolved_revision or None,
                    )
                )
            except Exception:
                continue
            files.append({"path": file_name, "digest": _sha256_file(local_path)})
    except Exception:
        files = []

    return {
        "digest": _sha256_json(files),
        "revision": resolved_revision,
        "files": files,
    }


def normalize_reference_filename(model_name: str) -> str:
    resolved = resolve_hf_name(model_name)
    return f"{resolved.replace('/', '_')}.json"


def load_conversations_from_json(path: Path) -> list[list[dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("conversations"), list):
        raw_conversations = payload["conversations"]
    elif isinstance(payload, list):
        raw_conversations = payload
    else:
        raise ValueError(f"Prompt file must be a list or object with conversations: {path}")

    conversations: list[list[dict[str, str]]] = []
    for conv in raw_conversations:
        if not isinstance(conv, list):
            raise ValueError(f"Invalid conversation entry in {path}")
        normalized: list[dict[str, str]] = []
        for message in conv:
            if not isinstance(message, dict):
                raise ValueError(f"Invalid message entry in {path}")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(f"Messages must include string role/content in {path}")
            normalized.append({"role": role, "content": content})
        conversations.append(normalized)
    return conversations


def build_reference_bundle(
    *,
    model_name: str,
    generation_base_url: str,
    conversations: list[list[dict[str, str]]],
    max_tokens: int,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout_seconds: int,
    provider_label: str,
    generation_model: str | None = None,
    tokenizer_revision: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    resolved_model = resolve_hf_name(model_name)

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if tokenizer_revision and tokenizer_revision.strip():
        tokenizer_kwargs["revision"] = tokenizer_revision.strip()
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, **tokenizer_kwargs)

    sequences: list[dict[str, list[int]]] = []
    decoded_output_text: list[str] = []

    iterator = conversations
    if verbose:
        iterator = tqdm(conversations, desc=f"Generating reference tokens ({provider_label})")

    target_model = generation_model or resolved_model
    for conversation in iterator:
        rendered = render_conversation_for_tokenization(
            tokenizer,
            conversation,
            add_generation_prompt=True,
            model_name=resolved_model,
        )
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        generated = generate_completion_tokens(
            base_url=generation_base_url,
            model=target_model,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        sequences.append(
            {
                "prompt_token_ids": [int(token) for token in generated.prompt_token_ids],
                "output_token_ids": [int(token) for token in generated.output_token_ids],
            }
        )
        decoded_output_text.append(
            tokenizer.decode(
                generated.output_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    tokenizer_meta = _load_tokenizer_metadata(resolved_model, tokenizer_revision)
    prompt_hash = _prompt_token_matrix_hash(
        [sequence["prompt_token_ids"] for sequence in sequences]
    )

    return {
        "model": resolved_model,
        "provider": provider_label,
        "parameters": {
            "n_prompts": len(sequences),
            "max_tokens": int(max_tokens),
            "seed": int(seed),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "prompt_token_ids_sha256": prompt_hash,
        },
        "tokenizer": tokenizer_meta,
        "conversations": conversations[: len(sequences)],
        "sequences": sequences,
        "decoded_output_text": decoded_output_text,
    }


def save_reference_bundle(bundle: dict[str, Any], *, output_dir: Path) -> Path:
    model_name = str(bundle.get("model", "unknown-model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / normalize_reference_filename(model_name)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return output_path
