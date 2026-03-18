"""OpenAI-compatible token generation helpers for reference bundle creation."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

_TOKEN_ID_RE = re.compile(r"^token_id:(-?\d+)$")


@dataclass(frozen=True)
class CompletionTokens:
    """Tokenized prompt/output result from a single completion request."""

    prompt_token_ids: list[int]
    output_token_ids: list[int]
    completion_tokens: int


def _resolve_service_url(base_url: str, suffix: str) -> str:
    parsed = urlparse(base_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL: {base_url!r}")
    path = parsed.path.rstrip("/")
    resolved_path = f"{path}{suffix}" if path else suffix
    return urlunparse(parsed._replace(path=resolved_path))


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Completion response must be a JSON object.")
    return decoded


def _parse_token_id(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    match = _TOKEN_ID_RE.match(value.strip())
    if not match:
        return None
    return int(match.group(1))


def _extract_response_token_ids(response: dict[str, Any]) -> list[int]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices returned in completion response.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Invalid completion choice payload.")

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError("Completion response did not include logprobs payload.")

    content = logprobs.get("content")
    if isinstance(content, list):
        token_ids = []
        for row in content:
            if not isinstance(row, dict):
                continue
            parsed = _parse_token_id(row.get("token_id"))
            if parsed is not None:
                token_ids.append(parsed)
        if token_ids:
            return token_ids

    tokens = logprobs.get("tokens")
    if isinstance(tokens, list):
        maybe_ids: list[int] = []
        for token in tokens:
            parsed = _parse_token_id(token)
            if parsed is None:
                maybe_ids = []
                break
            maybe_ids.append(parsed)
        if maybe_ids:
            return maybe_ids

    raise ValueError(
        "Unable to parse token ids from response. "
        "Expected logprobs.content[*].token_id or logprobs.tokens token_id:<int> strings."
    )


def _split_prompt_and_completion(
    token_ids: list[int],
    usage: dict[str, Any] | None,
    prompt_token_ids_fallback: list[int],
) -> CompletionTokens:
    prompt_tokens = None
    completion_tokens = None
    if isinstance(usage, dict):
        raw_prompt = usage.get("prompt_tokens")
        raw_completion = usage.get("completion_tokens")
        if isinstance(raw_prompt, int):
            prompt_tokens = raw_prompt
        if isinstance(raw_completion, int):
            completion_tokens = raw_completion

    if prompt_tokens is None:
        prompt_tokens = len(prompt_token_ids_fallback)

    if completion_tokens is None:
        completion_tokens = len(token_ids) - prompt_tokens
    if completion_tokens < 0:
        completion_tokens = len(token_ids)

    split_index = prompt_tokens
    if split_index > len(token_ids):
        if completion_tokens == len(token_ids):
            output_ids = [int(token) for token in token_ids[:completion_tokens]]
            return CompletionTokens(
                prompt_token_ids=[int(token) for token in prompt_token_ids_fallback],
                output_token_ids=output_ids,
                completion_tokens=len(output_ids),
            )
        raise ValueError(
            "Completion response usage.prompt_tokens exceeds parsed token payload "
            f"(prompt_tokens={prompt_tokens}, parsed_tokens={len(token_ids)})."
        )

    prompt_ids = [int(token) for token in token_ids[:split_index]]
    output_ids = [int(token) for token in token_ids[split_index:]]
    if completion_tokens < len(output_ids):
        output_ids = output_ids[:completion_tokens]
    elif completion_tokens > len(output_ids):
        completion_tokens = len(output_ids)

    return CompletionTokens(
        prompt_token_ids=prompt_ids,
        output_token_ids=output_ids,
        completion_tokens=completion_tokens,
    )


def generate_completion_tokens(
    *,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    timeout_seconds: int,
    extra_body: dict[str, Any] | None = None,
) -> CompletionTokens:
    """Generate output token IDs from an OpenAI-compatible completions API."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": [int(token) for token in prompt_token_ids],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "seed": int(seed),
        "echo": False,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
    }
    if extra_body:
        payload.update(extra_body)

    response = _post_json(
        _resolve_service_url(base_url, "/v1/completions"),
        payload,
        timeout_seconds,
    )
    token_ids = _extract_response_token_ids(response)
    return _split_prompt_and_completion(
        token_ids=token_ids,
        usage=response.get("usage") if isinstance(response.get("usage"), dict) else None,
        prompt_token_ids_fallback=prompt_token_ids,
    )
