"""High-level audit interface for verifying LLM provider outputs."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import openai
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from token_difr.api import verify_outputs_openai_compatible
from token_difr.common import compute_metrics_summary
from token_difr.model_registry import get_fireworks_name, get_openrouter_name, resolve_hf_name
from token_difr.openrouter_api import generate_openrouter_responses, tokenize_openrouter_responses


class FireworksVerificationError(RuntimeError):
    """Raised when Fireworks-side verification cannot be completed."""


@dataclass
class AuditResult:
    """Result of auditing a provider's outputs."""

    exact_match_rate: float
    avg_prob: float
    avg_margin: float
    avg_logit_rank: float
    avg_gumbel_rank: float
    infinite_margin_rate: float
    total_tokens: int
    n_sequences: int

    def __repr__(self) -> str:
        return (
            f"AuditResult({self.exact_match_rate:.1%} match rate, "
            f"{self.total_tokens} tokens across {self.n_sequences} sequences)"
        )


def _resolve_fireworks_verification_model(hf_model: str, fireworks_verification_model: str | None = None) -> str:
    """Resolve Fireworks verification target.

    When an override is provided, use it directly (supports deployment paths like
    ``accounts/<account>/deployments/<deployment-id>``). Otherwise, look up the
    default model mapping from FIREWORKS_MODEL_REGISTRY.
    """
    if fireworks_verification_model:
        return fireworks_verification_model
    return get_fireworks_name(hf_model)


def _normalize_openai_base_url(raw_base_url: str, *, ensure_v1_path: bool) -> str:
    """Normalize a verification base URL while preserving query parameters."""
    parsed = urlparse(raw_base_url.strip())
    path = parsed.path.rstrip("/")
    if ensure_v1_path and not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    if not path:
        path = "/"
    return urlunparse(parsed._replace(path=path))


async def _audit_provider_async(
    conversations: list[list[dict[str, str]]],
    model: str,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
    seed: int = 42,
    top_k: int = 50,
    top_p: float = 0.95,
    concurrency: int = 20,
    fireworks_verification_model: str | None = None,
    verification_backend: str = "fireworks",
    verification_model: str | None = None,
    verification_base_url: str | None = None,
    verification_api_key: str | None = None,
) -> AuditResult:
    """Async implementation of audit_provider."""
    hf_model = resolve_hf_name(model)

    # Get API keys
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Get OpenRouter model name (uses registry or falls back to lowercase)
    openrouter_model = get_openrouter_name(hf_model)

    # Create clients
    openrouter_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    backend = verification_backend.strip().lower()
    if backend == "fireworks":
        api_key = verification_api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable not set")

        if verification_model is not None:
            resolved_verification_model = verification_model
        else:
            # Backward-compatible alias support for existing callers.
            explicit_fireworks_model = fireworks_verification_model
            try:
                resolved_verification_model = _resolve_fireworks_verification_model(
                    hf_model, explicit_fireworks_model
                )
            except Exception as exc:
                if explicit_fireworks_model is None:
                    raise FireworksVerificationError(str(exc)) from exc
                raise

        raw_base_url = verification_base_url or "https://api.fireworks.ai/inference/v1"
        verifier_base_url = _normalize_openai_base_url(raw_base_url, ensure_v1_path=False)
        request_extra_body: dict[str, object] | None = None
    elif backend == "modal":
        api_key = verification_api_key or os.environ.get("MODAL_VERIFICATION_API_KEY") or "modal-verification"
        raw_base_url = verification_base_url or os.environ.get("MODAL_VERIFICATION_BASE_URL")
        if not raw_base_url:
            raise ValueError(
                "Modal verification requires --verification-base-url or MODAL_VERIFICATION_BASE_URL."
            )
        verifier_base_url = _normalize_openai_base_url(raw_base_url, ensure_v1_path=True)
        resolved_verification_model = verification_model or hf_model
        request_extra_body = {"return_tokens_as_token_ids": True}
    else:
        raise ValueError(
            f"Unsupported verification backend {verification_backend!r}. Expected 'fireworks' or 'modal'."
        )

    verification_client = AsyncOpenAI(
        api_key=api_key,
        base_url=verifier_base_url,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Generate responses via OpenRouter
    responses = await generate_openrouter_responses(
        client=openrouter_client,
        conversations=conversations,
        model=openrouter_model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        concurrency=concurrency,
    )

    # Tokenize responses
    sequences = tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens)

    # Verify via selected backend
    try:
        results = await verify_outputs_openai_compatible(
            sequences,
            vocab_size=vocab_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            client=verification_client,
            model=resolved_verification_model,
            topk_logprobs=5,
            backend_label=f"{backend} API",
            request_extra_body=request_extra_body,
        )
    except Exception as exc:
        if backend == "fireworks":
            raise FireworksVerificationError(str(exc)) from exc
        raise

    # Compute summary
    summary = compute_metrics_summary(results)

    return AuditResult(
        exact_match_rate=summary["exact_match_rate"],
        avg_prob=summary["avg_prob"],
        avg_margin=summary["avg_margin"],
        avg_logit_rank=summary["avg_logit_rank"],
        avg_gumbel_rank=summary["avg_gumbel_rank"],
        infinite_margin_rate=summary["infinite_margin_rate"],
        total_tokens=summary["total_tokens"],
        n_sequences=len(sequences),
    )


def audit_provider(
    conversations: list[list[dict[str, str]]],
    model: str,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
    seed: int = 42,
    top_k: int = 50,
    top_p: float = 0.95,
    concurrency: int = 20,
    fireworks_verification_model: str | None = None,
    verification_backend: str = "fireworks",
    verification_model: str | None = None,
    verification_base_url: str | None = None,
    verification_api_key: str | None = None,
) -> AuditResult:
    """Audit a provider by generating responses and verifying against a reference backend.

    This function:
    1. Generates responses via OpenRouter using the specified provider
    2. Tokenizes the responses using the model's tokenizer
    3. Verifies the token sequences against Fireworks logprobs
    4. Returns an AuditResult with verification metrics

    Args:
        conversations: List of conversations, where each conversation is a list of
            message dicts with 'role' and 'content' keys. Use construct_prompts()
            to get a default dataset.
        model: HuggingFace model name (e.g., "meta-llama/Llama-3.3-70B-Instruct").
            Must be in FIREWORKS_MODEL_REGISTRY unless fireworks_verification_model
            is provided.
        provider: OpenRouter provider to use (e.g., "groq", "moonshotai").
            If None, OpenRouter will choose automatically.
        temperature: Sampling temperature. Use 0.0 for deterministic outputs.
        max_tokens: Maximum tokens to generate per response.
        seed: Random seed for reproducibility.
        top_k: Top-k sampling parameter for verification.
        top_p: Top-p (nucleus) sampling parameter for verification.
        concurrency: Number of concurrent API requests.
        fireworks_verification_model: Optional Fireworks verification target.
            Can be a serverless model path
            (e.g., "accounts/fireworks/models/qwen3-8b") or an on-demand deployment
            path (e.g., "accounts/<account>/deployments/<deployment-id>").
        verification_backend: Verification backend ('fireworks' or 'modal').
        verification_model: Optional backend-specific override model/deployment.
        verification_base_url: Optional backend base URL override.
        verification_api_key: Optional backend API key override.

    Returns:
        AuditResult with verification metrics.

    Raises:
        ValueError: If API keys are not set or model is not in registry when no
            fireworks_verification_model is provided.
        FireworksVerificationError: If Fireworks verification cannot be completed.

    Example:
        >>> from token_difr import construct_prompts, audit_provider
        >>> prompts = construct_prompts(n_prompts=50, model_name="meta-llama/Llama-3.3-70B-Instruct")
        >>> result = audit_provider(prompts, "meta-llama/Llama-3.3-70B-Instruct", provider="groq")
        >>> print(result)
        AuditResult(98.3% match rate, 4521 tokens across 50 sequences)
    """
    return asyncio.run(
        _audit_provider_async(
            conversations=conversations,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_k=top_k,
            top_p=top_p,
            concurrency=concurrency,
            fireworks_verification_model=fireworks_verification_model,
            verification_backend=verification_backend,
            verification_model=verification_model,
            verification_base_url=verification_base_url,
            verification_api_key=verification_api_key,
        )
    )
