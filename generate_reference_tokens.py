# ruff: noqa: E402

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

TOKEN_DIFR_ROOT = Path(__file__).resolve().parent
SRC_DIR = TOKEN_DIFR_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANONICAL_PROMPTS_FILE = TOKEN_DIFR_ROOT / "reference_tokens" / "canonical_prompts.json"

load_dotenv()

from token_difr import construct_prompts
from token_difr.model_registry import resolve_hf_name
from token_difr.reference_tokens import (
    build_reference_bundle,
    load_conversations_from_json,
    normalize_reference_filename,
    save_reference_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference token bundles for a registered model."
    )
    parser.add_argument("model", help="HuggingFace model name or registered alias.")
    parser.add_argument(
        "--generation-backend",
        choices=("local", "modal"),
        default="local",
        help="Where to run reference generation requests.",
    )
    parser.add_argument(
        "--local-base-url",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible base URL for local generation.",
    )
    parser.add_argument(
        "--modal-base-url",
        default=None,
        help="OpenAI-compatible base URL for Modal generation (or set MODAL_BASE_URL).",
    )
    parser.add_argument(
        "--generation-model",
        default=None,
        help="Override model/deployment identifier sent to the generation backend.",
    )
    parser.add_argument(
        "--provider-label",
        default=None,
        help="Provider label stored in the output bundle (defaults to generation backend).",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional JSON file containing conversations or {conversations:[...]} payload.",
    )
    parser.add_argument("--n-prompts", type=int, default=100, help="Number of prompts to generate.")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum generated tokens per prompt.")
    parser.add_argument("--seed", type=int, default=424242, help="Generation seed.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Generation top-k.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Generation top-p.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout per completion request.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt used when constructing prompts from WildChat.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Optional tokenizer revision to pin while generating metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TOKEN_DIFR_ROOT / "reference_tokens"),
        help="Directory to write reference bundles.",
    )
    return parser.parse_args()


def _resolve_generation_base_url(args: argparse.Namespace) -> str:
    if args.generation_backend == "local":
        return str(args.local_base_url).strip()
    modal_base = args.modal_base_url or os.environ.get("MODAL_BASE_URL")
    if not modal_base or not str(modal_base).strip():
        raise ValueError("Modal generation requires --modal-base-url or MODAL_BASE_URL.")
    return str(modal_base).strip()


def _discover_prompt_source(reference_dir: Path, target_file: Path) -> Path | None:
    if target_file.is_file():
        return target_file

    candidates = sorted(reference_dir.glob("*.json"))
    if candidates:
        return candidates[0]
    return None


def _load_conversations(args: argparse.Namespace, model_name: str, output_dir: Path) -> list[list[dict[str, str]]]:
    prompt_source: Path | None = None
    if args.prompts_file:
        prompt_source = Path(args.prompts_file).expanduser().resolve()
    elif CANONICAL_PROMPTS_FILE.is_file():
        prompt_source = CANONICAL_PROMPTS_FILE
    else:
        prompt_source = _discover_prompt_source(
            output_dir,
            output_dir / normalize_reference_filename(model_name),
        )

    if prompt_source is not None:
        conversations = load_conversations_from_json(prompt_source)
    else:
        conversations = construct_prompts(
            n_prompts=args.n_prompts,
            model_name=model_name,
            system_prompt=args.system_prompt,
        )

    if len(conversations) < args.n_prompts:
        raise ValueError(
            f"Need at least {args.n_prompts} prompts but only found {len(conversations)}."
        )
    return conversations[: args.n_prompts]


def main() -> None:
    args = parse_args()
    model_name = resolve_hf_name(args.model)
    generation_base_url = _resolve_generation_base_url(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    conversations = _load_conversations(args, model_name, output_dir)

    provider_label = args.provider_label or args.generation_backend
    bundle = build_reference_bundle(
        model_name=model_name,
        generation_base_url=generation_base_url,
        conversations=conversations,
        max_tokens=args.max_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout_seconds=args.timeout_seconds,
        provider_label=provider_label,
        generation_model=args.generation_model,
        tokenizer_revision=args.tokenizer_revision,
        verbose=True,
    )
    output_path = save_reference_bundle(bundle, output_dir=output_dir)
    print(f"Saved reference tokens for {model_name}: {output_path}")


if __name__ == "__main__":
    main()
