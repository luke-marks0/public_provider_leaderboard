# inference-provider-leaderboard

This repository uses `token-difr/` as the inference-audit backend.

`token-difr` is responsible for:
- running local and Modal vLLM reference servers,
- generating reference token bundles,
- auditing provider outputs against a verification backend,
- and checking whether two inference implementations behave equivalently under greedy decoding.

## Backend location

All implementation logic lives under:
- `token-difr/`

Primary backend entry scripts:
- `token-difr/audit.py`
- `token-difr/generate_reference_tokens.py`
- `token-difr/serve.py`
- `token-difr/modal_vllm_app.py`

## Setup

From repository root:

```bash
cd token-difr
uv sync --extra vllm
```

Set required environment variables:

```bash
export OPENROUTER_API_KEY="..."
export FIREWORKS_API_KEY="..."                     # needed for Fireworks verification
export MODAL_VERIFICATION_BASE_URL="https://..."   # optional fixed modal verification endpoint
export MODAL_VERIFICATION_API_KEY="..."            # optional
export MODAL_BASE_URL="https://..."                # optional for modal generation
export TOKEN_DIFR_MODAL_AUDIT_MAX_MODEL_LEN_CAP=8192  # optional cap for auto-managed modal audits
export TOKEN_DIFR_MODAL_VERIFICATION_CONCURRENCY=1     # optional modal verification request concurrency
```

## Start/stop reference servers

Run from `token-difr/`.

Local vLLM:

```bash
uv run serve.py local start --model Qwen/Qwen3-8B --port 8000
uv run serve.py local list
uv run serve.py local stop --all
```

Modal vLLM:

```bash
uv run serve.py modal start --model Qwen/Qwen3-8B --gpu H100
uv run serve.py modal list
uv run serve.py modal stop --all
```

Notes:
- `serve.py modal start` deploys `token-difr/modal_vllm_app.py` by default (`--no-deploy` to reuse existing deployment).
- During deploy, `serve.py` sets `TOKEN_DIFR_MODAL_GPU` from `--gpu` so the Modal runtime matches each model profile.
- Modal server defaults now use `min_containers=0` so idle servers can scale to zero.
- Default `scaledown_window_seconds` is `60` (override per model config if needed).
- Tracked runtime state is stored in `token-difr/state/servers.json`.

## Generate reference tokens

Run from `token-difr/`.

Generate from local backend:

```bash
uv run generate_reference_tokens.py Qwen/Qwen3-8B \
  --generation-backend local \
  --local-base-url http://127.0.0.1:8000
```

Generate from Modal backend:

```bash
uv run generate_reference_tokens.py Qwen/Qwen3-8B \
  --generation-backend modal \
  --modal-base-url "https://<modal-endpoint>"
```

Use real prompts:

```bash
uv run generate_reference_tokens.py Qwen/Qwen3-8B \
  --generation-backend local \
  --local-base-url http://127.0.0.1:8000 \
  --prompts-file /path/to/prompts.json
```

Reference bundles are written to `token-difr/reference_tokens/`.

## Audit provider outputs

Run from `token-difr/`.

Default Fireworks verification:

```bash
uv run audit.py Qwen/Qwen3-8B --reference-tokens
```

Modal verification backend:

```bash
uv run audit.py Qwen/Qwen3-8B \
  --reference-tokens \
  --verification-backend modal \
  --modal-verification-base-url "https://<modal-endpoint>"
```

Auto-managed per-model Modal verification (no fixed endpoint required):

```bash
uv run audit.py Qwen/Qwen3-8B openai/gpt-oss-120b \
  --reference-tokens \
  --verification-backend modal
```

Results are written under `token-difr/audit_results/` when run from `token-difr/`.

## Audit inference implementations (A vs B)

Use this when you want to validate implementation correctness directly.

1. Generate baseline reference tokens from implementation A.
2. Verify those sequences against implementation B with the same model and sampling params.
3. Repeat in reverse (B -> A).

Interpretation:
- Primary metric: `exact_match_rate`
- For greedy decoding (`temperature=0`), equivalent implementations should match at very high rates.
- Large drops usually indicate model/config/prompt/tokenization differences.

### Direct verification example

Run from `token-difr/`:

```python
import asyncio
import json
import os

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from token_difr.api import verify_outputs_openai_compatible
from token_difr.common import TokenSequence, compute_metrics_summary

MODEL = "Qwen/Qwen3-8B"
BUNDLE_PATH = "reference_tokens/Qwen_Qwen3-8B.json"
VERIFY_BASE_URL = os.environ["VERIFY_BASE_URL"]

with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
    bundle = json.load(f)

sequences = [TokenSequence.from_dict(s) for s in bundle["sequences"]]
params = bundle["parameters"]

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


async def main() -> None:
    client = AsyncOpenAI(
        api_key=os.environ.get("MODAL_VERIFICATION_API_KEY", "modal-verification"),
        base_url=VERIFY_BASE_URL,
    )

    metrics = await verify_outputs_openai_compatible(
        outputs=sequences,
        vocab_size=len(tokenizer),
        temperature=float(params["temperature"]),
        top_k=int(params["top_k"]),
        top_p=float(params["top_p"]),
        seed=int(params["seed"]),
        client=client,
        model=MODEL,
        topk_logprobs=5,
        backend_label="implementation-under-test",
        request_extra_body={"return_tokens_as_token_ids": True},
    )

    print(compute_metrics_summary(metrics))


asyncio.run(main())
```

## Migrated deterministic configs

Configs migrated from `deterministic_inference_server` are in:
- `token-difr/configs/`

They are intentionally simplified to current supported local vLLM + Modal fields:
- `model`
- `served_model_name`
- `tensor_parallel_size`
- `dtype`
- `gpu_memory_utilization`
- `max_model_len`
- `max_num_seqs`
- `enforce_eager`
- `trust_remote_code`
- `modal_gpu`
- `modal_min_containers`
- `modal_max_containers`
- `modal_scaledown_window_seconds`

## Current limitations

- Verification metrics are currently focused on greedy-token correctness checks.
- Provider audits depend on OpenRouter model/provider availability.
- OpenAI-compatible completion/logprob behavior must match expected token-id formats.
