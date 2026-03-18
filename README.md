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
uv pip install -e token-difr
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

Reference bundles are written to `token-difr/reference_tokens/`.

## Audit provider outputs

Run from `token-difr/`.

Default Fireworks verification:

```bash
uv run audit.py Qwen/Qwen3-8B
```

Modal verification backend:
```bash
uv run audit.py Qwen/Qwen3-8B openai/gpt-oss-120b \
  --verification-backend modal
```

Results are written under `token-difr/audit_results/` when run from `token-difr/`.
