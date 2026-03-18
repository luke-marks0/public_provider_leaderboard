# inference-trust

This is a fork of [adamkarvonen/token-difr](https://github.com/adamkarvonen/token-difr) that supports reference token generation (both locally and via Modal), as well as scheduled audits of all OpenRouter providers for some model, and verification via Modal.

## Setup

```bash
uv pip install -e .
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

Local vLLM:

```bash
python serve.py local start --model Qwen/Qwen3-8B
python serve.py local list
python serve.py local stop --all
```

Modal vLLM:

```bash
python serve.py modal start --model Qwen/Qwen3-8B --gpu H100
python serve.py modal list
python serve.py modal stop --all
```

Notes:
- local startup validates that the active environment already has `vllm` and CUDA-enabled PyTorch,
  injects `torch/lib` into `LD_LIBRARY_PATH`, and auto-caches the Harmony tokenizer vocab needed by
  local `openai/gpt-oss-*` runs.
- `serve.py modal start` deploys `modal_vllm_app.py` by default (`--no-deploy` to reuse existing deployment).
- During deploy, `serve.py` sets `TOKEN_DIFR_MODAL_GPU` from `--gpu` so the Modal runtime matches each model profile.
- Modal server defaults use `min_containers=0` so idle servers can scale to zero.
- Default `scaledown_window_seconds` is `60` (override per model config if needed).
- Tracked runtime state is stored in `state/servers.json`.

## Generate reference tokens

Generate from local backend:

```bash
python generate_reference_tokens.py Qwen/Qwen3-8B \
  --generation-backend local \
  --local-base-url http://127.0.0.1:8000
```

Generate from Modal backend:

```bash
python generate_reference_tokens.py Qwen/Qwen3-8B \
  --generation-backend modal \
  --modal-base-url "https://<modal-endpoint>"
```

Reference bundles are written to `reference_tokens/`.

## Audit provider outputs

Default Fireworks verification:

```bash
python audit.py Qwen/Qwen3-8B
```

Modal verification backend:
```bash
python audit.py Qwen/Qwen3-8B openai/gpt-oss-120b \
  --verification-backend modal
```

Results are written under `audit_results/`.
