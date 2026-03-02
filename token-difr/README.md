# token-difr (backend)

`token-difr` is the backend package used by `inference-provider-leaderboard` for:
- local/Modal vLLM lifecycle management,
- reference token bundle generation,
- provider verification/auditing workflows.

Primary documentation now lives at the repository root:
- `../README.md`

Quick start from repo root:

```bash
cd token-difr
uv sync --extra vllm
uv run serve.py local list
```
