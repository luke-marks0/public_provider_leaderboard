"""Unit tests for lightweight server management helpers."""

from types import SimpleNamespace

import serve


def test_build_local_command_includes_required_vllm_flags() -> None:
    args = SimpleNamespace(
        model="qwen/qwen3-8b",
        host="0.0.0.0",
        port=8000,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        served_model_name="qwen3-8b",
        trust_remote_code=True,
        extra_args="--max-num-seqs 32",
    )
    command = serve._build_local_command(args)

    assert command[:3] == [serve.sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in command
    assert "Qwen/Qwen3-8B" in command
    assert "--tensor-parallel-size" in command
    assert "--trust-remote-code" in command
    assert "--max-num-seqs" in command


def test_append_query_preserves_existing_params() -> None:
    url = "https://example.modal.run/path?existing=1"
    result = serve._append_query(url, {"model_name": "Qwen/Qwen3-8B"})
    assert "existing=1" in result
    assert "model_name=Qwen%2FQwen3-8B" in result


def test_build_modal_query_params_encodes_expected_fields() -> None:
    entry = {
        "model": "Qwen/Qwen3-8B",
        "served_model_name": "qwen3-8b",
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 0,
        "trust_remote_code": True,
    }
    params = serve._build_modal_query_params(entry)
    assert params["model_name"] == "Qwen/Qwen3-8B"
    assert params["trust_remote_code"] is True
