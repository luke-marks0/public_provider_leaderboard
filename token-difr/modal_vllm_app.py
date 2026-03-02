import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = os.environ.get("TOKEN_DIFR_MODAL_APP_NAME", "token-difr-vllm")
DEFAULT_VLLM_IMAGE = os.environ.get(
    "TOKEN_DIFR_MODAL_VLLM_IMAGE",
    "docker.io/vllm/vllm-openai@sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0",
)
HF_VOLUME_NAME = os.environ.get("TOKEN_DIFR_MODAL_HF_VOLUME", "token-difr-hf-cache")
HF_SECRET_NAME = os.environ.get("TOKEN_DIFR_MODAL_HF_SECRET", "huggingface-secret")
SERVE_PORT = 8000
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("TOKEN_DIFR_MODAL_STARTUP_TIMEOUT_SECONDS", "1200"))


def _wait_for_openai_server(base_url: str, timeout_seconds: int, poll_seconds: float = 2.0) -> bool:
    deadline = time.time() + timeout_seconds
    endpoint = base_url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(poll_seconds)
    return False


image = modal.Image.from_registry(DEFAULT_VLLM_IMAGE, add_python="3.11").entrypoint([])
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/data/hf": modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    min_containers=0,
    scaledown_window=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=256)
class VllmServer:
    model_name: str = modal.parameter()
    served_model_name: str = modal.parameter(default="")
    tensor_parallel_size: int = modal.parameter(default=1)
    dtype: str = modal.parameter(default="auto")
    gpu_memory_utilization: str = modal.parameter(default="0.9")
    max_model_len: int = modal.parameter(default=0)
    trust_remote_code: bool = modal.parameter(default=True)

    @modal.enter()
    def start(self) -> None:
        command = [
            "vllm",
            "serve",
            str(self.model_name),
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVE_PORT),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--dtype",
            str(self.dtype),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.max_model_len > 0:
            command.extend(["--max-model-len", str(self.max_model_len)])
        if self.served_model_name:
            command.extend(["--served-model-name", str(self.served_model_name)])
        if self.trust_remote_code:
            command.append("--trust-remote-code")

        env = os.environ.copy()
        env.setdefault("HF_HOME", "/data/hf")
        self._process = subprocess.Popen(command, env=env)

        ready = _wait_for_openai_server(
            base_url=f"http://127.0.0.1:{SERVE_PORT}",
            timeout_seconds=STARTUP_TIMEOUT_SECONDS,
        )
        if not ready:
            self._process.terminate()
            raise RuntimeError("vLLM server did not become ready before timeout.")

    @modal.web_server(port=SERVE_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
    def serve(self) -> None:
        return

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is not None:
            return
        process.terminate()
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.5)
        process.kill()
