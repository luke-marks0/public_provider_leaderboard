# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

TOKEN_DIFR_ROOT = Path(__file__).resolve().parent
SRC_DIR = TOKEN_DIFR_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv()

from token_difr.model_registry import resolve_hf_name

STATE_DIR = TOKEN_DIFR_ROOT / "state"
STATE_FILE = STATE_DIR / "servers.json"
LOG_DIR = STATE_DIR / "logs"
DEFAULT_LOCAL_HOST = "0.0.0.0"
DEFAULT_LOCAL_PORT = 8000
DEFAULT_MODAL_APP_NAME = "token-difr-vllm"
DEFAULT_MODAL_CLASS_NAME = "VllmServer"
DEFAULT_MODAL_GPU = "H100"
DEFAULT_MODAL_MIN_CONTAINERS = 0
DEFAULT_MODAL_MAX_CONTAINERS = 1
DEFAULT_MODAL_SCALEDOWN_WINDOW_SECONDS = 60
O200K_BASE_TIKTOKEN_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
O200K_BASE_TIKTOKEN_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
CONFIG_DIR = TOKEN_DIFR_ROOT / "configs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sanitize_name(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "server"


def _to_int(value: Any, default: int, *, min_value: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _to_float(value: Any, default: float, *, min_value: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _config_file_candidates(hf_model: str) -> list[Path]:
    candidates: list[Path] = []
    dedupe: set[str] = set()

    model_tail = hf_model.split("/", 1)[-1]
    for raw_name in (model_tail, hf_model):
        slug = _sanitize_name(raw_name)
        if slug in dedupe:
            continue
        dedupe.add(slug)
        candidates.append(CONFIG_DIR / f"{slug}.json")

    return candidates


def _load_local_profile(hf_model: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "served_model_name": "",
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 0,
        "enforce_eager": False,
        "trust_remote_code": True,
    }

    for candidate in _config_file_candidates(hf_model):
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("served_model_name"), str):
            profile["served_model_name"] = payload["served_model_name"]
        profile["tensor_parallel_size"] = _to_int(payload.get("tensor_parallel_size"), profile["tensor_parallel_size"], min_value=1)
        if isinstance(payload.get("dtype"), str) and payload["dtype"].strip():
            profile["dtype"] = payload["dtype"].strip()
        profile["gpu_memory_utilization"] = _to_float(
            payload.get("gpu_memory_utilization"),
            profile["gpu_memory_utilization"],
            min_value=0.0,
        )
        profile["max_model_len"] = _to_int(payload.get("max_model_len"), profile["max_model_len"], min_value=0)
        if isinstance(payload.get("enforce_eager"), bool):
            profile["enforce_eager"] = payload["enforce_eager"]
        if isinstance(payload.get("trust_remote_code"), bool):
            profile["trust_remote_code"] = payload["trust_remote_code"]
        break

    return profile


def _resolve_local_start_settings(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    profile = _load_local_profile(model_name)
    return {
        "served_model_name": args.served_model_name if args.served_model_name is not None else profile["served_model_name"],
        "tensor_parallel_size": int(args.tensor_parallel_size) if args.tensor_parallel_size is not None else int(profile["tensor_parallel_size"]),
        "dtype": str(args.dtype) if args.dtype is not None else str(profile["dtype"]),
        "gpu_memory_utilization": float(args.gpu_memory_utilization) if args.gpu_memory_utilization is not None else float(profile["gpu_memory_utilization"]),
        "max_model_len": int(args.max_model_len) if args.max_model_len is not None else int(profile["max_model_len"]),
        "enforce_eager": bool(args.enforce_eager) if args.enforce_eager is not None else bool(profile["enforce_eager"]),
        "trust_remote_code": bool(args.trust_remote_code) if args.trust_remote_code is not None else bool(profile["trust_remote_code"]),
    }


def _read_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {"local_servers": {}, "modal_servers": {}}
    payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"local_servers": {}, "modal_servers": {}}
    local_servers = payload.get("local_servers")
    modal_servers = payload.get("modal_servers")
    if not isinstance(local_servers, dict):
        local_servers = {}
    if not isinstance(modal_servers, dict):
        modal_servers = {}
    return {"local_servers": local_servers, "modal_servers": modal_servers}


def _write_state(payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_openai_server(base_url: str, timeout_seconds: int, poll_seconds: float = 1.0) -> bool:
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


def _append_query(url: str, params: dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    encoded = urllib.parse.urlencode(query)
    return urllib.parse.urlunparse(parsed._replace(query=encoded))


def _normalize_base_url(raw_url: str) -> str:
    parsed = urllib.parse.urlparse(raw_url.strip())
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urllib.parse.urlunparse(parsed._replace(path=path))


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    existing = env.get(key, "").strip()
    if not existing:
        env[key] = value
        return
    parts = [part for part in existing.split(":") if part]
    if value not in parts:
        env[key] = ":".join([value, *parts])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_o200k_base_vocab() -> Path:
    encodings_dir = STATE_DIR / "encodings"
    target_path = encodings_dir / "o200k_base.tiktoken"
    if target_path.is_file():
        if _sha256_file(target_path) == O200K_BASE_TIKTOKEN_SHA256:
            return encodings_dir
        target_path.unlink()

    encodings_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        O200K_BASE_TIKTOKEN_URL,
        headers={"User-Agent": "token-difr-serve/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != O200K_BASE_TIKTOKEN_SHA256:
        raise RuntimeError(
            "Downloaded o200k_base.tiktoken digest mismatch: "
            f"expected {O200K_BASE_TIKTOKEN_SHA256}, got {digest}."
        )

    target_path.write_bytes(payload)
    return encodings_dir


def _build_local_runtime_env(model_name: str) -> dict[str, str]:
    env = os.environ.copy()

    try:
        import vllm  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Local vLLM startup requires `vllm` in the active environment. "
            "Reinstall `token-difr` so the default dependencies are present, for example:\n"
            "uv pip install -e ."
        ) from exc

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Local vLLM startup requires PyTorch to be installed in the active environment."
        ) from exc

    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(
            "Local vLLM startup requires a CUDA-enabled PyTorch build. "
            "Reinstall `token-difr` with uv so it resolves the CUDA PyTorch wheels, for example:\n"
            "uv pip install -e ."
        )

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    libtorch_cuda = torch_lib_dir / "libtorch_cuda.so"
    if not libtorch_cuda.is_file():
        raise RuntimeError(
            f"Expected CUDA runtime library not found: {libtorch_cuda}. "
            "Reinstall a CUDA-enabled PyTorch build before starting local vLLM."
        )

    _prepend_env_path(env, "LD_LIBRARY_PATH", str(torch_lib_dir))

    if resolve_hf_name(model_name).startswith("openai/gpt-oss") and not env.get("TIKTOKEN_ENCODINGS_BASE"):
        encodings_dir = _ensure_o200k_base_vocab()
        env["TIKTOKEN_ENCODINGS_BASE"] = f"{encodings_dir}/"

    return env


def _build_local_command(args: argparse.Namespace, settings: dict[str, Any] | None = None) -> list[str]:
    model_name = resolve_hf_name(args.model)
    if settings is None:
        settings = {
            "served_model_name": getattr(args, "served_model_name", ""),
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
            "dtype": getattr(args, "dtype", "auto"),
            "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", 0.9),
            "max_model_len": getattr(args, "max_model_len", 0),
            "enforce_eager": getattr(args, "enforce_eager", False),
            "trust_remote_code": getattr(args, "trust_remote_code", True),
        }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(settings["tensor_parallel_size"]),
        "--dtype",
        str(settings["dtype"]),
        "--gpu-memory-utilization",
        str(settings["gpu_memory_utilization"]),
    ]
    if int(settings["max_model_len"]) > 0:
        command.extend(["--max-model-len", str(settings["max_model_len"])])
    if str(settings["served_model_name"]).strip():
        command.extend(["--served-model-name", str(settings["served_model_name"])])
    if bool(settings["enforce_eager"]):
        command.append("--enforce-eager")
    if bool(settings["trust_remote_code"]):
        command.append("--trust-remote-code")
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return command


def _local_start(args: argparse.Namespace) -> None:
    state = _read_state()
    local_servers = dict(state["local_servers"])
    model_name = resolve_hf_name(args.model)
    name = args.name or f"{_sanitize_name(model_name)}-{args.port}"

    existing = local_servers.get(name)
    if isinstance(existing, dict):
        pid = existing.get("pid")
        if isinstance(pid, int) and _pid_is_running(pid):
            raise RuntimeError(f"Local server {name!r} is already running (pid={pid}).")
        local_servers.pop(name, None)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"local-{name}.log"
    settings = _resolve_local_start_settings(args, model_name)
    command = _build_local_command(args, settings)
    runtime_env = _build_local_runtime_env(model_name)

    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=runtime_env,
            start_new_session=True,
        )

    probe_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    base_url = f"http://{probe_host}:{args.port}"
    if not _wait_for_openai_server(base_url, timeout_seconds=args.start_timeout):
        if _pid_is_running(process.pid):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise RuntimeError(
            f"Local vLLM server did not become ready on {base_url} within {args.start_timeout}s. "
            f"See log: {log_path}"
        )

    local_servers[name] = {
        "name": name,
        "model": model_name,
        "host": args.host,
        "port": int(args.port),
        "pid": int(process.pid),
        "base_url": _normalize_base_url(base_url),
        "log_file": str(log_path),
        "started_at_utc": _utc_now_iso(),
        "command": command,
        **settings,
    }
    _write_state({"local_servers": local_servers, "modal_servers": state["modal_servers"]})
    print(f"Started local vLLM server {name}: pid={process.pid} base_url={_normalize_base_url(base_url)}")


def _stop_process(pid: int, timeout_seconds: int = 20) -> bool:
    if not _pid_is_running(pid):
        return True

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.5)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
    return not _pid_is_running(pid)


def _local_stop(args: argparse.Namespace) -> None:
    state = _read_state()
    local_servers = dict(state["local_servers"])

    if args.all:
        target_names = list(local_servers.keys())
    elif args.name:
        target_names = [args.name]
    else:
        raise ValueError("Specify --name or --all for local stop.")

    for name in target_names:
        entry = local_servers.get(name)
        if not isinstance(entry, dict):
            print(f"Local server {name!r} not found in state.")
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int):
            print(f"Local server {name!r} has invalid pid in state; removing entry.")
            local_servers.pop(name, None)
            continue
        stopped = _stop_process(pid)
        if stopped:
            print(f"Stopped local server {name} (pid={pid}).")
        else:
            print(f"Failed to stop local server {name} (pid={pid}).")
        local_servers.pop(name, None)

    _write_state({"local_servers": local_servers, "modal_servers": state["modal_servers"]})


def _local_list() -> None:
    state = _read_state()
    local_servers = state["local_servers"]
    if not local_servers:
        print("No local servers recorded.")
        return
    for name in sorted(local_servers.keys()):
        entry = local_servers[name]
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        status = "running" if isinstance(pid, int) and _pid_is_running(pid) else "stopped"
        print(
            f"{name}: status={status} model={entry.get('model')} "
            f"pid={pid} base_url={entry.get('base_url')}"
        )


def _import_modal() -> Any:
    try:
        import modal  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Modal SDK is required for modal server management. "
            "Install with `pip install modal` and run `modal token new`."
        ) from exc
    return modal


def _build_modal_query_params(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": str(entry["model"]),
        "served_model_name": str(entry.get("served_model_name", "")),
        "tensor_parallel_size": int(entry["tensor_parallel_size"]),
        "dtype": str(entry["dtype"]),
        "gpu_memory_utilization": str(entry["gpu_memory_utilization"]),
        "max_model_len": int(entry["max_model_len"]),
        "max_num_seqs": int(entry.get("max_num_seqs", 0)),
        "enforce_eager": bool(entry.get("enforce_eager", False)),
        "trust_remote_code": bool(entry["trust_remote_code"]),
    }


def _stringify_query_params(params: dict[str, Any]) -> dict[str, str]:
    query: dict[str, str] = {}
    for key, value in params.items():
        if isinstance(value, bool):
            query[key] = "true" if value else "false"
        else:
            query[key] = str(value)
    return query


def _modal_start(args: argparse.Namespace) -> None:
    state = _read_state()
    modal_servers = dict(state["modal_servers"])

    model_name = resolve_hf_name(args.model)
    name = args.name or _sanitize_name(model_name)
    app_name = args.app_name or DEFAULT_MODAL_APP_NAME
    class_name = args.class_name or DEFAULT_MODAL_CLASS_NAME
    min_containers = int(args.min_containers)
    scaledown_window_seconds = int(args.scaledown_window_seconds)
    modal_app_file = (TOKEN_DIFR_ROOT / "modal_vllm_app.py").resolve()

    if args.deploy:
        deploy_env = os.environ.copy()
        deploy_env["TOKEN_DIFR_MODAL_GPU"] = str(args.gpu)
        subprocess.run(
            ["modal", "deploy", str(modal_app_file), "--name", app_name],
            check=True,
            env=deploy_env,
        )

    modal = _import_modal()
    cls_obj = modal.Cls.from_name(app_name, class_name)
    cls_obj = cls_obj.with_options(
        gpu=args.gpu,
        scaledown_window=scaledown_window_seconds,
    )

    entry = {
        "name": name,
        "model": model_name,
        "served_model_name": args.served_model_name or "",
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "dtype": args.dtype,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": int(args.max_num_seqs),
        "enforce_eager": bool(args.enforce_eager),
        "trust_remote_code": bool(args.trust_remote_code),
        "app_name": app_name,
        "class_name": class_name,
        "gpu": args.gpu,
        "min_containers": min_containers,
        "max_containers": int(args.max_containers),
        "scaledown_window_seconds": scaledown_window_seconds,
        "app_file": str(modal_app_file),
        "started_at_utc": _utc_now_iso(),
    }
    params = _build_modal_query_params(entry)
    instance = cls_obj(**params)
    instance.update_autoscaler(
        min_containers=min_containers,
        max_containers=int(args.max_containers),
        scaledown_window=scaledown_window_seconds,
    )
    web_url = instance.serve.get_web_url()
    if not isinstance(web_url, str) or not web_url.strip():
        raise RuntimeError("Failed to resolve Modal web endpoint URL.")
    base_url = _normalize_base_url(_append_query(web_url.strip(), _stringify_query_params(params)))
    entry["base_url"] = base_url

    modal_servers[name] = entry
    _write_state({"local_servers": state["local_servers"], "modal_servers": modal_servers})
    print(f"Started modal server {name}: base_url={base_url}")


def _modal_stop(args: argparse.Namespace) -> None:
    state = _read_state()
    modal_servers = dict(state["modal_servers"])

    if args.all:
        target_names = list(modal_servers.keys())
    elif args.name:
        target_names = [args.name]
    else:
        raise ValueError("Specify --name or --all for modal stop.")

    modal = _import_modal()
    for name in target_names:
        entry = modal_servers.get(name)
        if not isinstance(entry, dict):
            print(f"Modal server {name!r} not found in state.")
            continue

        try:
            cls_obj = modal.Cls.from_name(str(entry["app_name"]), str(entry["class_name"]))
            cls_obj = cls_obj.with_options(
                gpu=str(entry["gpu"]),
                scaledown_window=int(entry["scaledown_window_seconds"]),
            )
            params = _build_modal_query_params(entry)
            instance = cls_obj(**params)
            instance.update_autoscaler(
                min_containers=0,
                max_containers=int(entry.get("max_containers", DEFAULT_MODAL_MAX_CONTAINERS)),
                scaledown_window=0,
            )
            print(f"Scaled down modal server {name} to zero containers.")
        except Exception as exc:
            print(
                f"Modal server {name!r} could not be updated (likely already gone): {exc}. "
                "Removing stale state entry."
            )
        finally:
            modal_servers.pop(name, None)

    _write_state({"local_servers": state["local_servers"], "modal_servers": modal_servers})


def _modal_list() -> None:
    state = _read_state()
    modal_servers = state["modal_servers"]
    if not modal_servers:
        print("No modal servers recorded.")
        return
    for name in sorted(modal_servers.keys()):
        entry = modal_servers[name]
        if not isinstance(entry, dict):
            continue
        print(
            f"{name}: model={entry.get('model')} gpu={entry.get('gpu')} "
            f"min_containers={entry.get('min_containers')} base_url={entry.get('base_url')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight local/Modal vLLM server management for token-difr."
    )
    subparsers = parser.add_subparsers(dest="backend", required=True)

    local_parser = subparsers.add_parser("local", help="Manage local vLLM servers.")
    local_subparsers = local_parser.add_subparsers(dest="action", required=True)

    local_start = local_subparsers.add_parser("start", help="Start a local vLLM server.")
    local_start.add_argument("--name", default="", help="Server name used in local state.")
    local_start.add_argument("--model", required=True, help="HuggingFace model to serve.")
    local_start.add_argument("--host", default=DEFAULT_LOCAL_HOST, help="Host to bind.")
    local_start.add_argument("--port", type=int, default=DEFAULT_LOCAL_PORT, help="Port to bind.")
    local_start.add_argument("--served-model-name", default=None, help="Optional served model alias.")
    local_start.add_argument("--tensor-parallel-size", type=int, default=None)
    local_start.add_argument("--dtype", default=None)
    local_start.add_argument("--gpu-memory-utilization", type=float, default=None)
    local_start.add_argument("--max-model-len", type=int, default=None)
    local_start.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
    local_start.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    local_start.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    local_start.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    local_start.set_defaults(enforce_eager=None, trust_remote_code=None)
    local_start.add_argument("--extra-args", default="", help="Extra args appended to the vLLM command.")
    local_start.add_argument("--start-timeout", type=int, default=180)

    local_stop = local_subparsers.add_parser("stop", help="Stop local vLLM server(s).")
    local_stop.add_argument("--name", default="", help="Server name to stop.")
    local_stop.add_argument("--all", action="store_true", help="Stop all tracked local servers.")

    local_subparsers.add_parser("list", help="List tracked local servers.")

    modal_parser = subparsers.add_parser("modal", help="Manage remote Modal vLLM servers.")
    modal_subparsers = modal_parser.add_subparsers(dest="action", required=True)

    modal_start = modal_subparsers.add_parser("start", help="Start/scale a Modal vLLM server.")
    modal_start.add_argument("--name", default="", help="Server name used in local state.")
    modal_start.add_argument("--model", required=True, help="HuggingFace model to serve.")
    modal_start.add_argument("--served-model-name", default="", help="Optional served model alias.")
    modal_start.add_argument("--app-name", default=DEFAULT_MODAL_APP_NAME, help="Modal app name.")
    modal_start.add_argument("--class-name", default=DEFAULT_MODAL_CLASS_NAME, help="Modal class name.")
    modal_start.add_argument("--gpu", default=DEFAULT_MODAL_GPU, help="Modal GPU request (e.g. H100 or H100:4).")
    modal_start.add_argument("--min-containers", type=int, default=DEFAULT_MODAL_MIN_CONTAINERS)
    modal_start.add_argument("--max-containers", type=int, default=DEFAULT_MODAL_MAX_CONTAINERS)
    modal_start.add_argument("--scaledown-window-seconds", type=int, default=DEFAULT_MODAL_SCALEDOWN_WINDOW_SECONDS)
    modal_start.add_argument("--tensor-parallel-size", type=int, default=1)
    modal_start.add_argument("--dtype", default="auto")
    modal_start.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    modal_start.add_argument("--max-model-len", type=int, default=0)
    modal_start.add_argument("--max-num-seqs", type=int, default=0)
    modal_start.add_argument("--enforce-eager", action="store_true", default=False)
    modal_start.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    modal_start.add_argument("--trust-remote-code", action="store_true", default=True)
    modal_start.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    modal_start.add_argument(
        "--deploy",
        action="store_true",
        default=True,
        help="Run `modal deploy` before starting (default true).",
    )
    modal_start.add_argument("--no-deploy", dest="deploy", action="store_false")

    modal_stop = modal_subparsers.add_parser("stop", help="Scale down Modal vLLM server(s) to zero.")
    modal_stop.add_argument("--name", default="", help="Server name to stop.")
    modal_stop.add_argument("--all", action="store_true", help="Stop all tracked modal servers.")

    modal_subparsers.add_parser("list", help="List tracked modal servers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "local":
        if args.action == "start":
            _local_start(args)
            return
        if args.action == "stop":
            _local_stop(args)
            return
        if args.action == "list":
            _local_list()
            return

    if args.backend == "modal":
        if args.action == "start":
            _modal_start(args)
            return
        if args.action == "stop":
            _modal_stop(args)
            return
        if args.action == "list":
            _modal_list()
            return

    raise ValueError(f"Unsupported command: backend={args.backend} action={getattr(args, 'action', None)}")


if __name__ == "__main__":
    main()
