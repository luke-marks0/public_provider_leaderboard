# ruff: noqa: E402

import argparse
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
DEFAULT_MODAL_MIN_CONTAINERS = 1
DEFAULT_MODAL_SCALEDOWN_WINDOW_SECONDS = 600


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sanitize_name(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "server"


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


def _build_local_command(args: argparse.Namespace) -> list[str]:
    model_name = resolve_hf_name(args.model)
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
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len and args.max_model_len > 0:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.served_model_name:
        command.extend(["--served-model-name", args.served_model_name])
    if args.trust_remote_code:
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
    command = _build_local_command(args)

    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
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
        subprocess.run(
            ["modal", "deploy", str(modal_app_file), "--name", app_name],
            check=True,
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
        "trust_remote_code": bool(args.trust_remote_code),
        "app_name": app_name,
        "class_name": class_name,
        "gpu": args.gpu,
        "min_containers": min_containers,
        "scaledown_window_seconds": scaledown_window_seconds,
        "app_file": str(modal_app_file),
        "started_at_utc": _utc_now_iso(),
    }
    params = _build_modal_query_params(entry)
    instance = cls_obj(**params)
    instance.update_autoscaler(
        min_containers=min_containers,
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

        cls_obj = modal.Cls.from_name(str(entry["app_name"]), str(entry["class_name"]))
        cls_obj = cls_obj.with_options(
            gpu=str(entry["gpu"]),
            scaledown_window=int(entry["scaledown_window_seconds"]),
        )
        params = _build_modal_query_params(entry)
        instance = cls_obj(**params)
        instance.update_autoscaler(min_containers=0)
        modal_servers.pop(name, None)
        print(f"Scaled down modal server {name} to zero containers.")

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
    local_start.add_argument("--served-model-name", default="", help="Optional served model alias.")
    local_start.add_argument("--tensor-parallel-size", type=int, default=1)
    local_start.add_argument("--dtype", default="auto")
    local_start.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    local_start.add_argument("--max-model-len", type=int, default=0)
    local_start.add_argument("--trust-remote-code", action="store_true", default=True)
    local_start.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
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
    modal_start.add_argument("--scaledown-window-seconds", type=int, default=DEFAULT_MODAL_SCALEDOWN_WINDOW_SECONDS)
    modal_start.add_argument("--tensor-parallel-size", type=int, default=1)
    modal_start.add_argument("--dtype", default="auto")
    modal_start.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    modal_start.add_argument("--max-model-len", type=int, default=0)
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
