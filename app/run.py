from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT_DIR / "app" / "main.py"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def wait_for_backend(url: str, timeout_seconds: int = 15) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1)
            if response.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    railway_port = os.getenv("PORT")
    api_host = os.getenv("API_HOST", "127.0.0.1")
    api_port = os.getenv("API_PORT", "8000")
    streamlit_host = os.getenv(
        "STREAMLIT_HOST",
        "0.0.0.0" if railway_port else "127.0.0.1",
    )
    streamlit_port = os.getenv("STREAMLIT_PORT", railway_port or "8501")
    api_base_url = os.getenv("API_BASE_URL", f"http://127.0.0.1:{api_port}")
    api_reload = env_flag("API_RELOAD", default=railway_port is None)
    streamlit_headless = env_flag("STREAMLIT_HEADLESS", default=railway_port is not None)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT_DIR)
        if not existing_pythonpath
        else f"{ROOT_DIR}{os.pathsep}{existing_pythonpath}"
    )
    env["API_BASE_URL"] = api_base_url

    backend_process: subprocess.Popen[bytes] | None = None
    frontend_process: subprocess.Popen[bytes] | None = None

    def handle_signal(_: int, __: object) -> None:
        stop_process(frontend_process)
        stop_process(backend_process)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        backend_command = [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            api_host,
            "--port",
            api_port,
        ]
        if api_reload:
            backend_command.append("--reload")

        backend_process = subprocess.Popen(
            backend_command,
            cwd=ROOT_DIR,
            env=env,
        )

        if not wait_for_backend(f"{api_base_url.rstrip('/')}/health"):
            print("Backend did not become ready in time.", file=sys.stderr)
            return 1

        frontend_command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(STREAMLIT_APP),
            "--server.address",
            streamlit_host,
            "--server.port",
            streamlit_port,
            "--server.headless",
            str(streamlit_headless).lower(),
            "--browser.gatherUsageStats",
            "false",
        ]

        frontend_process = subprocess.Popen(
            frontend_command,
            cwd=ROOT_DIR,
            env=env,
        )

        if frontend_process.poll() is not None:
            return frontend_process.returncode or 1

        return frontend_process.wait()
    finally:
        stop_process(frontend_process)
        stop_process(backend_process)


if __name__ == "__main__":
    raise SystemExit(main())
