# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
runtime.py — Runtime environment + service orchestration.

Three modes:
  * docker   — render/launch docker compose services (GPU exposed via toolkit)
  * native   — create a venv and install pip requirements
  * systemd  — install a user/system service unit for server mode

Everything is idempotent and honours dry_run. No system files are modified
without an explicit confirmation callback (`confirm`).
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from typing import Callable, List, Optional

CONFIRM_YES: Callable[[str], bool] = lambda _msg: True


def _run(cmd: List[str], dry_run: bool, cwd: Optional[str] = None) -> int:
    printable = " ".join(shlex.quote(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return 0
    print(f"[run] {printable}")
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


# ---------------------------------------------------------------------------
# Native venv
# ---------------------------------------------------------------------------
def native_setup(venv_dir: str = ".venv", requirements: str = "requirements.txt",
                 *, dry_run: bool = False) -> int:
    """Create a virtualenv and install requirements. Idempotent."""
    if not os.path.exists(venv_dir):
        rc = _run([sys.executable, "-m", "venv", venv_dir], dry_run)
        if rc != 0 and not dry_run:
            return rc
    pip = os.path.join(venv_dir, "bin", "pip")
    if os.name == "nt":
        pip = os.path.join(venv_dir, "Scripts", "pip.exe")
    _run([pip, "install", "--upgrade", "pip"], dry_run)
    if os.path.exists(requirements) or dry_run:
        return _run([pip, "install", "-r", requirements], dry_run)
    return 0


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
def docker_available() -> bool:
    return shutil.which("docker") is not None


def nvidia_container_toolkit_ok() -> bool:
    """Best-effort probe for GPU passthrough support."""
    if not docker_available():
        return False
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def docker_up(service: Optional[str] = None,
              compose_file: str = "docker/docker-compose.yml",
              *, dry_run: bool = False) -> int:
    cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
    if service:
        cmd.append(service)
    return _run(cmd, dry_run)


def docker_down(compose_file: str = "docker/docker-compose.yml",
                *, dry_run: bool = False) -> int:
    return _run(["docker", "compose", "-f", compose_file, "down"], dry_run)


def docker_status(compose_file: str = "docker/docker-compose.yml") -> str:
    if not docker_available():
        # Expected inside the manager container: the default compose service
        # deliberately does not mount /var/run/docker.sock, because a panel
        # that can drive the host's Docker daemon is a root shell on the host.
        if os.path.exists("/.dockerenv"):
            return ("docker status unavailable in this container "
                    "(no Docker socket mounted — this is the safe default; "
                    "see the manager-privileged compose profile)")
        return "docker not installed"
    r = subprocess.run(["docker", "compose", "-f", compose_file, "ps"],
                       capture_output=True, text=True)
    return r.stdout or r.stderr


# ---------------------------------------------------------------------------
# systemd (server mode)
# ---------------------------------------------------------------------------
SYSTEMD_TEMPLATE = """[Unit]
Description=Croft — {name}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={exec_start}
Restart=on-failure
RestartSec=5
Environment=HF_HOME={workdir}/models/.hf-cache

[Install]
WantedBy=default.target
"""


def install_systemd_unit(name: str, exec_start: str, workdir: str, *,
                         user_scope: bool = True,
                         confirm: Callable[[str], bool] = CONFIRM_YES,
                         dry_run: bool = False) -> Optional[str]:
    """
    Write a systemd unit and enable it. `user_scope=True` installs to
    ~/.config/systemd/user (no root). System scope requires confirmation.
    """
    unit = SYSTEMD_TEMPLATE.format(name=name, workdir=workdir, exec_start=exec_start)
    if user_scope:
        unit_dir = os.path.expanduser("~/.config/systemd/user")
        ctl = ["systemctl", "--user"]
    else:
        if not confirm(f"Install SYSTEM service {name} to /etc/systemd/system "
                       "(requires sudo)?"):
            print("Aborted system service install.")
            return None
        unit_dir = "/etc/systemd/system"
        ctl = ["sudo", "systemctl"]
    unit_path = os.path.join(unit_dir, f"croft-{name}.service")
    if dry_run:
        print(f"[dry-run] write unit -> {unit_path}\n{unit}")
    else:
        os.makedirs(unit_dir, exist_ok=True)
        with open(unit_path, "w", encoding="utf-8") as fh:
            fh.write(unit)
    _run(ctl + ["daemon-reload"], dry_run)
    _run(ctl + ["enable", "--now", f"croft-{name}.service"], dry_run)
    return unit_path


# ---------------------------------------------------------------------------
# Server launch commands (documented, used by main.py "serve")
#
# Each entry is an argv *list*, not a string, so `serve` can hand it straight
# to subprocess without a shell — a model path is a filesystem path the user
# chose, and it has no business being re-parsed by sh.
#
# Bind addresses are loopback. These servers have no authentication of any
# kind: a `--host 0.0.0.0` default publishes an unauthenticated inference
# endpoint (and, for the web UIs, a file browser) to the whole network the
# moment someone runs the printed command.
# ---------------------------------------------------------------------------
SERVE_COMMANDS = {
    "llama.cpp": ["third_party/llama.cpp/build/bin/llama-server",
                  "-m", "{model}", "-c", "4096",
                  "--host", "127.0.0.1", "--port", "8080"],
    "vllm": [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
             "--model", "{model}", "--host", "127.0.0.1", "--port", "8000",
             "--gpu-memory-utilization", "0.90"],
    "tgi": ["text-generation-launcher", "--model-id", "{model}",
            "--hostname", "127.0.0.1", "--port", "8081"],
    "text-generation-webui": [sys.executable, "server.py",
                              "--model", "{model}", "--api"],
    "sd-webui": [sys.executable, "launch.py", "--api", "--xformers"],
}


def serve_argv(backend: str, model: str) -> List[str]:
    """Return the launch command as an argv list with `model` substituted."""
    tmpl = SERVE_COMMANDS.get(backend)
    if not tmpl:
        raise KeyError(f"Unknown backend '{backend}'. "
                       f"Choose from {list(SERVE_COMMANDS)}")
    return [part.replace("{model}", model) for part in tmpl]


def serve_command(backend: str, model: str) -> str:
    """The same command, shell-quoted for display and for systemd's ExecStart."""
    return " ".join(shlex.quote(part) for part in serve_argv(backend, model))


if __name__ == "__main__":  # pragma: no cover
    print("docker available:", docker_available())
    print(serve_command("llama.cpp", "models/mistral-7b-q4.gguf"))
