#!/usr/bin/env python3
"""
main.py — Local LLM Chat installer & manager (CLI/TUI orchestrator).

Subcommands:
    detect       Print the hardware capability report (JSON).
    recommend    Rank models that will run on this machine.
    install      Interactive/unattended install of a selected model + backend.
    pull         Download a specific model id from the manifest (resumable).
    quantize     Convert/quantize a downloaded model (llama.cpp / AutoGPTQ).
    serve        Print/launch the serve command for a backend + model.
    status       Show installed models + running services.
    uninstall    Remove a model (and optionally services) with confirmation.

Design goals: idempotent, dry-run-able (--dry-run), consent before any
network download or system change, graceful degradation with clear messages.

`rich` and `typer` are used when available but are NOT required — the CLI
falls back to argparse + plain print so it runs on a bare Python 3.10+.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Allow `python installer/main.py ...` as well as `python -m installer.main`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import hardware, recommend as rec, downloader, quantize, runtime  # noqa: E402

try:
    from rich.console import Console       # type: ignore
    from rich.table import Table           # type: ignore
    _console = Console()
except Exception:  # pragma: no cover
    _console = None

MODELS_DIR = os.environ.get("LLM_MODELS_DIR", "models")
STATE_FILE = os.path.join(MODELS_DIR, "installed.json")


# ---------------------------------------------------------------------------
# Small output helpers (rich-optional)
# ---------------------------------------------------------------------------
def info(msg: str) -> None:
    if _console:
        _console.print(msg)
    else:
        print(msg)


def confirm(prompt: str, assume_yes: bool = False) -> bool:
    if assume_yes:
        info(f"{prompt} [auto-yes]")
        return True
    if not sys.stdin.isatty():
        info(f"{prompt} — non-interactive, defaulting to NO. Use --yes to accept.")
        return False
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {"models": {}}
    return {"models": {}}


def _save_state(state: Dict[str, Any]) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_recommendations(reco: Dict[str, Any], kind: str = "text") -> None:
    entries = reco["recommendations"].get(kind, [])
    info(f"\n[Tier: {reco['tier']}]  accelerator={reco['accelerator']}  "
         f"VRAM={reco['vram_gb']}GB  RAM={reco['ram_gb']}GB  disk={reco['disk_free_gb']}GB")
    if _console:
        table = Table(title=f"{kind.upper()} model recommendations")
        for col in ("id", "params", "fmt", "device", "need GB", "disk GB",
                    "fits", "license"):
            table.add_column(col)
        for e in entries:
            table.add_row(
                e["id"], f"{e['params_billion']:g}B", e["format"], e["device"],
                str(e["required_memory_gb"]), f"{e['approx_disk_gb']:g}",
                "✓" if e["fits"] else "✗",
                ("gated:" if e["gated"] else "") + e["license"])
        _console.print(table)
    else:
        print(f"\n{kind.upper()} recommendations:")
        for e in entries:
            fit = "OK " if e["fits"] else "no "
            print(f"  [{fit}] {e['id']:<22} {e['params_billion']:>5g}B "
                  f"{e['format']:<12} need {e['required_memory_gb']:>5}GB "
                  f"disk {e['approx_disk_gb']:>4g}GB  {e['license']}"
                  + ("  (gated)" if e["gated"] else ""))


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------
def cmd_detect(args) -> int:
    report = hardware.detect_all(disk_path=MODELS_DIR if os.path.isdir(MODELS_DIR) else ".")
    print(json.dumps(report, indent=2))
    return 0


def cmd_recommend(args) -> int:
    report = hardware.detect_all()
    reco = rec.recommend(report, kinds=args.kind or None)
    if args.json:
        print(json.dumps(reco, indent=2))
        return 0
    for kind in (args.kind or ["text", "image", "video"]):
        render_recommendations(reco, kind)
    if reco["default_text_model"]:
        info(f"\nDefault suggestion: install `{reco['default_text_model']}` "
             f"→  python -m installer.main install --model {reco['default_text_model']}")
    return 0


def _find_model(model_id: str):
    for m in rec.load_manifest():
        if m.id == model_id:
            return m
    return None


def cmd_pull(args) -> int:
    m = _find_model(args.model)
    if not m:
        info(f"Unknown model id: {args.model}. Run `recommend` to list ids.")
        return 2
    if m.gated:
        info(f"⚠ '{m.id}' is license-gated ({m.license}).")
        info(f"   Accept the license at https://huggingface.co/{m.hf_repo} "
             "and provide an HF token (--token or $HF_TOKEN).")
        if not confirm("Have you accepted the license and set a token?", args.yes):
            return 1
    dest_dir = os.path.join(MODELS_DIR, m.id)
    info(f"Downloading {m.hf_repo} → {dest_dir}  (~{m.approx_disk_gb:g} GB)")
    if not confirm(f"Proceed with download of ~{m.approx_disk_gb:g} GB?", args.yes):
        return 1
    try:
        downloader.hf_download(m.hf_repo, dest_dir, token=args.token,
                               dry_run=args.dry_run)
    except Exception as exc:
        info(f"✗ Download failed: {exc}")
        return 1
    state = _load_state()
    state["models"][m.id] = {"repo": m.hf_repo, "path": dest_dir,
                             "kind": m.kind, "format": m.format,
                             "license": m.license}
    if not args.dry_run:
        _save_state(state)
    info(f"✓ {m.id} ready in {dest_dir}")
    return 0


def cmd_quantize(args) -> int:
    src = args.src
    if args.method == "gguf":
        out_f16 = args.out or (src.rstrip("/") + "-f16.gguf")
        quantize.llamacpp_convert_to_gguf(src, out_f16, dry_run=args.dry_run)
        out_q = out_f16.replace("-f16.gguf", f"-{args.quant.lower()}.gguf")
        return quantize.llamacpp_quantize(out_f16, out_q, quant=args.quant,
                                          dry_run=args.dry_run)
    if args.method == "gptq":
        out = args.out or (src.rstrip("/") + "-gptq")
        return quantize.autogptq_quantize(src, out, bits=args.bits,
                                          dry_run=args.dry_run)
    if args.method == "bnb":
        print(quantize.bitsandbytes_recipe(src))
        return 0
    info(f"Unknown method {args.method}")
    return 2


def cmd_serve(args) -> int:
    cmd = runtime.serve_command(args.backend, args.model)
    info(f"Serve command:\n  {cmd}")
    if args.run:
        if not confirm("Launch this server now?", args.yes):
            return 1
        os.system(cmd)  # noqa: S605 (explicit user-confirmed launch)
    return 0


def cmd_install(args) -> int:
    """End-to-end: detect → recommend → (choose) → pull → set up runtime."""
    report = hardware.detect_all()
    reco = rec.recommend(report)
    render_recommendations(reco, "text")
    model_id = args.model or reco["default_text_model"]
    if not model_id:
        info("No compatible model found automatically. Pick one with --model.")
        return 2
    info(f"\nSelected model: {model_id}")

    # Runtime environment
    mode = args.mode
    if mode == "auto":
        mode = "docker" if runtime.docker_available() else "native"
    info(f"Runtime mode: {mode}")
    if mode == "native":
        runtime.native_setup(dry_run=args.dry_run)
    elif mode == "docker":
        if not runtime.docker_available():
            info("Docker not found. Install Docker or use --mode native.")
            return 1

    # Pull the model
    pull_args = argparse.Namespace(model=model_id, token=args.token,
                                   yes=args.yes, dry_run=args.dry_run)
    rc = cmd_pull(pull_args)
    if rc != 0:
        return rc

    # Optional systemd service
    if args.service and mode == "native":
        m = _find_model(model_id)
        backend = "llama.cpp" if m and m.format == "gguf" else "vllm"
        model_path = os.path.join(MODELS_DIR, model_id)
        exec_start = runtime.serve_command(backend, model_path)
        runtime.install_systemd_unit(
            "text", exec_start, os.path.abspath("."),
            user_scope=not args.system, confirm=lambda msg: confirm(msg, args.yes),
            dry_run=args.dry_run)
    info("\n✓ Install complete. Start chatting with `python -m installer.main serve "
         f"--backend llama.cpp --model {os.path.join(MODELS_DIR, model_id)} --run`")
    return 0


def cmd_status(args) -> int:
    state = _load_state()
    info("Installed models:")
    if not state["models"]:
        info("  (none)")
    for mid, meta in state["models"].items():
        exists = os.path.isdir(meta.get("path", ""))
        info(f"  {'✓' if exists else '✗'} {mid:<22} {meta['kind']:<6} "
             f"{meta['format']:<12} {meta['path']}")
    info("\nDocker services:")
    info(runtime.docker_status())
    return 0


def cmd_uninstall(args) -> int:
    state = _load_state()
    if args.model not in state["models"]:
        info(f"{args.model} is not installed.")
        return 1
    path = state["models"][args.model]["path"]
    if not confirm(f"Delete model files at {path}?", args.yes):
        return 1
    import shutil
    if args.dry_run:
        info(f"[dry-run] would remove {path}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
    del state["models"][args.model]
    if not args.dry_run:
        _save_state(state)
    info(f"✓ Removed {args.model}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="local-llm",
        description="Local LLM Chat installer & manager")
    p.add_argument("--dry-run", action="store_true",
                   help="preview actions without downloading or changing the system")
    p.add_argument("--yes", "-y", action="store_true",
                   help="assume yes for all confirmations (unattended)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Accept the global flags after the subcommand too (argparse otherwise
    # requires them before it), so `... pull --model X --dry-run` also works.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", dest="dry_run",
                        default=argparse.SUPPRESS)
    common.add_argument("--yes", "-y", action="store_true", dest="yes",
                        default=argparse.SUPPRESS)

    def add(name):
        return sub.add_parser(name, parents=[common])

    add("detect").set_defaults(func=cmd_detect)

    pr = add("recommend")
    pr.add_argument("--kind", action="append", choices=["text", "image", "video"])
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_recommend)

    pi = add("install")
    pi.add_argument("--model")
    pi.add_argument("--mode", choices=["auto", "docker", "native"], default="auto")
    pi.add_argument("--token", help="HF token for gated models")
    pi.add_argument("--service", action="store_true", help="install systemd unit")
    pi.add_argument("--system", action="store_true",
                    help="system-scope service (requires sudo)")
    pi.set_defaults(func=cmd_install)

    pp = add("pull")
    pp.add_argument("--model", required=True)
    pp.add_argument("--token")
    pp.set_defaults(func=cmd_pull)

    pq = add("quantize")
    pq.add_argument("src", help="path to HF model dir (or repo for bnb recipe)")
    pq.add_argument("--method", choices=["gguf", "gptq", "bnb"], default="gguf")
    pq.add_argument("--quant", default="Q4_K_M", help="GGUF quant type")
    pq.add_argument("--bits", type=int, default=4)
    pq.add_argument("--out")
    pq.set_defaults(func=cmd_quantize)

    ps = add("serve")
    ps.add_argument("--backend", required=True,
                    choices=list(runtime.SERVE_COMMANDS))
    ps.add_argument("--model", required=True)
    ps.add_argument("--run", action="store_true")
    ps.set_defaults(func=cmd_serve)

    add("status").set_defaults(func=cmd_status)

    pu = add("uninstall")
    pu.add_argument("--model", required=True)
    pu.set_defaults(func=cmd_uninstall)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # propagate top-level flags that subcommands read
    for flag in ("dry_run", "yes"):
        if not hasattr(args, flag):
            setattr(args, flag, getattr(args, flag, False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        info("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
