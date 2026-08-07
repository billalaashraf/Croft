# Security & Licensing checklist

## Model provenance & licensing
- [ ] Only official Hugging Face / GitHub sources are referenced; no restricted
      weights are bundled in this repo — the manifest stores pointers only.
- [ ] `gated: true` models require an explicit license-acceptance prompt in the
      installer flow **before** any bytes are downloaded (see `cmd_pull`).
- [ ] Provide your own **read-scope** Hugging Face token via `--token` or
      `$HF_TOKEN`; never commit tokens. Prefer `huggingface-cli login` which
      stores the token in `~/.cache/huggingface` with user-only permissions.
- [ ] Each installed model records its `license` and `source_url` in
      `models/installed.json` for auditability.
- [ ] Non-commercial licenses (e.g. Stable Video Diffusion) are labelled; do not
      use them commercially.

## Integrity
- [ ] Single-file downloads (GGUF via `download_url`) are SHA256-verified against
      the manifest's `integrity_sha256`; a mismatch aborts and preserves the file
      for inspection.
- [ ] HF snapshot repos are verified by commit hash + per-file ETag by
      `huggingface_hub` (resumable, tamper-evident).
- [ ] Bootstrap verifies the installer package checksum (`LLM_PKG_SHA256`) before
      extraction; missing checksum triggers a loud warning.

## System safety
- [ ] No root-level change happens without confirmation; CUDA/ROCm drivers are
      *advised*, never auto-installed, to avoid breaking the graphics stack.
- [ ] `--dry-run` previews every action; `--yes` is required for unattended runs.
- [ ] systemd units default to **user scope** (`--user`, no sudo); system scope
      requires an explicit confirmation.
- [ ] Uninstall prompts before deleting weights.

## Filesystem permission model for model storage
- [ ] Store weights under a dedicated `models/` dir owned by the service user,
      mode `0750` (owner rwx, group rx, world none):
      `install -d -m 0750 -o "$USER" -g "$USER" models`
- [ ] The HF cache (`HF_HOME=models/.hf-cache`) inherits the same restriction.
- [ ] In Docker, mount `./models` read-write only into inference containers; the
      manager mounts the Docker socket read-only where possible.
- [ ] Do not expose serving ports (8080/8000/7860/8090) to untrusted networks;
      bind to localhost or place behind an authenticated reverse proxy.
