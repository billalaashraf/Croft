# Croft — automated installer & manager

[![CI](https://github.com/billalaashraf/Croft/actions/workflows/ci.yml/badge.svg)](https://github.com/billalaashraf/Croft/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](pyproject.toml)

A cross-platform, automated installer and management system for a **locally
hosted** chat server with **text, image, and video** generation. It probes your
hardware, tells you which models will actually run, downloads them (resumable +
checksum-verified), configures a runtime (Docker, virtualenv, or systemd
service), and gives you a CLI/TUI plus a local **chat web app** (talk to your
models, pick one per task, with saved history).

Everything runs locally. No paid cloud APIs. Weights come from official sources
or Hugging Face (with your own read-scope token for gated models).

The `bootstrap_install.sh` script turns this into a single guided command: it
installs dependencies, walks you through picking text/image/video models,
downloads them (resumably), installs the engines needed to run them, and starts
the chat + image web app in the background. Runs are resumable — an interrupted
install can be continued where it left off.

## Project layout

Directory-level on purpose. A file-by-file map in a README is guaranteed to
drift — this one used to list two of the six test files.

| Path | What lives here |
|------|-----------------|
| `bootstrap_install.sh` | The guided install: prereqs, venv + deps, model picker, resume, engines, web UI |
| `bootstrap_install.ps1` | Windows → WSL2 bootstrap stub |
| `webui.sh` | start / stop / restart / status / logs for the web app |
| `models_manifest.json` | Model catalog + schema. Drives recommendation, download, revision pinning and integrity |
| `installer/` | The CLI: hardware probe, VRAM formula and ranking, resumable downloader, quantisation, docker/venv/systemd orchestration |
| `webui/` | The FastAPI app — chat, images, video, the manager panel, SQLite history, and the self-contained SPA in `static/` |
| `docker/` | Compose stack and Dockerfiles for the text, SD and manager services |
| `services/` | systemd units and a launchd agent for running Croft as a service |
| `tests/` | pytest suite plus `test_bootstrap.sh` for the shell paths pytest cannot reach |
| `docs/` | [COMPATIBILITY.md](docs/COMPATIBILITY.md) (the formula and tier tables) and [SECURITY.md](docs/SECURITY.md) (the threat model) |

## Installation

### Prerequisites

- **OS**: Linux, macOS, or Windows via **WSL2**.
- **Python 3.10+** with `venv` (`python3 --version`). The bootstrap can install it
  on Debian/Ubuntu (`apt`) and macOS (Homebrew); otherwise install it first.
- **Disk space** for the models you choose — a 7B GGUF chat model is ≈ 4–5 GB, an
  SDXL image model ≈ 7 GB+.
- **Git or curl** to get the code, plus a C compiler for the chat engine
  (`xcode-select --install` on macOS; `build-essential` on Debian/Ubuntu).
- **Optional**: an NVIDIA GPU + CUDA driver, or Apple Silicon (MPS), for speed —
  everything also runs on CPU. Docker only if you want the containerised backends.
- **Optional**: a Hugging Face token in `HF_TOKEN` for gated models (Llama, etc.).

### Step 0 — get the code

```bash
git clone https://github.com/billalaashraf/Croft.git croft && cd croft
# …or download and extract the archive, then: cd croft
```

### Option A — one guided command (recommended)

```bash
chmod +x bootstrap_install.sh
./bootstrap_install.sh
```

Run from inside the checkout, this does **every** step for you — detect hardware,
create a `.venv` and install dependencies, recommend models, let you **pick which
text / image / video models to install**, download them (resumable), install the
engines that run them, and launch the web app in the background. The full,
per-step breakdown and the interactive picker are documented in
[What the bootstrap does](#what-the-bootstrap-does).

When it finishes, open **http://127.0.0.1:8090**.

Unattended (CI / headless): `./bootstrap_install.sh --yes` installs the
recommended default text model and starts the app with no prompts. Preview
without changing anything: `./bootstrap_install.sh --dry-run`.

### Option B — manual, step by step

Prefer to run each step yourself (or can't use the bootstrap)? This is exactly
what Option A automates:

```bash
# 1. Create a virtualenv and install core dependencies
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. See what your machine can run (marks each model fits / won't fit)
python3 -m installer.main recommend

# 3. Download a model — the auto-recommended one, or a specific id from step 2
python3 -m installer.main install                        # best-fitting text model
python3 -m installer.main install --model qwen2.5-7b-q4  # a specific model

# 4. Install the engine that RUNS the model (weights alone can't run) — once:
pip install llama-cpp-python                              # chat: embedded GGUF
pip install torch diffusers transformers accelerate safetensors pillow  # image gen

# 5. Start the web app (background)
./webui.sh start                                         # → http://127.0.0.1:8090
# …or run it in the foreground: uvicorn webui.app:app --host 127.0.0.1 --port 8090
```

Open **http://127.0.0.1:8090**, choose a model + task, and chat. The first
message after a start pauses ~10–15 s while the model loads into memory, then
stays resident and responds quickly.

Every `installer.main` command supports `--dry-run` (preview, no
downloads/changes) and `--yes` (assume-yes).

### Platform notes

- **macOS (Apple Silicon)**: the MPS backend is used automatically; `llama-cpp-python`
  and `torch` build and run with Metal. Needs Xcode command-line tools to compile
  the chat engine.
- **Linux + NVIDIA**: install the CUDA driver, and the NVIDIA Container Toolkit if
  you use Docker — the bootstrap advises but never auto-installs these.
- **Windows**: "Windows support" here means **WSL2**, and only WSL2.
  `bootstrap_install.ps1` checks that WSL is installed and that your default
  distro is version 2 (offering to install or convert), then hands off to
  `bootstrap_install.sh` inside it and smoke-tests the result. Native Windows
  is out of scope because the inference stacks this installs — vLLM, TGI, and
  the CUDA/Metal builds of llama.cpp — have no native-Windows support; CUDA
  reaches WSL through the host NVIDIA driver. Windows forwards `localhost` into
  WSL2, so the printed URL opens in a Windows browser unchanged.
- **CPU-only**: fully supported — pick a small quantised GGUF (e.g. a 3B) for
  usable speed.

## What the bootstrap does

`bootstrap_install.sh` is a guided, resumable installer for Linux, macOS, and
WSL2. In order it:

1. **Detects** your OS, architecture, and accelerator (CUDA / ROCm / Apple MPS / CPU).
2. **Locates itself.** Run from inside a checkout (the installer sits beside it)
   and it installs *in place*; otherwise it falls back to `$HOME/croft`
   and fetches the package (tarball via `LLM_PKG_URL`, else `git clone`).
3. **Ensures prerequisites** — Python in both modes, plus Docker for docker mode.
   GPU drivers are advised, never auto-installed.
4. **Creates a `.venv` and installs core dependencies** (`huggingface_hub`,
   `requests`, `rich`, `uvicorn`, …) *before* any download, then runs the
   installer through that venv. This is what lets the machine download at all —
   without it the download fails with `huggingface_hub not installed`.
5. **Recommends models** for your tier and opens an interactive picker.
6. **Downloads** your selections (resumable + checksum-verified).
7. **Installs the inference engines** the chosen models need so they actually
   run — chat/GGUF → `llama-cpp-python` (embedded), image/video → the
   `torch` + `diffusers` stack — into the same venv the web app uses. Native
   mode only (docker runs models in containers). Skip with `LLM_NO_ENGINES=1`.
8. **Launches the chat + image web app** detached and prints its URL.

### Choosing models (interactive)

After the recommendation report you're walked through **three pickers in order —
text, then image, then video**. Each lists the models for that kind and accepts:

| Input | Meaning |
|-------|---------|
| numbers (`1 3`, `2,4`) | install those models |
| `d` | the recommended default (text only) |
| `a` | every model of this kind that fits (already-installed skipped, partials resumed) |
| `s` | skip this kind |

Every row is tagged **`✓ installed`** (already complete), **`◐ resume`** (a partial
download to continue), **`fits`**, or **`won't fit`**. Skip all three kinds and
you're offered **restart** or **quit**.

With `--yes` the bootstrap is unattended: it installs the recommended default
text model and skips image/video.

### Background web panel

When the installs finish, the bootstrap starts `webui/app.py` in the background
and prints a browser URL (default **http://127.0.0.1:8090**):

| Variable | Default | Effect |
|----------|---------|--------|
| `LLM_WEBUI_HOST` | `127.0.0.1` | bind address. Setting `0.0.0.0` publishes the app to every network this machine is on — do that only with `LLM_WEBUI_TOKEN` set to a value you chose and `LLM_WEBUI_ALLOWED_HOSTS` naming the hostname you will use. Read [docs/SECURITY.md](docs/SECURITY.md) first |
| `LLM_WEBUI_PORT` | `8090` | port |
| `LLM_WEBUI_TOKEN` | generated | access token for `/api/*`. Generated once into `.webui_token` (mode 0600) when unset |
| `LLM_WEBUI_ALLOWED_HOSTS` | unset | extra comma-separated hostnames the `Host` allow-list will accept |
| `LLM_NO_WEBUI` | unset | set to `1` to skip launching it |

**The URL carries an access token.** The bootstrap prints
`http://127.0.0.1:8090/#t=<token>` — open that link rather than the bare
address. The token sits in the fragment, which browsers never send to a server,
so it reaches the page and goes no further. Lost it? It is in `.webui_token`,
and the app also logs the link on startup; paste it into the unlock box.

Loopback is not on its own a permission boundary — any process on the machine
can reach it, and a web page can get there by DNS rebinding — so the app also
checks the `Host` header and requires that token on every `/api` call. Details
and the full threat model: [docs/SECURITY.md](docs/SECURITY.md).

Native mode logs to `webui.log` and writes a PID to `.webui.pid` (stop it with
`kill $(cat .webui.pid)`); docker mode runs the `manager` compose service (stop
it with `docker compose -f docker/docker-compose.yml stop manager`).

### Resuming an interrupted install

Installs resume at two levels:

- **Download level** (always on): an interrupted download keeps its partial
  files and continues byte-for-byte on the next attempt — no progress lost.
- **Session level**: the bootstrap journals your selections to
  `.bootstrap_session` and marks each done as it completes. If a run is
  interrupted, the next run detects the unfinished models and offers **resume**
  (reinstall just those) or **start fresh** — and `--yes` auto-resumes.
  Independently, any partially-downloaded model shows as `◐ resume` in the
  picker, so you can always continue it manually.

To resume after a download process was stopped, re-run the bootstrap and pick the
`◐ resume` model, or continue it directly (it picks up from the partial files):

```bash
python3 -m installer.main pull --model sdxl-turbo --yes
```

### Bootstrap flags & environment

```bash
./bootstrap_install.sh --mode native --yes      # unattended native install
./bootstrap_install.sh --dir /path/to/checkout  # override install location
./bootstrap_install.sh --dry-run                # preview everything, touch nothing
```

Environment: `LLM_INSTALL_DIR` (install location), `LLM_MODELS_DIR` (model root,
default `models`), `LLM_REPO_URL` / `LLM_PKG_URL` / `LLM_PKG_SHA256` (package
source when not run from a checkout), `LLM_NO_ENGINES=1` (skip the engine
install), plus the `LLM_WEBUI_*` / `LLM_NO_WEBUI` variables above.

## Interactive vs unattended

```bash
# Interactive — prompts for consent before each download / system change
python3 -m installer.main install --mode native

# Unattended — CI/headless, accept all, specific model, HF token via env
export HF_TOKEN=hf_xxx
python3 -m installer.main --yes install \
    --model llama3.1-8b-fp16 --mode docker --service
```

Environment variables: `HF_TOKEN` (gated models), `LLM_MODELS_DIR` (model
storage root, default `./models`), `LLM_INSTALL_DIR`, `TEXT_MODEL_ID`,
`TGI_QUANTIZE`, `GGUF_FILE` (compose).

## Downloading a specific model + quantizing it

```bash
# Pull a gated model (installer prompts you to accept the license first)
python3 -m installer.main pull --model llama3.1-8b-fp16 --token $HF_TOKEN

# Convert HF weights → GGUF → 4-bit for CPU/edge (llama.cpp)
python3 -m installer.main quantize models/llama3.1-8b-fp16 \
    --method gguf --quant Q4_K_M

# 4-bit GPTQ (GPU)
python3 -m installer.main quantize meta-llama/Llama-3.1-8B-Instruct --method gptq

# Print a bitsandbytes 4-bit runtime recipe (no offline conversion)
python3 -m installer.main quantize mistralai/Mistral-7B-Instruct-v0.2 --method bnb
```

## Switching backends & starting image/video

```bash
# Text backends
python3 -m installer.main serve --backend llama.cpp --model <gguf>   # CPU / small GPU
python3 -m installer.main serve --backend vllm      --model <dir>    # GPU throughput
python3 -m installer.main serve --backend tgi       --model <repo>   # HF TGI

# Docker: bring up profiles
docker compose -f docker/docker-compose.yml --profile text  up -d    # text
docker compose -f docker/docker-compose.yml --profile image up -d    # Stable Diffusion
docker compose -f docker/docker-compose.yml --profile manager up -d  # status panel :8090
```

## Chat web app

The web UI at **http://127.0.0.1:8090** is a full local studio — **chat**,
**text-to-image**, and **video**, switchable from the top-left. The
bootstrap launches it in the background (see
[Background web panel](#background-web-panel)). To control it yourself, use the
`webui.sh` helper (start / stop / restart / status / logs):

```bash
./webui.sh start          # launch in the background (creates .venv if needed)
./webui.sh status         # up? and is it the chat app or the old panel?
./webui.sh restart        # stop then start
./webui.sh stop
./webui.sh logs -f        # follow the log

# Manage the Docker `manager` container instead of a native process:
LLM_WEBUI_MODE=docker ./webui.sh restart   # rebuilds + restarts the container
```

It reuses `.webui.pid` / `webui.log` and honours `LLM_WEBUI_HOST` /
`LLM_WEBUI_PORT`. Or start it directly:
`uvicorn webui.app:app --host 127.0.0.1 --port 8090`.

What it does:

- **Chat with your installed models** — streaming replies, rendered markdown /
  code blocks, entirely on your machine.
- **Pick a model + task per conversation.** The model dropdown lists your
  installed text models; the task preset (General / Coding / Creative / Precise)
  sets the system prompt and sampling — this is how you point a conversation at
  the right model for the job.
- **Full history.** Every conversation and message is saved in SQLite and listed
  in the sidebar; reopen, continue, or delete any past chat.
- **Attach files to analyse.** Use the paperclip (or drag a file onto the chat)
  to attach text/code/JSON/CSV/Markdown, `.pdf`, or `.docx`; the extracted text
  is folded into your message so the model can summarise, review, or answer
  questions about it — and it stays in context for follow-up turns. Extraction
  is capped (`LLM_FILE_MAXCHARS`, default 16000) so a big file can't overflow the
  context. PDFs need `pypdf`, `.docx` needs `python-docx`; images aren't analysed
  by a text model. Uploads are kept under `models/uploads/`.

### How it talks to a model

Two interchangeable backends, resolved per model:

1. **Embedded GGUF** — if `llama-cpp-python` is installed, a GGUF model runs
   in-process with no separate server (`pip install llama-cpp-python`).
2. **OpenAI-compatible endpoint** — point a model (or a global default) at any
   `/v1`-style server: the project's own `serve` command, vLLM, TGI, or Ollama.
   Set it via the in-app settings (the sliders icon); common local servers
   (`:8080`, `:8000`, `:8081`, `:11434`) are auto-detected. Optional bearer via
   `LLM_CHAT_API_KEY`.

For example, serve your installed model and the chat app will use it:

```bash
python3 -m installer.main serve --backend llama.cpp \
    --model models/qwen2.5-7b-q4/*.gguf --run     # OpenAI API on :8080
# then set the default endpoint to http://127.0.0.1:8080/v1 in the chat settings
```

Chat-app environment: `LLM_CHAT_DB` (history DB, default `<models>/chat.db`),
`LLM_CHAT_API_KEY` (endpoint auth), `LLM_OLLAMA_HOST` (default
`http://127.0.0.1:11434`). Every variable Croft reads is listed with its default
in [.env.example](.env.example).

### Images (text-to-image)

The **Images** tab (top-left switch) generates pictures from your installed
image models (e.g. `sdxl-turbo`) — prompt in, image out, all local. Results are
saved to `models/sd-outputs/` and kept in a gallery (sidebar + main grid) that
persists across restarts.

It needs the diffusers stack, imported lazily — until then the tab shows a
one-line install hint instead of failing:

```bash
pip install torch diffusers transformers accelerate safetensors pillow
```

Turbo/LCM models default to a few steps and no guidance; others to 25 steps.
Env: `LLM_SD_DTYPE` (e.g. `float32` if fp16 gives black images on MPS),
`LLM_MODELS_DIR` (outputs go to `<models>/sd-outputs`). Image APIs:
`/api/image/models`, `/api/image/generate`, `/api/image/history`,
`/api/image/file/{id}`.

### Video (text-to-video / image-to-video)

The **Video** tab generates clips from installed video models:

- **Text-to-video** (AnimateDiff, e.g. `animatediff-sd15`) — a prompt → a short
  clip. AnimateDiff rides on an SD1.5 base (`LLM_ANIMATEDIFF_BASE`, default
  `runwayml/stable-diffusion-v1-5`).
- **Image-to-video** (Stable Video Diffusion, e.g. `svd-xt`) — attach a source
  image (the source-image button appears automatically for these models) → an
  animated clip.

Set frames / fps / steps in the composer. Clips render as `.mp4` (via
`imageio-ffmpeg`) or `.gif` fallback, save to `models/sd-outputs/`, and appear in
a gallery that plays on hover. Needs the same diffusers stack as images plus
`imageio imageio-ffmpeg` — until then the tab shows an install hint. Install a
video model first (`python3 -m installer.main install --model animatediff-sd15`).
Video APIs: `/api/video/models`, `/api/video/generate`, `/api/video/history`,
`/api/video/file/{id}`.

### Manager panel & APIs

The original status panel lives at **`/manager`** (hardware report, installed
models, per-GPU VRAM). JSON APIs: `/api/hardware`, `/api/recommend`,
`/api/status`, plus the chat APIs (`/api/models`, `/api/conversations`,
`/api/conversations/{id}/chat`, `/api/settings`).

## How model recommendation works

See `docs/COMPATIBILITY.md` for the full formula and tables. In short:
`required_GB = params × bytes_per_param × 1.5 + 1`. The engine measures each
model against your VRAM (or RAM for CPU GGUF), marks `fits`, and ranks fitting
models by capability. GPU-too-small automatically demotes to a CPU/GGUF or
frame-by-frame variant.

## Adding custom models

Append an entry to `models_manifest.json` following the `schema` block at the
top of that file. Two fields are what make an entry trustworthy, and the test
suite enforces both:

- **`integrity_sha256`** — required for any entry with a `download_url`. A
  direct download with no hash is an unverifiable download.
- **`revision`** — required for any entry with an `hf_repo`. Pin the commit SHA,
  not `main`: branches move and can be force-pushed, so an unpinned entry means
  two people installing the same id weeks apart can get different weights with
  no way to notice. Get it with:

  ```bash
  curl -s https://huggingface.co/api/models/<owner>/<repo> | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'
  ```

Set `gated: true` if the repo requires accepting a licence on Hugging Face. The
new id is immediately usable: `python3 -m installer.main pull --model <your-id>`.

If the file fails to parse, Croft warns on stderr and falls back to the built-in
catalog — so a typo shows up as a warning rather than a model that silently
never appears.

## Design decisions & limitations

- **Consent-first**: no download or system change happens without a prompt
  (or explicit `--yes`). CUDA/ROCm drivers are *checked and advised*, never
  auto-installed, to avoid breaking your graphics stack.
- **Graceful degradation**: missing `psutil`/`pynvml`/vendor tools fall back to
  `/proc`, `sysctl`, or best-effort probes; warnings surface in the report.
- **No bundled restricted weights**: only pointers to official repos.
- **Estimates are heuristics**: real VRAM use varies with context length, batch
  size, and framework. Numbers are deliberately conservative (1.5× overhead).
- **Windows**: GPU inference is via WSL2 only (native Windows CUDA for these
  stacks is unsupported here).
- **Local, not trusted**: binding to loopback is not a permission boundary, so
  the app requires an access token and validates the `Host` header. It is
  single-user by design — one token is one level of access, all of it. See
  [docs/SECURITY.md](docs/SECURITY.md).

## Uninstall

```bash
python3 -m installer.main uninstall --model mistral-7b-q4   # removes weights (prompts)
kill $(cat .webui.pid)                                      # stop the native web panel
docker compose -f docker/docker-compose.yml down            # stop containers
systemctl --user disable --now croft-text                   # stop and disable the service
```

`disable --now` stops the unit and removes its autostart symlink, but leaves the
unit file in place. To remove it completely:

```bash
systemctl --user disable --now croft-text
rm ~/.config/systemd/user/croft-text.service
systemctl --user daemon-reload
```

On macOS the equivalent is `launchctl unload
~/Library/LaunchAgents/com.croft.manager.plist` followed by deleting that file.

## Testing

```bash
pip install -e '.[dev]'      # or: pip install pytest httpx
pytest -q
```

CI runs the same suite on Python 3.10–3.14 across Ubuntu and macOS, plus
`shellcheck`, a lockfile install, and a nightly job that installs the real
inference engines and asserts they import — see `.github/workflows/ci.yml`.

## Contributing

Bug reports and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md)
covers the dev setup, what CI will check, and the conventions this codebase
follows. Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

Found a security problem? **Don't open a public issue** — use
[GitHub's private vulnerability reporting](https://github.com/billalaashraf/Croft/security/advisories/new).
See [docs/SECURITY.md](docs/SECURITY.md).

## License

Croft is licensed under the [Apache License 2.0](LICENSE).

**This does not cover model weights.** Croft bundles none — it downloads them
from their original repositories, and each carries its own licence. Some are
non-commercial (Stable Video Diffusion, SDXL-Turbo); some are gated and require
you to accept terms on Hugging Face before any bytes move, which the installer
enforces as an explicit step. Every model's licence is recorded in
[models_manifest.json](models_manifest.json) and, once installed, in
`models/installed.json` alongside the exact commit it came from. Accepting them
is your responsibility.

See [docs/SECURITY.md](docs/SECURITY.md) for the threat model — what the app
defends against, what it explicitly does not, and what changes if you bind it to
anything other than loopback — and [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)
for hardware notes.
