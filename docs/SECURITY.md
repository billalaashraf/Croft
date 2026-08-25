# Security

This document states what Local LLM Chat defends against, what it does not,
and what changes if you move it off its defaults. It is written to be read
before you expose anything, not after.

## Threat model

**The design point is a single user, on one machine, reaching the app over
loopback.** Everything below follows from that.

In scope — attacks the app is built to survive:

| Threat | Defence |
|---|---|
| Another account on this machine reaching the app on `127.0.0.1` | A random access token, in a mode-0600 file only your user can read |
| A web page you have open using **DNS rebinding** to reach the app from inside your browser | `Host` header allow-list; rebinding must present the attacker's own hostname |
| A web page making a cross-site request that changes state (CSRF) | Mutating requests require a **custom header**, which needs a CORS preflight this app never answers. No CORS headers are sent, deliberately |
| Another local user reading your chats, uploads or endpoints off disk | `chat.db` is 0600, `models/uploads/` is 0700, uploads are 0600 |
| A tampered or truncated model download | SHA256 verified against the manifest for direct downloads; commit hash + per-file ETag for HF snapshots |
| A hostile value in an environment variable or a path becoming a shell command | No `eval`, no `os.system`, no `shell=True`; every subprocess takes an argv list |

Out of scope — say so plainly rather than implying cover:

- **A hostile local user who is already root, or who is you.** Anything that
  can read your home directory can read the token.
- **Malicious model weights.** A GGUF or a `.safetensors` file is loaded by
  `llama.cpp` / `diffusers`, and a checksum proves only that you got the bytes
  the manifest names — not that those bytes are safe. Download from
  repositories you have reason to trust.
- **Prompt injection through an attached document.** Text extracted from an
  upload goes into the model's context. The model has no tools here, so the
  blast radius is its own reply, but do not treat that reply as trusted.
- **Multi-user or internet-facing deployment.** There are no user accounts, no
  roles and no audit log. One token is one level of access: all of it.

## The access token

Every `/api/*` route requires it. The token is:

- read from `LLM_WEBUI_TOKEN` if set, otherwise generated once and written to
  `.webui_token` (mode 0600) beside the pidfile;
- printed at startup as a ready-to-open URL —
  `http://127.0.0.1:8090/#t=<token>`. It sits in the **fragment**, which
  browsers never send to a server, so the secret reaches the page and stops
  there;
- accepted in three ways, which are not equivalent:

| Form | Accepted for |
|---|---|
| `X-LLM-Token: <token>` header | everything |
| `Authorization: Bearer <token>` | everything |
| `llm_token` cookie | **GET/HEAD only** |

The cookie exists so plain `<img src>` links work. It is refused for anything
that mutates state, because a cookie is exactly what a cross-site form post
would carry automatically. That is the CSRF boundary.

Rotate the token by deleting `.webui_token` and restarting; every open tab will
need the new link.

```bash
# what a client without the token gets
curl -X POST http://127.0.0.1:8090/api/settings          # 401
curl -H 'Host: evil.example' http://127.0.0.1:8090/api/models   # 421
```

## Binding to something other than loopback

`LLM_WEBUI_HOST=0.0.0.0` publishes the app to every network the machine is on.
If you do it:

1. The token is now the **only** thing between the network and your chat
   history and settings. Set `LLM_WEBUI_TOKEN` to a value you generated
   yourself rather than relying on the on-disk file.
2. Set `LLM_WEBUI_ALLOWED_HOSTS` to the hostname you will actually use.
   Hostnames are refused unless listed; bare IP addresses are allowed under a
   wildcard bind, since DNS rebinding needs a *name*.
3. Prefer an SSH tunnel or a reverse proxy that terminates TLS. This app speaks
   plain HTTP: on an untrusted network the token and every message are readable
   in transit.

The inference backends are a separate matter, and worse: `llama-server`, vLLM,
TGI and the Stable Diffusion web UI have **no authentication at all**. The
serve commands and compose file bind them to `127.0.0.1` for that reason. A
`--host 0.0.0.0` there is an open inference endpoint, and for the SD web UI, a
file browser as well.

## Docker

- Every published port is bound to `127.0.0.1` in `docker/docker-compose.yml`.
  This matters more than it looks: Docker writes its own iptables rules, so a
  bare `8090:8090` is reachable from the whole network **and bypasses a host
  firewall**.
- The `manager` service does **not** mount `/var/run/docker.sock`. Anything
  that can talk to a container holding that socket can start a privileged
  container and own the host. Without it, `docker_status()` reports
  "unavailable in this container" and everything else works.
- If you want the compose readout anyway, opt in explicitly:
  `docker compose --profile manager-privileged up -d`. Understand that you have
  made the panel root-equivalent on the host.
- The manager image runs as uid 10001, not root.
- `--enable-insecure-extension-access` is not set on the SD web UI. It lets
  anything that reaches that port install and run arbitrary extension code.

## Installing and downloading

- Nothing is downloaded or changed without a prompt, or an explicit `--yes`.
  `--dry-run` previews every action.
- Gated models require you to confirm you have accepted the licence before any
  bytes move.
- The Docker install script is downloaded to `/tmp/get-docker.sh` and you are
  told where it is before it runs as root — instead of `curl | sh`, which gives
  you no chance to look.
- systemd units default to user scope. System scope asks first.
- Store weights under a directory owned by the service user, mode 0750:
  `install -d -m 0750 -o "$USER" -g "$USER" models`. `HF_HOME` inherits it.

## Licensing

Model licences are your responsibility. Each installed model records its
`license` and `source_url` in `models/installed.json` for auditability, and
non-commercial licences (Stable Video Diffusion, for instance) are labelled in
the manifest. Provide a **read-scope** Hugging Face token via `--token` or
`$HF_TOKEN`; never commit it. `huggingface-cli login` stores it in
`~/.cache/huggingface` with user-only permissions.

## Reporting a problem

Open an issue with reproduction steps. If the issue is exploitable and you
would rather not post it publicly, say so in the issue without the details and
ask for a private channel first.
