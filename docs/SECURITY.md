# Security

This document states what Croft defends against, what it does not,
and what changes if you move it off its defaults. It is written to be read
before you expose anything, not after.

## Threat model

**The design point is a single user, on one machine, reaching the app over
loopback.** Everything below follows from that.

In scope — attacks the app is built to survive:

| Threat | Defence |
|---|---|
| Another account on this machine reaching the app on `127.0.0.1` | A random access token, in a mode-0600 file only your user can read. It is printed as a URL only to an interactive terminal — never into a log file — and `webui.sh` sets `umask 077` so the logs are 0600 too |
| A second listener being the weak half of the pair | The diffusion worker on :7862 enforces the same two gates as the app, from the same token file. It is not reachable without the token, and it refuses an unknown `Host` |
| A web page you have open using **DNS rebinding** to reach the app from inside your browser | `Host` header allow-list; rebinding must present the attacker's own hostname |
| A web page making a cross-site request that changes state (CSRF) | Mutating requests require a **custom header**, which needs a CORS preflight this app never answers. No CORS headers are sent, deliberately |
| Another local user reading your chats, uploads or endpoints off disk | `chat.db` is 0600, `models/uploads/` is 0700, uploads are 0600, logs are 0600. These are re-applied on every start, so an install created before the rule existed is repaired rather than left as it was |
| A tampered or truncated model download | SHA256 verified against the manifest for direct downloads; for HF snapshots, a commit SHA pinned in the manifest plus per-file ETag |
| Weights changing under you between installs | Every `hf_repo` entry pins a `revision`. `main` moves and can be force-pushed; a pinned commit means two installs weeks apart fetch identical bytes, and the commit is recorded in `models/installed.json` |
| A hostile value in an environment variable or a path becoming a shell command | No `eval`, no `os.system`, no `shell=True`; every subprocess takes an argv list, and the values that reach one are checked against a closed set |
| Your conversations or `LLM_CHAT_API_KEY` being sent somewhere you did not choose | An inference endpoint must resolve to loopback. Anything else is refused unless you set `LLM_ALLOW_REMOTE_ENDPOINT=1`, and link-local addresses are refused even then. Redirects are disabled, so a 302 cannot relocate the request |
| A downloaded weights file executing code when it loads | Every pipeline is loaded with `use_safetensors=True`. This is a format restriction, not a guarantee about the weights — see below |

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
  roles and no audit log. One token is one level of access: all of it. Refused
  requests are logged to stderr, which is enough to notice probing and not
  enough to reconstruct what happened.
- **Denial of service.** Generation parameters are bounded so a single request
  cannot ask for an unbounded allocation, and uploads are capped as they stream.
  Neither is rate limiting: anything holding the token can keep the accelerator
  busy for as long as it likes.

## The access token

Every route requires it except `/` (the unlock page, which is static HTML) and
`/healthz` (liveness, which reports only "ok" and a version). The list of public
paths is `webui.app.PUBLIC_PATHS`, and a test walks the router to assert nothing
else answers without a credential — the gate used to be "anything under `/api/`",
which silently left `/manager` open. The token is:

- read from `LLM_WEBUI_TOKEN` if set, otherwise generated once and written to
  `.webui_token` (mode 0600) beside the pidfile;
- printed at startup as a ready-to-open URL —
  `http://127.0.0.1:8090/#t=<token>` — **only when stdout is a terminal**. It
  sits in the **fragment**, which browsers never send to a server, so the secret
  reaches the page and stops there. Started by `webui.sh` or a service unit,
  stdout is a log file, and printing it there copied the secret into a
  world-readable file: get the link with `./webui.sh url` instead;
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
- The Docker install script is downloaded to a `mktemp -d` directory (mode
  0700) and you are told where it is before it runs as root — instead of
  `curl | sh`, which gives you no chance to look. A fixed `/tmp` path would let
  another local user swap the file during the confirmation prompt, which is a
  race that ends in root.
- `LLM_PKG_URL` requires `LLM_PKG_SHA256`. The archive is extracted and its
  installer is executed, so an unverified one is remote code execution; the
  bootstrap refuses rather than warning and continuing. The download is
  restricted to `--proto '=https'`, and extraction uses `--no-same-owner
  --no-same-permissions` so an archive cannot choose the mode of what it writes.
- `requirements.lock` carries a hash for every artefact and is installed with
  `pip --require-hashes`, so a substituted wheel fails rather than installing.
- Joining the `docker` group is root-equivalent on the host; the bootstrap says
  so and asks separately before doing it.
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

## Reporting a vulnerability

**Use [private vulnerability reporting](https://github.com/billalaashraf/Croft/security/advisories/new).**
It opens a draft advisory only you and the maintainers can see, so nothing is
disclosed before there is a fix.

Please do not open a public issue for anything exploitable — including an issue
that says "I found something, contact me". That still announces a vulnerability
exists and points people at the right place to look.

What helps: the version or commit, the platform, and the smallest reproduction
you have. If you are not sure whether something counts, report it privately
anyway; a false alarm costs far less than the alternative.

Expect an acknowledgement within a week. This is a small project with no
security team and no bounty — the honest commitment is that reports are read
and acted on, not that they are triaged within hours.

For non-security bugs, [open an issue](https://github.com/billalaashraf/Croft/issues/new/choose).
