# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
webui/auth.py — local access control for the web app.

The app binds to loopback by default, but loopback is not a permission
boundary. Two things can reach it that you did not invite:

  * any other process or user account on this machine, and
  * any web page you have open, via DNS rebinding — a site that resolves its
    own hostname to 127.0.0.1 gets same-origin access to this app from inside
    your browser. That is the realistic remote attack against a localhost
    service, and here it would let a page repoint `default_endpoint` at an
    attacker's server and quietly receive every future chat.

So there are two gates, and they answer different attacks:

  1. **Host allow-list.** A rebinding attack has to send the attacker's own
     hostname in the `Host` header, because that is what the browser puts
     there. Rejecting any `Host` we do not recognise ends it. Literal IPs are
     allowed when the app is deliberately bound to a wildcard address, since a
     rebinding attack needs a *name*.
  2. **A shared secret.** A random token, written to a mode-0600 file that only
     your user can read, must accompany every `/api` call. That is what stops
     the other account on the box, and what makes a `0.0.0.0` bind survivable.

The token is accepted three ways, and the difference matters:

  * `X-LLM-Token` header, or `Authorization: Bearer …` — a *custom* header,
    which a cross-origin page cannot set without a CORS preflight this app
    never answers. These are the strong forms.
  * an `llm_token` cookie — convenient for plain `<img src>` links, but a
    cookie is attached by the browser automatically, so it is accepted for
    safe (GET/HEAD) requests only. Anything that mutates state needs a header.

Deliberately absent: CORS headers. Their absence is part of the defence.

Env:
  LLM_WEBUI_TOKEN         use this token instead of generating one
  LLM_WEBUI_TOKEN_FILE    where to persist it (default <project>/.webui_token)
  LLM_WEBUI_HOST          the bind address, used to decide the Host policy
  LLM_WEBUI_ALLOWED_HOSTS extra comma-separated hostnames to accept
"""
from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
from typing import Optional, Set

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOKEN_FILE = os.environ.get("LLM_WEBUI_TOKEN_FILE") or os.path.join(
    _PROJECT_ROOT, ".webui_token")

# Hosts that always mean "this machine".
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}

_TOKEN: Optional[str] = None


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------
def _write_token(path: str, token: str) -> None:
    """Write the token so only the owning user can read it.

    The mode is applied to the descriptor at creation time rather than
    chmod-ed afterwards, so there is no window in which the file exists and is
    world-readable.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (token + "\n").encode("ascii"))
    finally:
        os.close(fd)
    try:                       # a pre-existing file may have looser bits
        os.chmod(path, 0o600)
    except OSError:            # pragma: no cover — e.g. exotic filesystems
        pass


def get_token() -> str:
    """Return the access token, creating and persisting one on first use."""
    global _TOKEN
    if _TOKEN:
        return _TOKEN

    env = (os.environ.get("LLM_WEBUI_TOKEN") or "").strip()
    if env:
        _TOKEN = env
        return _TOKEN

    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="ascii") as fh:
                existing = fh.read().strip()
            if existing:
                _TOKEN = existing
                return _TOKEN
        except OSError:
            pass

    _TOKEN = secrets.token_urlsafe(32)
    try:
        _write_token(TOKEN_FILE, _TOKEN)
    except OSError as exc:     # pragma: no cover — read-only project dir
        print(f"[auth] warning: could not persist token to {TOKEN_FILE}: {exc}")
    return _TOKEN


def reset_token_cache() -> None:
    """Forget the in-process token. Used by tests."""
    global _TOKEN
    _TOKEN = None


def token_matches(candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    return hmac.compare_digest(candidate, get_token())


# ---------------------------------------------------------------------------
# Host header policy
# ---------------------------------------------------------------------------
def _strip_port(host: str) -> str:
    """`example.com:8090` -> `example.com`; `[::1]:8090` -> `::1`."""
    host = host.strip().lower()
    if host.startswith("["):                       # bracketed IPv6 literal
        end = host.find("]")
        return host[1:end] if end > 0 else host
    # Bare IPv6 (no brackets, several colons) carries no port to strip.
    if host.count(":") > 1:
        return host
    return host.rsplit(":", 1)[0] if ":" in host else host


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def allowed_hosts() -> Set[str]:
    extra = os.environ.get("LLM_WEBUI_ALLOWED_HOSTS", "")
    hosts = {h.strip().lower() for h in extra.split(",") if h.strip()}
    bind = (os.environ.get("LLM_WEBUI_HOST") or "127.0.0.1").strip().lower()
    if bind:
        hosts.add(bind)
    return hosts | _LOOPBACK_NAMES


def _bound_to_wildcard() -> bool:
    return (os.environ.get("LLM_WEBUI_HOST") or "127.0.0.1").strip() in (
        "0.0.0.0", "::", "[::]", "*")


def host_allowed(header: Optional[str]) -> bool:
    """True if this `Host` header may be served.

    A missing Host is rejected: HTTP/1.1 requires it, so its absence is either
    a broken client or someone probing.
    """
    if not header:
        return False
    host = _strip_port(header)
    if not host:
        return False
    if host in allowed_hosts():
        return True
    # Reaching a wildcard bind over the LAN means addressing it by IP. A
    # rebinding attack cannot: the browser sends the attacker's *name*.
    return _bound_to_wildcard() and _is_ip_literal(host)


# ---------------------------------------------------------------------------
# Request authorisation
# ---------------------------------------------------------------------------
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def credential_from(request) -> tuple:
    """Return (token, strong) from a Starlette request.

    `strong` marks credentials a cross-origin page cannot forge: custom
    headers are blocked by the browser without a CORS preflight, which this
    app never answers. A cookie rides along automatically, so it is not.
    """
    header = request.headers.get("x-llm-token")
    if header:
        return header.strip(), True
    authorization = request.headers.get("authorization") or ""
    if authorization[:7].lower() == "bearer ":
        return authorization[7:].strip(), True
    cookie = request.cookies.get("llm_token")
    if cookie:
        return cookie.strip(), False
    return None, False


def authorize(request) -> Optional[str]:
    """Return None when the request may proceed, else a reason string."""
    candidate, strong = credential_from(request)
    if not candidate:
        return "missing access token"
    if not token_matches(candidate):
        return "invalid access token"
    if not strong and request.method.upper() not in SAFE_METHODS:
        # A cookie alone is exactly what a cross-site form post would carry.
        return "this request requires the X-LLM-Token header"
    return None
