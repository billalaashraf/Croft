# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
Web app access control and data hygiene.

These are the tests that would have caught the app shipping with no auth at
all: they assert the two gates exist (token, Host allow-list), that a cookie is
not a substitute for a header on a mutating request, and that deleting a
conversation actually deletes the files attached to it.

Everything runs against a temporary models dir, so no test touches a real
chat.db or a real upload.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi is needed for the web app tests")

TOKEN = "test-token-not-a-real-secret"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient over a freshly imported app rooted at a temp models dir.

    These modules read their paths at import time, so the environment has to be
    set *before* the reload — and every module that cached a path has to be
    reloaded, not only the app.
    """
    monkeypatch.setenv("LLM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LLM_CHAT_DB", str(tmp_path / "models" / "chat.db"))
    monkeypatch.setenv("LLM_WEBUI_TOKEN", TOKEN)
    monkeypatch.setenv("LLM_WEBUI_HOST", "127.0.0.1")
    monkeypatch.delenv("LLM_WEBUI_ALLOWED_HOSTS", raising=False)

    from webui import auth, chatstore, files
    for module in (auth, files, chatstore):
        importlib.reload(module)
    auth.reset_token_cache()

    import webui.app as appmod
    importlib.reload(appmod)

    with fastapi_testclient.TestClient(appmod.app) as c:
        c.headers.update({"Host": "127.0.0.1:8090"})
        yield c


def auth_headers():
    return {"X-LLM-Token": TOKEN}


# ---------------------------------------------------------------------------
# The token gate
# ---------------------------------------------------------------------------
def test_api_requires_a_token(client):
    assert client.get("/api/models").status_code == 401


def test_mutating_route_requires_a_token(client):
    r = client.post("/api/settings", json={"default_endpoint": "http://evil"})
    assert r.status_code == 401
    assert r.json()["unauthorized"] is True


def test_a_wrong_token_is_refused(client):
    assert client.get("/api/models",
                      headers={"X-LLM-Token": "wrong"}).status_code == 401


def test_the_right_token_is_accepted(client):
    assert client.get("/api/models", headers=auth_headers()).status_code == 200


def test_bearer_form_is_accepted(client):
    r = client.get("/api/models", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_the_page_itself_is_not_gated(client):
    """`/` must serve without a token, or the unlock screen can never load."""
    assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# Cookie vs header: the CSRF boundary
# ---------------------------------------------------------------------------
def test_cookie_alone_can_read(client):
    client.cookies.set("llm_token", TOKEN)
    assert client.get("/api/models").status_code == 200


def test_cookie_alone_cannot_write(client):
    """A cookie is what a cross-site form post carries automatically, so it
    must not be enough to change anything."""
    client.cookies.set("llm_token", TOKEN)
    r = client.post("/api/settings", json={"default_endpoint": "http://evil"})
    assert r.status_code == 401
    assert "header" in r.json()["error"]


def test_cookie_plus_header_can_write(client):
    client.cookies.set("llm_token", TOKEN)
    r = client.post("/api/settings", json={"default_endpoint": "http://ok"},
                    headers=auth_headers())
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# The Host allow-list (DNS rebinding)
# ---------------------------------------------------------------------------
def test_foreign_host_is_refused_even_with_a_valid_token(client):
    """A rebinding attack holds a valid session but has to send its own
    hostname, which is what gives it away."""
    r = client.get("/api/models",
                   headers={**auth_headers(), "Host": "evil.example"})
    assert r.status_code == 421


def test_foreign_host_is_refused_on_the_page_too(client):
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 421


@pytest.mark.parametrize("host", ["127.0.0.1:8090", "localhost:8090",
                                  "localhost", "[::1]:8090"])
def test_loopback_hosts_are_accepted(client, host):
    r = client.get("/api/models", headers={**auth_headers(), "Host": host})
    assert r.status_code == 200


def test_allowlisted_host_is_accepted(client, monkeypatch):
    monkeypatch.setenv("LLM_WEBUI_ALLOWED_HOSTS", "workstation.local")
    r = client.get("/api/models",
                   headers={**auth_headers(), "Host": "workstation.local:8090"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Uploads and deletion
# ---------------------------------------------------------------------------
def test_oversized_upload_is_rejected(client, monkeypatch):
    from webui import files
    monkeypatch.setattr(files, "MAX_BYTES", 16)
    r = client.post("/api/attach", headers=auth_headers(),
                    files={"file": ("big.txt", b"x" * 64, "text/plain")})
    assert r.status_code == 413


def test_uploads_are_written_owner_only(client):
    r = client.post("/api/attach", headers=auth_headers(),
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 200
    from webui import chatstore
    row = chatstore.get_attachment(r.json()["id"])
    assert oct(os.stat(row["path"]).st_mode & 0o777) == "0o600"


def test_the_database_is_owner_only(client):
    from webui import chatstore
    assert oct(os.stat(chatstore.db_path()).st_mode & 0o777) == "0o600"


def test_deleting_a_conversation_removes_its_uploads(client):
    """The whole point: a deleted chat must not leave the document behind, nor
    its extracted text in the database."""
    from webui import chatstore

    up = client.post("/api/attach", headers=auth_headers(),
                     files={"file": ("secret.txt", b"confidential", "text/plain")})
    aid = up.json()["id"]
    path = chatstore.get_attachment(aid)["path"]
    assert os.path.exists(path)

    convo = client.post("/api/conversations", headers=auth_headers(),
                        json={"title": "t", "task": "general"}).json()
    # Bind the attachment to the turn exactly as api_chat records it.
    chatstore.add_message(convo["id"], "user", "look at this", None,
                          attachments=[{"id": aid, "filename": "secret.txt",
                                        "kind": "text", "chars": 12,
                                        "truncated": False}])

    r = client.delete(f"/api/conversations/{convo['id']}", headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["attachments_removed"] == 1

    assert not os.path.exists(path), "the uploaded file is still on disk"
    assert chatstore.get_attachment(aid) is None, \
        "the extracted text is still in the database"


def test_deleting_a_missing_conversation_reports_not_deleted(client):
    r = client.delete("/api/conversations/does-not-exist", headers=auth_headers())
    assert r.json()["ok"] is False


def test_video_source_image_outside_uploads_is_refused(client):
    """The path comes from one of our own rows, but a row is not a guarantee."""
    from webui import chatstore
    row = chatstore.add_attachment(None, "passwd", "/etc/passwd", "text", 0,
                                   False, "")
    r = client.post("/api/video/generate", headers=auth_headers(),
                    json={"model_id": "svd-xt", "source_image_id": row["id"]})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid source image"


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------
def test_healthz_needs_no_token(client):
    """A container healthcheck cannot hold a secret. /healthz is outside /api
    for that reason, and the compose healthchecks depend on it staying there."""
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_healthz_leaks_nothing(client):
    """Unauthenticated means it must stay boring: a version string, no hardware
    report, no model list, no filesystem paths."""
    body = client.get("/healthz").json()
    assert set(body) == {"status", "version"}


def test_healthz_still_obeys_the_host_allow_list(client):
    """Being tokenless does not make it a hole in the rebinding defence."""
    r = client.get("/healthz", headers={"Host": "evil.example"})
    assert r.status_code == 421
