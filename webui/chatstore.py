# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
webui/chatstore.py — SQLite persistence for the chat UI.

Stores conversations, their messages, and a small key/value settings table
(used for per-model inference endpoints). Uses only the stdlib `sqlite3`, so it
adds no dependency. A fresh connection is opened per call — simple and safe
under FastAPI's threadpool; the write volume here is tiny.

DB location: $LLM_CHAT_DB, else <LLM_MODELS_DIR|models>/chat.db.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

_DB_PATH = os.environ.get("LLM_CHAT_DB") or os.path.join(
    os.environ.get("LLM_MODELS_DIR", "models"), "chat.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    model_id      TEXT,
    task          TEXT,
    system_prompt TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,          -- user | assistant | system
    content         TEXT NOT NULL,
    model_id        TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id              TEXT PRIMARY KEY,
    model_id        TEXT,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT,
    params          TEXT,               -- JSON: steps, width, height, guidance, seed
    path            TEXT NOT NULL,       -- PNG on disk
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at DESC);
CREATE TABLE IF NOT EXISTS videos (
    id           TEXT PRIMARY KEY,
    model_id     TEXT,
    prompt       TEXT,
    params       TEXT,                  -- JSON: frames, fps, steps, source
    path         TEXT NOT NULL,          -- mp4/gif on disk
    fmt          TEXT,                   -- mp4 | gif
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_videos_created ON videos(created_at DESC);
CREATE TABLE IF NOT EXISTS attachments (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    filename        TEXT NOT NULL,
    path            TEXT,                -- raw file on disk
    kind            TEXT,                -- text | pdf | docx | image | binary | error
    chars           INTEGER DEFAULT 0,
    truncated       INTEGER DEFAULT 0,
    text            TEXT,                -- extracted content fed to the model
    created_at      REAL NOT NULL
);
"""

# Columns added after v1; applied idempotently in init_db().
_MIGRATIONS = [
    ("messages", "attachments", "ALTER TABLE messages ADD COLUMN attachments TEXT"),
]


def _now() -> float:
    return time.time()


def _restrict(path: str, mode: int) -> None:
    """Best-effort chmod. Never fatal — a Windows or FAT mount has no bits."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _connect() -> sqlite3.Connection:
    parent = os.path.dirname(os.path.abspath(_DB_PATH)) or "."
    os.makedirs(parent, exist_ok=True)
    fresh = not os.path.exists(_DB_PATH)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if fresh:
        # Every message, every uploaded document's extracted text and every
        # configured endpoint lives in this file. sqlite3 creates it 0644, so
        # on a shared machine any other account could simply read the lot.
        # Tighten it once, at creation, rather than on every connect.
        _restrict(_DB_PATH, 0o600)
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        for table, column, ddl in _MIGRATIONS:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            if column not in cols:
                conn.execute(ddl)


def db_path() -> str:
    return _DB_PATH


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
def create_conversation(title: str, model_id: Optional[str],
                        task: Optional[str], system_prompt: Optional[str]) -> Dict[str, Any]:
    cid = uuid.uuid4().hex
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id,title,model_id,task,system_prompt,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (cid, title or "New chat", model_id, task, system_prompt, ts, ts))
    return get_conversation(cid)


def list_conversations() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT c.*, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count "
            "FROM conversations c ORDER BY c.updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    return dict(row) if row else None


def update_conversation(cid: str, **fields: Any) -> Optional[Dict[str, Any]]:
    allowed = {"title", "model_id", "task", "system_prompt"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return get_conversation(cid)
    cols = ", ".join(f"{k}=?" for k in sets)
    with _connect() as conn:
        conn.execute(f"UPDATE conversations SET {cols}, updated_at=? WHERE id=?",
                     (*sets.values(), _now(), cid))
    return get_conversation(cid)


def touch_conversation(cid: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), cid))


def conversation_attachment_ids(cid: str) -> List[str]:
    """Every attachment id this conversation refers to.

    Two places to look, because /api/attach uploads a file before the turn it
    belongs to exists: the row's own `conversation_id` (often NULL for exactly
    that reason) and the JSON blob recorded on each user message. Reading only
    the column would have missed nearly all of them.
    """
    ids: List[str] = []
    seen = set()
    with _connect() as conn:
        for (blob,) in conn.execute(
                "SELECT attachments FROM messages WHERE conversation_id=? "
                "AND attachments IS NOT NULL", (cid,)):
            try:
                for a in json.loads(blob) or []:
                    aid = a.get("id")
                    if aid and aid not in seen:
                        seen.add(aid)
                        ids.append(aid)
            except (ValueError, AttributeError):
                continue                      # a malformed blob is not fatal
        for (aid,) in conn.execute(
                "SELECT id FROM attachments WHERE conversation_id=?", (cid,)):
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
    return ids


def delete_conversation(cid: str) -> Tuple[bool, List[str]]:
    """Delete a conversation and everything hanging off it.

    Returns (deleted, paths) — `paths` being the on-disk attachments removed,
    for the caller to unlink. A tuple rather than just the list because an
    empty list is ambiguous: a conversation with no attachments and one that
    never existed would look identical.

    Deleting a chat used to leave the uploaded documents on disk *and* their
    extracted text, in full, in the database. That makes "delete" a promise the
    app was not keeping.
    """
    aids = conversation_attachment_ids(cid)
    paths: List[str] = []
    with _connect() as conn:
        for aid in aids:
            row = conn.execute(
                "SELECT path FROM attachments WHERE id=?", (aid,)).fetchone()
            if row and row["path"]:
                paths.append(row["path"])
        cur = conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        for aid in aids:
            conn.execute("DELETE FROM attachments WHERE id=?", (aid,))
        deleted = cur.rowcount > 0
    return deleted, (paths if deleted else [])


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def add_message(conversation_id: str, role: str, content: str,
                model_id: Optional[str] = None,
                attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    mid = uuid.uuid4().hex
    ts = _now()
    att_json = json.dumps(attachments) if attachments else None
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id,conversation_id,role,content,model_id,created_at,attachments)"
            " VALUES (?,?,?,?,?,?,?)",
            (mid, conversation_id, role, content, model_id, ts, att_json))
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, conversation_id))
    return {"id": mid, "conversation_id": conversation_id, "role": role,
            "content": content, "model_id": model_id, "created_at": ts,
            "attachments": attachments or []}


def list_messages(conversation_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
            (conversation_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["attachments"] = json.loads(d["attachments"]) if d.get("attachments") else []
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Attachments (uploaded files, with extracted text for the model)
# ---------------------------------------------------------------------------
def add_attachment(conversation_id: Optional[str], filename: str, path: Optional[str],
                   kind: str, chars: int, truncated: bool, text: str) -> Dict[str, Any]:
    aid = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO attachments (id,conversation_id,filename,path,kind,chars,truncated,text,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, conversation_id, filename, path, kind, chars, 1 if truncated else 0, text, _now()))
    return {"id": aid, "filename": filename, "kind": kind, "chars": chars,
            "truncated": bool(truncated)}


def get_attachment(aid: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM attachments WHERE id=?", (aid,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Settings (key/value, JSON-encoded values)
# ---------------------------------------------------------------------------
def get_setting(key: str, default: Any = None) -> Any:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_setting(key: str, value: Any) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings (key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))


def all_settings() -> Dict[str, Any]:
    with _connect() as conn:
        rows = conn.execute("SELECT key,value FROM settings").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


# ---------------------------------------------------------------------------
# Images (generation history)
# ---------------------------------------------------------------------------
def add_image(model_id: Optional[str], prompt: str, negative_prompt: Optional[str],
              params: Dict[str, Any], path: str) -> Dict[str, Any]:
    iid = uuid.uuid4().hex
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO images (id,model_id,prompt,negative_prompt,params,path,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (iid, model_id, prompt, negative_prompt, json.dumps(params), path, ts))
    return get_image(iid)


def list_images(limit: int = 100) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM images ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d["params"]) if d.get("params") else {}
        out.append(d)
    return out


def get_image(iid: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["params"] = json.loads(d["params"]) if d.get("params") else {}
    return d


def delete_image(iid: str) -> Optional[str]:
    """Delete the DB row; return the file path so the caller can unlink it."""
    img = get_image(iid)
    if not img:
        return None
    with _connect() as conn:
        conn.execute("DELETE FROM images WHERE id=?", (iid,))
    return img.get("path")


# ---------------------------------------------------------------------------
# Videos (generation history)
# ---------------------------------------------------------------------------
def add_video(model_id: Optional[str], prompt: Optional[str], params: Dict[str, Any],
              path: str, fmt: str) -> Dict[str, Any]:
    vid = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO videos (id,model_id,prompt,params,path,fmt,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (vid, model_id, prompt, json.dumps(params), path, fmt, _now()))
    return get_video(vid)


def list_videos(limit: int = 100) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM videos ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d["params"]) if d.get("params") else {}
        out.append(d)
    return out


def get_video(vid: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["params"] = json.loads(d["params"]) if d.get("params") else {}
    return d


def delete_video(vid: str) -> Optional[str]:
    v = get_video(vid)
    if not v:
        return None
    with _connect() as conn:
        conn.execute("DELETE FROM videos WHERE id=?", (vid,))
    return v.get("path")
