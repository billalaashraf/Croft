"""
webui/files.py — save uploaded files and extract analysable text.

Turns an uploaded file into plain text the local model can reason over:
  * text / code / json / csv / markdown / logs  → decoded directly (UTF-8)
  * .pdf   → pypdf, if installed
  * .docx  → python-docx, if installed
  * images → no text (a text model can't see them; noted for the model)
  * other binary → skipped with a note

Extraction is capped (LLM_FILE_MAXCHARS, default 16000) so a big file can't blow
the context window; the cap is reported so the UI/model know it was truncated.
Raw uploads are kept under <LLM_MODELS_DIR>/uploads for reference.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional, Tuple

MODELS_DIR = os.environ.get("LLM_MODELS_DIR", "models")
UPLOAD_DIR = os.path.join(MODELS_DIR, "uploads")
MAX_CHARS = int(os.environ.get("LLM_FILE_MAXCHARS", "16000"))
MAX_BYTES = int(os.environ.get("LLM_FILE_MAXBYTES", str(25 * 1024 * 1024)))  # 25 MB

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".heic"}
# Extensions we always treat as text even if they contain odd bytes.
_TEXT_HINT = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json",
              ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".html",
              ".htm", ".css", ".js", ".ts", ".jsx", ".tsx", ".py", ".rb", ".go",
              ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".sh",
              ".bash", ".zsh", ".sql", ".r", ".swift", ".kt", ".lua", ".pl", ".m"}


def deps() -> dict:
    def has(mod):
        try:
            __import__(mod); return True
        except Exception:
            return False
    return {"pdf": has("pypdf"), "docx": has("docx")}


def save_upload(data: bytes, filename: str) -> str:
    """Write an upload under UPLOAD_DIR and return its path.

    Permissions are set at creation, not afterwards: between an open() and a
    later chmod() the file exists and is world-readable, and these are
    documents the user chose to hand to a *local* model.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        os.chmod(UPLOAD_DIR, 0o700)
    except OSError:
        pass
    safe = os.path.basename(filename or "file")
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{safe}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def is_within_uploads(path: Optional[str]) -> bool:
    """True if `path` really resolves inside UPLOAD_DIR.

    Compared after resolving symlinks, because `uploads/x -> /etc/shadow` and
    `uploads/../../etc/shadow` both pass a plain prefix test on the string.
    Used wherever a client-supplied path reaches the filesystem.
    """
    if not path:
        return False
    try:
        root = os.path.realpath(UPLOAD_DIR)
        target = os.path.realpath(path)
    except OSError:
        return False
    return os.path.commonpath([root, target]) == root and target != root


def _pdf_text(path: str) -> str:
    from pypdf import PdfReader  # type: ignore
    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(path: str) -> str:
    import docx  # type: ignore
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(path: str, filename: str) -> Tuple[str, str, bool]:
    """Return (text, kind, truncated). `text` is '' when nothing analysable."""
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".pdf":
            kind, text = "pdf", _pdf_text(path)
        elif ext == ".docx":
            kind, text = "docx", _docx_text(path)
        elif ext in IMAGE_EXTS:
            return "", "image", False
        else:
            with open(path, "rb") as fh:
                raw = fh.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                if ext in _TEXT_HINT:
                    text = raw.decode("utf-8", errors="replace")
                else:
                    return "", "binary", False   # not text we can trust
            kind = "text"
    except ModuleNotFoundError as exc:
        return f"[cannot read {ext} — missing dependency: {exc.name}]", "error", False
    except Exception as exc:  # pragma: no cover
        return f"[could not extract text: {exc}]", "error", False

    truncated = False
    if len(text) > MAX_CHARS:
        text, truncated = text[:MAX_CHARS], True
    return text, kind, truncated
