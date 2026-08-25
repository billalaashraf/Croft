"""
Downloader resume, retry and integrity behaviour — with no network.

`requests.get` is replaced by a fake that returns scripted responses, so these
exercise the branches that only show up on a flaky connection: a partial
`.part` file, a server that honours `Range` (206), one that ignores it (200),
one that says the range is unsatisfiable (416), and a run that never succeeds.

Those branches are exactly where a download corrupts a file quietly rather than
failing loudly, which is why they are worth pinning down.
"""
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import downloader  # noqa: E402

PAYLOAD = b"".join(bytes([i % 251]) for i in range(4096))
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    """The slice of `requests.Response` that http_download actually uses."""

    def __init__(self, status_code, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk):
        for i in range(0, len(self._body), chunk):
            yield self._body[i:i + chunk]


class FakeRequests:
    """Serves a scripted sequence of responses and records what was asked."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, stream=None, timeout=None):
        self.calls.append(dict(headers or {}))
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Retries back off up to 30s; the tests should not actually wait."""
    monkeypatch.setattr(downloader.time, "sleep", lambda _s: None)


def full_response():
    return FakeResponse(200, PAYLOAD, {"Content-Length": str(len(PAYLOAD))})


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def test_download_verifies_a_matching_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "requests", FakeRequests([full_response()]))
    dest = str(tmp_path / "model.gguf")
    assert downloader.http_download(dest=dest, url="https://x/model.gguf",
                                    expected_sha256=DIGEST) == dest
    assert open(dest, "rb").read() == PAYLOAD


def test_a_hash_mismatch_raises_and_keeps_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "requests", FakeRequests([full_response()]))
    dest = str(tmp_path / "model.gguf")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        downloader.http_download("https://x/model.gguf", dest,
                                 expected_sha256="00" * 32)
    # Kept deliberately: you cannot inspect a file that was deleted on failure.
    assert os.path.exists(dest)


def test_an_already_complete_file_is_not_downloaded_again(tmp_path, monkeypatch):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(PAYLOAD)
    fake = FakeRequests([])          # any request at all would IndexError
    monkeypatch.setattr(downloader, "requests", fake)
    downloader.http_download("https://x/model.gguf", str(dest),
                             expected_sha256=DIGEST)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def test_a_partial_file_resumes_with_a_range_header(tmp_path, monkeypatch):
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(PAYLOAD[:1000])
    fake = FakeRequests([FakeResponse(
        206, PAYLOAD[1000:], {"Content-Length": str(len(PAYLOAD) - 1000)})])
    monkeypatch.setattr(downloader, "requests", fake)

    downloader.http_download("https://x/model.gguf", str(dest),
                             expected_sha256=DIGEST)
    assert fake.calls[0]["Range"] == "bytes=1000-"
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "model.gguf.part").exists()


def test_a_server_ignoring_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    """A 200 answer to a Range request is the whole file. Appending it to the
    partial would double the head of the file and corrupt it silently."""
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(PAYLOAD[:1000])
    monkeypatch.setattr(downloader, "requests", FakeRequests([full_response()]))

    downloader.http_download("https://x/model.gguf", str(dest),
                             expected_sha256=DIGEST)
    assert dest.read_bytes() == PAYLOAD


def test_416_treats_the_partial_as_complete(tmp_path, monkeypatch):
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(PAYLOAD)
    monkeypatch.setattr(downloader, "requests", FakeRequests([FakeResponse(416)]))

    downloader.http_download("https://x/model.gguf", str(dest),
                             expected_sha256=DIGEST)
    assert dest.read_bytes() == PAYLOAD


def test_416_with_nothing_to_resume_is_an_error(tmp_path, monkeypatch):
    """Without a partial file there was no Range to reject, so a 416 means the
    server is confused — not that we already hold the file."""
    monkeypatch.setattr(downloader, "requests",
                        FakeRequests([FakeResponse(416)] * 5))
    with pytest.raises(RuntimeError):
        downloader.http_download("https://x/model.gguf",
                                 str(tmp_path / "model.gguf"), max_retries=2)


def test_a_stale_range_header_is_not_reused_after_a_retry(tmp_path, monkeypatch):
    """Regression: the headers dict was built once and mutated, so a Range from
    an attempt that had a partial file survived into a later attempt that did
    not — and the server answered with bytes from the middle of the file."""
    dest = tmp_path / "model.gguf"
    part = tmp_path / "model.gguf.part"
    part.write_bytes(PAYLOAD[:1000])

    class VanishingPart(FakeRequests):
        def get(self, *a, **kw):
            # Simulate the partial being cleaned up between attempts.
            if part.exists():
                part.unlink()
            return super().get(*a, **kw)

    fake = VanishingPart([ConnectionError("dropped"), full_response()])
    monkeypatch.setattr(downloader, "requests", fake)

    downloader.http_download("https://x/model.gguf", str(dest),
                             expected_sha256=DIGEST)
    assert "Range" in fake.calls[0], "first attempt should have resumed"
    assert "Range" not in fake.calls[1], "second attempt had nothing to resume"
    assert dest.read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Retry and failure
# ---------------------------------------------------------------------------
def test_it_retries_and_then_succeeds(tmp_path, monkeypatch):
    fake = FakeRequests([ConnectionError("reset"), ConnectionError("reset"),
                         full_response()])
    monkeypatch.setattr(downloader, "requests", fake)
    dest = str(tmp_path / "model.gguf")
    downloader.http_download("https://x/model.gguf", dest, expected_sha256=DIGEST)
    assert len(fake.calls) == 3


def test_exhausting_retries_raises_rather_than_returning_a_missing_path(
        tmp_path, monkeypatch):
    """Regression: the loop could exhaust its retries and fall through to the
    verification step. With no expected hash to check, `verify_sha256` returned
    True and the caller got back the path of a file that was never written."""
    monkeypatch.setattr(downloader, "requests",
                        FakeRequests([ConnectionError("down")] * 3))
    dest = str(tmp_path / "model.gguf")
    with pytest.raises(RuntimeError, match="Download failed after"):
        downloader.http_download("https://x/model.gguf", dest, max_retries=3)
    assert not os.path.exists(dest)


def test_dry_run_touches_neither_network_nor_disk(tmp_path, monkeypatch):
    fake = FakeRequests([])
    monkeypatch.setattr(downloader, "requests", fake)
    dest = str(tmp_path / "model.gguf")
    assert downloader.http_download("https://x/model.gguf", dest,
                                    dry_run=True) == dest
    assert not os.path.exists(dest)
    assert fake.calls == []
