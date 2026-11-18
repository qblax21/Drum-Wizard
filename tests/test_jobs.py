import asyncio
import json
import time
import urllib.error
import urllib.request

import pytest

from drumwizard import server
from drumwizard.jobs import INTERRUPTED, cleanup, load_jobs, save_job
from test_server import PORT, _make_job, _post, _serve

DAY = 86400


def _job(tmp, jid, status="done", age_days=0.0, **extra):
    save_job(tmp, jid, {"status": status, "name": jid, "created": time.time() - age_days * DAY, **extra})
    (tmp / jid / "input.wav").write_bytes(b"x")


def test_jobs_survive_a_restart(tmp_path):
    with _serve(PORT + 6) as base:
        jid, j = _make_job(base, tmp_path)
    assert (server.WORK / jid / "job.json").exists()
    again = load_jobs(server.WORK)            # what a freshly started server would read
    assert again[jid]["status"] == "done" and again[jid]["hits"] == j["hits"] and again[jid]["bpm"] == j["bpm"]


def test_job_cut_off_by_a_restart_is_marked_failed(tmp_path):
    _job(tmp_path, "a", "processing"); _job(tmp_path, "b", "queued"); _job(tmp_path, "c", "done")
    (tmp_path / "d").mkdir(); (tmp_path / "d" / "job.json").write_text("{not json")
    jobs = load_jobs(tmp_path)
    assert sorted(jobs) == ["a", "b", "c"]
    assert jobs["a"]["status"] == jobs["b"]["status"] == "error" and jobs["a"]["error"] == INTERRUPTED
    assert json.loads((tmp_path / "a" / "job.json").read_text())["status"] == "error"   # and the file says so too


def test_cleanup_deletes_only_jobs_older_than_seven_days(tmp_path):
    _job(tmp_path, "old", age_days=8); _job(tmp_path, "new", age_days=6.9)
    _job(tmp_path, "running_old", "processing", age_days=9)
    jobs = {"old": {}, "new": {}, "running_old": {"status": "processing"}}
    assert cleanup(tmp_path, jobs, days=7) == ["old"]
    assert not (tmp_path / "old").exists() and (tmp_path / "new").exists() and (tmp_path / "running_old").exists()
    assert "old" not in jobs and "new" in jobs


def test_cleanup_reads_age_from_disk_and_expires_folders_without_job_json(tmp_path):
    _job(tmp_path, "ondisk", age_days=10)                     # not in memory: age comes from job.json
    (tmp_path / "bare").mkdir(); old = time.time() - 10 * DAY
    import os; os.utime(tmp_path / "bare", (old, old))        # no job.json: folder mtime
    assert sorted(cleanup(tmp_path, {}, days=7)) == ["bare", "ondisk"]


def test_keep_days_setting(tmp_path, monkeypatch):
    _job(tmp_path, "j", age_days=2)
    assert cleanup(tmp_path, {}) == []                        # default 7 days
    monkeypatch.setenv("DRUMWIZARD_KEEP_DAYS", "1")
    assert cleanup(tmp_path, {}) == ["j"]


def test_upload_over_the_cap_is_rejected_with_413(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUMWIZARD_MAX_UPLOAD_MB", "0.01")    # ~10 KB
    with _serve(PORT + 7) as base:
        before = set(p.name for p in server.WORK.iterdir())
        body = b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"d.wav\"\r\n\r\n" + b"x" * 50_000 + b"\r\n--b--\r\n"
        req = urllib.request.Request(base + "/api/jobs", body, {"Content-Type": "multipart/form-data; boundary=b"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 413 and b"limit" in e.value.read()
        assert set(p.name for p in server.WORK.iterdir()) == before          # nothing was stored
        assert json.loads(urllib.request.urlopen(base + "/api/features").read())["max_upload_mb"] == 0


def test_chunked_upload_without_content_length_is_cut_off(monkeypatch):
    monkeypatch.setenv("DRUMWIZARD_MAX_UPLOAD_MB", "0.01")
    read = []

    async def inner(scope, receive, send):
        while True:
            msg = await receive()
            read.append(len(msg.get("body", b"")))
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    chunks = iter([{"type": "http.request", "body": b"x" * 4096, "more_body": True} for _ in range(100)])

    async def receive():
        return next(chunks)

    sent = []

    async def send(m):
        sent.append(m)

    scope = {"type": "http", "method": "POST", "path": "/api/jobs", "headers": []}
    asyncio.run(server.UploadLimit(inner)(scope, receive, send))
    assert sent[0]["status"] == 413 and sum(read) < 20_000       # stopped reading right after the cap


def test_upload_under_the_cap_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUMWIZARD_MAX_UPLOAD_MB", "5")
    with _serve(PORT + 8) as base:
        jid, j = _make_job(base, tmp_path)
        assert j["status"] == "done"
