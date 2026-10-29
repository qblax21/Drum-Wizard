import json
import threading
import time
import urllib.request
import uuid

import numpy as np
import soundfile as sf
import uvicorn

from drumwizard.server import app
from test_pipeline import _hit, SR

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"


def _get(path):
    return urllib.request.urlopen(BASE + path).read()


def test_job_flow(tmp_path):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(.05)
    try:
        y = np.zeros(SR * 2, dtype=np.float32)
        for i, k in enumerate(["kick", "hat", "snare", "hat"]):
            h = _hit(k); s = int(i * .5 * SR); y[s:s + len(h)] += h
        p = tmp_path / "d.wav"; sf.write(p, y, SR)

        b = uuid.uuid4().hex
        body = (f'--{b}\r\nContent-Disposition: form-data; name="is_stem"\r\n\r\ntrue\r\n'
                f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="d.wav"\r\n'
                f'Content-Type: audio/wav\r\n\r\n').encode() + p.read_bytes() + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(BASE + "/api/jobs", body,
                                     {"Content-Type": f"multipart/form-data; boundary={b}"})
        jid = json.load(urllib.request.urlopen(req))["id"]
        for _ in range(60):
            j = json.loads(_get(f"/api/jobs/{jid}"))
            if j["status"] in ("done", "error"):
                break
            time.sleep(.5)
        assert j["status"] == "done", j
        assert _get(f"/api/jobs/{jid}/samples/kick.wav")[:4] == b"RIFF"
        req = urllib.request.Request(f"{BASE}/api/jobs/{jid}/midi",
                                     json.dumps({"notes": {"kick": 60}}).encode(),
                                     {"Content-Type": "application/json"})
        assert urllib.request.urlopen(req).read()[:4] == b"MThd"
        assert b"Drum Wizard" in _get("/")
    finally:
        srv.should_exit = True


def test_rebuild_samples_after_reassign(tmp_path):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT + 1, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(.05)
    base = f"http://127.0.0.1:{PORT + 1}"
    try:
        y = np.zeros(SR * 2, dtype=np.float32)
        for i, k in enumerate(["kick", "hat", "snare", "hat"]):
            h = _hit(k); s = int(i * .5 * SR); y[s:s + len(h)] += h
        p = tmp_path / "d.wav"; sf.write(p, y, SR)
        b = uuid.uuid4().hex
        body = (f'--{b}\r\nContent-Disposition: form-data; name="is_stem"\r\n\r\ntrue\r\n'
                f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="d.wav"\r\n'
                f'Content-Type: audio/wav\r\n\r\n').encode() + p.read_bytes() + f"\r\n--{b}--\r\n".encode()
        jid = json.load(urllib.request.urlopen(urllib.request.Request(
            base + "/api/jobs", body, {"Content-Type": f"multipart/form-data; boundary={b}"})))["id"]
        for _ in range(60):
            j = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid}").read())
            if j["status"] != "processing" and j["status"] != "queued":
                break
            time.sleep(.5)
        hits = j["hits"]
        for h in hits:  # relabel every hit as a tom
            h["piece"] = "tom_low"
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f"{base}/api/jobs/{jid}/samples", json.dumps({"hits": hits}).encode(),
            {"Content-Type": "application/json"})))
        assert r["pieces"] == ["tom_low"]
        assert urllib.request.urlopen(f"{base}/api/jobs/{jid}/samples/tom_low.wav").read()[:4] == b"RIFF"
    finally:
        srv.should_exit = True
