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
        for h in hits:  # relabel every hit as the lowest tom
            h["piece"], h["variant"] = "tom", 0
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f"{base}/api/jobs/{jid}/samples", json.dumps({"hits": hits}).encode(),
            {"Content-Type": "application/json"})))
        assert r["pieces"] == ["tom_0"]
        assert urllib.request.urlopen(f"{base}/api/jobs/{jid}/samples/tom_0.wav").read()[:4] == b"RIFF"
    finally:
        srv.should_exit = True


# --- hit editing: add / move / validate -------------------------------------------------------------------------

import contextlib
import io
import urllib.error

import mido


@contextlib.contextmanager
def _serve(port):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True


def _post(url, obj):
    return urllib.request.urlopen(urllib.request.Request(url, json.dumps(obj).encode(),
                                                         {"Content-Type": "application/json"}))


def _make_job(base, tmp_path):
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
        if j["status"] in ("done", "error"):
            break
        time.sleep(.5)
    assert j["status"] == "done", j
    return jid, j


def _midi_times(data, bpm):
    """[(seconds, note)] of the note_ons in a MIDI file written at `bpm`."""
    mid, t, out = mido.MidiFile(file=io.BytesIO(data)), 0, []
    for m in mid.tracks[0]:
        t += m.time
        if m.type == "note_on":
            out.append((t / 480 * 60 / bpm, m.note))
    return out


def test_added_hit_gets_a_sample_and_a_midi_note(tmp_path):
    with _serve(PORT + 2) as base:
        jid, j = _make_job(base, tmp_path)
        hits = [dict(h) for h in j["hits"]]
        hits.append({"time": 1.25, "piece": "tom", "velocity": 90, "variant": 3})   # no start/end: a new hit
        r = json.load(_post(f"{base}/api/jobs/{jid}/samples", {"hits": hits}))
        assert "tom_3" in r["pieces"]
        new = r["hits"][-1]
        assert new["start"] == round(1.25 * SR) and new["end"] > new["start"]
        old = j["hits"]
        assert [(h["time"], h["piece"], h["start"]) for h in r["hits"][:-1]] == [(h["time"], h["piece"], h["start"]) for h in old]
        for before, after in zip(old, r["hits"]):   # a slice only changes by being cut where the new hit begins
            assert after["end"] == before["end"] or after["end"] == new["start"]
        assert urllib.request.urlopen(f"{base}/api/jobs/{jid}/samples/tom_3.wav").read()[:4] == b"RIFF"
        data = _post(f"{base}/api/jobs/{jid}/midi", {"notes": {}, "hits": hits}).read()
        got = _midi_times(data, j["bpm"])
        assert any(abs(t - 1.25) < 0.01 and n == 47 for t, n in got)   # tom_3 -> GM 47


def test_moved_hit_is_resliced_and_exported_at_the_new_time(tmp_path):
    with _serve(PORT + 3) as base:
        jid, j = _make_job(base, tmp_path)
        hits = [dict(h) for h in j["hits"]]
        k = next(i for i, h in enumerate(hits) if h["piece"] == "snare")
        old = hits[k]["time"]
        hits[k]["time"] = old + 0.045   # nine 5 ms steps
        r = json.load(_post(f"{base}/api/jobs/{jid}/samples", {"hits": hits}))
        moved = r["hits"][k]
        assert moved["start"] == round((old + 0.045) * SR) and moved["end"] > moved["start"]
        assert "snare" in r["pieces"]
        got = _midi_times(_post(f"{base}/api/jobs/{jid}/midi", {"notes": {}, "hits": hits}).read(), j["bpm"])
        assert any(abs(t - (old + 0.045)) < 0.01 and n == 38 for t, n in got)
        assert not any(abs(t - old) < 0.01 and n == 38 for t, n in got)


def test_bad_edits_are_rejected_or_clamped(tmp_path):
    with _serve(PORT + 4) as base:
        jid, j = _make_job(base, tmp_path)
        for url in (f"{base}/api/jobs/{jid}/samples", f"{base}/api/jobs/{jid}/midi"):
            body = {"hits": [{"time": 0.5, "piece": "cowbell", "velocity": 90}]}
            with __import__("pytest").raises(urllib.error.HTTPError) as e:
                _post(url, {**body, "notes": {}})
            assert e.value.code == 422
        hits = [{"time": 99.0, "piece": "kick", "velocity": 500}]   # past the end, velocity out of range
        r = json.load(_post(f"{base}/api/jobs/{jid}/samples", {"hits": hits}))
        assert r["hits"][0]["time"] == j["duration"] and r["hits"][0]["velocity"] == 127
