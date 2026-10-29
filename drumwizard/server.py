"""Local web UI backend. Run: drumwizard-ui  (serves http://127.0.0.1:8765)"""
import io
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .export import write_midi
from .gm import GM_NOTES
from .pieces import Hit
from .pipeline import run_pipeline

WORK = Path(tempfile.mkdtemp(prefix="drumwizard-"))
JOBS: dict[str, dict] = {}
app = FastAPI(title="Drum Wizard")


def _run(job_id: str, src: Path, is_stem: bool):
    job = JOBS[job_id]
    try:
        job.update(status="processing", **run_pipeline(src, WORK / job_id / "out", is_stem))
        job["status"] = "done"
    except Exception as e:  # surfaced to the UI
        job.update(status="error", error=str(e))


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), is_stem: bool = Form(False)):
    job_id = uuid.uuid4().hex[:10]
    d = WORK / job_id
    d.mkdir(parents=True)
    src = d / ("input" + Path(file.filename or "x.wav").suffix)
    src.write_bytes(await file.read())
    JOBS[job_id] = {"status": "queued", "name": file.filename}
    threading.Thread(target=_run, args=(job_id, src, is_stem), daemon=True).start()
    return {"id": job_id}


def _job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return {**_job(job_id), "default_notes": GM_NOTES}


@app.get("/api/jobs/{job_id}/audio")
def audio(job_id: str):
    _job(job_id)
    return FileResponse(WORK / job_id / "out" / "drums.wav", media_type="audio/wav")


@app.get("/api/jobs/{job_id}/samples/{piece}.wav")
def sample(job_id: str, piece: str):
    _job(job_id)
    p = WORK / job_id / "out" / "samples" / f"{Path(piece).name}.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


class MidiReq(BaseModel):
    notes: dict[str, int]
    hits: list[dict] | None = None  # client may send an edited hit list
    bpm: float | None = None


@app.post("/api/jobs/{job_id}/midi")
def midi(job_id: str, req: MidiReq):
    job = _job(job_id)
    notes = {**GM_NOTES, **{k: max(0, min(127, v)) for k, v in req.notes.items()}}
    hits = [Hit(**h) for h in (req.hits if req.hits is not None else job["hits"])]
    buf = io.BytesIO()
    tmp = WORK / job_id / "tmp.mid"
    write_midi(hits, tmp, req.bpm or job["bpm"], notes)
    buf.write(tmp.read_bytes())
    return Response(buf.getvalue(), media_type="audio/midi",
                    headers={"Content-Disposition": 'attachment; filename="drums.mid"'})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
