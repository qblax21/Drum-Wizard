"""Job storage for the web UI: jobs live in folders under a data directory, survive restarts, and expire.

Layout: <data dir>/<job id>/{input.*, out/, job.json}. job.json holds the job dict the UI polls (status, hits, ...).
Settings (environment): DRUMWIZARD_DATA (data dir), DRUMWIZARD_KEEP_DAYS (default 7), DRUMWIZARD_MAX_UPLOAD_MB (default 300).
"""
import json
import os
import shutil
import time
from pathlib import Path

KEEP_DAYS = 7.0
MAX_UPLOAD_MB = 300.0
INTERRUPTED = "Interrupted by a restart of Drum Wizard. Upload the file again."


def data_dir() -> Path:
    env = os.environ.get("DRUMWIZARD_DATA")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "DrumWizard" / "jobs"


def keep_days() -> float:
    return float(os.environ.get("DRUMWIZARD_KEEP_DAYS", KEEP_DAYS))


def max_upload_bytes() -> int:
    return int(float(os.environ.get("DRUMWIZARD_MAX_UPLOAD_MB", MAX_UPLOAD_MB)) * 1024 * 1024)


def save_job(work: Path, job_id: str, job: dict) -> None:
    """Write job.json atomically so a crash never leaves half a file."""
    d = work / job_id
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job), encoding="utf-8")
    os.replace(tmp, d / "job.json")


def load_jobs(work: Path) -> dict[str, dict]:
    """Read every saved job. A job that was queued or running when the server stopped cannot continue: mark it failed."""
    jobs = {}
    for f in work.glob("*/job.json"):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue   # unreadable: leave the folder for cleanup to expire
        if job.get("status") in ("queued", "processing"):
            job.update(status="error", error=INTERRUPTED)
            save_job(work, f.parent.name, job)
        jobs[f.parent.name] = job
    return jobs


def cleanup(work: Path, jobs: dict[str, dict], days: float | None = None, now: float | None = None) -> list[str]:
    """Delete job folders older than `days` (by creation time; folders without job.json use their mtime) and forget
    them. Jobs still running are kept. Returns the removed folder names."""
    days = keep_days() if days is None else days
    cutoff = (time.time() if now is None else now) - days * 86400
    removed = []
    for d in work.iterdir():
        if not d.is_dir():
            continue
        job = jobs.get(d.name)
        if job and job.get("status") in ("queued", "processing"):
            continue
        created = (job or {}).get("created")
        if created is None:
            try:
                created = json.loads((d / "job.json").read_text(encoding="utf-8")).get("created")
            except (OSError, ValueError):
                created = None
        if created is None:
            created = d.stat().st_mtime
        if created < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            jobs.pop(d.name, None)
            removed.append(d.name)
    return removed
