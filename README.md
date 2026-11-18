# Drum Wizard

Song -> drum stem -> individual drum pieces -> MIDI / triggerable samples.

## Pipeline
1. **Drum isolation** (`stems.py`): Demucs `htdemucs` pulls the drum stem from a full mix.
2. **Piece detection** (`pieces.py`): onset detection, then each hit is classified as kick, snare, toms, closed/open hi-hat or cymbal.
3. **Export** (`export.py`): a General-MIDI drum `.mid` file (channel 10), plus one-shot WAV samples per piece.
4. **Live** (`live.py`): play the samples when your e-drum kit sends MIDI notes.

## Use
```
pip install -e '.[separation,live]'
drumwizard song.mp3 -o out          # full pipeline
drumwizard drums.wav --stem -o out  # skip Demucs if you already have a drum stem
drumwizard-ui                        # web UI (pip install -e '.[server]')
```

## Web UI

`drumwizard-ui` serves http://127.0.0.1:8765 (this PC only). `--host 0.0.0.0` or a Tailscale address makes it reachable from other devices; `--port` changes the port.

**Editing hits** (select a hit by clicking it; edits re-cut the samples and are used by the MIDI download):
- *Add*: double-click a lane on the timeline (or press "+ Hit at playhead"), choose the piece, "Add hit". It gets the average velocity of that piece.
- *Move*: drag a hit with the mouse (snaps to 5 ms), or Shift+Left/Right or the "5 ms" buttons (5 ms per step). The sample is re-cut from the new position.
- *Reassign / delete*: keys 1-9 / Del, or the "Change to" menu. Left/Right select the previous/next hit, Esc deselects.

**Jobs are kept on disk** under `%LOCALAPPDATA%\DrumWizard\jobs` (`~/.local/share/DrumWizard/jobs` elsewhere), so an uploaded song and its result survive a restart. A job that was still running when the server stopped is shown as failed; upload it again. Jobs older than 7 days are deleted when the server starts, on each new upload, and hourly. Settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `DRUMWIZARD_DATA` | see above | where jobs are stored |
| `DRUMWIZARD_KEEP_DAYS` | `7` | delete jobs older than this |
| `DRUMWIZARD_MAX_UPLOAD_MB` | `300` | largest accepted upload (bigger gets HTTP 413) |

Edits to a job's hits live in the browser until you download the MIDI file; reloading the page starts again from the detected hits.

**Start at Windows login** (Task Scheduler, no admin needed). From the project folder, after `pip install -e '.[server]'` in `.venv`:
```
powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1                          # this PC only
powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1 -HostName 100.70.135.52   # also over Tailscale (use your own address)
Start-ScheduledTask -TaskName DrumWizardUI                                                       # start now, without logging out
```
The server runs hidden as task `DrumWizardUI`, restarts 30 s after it stops (e.g. the Tailscale address was not up yet), and logs to `%LOCALAPPDATA%\DrumWizard\server.log`. The default host stays `127.0.0.1`; only pass another one if you want other devices to reach it. Remove with `scripts\uninstall-autostart.ps1`. If you move the project folder, run the install script again.

## Tests

`pytest -q` runs everything. Tests marked `slow` (real Demucs / DrumSep models, downloads) are skipped unless you run `pytest -m slow`; CI runs `pytest -q -m "not slow"` (see `.github/workflows/ci.yml`).

## Status / next steps
- Done: pipeline skeleton, MIDI export, sample slicing, synthetic-audio test.
- Piece classification is a rule-based baseline; thresholds need tuning on real recordings.
  The next upgrade is a learned model (e.g. a DrumSep-style per-piece separator) behind `classify_hit`.
- Done: local web UI (`drumwizard-ui`, then open http://127.0.0.1:8765): upload, hit timeline with playback, select a hit to reassign it to another piece or delete it (keys 1-9, Del), add a missed hit, move a hit in 5 ms steps, audition samples, per-piece MIDI note map, MIDI download, and live kit triggering through browser Web MIDI (Chrome/Edge).
- Cymbals: stay one "Cymbal / crash" piece by default. If 2-3 clearly different cymbals are detected (by brightness and decay), the UI offers a "Split cymbals" toggle giving each its own lane, sample and MIDI note (49, 57, 51). 51 is also the ride's note, so when a ride exists (DrumSep finds one) the split cymbals are crashes only: 49, 57 and 52 (china; 55 is the next free crash note).
- Not built yet: a packaged desktop shell (Tauri/Electron), `live.py` and Web MIDI untested without hardware.
