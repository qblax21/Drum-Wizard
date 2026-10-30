# Drum Wizard: project context

Goal: a local desktop-style app (like "QT Stems") that takes a song, isolates the drum stem, splits that into
individual drum pieces, and maps each to MIDI so it can drive an e-drum kit. Output is both a `.mid` file
(General MIDI drum notes, channel 10) and one-shot samples triggerable from a MIDI kit. Eventually may be sold,
so keep the licensing and copyright notes in README in mind (not legal advice; check Demucs weights and any
future model licenses).

## Layout
- `drumwizard/stems.py`: Demucs `htdemucs` drum isolation (needs `[separation]` extra). **Never run on real audio yet.**
- `drumwizard/pieces.py`: onset detection + rule-based hit classification (`classify_hit`, the seam to swap for a learned model); `assign_cymbal_variants` clusters cymbal hits into 1-3 distinct cymbals.
- `drumwizard/export.py`: MIDI writer (`note_for` resolves per-cymbal notes) and per-piece one-shot sample slicing.
- `drumwizard/pipeline.py`: `run_pipeline()` used by both CLI and server.
- `drumwizard/server.py` + `static/index.html`: FastAPI backend and single-page vanilla-JS UI (`drumwizard-ui` -> http://127.0.0.1:8765).
- `drumwizard/live.py`: python-rtmidi live trigger, **untested without hardware**. The UI also does kit triggering via browser Web MIDI (Chrome/Edge).
- `tests/`: synthetic-audio tests (pytest). Run `pytest -q`.

## Setup on a new machine
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e '.[server,separation,live,dev]'
# NVIDIA GPU: install CUDA PyTorch from pytorch.org first for much faster Demucs
pytest -q
drumwizard-ui
```

## Done
Pipeline, MIDI export, samples, web UI (timeline, playback, audition, per-piece MIDI note map, MIDI download,
Web MIDI kit triggering), hit select/reassign/delete (keys 1-9, Del, arrows), cymbal split toggle (collapsed by default).

## Known problems / next steps (highest value first)
1. **Test on real songs/drum stems.** Everything so far was tuned on synthetic audio. Classifier thresholds are guesses.
2. Cymbal tail suppression (`_suppress_cymbal_tails`) and a >6 kHz hat/cymbal layer (`_add_top_layer`) are in but only count-checked on the Dragon stem (kick 2028, snare 516, tom_high 463, cymbal 694, hats 653); nobody has listened to the result yet. Kick and tom counts look high; thresholds are still guesses.
3. Pieces are not truly separated: overlapping sounds (cymbal ringing over a kick) confuse classification and samples carry bleed. A learned per-piece model (DrumSep-style) behind `classify_hit` is the real fix; check its license for commercial use.
4. UI can't add a missed hit or change a hit's timing.
5. Package as a desktop app (Tauri/Electron + bundled Python) once quality is acceptable.
6. Test `live.py` / Web MIDI with a real e-drum kit.

## Conventions
Match the existing terse style. Prefer small, tested changes; run `pytest -q` before committing. Don't add dependencies
casually: a package name that comes only from an error message should be verified before installing.
