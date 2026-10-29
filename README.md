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
```

## Status / next steps
- Done: pipeline skeleton, MIDI export, sample slicing, synthetic-audio test.
- Piece classification is a rule-based baseline; thresholds need tuning on real recordings.
  The next upgrade is a learned model (e.g. a DrumSep-style per-piece separator) behind `classify_hit`.
- Not built yet: desktop UI (planned: local FastAPI backend + Tauri/web frontend), note-map editor, `live.py` untested without hardware.
