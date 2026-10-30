"""Outputs: MIDI file of hits, and per-piece one-shot samples for triggering."""
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from .gm import GM_NOTES, TOM_VARIANT_NOTES
from .pieces import Hit


def note_for(hit: Hit, notes: dict) -> int:
    """Per-cymbal note if the caller mapped one (split mode), else the shared piece note."""
    if hit.piece == "cymbal" and f"cymbal_{hit.variant}" in notes:
        return notes[f"cymbal_{hit.variant}"]
    if hit.piece == "tom":
        k = f"tom_{hit.variant}"
        return notes.get(k, TOM_VARIANT_NOTES.get(k, notes["tom"]))
    return notes[hit.piece]


def write_midi(hits: list[Hit], path: str | Path, bpm: float = 120.0, notes: dict | None = None) -> None:
    notes = notes or GM_NOTES
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    events = []
    for h in hits:
        t = int(round(h.time * bpm / 60 * tpb))
        n = note_for(h, notes)
        events += [(t, 1, mido.Message("note_on", channel=9, note=n, velocity=h.velocity)),
                   (t + tpb // 8, 0, mido.Message("note_off", channel=9, note=n, velocity=0))]
    last = 0
    for t, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        tr.append(msg.copy(time=t - last))
        last = t
    mid.save(str(path))


def _sample_key(h: Hit) -> str:
    return f"{h.piece}_{h.variant}" if h.piece in ("cymbal", "tom") else h.piece


def write_samples(y: np.ndarray, sr: int, hits: list[Hit], out_dir: str | Path) -> dict[str, Path]:
    """Save the cleanest (loudest, least-bleeding) hit of each piece as a one-shot WAV.

    Toms get one `tom_<slot>.wav` per pitch. Cymbals get a combined `cymbal.wav` plus one `cymbal_<n>.wav` per detected variant.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def better(h, cur):
        return cur is None or h.velocity > cur.velocity or (
            h.velocity == cur.velocity and h.end - h.start > cur.end - cur.start)

    best: dict[str, Hit] = {}
    for h in hits:
        keys = [_sample_key(h)] if h.piece == "tom" else [h.piece] + ([_sample_key(h)] if h.piece == "cymbal" else [])
        for k in keys:
            if better(h, best.get(k)):
                best[k] = h
    paths = {}
    for key, h in best.items():
        seg = y[h.start:h.end].copy()
        fade = min(len(seg), int(0.01 * sr))
        if fade:
            seg[-fade:] *= np.linspace(1, 0, fade)
        seg /= np.abs(seg).max() + 1e-12
        p = out_dir / f"{key}.wav"
        sf.write(str(p), seg, sr)
        paths[key] = p
    return paths
