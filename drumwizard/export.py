"""Outputs: MIDI file of hits, and per-piece one-shot samples for triggering."""
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from .gm import GM_NOTES
from .pieces import Hit


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
        n = notes[h.piece]
        events += [(t, 1, mido.Message("note_on", channel=9, note=n, velocity=h.velocity)),
                   (t + tpb // 8, 0, mido.Message("note_off", channel=9, note=n, velocity=0))]
    last = 0
    for t, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        tr.append(msg.copy(time=t - last))
        last = t
    mid.save(str(path))


def write_samples(y: np.ndarray, sr: int, hits: list[Hit], out_dir: str | Path) -> dict[str, Path]:
    """Save the cleanest (loudest, least-bleeding) hit of each piece as a one-shot WAV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, Hit] = {}
    for h in hits:
        if h.piece not in best or h.velocity > best[h.piece].velocity or (
            h.velocity == best[h.piece].velocity and h.end - h.start > best[h.piece].end - best[h.piece].start
        ):
            best[h.piece] = h
    paths = {}
    for piece, h in best.items():
        seg = y[h.start:h.end].copy()
        fade = min(len(seg), int(0.01 * sr))
        if fade:
            seg[-fade:] *= np.linspace(1, 0, fade)
        seg /= np.abs(seg).max() + 1e-12
        p = out_dir / f"{piece}.wav"
        sf.write(str(p), seg, sr)
        paths[piece] = p
    return paths
