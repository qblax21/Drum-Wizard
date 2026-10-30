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


# Seconds of undisturbed audio wanted per piece when choosing which hit to cut a sample from.
SAMPLE_WANT = {"kick": 0.25, "snare": 0.3, "tom": 0.35, "hihat_closed": 0.12, "hihat_open": 0.3, "cymbal": 1.0}
COINCIDE = 0.02   # s: another piece landing this close to a hit bleeds into its attack


def _room(hits: list[Hit], sr: int) -> list[tuple[bool, int]]:
    """Per hit: (does another piece land on it, samples until the next hit that is not simultaneous with it)."""
    order = sorted(range(len(hits)), key=lambda i: hits[i].start)
    starts = np.array([hits[i].start for i in order])
    out = [(False, 0)] * len(hits)
    for pos, i in enumerate(order):
        h, c = hits[i], int(COINCIDE * sr)
        lo, hi = np.searchsorted(starts, h.start - c), np.searchsorted(starts, h.start + c, side="right")
        near = [hits[order[j]] for j in range(lo, hi) if order[j] != i]
        stacked = any(o.piece != h.piece for o in near)
        later = starts[np.searchsorted(starts, h.start + c, side="right"):]
        nxt = int(later[0]) if len(later) else h.end
        out[i] = (stacked, min(h.end, nxt) - h.start)
    return out


def write_samples(y: np.ndarray, sr: int, hits: list[Hit], out_dir: str | Path,
                  sources: dict[str, np.ndarray] | None = None) -> dict[str, Path]:
    """Save the cleanest hit of each piece as a one-shot WAV.

    Cleanest = nothing else playing on its attack, then the most undisturbed room before the next hit (up to
    SAMPLE_WANT), then loudest. Toms get one `tom_<slot>.wav` per pitch. Cymbals get a combined `cymbal.wav`
    plus one `cymbal_<n>.wav` per detected variant. `sources` maps a piece to its own stem (DrumSep); else all
    samples are cut from `y`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    room = _room(hits, sr)

    def score(i):
        stacked, n = room[i]
        return (not stacked, min(n, int(SAMPLE_WANT.get(hits[i].piece, 0.3) * sr)), hits[i].velocity)

    best: dict[str, int] = {}
    for i, h in enumerate(hits):
        keys = [_sample_key(h)] if h.piece == "tom" else [h.piece] + ([_sample_key(h)] if h.piece == "cymbal" else [])
        for k in keys:
            if k not in best or score(i) > score(best[k]):
                best[k] = i
    paths = {}
    for key, i in best.items():
        h = hits[i]
        src = sources.get(h.piece, y) if sources else y
        seg = src[h.start:h.start + max(room[i][1], 32)].copy()
        loud = np.nonzero(np.abs(seg) > 0.003 * (np.abs(seg).max() + 1e-12))[0]   # drop the silent tail (-50 dB)
        seg = seg[: min(len(seg), int(loud[-1]) + int(0.02 * sr))] if len(loud) else seg
        fade = min(len(seg), int(0.03 * sr))
        if fade:
            seg[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
        seg /= np.abs(seg).max() + 1e-12
        p = out_dir / f"{key}.wav"
        sf.write(str(p), seg, sr)
        paths[key] = p
    return paths
