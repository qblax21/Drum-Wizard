"""Outputs: MIDI file of hits, and per-piece one-shot samples for triggering."""
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from .gm import GM_NOTES, TOM_VARIANT_NOTES
from .pieces import Hit, pitch_feats


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
SAMPLE_WANT = {"kick": 0.25, "snare": 0.3, "tom": 0.35, "hihat_closed": 0.12, "hihat_open": 0.3, "cymbal": 1.0, "ride": 1.0}
SAMPLE_MAX = {"kick": 0.4, "snare": 0.5, "tom": 0.5, "hihat_closed": 0.15, "hihat_open": 0.6, "cymbal": 2.5, "ride": 2.0}
TAIL_DB = {"kick": 0.01, "snare": 0.01, "tom": 0.02}   # trim to this fraction of peak (stem bleed rings on below it)
NO_REATTACK = ("hihat_closed", "hihat_open", "cymbal", "ride")   # sustained pieces: cut where a second hit starts
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


def _quality(seg: np.ndarray, piece: str, sr: int) -> float:
    """Tie-breaker between equally roomy hits: fastest-decaying closed hat, most tonal (cleanest-pitched) tom."""
    if piece == "hihat_closed":
        a = np.abs(seg[: int(0.12 * sr)])
        return -float(a[len(a) // 2:].mean() / (a[: len(a) // 4].mean() + 1e-12)) if len(a) > 64 else -1.0
    if piece == "tom":
        pf = pitch_feats(seg, sr)
        return pf[1] - pf[2] if pf else -1.0
    return 0.0


def _cut_reattack(seg: np.ndarray, sr: int) -> np.ndarray:
    """End a sample where a second hit begins (a rise in the 5 ms envelope after the first 50 ms)."""
    import librosa
    on = librosa.onset.onset_detect(y=seg, sr=sr, units="samples", delta=0.08, wait=int(0.02 * sr / 512), backtrack=False)
    late = [int(s) for s in on if s > int(0.05 * sr)]
    return seg[: max(late[0] - int(0.004 * sr), 32)] if late else seg


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
    # separate stems: other pieces are not bleed, so only the piece's own next onset (hit.end) limits a sample
    room = [(False, h.end - h.start) for h in hits] if sources else _room(hits, sr)

    def source(h):
        return sources.get(h.piece, y) if sources else y

    cut: dict[int, np.ndarray] = {}

    def slice_of(i):
        if i not in cut:
            h = hits[i]
            seg = source(h)[h.start:h.start + min(max(room[i][1], 32), int(SAMPLE_MAX.get(h.piece, 1.5) * sr))]
            cut[i] = _cut_reattack(seg, sr) if h.piece in NO_REATTACK else seg
        return cut[i]

    def score(i):
        h, stacked = hits[i], room[i][0]
        seg = slice_of(i)
        q = _quality(seg, h.piece, sr) if h.piece in ("hihat_closed", "tom") else 0.0
        return (not stacked, min(len(seg), int(SAMPLE_WANT.get(h.piece, 0.3) * sr)), round(q, 2), h.velocity)

    best: dict[str, int] = {}
    for i, h in enumerate(hits):
        keys = [_sample_key(h)] if h.piece == "tom" else [h.piece] + ([_sample_key(h)] if h.piece == "cymbal" else [])
        for k in keys:
            if k not in best or score(i) > score(best[k]):
                best[k] = i
    paths = {}
    for key, i in best.items():
        h = hits[i]
        seg = slice_of(i).copy()
        loud = np.nonzero(np.abs(seg) > TAIL_DB.get(h.piece, 0.003) * (np.abs(seg).max() + 1e-12))[0]   # drop the quiet tail
        seg = seg[: min(len(seg), int(loud[-1]) + int(0.02 * sr))] if len(loud) else seg
        fade = min(len(seg), int(0.03 * sr))
        if fade:
            seg[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
        seg /= np.abs(seg).max() + 1e-12
        p = out_dir / f"{key}.wav"
        sf.write(str(p), seg, sr)
        paths[key] = p
    return paths
