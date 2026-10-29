"""Stage 2: find every drum hit in the drum stem and classify it into a kit piece.

This is a signal-processing baseline (onset detection + spectral features).
`classify_hit` is the single seam to swap for a learned model later.
"""
from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class Hit:
    time: float       # seconds
    piece: str        # key of gm.GM_NOTES
    velocity: int     # 1..127
    start: int        # sample index of slice start
    end: int          # sample index of slice end (before next hit, capped)


def classify_hit(seg: np.ndarray, sr: int) -> str:
    """Classify a short windowed hit from spectral shape and decay."""
    n = min(len(seg), int(0.25 * sr))
    seg = seg[:n]
    if n < 64:
        return "hihat_closed"
    spec = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    tot = spec.sum() + 1e-12
    low = spec[f < 150].sum() / tot
    mid = spec[(f >= 150) & (f < 1500)].sum() / tot
    high = spec[f >= 5000].sum() / tot
    centroid = (f * spec).sum() / tot
    flat = librosa.feature.spectral_flatness(y=seg, n_fft=min(2048, n)).mean()

    # decay: fraction of energy remaining in the back half of the window
    env = np.abs(seg)
    tail = env[n // 2:].mean() / (env[: n // 4].mean() + 1e-12)

    if low > 0.5 and centroid < 600:
        return "kick"
    # snares are broadband but carry a mid-range body that hats/cymbals lack
    if mid > 0.12 and tail < 0.35:
        return "snare"
    if high > 0.35 or centroid > 4000:
        if tail > 0.35:
            return "cymbal" if centroid < 7000 or tail > 0.6 else "hihat_open"
        return "hihat_closed"
    if flat > 0.05 and mid > 0.35:
        return "snare"
    return "tom_low" if centroid < 400 else "tom_high"


def detect_hits(y: np.ndarray, sr: int) -> list[Hit]:
    # pad so a hit at t=0 still has a pre-roll for the detector
    pad = int(0.1 * sr)
    onsets = librosa.onset.onset_detect(
        y=np.pad(y, (pad, 0)), sr=sr, units="samples", backtrack=False, delta=0.1, wait=int(0.03 * sr / 512)
    )
    lead = int(0.003 * sr)
    starts = np.maximum(onsets - pad - lead, 0)
    peak = np.abs(y).max() + 1e-12
    hits = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else len(y)
        e = min(nxt, s + int(1.5 * sr))
        seg = y[s:e]
        if len(seg) < 32:
            continue
        amp = np.abs(seg[: int(0.05 * sr)]).max() / peak
        vel = int(np.clip(round(20 + 107 * amp ** 0.7), 1, 127))
        hits.append(Hit(float(s / sr), classify_hit(seg, sr), vel, int(s), int(e)))
    return hits
