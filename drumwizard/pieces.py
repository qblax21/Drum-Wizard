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
    variant: int = 0  # which distinct cymbal (only meaningful when piece == "cymbal")


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


MAX_CYMBALS = 3


def assign_cymbal_variants(y: np.ndarray, sr: int, hits: list[Hit]) -> int:
    """Cluster cymbal hits by brightness and decay; set Hit.variant. Returns how many cymbals were found.

    Only splits when the clusters are clearly separate and each has >= 2 hits, so a
    single cymbal played at varying dynamics stays one piece. Variant 0 is the darkest.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    cym = [h for h in hits if h.piece == "cymbal"]
    if not cym:
        return 0
    MIN_DECAY_WINDOW = 0.3   # shorter windows are cut off by the next hit, so their decay can't be measured
    feats = []
    for h in cym:
        seg = y[h.start:min(h.end, h.start + int(1.5 * sr))]
        n = min(len(seg), int(0.4 * sr))
        spec = np.abs(np.fft.rfft(seg[:n] * np.hanning(n))) ** 2
        f = np.fft.rfftfreq(n, 1 / sr)
        centroid = (f * spec).sum() / (spec.sum() + 1e-12)
        env = np.sqrt(np.convolve(seg ** 2, np.ones(512) / 512, mode="same")) + 1e-12
        below = np.nonzero(env < env[: int(0.03 * sr)].max() * 0.1)[0]   # time to fall 20 dB
        decay = (below[0] if len(below) else len(seg)) / sr
        feats.append([np.log(centroid + 1), np.log(decay + 0.05), len(seg) / sr >= MIN_DECAY_WINDOW])
    F = np.array(feats, dtype=float)
    long_ = F[:, 2] > 0
    if long_.sum() < 4:
        return 1
    # fixed scales (not data-relative): 1.0 = ~35% brightness difference or ~1.8x decay difference
    X = F[long_, :2] / np.array([0.3, 0.6])

    best_k, best_labels, best_score = 1, np.zeros(len(X), int), 0.6   # min silhouette to accept a split
    best_centers = X.mean(0, keepdims=True)
    for k in range(2, min(MAX_CYMBALS, len(X) // 2) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        if np.bincount(labels).min() < 2:
            continue
        c = km.cluster_centers_
        if min(np.linalg.norm(c[i] - c[j]) for i in range(k) for j in range(i)) < 1.0:
            continue   # clusters too close together: same cymbal at different dynamics
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_labels, best_score, best_centers = k, labels, score, c
    # darkest cluster = variant 0
    rank = {int(c): r for r, c in enumerate(np.argsort(best_centers[:, 0]))}
    li = 0
    for h, is_long, row in zip(cym, long_, F):
        if is_long:
            h.variant = rank[int(best_labels[li])]
            li += 1
        else:   # cut-off hit: nearest cluster by brightness only
            h.variant = rank[int(np.argmin(np.abs(best_centers[:, 0] - row[0] / 0.3)))]
    return best_k


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
    hits = _add_top_layer(y, sr, hits, peak)
    hits = _suppress_cymbal_tails(y, sr, hits)
    assign_cymbal_variants(y, sr, hits)
    return hits


HP_HZ = 6000
COINCIDE = 0.03       # s: a high-band onset this close to a full-band hit is the same event
TAIL_TAU = 0.45       # s: assumed cymbal amplitude decay constant
TAIL_MARGIN = 1.5     # a hit must beat the decaying previous cymbal by this factor to count as new


def _hp(y: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, HP_HZ, "hp", fs=sr, output="sos"), y)


def _add_top_layer(y: np.ndarray, sr: int, hits: list[Hit], peak: float) -> list[Hit]:
    """Full-band detection labels a hat/cymbal hit under a kick or tom as that drum, or misses it.
    Detect onsets in the >6 kHz band and add hat/cymbal hits the full-band pass did not already report."""
    yh = _hp(y, sr)
    hp_peak = np.abs(yh).max() + 1e-12
    pad = int(0.1 * sr)
    on = librosa.onset.onset_detect(y=np.pad(yh, (pad, 0)), sr=sr, units="samples", backtrack=False, delta=0.1,
                                    wait=int(0.03 * sr / 512)) - pad
    times = np.array([h.start for h in hits]) / sr
    lead = int(0.003 * sr)
    starts = np.maximum(on - lead, 0)
    out = list(hits)
    for i, s in enumerate(starts):
        t = s / sr
        near = np.nonzero(np.abs(times - t) < COINCIDE)[0] if len(times) else []
        if any(hits[j].piece in ("snare", "hihat_closed", "hihat_open", "cymbal") for j in near):
            continue   # already a hat/cymbal, or snare wires: not a separate hat
        nxt = starts[i + 1] if i + 1 < len(starts) else len(y)
        e = int(min(nxt, s + 1.5 * sr))
        seg = yh[s:e]
        if len(seg) < 32:
            continue
        amp = np.abs(seg[: int(0.05 * sr)]).max() / hp_peak
        if amp < 0.05:
            continue
        piece = classify_hit(seg, sr)
        if piece not in ("hihat_closed", "hihat_open", "cymbal"):
            continue
        vel = int(np.clip(round(20 + 107 * amp ** 0.7), 1, 127))
        out.append(Hit(float(t), piece, vel, int(s), e))
    return sorted(out, key=lambda h: h.start)


def _suppress_cymbal_tails(y: np.ndarray, sr: int, hits: list[Hit]) -> list[Hit]:
    """Drop cymbal/open-hat hits that are just the decaying tail of an earlier cymbal re-triggering the detector."""
    yh = _hp(y, sr)
    win = int(0.04 * sr)
    keep: list[Hit] = []
    ref_t, ref_a = -1e9, 0.0   # last real cymbal: when, and its high-band peak
    for h in hits:
        a = np.abs(yh[h.start:h.start + win]).max() if h.start < len(yh) else 0.0
        if h.piece in ("cymbal", "hihat_open"):
            expected = ref_a * np.exp(-(h.start / sr - ref_t) / TAIL_TAU)
            if a < TAIL_MARGIN * expected:
                continue
            ref_t, ref_a = h.start / sr, a
        keep.append(h)
    return keep
