"""Validate and re-slice a hit list the user edited in the UI (hits added, moved, reassigned or deleted)."""
import numpy as np

from .drumsep import SOURCE_FILE
from .gm import GM_NOTES
from .pieces import Hit

SLACK = 0.0005        # s: a hit whose start is this close to its time was not moved
MAX_TAIL = 1.5        # s: longest slice a hit owns, as in detection
MIN_SLICE = 32


def clean_hits(raw: list[dict], duration: float) -> list[Hit]:
    """Raw client hits -> Hits. Keeps the client's order. Unknown pieces raise ValueError; time is clamped into the
    track, velocity into 1..127. start/end are placeholders until `reslice` fixes them."""
    hits = []
    for r in raw:
        if r.get("piece") not in GM_NOTES:
            raise ValueError(f"unknown piece {r.get('piece')!r}")
        if not isinstance(r.get("time"), (int, float)):
            raise ValueError("hit has no time")
        t = float(np.clip(r["time"], 0.0, max(duration, 0.0)))
        hits.append(Hit(t, r["piece"], int(np.clip(r.get("velocity", 100), 1, 127)),
                        int(r.get("start", -1)), int(r.get("end", -1)), int(r.get("variant", 0))))
    return hits


def reslice(hits: list[Hit], sr: int, n_samples: int, grouped: bool) -> list[Hit]:
    """Give moved and new hits a slice that matches their time.

    A hit is moved/new when its start no longer sits at its time. It starts at time*sr and ends at the next onset
    that limits its sample: the next hit of the same stem when `grouped` (DrumSep stems: other pieces are not
    bleed), else the next hit of any piece. The hit before it in that group is cut short if the edit landed inside
    its slice. Everything else keeps the slice detection gave it. Order is preserved.
    """
    def group(h):
        return SOURCE_FILE.get(h.piece, h.piece) if grouped else None

    edited = [i for i, h in enumerate(hits) if h.start < 0 or h.end <= h.start or abs(h.start - h.time * sr) > SLACK * sr]
    for i in edited:
        h = hits[i]
        h.start = min(int(round(h.time * sr)), max(n_samples - MIN_SLICE, 0))
    for i in edited:
        h = hits[i]
        others = [o for j, o in enumerate(hits) if j != i and group(o) == group(h)]
        later = [o.start for o in others if o.start > h.start]
        h.end = int(min(min(later) if later else n_samples, h.start + MAX_TAIL * sr, n_samples))
        h.end = max(h.end, min(h.start + MIN_SLICE, n_samples))
        earlier = [o for o in others if o.start < h.start and o.end > h.start]
        if earlier:
            prev = max(earlier, key=lambda o: o.start)
            prev.end = max(h.start, prev.start + MIN_SLICE)
    return hits
