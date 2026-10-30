"""Optional stage 2b: split the drum stem into per-piece stems with DrumSep (MDX23C) and find hits in each stem.

Needs the `drumsep` extra. The DrumSep weights (aufr33 & jarredou) have no stated licence: treat them as
non-commercial until that is clarified, which is why this stage is optional and sits behind `separate_and_detect`.
"""
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .pieces import Hit, assign_cymbal_variants, assign_tom_variants

GATE = 0.12           # onsets quieter than this fraction of a stem's loud hits are bleed from the other pieces
CUT = 0.04            # ...but any onset louder than this still ends the previous hit, so samples never span two hits
CROSS = 0.03          # s: ride and crash onsets this close are one hit; the louder stem wins
RAW_TO_PIECE = {"kick": "kick", "snare": "snare", "toms": "tom", "hh": "hihat", "ride": "ride", "crash": "cymbal"}
SOURCE_FILE = {"kick": "kick", "snare": "snare", "tom": "tom", "hihat_closed": "hihat", "hihat_open": "hihat",
               "cymbal": "cymbal", "ride": "ride"}


def separate_stems(drums_wav: str | Path, device: str | None = None) -> tuple[dict[str, np.ndarray], int]:
    """Run DrumSep on a drum file. Returns ({piece group: mono audio}, sample rate)."""
    try:
        from mdxnet_infer import separate
    except ImportError as e:
        raise RuntimeError("Install DrumSep support: pip install 'drumwizard[drumsep]'") from e
    with tempfile.TemporaryDirectory(prefix="drumsep-") as d:
        paths = separate(drums_wav, d, "drumsep-6stem", device=device, progress=False)
        out: dict[str, np.ndarray] = {}
        for name, p in paths.items():
            piece = RAW_TO_PIECE.get(name)
            if piece:
                y, sr = sf.read(str(p), dtype="float32", always_2d=True)
                out[piece] = y.mean(1)
    return out, sr


def _onsets(y: np.ndarray, sr: int) -> list[tuple[int, float, float]]:
    """(start sample, loudness relative to the stem's loud hits, absolute peak) for every onset above CUT."""
    pad = int(0.1 * sr)
    on = librosa.onset.onset_detect(y=np.pad(y, (pad, 0)), sr=sr, units="samples", backtrack=False, delta=0.1,
                                    wait=int(0.03 * sr / 512)) - pad
    on = np.maximum(on - int(0.003 * sr), 0)
    if not len(on):
        return []
    pk = np.array([np.abs(y[s:s + int(0.04 * sr)]).max() for s in on])
    ref = np.percentile(pk, 98) + 1e-12
    return [(int(s), float(p / ref), float(p)) for s, p in zip(on, pk) if p >= CUT * ref]


def _velocity(amp: float) -> int:
    return int(np.clip(round(20 + 107 * min(amp, 1.2) ** 0.7), 1, 127))


def _is_open(y: np.ndarray, sr: int, s: int, e: int) -> bool:
    seg = np.abs(y[s:e][: int(0.25 * sr)])
    if len(seg) < int(0.12 * sr):
        return False
    return seg[len(seg) // 2:].mean() / (seg[: len(seg) // 4].mean() + 1e-12) > 0.35


def detect_from_stems(stems: dict[str, np.ndarray], sr: int) -> tuple[list[Hit], dict[str, np.ndarray]]:
    ons = {piece: _onsets(y, sr) for piece, y in stems.items()}
    if "ride" in ons and "cymbal" in ons:   # one cymbal strike shows up in both stems: keep the louder
        a, b = ons["ride"], ons["cymbal"]
        drop_a, drop_b = set(), set()
        for i, (s, _, pa) in enumerate(a):
            for j, (t, _, pb) in enumerate(b):
                if abs(s - t) < CROSS * sr:
                    (drop_b if pa >= pb else drop_a).add(j if pa >= pb else i)
        ons["ride"] = [o for i, o in enumerate(a) if i not in drop_a]
        ons["cymbal"] = [o for j, o in enumerate(b) if j not in drop_b]
    hits: list[Hit] = []
    for piece, y in stems.items():
        lst = ons[piece]
        for i, (s, amp, _) in enumerate(lst):
            if amp < GATE:
                continue
            e = int(min(lst[i + 1][0] if i + 1 < len(lst) else len(y), s + 1.5 * sr))
            if e - s < 32:
                continue
            kind = piece
            if piece == "hihat":
                kind = "hihat_open" if _is_open(y, sr, s, e) else "hihat_closed"
            hits.append(Hit(float(s / sr), kind, _velocity(amp), s, e))
    hits.sort(key=lambda h: h.start)
    for fn, piece in ((assign_cymbal_variants, "cymbal"), (assign_tom_variants, "tom")):
        fn(stems.get(piece, np.zeros(1, np.float32)), sr, hits)
    sources = {k: stems[v] for k, v in SOURCE_FILE.items() if v in stems}
    return hits, sources


def save_sources(stems: dict[str, np.ndarray], sr: int, out_dir: str | Path) -> None:
    """Keep the per-piece stems so samples can be re-cut after the user edits hit labels."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, y in stems.items():
        sf.write(str(out_dir / f"{name}.wav"), y, sr, subtype="PCM_16")


def load_sources(out_dir: str | Path) -> dict[str, np.ndarray] | None:
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    files = {p.stem: sf.read(str(p), dtype="float32")[0] for p in out_dir.glob("*.wav")}
    return {piece: files[f] for piece, f in SOURCE_FILE.items() if f in files}


def separate_and_detect(drums_wav: str | Path, stems_dir: str | Path, device: str | None = None):
    """DrumSep a drum file, save the per-piece stems to stems_dir, return (hits, per-piece sources)."""
    stems, sr = separate_stems(drums_wav, device)
    save_sources(stems, sr, stems_dir)
    return detect_from_stems(stems, sr)
