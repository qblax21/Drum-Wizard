from dataclasses import asdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .export import write_samples
from .pieces import detect_hits
from .stems import load_drum_stem, separate_drums


def run_pipeline(src: str | Path, out: str | Path, is_stem: bool = False) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    y, sr = load_drum_stem(src) if is_stem else separate_drums(src)
    sf.write(str(out / "drums.wav"), y, sr)
    hits = detect_hits(y, sr)
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    write_samples(y, sr, hits, out / "samples")
    return {"bpm": tempo if tempo > 0 else 120.0, "duration": len(y) / sr,
            "hits": [asdict(h) for h in hits]}
