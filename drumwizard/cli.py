import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .export import write_midi, write_samples
from .pieces import detect_hits
from .stems import load_drum_stem, separate_drums


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drumwizard")
    ap.add_argument("input", help="song file (or a drum stem with --stem)")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--stem", action="store_true", help="input is already an isolated drum stem")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    y, sr = load_drum_stem(a.input) if a.stem else separate_drums(a.input)
    if not a.stem:
        sf.write(str(out / "drums.wav"), y, sr)
    hits = detect_hits(y, sr)
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    bpm = tempo if tempo > 0 else 120.0
    write_midi(hits, out / "drums.mid", bpm)
    paths = write_samples(y, sr, hits, out / "samples")
    counts = {}
    for h in hits:
        counts[h.piece] = counts.get(h.piece, 0) + 1
    print(f"{len(hits)} hits @ {bpm:.0f} bpm: {counts}")
    print(f"MIDI: {out / 'drums.mid'}  samples: {[p.name for p in paths.values()]}")


if __name__ == "__main__":
    main()
