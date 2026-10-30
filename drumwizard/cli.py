import argparse
from pathlib import Path

from .export import write_midi


from .pipeline import run_pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drumwizard")
    ap.add_argument("input", help="song file (or a drum stem with --stem)")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--stem", action="store_true", help="input is already an isolated drum stem")
    ap.add_argument("--drumsep", action="store_true", help="split the drum stem into pieces with DrumSep (needs the drumsep extra)")
    a = ap.parse_args(argv)

    from .pieces import Hit
    res = run_pipeline(a.input, a.out, a.stem, a.drumsep)
    out = Path(a.out)
    hits = [Hit(**h) for h in res["hits"]]
    bpm = res["bpm"]
    write_midi(hits, out / "drums.mid", bpm)
    paths = {p.stem: p for p in (out / "samples").glob("*.wav")}
    counts = {}
    for h in hits:
        counts[h.piece] = counts.get(h.piece, 0) + 1
    print(f"{len(hits)} hits @ {bpm:.0f} bpm: {counts}")
    print(f"MIDI: {out / 'drums.mid'}  samples: {[p.name for p in paths.values()]}")


if __name__ == "__main__":
    main()
