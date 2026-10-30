import numpy as np

from drumwizard.drumsep import detect_from_stems, load_sources, save_sources
from drumwizard.export import write_samples

SR = 44100


def _tone(f, n=SR // 4, amp=0.8):
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * f * t) * np.exp(-t * 14)).astype(np.float32)


def _stem(hits, amp=1.0, seconds=8):
    y = np.zeros(SR * seconds, dtype=np.float32)
    for t, f in hits:
        s = int(t * SR)
        y[s:s + SR // 4] += _tone(f, amp=amp)
    return y


def test_stems_give_pieces_and_ignore_bleed():
    kick = _stem([(0.5 * i, 50) for i in range(10)])
    toms = _stem([(0.4 + 0.6 * i, f) for i, f in enumerate([90, 140, 200] * 3)])
    toms += 0.03 * _stem([(0.5 * i, 50) for i in range(10)])   # kick bleed in the toms stem, ~-30 dB
    hits, sources = detect_from_stems({"kick": kick, "tom": toms}, SR)
    pieces = [h.piece for h in hits]
    assert pieces.count("kick") == 10
    assert pieces.count("tom") == 9                               # the bleed is gated out
    assert len({h.variant for h in hits if h.piece == "tom"}) == 3
    assert set(sources) == {"kick", "tom"}


def test_samples_cut_from_each_pieces_own_stem(tmp_path):
    kick = _stem([(0.5 * i, 50) for i in range(6)])
    toms = _stem([(0.25 + 0.5 * i, 120) for i in range(6)])
    hits, sources = detect_from_stems({"kick": kick, "tom": toms}, SR)
    save_sources({"kick": kick, "tom": toms}, SR, tmp_path / "stems")
    assert set(load_sources(tmp_path / "stems")) == {"kick", "tom"}
    paths = write_samples(kick + toms, SR, hits, tmp_path / "s", load_sources(tmp_path / "stems"))
    assert "kick.wav" in {p.name for p in paths.values()} and any(k.startswith("tom_") for k in paths)
