import numpy as np

from drumwizard.pieces import detect_hits

SR = 44100


def _kick(n=SR // 4):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 40)) * t) * np.exp(-t * 18)


def _hat(n=SR // 10):
    t = np.arange(n) / SR
    return np.random.default_rng(1).standard_normal(n) * np.exp(-t * 80) * 0.4


def _place(y, h, s):
    y[s:s + len(h)] += h.astype(np.float32)


def test_hat_under_kick_is_found():
    y = np.zeros(SR * 3, dtype=np.float32)
    for i in range(4):
        s = int(i * 0.5 * SR)
        _place(y, _kick(), s)
        _place(y, _hat(), s)
    pieces = [h.piece for h in detect_hits(y, SR)]
    assert pieces.count("kick") == 4
    assert sum(p.startswith("hihat") for p in pieces) == 4


def test_cymbal_tail_not_retriggered():
    n = SR * 2
    t = np.arange(n) / SR
    y = (np.random.default_rng(2).standard_normal(n) * np.exp(-t * 2.5)).astype(np.float32)
    y *= np.hanning(n) ** 0.01
    y *= 0.8 / np.abs(y).max()
    hits = detect_hits(y, SR)
    assert sum(h.piece in ("cymbal", "hihat_open") for h in hits) <= 1


def _tom(f, n=SR // 4):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 14) * 0.8


def test_toms_split_by_pitch_and_map_to_gm_notes():
    from drumwizard.export import note_for
    from drumwizard.gm import TOM_VARIANT_NOTES, GM_NOTES
    y = np.zeros(SR * 12, dtype=np.float32)
    freqs = [95, 135, 190] * 5
    for i, f in enumerate(freqs):
        _place(y, _tom(f), int((0.3 + i * 0.7) * SR))
    hits = [h for h in detect_hits(y, SR) if h.piece == "tom"]
    assert len(hits) == 15
    assert len({h.variant for h in hits}) == 3
    by_f = {}
    for h, f in zip(hits, freqs):
        by_f.setdefault(f, set()).add(h.variant)
    assert all(len(v) == 1 for v in by_f.values())                      # one tom per pitch
    assert max(by_f[95]) < max(by_f[135]) < max(by_f[190])              # higher pitch -> higher slot
    assert note_for(hits[0], GM_NOTES) == TOM_VARIANT_NOTES[f"tom_{hits[0].variant}"]


def test_low_kick_pitch_is_not_a_tom():
    y = np.zeros(SR * 3, dtype=np.float32)
    for i in range(4):
        t = np.arange(SR // 3) / SR
        _place(y, np.sin(2 * np.pi * 52 * t) * np.exp(-t * 12), int((0.2 + i * 0.6) * SR))
    assert {h.piece for h in detect_hits(y, SR)} == {"kick"}
