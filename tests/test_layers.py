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
