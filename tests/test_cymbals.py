import numpy as np
from scipy.signal import butter, sosfilt

from drumwizard.pieces import Hit, assign_cymbal_variants
from drumwizard.export import note_for
from drumwizard.gm import GM_NOTES, CYMBAL_VARIANT_NOTES

SR = 44100


def _cym(lo, hi, tau, seed):
    n = SR
    t = np.arange(n) / SR
    x = np.random.default_rng(seed).standard_normal(n) * np.exp(-t / tau)
    return sosfilt(butter(4, [lo, hi], "band", fs=SR, output="sos"), x)


def _track(kinds):
    y = np.zeros(SR * (len(kinds) + 1), dtype=np.float32)
    hits = []
    for i, (lo, hi, tau) in enumerate(kinds):
        s = i * SR
        h = _cym(lo, hi, tau, i)
        y[s:s + len(h)] += h
        hits.append(Hit(float(i), "cymbal", 100, s, s + SR))
    return y, hits


def test_two_distinct_cymbals_split():
    dark, bright = (3000, 8000, 0.9), (9000, 18000, 0.25)
    y, hits = _track([dark, bright] * 4)
    assert assign_cymbal_variants(y, SR, hits) == 2
    assert {h.variant for h in hits[0::2]} == {0}
    assert {h.variant for h in hits[1::2]} == {1}   # brighter cymbal is variant 1


def test_single_cymbal_stays_collapsed():
    y, hits = _track([(3000, 8000, 0.9)] * 6)
    assert assign_cymbal_variants(y, SR, hits) == 1
    assert {h.variant for h in hits} == {0}


def test_midi_note_follows_split_state():
    h = Hit(0, "cymbal", 100, 0, 10, variant=1)
    assert note_for(h, GM_NOTES) == 49                                 # collapsed: shared note
    assert note_for(h, {**GM_NOTES, **CYMBAL_VARIANT_NOTES}) == 57     # split: per-cymbal note
