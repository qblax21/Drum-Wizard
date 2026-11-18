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


def _midi_notes(hits, notes, tmp_path):
    import mido
    from drumwizard.export import write_midi
    write_midi(hits, tmp_path / "x.mid", 120, notes)
    return [m.note for m in mido.MidiFile(tmp_path / "x.mid").tracks[0] if m.type == "note_on"]


def test_ride_and_third_split_cymbal_do_not_share_note_51(tmp_path):
    split = {**GM_NOTES, **CYMBAL_VARIANT_NOTES}
    assert split["cymbal_2"] == split["ride"] == 51          # the clash: both default to 51
    cym = [Hit(i * 0.5, "cymbal", 100, 0, 10, variant=i) for i in range(3)]
    ride = Hit(2.0, "ride", 100, 0, 10)
    assert _midi_notes(cym, split, tmp_path) == [49, 57, 51]          # no ride: the bright cymbal may be the ride
    got = _midi_notes(cym + [ride], split, tmp_path)
    assert got == [49, 57, 52, 51]                                    # ride keeps 51, cymbals are crashes only
    assert set(got[:3]) <= {49, 57, 52, 55}


def test_crash_only_respects_free_crash_notes_and_leaves_other_notes():
    from drumwizard.gm import crash_only, cymbal_variant_notes
    assert cymbal_variant_notes(True)["cymbal_2"] == 52 and cymbal_variant_notes(False)["cymbal_2"] == 51
    n = {**GM_NOTES, "cymbal_0": 49, "cymbal_1": 52, "cymbal_2": 51, "tom_0": 41}
    out = crash_only(n)
    assert out["cymbal_2"] == 57 and out["cymbal_0"] == 49 and out["cymbal_1"] == 52 and out["tom_0"] == 41
