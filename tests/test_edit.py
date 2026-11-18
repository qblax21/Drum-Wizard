from drumwizard.edit import clean_hits, reslice
from drumwizard.pieces import Hit

SR = 1000


def _h(t, piece="snare", start=None, end=None):
    s = round(t * SR) if start is None else start
    return Hit(t, piece, 100, s, s + 300 if end is None else end)


def test_untouched_hits_keep_their_slices():
    hits = [_h(0.1), _h(0.5)]
    before = [(h.start, h.end) for h in hits]
    reslice(hits, SR, 2000, grouped=False)
    assert [(h.start, h.end) for h in hits] == before


def test_new_hit_ends_at_the_next_onset_and_cuts_the_previous_one_short():
    hits = [_h(0.1, end=900), _h(0.8), Hit(0.4, "snare", 100, -1, -1)]
    reslice(hits, SR, 2000, grouped=False)
    assert (hits[2].start, hits[2].end) == (400, 800)
    assert hits[0].end == 400


def test_with_stems_only_the_same_stem_limits_a_slice():
    hits = [_h(0.6, "kick"), Hit(0.4, "snare", 100, -1, -1), _h(0.7, "snare")]
    reslice(hits, SR, 2000, grouped=True)
    assert hits[1].end == 700   # the kick at 0.6 is another stem and is ignored
    reslice(hits, SR, 2000, grouped=False)


def test_clean_hits_clamps_and_rejects():
    h = clean_hits([{"time": -3, "piece": "kick", "velocity": 0}, {"time": 9, "piece": "tom", "variant": 2}], 5.0)
    assert (h[0].time, h[0].velocity, h[1].time, h[1].variant) == (0.0, 1, 5.0, 2)
    for bad in ({"time": 1, "piece": "gong"}, {"piece": "kick"}):
        try:
            clean_hits([bad], 5.0)
        except ValueError:
            continue
        raise AssertionError(bad)
