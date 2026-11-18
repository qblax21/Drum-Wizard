"""Run the real separation models on a short synthetic clip. Slow (model load, GPU or minutes on CPU) and the weights
must be downloaded already, so these are marked `slow` and skipped by default and in CI. Run them with: pytest -m slow"""
import numpy as np
import pytest
import soundfile as sf

from test_pipeline import SR, _hit

pytestmark = pytest.mark.slow


def _clip(tmp_path, with_chords=False):
    y = np.zeros(SR * 6, dtype=np.float32)
    for i, k in enumerate(["kick", "hat", "snare", "hat"] * 3):
        h = _hit(k); s = int(i * .5 * SR); y[s:s + len(h)] += h
    if with_chords:
        t = np.arange(len(y)) / SR
        y += 0.2 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 330 * t)
    p = tmp_path / "clip.wav"
    sf.write(p, y, SR)
    return p


def test_demucs_isolates_a_drum_track(tmp_path):
    pytest.importorskip("demucs")
    from drumwizard.stems import separate_drums
    y, sr = separate_drums(_clip(tmp_path, with_chords=True))
    assert sr == 44100 and abs(len(y) - 6 * SR) < SR // 10 and np.isfinite(y).all() and np.abs(y).max() > 0.01


def test_drumsep_splits_a_drum_clip_into_piece_stems(tmp_path):
    pytest.importorskip("mdxnet_infer")
    from drumwizard.drumsep import load_sources, separate_and_detect
    hits, sources = separate_and_detect(_clip(tmp_path), tmp_path / "stems")
    assert hits and {"kick", "snare"} <= set(sources)
    assert set(load_sources(tmp_path / "stems")) >= {"kick", "snare"}   # saved for later re-cutting
