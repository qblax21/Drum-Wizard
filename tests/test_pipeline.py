import numpy as np
import mido
import soundfile as sf

from drumwizard.cli import main

SR = 44100


def _hit(kind, n=SR // 4):
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    if kind == "kick":
        return np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 40)) * t) * np.exp(-t * 18)
    if kind == "snare":
        return (0.6 * rng.standard_normal(n) * np.exp(-t * 20) + 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 25))
    return rng.standard_normal(n) * np.exp(-t * 80)  # closed hat


def test_synthetic_beat(tmp_path):
    y = np.zeros(SR * 4, dtype=np.float32)
    seq = ["kick", "hat", "snare", "hat"] * 2
    for i, k in enumerate(seq):
        s = int(i * 0.5 * SR)
        h = _hit(k)
        y[s:s + len(h)] += h.astype(np.float32)
    sf.write(tmp_path / "d.wav", y, SR)
    main([str(tmp_path / "d.wav"), "--stem", "-o", str(tmp_path / "o")])
    mid = mido.MidiFile(tmp_path / "o" / "drums.mid")
    notes = [m.note for m in mid.tracks[0] if m.type == "note_on"]
    assert notes == [36, 42, 38, 42] * 2
    assert (tmp_path / "o" / "samples" / "kick.wav").exists()
