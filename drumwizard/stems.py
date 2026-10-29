"""Stage 1: isolate the drum stem from a full mix using Demucs."""
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def separate_drums(song: str | Path, model: str = "htdemucs") -> tuple[np.ndarray, int]:
    """Return (mono drum audio, sample rate). Needs the `separation` extra."""
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        from demucs.pretrained import get_model
    except ImportError as e:
        raise RuntimeError("Install separation support: pip install 'drumwizard[separation]'") from e

    m = get_model(model)
    m.eval()
    wav = AudioFile(str(song)).read(streams=0, samplerate=m.samplerate, channels=m.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        out = apply_model(m, wav[None], device=device, split=True, progress=True)[0]
    out = out * ref.std() + ref.mean()
    drums = out[m.sources.index("drums")].mean(0).cpu().numpy()
    return drums, m.samplerate


def load_drum_stem(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an already-isolated drum stem (skips Demucs)."""
    y, sr = sf.read(str(path), always_2d=True)
    return y.mean(1).astype(np.float32), sr
