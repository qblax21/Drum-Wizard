"""Live use: play one-shot samples when the e-drum kit sends notes, and/or forward MIDI."""
from pathlib import Path

import soundfile as sf


def list_ports() -> list[str]:
    import rtmidi
    return rtmidi.MidiIn().get_ports()


def run(samples: dict[int, Path], port: int = 0) -> None:
    """samples maps incoming MIDI note number -> wav file. Blocks until Ctrl-C."""
    import rtmidi
    import sounddevice as sd

    loaded = {n: sf.read(str(p), dtype="float32") for n, p in samples.items()}
    mi = rtmidi.MidiIn()
    mi.open_port(port)

    def cb(event, _):
        (status, note, vel), _dt = event
        if status & 0xF0 == 0x90 and vel > 0 and note in loaded:
            data, sr = loaded[note]
            sd.play(data * (vel / 127), sr)

    mi.set_callback(cb)
    print("Listening... Ctrl-C to stop")
    try:
        while True:
            import time; time.sleep(0.5)
    except KeyboardInterrupt:
        pass
