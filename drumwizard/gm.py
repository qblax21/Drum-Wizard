"""General MIDI drum-map note numbers for each drum piece we detect."""

GM_NOTES = {
    "kick": 36,
    "snare": 38,
    "tom_low": 45,
    "tom_high": 50,
    "hihat_closed": 42,
    "hihat_open": 46,
    "cymbal": 49,  # crash
}
PIECES = list(GM_NOTES)
