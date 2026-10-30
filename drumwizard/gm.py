"""General MIDI drum-map note numbers for each drum piece we detect."""

GM_NOTES = {
    "kick": 36,
    "snare": 38,
    "tom": 45,
    "hihat_closed": 42,
    "hihat_open": 46,
    "cymbal": 49,  # crash
    "ride": 51,
}
PIECES = list(GM_NOTES)

# Notes used when the UI splits "cymbal" into separate cymbals (variant 0,1,2): crash 1, crash 2, ride
CYMBAL_VARIANT_NOTES = {"cymbal_0": 49, "cymbal_1": 57, "cymbal_2": 51}

# Toms are split by pitch: variant = slot in this list (low floor tom ... high tom). A kit with fewer toms uses spread-out slots.
TOM_NOTES = [41, 43, 45, 47, 48, 50]
TOM_VARIANT_NOTES = {f"tom_{i}": n for i, n in enumerate(TOM_NOTES)}
