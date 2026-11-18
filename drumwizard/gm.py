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

# With a separate ride piece (note 51) the split cymbals are all crashes, so the third becomes a china instead of
# a second ride that would share note 51.
CYMBAL_VARIANT_NOTES_WITH_RIDE = {"cymbal_0": 49, "cymbal_1": 57, "cymbal_2": 52}
CRASH_NOTES = [49, 57, 52, 55]   # crash 1, crash 2, china, splash
RIDE_NOTES = (51, 53, 59)


def cymbal_variant_notes(has_ride: bool) -> dict:
    return CYMBAL_VARIANT_NOTES_WITH_RIDE if has_ride else CYMBAL_VARIANT_NOTES


def crash_only(notes: dict) -> dict:
    """Notes with every split cymbal moved off the ride's note(s) onto a free crash note."""
    out, ride = dict(notes), notes.get("ride", 51)
    keys = sorted(k for k in out if k.startswith("cymbal_"))
    for k in keys:
        if out[k] == ride or out[k] in RIDE_NOTES:
            taken = {out[o] for o in keys if o != k}
            prefer = [CYMBAL_VARIANT_NOTES_WITH_RIDE.get(k, CRASH_NOTES[0])] + CRASH_NOTES   # the variant's own crash note first
            out[k] = next((n for n in prefer if n not in taken and n != ride), CRASH_NOTES[0])
    return out


# Toms are split by pitch: variant = slot in this list (low floor tom ... high tom). A kit with fewer toms uses spread-out slots.
TOM_NOTES = [41, 43, 45, 47, 48, 50]
TOM_VARIANT_NOTES = {f"tom_{i}": n for i, n in enumerate(TOM_NOTES)}
