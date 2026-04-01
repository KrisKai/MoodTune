from config import EMOTION_MUSIC_PARAMS

# Scale definitions: MIDI note offsets from root
SCALES = {
    'major':      [0, 2, 4, 5, 7, 9, 11],
    'minor':      [0, 2, 3, 5, 7, 8, 10],
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
}


def get_scale_notes(scale_name, octave):
    """Get MIDI note numbers for a scale at a given octave."""
    root = octave * 12  # C of that octave
    offsets = SCALES[scale_name]
    # Return notes across 2 octaves for melodic range
    notes = []
    for oct_offset in range(2):
        for offset in offsets:
            note = root + oct_offset * 12 + offset
            if 0 <= note <= 127:
                notes.append(note)
    return notes


def get_music_params(emotion):
    """Get musical parameters for a given emotion."""
    return EMOTION_MUSIC_PARAMS.get(emotion, EMOTION_MUSIC_PARAMS['neutral'])
