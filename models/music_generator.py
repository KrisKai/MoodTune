import random
import numpy as np

from utils.emotion_music_map import get_scale_notes, get_music_params
from config import MELODY_BEATS


def generate_melody(emotion, num_beats=MELODY_BEATS, seed=None):
    """Generate a melody based on the detected emotion.

    Returns a list of (midi_note, duration_beats, velocity) tuples.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    params = get_music_params(emotion)
    scale_notes = get_scale_notes(params['scale'], params['octave'])
    density = params['note_density']
    velocity = params['velocity']
    tempo = params['tempo']

    melody = []
    current_idx = len(scale_notes) // 2  # Start in the middle of the range

    beat = 0
    while beat < num_beats:
        # Decide if this beat has a note or rest
        if random.random() < density:
            # Choose duration: mostly quarter notes, some half/eighth
            dur_choices = [0.5, 1.0, 1.0, 1.0, 2.0]
            duration = random.choice(dur_choices)
            duration = min(duration, num_beats - beat)

            # Melodic motion: prefer stepwise, occasional leaps
            step = random.choices(
                [-2, -1, 0, 1, 2],
                weights=[0.1, 0.3, 0.2, 0.3, 0.1],
            )[0]
            current_idx = max(0, min(len(scale_notes) - 1, current_idx + step))

            note = scale_notes[current_idx]
            vel = velocity + random.randint(-10, 10)
            vel = max(30, min(127, vel))

            melody.append((note, duration, vel))
            beat += duration
        else:
            # Rest
            melody.append((0, 0.5, 0))  # note=0, velocity=0 means rest
            beat += 0.5

    # End on the tonic (first note of scale) for resolution
    if melody:
        tonic = scale_notes[0]
        melody.append((tonic, 2.0, velocity))

    return melody, params
