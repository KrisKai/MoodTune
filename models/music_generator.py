import os
import random
import numpy as np
import torch

from utils.emotion_music_map import get_scale_notes, get_music_params
from config import (
    MELODY_BEATS, EMOTION_LABELS, LSTM_MODEL_PATH, MIDI_VOCAB_SIZE,
)


# ── LSTM-Based Generation ──

def _load_lstm_model(device='cpu'):
    """Try to load the trained LSTM model. Returns None if not available."""
    try:
        from models.melody_lstm import MelodyLSTM
        if not os.path.exists(LSTM_MODEL_PATH):
            return None
        model = MelodyLSTM()
        model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def generate_melody_lstm(emotion, num_notes=20, temperature=0.8, device='cpu'):
    """Generate a melody using the trained LSTM model.

    Returns (melody, params) or None if LSTM model is not available.
    """
    model = _load_lstm_model(device)
    if model is None:
        return None

    emo_idx = EMOTION_LABELS.index(emotion)
    params = get_music_params(emotion)
    scale_notes = get_scale_notes(params['scale'], params['octave'])

    # Build scale mask to keep notes in key
    scale_mask = torch.zeros(MIDI_VOCAB_SIZE)
    for n in scale_notes:
        if 0 <= n < MIDI_VOCAB_SIZE:
            scale_mask[n] = 1.0
    scale_mask = scale_mask.to(device)

    # Seed with first 4 notes of the scale
    seed_notes = scale_notes[:4]

    # Generate
    raw_notes = model.generate(
        emotion_idx=emo_idx,
        start_notes=seed_notes,
        num_notes=num_notes,
        temperature=temperature,
        scale_mask=scale_mask,
        device=device,
    )

    # Convert to melody format: (midi_note, duration_beats, velocity)
    velocity = params['velocity']
    density = params['note_density']
    melody = []

    for note in raw_notes:
        if note == 0 or random.random() > density:
            melody.append((0, 0.5, 0))  # Rest
        else:
            dur_choices = [0.5, 1.0, 1.0, 1.0, 2.0]
            duration = random.choice(dur_choices)
            vel = velocity + random.randint(-10, 10)
            vel = max(30, min(127, vel))
            melody.append((note, duration, vel))

    # End on tonic
    tonic = scale_notes[0]
    melody.append((tonic, 2.0, velocity))

    return melody, params


# ── Rule-Based Generation (Fallback) ──

def generate_melody_rulebased(emotion, num_beats=MELODY_BEATS, seed=None):
    """Rule-based melody generator. Used as fallback when LSTM is not trained."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    params = get_music_params(emotion)
    scale_notes = get_scale_notes(params['scale'], params['octave'])
    density = params['note_density']
    velocity = params['velocity']

    melody = []
    current_idx = len(scale_notes) // 2

    beat = 0
    while beat < num_beats:
        if random.random() < density:
            dur_choices = [0.5, 1.0, 1.0, 1.0, 2.0]
            duration = random.choice(dur_choices)
            duration = min(duration, num_beats - beat)

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
            melody.append((0, 0.5, 0))
            beat += 0.5

    if melody:
        tonic = scale_notes[0]
        melody.append((tonic, 2.0, velocity))

    return melody, params


# ── Main Entry Point ──

def generate_melody(emotion, num_beats=MELODY_BEATS, seed=None, use_lstm=True):
    """Generate a melody for the given emotion.

    Tries LSTM model first. Falls back to rule-based if LSTM is not available.

    Returns:
        melody: list of (midi_note, duration_beats, velocity) tuples
        params: dict of musical parameters used
    """
    if use_lstm:
        result = generate_melody_lstm(emotion, num_notes=num_beats + 4)
        if result is not None:
            return result

    # Fallback to rule-based
    return generate_melody_rulebased(emotion, num_beats=num_beats, seed=seed)
