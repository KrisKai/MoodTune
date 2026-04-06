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


def _generate_phrase(model, emo_idx, scale_notes, scale_mask, num_notes, temperature, device):
    """Generate a single phrase of notes using the LSTM."""
    seed_notes = random.sample(scale_notes[:8], min(4, len(scale_notes)))
    raw_notes = model.generate(
        emotion_idx=emo_idx,
        start_notes=seed_notes,
        num_notes=num_notes,
        temperature=temperature,
        scale_mask=scale_mask,
        device=device,
    )
    return raw_notes


def _notes_to_melody(raw_notes, params, variation=0):
    """Convert raw MIDI notes to melody format with musical durations.

    variation: 0 = original, >0 = add slight rhythmic/velocity variation.
    """
    velocity = params['velocity']
    density = params['note_density']
    melody = []

    for i, note in enumerate(raw_notes):
        if note == 0 or random.random() > density:
            melody.append((0, 0.5, 0))  # Rest
        else:
            dur_choices = [0.5, 1.0, 1.0, 1.0, 2.0]
            duration = random.choice(dur_choices)
            # Add variation to repeated phrases
            vel = velocity + random.randint(-10 - variation * 3, 10 + variation * 3)
            vel = max(30, min(127, vel))
            # Occasional rhythmic variation on repeats
            if variation > 0 and random.random() < 0.2:
                duration = random.choice([0.5, 0.5, 1.0, 1.5])
            melody.append((note, duration, vel))

    return melody


def generate_melody_lstm(emotion, target_beats=300, temperature=0.8, device='cpu'):
    """Generate a structured melody using the trained LSTM model.

    Creates song-like structure: Intro → Verse → Chorus → Verse → Chorus → Outro
    Each section uses LSTM generation with phrase repetition and variation.

    Returns (melody, params) or None if LSTM model is not available.
    """
    model = _load_lstm_model(device)
    if model is None:
        return None

    emo_idx = EMOTION_LABELS.index(emotion)
    params = get_music_params(emotion)
    scale_notes = get_scale_notes(params['scale'], params['octave'])
    tonic = scale_notes[0]

    # Build scale mask to keep notes in key
    scale_mask = torch.zeros(MIDI_VOCAB_SIZE)
    for n in scale_notes:
        if 0 <= n < MIDI_VOCAB_SIZE:
            scale_mask[n] = 1.0
    scale_mask = scale_mask.to(device)

    # --- Song Structure ---
    # Generate core phrases, then arrange into sections with repetition
    phrase_len = 16  # notes per phrase

    # Generate unique phrases
    intro_notes = _generate_phrase(model, emo_idx, scale_notes, scale_mask, phrase_len, temperature * 0.9, device)
    verse_notes = _generate_phrase(model, emo_idx, scale_notes, scale_mask, phrase_len * 2, temperature, device)
    chorus_notes = _generate_phrase(model, emo_idx, scale_notes, scale_mask, phrase_len * 2, temperature * 1.1, device)
    bridge_notes = _generate_phrase(model, emo_idx, scale_notes, scale_mask, phrase_len, temperature * 0.85, device)

    melody = []
    total_beats = 0

    def add_section(raw_notes, variation=0, label=""):
        nonlocal total_beats
        section = _notes_to_melody(raw_notes, params, variation=variation)
        for note_tuple in section:
            if total_beats >= target_beats:
                break
            melody.append(note_tuple)
            total_beats += note_tuple[1]

    # Intro (softer)
    add_section(intro_notes, variation=0, label="intro")

    # Verse 1
    add_section(verse_notes, variation=0, label="verse1")

    # Chorus 1
    add_section(chorus_notes, variation=0, label="chorus1")

    # Verse 2 (slight variation of verse 1)
    add_section(verse_notes, variation=1, label="verse2")

    # Chorus 2
    add_section(chorus_notes, variation=1, label="chorus2")

    # Bridge
    add_section(bridge_notes, variation=0, label="bridge")

    # Fill remaining time by repeating chorus/verse with increasing variation
    repeat = 2
    while total_beats < target_beats:
        add_section(verse_notes, variation=repeat, label=f"verse_r{repeat}")
        add_section(chorus_notes, variation=repeat, label=f"chorus_r{repeat}")
        repeat += 1
        if repeat > 5:
            # Generate fresh material if we still need more
            fresh = _generate_phrase(model, emo_idx, scale_notes, scale_mask, phrase_len * 2, temperature, device)
            add_section(fresh, variation=0, label="fresh")

    # Outro: slow down to tonic
    melody.append((tonic, 2.0, params['velocity']))
    melody.append((tonic, 4.0, int(params['velocity'] * 0.7)))

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
        result = generate_melody_lstm(emotion, target_beats=num_beats, temperature=0.8)
        if result is not None:
            return result

    # Fallback to rule-based
    return generate_melody_rulebased(emotion, num_beats=num_beats, seed=seed)
