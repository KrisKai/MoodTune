import io
import numpy as np
from scipy.io import wavfile

from config import SAMPLE_RATE


def midi_to_freq(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def generate_tone(freq, duration_seconds, sample_rate=SAMPLE_RATE, velocity=100):
    """Generate a single tone with full ADSR envelope and rich harmonics."""
    num_samples = int(sample_rate * duration_seconds)
    if num_samples == 0:
        return np.zeros(0)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

    # Slight vibrato for warmth (5 Hz, subtle depth)
    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * t)
    f = freq * vibrato

    # Mix of sine and harmonics for richer sound
    wave = (0.55 * np.sin(2 * np.pi * f * t) +
            0.25 * np.sin(2 * np.pi * f * 2 * t) +
            0.12 * np.sin(2 * np.pi * f * 3 * t) +
            0.08 * np.sin(2 * np.pi * f * 4 * t))

    # Full ADSR envelope
    attack_time = 0.03    # 30ms attack
    decay_time = 0.1      # 100ms decay
    sustain_level = 0.7   # sustain at 70%
    release_time = min(0.15, duration_seconds * 0.2)  # 150ms or 20% of note

    envelope = np.ones(num_samples)
    for i in range(num_samples):
        time = t[i]
        time_from_end = duration_seconds - time

        if time < attack_time:
            # Attack
            envelope[i] = time / attack_time
        elif time < attack_time + decay_time:
            # Decay
            decay_progress = (time - attack_time) / decay_time
            envelope[i] = 1.0 - (1.0 - sustain_level) * decay_progress
        elif time_from_end < release_time:
            # Release
            envelope[i] = sustain_level * (time_from_end / release_time)
        else:
            # Sustain
            envelope[i] = sustain_level

    wave *= envelope * (velocity / 127.0)
    return wave


def melody_to_wav(melody, tempo):
    """Convert a melody (list of (note, duration_beats, velocity)) to a WAV byte buffer.

    Args:
        melody: List of (midi_note, duration_in_beats, velocity) tuples.
                midi_note=0 means rest.
        tempo: BPM for playback speed.

    Returns:
        BytesIO buffer containing WAV data.
    """
    beat_duration = 60.0 / tempo
    audio_segments = []

    for note, dur_beats, vel in melody:
        duration_sec = dur_beats * beat_duration

        if note == 0 or vel == 0:
            # Rest
            num_samples = int(SAMPLE_RATE * duration_sec)
            audio_segments.append(np.zeros(num_samples))
        else:
            freq = midi_to_freq(note)
            tone = generate_tone(freq, duration_sec, velocity=vel)
            audio_segments.append(tone)

    if not audio_segments:
        return None

    audio = np.concatenate(audio_segments)

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8

    audio_int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio_int16)
    buf.seek(0)
    return buf
