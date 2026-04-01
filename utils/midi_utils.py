import io
import numpy as np
from scipy.io import wavfile

from config import SAMPLE_RATE


def midi_to_freq(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def generate_tone(freq, duration_seconds, sample_rate=SAMPLE_RATE, velocity=100):
    """Generate a single tone with attack-decay envelope."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)

    # Mix of sine and harmonics for richer sound
    wave = (0.6 * np.sin(2 * np.pi * freq * t) +
            0.25 * np.sin(2 * np.pi * freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * freq * 3 * t))

    # ADSR-like envelope: quick attack, gradual decay
    attack = np.minimum(t / 0.05, 1.0)  # 50ms attack
    decay = np.exp(-t * 1.5)
    envelope = attack * decay

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
