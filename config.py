import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Emotion labels (FER2013 order)
EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
NUM_CLASSES = 7

# ── FER Model ──
IMG_SIZE = 224                  # EfficientNet-B0 native resolution for max accuracy
BATCH_SIZE = 32                 # Smaller batch for 224x224 (more GPU memory per image)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING = 0.1           # Reduces overconfidence on noisy FER2013 labels
FER_BACKBONE = 'efficientnet_b0'  # Options: 'resnet18', 'efficientnet_b0'
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'fer_best.pth')

# ── Data Paths ──
FER_DATA_DIR = os.path.join(BASE_DIR, 'data', 'fer2013')
SONGS_CSV = os.path.join(BASE_DIR, 'data', 'songs.csv')
MIDI_DATA_DIR = os.path.join(BASE_DIR, 'data', 'midi')

# ── Music Generation (LSTM) ──
SAMPLE_RATE = 44100
MELODY_BEATS = 300              # ~3 minutes at 100 BPM
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_EMBED_DIM = 64
LSTM_SEQ_LEN = 64               # Sequence length for training
LSTM_EPOCHS = 100
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 32
MIDI_VOCAB_SIZE = 128            # MIDI note range 0-127
LSTM_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'melody_lstm.pth')

# ── Emotion to Musical Parameters ──
EMOTION_MUSIC_PARAMS = {
    'happy':     {'tempo': 120, 'scale': 'major',      'octave': 5, 'note_density': 0.8, 'velocity': 100},
    'sad':       {'tempo': 60,  'scale': 'minor',      'octave': 4, 'note_density': 0.4, 'velocity': 60},
    'angry':     {'tempo': 140, 'scale': 'minor',      'octave': 4, 'note_density': 0.9, 'velocity': 120},
    'fearful':   {'tempo': 90,  'scale': 'minor',      'octave': 4, 'note_density': 0.6, 'velocity': 50},
    'surprised': {'tempo': 130, 'scale': 'major',      'octave': 5, 'note_density': 0.7, 'velocity': 90},
    'neutral':   {'tempo': 100, 'scale': 'major',      'octave': 4, 'note_density': 0.5, 'velocity': 80},
    'disgusted': {'tempo': 80,  'scale': 'minor',      'octave': 3, 'note_density': 0.5, 'velocity': 70},
}

# ── Mood Improvement Mapping for Song Recommendations ──
MOOD_IMPROVEMENT_MAP = {
    'sad':       {'target_valence': (0.6, 1.0), 'target_energy': (0.4, 0.7), 'label': 'uplifting'},
    'angry':     {'target_valence': (0.5, 0.8), 'target_energy': (0.2, 0.5), 'label': 'calming'},
    'fearful':   {'target_valence': (0.6, 0.9), 'target_energy': (0.3, 0.6), 'label': 'reassuring'},
    'disgusted': {'target_valence': (0.5, 0.8), 'target_energy': (0.4, 0.7), 'label': 'refreshing'},
    'happy':     {'target_valence': (0.7, 1.0), 'target_energy': (0.6, 1.0), 'label': 'keep vibing'},
    'surprised': {'target_valence': (0.5, 0.8), 'target_energy': (0.4, 0.7), 'label': 'grounding'},
    'neutral':   {'target_valence': (0.5, 0.8), 'target_energy': (0.4, 0.7), 'label': 'energizing'},
}
