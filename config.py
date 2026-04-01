import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Emotion labels (FER2013 order)
EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
NUM_CLASSES = 7

# FER model
IMG_SIZE = 48
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
EARLY_STOP_PATIENCE = 5
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'fer_resnet18_best.pth')

# Data paths
FER_DATA_DIR = os.path.join(BASE_DIR, 'data', 'fer2013')
SONGS_CSV = os.path.join(BASE_DIR, 'data', 'songs.csv')

# Music generation
SAMPLE_RATE = 44100
MELODY_BEATS = 16

# Emotion to musical parameters
EMOTION_MUSIC_PARAMS = {
    'happy':     {'tempo': 120, 'scale': 'major',      'octave': 5, 'note_density': 0.8, 'velocity': 100},
    'sad':       {'tempo': 60,  'scale': 'minor',      'octave': 4, 'note_density': 0.4, 'velocity': 60},
    'angry':     {'tempo': 140, 'scale': 'minor',      'octave': 4, 'note_density': 0.9, 'velocity': 120},
    'fearful':   {'tempo': 90,  'scale': 'minor',      'octave': 4, 'note_density': 0.6, 'velocity': 50},
    'surprised': {'tempo': 130, 'scale': 'major',      'octave': 5, 'note_density': 0.7, 'velocity': 90},
    'neutral':   {'tempo': 100, 'scale': 'major',      'octave': 4, 'note_density': 0.5, 'velocity': 80},
    'disgusted': {'tempo': 80,  'scale': 'minor',      'octave': 3, 'note_density': 0.5, 'velocity': 70},
}

# Mood improvement mapping for song recommendations
MOOD_IMPROVEMENT_MAP = {
    'sad':       {'target_valence': (0.6, 1.0), 'target_energy': (0.4, 0.7), 'label': 'uplifting'},
    'angry':     {'target_valence': (0.5, 0.8), 'target_energy': (0.2, 0.5), 'label': 'calming'},
    'fearful':   {'target_valence': (0.6, 0.9), 'target_energy': (0.3, 0.6), 'label': 'reassuring'},
    'disgusted': {'target_valence': (0.5, 0.8), 'target_energy': (0.4, 0.7), 'label': 'refreshing'},
    'happy':     {'target_valence': (0.7, 1.0), 'target_energy': (0.6, 1.0), 'label': 'keep vibing'},
    'surprised': {'target_valence': (0.5, 0.8), 'target_energy': (0.4, 0.7), 'label': 'grounding'},
    'neutral':   {'target_valence': (0.5, 0.8), 'target_energy': (0.4, 0.7), 'label': 'energizing'},
}
