# MoodTune - Emotion-Powered Music Generation & Recommendation

## What This Is
Deep Learning final project (UAB Spring 2026). Webcam captures facial expression → classifies emotion with fine-tuned EfficientNet-B0 → LSTM generates a melody matching mood → recommends songs to improve mood. Streamlit web app. Two trained neural networks.

## Quick Start
```bash
pip install -r requirements.txt
# Download FER2013 (image-folder format) from Kaggle into data/fer2013/
python3 training/train_fer.py                     # Train emotion model (~30 epochs)
python3 training/train_music.py --generate-synthetic  # Train LSTM melody generator
streamlit run app.py                              # Launch web app
```

## Project Structure
```
├── app.py                      # Streamlit web app (main entry point)
├── config.py                   # ALL constants: labels, hyperparams, music params, LSTM params, mood mappings
├── requirements.txt            # torch, torchvision, opencv, streamlit, pandas, scipy, numpy, sklearn, matplotlib, mido, seaborn
├── .gitignore
├── data/
│   ├── fer2013/                # FER2013 dataset - train/ and test/ subdirs (gitignored)
│   ├── midi/                   # Optional: MIDI files organized by emotion subdirs (gitignored)
│   └── songs.csv               # 50 curated songs with title, artist, genre, mood_tag, valence, energy, url
├── models/
│   ├── fer_model.py            # EmotionResNet (ResNet18) + EmotionEfficientNet (EfficientNet-B0)
│   │                           #   build_model(), load_trained_model(), predict_emotion()
│   ├── melody_lstm.py          # MelodyLSTM: 2-layer LSTM conditioned on emotion embedding
│   │                           #   forward(), generate() with temperature + scale_mask
│   └── music_generator.py      # generate_melody() → tries LSTM first, falls back to rule-based
│                               #   generate_melody_lstm(), generate_melody_rulebased()
├── training/
│   ├── train_fer.py            # FER pipeline: EfficientNet-B0, Mixup, label smoothing, CosineAnnealing
│   └── train_music.py          # LSTM pipeline: MIDI parsing, synthetic data generation, next-note prediction
├── utils/
│   ├── face_detection.py       # FaceDetector class: Haar cascade, detect_and_preprocess(), draw_bbox()
│   ├── emotion_music_map.py    # SCALES dict, get_scale_notes(), get_music_params()
│   ├── song_recommender.py     # recommend_songs(emotion, n) → filters by valence/energy for mood IMPROVEMENT
│   └── midi_utils.py           # melody_to_wav() → sine+harmonics synthesis with ADSR envelope → WAV BytesIO
├── checkpoints/                # fer_best.pth + melody_lstm.pth saved here (gitignored)
├── docs/
│   └── MoodTune_Report.md      # Full project report with techniques
└── notebooks/                  # EDA and experiment notebooks
```

## Two Trained Neural Networks

### 1. Emotion Classifier (EfficientNet-B0)
- **Architecture**: torchvision EfficientNet-B0, pretrained ImageNet, fine-tuned on FER2013
- **Input**: 96x96 grayscale (upscaled from 48x48) → repeated to 3ch
- **Output**: 7 emotion classes
- **Accuracy improvements over v1 (ResNet18 65-70%)**:
  - EfficientNet-B0 backbone (stronger features)
  - Label smoothing 0.1 (handles FER2013 label noise)
  - Mixup augmentation (50% probability, alpha=0.2)
  - RandomErasing augmentation
  - CosineAnnealingWarmRestarts scheduler
  - Gradient clipping (max_norm=1.0)
  - Weight decay 1e-4
  - Larger input 96x96
- **Expected accuracy**: 73-78%

### 2. Melody LSTM
- **Architecture**: 2-layer LSTM, note_embed(64) + emotion_embed(64) → LSTM(256) → FC(128)
- **Conditioning**: Emotion embedding concatenated with note embedding at every timestep
- **Training data**: Synthetic (rule-based patterns per emotion) or real MIDI files in data/midi/{emotion}/
- **Generation**: Autoregressive, temperature-controlled sampling with scale mask
- **Checkpoint**: checkpoints/melody_lstm.pth

## Key Architecture Decisions
- **FER model**: EfficientNet-B0 (upgraded from ResNet18). Grayscale repeated to 3ch. 7-class output.
- **Music generation**: LSTM (primary) with rule-based fallback. Emotion conditions the LSTM via embedding concatenation. Scale mask forces notes to stay in key.
- **Audio**: Direct sine-wave synthesis with harmonics (no MIDI/FluidSynth dependency). scipy.io.wavfile → BytesIO → st.audio().
- **Song recommendation**: Mood-IMPROVEMENT logic (sad→uplifting, angry→calming). Filters songs.csv by valence/energy.
- **Web app**: st.camera_input() for snapshots. Sidebar shows model status (FER ready / LSTM ready).

## Emotion Labels (FER2013 order)
`['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']`

## Config Location
ALL tunable values live in `config.py`: paths, FER hyperparams, LSTM hyperparams, emotion-music params, mood-improvement map. Check there first.

## Training Details

### FER Training (train_fer.py)
- Dataset: FER2013 image-folder format, 96x96 upscaled
- Augmentation: HFlip, Rotation(15), Affine(translate=0.1, scale=0.9-1.1), ColorJitter(0.3, 0.2), RandomErasing(p=0.25), Mixup(alpha=0.2)
- Loss: CrossEntropyLoss with class weights + label_smoothing=0.1
- Optimizer: Adam lr=1e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
- Early stopping: patience=7
- Device: auto-detects cuda > mps > cpu

### LSTM Training (train_music.py)
- Data: MIDI files (mido parser) OR synthetic generation (--generate-synthetic flag)
- Synthetic: 700 sequences, emotion-specific melodic patterns (directional bias per emotion)
- Loss: CrossEntropyLoss(ignore_index=0) for next-note prediction
- Optimizer: Adam lr=1e-3, weight_decay=1e-5
- Scheduler: ReduceLROnPlateau(patience=10)
- Sequence length: 64, batch size: 32, 100 epochs max
- Sliding window with 50% overlap for training samples

## Dependencies (Python 3.11+ recommended)
torch, torchvision, numpy, opencv-python, Pillow, streamlit, pandas, matplotlib, scikit-learn, scipy, seaborn, mido
