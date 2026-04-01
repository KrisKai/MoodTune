# MoodTune - Emotion-Powered Music Generation & Recommendation

## What This Is
Deep Learning final project (UAB Spring 2026). Webcam captures facial expression → classifies emotion with fine-tuned ResNet18 → generates a melody matching mood → recommends songs to improve mood. Streamlit web app.

## Quick Start
```bash
pip install -r requirements.txt
# Download FER2013 (image-folder format) from Kaggle into data/fer2013/
python3 training/train_fer.py   # Train emotion model (~25 epochs, auto-detects MPS)
streamlit run app.py            # Launch web app
```

## Project Structure
```
├── app.py                      # Streamlit web app (main entry point)
├── config.py                   # ALL constants: labels, hyperparams, music params, mood mappings
├── requirements.txt            # torch, torchvision, opencv, streamlit, pandas, scipy, numpy, sklearn, matplotlib
├── .gitignore
├── data/
│   ├── fer2013/                # FER2013 dataset - train/ and test/ subdirs (gitignored)
│   └── songs.csv               # 50 curated songs with title, artist, genre, mood_tag, valence, energy, url
├── models/
│   ├── fer_model.py            # EmotionResNet (ResNet18, 1ch grayscale→3ch repeat, 7-class output)
│   │                           #   load_trained_model(), predict_emotion() helpers
│   └── music_generator.py      # generate_melody(emotion) → list of (midi_note, duration, velocity) + params
│                               #   Rule-based: stepwise motion, scale-constrained, tonic resolution
├── training/
│   ├── train_fer.py            # Full pipeline: ImageFolder loader, augmentation, Adam+ReduceLROnPlateau,
│   │                           #   CrossEntropyLoss with class weights, early stopping, confusion matrix
│   └── train_music.py          # (placeholder for optional LSTM melody training)
├── utils/
│   ├── face_detection.py       # FaceDetector class: Haar cascade, detect_and_preprocess(), draw_bbox()
│   ├── emotion_music_map.py    # SCALES dict, get_scale_notes(), get_music_params()
│   ├── song_recommender.py     # recommend_songs(emotion, n) → filters by valence/energy for mood IMPROVEMENT
│   └── midi_utils.py           # melody_to_wav() → sine+harmonics synthesis with ADSR envelope → WAV BytesIO
├── checkpoints/                # fer_resnet18_best.pth saved here (gitignored)
└── notebooks/                  # EDA and experiment notebooks
```

## Key Architecture Decisions
- **FER model**: Fine-tuned torchvision ResNet18. Grayscale input repeated to 3 channels (preserves pretrained conv1 weights). Output: 7 classes matching FER2013 labels.
- **Audio**: Direct sine-wave synthesis with harmonics (no MIDI/FluidSynth dependency). `scipy.io.wavfile` → BytesIO → `st.audio()`.
- **Music generation**: Rule-based (not LSTM). Emotion → musical params (tempo, scale, octave, density, velocity) defined in `config.py:EMOTION_MUSIC_PARAMS`. Melody constrained to scale notes with stepwise motion preference.
- **Song recommendation**: Mood-IMPROVEMENT logic (sad→uplifting, angry→calming). Mapping in `config.py:MOOD_IMPROVEMENT_MAP`. Filters `songs.csv` by target valence/energy ranges.
- **Web app**: `st.camera_input()` for snapshots (not streamlit-webrtc). Sidebar has threshold/count sliders.

## Emotion Labels (FER2013 order)
`['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']`

## Config Location
ALL tunable values live in `config.py`: paths, hyperparams, emotion-music params, mood-improvement map. Check there first.

## Training Details
- Dataset: FER2013 image-folder format (`data/fer2013/train/`, `data/fer2013/test/`)
- Augmentation: HFlip, Rotation(10), Affine(translate=0.1), ColorJitter(brightness=0.2)
- Optimizer: Adam lr=1e-4, scheduler ReduceLROnPlateau(patience=3)
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Early stopping: patience=5, saves best model to `checkpoints/fer_resnet18_best.pth`
- Device: auto-detects cuda > mps > cpu
- Expected accuracy: 65-70%

## Dependencies (Python 3.11+ recommended, 3.14 may have compatibility issues)
torch, torchvision, numpy, opencv-python, Pillow, streamlit, pandas, matplotlib, scikit-learn, scipy
