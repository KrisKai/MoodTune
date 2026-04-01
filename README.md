# MoodTune

**Emotion-Powered Music Generation & Recommendation**

A deep learning web app that captures your facial expression via webcam, detects your emotion, generates a melody matching your mood, and recommends songs to improve how you feel.

> Deep Learning Final Project — UAB Spring 2026

---

## Demo

```
📸 Webcam Capture  →  🧠 Emotion Detection  →  🎶 Melody Generation  →  🎧 Song Recommendations
```

1. **Capture** — Take a photo with your webcam
2. **Detect** — AI classifies your expression into one of 7 emotions
3. **Generate** — A unique melody is created based on your mood (tempo, scale, dynamics)
4. **Recommend** — Songs are suggested to *improve* your mood (sad → uplifting, angry → calming)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FER2013 Dataset

Download the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) (image-folder format) from Kaggle and place it in `data/fer2013/`:

```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/
```

Expected structure:
```
data/fer2013/
  train/
    angry/ disgusted/ fearful/ happy/ sad/ surprised/ neutral/
  test/
    angry/ disgusted/ fearful/ happy/ sad/ surprised/ neutral/
```

### 3. Train the Emotion Model

```bash
python3 training/train_fer.py
```

- Auto-detects GPU (CUDA / Apple MPS) or falls back to CPU
- Trains for ~25 epochs with early stopping
- Saves best model to `checkpoints/fer_resnet18_best.pth`
- Expected accuracy: 65-70%

### 4. Launch the App

```bash
streamlit run app.py
```

---

## Project Structure

```
MoodTune/
├── app.py                  # Streamlit web app (main entry point)
├── config.py               # All constants: labels, hyperparams, music params, mood mappings
├── requirements.txt        # Python dependencies
├── data/
│   ├── fer2013/            # FER2013 dataset (not included, download from Kaggle)
│   └── songs.csv           # 50 curated songs with valence, energy, and Spotify links
├── models/
│   ├── fer_model.py        # EmotionResNet — fine-tuned ResNet18 (7-class emotion classifier)
│   └── music_generator.py  # Rule-based melody generator (scale-constrained, stepwise motion)
├── training/
│   └── train_fer.py        # Full training pipeline with augmentation, class weights, early stopping
├── utils/
│   ├── face_detection.py   # Haar Cascade face detection + preprocessing
│   ├── emotion_music_map.py# Emotion → musical parameters (tempo, scale, octave, dynamics)
│   ├── song_recommender.py # Content-based filtering with mood-improvement logic
│   └── midi_utils.py       # Sine-wave synthesis with harmonics and ADSR envelope
├── checkpoints/            # Trained model weights (gitignored)
├── docs/
│   └── MoodTune_Report.md  # Full project report with all techniques documented
└── notebooks/              # EDA and experiment notebooks
```

---

## Techniques Used

### Facial Expression Recognition
- **Transfer Learning** — ResNet-18 pretrained on ImageNet, fine-tuned on FER2013
- **Data Augmentation** — Horizontal flip, rotation, affine transforms, brightness jitter
- **Class Imbalance Handling** — Inverse-frequency weighted CrossEntropyLoss
- **Optimization** — Adam (lr=1e-4) + ReduceLROnPlateau + early stopping
- **Face Detection** — OpenCV Haar Cascade classifier

### Music Generation
- **Emotion-to-Music Mapping** — Each emotion maps to tempo, scale (major/minor), octave, note density, and velocity based on music psychology research
- **Procedural Melody** — Constrained random walk with stepwise motion preference, scale-locked notes, tonic resolution
- **Harmonic Synthesis** — Fundamental + 2 overtones with ADSR envelope, rendered as 44.1kHz WAV

### Song Recommendation
- **Mood Improvement Logic** — Recommends songs to shift mood positively (not match it)
- **Content-Based Filtering** — Filters by Spotify audio features (valence, energy)

---

## Emotion Labels

| Emotion | Melody Style | Recommendation |
|---|---|---|
| Happy | Fast major, bright, energetic | Keep vibing |
| Sad | Slow minor, sparse, soft | Uplifting tracks |
| Angry | Fast minor, dense, loud | Calming tracks |
| Fearful | Moderate minor, soft | Reassuring tracks |
| Surprised | Fast major, moderate | Grounding tracks |
| Neutral | Moderate major, balanced | Energizing tracks |
| Disgusted | Slow minor, low octave | Refreshing tracks |

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Computer Vision | OpenCV (Haar Cascades) |
| Audio Synthesis | NumPy, SciPy |
| Web App | Streamlit |
| Data | Pandas, scikit-learn |
| Visualization | Matplotlib |

---

## Requirements

- Python 3.11+ recommended
- See [requirements.txt](requirements.txt) for all dependencies

---

## Documentation

Full project report with detailed technique explanations: [docs/MoodTune_Report.md](docs/MoodTune_Report.md)

---

## License

This project was created for educational purposes as part of a university course.
