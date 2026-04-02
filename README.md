# MoodTune

**Emotion-Powered Music Generation & Recommendation**

A deep learning web app that captures your facial expression via webcam, detects your emotion using a fine-tuned EfficientNet-B0, generates an original melody with an emotion-conditioned LSTM, and recommends songs to improve how you feel. Two trained neural networks, fully local, no API calls.

> Deep Learning Final Project — UAB Spring 2026

---

## Pipeline

```
📸 Webcam Capture  →  🧠 EfficientNet-B0  →  🎶 LSTM Melody  →  🎧 Song Recommendations
```

1. **Capture** — Take a photo with your webcam
2. **Detect** — EfficientNet-B0 classifies your expression into one of 7 emotions
3. **Generate** — Emotion-conditioned LSTM generates an original melody (scale-locked, tempo-matched)
4. **Recommend** — Songs are suggested to *improve* your mood (sad → uplifting, angry → calming)

---

## Why Train Locally vs. Calling an API?

| | Our Approach (Local Training) | API-Based (e.g. DeepFace, AWS Rekognition) |
|---|---|---|
| **FER Accuracy** | ~73-78% on FER2013 | ~75-80% (cloud models trained on larger private datasets) |
| **Music Generation** | Trained LSTM — learned emotion-specific patterns | Would need MusicGen API / OpenAI Jukebox — black box |
| **Latency** | Instant (on-device inference) | 200-500ms+ per API call (network dependent) |
| **Cost** | Free (your GPU/CPU) | $0.001-0.01+ per image/request |
| **Privacy** | Photos never leave your machine | Facial data sent to third-party servers |
| **Learning Value** | Full understanding of model architecture, training, and tuning | Just calling endpoints — no DL knowledge demonstrated |
| **Offline** | Works without internet | Requires internet connection |
| **Customizable** | Can fine-tune for specific use cases, swap backbones, adjust augmentation | Limited to what the API exposes |

### Accuracy Breakdown

| Model | Expected Accuracy | Notes |
|---|---|---|
| **Our EfficientNet-B0** (v2) | **~73-78%** | Fine-tuned on FER2013, label smoothing, Mixup, 96x96 input |
| Our ResNet-18 (v1) | ~65-70% | Baseline fine-tune, basic augmentation, 48x48 input |
| DeepFace (API/library) | ~75-80% | Ensemble of multiple models, larger training data |
| AWS Rekognition | ~80-85% | Proprietary model, trained on millions of faces |
| Human agreement on FER2013 | ~65±5% | FER2013 labels are noisy — humans disagree too |

**Key insight:** Our model is within 2-5% of cloud APIs on FER2013 — and FER2013 itself has ~65% human agreement due to label noise, so the accuracy ceiling is inherently limited. The improvements from v1→v2 are significant:

| Improvement | Technique | Estimated Gain |
|---|---|---|
| Stronger backbone | EfficientNet-B0 (vs ResNet-18) | +3-4% |
| Label smoothing | Reduces overconfidence on noisy labels | +1-2% |
| Mixup augmentation | Blends training samples for regularization | +1-2% |
| Larger input | 96x96 (vs 48x48) | +1-2% |
| RandomErasing | Occlusion robustness | +0.5-1% |
| CosineAnnealing | Better LR schedule than plateau-based | +0.5-1% |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FER2013 Dataset

Download the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) (image-folder format) from Kaggle:

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

### 3. Train Both Models

```bash
# Train emotion classifier (~30 epochs, auto-detects CUDA/MPS/CPU)
python3 training/train_fer.py

# Train LSTM melody generator (no MIDI files needed — uses synthetic data)
python3 training/train_music.py --generate-synthetic
```

Or if you have your own MIDI files organized by emotion:
```bash
# Place .mid files in data/midi/happy/, data/midi/sad/, etc.
python3 training/train_music.py
```

### 4. Launch the App

```bash
streamlit run app.py
```

---

## Project Structure

```
MoodTune/
├── app.py                  # Streamlit web app (main entry point)
├── config.py               # All constants: FER params, LSTM params, music mappings
├── requirements.txt
├── data/
│   ├── fer2013/            # FER2013 dataset (download from Kaggle)
│   ├── midi/               # Optional: MIDI files by emotion for LSTM training
│   └── songs.csv           # 50 curated songs with valence, energy, Spotify links
├── models/
│   ├── fer_model.py        # EmotionEfficientNet + EmotionResNet (configurable backbone)
│   ├── melody_lstm.py      # MelodyLSTM — 2-layer LSTM conditioned on emotion embedding
│   └── music_generator.py  # generate_melody() — LSTM primary, rule-based fallback
├── training/
│   ├── train_fer.py        # FER pipeline: Mixup, label smoothing, CosineAnnealing
│   └── train_music.py      # LSTM pipeline: MIDI parsing, synthetic data, next-note prediction
├── utils/
│   ├── face_detection.py   # Haar Cascade face detection + preprocessing
│   ├── emotion_music_map.py# Emotion → musical parameters mapping
│   ├── song_recommender.py # Content-based filtering with mood-improvement logic
│   └── midi_utils.py       # Sine-wave + harmonics synthesis with ADSR envelope
├── checkpoints/            # fer_best.pth + melody_lstm.pth (gitignored)
├── docs/
│   └── MoodTune_Report.md  # Full project report with techniques
└── notebooks/              # EDA and experiment notebooks
```

---

## Two Trained Neural Networks

### 1. Emotion Classifier — EfficientNet-B0

| Detail | Value |
|---|---|
| Architecture | EfficientNet-B0 (pretrained ImageNet) |
| Input | 96x96 grayscale → 3ch |
| Output | 7 emotions |
| Loss | CrossEntropyLoss + class weights + label smoothing (0.1) |
| Augmentation | HFlip, Rotation(15), Affine, ColorJitter, RandomErasing, Mixup |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts |
| Expected Accuracy | 73-78% |

### 2. Melody Generator — Emotion-Conditioned LSTM

| Detail | Value |
|---|---|
| Architecture | 2-layer LSTM (hidden=256, embed=64) |
| Conditioning | Emotion embedding concatenated with note embedding per timestep |
| Training Data | Synthetic emotion-specific patterns (700 sequences) or real MIDI |
| Output | Autoregressive note sequence, scale-masked, temperature-sampled |
| Synthesis | Sine + harmonics with ADSR envelope → 44.1kHz WAV |

---

## Techniques Used

### Facial Expression Recognition
- **Transfer Learning** — EfficientNet-B0 pretrained on ImageNet, fine-tuned on FER2013
- **Label Smoothing (0.1)** — Prevents overconfidence on FER2013's noisy labels
- **Mixup Augmentation** — Blends pairs of images and labels for better generalization
- **RandomErasing** — Simulates occlusion for robustness
- **Class Imbalance Handling** — Inverse-frequency weighted loss
- **Cosine Annealing** — Warm restarts LR schedule for better convergence
- **Gradient Clipping** — Stabilizes training
- **Face Detection** — OpenCV Haar Cascade classifier

### Music Generation
- **Emotion-Conditioned LSTM** — Learns melodic patterns specific to each emotion
- **Scale Masking** — Generated notes constrained to musically valid scales
- **Temperature Sampling** — Controls randomness vs. coherence of melodies
- **Emotion-to-Music Mapping** — Tempo, scale (major/minor), octave, dynamics per emotion
- **Rule-Based Fallback** — Works even without trained LSTM
- **Harmonic Synthesis** — Fundamental + 2 overtones with ADSR envelope

### Song Recommendation
- **Mood Improvement Logic** — Recommends songs to shift mood positively (not match it)
- **Content-Based Filtering** — Filters by Spotify audio features (valence, energy)

---

## Emotion-to-Music Mapping

| Emotion | Tempo | Scale | Melody Style | Song Recommendation |
|---|---|---|---|---|
| Happy | 120 BPM | Major | Bright, energetic, upward motion | Keep vibing |
| Sad | 60 BPM | Minor | Slow, sparse, descending | Uplifting tracks |
| Angry | 140 BPM | Minor | Dense, loud, erratic leaps | Calming tracks |
| Fearful | 90 BPM | Minor | Soft, narrow range, descending | Reassuring tracks |
| Surprised | 130 BPM | Major | Fast, moderate dynamics | Grounding tracks |
| Neutral | 100 BPM | Major | Balanced, moderate | Energizing tracks |
| Disgusted | 80 BPM | Minor | Low octave, moderate density | Refreshing tracks |

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Emotion Model | EfficientNet-B0 (fine-tuned) |
| Music Model | LSTM (emotion-conditioned) |
| Computer Vision | OpenCV (Haar Cascades) |
| Audio Synthesis | NumPy, SciPy (sine + harmonics) |
| MIDI Parsing | mido |
| Web App | Streamlit |
| Data | Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## Requirements

- Python 3.11+ recommended
- See [requirements.txt](requirements.txt) for all dependencies
- GPU recommended for training (CUDA or Apple MPS), CPU works but slower

---

## Documentation

Full project report with detailed technique explanations: [docs/MoodTune_Report.md](docs/MoodTune_Report.md)

---

## License

This project was created for educational purposes as part of a university course.
