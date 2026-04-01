# MoodTune: Emotion-Powered Music Generation and Recommendation

**Course:** Deep Learning — UAB Spring 2026
**Project Type:** Final Project

---

## Abstract

MoodTune is a multimodal deep learning system that bridges computer vision and computational music. The application captures a user's facial expression through a webcam, classifies the emotion using a fine-tuned ResNet-18 convolutional neural network, generates a short original melody that reflects the detected mood, and recommends existing songs designed to improve the user's emotional state. The system integrates transfer learning for facial expression recognition, rule-based procedural music generation grounded in music theory, harmonic audio synthesis using digital signal processing, and content-based filtering for song recommendation — all delivered through an interactive Streamlit web application.

---

## 1. Introduction

### 1.1 Problem Statement

Emotional well-being is closely tied to music. Research in music psychology shows that certain musical properties — tempo, mode (major vs. minor), dynamics — systematically correspond to emotional states. However, selecting music that matches or improves one's mood typically requires conscious effort and self-awareness.

This project asks: **can we automate the process of detecting a person's emotional state from their face and responding with appropriate music — both generated and curated — to support their well-being?**

### 1.2 Motivation

- **Facial expressions** are one of the most direct, universally recognized signals of human emotion, making them an ideal input for an emotion-aware system.
- **Deep learning** has made significant advances in facial expression recognition (FER), achieving human-level performance on benchmark datasets.
- **Music generation** from emotional parameters demonstrates how learned perception can drive creative output.
- The combination of **perception (vision) + generation (audio) + recommendation** showcases a full multimodal AI pipeline.

### 1.3 Goals

1. Train a CNN-based emotion classifier that achieves competitive accuracy on the FER2013 benchmark.
2. Map detected emotions to musically meaningful parameters (tempo, scale, dynamics).
3. Generate short, listenable melodies that audibly differ across emotional categories.
4. Recommend existing songs that would shift the user's mood in a positive direction.
5. Deliver all of the above through an interactive, browser-based interface.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
                          MoodTune System Pipeline
                          ========================

  WEBCAM INPUT                PERCEPTION                 GENERATION & OUTPUT
 +-----------+          +-------------------+          +---------------------+
 |           |          |                   |          |                     |
 |  Browser  |  image   |  Face Detection   |  face   |  Melody Generator   |
 |  Webcam   +--------->+  (Haar Cascade)   +-------->+  (Rule-Based)       |
 |           |          |                   |  crop   |         |           |
 +-----------+          +--------+----------+         |         v           |
                                 |                    |  Audio Synthesis    |
                                 | 48x48              |  (Sine + Harmonics) |
                                 | grayscale          |         |           |
                                 v                    |         v           |
                        +-------------------+         |  WAV Audio Player   |
                        |                   |         |                     |
                        |  EmotionResNet    |         +---------------------+
                        |  (ResNet-18)      |
                        |  7-class output   |         +---------------------+
                        |                   |         |                     |
                        +--------+----------+         |  Song Recommender   |
                                 |                    |  (Content-Based     |
                                 | emotion            |   Filtering)        |
                                 | + confidence       |         |           |
                                 |                    |         v           |
                                 +------------------->+  Mood-Improvement   |
                                                      |  Song List          |
                                                      |                     |
                                                      +---------------------+
```

### 2.2 Component Summary

| Component | Technique | Input | Output |
|---|---|---|---|
| Face Detection | Haar Cascade Classifier | BGR webcam frame | Cropped face region + bounding box |
| Emotion Classification | Fine-tuned ResNet-18 | 48x48 grayscale tensor | 7-class probability distribution |
| Music Generation | Rule-based procedural generation | Emotion label | List of (MIDI note, duration, velocity) |
| Audio Synthesis | Harmonic sine-wave synthesis + ADSR | Note sequence + tempo | WAV audio buffer |
| Song Recommendation | Content-based filtering | Emotion label | Filtered song list (mood improvement) |
| Web Interface | Streamlit | All of the above | Interactive browser app |

---

## 3. Techniques and Methods

### 3.1 Facial Expression Recognition with Transfer Learning

#### 3.1.1 Transfer Learning

Transfer learning is a technique where a model trained on a large dataset (source task) is adapted to a different but related task (target task). Instead of training a deep CNN from scratch — which would require far more data and compute than available — we start from a model pretrained on ImageNet (1.2 million natural images, 1000 classes) and fine-tune it for facial expression recognition (35,887 images, 7 classes).

**Why it works:** The early layers of a CNN learn general visual features (edges, textures, shapes) that are useful across many vision tasks. Only the later, more task-specific layers need to be retrained for the new domain.

#### 3.1.2 ResNet-18 Architecture

We use **ResNet-18** (Residual Network with 18 layers) as the backbone. ResNet introduced **skip connections** (residual connections) that add the input of a block directly to its output:

```
output = F(x) + x
```

This solves the **vanishing gradient problem** in deep networks by providing a direct gradient path through the skip connection, enabling effective training of deeper architectures. ResNet-18 contains:

- 1 initial convolution + batch normalization + max pooling
- 4 residual layer groups (2 blocks each), with progressively more filters: 64 -> 128 -> 256 -> 512
- Global average pooling
- Fully connected classification layer

**Modifications for our task:**

1. **Input adaptation:** FER2013 images are 48x48 grayscale (1 channel), but ResNet-18 expects 224x224 RGB (3 channels). Rather than modifying the first convolutional layer (which would discard pretrained weights), we **repeat the grayscale channel 3 times** at inference: `x.repeat(1, 3, 1, 1)`. This preserves all pretrained conv1 weights since they still receive 3-channel input.

2. **Output layer replacement:** The original 1000-class ImageNet head is replaced with a new fully connected layer: `Linear(512, 7)` mapping to 7 emotion classes.

#### 3.1.3 Data Augmentation

To combat overfitting on the relatively small FER2013 training set (~28,000 images), we apply the following augmentations during training:

| Augmentation | Parameters | Purpose |
|---|---|---|
| Random Horizontal Flip | p=0.5 | Faces can be symmetric; doubles effective dataset |
| Random Rotation | +/-10 degrees | Handles slight head tilts |
| Random Affine | translate=10% | Handles face position variation |
| Color Jitter | brightness=0.2 | Handles lighting variation |

These transforms are applied randomly each epoch, so the model sees slightly different versions of each training image, improving generalization.

#### 3.1.4 Handling Class Imbalance

FER2013 is significantly **imbalanced** — the "disgust" class has far fewer samples than "happy" or "neutral." A naive model would bias toward majority classes.

**Solution: Inverse-frequency class weighting in the loss function.**

```
weight_c = N_total / (N_classes x N_c)
```

Where `N_c` is the number of samples in class `c`. Classes with fewer samples receive higher weights, penalizing the model more for misclassifying rare emotions. These weights are passed to `CrossEntropyLoss`.

#### 3.1.5 Cross-Entropy Loss

The standard loss function for multi-class classification:

```
L = -sum(y_c * log(p_c) * w_c)
```

Where `y_c` is the one-hot ground truth, `p_c` is the predicted probability (after softmax), and `w_c` is the class weight. It penalizes confident wrong predictions heavily due to the logarithm.

#### 3.1.6 Optimization: Adam with Learning Rate Scheduling

- **Adam optimizer** (Adaptive Moment Estimation) with learning rate = 1e-4. Adam combines momentum (tracks exponential moving average of gradients) and RMSProp (tracks exponential moving average of squared gradients) to adapt the learning rate per-parameter. The low initial LR is appropriate for fine-tuning — we want small updates to preserve pretrained features.

- **ReduceLROnPlateau scheduler**: Monitors validation accuracy. If no improvement for 3 consecutive epochs (patience=3), the learning rate is halved (factor=0.5). This allows the optimizer to make finer adjustments as training converges.

#### 3.1.7 Early Stopping

Training halts if validation accuracy does not improve for 5 consecutive epochs (patience=5). The model checkpoint with the best validation accuracy is saved and used for inference. This prevents overfitting by stopping before the model memorizes training noise.

#### 3.1.8 Hardware Acceleration

The training script auto-detects the best available device:
- **CUDA** (NVIDIA GPU) — fastest option
- **MPS** (Apple Metal Performance Shaders) — GPU acceleration on Apple Silicon Macs
- **CPU** — fallback

---

### 3.2 Face Detection with Haar Cascades

Before emotion classification, we must locate and extract the face from the webcam frame.

**Haar Cascade Classifier** (Viola-Jones, 2001) is a classical machine learning approach to object detection:

1. **Haar-like features**: Rectangular filters that compute the difference between pixel sums in adjacent regions, capturing edges, lines, and contrast patterns.
2. **Integral image**: A preprocessing step that allows any rectangular sum to be computed in O(1), making feature computation extremely fast.
3. **AdaBoost cascade**: A sequence of increasingly complex classifiers. Each stage quickly rejects non-face regions, so only candidate regions pass to the next stage. This gives near-real-time performance.

**Our configuration:**
- Cascade: `haarcascade_frontalface_default.xml` (OpenCV built-in)
- Scale factor: 1.3 (image is scaled down by 30% at each pyramid level)
- Min neighbors: 5 (requires 5 overlapping detections to confirm a face, reducing false positives)
- Selection: largest detected face (by area)

**Preprocessing pipeline after detection:**
1. Crop the face region from the grayscale frame
2. Resize to 48x48 pixels (FER2013 standard)
3. Convert to tensor
4. Normalize to mean=0.5, std=0.5
5. Add batch dimension: shape becomes (1, 1, 48, 48)

---

### 3.3 Emotion-to-Music Parameter Mapping

The bridge between perception and generation is a hand-crafted mapping grounded in **music psychology research**. Each of the 7 emotions maps to a set of musical parameters:

| Emotion | Tempo (BPM) | Scale | Octave | Note Density | Velocity |
|---|---|---|---|---|---|
| Happy | 120 | Major | 5 | 80% | 100 |
| Sad | 60 | Minor | 4 | 40% | 60 |
| Angry | 140 | Minor | 4 | 90% | 120 |
| Fearful | 90 | Minor | 4 | 60% | 50 |
| Surprised | 130 | Major | 5 | 70% | 90 |
| Neutral | 100 | Major | 4 | 50% | 80 |
| Disgusted | 80 | Minor | 3 | 50% | 70 |

**Rationale from music theory and psychology:**

- **Tempo**: Faster tempos (>120 BPM) feel energetic and agitated; slower tempos (<80 BPM) feel contemplative and melancholic.
- **Scale/Mode**: Major scales sound bright and happy; minor scales sound dark and sad. This is one of the most robust emotion-music associations.
- **Octave**: Higher pitches feel lighter and more cheerful; lower pitches feel heavier and more somber.
- **Note density**: Busy passages feel energetic or anxious; sparse passages feel calm or desolate.
- **Velocity (dynamics)**: Loud notes convey intensity (anger, excitement); soft notes convey gentleness (sadness, fear).

---

### 3.4 Procedural Melody Generation

The melody generator uses a **constrained random walk** algorithm to produce musically coherent sequences.

#### 3.4.1 Scale Construction

Musical scales are built as sets of MIDI note numbers. For example, C major starting at octave 5 (MIDI root = 60):

```
C Major = [60, 62, 64, 65, 67, 69, 71]  (C, D, E, F, G, A, B)
C Minor = [60, 62, 63, 65, 67, 68, 70]  (C, D, Eb, F, G, Ab, Bb)
```

Notes span 2 octaves to provide sufficient melodic range.

#### 3.4.2 Melody Algorithm

```
1. Start at the MIDDLE of the available scale notes
2. For each beat (total = 16 beats):
   a. Roll density check: random() < note_density?
      - YES: generate a note
      - NO:  insert a rest (silence)
   b. If generating a note:
      - Choose step direction (weighted random):
          -2 steps: 10%  (downward leap)
          -1 step:  30%  (stepwise down)  <-- PREFERRED
           0 steps: 20%  (repeat)
          +1 step:  30%  (stepwise up)    <-- PREFERRED
          +2 steps: 10%  (upward leap)
      - Move current position by step (clamped to scale bounds)
      - Pick the note at new position from the scale array
      - Choose duration: mostly quarter notes (1 beat),
        some eighth (0.5) and half (2) notes
      - Add velocity variation: base +/- 10 (random)
3. End on the TONIC (root note) with a long duration (2 beats)
```

**Key musical principles applied:**

- **Stepwise motion preference (60%)**: Real melodies predominantly move by small intervals. Large leaps are rare and dramatic.
- **Scale constraint**: Every generated note belongs to the target scale, preventing dissonant "wrong notes."
- **Tonic resolution**: Ending on the root note creates a sense of musical closure.
- **Rhythmic variation**: Mixing note durations creates rhythmic interest rather than monotony.

---

### 3.5 Digital Audio Synthesis

Generated melodies are converted directly to audio waveforms — no external MIDI synthesizer needed.

#### 3.5.1 MIDI-to-Frequency Conversion

The standard equal temperament formula:

```
f = 440 x 2^((n - 69) / 12)
```

Where `n` is the MIDI note number and 440 Hz is the reference pitch (A4, MIDI note 69). Each semitone is a factor of 2^(1/12) apart.

#### 3.5.2 Additive Harmonic Synthesis

A pure sine wave sounds thin and artificial. To create a richer, more natural tone, we mix the fundamental frequency with its first two **harmonics** (overtones):

```
signal(t) = 0.60 x sin(2pi x f x t)       -- fundamental
           + 0.25 x sin(2pi x 2f x t)      -- 2nd harmonic (octave)
           + 0.15 x sin(2pi x 3f x t)      -- 3rd harmonic (octave + fifth)
```

This is the principle behind **additive synthesis**, one of the foundational techniques in electronic music and sound design. The relative amplitudes of harmonics determine the **timbre** (tonal color) of the sound.

#### 3.5.3 ADSR Envelope

Raw synthesized tones that start and stop abruptly sound harsh and unnatural. An **envelope** shapes the amplitude over time:

```
Attack:  Linear ramp from 0 to 1 over 50ms
Decay:   Exponential decay: e^(-1.5t)
```

This creates a piano-like quality — the note begins quickly, then gradually fades. The velocity parameter further scales the envelope, controlling overall loudness.

#### 3.5.4 Audio Pipeline

1. For each note in the melody, generate the waveform (or silence for rests)
2. Concatenate all waveforms sequentially
3. **Peak normalize** to 80% amplitude (prevents clipping)
4. Quantize from float64 to int16 (16-bit PCM, CD standard)
5. Write to an in-memory WAV buffer at 44,100 Hz sample rate
6. The buffer is passed directly to the Streamlit audio player

---

### 3.6 Content-Based Song Recommendation

#### 3.6.1 Mood Improvement Logic

Unlike typical recommendation systems that match content to preference, MoodTune uses a **mood improvement** strategy: it recommends music that would shift the user's emotional state in a positive direction.

| Detected Emotion | Target Mood | Valence Range | Energy Range |
|---|---|---|---|
| Sad | Uplifting | 0.6 - 1.0 | 0.4 - 0.7 |
| Angry | Calming | 0.5 - 0.8 | 0.2 - 0.5 |
| Fearful | Reassuring | 0.6 - 0.9 | 0.3 - 0.6 |
| Disgusted | Refreshing | 0.5 - 0.8 | 0.4 - 0.7 |
| Happy | Keep Vibing | 0.7 - 1.0 | 0.6 - 1.0 |
| Surprised | Grounding | 0.5 - 0.8 | 0.4 - 0.7 |
| Neutral | Energizing | 0.5 - 0.8 | 0.4 - 0.7 |

#### 3.6.2 Audio Features

Songs are characterized by two key features (derived from Spotify's audio analysis):

- **Valence (0.0 - 1.0)**: Musical positiveness. High valence = happy, cheerful, euphoric. Low valence = sad, depressed, angry.
- **Energy (0.0 - 1.0)**: Perceptual intensity and activity. High energy = fast, loud, noisy. Low energy = soft, mellow, ambient.

#### 3.6.3 Filtering Algorithm

```
1. Look up the target valence and energy ranges for the detected emotion
2. Filter the song dataset: valence IN target range AND energy IN target range
3. If fewer than N songs match:
   - Relax both ranges by +/- 0.1 and re-filter
4. If still insufficient:
   - Fall back to random selection from the full catalog
5. Randomly sample N songs from the filtered set
6. Return song metadata (title, artist, genre, Spotify URL)
```

This is a **content-based filtering** approach — recommendations are based on the attributes (features) of the items, not on user history or collaborative signals.

---

## 4. Dataset

### 4.1 FER2013 (Facial Expression Recognition)

- **Source**: Kaggle (image-folder format)
- **Size**: 35,887 labeled images
- **Resolution**: 48 x 48 pixels, grayscale
- **Classes**: 7 emotions — angry, disgusted, fearful, happy, sad, surprised, neutral
- **Split**: Separate `train/` and `test/` directories
- **Known limitations**: Label noise (some images are ambiguous or mislabeled), low resolution, class imbalance (notably few "disgust" samples)
- **Expected accuracy ceiling**: ~75% (state-of-the-art); our target: 65-70%

### 4.2 Song Recommendation Dataset

- **File**: `data/songs.csv`
- **Size**: 50 curated songs across multiple genres
- **Columns**: title, artist, genre, mood_tag, valence, energy, url
- **Genres represented**: Pop, Rock, Soul, Folk, Classical, Ambient, Hip-Hop, Jazz, Reggae, Indie, Funk, Alternative
- **Mood tags**: happy, calm, uplifting, energetic, reassuring, melancholic

---

## 5. Results

### 5.1 Emotion Classification

- **Expected test accuracy**: 65-70% on FER2013
- **Evaluation metrics**: Confusion matrix, per-class precision/recall/F1, overall accuracy
- **Common confusions**: Fear vs. Surprise (similar facial features), Sad vs. Neutral (subtle differences), Angry vs. Disgusted (overlapping expressions)

### 5.2 Music Generation

Each emotion produces audibly distinct melodies:
- **Happy**: Fast (120 BPM), bright major scale, frequent notes, moderate-loud dynamics
- **Sad**: Slow (60 BPM), dark minor scale, sparse notes with rests, soft dynamics
- **Angry**: Very fast (140 BPM), dense minor-scale patterns, loud dynamics
- **Neutral**: Moderate tempo (100 BPM), balanced major scale, medium density

### 5.3 Song Recommendations

The mood-improvement logic correctly maps negative emotions to positive musical targets (e.g., sadness triggers uplifting recommendations with high valence), while positive emotions receive reinforcing recommendations.

---

## 6. Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Deep Learning | PyTorch + torchvision | >= 2.0 |
| Computer Vision | OpenCV | >= 4.8 |
| Web Framework | Streamlit | >= 1.28 |
| Audio Processing | SciPy + NumPy | >= 1.11 / >= 1.24 |
| Data Processing | Pandas | >= 2.0 |
| Visualization | Matplotlib | >= 3.7 |
| Evaluation | scikit-learn | >= 1.3 |
| Image Processing | Pillow | >= 10.0 |
| Language | Python | 3.11+ recommended |

---

## 7. How to Run

### 7.1 Installation

```bash
cd "final project"
pip install -r requirements.txt
```

### 7.2 Dataset Setup

Download FER2013 in image-folder format from Kaggle:
```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/
```

The expected structure:
```
data/fer2013/
  train/
    angry/
    disgusted/
    fearful/
    happy/
    sad/
    surprised/
    neutral/
  test/
    (same subdirectories)
```

### 7.3 Training

```bash
python3 training/train_fer.py
```

This will:
- Load and augment the FER2013 dataset
- Fine-tune ResNet-18 for up to 25 epochs
- Auto-detect GPU (CUDA/MPS) or fall back to CPU
- Save the best model to `checkpoints/fer_resnet18_best.pth`
- Generate training curves and confusion matrix plots

### 7.4 Running the App

```bash
streamlit run app.py
```

Open the browser URL shown in the terminal. The app provides:
1. Webcam capture button
2. Real-time emotion detection with confidence scores
3. Generated melody audio player
4. Song recommendations with Spotify links

---

## 8. References

1. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. — ResNet architecture with skip connections.

2. **Goodfellow, I. J., et al.** (2013). Challenges in Representation Learning: A Report on Three Machine Learning Contests. *Neural Information Processing*. — Original FER2013 dataset and challenge.

3. **Viola, P., & Jones, M.** (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. *CVPR 2001*. — Haar Cascade face detection.

4. **Kingma, D. P., & Ba, J.** (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. — Adam optimizer.

5. **Juslin, P. N., & Sloboda, J.** (2010). *Handbook of Music and Emotion: Theory, Research, Applications*. Oxford University Press. — Music-emotion associations (tempo, mode, dynamics).

6. **Deng, J., et al.** (2009). ImageNet: A Large-Scale Hierarchical Image Database. *CVPR 2009*. — ImageNet dataset used for pretraining.

7. **Yosinski, J., et al.** (2014). How Transferable Are Features in Deep Neural Networks? *NeurIPS 2014*. — Transfer learning theory and layer-by-layer feature reuse.
