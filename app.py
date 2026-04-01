import sys
import os
import numpy as np
import cv2
import torch
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMOTION_LABELS, MODEL_PATH
from models.fer_model import EmotionResNet, predict_emotion
from models.music_generator import generate_melody
from utils.face_detection import FaceDetector
from utils.midi_utils import melody_to_wav
from utils.song_recommender import recommend_songs

# --- Page Config ---
st.set_page_config(
    page_title="MoodTune",
    page_icon="🎵",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("MoodTune")
    st.markdown("*Emotion-powered music generation & recommendation*")
    st.divider()
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.2, 0.05)
    num_recommendations = st.slider("Number of song recommendations", 1, 10, 5)
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Take a photo with your webcam\n"
        "2. AI detects your facial expression\n"
        "3. A melody is generated matching your mood\n"
        "4. Songs are recommended to improve your mood"
    )
    st.divider()
    st.caption("Deep Learning Final Project - UAB Spring 2026")


# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    device = 'cpu'
    model = EmotionResNet(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_detector():
    return FaceDetector()


# --- Main ---
st.title("MoodTune 🎵")
st.subheader("Capture your expression, hear your mood")

model = load_model()
detector = load_detector()

if model is None:
    st.warning(
        "No trained model found. Please train the FER model first:\n\n"
        "```bash\n"
        "cd 'final project'\n"
        "python training/train_fer.py\n"
        "```\n\n"
        "You'll need the FER2013 dataset in `data/fer2013/` (download from Kaggle)."
    )
    st.stop()

# --- Webcam Capture ---
st.markdown("### 📸 Capture Your Expression")
img_file = st.camera_input("Take a photo")

if img_file is not None:
    # Decode image
    file_bytes = np.frombuffer(img_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect face and predict emotion
    face_tensor, bbox = detector.detect_and_preprocess(frame)

    if face_tensor is None:
        st.error("No face detected. Please try again with better lighting and face the camera directly.")
    else:
        emotion, confidence, probs = predict_emotion(model, face_tensor)

        if confidence < confidence_threshold:
            st.warning(f"Low confidence ({confidence:.0%}). Try retaking the photo.")
        else:
            # --- Results Layout ---
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Detected Expression")
                # Draw bbox on image
                annotated = detector.draw_bbox(frame.copy(), bbox, emotion, confidence)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            with col2:
                st.markdown("### Emotion Analysis")

                # Emotion label with large text
                EMOTION_EMOJI = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠',
                    'fearful': '😨', 'surprised': '😲', 'neutral': '😐',
                    'disgusted': '🤢',
                }
                emoji = EMOTION_EMOJI.get(emotion, '')
                st.metric(
                    label="Detected Emotion",
                    value=f"{emoji} {emotion.capitalize()}",
                    delta=f"{confidence:.1%} confidence",
                )

                # Confidence bar chart
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ['#ff6b6b' if EMOTION_LABELS[i] == emotion else '#4ecdc4'
                          for i in range(len(EMOTION_LABELS))]
                ax.barh(EMOTION_LABELS, probs, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Confidence')
                ax.set_title('Emotion Probabilities')
                plt.tight_layout()
                st.pyplot(fig)

            st.divider()

            # --- Music Generation ---
            st.markdown("### 🎶 Generated Melody")
            st.markdown(f"Creating a melody to match your **{emotion}** mood...")

            melody, params = generate_melody(emotion)
            wav_buffer = melody_to_wav(melody, params['tempo'])

            if wav_buffer:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.audio(wav_buffer, format='audio/wav')
                with col_b:
                    st.markdown("**Musical Parameters:**")
                    st.markdown(f"- Tempo: {params['tempo']} BPM")
                    st.markdown(f"- Scale: {params['scale'].capitalize()}")
                    st.markdown(f"- Octave: {params['octave']}")
                    st.markdown(f"- Density: {params['note_density']:.0%}")

                # Regenerate button
                if st.button("🔄 Generate New Melody"):
                    import random
                    melody, params = generate_melody(emotion, seed=random.randint(0, 99999))
                    wav_buffer = melody_to_wav(melody, params['tempo'])
                    if wav_buffer:
                        st.audio(wav_buffer, format='audio/wav')

            st.divider()

            # --- Song Recommendations ---
            st.markdown("### 🎧 Song Recommendations")
            try:
                songs_df, improvement_label = recommend_songs(emotion, n=num_recommendations)
                st.markdown(
                    f"Since you seem **{emotion}**, here are some **{improvement_label}** tracks:"
                )
                for _, row in songs_df.iterrows():
                    st.markdown(
                        f"**{row['title']}** by {row['artist']} "
                        f"({row['genre']}) — "
                        f"[Listen on Spotify]({row['url']})"
                    )
            except Exception as e:
                st.error(f"Could not load song recommendations: {e}")
