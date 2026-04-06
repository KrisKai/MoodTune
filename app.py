import sys
import os
import numpy as np
import cv2
import torch
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMOTION_LABELS, MODEL_PATH, LSTM_MODEL_PATH
from models.fer_model import build_model, predict_emotion
from models.music_generator import generate_melody
from utils.face_detection import FaceDetector
from utils.midi_utils import melody_to_wav
from utils.song_recommender import recommend_songs

# --- Page Config ---
st.set_page_config(
    page_title="MoodTune",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main background & font */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Hero title */
    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Emotion result card */
    .emotion-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #333;
    }
    .emotion-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.3rem;
    }
    .emotion-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.2rem;
    }
    .emotion-confidence {
        font-size: 1rem;
        color: #aaa;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }

    /* Song card */
    .song-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    .song-card:hover {
        transform: translateX(4px);
    }
    .song-title {
        font-size: 1rem;
        font-weight: 600;
        color: #fff;
    }
    .song-artist {
        font-size: 0.85rem;
        color: #aaa;
    }
    .song-genre {
        display: inline-block;
        background: #667eea33;
        color: #667eea;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-top: 4px;
    }

    /* Music params pill */
    .param-pill {
        display: inline-block;
        background: #667eea22;
        color: #667eea;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px 4px;
        font-weight: 500;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #fff;
    }

    /* Status indicators */
    .status-ready {
        color: #4ecdc4;
        font-weight: 600;
    }
    .status-not-ready {
        color: #ff6b6b;
        font-weight: 600;
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}

    /* Camera input styling */
    [data-testid="stCameraInput"] {
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("# 🎵 MoodTune")
    st.markdown("*Emotion-powered music generation*")
    st.divider()

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.2, 0.05,
                                     help="Minimum confidence to accept a prediction")
    num_recommendations = st.slider("Song recommendations", 1, 10, 5)
    st.divider()

    # Model status
    st.markdown("### System Status")
    fer_ready = os.path.exists(MODEL_PATH)
    lstm_ready = os.path.exists(LSTM_MODEL_PATH)

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        if fer_ready:
            st.markdown("🟢 **FER Model**")
        else:
            st.markdown("🔴 **FER Model**")
    with col_s2:
        if lstm_ready:
            st.markdown("🟢 **Melody LSTM**")
        else:
            st.markdown("🟡 **Rule-based**")

    st.divider()
    st.markdown("### How It Works")
    st.markdown("""
    1. 📸 **Capture** your expression
    2. 🧠 **AI analyzes** your emotion
    3. 🎶 **LSTM generates** a melody
    4. 🎧 **Songs recommended** to improve mood
    """)
    st.divider()
    st.caption("Deep Learning Final Project — UAB Spring 2026")


# --- Load Models ---
@st.cache_resource
def load_fer_model():
    if not os.path.exists(MODEL_PATH):
        return None
    device = 'cpu'
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_detector():
    return FaceDetector()


# --- Hero Header ---
st.markdown('<p class="hero-title">MoodTune</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Capture your expression. Hear your mood. Discover new music.</p>',
            unsafe_allow_html=True)

model = load_fer_model()
detector = load_detector()

if model is None:
    st.error("⚠️ No trained emotion model found.")
    st.markdown("""
    **To get started, train the models:**
    ```bash
    python3 training/train_fer.py                        # Emotion classifier
    python3 training/train_music.py --generate-synthetic  # Melody LSTM
    ```
    You'll need FER2013 in `data/fer2013/` ([Download from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)).
    """)
    st.stop()

# --- Webcam Capture ---
st.markdown('<p class="section-header">📸 Capture Your Expression</p>', unsafe_allow_html=True)

col_cam, col_space = st.columns([2, 1])
with col_cam:
    img_file = st.camera_input("Take a photo", label_visibility="collapsed")

if img_file is not None:
    # Decode image
    file_bytes = np.frombuffer(img_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect face and predict emotion
    face_tensor, bbox = detector.detect_and_preprocess(frame)

    if face_tensor is None:
        st.warning("😕 No face detected. Try better lighting and face the camera directly.")
    else:
        emotion, confidence, probs = predict_emotion(model, face_tensor)

        if confidence < confidence_threshold:
            st.warning(f"🤔 Low confidence ({confidence:.0%}). Try retaking the photo with a clearer expression.")
        else:
            # ===== RESULTS SECTION =====
            st.markdown("---")

            # --- Emotion Detection ---
            st.markdown('<p class="section-header">🧠 Emotion Analysis</p>', unsafe_allow_html=True)

            EMOTION_EMOJI = {
                'happy': '😊', 'sad': '😢', 'angry': '😠',
                'fearful': '😨', 'surprised': '😲', 'neutral': '😐',
                'disgusted': '🤢',
            }
            EMOTION_COLOR = {
                'happy': '#FFD93D', 'sad': '#6C9BCF', 'angry': '#FF6B6B',
                'fearful': '#A66CFF', 'surprised': '#F7A4A4', 'neutral': '#B2B2B2',
                'disgusted': '#6BCB77',
            }

            col1, col2, col3 = st.columns([1.2, 1, 1.2])

            with col1:
                # Annotated photo
                annotated = detector.draw_bbox(frame.copy(), bbox, emotion, confidence)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            with col2:
                # Emotion card
                emoji = EMOTION_EMOJI.get(emotion, '🎭')
                color = EMOTION_COLOR.get(emotion, '#667eea')
                st.markdown(f"""
                <div class="emotion-card">
                    <div class="emotion-emoji">{emoji}</div>
                    <div class="emotion-label" style="color: {color};">{emotion.capitalize()}</div>
                    <div class="emotion-confidence">{confidence:.1%} confidence</div>
                </div>
                """, unsafe_allow_html=True)

                # Top 3 emotions
                st.markdown("")
                sorted_indices = np.argsort(probs)[::-1]
                for rank, idx in enumerate(sorted_indices[:3]):
                    label = EMOTION_LABELS[idx]
                    prob = probs[idx]
                    bar_emoji = EMOTION_EMOJI.get(label, '')
                    st.markdown(f"{bar_emoji} **{label.capitalize()}** — {prob:.1%}")

            with col3:
                # Confidence chart
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')

                colors = [EMOTION_COLOR.get(EMOTION_LABELS[i], '#667eea') if EMOTION_LABELS[i] == emotion
                          else '#333333' for i in range(len(EMOTION_LABELS))]
                bars = ax.barh(EMOTION_LABELS, probs, color=colors, height=0.6, edgecolor='none')

                # Highlight the detected emotion bar
                for i, bar in enumerate(bars):
                    if EMOTION_LABELS[i] == emotion:
                        bar.set_edgecolor(EMOTION_COLOR.get(emotion, '#667eea'))
                        bar.set_linewidth(2)

                ax.set_xlim(0, 1)
                ax.set_xlabel('Confidence', color='#888', fontsize=10)
                ax.tick_params(colors='#888', labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#333')
                ax.spines['left'].set_color('#333')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ===== MUSIC GENERATION =====
            st.markdown("---")
            st.markdown('<p class="section-header">🎶 AI-Generated Melody</p>', unsafe_allow_html=True)

            lstm_available = os.path.exists(LSTM_MODEL_PATH)
            gen_type = "LSTM Neural Network" if lstm_available else "Rule-based Algorithm"

            melody, params = generate_melody(emotion, use_lstm=lstm_available)
            wav_buffer = melody_to_wav(melody, params['tempo'])

            if wav_buffer:
                col_audio, col_params = st.columns([2, 1])

                with col_audio:
                    st.markdown(f"🧠 Generated by **{gen_type}** for your **{emotion}** mood")
                    st.audio(wav_buffer, format='audio/wav')

                    if st.button("🔄 Generate New Melody", use_container_width=True):
                        melody, params = generate_melody(emotion, use_lstm=lstm_available)
                        wav_buffer = melody_to_wav(melody, params['tempo'])
                        if wav_buffer:
                            st.audio(wav_buffer, format='audio/wav')

                with col_params:
                    st.markdown("**Musical Parameters**")
                    params_html = f"""
                    <div>
                        <span class="param-pill">🎹 {gen_type.split()[0]}</span>
                        <span class="param-pill">🎵 {params['tempo']} BPM</span>
                        <span class="param-pill">🎼 {params['scale'].capitalize()}</span>
                        <span class="param-pill">🔊 Octave {params['octave']}</span>
                        <span class="param-pill">📝 {len(melody)} notes</span>
                    </div>
                    """
                    st.markdown(params_html, unsafe_allow_html=True)

                    # Duration estimate
                    total_beats = sum(n[1] for n in melody)
                    duration_sec = total_beats * (60.0 / params['tempo'])
                    mins = int(duration_sec // 60)
                    secs = int(duration_sec % 60)
                    st.markdown(f"⏱️ Duration: **{mins}:{secs:02d}**")

            # ===== SONG RECOMMENDATIONS =====
            st.markdown("---")
            st.markdown('<p class="section-header">🎧 Recommended Songs</p>', unsafe_allow_html=True)

            try:
                songs_df, improvement_label = recommend_songs(emotion, n=num_recommendations)

                st.markdown(
                    f"Since you're feeling **{emotion}**, here are some **{improvement_label}** tracks to improve your mood:"
                )
                st.markdown("")

                # Display songs in a clean grid
                cols_per_row = min(3, len(songs_df))
                rows = (len(songs_df) + cols_per_row - 1) // cols_per_row

                song_list = list(songs_df.iterrows())
                for row_idx in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        song_idx = row_idx * cols_per_row + col_idx
                        if song_idx < len(song_list):
                            _, row = song_list[song_idx]
                            with cols[col_idx]:
                                st.markdown(f"""
                                <div class="song-card">
                                    <div class="song-title">🎵 {row['title']}</div>
                                    <div class="song-artist">{row['artist']}</div>
                                    <span class="song-genre">{row['genre']}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                st.link_button("▶ Play on Spotify", row['url'],
                                               use_container_width=True)

            except Exception as e:
                st.error(f"Could not load song recommendations: {e}")

            # Footer
            st.markdown("---")
            st.markdown(
                '<p style="text-align: center; color: #555; font-size: 0.85rem;">'
                'Built with EfficientNet-B0 + LSTM | Deep Learning Final Project | UAB Spring 2026'
                '</p>',
                unsafe_allow_html=True,
            )
