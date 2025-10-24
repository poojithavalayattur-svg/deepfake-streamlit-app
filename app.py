import streamlit as st
import tempfile, os, shutil
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🧠 Deepfake Video Detection Prototype")
st.write("Upload a short video to check whether it might be real or fake using a simple ML-based approach. "
         "This demo runs on a lightweight mock model for educational purposes.")

# --- basic setup ---
FRAME_STEP = 10
MAX_FRAMES = 40
TARGET_SIZE = (128, 128)

@st.cache_resource
def load_feature_model():
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    return model

feat_model = load_feature_model()

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= MAX_FRAMES:
            break
        if idx % FRAME_STEP == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def crop_faces(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    crops = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            crops.append(frame[y:y+h, x:x+w])
    return crops if crops else frames

def extract_features(frames):
    features = []
    for frame in frames:
        img = cv2.resize(frame, TARGET_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        feat = feat_model.predict(x, verbose=0)
        features.append(feat.flatten())
    return np.vstack(features)

def predict_fake_probability(video_path):
    frames = extract_frames(video_path)
    crops = crop_faces(frames)
    feats = extract_features(crops)
    avg_val = np.mean(np.abs(feats)) % 1.0
    label = "FAKE" if avg_val > 0.5 else "REAL"
    return round(avg_val, 2), label

uploaded = st.file_uploader("Upload a short video (MP4/MOV)", type=["mp4", "mov"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    st.video(tfile.name)

    if st.button("Analyze Video"):
        with st.spinner("Analyzing video... please wait"):
            prob, label = predict_fake_probability(tfile.name)
        st.success(f"Result: {label}")
        st.write(f"Fake probability: **{prob*100:.1f}%**")
        st.info("This is a simplified prototype running on mock logic (not a trained model).")
