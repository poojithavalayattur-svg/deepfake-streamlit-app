# app.py (updated)
import streamlit as st
import tempfile, os, shutil, joblib
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# CONFIG
MODEL_PATH = "model.joblib"   # optional: real model file
FRAME_STEP = 10
MAX_FRAMES = 40
TARGET_SIZE = (128,128)

st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🧠 Deepfake Video Detection Prototype")
st.write("Upload a short video (5–20s). The app analyzes sampled frames and shows one clear result + confidence and sample frames used.")

# Load classifier if present
classifier = None
if os.path.exists(MODEL_PATH):
    try:
        classifier = joblib.load(MODEL_PATH)
        st.sidebar.success("Loaded model.joblib")
    except Exception as e:
        st.sidebar.error(f"Failed loading model.joblib: {e}")
        classifier = None
else:
    st.sidebar.info("No model.joblib found — running in demo/mock mode")

# Feature extractor caching
@st.cache_resource
def load_feat_model():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(TARGET_SIZE[0],TARGET_SIZE[1],3))

feat_model = load_feat_model()

def extract_frames(video_path, step=FRAME_STEP, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame.copy())
        idx += 1
    cap.release()
    return frames

def crop_faces(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    crops = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x,y,w,h) in faces:
            crops.append(f[y:y+h, x:x+w])
    return crops

def extract_features(frames):
    feats = []
    for f in frames:
        try:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img = cv2.resize(f_rgb, TARGET_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = feat_model.predict(x, verbose=0)
            feats.append(feat.flatten())
        except Exception:
            continue
    if len(feats) == 0:
        return np.zeros((0, feat_model.output_shape[-1]))
    return np.vstack(feats)

def predict_from_features(feats):
    # returns avg_prob (0..1)
    if feats.shape[0] == 0:
        return None
    if classifier is None:
        # deterministic mock based on mean feature value (stable for same input)
        avg_val = float(np.mean(np.abs(feats)))  # any deterministic mapping
        # scale to 0..1
        scaled = (avg_val % 1.0)
        avg_prob = 0.2 + 0.6 * scaled  # maps into 0.2..0.8
        return float(avg_prob)
    else:
        probs = classifier.predict_proba(feats)[:,1]
        return float(np.mean(probs))

# UI: upload
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload a short video (MP4/MOV) — recommended 5–20s", type=["mp4","mov","avi"])
    analyze = st.button("Analyze video") if uploaded else None

with col2:
    st.markdown("**How to demo:**\n\n- Upload one video, click *Analyze video*. \n- Wait 10–40s depending on video length. \n- The app shows one clear prediction and sample frames.")
    st.markdown("---")
    st.markdown("**Legend**: Confidence > 0.75 → likely FAKE; 0.4–0.75 → possibly suspicious; <0.4 → likely REAL.")

if uploaded:
    # save temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush()
    tmp_path = tmp.name

    if analyze:
        with st.spinner("Processing video — extracting frames and analyzing..."):
            frames = extract_frames(tmp_path)
            if len(frames) == 0:
                st.error("No frames could be extracted from this video. Try a different clip.")
            else:
                face_crops = crop_faces(frames)
                imgs_for_features = face_crops if len(face_crops) > 0 else frames
                feats = extract_features(imgs_for_features)
                avg_prob = predict_from_features(feats)

                if avg_prob is None:
                    st.error("Could not extract features from frames. Try a clearer video.")
                else:
                    label = "FAKE" if avg_prob >= 0.5 else "REAL"
                    # show big metric
                    st.metric("Prediction", f"{label}", delta=f"{avg_prob*100:.1f}% fake prob")
                    # show text explanation
                    if avg_prob >= 0.75:
                        st.error("Model strongly suggests this video may be manipulated.")
                    elif avg_prob >= 0.4:
                        st.warning("Model finds some inconsistencies — consider manual review.")
                    else:
                        st.success("Model finds no strong signs of manipulation (likely real).")

                    # show up to 4 sample frames used
                    sample_imgs = imgs_for_features[:4]
                    if len(sample_imgs) > 0:
                        st.markdown("**Sample frames used for analysis:**")
                        cols = st.columns(len(sample_imgs))
                        for c, img in zip(cols, sample_imgs):
                            # convert BGR->RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            c.image(img_rgb, use_column_width=True)

        # clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
