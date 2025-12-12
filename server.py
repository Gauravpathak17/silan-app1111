# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
import base64
import io
from typing import Optional

app = Flask(__name__)
CORS(app)  # allow frontend requests (Node proxy or direct)

# Load ML models once
MODEL_PATH = "models/gesture_model_v2.pkl"
SCALER_PATH = "models/scaler_v2.pkl"
LE_PATH = "models/label_encoder_v2.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LE_PATH)

# 63 features (x,y,z for 21 landmarks)
feature_columns = [str(i) for i in range(63)]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # optional if you want debugging visuals

def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """Convert base64 data URI to cv2 image (BGR)."""
    try:
        header, data = base64_string.split(",", 1)
    except ValueError:
        data = base64_string

    try:
        decoded = base64.b64decode(data)
        arr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        app.logger.error("Failed to decode base64 image: %s", e)
        return None

def extract_landmarks(img: np.ndarray) -> Optional[str]:
    """Extract 63 features + predict label."""
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) != 63:
            return None

        df = pd.DataFrame([landmarks], columns=feature_columns)
        scaled = scaler.transform(df)
        probs = model.predict_proba(scaled)[0]
        label = label_encoder.inverse_transform([int(np.argmax(probs))])[0]
        return label.upper()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json(force=True)
        img_b64 = req.get("image")

        if not img_b64:
            return jsonify({"error": "No image received"}), 400

        img = decode_base64_image(img_b64)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        label = extract_landmarks(img)
        return jsonify({"prediction": label or ""}), 200

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------
# HEALTH CHECK ENDPOINT (required for frontend dashboard)
# --------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/")
def home():
    return jsonify({"status": "Flask ML backend OK"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
