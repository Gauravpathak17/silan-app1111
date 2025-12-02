from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
import base64


app = Flask(__name__)
CORS(app)

# Load ML models
model = joblib.load("models/gesture_model_v2.pkl")
scaler = joblib.load("models/scaler_v2.pkl")
label_encoder = joblib.load("models/label_encoder_v2.pkl")

# 63 features (x, y, z for 21 landmarks)
feature_columns = [str(i) for i in range(63)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

def decode_base64_image(base64_string):
    header, data = base64_string.split(",", 1)
    decoded = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
    return img

def extract_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    pred_index = np.argmax(model.predict_proba(scaled)[0])
    label = label_encoder.inverse_transform([pred_index])[0]

    return label.upper()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json()
        img_b64 = req.get("image", None)

        if img_b64 is None:
            return jsonify({"error": "No image received"}), 400

        img = decode_base64_image(img_b64)
        label = extract_landmarks(img)

        if label is None:
            return jsonify({"prediction": ""})  # hand not detected

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return jsonify({"status": "Backend OK"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
