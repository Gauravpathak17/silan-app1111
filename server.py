from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model components
model = joblib.load("models/gesture_model_v2.pkl")
scaler = joblib.load("models/scaler_v2.pkl")
label_encoder = joblib.load("models/label_encoder_v2.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting: { "landmarks": [63 values] }
        if "landmarks" not in data:
            return jsonify({"error": "No landmarks provided"}), 400

        landmarks = data["landmarks"]

        if len(landmarks) != 63:
            return jsonify({"error": "Invalid landmark count"}), 400

        # Convert to array
        landmarks_np = np.array(landmarks).reshape(1, -1)

        # Scale
        scaled = scaler.transform(landmarks_np)

        # Predict
        pred = model.predict(scaled)[0]

        # Decode label
        label = label_encoder.inverse_transform([pred])[0]

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"status": "Backend OK"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
