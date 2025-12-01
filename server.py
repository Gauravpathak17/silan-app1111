import os
import requests
import pickle
from flask import Flask, jsonify

app = Flask(__name__)

def download_file(url, filename):
    """Download file from HuggingFace only if not already downloaded."""
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return

    print(f"Downloading {filename}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"{filename} downloaded successfully.")


# HuggingFace direct download URLs
model_url = "https://huggingface.co/gauravpathak11/silan-model/resolve/main/gesture_model_v2.pkl"
scaler_url = "https://huggingface.co/gauravpathak11/silan-model/resolve/main/scaler_v2.pkl"
encoder_url = "https://huggingface.co/gauravpathak11/silan-model/resolve/main/label_encoder_v2.pkl"

# Local file names
model_file = "gesture_model_v2.pkl"
scaler_file = "scaler_v2.pkl"
encoder_file = "label_encoder_v2.pkl"

# Download the files
download_file(model_url, model_file)
download_file(scaler_url, scaler_file)
download_file(encoder_url, encoder_file)

# Load model components
model = pickle.load(open(model_file, "rb"))
scaler = pickle.load(open(scaler_file, "rb"))
encoder = pickle.load(open(encoder_file, "rb"))

@app.route("/")
def home():
    return "SILAN MODEL V2 RUNNING SUCCESSFULLY!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
