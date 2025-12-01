import os
import requests
import pickle
from flask import Flask

app = Flask(__name__)

def download_file(url, filename):
    """Download files from Google Drive."""
    if os.path.exists(filename):
        print(f"{filename} already exists")
        return
    print(f"Downloading {filename}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"{filename} downloaded")

# Google Drive direct links
model_url = "https://drive.google.com/uc?export=download&id=1hjIMhfuXs4nb-jq2F6wmplG-WJjnivBu"
scaler_url = "https://drive.google.com/uc?export=download&id=1BPgZaviKms93Qk1Rq_WHVuSUdY7UDre0"
dataset_url = "https://drive.google.com/uc?export=download&id=1hxLgzFPD4EcnW9U29shplGj25bVEDCNi"

# Local filenames
model_file = "model.pkl"
scaler_file = "scaler.pkl"
dataset_file = "dataset.pkl"

# Download files
download_file(model_url, model_file)
download_file(scaler_url, scaler_file)
download_file(dataset_url, dataset_file)

# Load models
model = pickle.load(open(model_file, "rb"))
scaler = pickle.load(open(scaler_file, "rb"))

@app.route("/")
def home():
    return "SILAN MODEL LOADED SUCCESSFULLY!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
