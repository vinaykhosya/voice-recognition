import os
import requests

# 1) Configuration
# ----------------
# Path to your recorded audio file
AUDIO_DIR  = os.path.expanduser(r"C:\Users\VINAY\Desktop\python\esp")
AUDIO_FILE = "recorded_audio.wav"

# URL of your Flask predict endpoint
API_URL    = "http://127.0.0.1:5000/predict"


def call_predict(file_path):
    """Send a WAV file to the /predict endpoint and print the result."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, "rb") as wav:
        files = {"file": (os.path.basename(file_path), wav, "audio/wav")}
        print(f"📤 Uploading {file_path} → {API_URL}")
        resp = requests.post(API_URL, files=files)

    if resp.status_code == 200:
        data = resp.json()
        print("✅ Prediction result:")
        print(f"   • Label:      {data['predicted_label']}")
        print(f"   • Confidence: {data['confidence']:.4f}")
        print("   • All probabilities:")
        for label, p in data["all_probabilities"].items():
            print(f"     ‑ {label}: {p:.4f}")
    else:
        print(f"❌ Server returned {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    # Build full path
    audio_path = os.path.join(AUDIO_DIR, AUDIO_FILE)
    call_predict(audio_path)
