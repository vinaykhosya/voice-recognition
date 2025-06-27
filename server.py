import os
import json
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import HubertModel

# ─── CONFIG ────────────────────────────────────────────────
SAMPLE_RATE      = 16000
MAX_AUDIO_LENGTH = 6 * SAMPLE_RATE
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_WEIGHTS = r"final-hubert-audio-classifier-1744960186\model_state_dict.pt"
LABELS_JSON = r"final-hubert-audio-classifier-1744960186\label_mapping.json"


# ─── MODEL DEFINITION (copy‑pasted from your code) ─────────
class HubertForAudioClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.hubert     = HubertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.hubert.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.hubert(input_values=input_values, attention_mask=attention_mask)
        pooled  = outputs.last_hidden_state.mean(dim=1)
        logits  = self.classifier(pooled)
        return logits

# ─── AUDIO PREPROCESSING ───────────────────────────────────
def process_audio_for_inference(file_stream):
    # Read raw bytes from request, write to temp file or load directly
    data, sr = sf.read(file_stream, dtype="float32")
    if data.ndim>1:
        data = data.mean(axis=1)
    if sr!=SAMPLE_RATE:
        data = librosa.resample(y=data, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(data)>MAX_AUDIO_LENGTH:
        data = data[:MAX_AUDIO_LENGTH]
    else:
        padding = MAX_AUDIO_LENGTH - len(data)
        data = np.pad(data, (0, padding), 'constant')
    data = data/ (np.max(np.abs(data))+1e-9)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ─── LOAD LABEL MAPPING ────────────────────────────────────
with open(LABELS_JSON, "r") as f:
    label_mapping = json.load(f)

# ─── MODEL LOADING ─────────────────────────────────────────
num_labels = len(label_mapping)
model = HubertForAudioClassification("facebook/hubert-base-ls960", num_labels)
state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE).eval()

# ─── FLASK APP ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"no file part"}), 400
    f = request.files["file"]
    if f.filename=="":
        return jsonify({"error":"no selected file"}), 400

    try:
        # Preprocess
        input_tensor = process_audio_for_inference(f)
        mask         = torch.ones_like(input_tensor).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor, attention_mask=mask)
            probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()

        idx   = int(np.argmax(probs))
        label = label_mapping[str(idx)]
        conf  = float(probs[idx])
        all_p = { label_mapping[str(i)]: float(probs[i]) for i in range(len(probs)) }

        return jsonify({
            "predicted_label": label,
            "confidence":      conf,
            "all_probabilities": all_p
        })

    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__=="__main__":
    # host 0.0.0.0 if you want other devices to reach it
    app.run(host="127.0.0.1", port=5000, debug=False)
