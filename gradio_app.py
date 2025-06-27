import os
import json
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from transformers import HubertModel

# ─── CONFIG ────────────────────────────────────────────────
SAMPLE_RATE      = 16000
MAX_AUDIO_LEN    = 6 * SAMPLE_RATE
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR        = "final-hubert-audio-classifier-1744960186"

MODEL_WEIGHTS    = os.path.join(MODEL_DIR, "model_state_dict.pt")
LABELS_JSON      = os.path.join(MODEL_DIR, "label_mapping.json")

# ─── MODEL DEFINITION ──────────────────────────────────────
class HubertForAudioClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.hubert     = HubertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.hubert.config.hidden_size, num_labels)
    def forward(self, x, attention_mask=None):
        out = self.hubert(input_values=x, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

# ─── VERIFY FILES ──────────────────────────────────────────
assert os.path.exists(MODEL_WEIGHTS), f"Missing: {MODEL_WEIGHTS}"
assert os.path.exists(LABELS_JSON),  f"Missing: {LABELS_JSON}"

# ─── LOAD LABELS & MODEL ───────────────────────────────────
with open(LABELS_JSON, "r") as f:
    labels = json.load(f)
num_labels = len(labels)

model = HubertForAudioClassification("facebook/hubert-base-ls960", num_labels)
state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE).eval()

# ─── PREPROCESS ────────────────────────────────────────────
def preprocess(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(audio) > MAX_AUDIO_LEN:
        audio = audio[:MAX_AUDIO_LEN]
    else:
        pad = MAX_AUDIO_LEN - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return tensor

# ─── PREDICTION ────────────────────────────────────────────
def predict_gradio(wav_filepath):
    x = preprocess(wav_filepath)
    mask = torch.ones_like(x).to(DEVICE)
    with torch.no_grad():
        logits = model(x, attention_mask=mask)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()
    idx    = int(np.argmax(probs))
    label  = labels[str(idx)]
    conf   = float(probs[idx])
    all_p  = {labels[str(i)]: float(probs[i]) for i in range(len(probs))}
    top3   = sorted(all_p.items(), key=lambda x: x[1], reverse=True)[:3]
    return label, f"{conf:.2%}", dict(top3)

# ─── GRADIO INTERFACE ──────────────────────────────────────
iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Audio(sources=["upload"], type="filepath", label="Upload WAV"),
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.Textbox(label="Confidence"),
        gr.Label(num_top_classes=3, label="Top‑3 Probabilities")
    ],
    title="HuBERT Audio Classifier",
    description="Upload a WAV file (≤6 s) and see the predicted class.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()  # add share=True if you want a public URL
