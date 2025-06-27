import torch
import numpy as np
import librosa
import json
import soundfile as sf
import torch.nn as nn
from transformers import HubertModel
from torch.nn import functional as F
import os

# Constants (matching those used during training)
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 6 * SAMPLE_RATE  # 6 seconds at 16kHz
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model class (same as in your training code)
class HubertForAudioClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.hubert.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.hubert(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else logits

def process_audio_for_inference(file_path, max_length=MAX_AUDIO_LENGTH):
    """
    Process WAV audio file for inference
    """
    try:
        # Load the audio file
        audio_data, sr = sf.read(file_path)
        
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Handle length (trim or pad)
        if len(audio_data) > max_length:
            audio_data = audio_data[:max_length]
        else:
            padding = max_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), 'constant')
        
        # Normalize
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
        
        return audio_data.astype(np.float32)
    
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return np.zeros(max_length, dtype=np.float32)

def predict_audio(file_path, model, label_mapping):
    """
    Run inference on an audio file using the trained HuBERT model
    
    Args:
        file_path: Path to the audio file
        model: Loaded HuBERT classification model
        label_mapping: Dictionary mapping class indices to class names
        
    Returns:
        predicted_class: The predicted class name
        confidence: Confidence score for the prediction
        all_probabilities: Dictionary with probabilities for all classes
    """
    # Process the audio file
    processed_audio = process_audio_for_inference(file_path)
    
    # Convert to tensor and add batch dimension
    input_values = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Create attention mask (all ones, same shape as input_values)
    attention_mask = torch.ones_like(input_values).to(DEVICE)
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
    
    # Handle different output types
    if isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs
    
    # Get predictions
    probabilities = F.softmax(logits, dim=1)[0]
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class = label_mapping[str(predicted_class_idx)]
    confidence = probabilities[predicted_class_idx].item()
    
    # Get all probabilities
    all_probabilities = {label_mapping[str(i)]: probabilities[i].item() for i in range(len(label_mapping))}
    
    return predicted_class, confidence, all_probabilities

def main():
    """
    Test inference on the specified WAV file
    """
    # File to test
    test_audio_path = "recorded_audio.wav"
    
    # Model and label mapping paths
    model_path = "final-hubert-audio-classifier-1744960186/model_state_dict.pt"
    label_path = "final-hubert-audio-classifier-1744960186\label_mapping.json"
    
    # Debug checks
    print(f"Testing audio file exists: {os.path.exists(test_audio_path)}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"Label mapping exists: {os.path.exists(label_path)}")
    
    # Load the label mapping
    try:
        with open(label_path, 'r') as f:
            label_mapping = json.load(f)
        print(f"Successfully loaded label mapping: {label_mapping}")
    except Exception as e:
        print(f"Error loading label mapping: {e}")
        return
    
    # Initialize the model with pretrained hubert
    try:
        num_labels = len(label_mapping)
        print(f"Initializing model with {num_labels} labels...")
        model = HubertForAudioClassification("facebook/hubert-base-ls960", num_labels)
        print("Model initialized successfully")
        
        # Load the trained model weights
        print(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully")
        
        model = model.to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run prediction
    try:
        print(f"Processing audio file: {test_audio_path}")
        predicted_class, confidence, all_probs = predict_audio(test_audio_path, model, label_mapping)
        
        # Print results
        print(f"\nTest file: {test_audio_path}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print("\nAll class probabilities:")
        for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {prob:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()