from flask import Flask, send_file
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading

app = Flask(__name__)

fs = 16000  # Sampling frequency

def record_audio():
    print("Recording...")
    seconds = 6
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write("recorded_audio.wav", fs, recording)
    print("Recording done and saved.")

@app.route('/start_recording', methods=['GET'])
def start_recording():
    t = threading.Thread(target=record_audio)
    t.start()
    t.join()
    return send_file('recorded_audio.wav', mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
