from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import librosa
import soundfile as sf  # To save audio files
from transformers import pipeline
from response_logic import prompt_correction

import numpy as np
from scipy.io.wavfile import write
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from flask_cors import CORS

# Set up pipeline for TTS
tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-khm", device=-1)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

pipe = pipeline("automatic-speech-recognition", model="SSethisak/xlsr-khmer-fleur", device=-1)

def resample_audio(audio):
    try:
        # Load the audio file using librosa
        waveform, sample_rate = librosa.load(audio, sr=None)
        print(f"Original Sample Rate: {sample_rate}, Shape: {waveform.shape}")

        # Resample to 16 kHz if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate
            print("Resampled to 16 kHz")

        # Save the resampled audio
        resampled_path = os.path.join("uploads", f"resampled_{os.path.basename(audio)}")
        sf.write(resampled_path, waveform, sample_rate)
        print(f"Resampled audio saved at: {resampled_path}")
        return resampled_path
    except Exception as e:
        print(f"Error in resample_audio: {e}")
        raise

def audio_prompt_response(audio):
    try:
        resampled_audio = resample_audio(audio)
        print(resampled_audio)
        data, samplerate = sf.read(resampled_audio) # Extracting the information and sample rate of the audio from the audio file
        transcription = pipe(data)
        print("Transcription:", transcription["text"])
        correction = prompt_correction(transcription["text"])
        print("Correction:", correction)

        return transcription, correction
    except Exception as e:
        print(f"Error in audio_prompt_response: {e}")
        return f"Error: {e}"

app = Flask(__name__)

# Set up orgin for CORS
CORS(app, resources={r"/*": {"origins": os.environ.get("BACKEND_URL")}})
app.config['CORS_HEADERS'] = 'Content-Type'

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        transcription, correction = audio_prompt_response(filepath)
        transcription_text = transcription["text"]
        correction_text = correction
        # response_text = response.text

        #audio_url = url_for('serve_file', filename=audio_path)
        return jsonify({
            "transcription": transcription_text,
            "correction": correction_text,
            # "response": response_text,
            
        })
    return jsonify({"error": "Invalid file"}), 400

# @app.route('/output/<filename>')
# def serve_file(filename):
#     return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)