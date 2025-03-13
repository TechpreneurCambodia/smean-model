from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import librosa
import soundfile as sf  # To save audio files
from transformers import pipeline
from response_logic import prompt_correction, prompt_summary

from utils import *
import numpy as np
from scipy.io.wavfile import write
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from flask_cors import CORS

# Set up pipeline for TTS
tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-khm", device=-1)

# Set up pipeline for ASR
pipe = pipeline("automatic-speech-recognition", model="SSethisak/xlsr-khmer-fleur", device=-1)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

# Upload set up
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to resample the audio to 16 kHz
def resample_audio(audio):
    try:
        waveform, sample_rate = librosa.load(audio, sr=None)
        if waveform is None or sample_rate is None:
            raise ValueError("Failed to load audio file")

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate

        resampled_path = os.path.join(UPLOAD_FOLDER, f"resampled_{os.path.basename(audio)}")
        sf.write(resampled_path, waveform, sample_rate)
        return resampled_path
    except Exception as e:
        raise e

# Function to get the transcription and correction
def audio_prompt_response(audio):
    try:
        resampled_audio = resample_audio(audio)
        data, samplerate = sf.read(resampled_audio)
        transcription = pipe(data)
        correction = prompt_correction(transcription["text"])
        summary = prompt_summary(correction)
        return correction, summary
    except Exception as e:
        print(f"Error in audio processing: {e}")
        return f"Error: {e}"

'''
Application setup
'''
# Define the app
app = Flask(__name__)

# Set up origin for CORS
CORS(app, resources={r"/*": {"origins": os.environ.get("BACKEND_URL")}})
app.config['CORS_HEADERS'] = 'Content-Type'

# Set the upload and output folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER