from flask import Flask, render_template, request, jsonify
import os
import librosa
import soundfile as sf  # To save audio files
from transformers import pipeline
import soundfile as sf
from response_logic import prompt_category, prompt_response, prompt_correction

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
        response = prompt_response(correction)
        print("Response:", response.text)
        return transcription, correction, response
    except Exception as e:
        print(f"Error in audio_prompt_response: {e}")
        return f"Error: {e}"

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

        transcription, correction, response = audio_prompt_response(filepath)
        transcription_text = transcription["text"]
        correction_text = correction
        response_text = response.text
        return jsonify({
                "transcription": transcription_text,
                "correction": correction_text,
                "response": response_text
            })
    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    app.run(debug=True)
