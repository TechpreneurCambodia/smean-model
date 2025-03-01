from setup import *

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)   # Audio file need to be saved to generate transcription
        try:
            transcription, correction = audio_prompt_response(filepath)
            transcription_text = transcription['text']
            correction_text = correction
            print("Transcription: ", transcription_text)
        except Exception as e:
            transcription_text = "Error"
            correction_text = "Error"
            print(f"Error in audio processing: {e}")

        return model_response(
            title="Audio title1", 
            content=transcription_text, 
            noteType="audio", 
            transcription=transcription_text, 
            summary=correction_text,
        )
        
    return jsonify({
        "error": "Invalid file"}), 400  

@app.route('/upload-test', methods=['POST'])
def upload_test():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        return model_response()
    return jsonify({
        "error": "Invalid file",}), 400
if __name__ == '__main__':
    app.run(debug=True)