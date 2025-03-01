

from flask import jsonify


def model_response(title="Audio title1", content="Audio content1", noteType="audio", transcription="transcription", summary="summary"):
    return jsonify({
            "title": title,
            "content": content,
            "noteType": noteType,
            "transcription": transcription,
            "summary": summary
        })