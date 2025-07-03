
from flask import Flask, request, jsonify, Response, render_template_string
from synth_wrapper import synthesize_text

from flask import request, send_file

app = Flask(__name__)

@app.route("/speak", methods=["POST"])
def speak():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return {"error": "Missing text"}, 400

        print("🔊 Synthesizing text")
        buffer = synthesize_text(data["text"])
        print("📦 Buffer type: in server ", type(buffer))
        print("📦 Buffer size: in server ", buffer.getbuffer().nbytes)

        if not buffer:
            return {"error": "No audio buffer returned"}, 500

 
        print("📦 Buffer type: in server2 ", type(buffer))
        print("📦 Buffer size: in server2 ", buffer.getbuffer().nbytes)
        buffer.seek(0)  # 🔁 Reset stream just in case
        buffer.name = "speech.wav"  # Required for Flask MIME guessing
        print("✅ Synthesis complete")  # <-- You should see this now

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False
        )

    except Exception as e:
        print("❌ TTS synthesis error:", str(e))
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        print("❌ Failed to start server:", str(e))
        import traceback
        traceback.print_exc()
        exit(1)