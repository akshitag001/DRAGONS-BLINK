import threading
from flask import Flask, render_template, jsonify, request
from state import state
from pipeline import run_pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify({
        "blur_faces": state.blur_faces,
        "blur_cards": state.blur_cards,
        "fps": round(state.fps, 1),
        "total_faces": state.total_faces,
        "total_cards": state.total_cards,
        "is_running": state.is_running
    })

@app.route("/api/toggle", methods=["POST"])
def toggle():
    data = request.json
    setting = data.get("setting")
    value = data.get("value")
    
    if setting == "blur_faces":
        state.blur_faces = bool(value)
    elif setting == "blur_cards":
        state.blur_cards = bool(value)
        
    return jsonify({"status": "success"})

def start_pipeline_thread():
    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()

if __name__ == "__main__":
    print("Starting PrivacyGuard Pipeline in background...")
    start_pipeline_thread()
    
    print("Starting PrivacyGuard Dashboard Server on http://localhost:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
