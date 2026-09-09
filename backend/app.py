from flask import Flask, jsonify, request
from flask_cors import CORS

from ml_service import read_csv, summarize, train_classifier


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
CORS(app)


@app.get("/api/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/api/dataset/inspect")
def inspect_dataset():
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file."}), 400
    try:
        summary = summarize(read_csv(request.files["file"].read()))
        return jsonify(summary.__dict__)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/train")
def train():
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file."}), 400
    try:
        frame = read_csv(request.files["file"].read())
        result = train_classifier(
            frame,
            request.form.get("target", ""),
            request.form.get("model", "logistic_regression"),
            request.form.get("scale", "true").lower() == "true",
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "The file is larger than 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

