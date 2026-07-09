from pathlib import Path

from flask import Flask, render_template, request

from src.model import ensure_model
from src.predict import predict_crop
from src.validation import FEATURE_FIELDS, validate_form


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.pkl"

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FEATURE_FIELDS, values={}, errors={})


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate_form(request.form)
    if errors:
        return render_template("index.html", fields=FEATURE_FIELDS, values=values, errors=errors), 400

    ensure_model(MODEL_PATH)
    crop, probabilities = predict_crop(values, MODEL_PATH)
    return render_template("result.html", crop=crop, probabilities=probabilities, inputs=values)


if __name__ == "__main__":
    ensure_model(MODEL_PATH)
    app.run(debug=True)
