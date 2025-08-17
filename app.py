import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------------
# Configuration
# --------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}
MODEL_PATH = os.path.join("artifacts", "training", "model.keras")

# IMPORTANT: Ensure this order matches your training class_indices mapping.
# By default, Keras flow_from_directory sorts classes alphabetically.
CLASS_KEYS = [
    "cocci",
    "healthy",
    "ncd",
    "pcrcocci",
    "pcrhealthy",
    "pcrncd",
    "pcrsalmo",
]
DISPLAY_NAMES = {
    "cocci": "Coccidiosis",
    "healthy": "Healthy",
    "ncd": "Newcastle disease",
    "pcrcocci": "PCR Coccidiosis",
    "pcrhealthy": "PCR Healthy",
    "pcrncd": "PCR Newcastle disease",
    "pcrsalmo": "PCR Salmonella",
}

# --------------------------
# App setup
# --------------------------
app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
MODEL = load_model(MODEL_PATH, compile=False)

# --------------------------
# Helpers
# --------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path: str):
    # Match training preprocessing: 224x224 and rescale 1./255
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)
    return x

def predict_image(img_path: str):
    x = preprocess_image(img_path)
    preds = MODEL.predict(x, verbose=0)      # (1,7)
    if isinstance(preds, list):              # Safety for multi-head models
        preds = preds
    probs = preds                         # (7,)
    pred_idx = int(np.argmax(probs))         # scalar int
    pred_key = CLASS_KEYS[pred_idx]
    pred_name = DISPLAY_NAMES.get(pred_key, pred_key)
    pred_conf = float(probs[pred_idx])       # scalar float

    # Convert all to Python scalars for templating
    details = {
        DISPLAY_NAMES.get(k, k): round(float(p), 6)
        for k, p in zip(CLASS_KEYS, probs)
    }
    print("preds shape:", getattr(preds, "shape", None))
    print("probs shape:", getattr(probs, "shape", None))
    print("pred_idx:", pred_idx, type(pred_idx))
    print("pred_conf:", pred_conf, type(pred_conf))
    
    return {
        "pred_key": pred_key,
        "pred_name": pred_name,
        "pred_conf": round(pred_conf, 4),
        "probs": details,
    }

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Upload an image (png, jpg, jpeg, bmp, gif).")
        return redirect(url_for("index"))

    # Save file safely with unique name
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        result = predict_image(save_path)
        # Optional: delete uploaded file after prediction to save disk
        # os.remove(save_path)

        return render_template(
            "result.html",
            filename=filename,
            predicted_key=result["pred_key"],
            predicted_name=result["pred_name"],
            confidence=result["pred_conf"],
            details=result["probs"],
        )
    except Exception as e:
        # Helpful debug if anything goes wrong
        print("Prediction error:", repr(e))
        flash(f"Prediction failed: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    # Use a production server (e.g., gunicorn) in production
    app.run(host="0.0.0.0", port=5000, debug=True)
