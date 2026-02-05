from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, Multiply, Add
import tensorflow as tf
import os   # ✅ NEW

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"

# ================= UPLOAD CONFIG (NEW) =================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# =======================================================

# ---------------- Custom Layer ----------------
class ResidualAttention(Layer):
    def __init__(self, units, **kwargs):
        super(ResidualAttention, self).__init__(**kwargs)
        self.units = units
        self.dense1 = Dense(units, activation='relu')
        self.dense2 = Dense(units, activation='sigmoid')
        self.multiply = Multiply()
        self.add = Add()

    def call(self, inputs):
        x = self.dense1(inputs)
        attention = self.dense2(x)
        x = self.multiply([inputs, attention])
        x = self.add([inputs, x])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# ---------------- Load preprocessing objects and model ----------------
try:
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    best_models_list = joblib.load("best_models_list.pkl")
    model_path = best_models_list[0]
    model = load_model(model_path, custom_objects={'ResidualAttention': ResidualAttention})
    print("Loaded scaler, imputer and model:", model_path)
except Exception as e:
    print("Error loading model or preprocessing objects:", e)
    scaler = None
    imputer = None
    model = None

# ---------------- Feature Order ----------------
FEATURE_ORDER = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

# ---------------- Single Prediction Helper ----------------
def predict_heart_disease(input_dict):
    if imputer is None or scaler is None or model is None:
        raise RuntimeError("Model or preprocessing objects not loaded.")

    input_df = pd.DataFrame([[input_dict[f] for f in FEATURE_ORDER]],
                            columns=FEATURE_ORDER)

    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    input_scaled = input_scaled.reshape(1, input_scaled.shape[1], 1)

    prob = model.predict(input_scaled, verbose=0)[0][0]
    label = "Heart Disease DETECTED" if prob > 0.75 else "NO Heart Disease"
    return label, float(prob)

# ================= BULK PREDICTION HELPER (NEW) =================
def predict_from_dataframe(df):
    df = df[FEATURE_ORDER]

    df_imputed = imputer.transform(df)
    df_scaled = scaler.transform(df_imputed)
    df_scaled = df_scaled.reshape(df_scaled.shape[0], df_scaled.shape[1], 1)

    probs = model.predict(df_scaled, verbose=0).flatten()

    df["Probability (%)"] = (probs * 100).round(2)
    df["Prediction"] = df["Probability (%)"].apply(
        lambda x: "Heart Disease DETECTED" if x > 75 else "NO Heart Disease"
    )
    return df
# ================================================================

# ---------------- Routes ----------------

@app.route("/", methods=["GET"])
def index():
    return render_template("home.html")

@app.route("/predict", methods=["GET"])
def show_predict_form():
    defaults = {k: "" for k in FEATURE_ORDER}
    return render_template("predict.html", values=defaults, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_vals = {}
        for f in FEATURE_ORDER:
            val = request.form.get(f, "").strip()
            if val == "":
                flash(f"Please enter a value for '{f}'.", "warning")
                return redirect(url_for('show_predict_form'))
            input_vals[f] = float(val)

        label, prob = predict_heart_disease(input_vals)
        prob_percent = round(prob * 100, 2)

        return render_template(
            "predict.html",
            values=input_vals,
            result={"label": label, "prob": prob_percent}
        )

    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "danger")
        previous = {f: request.form.get(f, "") for f in FEATURE_ORDER}
        return render_template("predict.html", values=previous, result=None)

# ================= FILE UPLOAD ROUTE (NEW) =================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    results = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            flash("Please upload a CSV or Excel file", "warning")
            return render_template("upload.html")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                flash("Only CSV or Excel files are allowed", "danger")
                return render_template("upload.html")

            missing = set(FEATURE_ORDER) - set(df.columns)
            if missing:
                flash(f"Missing columns: {', '.join(missing)}", "danger")
                return render_template("upload.html")

            df_result = predict_from_dataframe(df.copy())
            results = df_result.to_dict(orient="records")

        except Exception as e:
            flash(f"Bulk prediction failed: {str(e)}", "danger")

    return render_template("upload.html", results=results)
# ===========================================================

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/risk")
def risk():
    return render_template("risk.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
