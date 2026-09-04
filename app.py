from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("random_forest_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    turbidity = float(request.form["turbidity"])
    do = float(request.form["do"])
    ph = float(request.form["ph"])
    temp = float(request.form["temp"])
    bod = float(request.form["bod"])

    features = np.array([[turbidity, do, ph, temp, bod]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
