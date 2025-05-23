import pandas as pd
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "API is active"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "Please provide a 'symptoms' list in JSON body."}), 400

    symptoms = data["symptoms"]

    # Load structure of the training data
    df = pd.read_csv("data/Training.csv")
    df.drop(columns=['Unnamed: 133'], errors='ignore', inplace=True)

    # Prepare input vector
    input_vector = pd.DataFrame([0] * (df.shape[1] - 1), index=df.columns[:-1]).T
    for symptom in symptoms:
        if symptom in input_vector.columns:
            input_vector.loc[0, symptom] = 1

    # Load model and label encoder
    with open("best_classifier.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoder_y.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Get prediction probabilities
    proba = model.predict_proba(input_vector)[0]
    top3_indices = proba.argsort()[-3:][::-1]
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    top3_probs = proba[top3_indices]

    top3_results = [
        {"Disease": disease, "Probability": round(float(prob), 4)}
        for disease, prob in zip(top3_diseases, top3_probs)
    ]

    return jsonify({"Top 3 Predictions": top3_results})

if __name__ == "__main__":
    app.run(debug=True)
