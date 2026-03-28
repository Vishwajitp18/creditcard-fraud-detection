from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("../model/model.pkl", "rb"))

# Feature names (IMPORTANT for correct mapping)
feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27",
    "V28","Amount"
]

@app.route("/")
def home():
    return " ML Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]

        # Convert to DataFrame (fixes warning + keeps feature order)
        df = pd.DataFrame([data], columns=feature_names)

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # Explainability (feature contribution approximation)
        contributions = df.iloc[0] * model.feature_importances_

        top_features = sorted(
            list(zip(feature_names, contributions)),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        return jsonify({
            "fraud": bool(prediction),
            "confidence": round(float(probability), 3),
            "top_factors": [
                {"feature": f, "impact": float(v)} for f, v in top_features
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)