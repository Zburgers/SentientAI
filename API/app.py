import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

THRESHOLD = 0.6

# ✅ Load Models
cnn_lstm_model = tf.keras.models.load_model("Models/cnn_lstm_xgb_model_v1.keras")
xgb_model = pickle.load(open("Models/xgb_classifier_v1.pkl", "rb"))

# ✅ Load Scaler
scaler = pickle.load(open("Models/scaler_v2.pkl", "rb"))

# ✅ API Definition
app = FastAPI()

class TrafficData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "🔥 DDoS Detection API Running on CPU!"}

@app.post("/predict")
def predict_traffic(data: TrafficData):
    try:
        # ✅ Convert Input to NumPy Array
        features = np.array(data.features).reshape(1, -1)  # Matches (1, 26)

        # ✅ Normalize Input
        features_scaled = scaler.transform(features)

        # ✅ Reshape for CNN-LSTM Model
        cnn_lstm_input = np.expand_dims(features_scaled, axis=-1)  # Matches (1, 26, 1)

        # ✅ CNN-LSTM Prediction
        lstm_prediction = cnn_lstm_model.predict(cnn_lstm_input)[0][0]

        # ✅ Extract CNN-LSTM Features
        feature_extractor = tf.keras.Model(
            inputs=cnn_lstm_model.input, 
            outputs=cnn_lstm_model.get_layer("feature_extraction").output
        )
        extracted_features = feature_extractor.predict(cnn_lstm_input)

        # ✅ XGBoost Prediction
        xgb_prediction = xgb_model.predict_proba(extracted_features)[:, 1][0]

        # ✅ Final Decision (Ensemble Model)
        final_prediction = (lstm_prediction + xgb_prediction) / 2
        attack_label = "🚨 DDoS Attack Detected!" if final_prediction > THRESHOLD else "✅ Normal Traffic"

        return {
            "alert": attack_label,
            "cnn_lstm_confidence": float(lstm_prediction),
            "xgb_confidence": float(xgb_prediction),
            "final_confidence": float(final_prediction)
        }

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}
