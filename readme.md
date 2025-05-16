# 🚨 SentientAI: DDoS Detection Dashboard

![Futuristic AI Cybersecurity](dump/SentientAI.png)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.0-brightgreen?logo=xgboost)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)](https://fastapi.tiangolo.com/)

---

## 🌟 Overview

**SentientAI** is an advanced DDoS detection system featuring a hybrid deep learning and machine learning model (CNN-LSTM + XGBoost) and a real-time interactive dashboard. It empowers cybersecurity teams to monitor, analyze, and respond to network threats with high accuracy and actionable insights.

---

## 🧠 Key Features

- **Hybrid Model:** Combines CNN-LSTM (deep learning) for feature extraction with XGBoost (machine learning) for robust classification.
- **Real-Time Dashboard:** Visualizes live network traffic, attack alerts, and model confidence scores.
- **API-Driven:** FastAPI backend for scalable, programmatic access to predictions.
- **Modern Stack:** Built with TensorFlow, XGBoost, FastAPI, and Streamlit.
- **Customizable:** Easily extendable for new features, data sources, or attack types.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Network Traffic Data] --> B[Preprocessing & Feature Engineering]
    B --> C[CNN-LSTM Model]
    C --> D[Feature Extraction (26D)]
    D --> E[XGBoost Classifier]
    E --> F[Prediction & Alert]
    F --> G[Streamlit Dashboard]
    F --> H[FastAPI Endpoint]
```

---

## 📊 Dashboard Preview

> _Add a screenshot or GIF of your Streamlit dashboard here for maximum impact!_

---

## 🔬 Model Details

- **Input Features:** 26 network traffic features (see below)
- **CNN-LSTM:** Extracts temporal and spatial patterns from traffic data
- **XGBoost:** Final classification using deep features
- **Ensemble Output:** Combines both model confidences for robust detection

**Feature List (26 features + Label):**
```
1. Flow Duration
2. Total Fwd Packets
3. Total Backward Packets
4. Fwd Packets Length Total
5. Bwd Packets Length Total
6. Packet Length Max
7. Packet Length Min
8. Flow IAT Mean
9. Flow IAT Std
10. Flow IAT Max
11. Flow IAT Min
12. Fwd IAT Total
13. Fwd IAT Mean
14. Bwd IAT Total
15. Bwd IAT Mean
16. Flow Bytes/s
17. Flow Packets/s
18. SYN Flag Count
19. RST Flag Count
20. ACK Flag Count
21. URG Flag Count
22. Bwd Packets/s
23. Fwd Packets/s
24. Down/Up Ratio
25. Subflow Fwd Bytes
26. Subflow Bwd Bytes
27. Protocol
28. Label (Target: 0 = Normal, 1 = DDoS Attack)
```

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/Zburgers/SentientAI
cd SentientAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Hybrid Model

```bash
python Scripts/train_hybrid_model.py
```

### 4. Start the API Server

```bash
uvicorn API.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Launch the Dashboard

```bash
streamlit run Dashboard/dashboard.py
```

---

## 🛡️ API Usage

- **POST** `/predict`  
  Send a JSON payload with 26 features to get a DDoS prediction.

```json
{
  "features": [0.1, 0.2, ..., 1.3]
}
```

**Response:**
```json
{
  "alert": "🚨 DDoS Attack Detected!",
  "cnn_lstm_confidence": 0.92,
  "xgb_confidence": 0.89,
  "final_confidence": 0.905
}
```

---

## 📦 Project Structure

```
SentientAI/
│
├── API/                # FastAPI backend
├── Data/               # Datasets
├── Models/             # Saved models
├── Notebooks/          # Jupyter notebooks
├── Scripts/            # Training & utility scripts
├── dump/               # Visual assets, logs, etc.
├── requirements.txt    # Dependencies
├── readme.md           # Project documentation
```

---

## 🖼️ Visual Assets

- ![Futuristic AI Cybersecurity](dump/DALL·E%202025-02-21%2002.44.46%20-%20A%20futuristic%20AI-powered%20cybersecurity%20system%20monitoring%20real-time%20network%20traffic.%20The%20image%20should%20feature%20a%20glowing%20digital%20brain%20integrated%20into%20a%20.webp)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgements

- TensorFlow, XGBoost, FastAPI, Streamlit, and the open-source community.
