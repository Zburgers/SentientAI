import requests
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "Data/DDoS_Dataset_v2.csv"

# ✅ Load Dataset
df = pd.read_csv(DATA_PATH)

# ✅ Separate Normal & Attack Samples
df_normal = df[df["Label"] == 0].drop(columns=["Label"])
df_attack = df[df["Label"] == 1].drop(columns=["Label"])

# ✅ Sample One Normal & One Attack Instance
normal_sample = df_normal.sample(n=1).values
attack_sample = df_attack.sample(n=1).values

# ✅ Reshape for Model
normal_input = normal_sample.reshape(1, 26)  # Matches (1, 26)
attack_input = attack_sample.reshape(1, 26)  # Matches (1, 26)

print(f"🔥 Normal Sample Shape: {normal_input.shape}")
print(f"🔥 Attack Sample Shape: {attack_input.shape}")

# 🚀 Send Normal Traffic Request
print("🚀 Sending Labeled Normal Traffic Request...")
response = requests.post(API_URL, json={"features": normal_input.tolist()[0]})
print(f"✅ Model Prediction for Normal: {response.json()}")

# 🚨 Send Labeled DDoS Attack Request
print("\n🚀 Sending Labeled DDoS Attack Request...")
response = requests.post(API_URL, json={"features": attack_input.tolist()[0]})
print(f"🚨 Model Prediction for DDoS: {response.json()}")
