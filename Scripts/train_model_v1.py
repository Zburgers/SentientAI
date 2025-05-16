import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle

# Load Dataset
df = pd.read_csv("DDoS_Dataset_v1.csv")

# Feature Selection (Using Top 30 Most Important Features)
important_features = [
    "URG Flag Count", "Bwd Packet Length Max", "Packet Length Min", 
    "Bwd Packets Length Total", "Fwd Packet Length Min", "Fwd Act Data Packets",
    "Bwd Packets/s", "Subflow Bwd Bytes", "Init Bwd Win Bytes", "Down/Up Ratio",
    "Bwd Packet Length Mean", "Init Fwd Win Bytes", "Fwd Packets Length Total",
    "Avg Bwd Segment Size", "Flow IAT Mean", "Flow IAT Std", "Fwd Packet Length Mean",
    "Avg Packet Size", "Fwd Packet Length Max", "Fwd IAT Mean", "Packet Length Mean",
    "Flow Bytes/s", "Fwd Packets/s", "Flow Duration", "Total Fwd Packets", 
    "Total Backward Packets", "ACK Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count"
]

# Keep only features that actually exist in dataset
available_features = [feature for feature in important_features if feature in df.columns]
print(f"🔥 Using {len(available_features)} features for training: {available_features}")


df = df[available_features + ["Label"]]  # ✅ Keep only selected features

# Balance Dataset (Equal Attack & Normal Samples)
df_attack = df[df["Label"] == 1]
df_normal = df[df["Label"] == 0]
df_attack_balanced = resample(df_attack, replace=True, n_samples=len(df_normal), random_state=42)
df_balanced = pd.concat([df_attack_balanced, df_normal]).sample(frac=1, random_state=42)

# Split Features & Labels
X = df_balanced.drop(columns=["Label"]).values
y = df_balanced["Label"].values

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for CNN & LSTM (Adding 3rd Dimension)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

df.to_csv("DDoS_Dataset_v1.csv", index=False)

# CNN-LSTM Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # Binary Classification (Attack or Normal)
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train Model (Increased Epochs for Better Learning)
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Save Model in `.keras` Format
model.save("cnn_lstm_model_v1.keras")
print("🔥 Model saved as `cnn_lstm_model_v1.keras`!")

# Save the StandardScaler
with open("scaler_v1.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🔥 Scaler saved successfully as `scaler_v1.pkl`!")

