import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Filtered_DDoS_Dataset.csv")

# Convert attack labels to binary (1 = Attack, 0 = Normal Traffic)
df["Label"] = df["Label"].apply(lambda x: 0 if "Benign" in str(x) else 1)

# Define selected features (Based on Feature Importance)
selected_features = [
    "URG Flag Count", "Bwd Packet Length Max", "Packet Length Min", "Bwd Packets Length Total",
    "Fwd Packet Length Min", "Fwd Act Data Packets", "Bwd Packets/s", "Subflow Bwd Bytes",
    "Init Bwd Win Bytes", "Down/Up Ratio", "Bwd Packet Length Mean", "Init Fwd Win Bytes",
    "Fwd Packets Length Total", "Avg Bwd Segment Size", "Flow IAT Mean", "Flow IAT Std",
    "Fwd Packet Length Mean", "Avg Packet Size", "Fwd Packet Length Max", "Fwd IAT Mean",
    "ACK Flag Count", "Flow Bytes/s", "RST Flag Count", "Fwd IAT Max", "Fwd Packets/s",
    "Flow IAT Max", "Fwd IAT Total", "Packet Length Max", "Subflow Fwd Bytes", "Flow Packets/s"
]

# Keep only selected features
df = df[selected_features + ["Label"]]

# Normalize features
scaler = StandardScaler()
X = df.drop(columns=["Label"])
y = df["Label"]
X_scaled = scaler.fit_transform(X)

# Save processed data
pd.DataFrame(X_scaled, columns=X.columns).to_csv("Processed_DDoS_Dataset.csv", index=False)
pd.DataFrame(y, columns=["Label"]).to_csv("DDoS_Labels.csv", index=False)

# Save the scaler for later use in API
import pickle
with open("scaler_v3.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🔥 Data Preprocessing Complete! Using top 30 features only.")
