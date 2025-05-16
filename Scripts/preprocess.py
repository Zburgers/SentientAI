from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load filtered dataset
df = pd.read_csv("Filtered_DDoS_Dataset.csv")

# Convert attack labels to binary
df["Label"] = df["Label"].apply(lambda x: 0 if "Benign" in str(x) else 1)

# Drop non-numeric or irrelevant columns
drop_cols = ["Flow ID", "Timestamp", "Source IP", "Destination IP"]  # Drop unnecessary identifiers
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Normalize features
scaler = StandardScaler()
X = df.drop(columns=["Label"])  # Features
y = df["Label"]  # Target (attack or not)

X_scaled = scaler.fit_transform(X)

# Save processed data
pd.DataFrame(X_scaled, columns=X.columns).to_csv("Processed_DDoS_Dataset.csv", index=False)
pd.DataFrame(y, columns=["Label"]).to_csv("DDoS_Labels.csv", index=False)

print("🔥 Data Preprocessing Complete! Saved as Processed_DDoS_Dataset.csv")
