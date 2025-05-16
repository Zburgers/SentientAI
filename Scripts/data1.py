import pandas as pd
import os

# Define selected files
files = [
    "CIC/LDAP-training.parquet",
    "CIC/MSSQL-training.parquet",
    "CIC/Syn-training.parquet",
    "CIC/UDP-training.parquet",
    "CIC/NetBIOS-training.parquet",
    "CIC/Portmap-training.parquet",
    "CIC/Syn-testing.parquet",
    "CIC/UDP-testing.parquet"
]


# Load & merge selected files
df_list = [pd.read_parquet(f) for f in files if os.path.exists(f)]
df = pd.concat(df_list, ignore_index=True)

# Print dataset summary
print("🔥 Loaded Data Shape:", df.shape)
print(df["Label"].value_counts())  # Check class distribution

# Save to CSV for further processing
df.to_csv("Filtered_DDoS_Dataset.csv", index=False)
