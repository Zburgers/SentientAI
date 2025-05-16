import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load optimized dataset
df = pd.read_csv("Optimized_DDoS_Dataset.csv")

# Separate features & labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for API use
with open("scaler_v3.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the optimized AI model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile & Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# Save trained model
model.save("ddos_model_v3.h5")

print("🔥 Training Complete! New model saved with 30 optimized features.")
