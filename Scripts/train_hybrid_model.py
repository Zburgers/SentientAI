import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🔥 Load Preprocessed Dataset
df = pd.read_csv("Data/DDoS_Dataset_v2.csv")
X = df.drop(columns=["Label"])
y = df["Label"]

# ✅ Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🚀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔥 Build CNN-LSTM Model
def build_cnn_lstm():
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1], 1))
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    
    # ✅ Fix Feature Extraction Output to 26 Instead of 128
    feature_output = tf.keras.layers.Dense(26, activation="relu", name="feature_extraction")(x)
    classification_output = tf.keras.layers.Dense(1, activation="sigmoid", name="classification")(feature_output)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

# 🚀 Train CNN-LSTM Model
cnn_lstm_model = build_cnn_lstm()
cnn_lstm_model.fit(
    X_train.reshape(-1, X_train.shape[1], 1), 
    y_train,
    epochs=10, batch_size=64, validation_split=0.2
)

# 🔥 Extract LSTM Features (NOW WITH 26-DIMENSION OUTPUT)
feature_extractor = tf.keras.Model(
    inputs=cnn_lstm_model.input, 
    outputs=cnn_lstm_model.get_layer("feature_extraction").output
)
X_train_embeddings = feature_extractor.predict(X_train.reshape(-1, X_train.shape[1], 1))
X_test_embeddings = feature_extractor.predict(X_test.reshape(-1, X_test.shape[1], 1))

print(f"🔥 LSTM Feature Extraction Shape: {X_train_embeddings.shape}")  # ✅ Should print (None, 26)

# 🚀 Train XGBoost Model with Optimized Settings
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
    n_jobs=4,
    use_label_encoder=False,
    eval_metric="logloss",
)

xgb_model.fit(X_train_embeddings, y_train)

# 💾 Save Models
cnn_lstm_model.save("Models/cnn_lstm_xgb_model_v1.keras")
pickle.dump(xgb_model, open("Models/xgb_classifier_v1.pkl", "wb"))

print("✅ Hybrid CNN-LSTM-XGBoost Model Trained & Saved!")
