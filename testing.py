# Import custom Quantum layer
from training import QuantumLayer
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score

# -------------------------------
# Load Models
# -------------------------------
# Load trained hybrid model
model = load_model(
    "/content/cloudburst_bilstm_quantum.h5",
    custom_objects={"QuantumLayer": QuantumLayer}
)

# Load trained RL agent
rl_model = PPO.load("/content/cloudburst_rl_agent.zip")

# -------------------------------
# Load & Prepare Dataset
# -------------------------------
data = pd.read_csv("/synthetic_weather_data.csv")

# Features and target
X = data.drop("cloudburst_risk", axis=1)
y = data["cloudburst_risk"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Model expects input shape:", model.input_shape)

# -------------------------------
# Reshape Data for BiLSTM Input
# -------------------------------
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# -------------------------------
# Evaluate Model
# -------------------------------
# Direct accuracy from model.evaluate()
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Train Accuracy (model.evaluate): {train_acc * 100:.2f}%")
print(f"Test Accuracy  (model.evaluate): {test_acc * 100:.2f}%")

# -------------------------------
# Accuracy from Predictions
# -------------------------------
# Predict on test set
y_pred_test = (model.predict(X_test) > 0.5).astype(int)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Predict on train set
y_pred_train = (model.predict(X_train) > 0.5).astype(int)
accuracy_train = accuracy_score(y_train, y_pred_train)

print(f"Train Accuracy (from predictions): {accuracy_train * 100:.2f}%")
print(f"Test Accuracy  (from predictions): {accuracy_test * 100:.2f}%")
