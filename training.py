#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid BiLSTM + Quantum + PPO RL for Cloudburst Classification

- Train/test split (stratified)
- MinMax scaling
- BiLSTM -> Dense -> QuantumLayer -> Dense
- EarlyStopping & ReduceLROnPlateau
- Optional naive oversampling
- Evaluation (accuracy, report, confusion)
- Saves: model (.h5), scaler (.pkl)
- PPO RL agent learns alerting policy from labels
- RL inference mode to produce final operational decision

Usage (Colab cell):
  !python train_evaluate_hybrid_cloudburst_rl.py --csv /content/synthetic_weather_data.csv \
      --epochs 120 --batch_size 32 --oversample 1 --train_rl 1 --rl_timesteps 20000

RL-only later (reusing saved model/scaler):
  !python train_evaluate_hybrid_cloudburst_rl.py --csv /content/synthetic_weather_data.csv \
      --train 0 --train_rl 1 --rl_timesteps 20000

RL inference on a CSV:
  !python train_evaluate_hybrid_cloudburst_rl.py --csv /content/new_dataset_cleaned.csv \
      --mode infer_rl --head 10
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import pennylane as qml

# RL
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ------------------------------- #
# Reproducibility
# ------------------------------- #
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ------------------------------- #
# Quantum Layer (PennyLane + TF)
# ------------------------------- #
class QuantumLayer(tf.keras.layers.Layer):
    """
    Wraps a PennyLane QNode so it plays nicely with Keras/TensorFlow.
    Encodes an n_qubits-length vector with RY, adds a CNOT chain,
    applies fixed Rot gates, returns expval(Z) per qubit.
    """
    def __init__(self, n_qubits, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = int(n_qubits)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="tf")
        def _qnode(inputs):
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.Rot(0.1, 0.2, 0.3, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = _qnode

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        def one_sample(x):
            out = self.qnode(x)
            return tf.convert_to_tensor(out, dtype=tf.float32)
        return tf.map_fn(
            one_sample,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32),
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

# ------------------------------- #
# Simple Oversampling (optional)
# ------------------------------- #
def oversample_minority(X, y, random_state=42):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y

    maj_class = classes[np.argmax(counts)]
    min_class = classes[np.argmin(counts)]
    n_maj, n_min = counts.max(), counts.min()
    if n_maj == n_min:
        return X, y

    min_idx = np.where(y == min_class)[0]
    rng = np.random.default_rng(random_state)
    add_idx = rng.choice(min_idx, size=(n_maj - n_min), replace=True)

    X_bal = np.concatenate([X, X[add_idx]], axis=0)
    y_bal = np.concatenate([y, y[add_idx]], axis=0)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]

# ------------------------------- #
# Build Hybrid Model
# ------------------------------- #
def build_model(n_features, n_qubits=4, lstm_units=64, dropout_main=0.30, dropout_head=0.20, lr=1e-3):
    inputs = Input(shape=(1, n_features))
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(inputs)
    x = Dropout(dropout_main)(x)
    q_in = Dense(n_qubits)(x)
    q_out = QuantumLayer(n_qubits)(q_in)
    x = Dense(32, activation='relu')(q_out)
    x = Dropout(dropout_head)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------- #
# Train & Evaluate Supervised Model
# ------------------------------- #
def train_and_evaluate(csv_path, epochs=150, batch_size=32,
                       test_size=0.2, oversample=False, seed=42,
                       n_qubits=4, lstm_units=64):
    set_seeds(seed)
    df = pd.read_csv(csv_path)

    feature_columns = [
        "temperature", "humidity", "pressure",
        "wind_speed", "cloud_cover", "ground_level_pressure"
    ]
    target_column = "cloudburst_risk"

    if not all(col in df.columns for col in feature_columns + [target_column]):
        raise ValueError(f"CSV must contain: {feature_columns + [target_column]}")

    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    if oversample:
        X_train_scaled, y_train = oversample_minority(X_train_scaled, y_train, random_state=seed)

    n_features = X_train_scaled.shape[1]
    X_train_lstm = X_train_scaled.reshape((-1, 1, n_features))
    X_test_lstm  = X_test_scaled.reshape((-1, 1, n_features))

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

    model = build_model(n_features=n_features, n_qubits=n_qubits, lstm_units=lstm_units)

    early = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5, verbose=1)

    _ = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_test_lstm, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, reduce],
        class_weight=cw,
        verbose=1
    )

    y_proba = model.predict(X_test_lstm, verbose=0).reshape(-1)
    y_pred = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    print("\n================= RESULTS =================")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    model_path = "cloudburst_bilstm_quantum.h5"
    scaler_path = "cloudburst_minmax_scaler.pkl"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved model to: {os.path.abspath(model_path)}")
    print(f"Saved scaler to: {os.path.abspath(scaler_path)}")

    # Return pieces the RL stage will need
    return {
        "model": model,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "feature_columns": feature_columns
    }

# ------------------------------- #
# RL Environment (label-driven)
# ------------------------------- #
class CloudburstEnv(gym.Env):
    """
    Observation: scaled feature vector (shape = n_features)
    Action: 0 = NO ALERT, 1 = ALERT
    Reward: +r_correct if action == ground-truth label, else r_wrong.
    """
    metadata = {"render_modes": []}

    def __init__(self, X_scaled, y, r_correct=1.0, r_wrong=-1.0):
        super().__init__()
        self.X = X_scaled.astype(np.float32)
        self.y = y.astype(int)
        self.n = len(self.y)
        self.idx = 0
        self.r_correct = float(r_correct)
        self.r_wrong = float(r_wrong)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = np.random.randint(0, self.n)
        return self.X[self.idx], {}

    def step(self, action):
        gt = self.y[self.idx]
        reward = self.r_correct if int(action) == int(gt) else self.r_wrong

        # next state: random sample (bandit-style)
        self.idx = np.random.randint(0, self.n)
        obs = self.X[self.idx]
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

# ------------------------------- #
# Train RL Agent
# ------------------------------- #
def train_rl_agent(X_train_scaled, y_train, rl_timesteps=20000, r_correct=1.0, r_wrong=-1.0):
    env = CloudburstEnv(X_train_scaled, y_train, r_correct=r_correct, r_wrong=r_wrong)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(rl_timesteps))
    rl_path = "cloudburst_rl_agent.zip"
    model.save(rl_path)
    print(f"Saved RL agent to: {os.path.abspath(rl_path)}")
    return rl_path

# ------------------------------- #
# RL Inference (final action)
# ------------------------------- #
def rl_infer_on_csv(csv_path, head=10):
    # Load artifacts
    model = load_model("cloudburst_bilstm_quantum.h5", custom_objects={"QuantumLayer": QuantumLayer})
    scaler = joblib.load("cloudburst_minmax_scaler.pkl")
    rl_model = PPO.load("cloudburst_rl_agent.zip")

    # Columns must match training
    feature_columns = [
        "temperature", "humidity", "pressure",
        "wind_speed", "cloud_cover", "ground_level_pressure"
    ]
    df = pd.read_csv(csv_path)
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    X = df[feature_columns].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # ML model probabilities
    probs = model.predict(X_lstm, verbose=0).reshape(-1)
    labels = np.where(probs >= 0.5, "HIGH RISK", "LOW RISK")

    # RL agent decisions
    rl_actions = []
    for i in range(X_scaled.shape[0]):
        action, _ = rl_model.predict(X_scaled[i], deterministic=True)
        rl_actions.append(["NO ALERT", "ALERT"][int(action)])

    out = pd.DataFrame(df[feature_columns].copy())
    out["Predicted_Probability"] = probs
    out["Model_Label"] = labels
    out["RL_Decision"] = rl_actions

    print(out.head(int(head)).to_string(index=False))
    save_path = "rl_inference_results.csv"
    out.to_csv(save_path, index=False)
    print(f"\nSaved full RL inference to: {os.path.abspath(save_path)}")

# ------------------------------- #
# CLI
# ------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Hybrid BiLSTM+Quantum + PPO RL for Cloudburst.")
    p.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "infer_rl"],
                   help="train (supervised + optional RL) or infer_rl (use saved RL agent)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--oversample", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--lstm_units", type=int, default=64)
    p.add_argument("--train", type=int, default=1, help="1=train supervised model, 0=skip")
    p.add_argument("--train_rl", type=int, default=1, help="1=train PPO on train split, 0=skip")
    p.add_argument("--rl_timesteps", type=int, default=20000)
    p.add_argument("--r_correct", type=float, default=1.0, help="Reward for correct alert/no-alert")
    p.add_argument("--r_wrong", type=float, default=-1.0, help="Penalty for wrong decision")
    p.add_argument("--head", type=int, default=10, help="Rows to print for inference")
    return p.parse_args()

def main():
    args = parse_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    set_seeds(args.seed)

    if args.mode == "train":
        relay = None
        if args.train:
            relay = train_and_evaluate(
                csv_path=args.csv,
                epochs=args.epochs,
                batch_size=args.batch_size,
                test_size=args.test_size,
                oversample=bool(args.oversample),
                seed=args.seed,
                n_qubits=args.n_qubits,
                lstm_units=args.lstm_units,
            )
        else:
            # Load scaler and prepare train split directly for RL if skipping supervised
            df = pd.read_csv(args.csv)
            feature_columns = [
                "temperature", "humidity", "pressure",
                "wind_speed", "cloud_cover", "ground_level_pressure"
            ]
            target_column = "cloudburst_risk"
            X = df[feature_columns].values.astype(np.float32)
            y = df[target_column].values.astype(int)
            X_train_raw, _, y_train, _ = train_test_split(
                X, y, test_size=args.test_size, random_state=args.seed, stratify=y
            )
            scaler = MinMaxScaler().fit(X_train_raw)
            joblib.dump(scaler, "cloudburst_minmax_scaler.pkl")
            X_train_scaled = scaler.transform(X_train_raw)
            relay = {"X_train_scaled": X_train_scaled, "y_train": y_train}

        if args.train_rl:
            X_train_scaled = relay["X_train_scaled"]
            y_train = relay["y_train"]
            train_rl_agent(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                rl_timesteps=args.rl_timesteps,
                r_correct=args.r_correct,
                r_wrong=args.r_wrong
            )

    elif args.mode == "infer_rl":
        rl_infer_on_csv(args.csv, head=args.head)

if __name__ == "__main__":
    main()
