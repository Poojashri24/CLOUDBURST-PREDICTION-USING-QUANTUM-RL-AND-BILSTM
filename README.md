# 🌩 Hybrid BiLSTM + Quantum CNN + Reinforcement Learning for Cloudburst Prediction

## 📌 Overview
This project implements a **hybrid deep learning model** combining:
- **BiLSTM (Bidirectional Long Short-Term Memory)** – for temporal sequence learning
- **Quantum CNN (Quantum Convolutional Neural Network)** – for enhanced feature extraction using quantum computing concepts
- **Reinforcement Learning Agent** – for adaptive model tuning and decision-making

The goal is to predict **cloudburst events** using meteorological data and advanced AI techniques.  
The dataset includes both **high-risk** and **low-risk** weather samples with parameters such as:
- Temperature
- Humidity
- Pressure
- Wind Speed
- Cloud Cover
- Ground Level Pressure

The model is capable of:
1. Training and evaluating prediction accuracy
2. Outputting **training accuracy** without retraining
3. Outputting **testing accuracy** after evaluation

---

## ⚙️ Installation
1. **Clone this repository**
```bash
git clone https://github.com/Poojashri24/CLOUDBURST-PREDICTION-USING-QUANTUM-RL-AND-BILSTM
cd cloudburst-hybrid-rl
```
2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```
3.**Install dependencies**

📦 Requirements

  - Python 3.8+
  
  - TensorFlow / Keras
  
  - NumPy
  
  - PennyLane (for Quantum CNN)
  
  - scikit-learn
  
  - Gym (for RL agent)
  
  - Matplotlib (optional for visualization)
  
4.**Train the model**

```bash
python training.py
```

🧠 Model Details

**BiLSTM Layer**

- Captures sequential dependencies in time-series meteorological data.

- Processes input in both forward and backward directions.

**Quantum CNN**

- Uses variational quantum circuits for convolution operations.

- Extracts high-dimensional features beyond classical CNN capabilities.

**Reinforcement Learning Agent**

- Monitors predictions and rewards the model for correct decisions.

- Can adjust learning rate, dropout, and activation functions dynamically.

5.**Test the model**
``` bash
pyhton testing.py
```
---

**📈 Example Results**

After training on sample weather data:

- Train Accuracy: 0.98
- Test Accuracy: 0.94



