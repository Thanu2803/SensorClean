# SensorClean: Sensor Data Noise Reduction Using Autoencoders

## 📌 Project Overview

SensorClean is a deep learning project designed to reduce noise from sensor data using a Deep Denoising Autoencoder.

The system takes noisy sensor readings and reconstructs a cleaner version of the original signal.

## 🎯 Objectives

- Clean sensor data
- Analyze sensor data using EDA
- Normalize sensor values
- Add artificial noise
- Train a denoising autoencoder
- Reconstruct clean sensor signals
- Evaluate denoising performance
- Visualize original, noisy and denoised signals
- Provide a Streamlit web application

## 🔄 Project Workflow

Dataset
↓
Data Cleaning
↓
EDA
↓
Normalization
↓
Noise Addition
↓
Train/Test Split
↓
Autoencoder Training
↓
Denoising
↓
Evaluation
↓
Visualization
↓
Streamlit Application

## 🧠 Autoencoder Architecture

Input Layer: 3 sensor features

Encoder:
- Dense Layer: 16 neurons
- Dense Layer: 8 neurons

Decoder:
- Dense Layer: 16 neurons
- Output Layer: 3 neurons

Activation Function: ReLU

Output activation: Sigmoid

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

## 📊 Sensors Used

- Temperature
- Humidity
- Pressure

## 📈 Evaluation Metrics

The model is evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Signal-to-Noise Ratio (SNR)

Lower MSE, MAE and RMSE indicate better reconstruction.

Higher SNR indicates better signal quality.

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- SciPy
- Streamlit

## ▶️ How to Run

### 1. Create virtual environment

```bash
python -m venv venv