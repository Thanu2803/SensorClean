import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
clean_data = pd.read_csv("data/y_test.csv")
noisy_data = pd.read_csv("data/X_test.csv")
denoised_data = pd.read_csv("data/denoised_sensor_data.csv")

# Convert to NumPy
clean = clean_data.values
noisy = noisy_data.values
denoised = denoised_data.values

# ==========================================
# MSE
# ==========================================

mse = mean_squared_error(clean, denoised)

# ==========================================
# MAE
# ==========================================

mae = mean_absolute_error(clean, denoised)

# ==========================================
# RMSE
# ==========================================

rmse = np.sqrt(mse)

# ==========================================
# SNR
# ==========================================

signal_power = np.mean(clean ** 2)
noise_power = np.mean((clean - denoised) ** 2)

snr = 10 * np.log10(signal_power / noise_power)

# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("MODEL EVALUATION RESULTS")
print("===================================")

print(f"MSE  : {mse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"SNR  : {snr:.2f} dB")

print("\n===================================")
print("EVALUATION COMPLETED!")
print("===================================")