import numpy as np
import pandas as pd


# Make results repeatable
np.random.seed(42)

# Number of sensor readings
n = 1000

# Timestamp
timestamp = np.arange(1, n + 1)


# ==========================================
# CREATE REALISTIC SENSOR SIGNALS
# ==========================================

temperature = (
    25
    + 3 * np.sin(np.linspace(0, 10 * np.pi, n))
    + np.random.normal(0, 0.15, n)
)

humidity = (
    60
    + 5 * np.sin(np.linspace(0, 8 * np.pi, n))
    + np.random.normal(0, 0.3, n)
)

pressure = (
    1012
    + 2 * np.sin(np.linspace(0, 6 * np.pi, n))
    + np.random.normal(0, 0.1, n)
)


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame({
    "timestamp": timestamp,
    "temperature": temperature,
    "humidity": humidity,
    "pressure": pressure
})


# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(
    "data/sensor_data.csv",
    index=False
)


print("===================================")
print("LARGE DATASET CREATED!")
print("===================================")

print("Number of samples:", len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved as:")
print("data/sensor_data.csv")