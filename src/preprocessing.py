import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD NORMALIZED DATA
# ==========================================

df = pd.read_csv("data/normalized_sensor_data.csv")

print("===== NORMALIZED DATA =====")
print(df.head())


# ==========================================
# 2. ADD NOISE
# ==========================================

noise_factor = 0.1

noise = np.random.normal(
    0,
    noise_factor,
    df.shape
)

noisy_data = df + noise

# Keep values between 0 and 1
noisy_data = np.clip(
    noisy_data,
    0,
    1
)


# ==========================================
# 3. CREATE NOISY DATAFRAME
# ==========================================

noisy_df = pd.DataFrame(
    noisy_data,
    columns=df.columns
)

print("\n===== NOISY DATA =====")
print(noisy_df.head())


# ==========================================
# 4. SAVE NOISY DATA
# ==========================================

noisy_df.to_csv(
    "data/noisy_sensor_data.csv",
    index=False
)


# ==========================================
# 5. PLOT CLEAN VS NOISY TEMPERATURE
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["temperature"],
    label="Clean Temperature"
)

plt.plot(
    noisy_df["temperature"],
    label="Noisy Temperature"
)

plt.title("Clean vs Noisy Temperature")
plt.xlabel("Time")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/noisy_vs_clean.png"
)

plt.show()


# ==========================================
# 6. COMPLETED
# ==========================================

print("\n===================================")
print("NOISE ADDITION COMPLETED!")
print("===================================")

print("\nNoisy data saved as:")
print("data/noisy_sensor_data.csv")

print("\nGraph saved as:")
print("results/noisy_vs_clean.png")