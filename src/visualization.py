import pandas as pd
import matplotlib.pyplot as plt

# Load data
clean_data = pd.read_csv("data/y_test.csv")
noisy_data = pd.read_csv("data/X_test.csv")
denoised_data = pd.read_csv("data/denoised_sensor_data.csv")

# ==========================================
# CREATE COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    clean_data["temperature"].values,
    label="Original / Clean"
)

plt.plot(
    noisy_data["temperature"].values,
    label="Noisy"
)

plt.plot(
    denoised_data["temperature"].values,
    label="Denoised"
)

plt.title("Original vs Noisy vs Denoised Temperature")
plt.xlabel("Test Sample")
plt.ylabel("Normalized Temperature")

plt.legend()
plt.grid(True)

# Save graph
plt.savefig(
    "results/original_noisy_denoised.png"
)

plt.show()

print("\n===================================")
print("VISUALIZATION COMPLETED!")
print("===================================")

print("\nGraph saved at:")
print("results/original_noisy_denoised.png")

