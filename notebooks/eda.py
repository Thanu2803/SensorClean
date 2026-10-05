import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/sensor_data.csv")

print("===== DATASET =====")
print(df.head())

print("\n===== DATASET INFORMATION =====")
print(df.info())

print("\n===== STATISTICS =====")
print(df.describe())


# ==========================================
# 2. TEMPERATURE GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp"],
    df["temperature"],
    marker="o"
)

plt.title("Temperature Sensor Data")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.grid(True)

plt.savefig("results/temperature.png")

plt.show()


# ==========================================
# 3. HUMIDITY GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp"],
    df["humidity"],
    marker="o"
)

plt.title("Humidity Sensor Data")
plt.xlabel("Time")
plt.ylabel("Humidity")
plt.grid(True)

plt.savefig("results/humidity.png")

plt.show()


# ==========================================
# 4. PRESSURE GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp"],
    df["pressure"],
    marker="o"
)

plt.title("Pressure Sensor Data")
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.grid(True)

plt.savefig("results/pressure.png")

plt.show()


# ==========================================
# 5. CORRELATION HEATMAP
# ==========================================

numeric_columns = [
    "temperature",
    "humidity",
    "pressure"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Sensor Correlation")

plt.savefig("results/correlation.png")

plt.show()


# ==========================================
# 6. EDA COMPLETED
# ==========================================

print("\n===================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("===================================")

print("\nGraphs saved in the results folder:")
print("1. temperature.png")
print("2. humidity.png")
print("3. pressure.png")
print("4. correlation.png")
