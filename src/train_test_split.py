import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# 1. LOAD CLEAN AND NOISY DATA
# ==========================================

clean_data = pd.read_csv(
    "data/normalized_sensor_data.csv"
)

noisy_data = pd.read_csv(
    "data/noisy_sensor_data.csv"
)


# ==========================================
# 2. SPLIT THE DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    noisy_data,
    clean_data,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 3. DISPLAY SHAPES
# ==========================================

print("===== TRAINING DATA =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n===== TESTING DATA =====")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# 4. SAVE DATA
# ==========================================

X_train.to_csv(
    "data/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/y_test.csv",
    index=False
)


# ==========================================
# 5. COMPLETED
# ==========================================

print("\n===================================")
print("TRAIN/TEST SPLIT COMPLETED!")
print("===================================")
