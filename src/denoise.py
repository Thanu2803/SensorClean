import pandas as pd
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/denoising_autoencoder.keras")

print("Trained model loaded successfully!")

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Convert noisy data to NumPy
X_test_array = X_test.values

# Predict denoised data
denoised_data = model.predict(X_test_array)

# Convert to DataFrame
denoised_df = pd.DataFrame(
    denoised_data,
    columns=X_test.columns
)

print("\n===== DENOISED DATA =====")
print(denoised_df.head())

# Save denoised data
denoised_df.to_csv(
    "data/denoised_sensor_data.csv",
    index=False
)

print("\n===================================")
print("DENOISING COMPLETED!")
print("===================================")

print("\nDenoised data saved at:")
print("data/denoised_sensor_data.csv")