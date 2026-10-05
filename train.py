import pandas as pd
from src.model import create_autoencoder

print("===== STARTING TRAINING =====")

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

print("Training data loaded!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

autoencoder = create_autoencoder(X_train.shape[1])

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("===== TRAINING MODEL =====")

history = autoencoder.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test, y_test),
    verbose=1
)

autoencoder.save("models/denoising_autoencoder.keras")

print("===================================")
print("MODEL TRAINING COMPLETED!")
print("===================================")
print("Model saved at:")
print("models/denoising_autoencoder.keras")
