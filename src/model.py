import tensorflow as tf
from tensorflow.keras import layers, models


# ==========================================
# 1. CREATE DENOISING AUTOENCODER
# ==========================================

def create_autoencoder(input_size):

    model = models.Sequential([
        
        # Encoder
        layers.Input(shape=(input_size,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),

        # Decoder
        layers.Dense(16, activation="relu"),
        layers.Dense(input_size, activation="sigmoid")
    ])

    return model


# ==========================================
# 2. CREATE MODEL
# ==========================================

autoencoder = create_autoencoder(3)


# ==========================================
# 3. COMPILE MODEL
# ==========================================

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


# ==========================================
# 4. DISPLAY MODEL
# ==========================================

autoencoder.summary()