import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SensorClean",
    page_icon="📡",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📡 SensorClean")

st.subheader(
    "Sensor Data Noise Reduction Using Autoencoders"
)

st.write(
    "Upload sensor data and use the trained "
    "autoencoder to reduce noise."
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_trained_model():

    return load_model(
        "models/denoising_autoencoder.keras"
    )


# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Sensor CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    # Read uploaded data
    df = pd.read_csv(uploaded_file)

    st.success("Sensor data uploaded successfully!")

    st.subheader("Original Sensor Data")

    st.dataframe(df)


    # ======================================
    # SELECT SENSOR COLUMNS
    # ======================================

    sensor_columns = [
        "temperature",
        "humidity",
        "pressure"
    ]

    if all(
        column in df.columns
        for column in sensor_columns
    ):

        sensor_data = df[sensor_columns]


        # ==================================
        # NORMALIZE
        # ==================================

        scaler = MinMaxScaler()

        normalized_data = scaler.fit_transform(
            sensor_data
        )


        # ==================================
        # LOAD MODEL
        # ==================================

        model = load_trained_model()


        # ==================================
        # DENOISE
        # ==================================

        denoised_normalized = model.predict(
            normalized_data
        )


        # Convert back to original scale
        denoised_data = scaler.inverse_transform(
            denoised_normalized
        )


        denoised_df = pd.DataFrame(
            denoised_data,
            columns=sensor_columns
        )


        # ==================================
        # DISPLAY RESULT
        # ==================================

        st.subheader("Denoised Sensor Data")

        st.dataframe(
            denoised_df
        )


        # ==================================
        # TEMPERATURE GRAPH
        # ==================================

        st.subheader(
            "Temperature: Original vs Denoised"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            sensor_data["temperature"],
            label="Original"
        )

        ax.plot(
            denoised_df["temperature"],
            label="Denoised"
        )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Temperature")

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


        # ==================================
        # HUMIDITY GRAPH
        # ==================================

        st.subheader(
            "Humidity: Original vs Denoised"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            sensor_data["humidity"],
            label="Original"
        )

        ax.plot(
            denoised_df["humidity"],
            label="Denoised"
        )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Humidity")

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


        # ==================================
        # PRESSURE GRAPH
        # ==================================

        st.subheader(
            "Pressure: Original vs Denoised"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            sensor_data["pressure"],
            label="Original"
        )

        ax.plot(
            denoised_df["pressure"],
            label="Denoised"
        )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Pressure")

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


        # ==================================
        # DOWNLOAD RESULT
        # ==================================

        csv = denoised_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Denoised Data",
            data=csv,
            file_name="denoised_sensor_data.csv",
            mime="text/csv"
        )

    else:

        st.error(
            "CSV must contain temperature, "
            "humidity and pressure columns."
        )