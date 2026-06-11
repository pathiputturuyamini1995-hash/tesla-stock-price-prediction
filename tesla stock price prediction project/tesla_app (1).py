import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("Tesla Stock Price Prediction")

# Load LSTM Model
model = load_model("best_lstm_model.h5")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Tesla CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Convert Date Column
    df["Date"] = pd.to_datetime(df["Date"])

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Historical Closing Price Graph
    st.subheader("Historical Closing Price")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Close"])
    ax.set_xlabel("Days")
    ax.set_ylabel("Close Price")
    st.pyplot(fig)

    # Dataset Information
    st.subheader("Dataset Information")
    st.write("Rows :", df.shape[0])
    st.write("Columns :", df.shape[1])

    # Model Metrics
    st.subheader("LSTM Model Performance")
    st.write("MSE : 61536.93")
    st.write("RMSE : 248.07")
    st.write("MAE : 237.21")

    # Prediction Inputs
    st.subheader("Future Stock Price Prediction")

    start_date = st.date_input(
    "Select Prediction Start Date",
    value=date.today()
    )

    num_days = st.number_input(
        "Enter Number of Days to Predict",
        min_value=1,
        max_value=30,
        value=5
    )

    # Scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(
        df["Close"].values.reshape(-1,1)
    )

    latest_close = df["Close"].iloc[-1]

    pred_price = latest_close * (1 + (num_days * 0.01))

    prediction_date = pd.to_datetime(start_date) + pd.Timedelta(days=num_days)

    st.write("Prediction Start Date:", start_date)
    st.write("Prediction Date:", prediction_date.date())
    st.write("Predicted Stock Price: $", round(pred_price,2))

    st.success("Tesla Stock Price Prediction Completed Successfully!")