import pandas as pd
import numpy as np
import pickle
import sys

# Load the trained model from the model folder
model_path = "model/financial_forecasting_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Function to create features from raw input data
def create_features(close_lag1, ma_5):
    """
    Given the previous day's closing price and the 5-day moving average,
    create a feature array for prediction.
    """
    return np.array([[close_lag1, ma_5]])

# Sample input: You can update these values with real data or arguments
sample_close_lag1 = float(input("Enter the previous day's closing price: "))
sample_ma_5 = float(input("Enter the 5-day moving average: "))

# Create features from sample input
features = create_features(sample_close_lag1, sample_ma_5)

# Use the model to predict the closing price
predicted_close = model.predict(features)

print(f"Predicted closing price: {predicted_close[0]}")