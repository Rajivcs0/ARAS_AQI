import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("https://github.com/Rajivcs0/ARAS_AQI/blob/main/air-quality-india.csv")

# Convert Timestamp to datetime format
# Convert Timestamp to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='%d-%m-%Y') 
# Select features and target variable
X = df[["Year", "Month", "Day", "Hour"]]
y = df["PM2.5"]

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")

import joblib
import numpy as np

# ---------- Save the model before running this (from training script) ----------
joblib.dump(model, "pm25_model.pkl")

# ----------- Take User Input and Predict PM2.5 ----------- #
print("\nEnter details to predict PM2.5 level:")

# Get user inputs
try:
    year = int(input("Enter Year (e.g. 2020): "))
    month = int(input("Enter Month (1-12): "))
    day = int(input("Enter Day (1-31): "))
    hour = int(input("Enter Hour (0-23): "))
except ValueError:
    print("Invalid input. Please enter valid integers.")
    exit()

# Load the trained model
try:
    model = joblib.load("pm25_model.pkl")
except FileNotFoundError:
    print("Model file not found. Make sure 'pm25_model.pkl' exists.")
    exit()

# Make prediction
user_input = np.array([[year, month, day, hour]])
predicted_pm25 = model.predict(user_input)[0]

# AQI Condition Output
def get_aqi_status(pm25):
    if pm25 <= 12:
        return "AQI is good."
    elif 12 < pm25 <= 55:
        return "AQI is sensitive for kids, elders and patients."
    elif 55 < pm25 <= 100:
        return "AQI is too dangerous so be careful to go outside."
    else:
        return "AQI is in too hazardous and dangerous for everyone to avoid air breathing outside."

# Display result
aqi_status = get_aqi_status(predicted_pm25)
print(f"\nPredicted PM2.5 Level: {predicted_pm25:.2f}")
print(f"AQI Status: {aqi_status}")
