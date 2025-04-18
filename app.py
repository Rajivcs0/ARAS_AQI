import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------- Load the dataset -------------------
try:
    df = pd.read_csv("/content/air-quality-india.csv")
except FileNotFoundError:
    print("❌ File not found. Make sure the CSV file path is correct.")
    exit()

# ------------------- Clean and Prepare Data -------------------
# Strip any extra spaces from column names
df.columns = df.columns.str.strip()

# Check if 'Timestamp' column exists
if "Timestamp" not in df.columns:
    print("❌ 'Timestamp' column is missing from the dataset.")
    print("Available columns:", df.columns.tolist())
    exit()

# Convert Timestamp to datetime safely
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
df = df.dropna(subset=["Timestamp"])  # Drop rows with invalid/missing timestamps

# Extract date/time features
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Hour"] = df["Timestamp"].dt.hour

# Drop rows with missing target values
df = df.dropna(subset=["PM2.5"])

# ------------------- Feature Selection -------------------
X = df[["Year", "Month", "Day", "Hour"]]
y = df["PM2.5"]

# ------------------- Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------- Train RandomForest -------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------- Evaluate the model -------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print("\n✅ Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")

# ------------------- Save the model -------------------
joblib.dump(model, "pm25_model.pkl")
print("\n✅ Model saved as 'pm25_model.pkl'.")

# ------------------- Take user input for prediction -------------------
print("\n🔍 Enter details to predict PM2.5 level:")

try:
    year = int(input("Enter Year (e.g. 2020): "))
    month = int(input("Enter Month (1-12): "))
    day = int(input("Enter Day (1-31): "))
    hour = int(input("Enter Hour (0-23): "))
except ValueError:
    print("❌ Invalid input. Please enter valid integers.")
    exit()

# Load the model
try:
    model = joblib.load("pm25_model.pkl")
except FileNotFoundError:
    print("❌ Model file not found. Make sure 'pm25_model.pkl' exists.")
    exit()

# Predict PM2.5
user_input = np.array([[year, month, day, hour]])
predicted_pm25 = model.predict(user_input)[0]

# ------------------- AQI Evaluation -------------------
def get_aqi_status(pm25):
    if pm25 <= 12:
        return "🟢 AQI is good."
    elif 12 < pm25 <= 55:
        return "🟡 AQI is sensitive for kids, elders and patients."
    elif 55 < pm25 <= 100:
        return "🟠 AQI is too dangerous so be careful to go outside."
    else:
        return "🔴 AQI is hazardous. Avoid breathing outside air."

# Output result
aqi_status = get_aqi_status(predicted_pm25)
print(f"\n📍 Predicted PM2.5 Level: {predicted_pm25:.2f} µg/m³")
print(f"📋 AQI Status: {aqi_status}")
