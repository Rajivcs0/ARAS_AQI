import streamlit as st
import numpy as np
import joblib
from datetime import datetime

import streamlit as st
import pandas as pd

st.set_page_config(page_title="PM2.5 Prediction", page_icon="🌫", layout="centered")

st.title("PM2.5 Prediction App")


# Load the trained model (cached for performance)
@st.cache_resource
def load_model():
    return joblib.load("/Users/thril/Desktop/ARAS1-AQI/pm25_model.pkl")

model = load_model()

# Page configuration
#st.set_page_config(page_title="PM2.5 Prediction", page_icon="🌫", layout="centered")

# Title and description
st.title("🌫 PM2.5 Air Quality Prediction")
st.markdown("""
This app predicts *PM2.5 levels* based on the date and time.
Use the sidebar to input values and hit *Enter* or click *Predict*.
""")

# Sidebar form for user input
with st.sidebar.form("pm25_form"):
    st.header("Date & Time Input")

    year = st.number_input("Year", min_value=2017, max_value=2025, step=1, value=2022)
    month = st.number_input("Month", min_value=1, max_value=12, step=1, value=1)
    day = st.number_input("Day", min_value=1, max_value=31, step=1, value=1)
    hour = st.number_input("Hour", min_value=0, max_value=23, step=1, value=0)

    submitted = st.form_submit_button("🎯 Predict PM2.5")

# On form submit
if submitted:
    try:
        # Validate input date
        valid_date = datetime(year, month, day, hour)

        # Prepare input for model
        input_features = np.array([[year, month, day, hour]])
        prediction = model.predict(input_features)[0]

        # Show result
        st.subheader("📈 Prediction Result")
        st.success(f"Predicted PM2.5 Level: {prediction:.2f} µg/m³")
        st.caption(f"Prediction for: {valid_date.strftime('%B %d, %Y at %I:%M %p')}")

        # Optional air quality interpretation
        if prediction <= 50:
            st.info("Air Quality: Good 😊")
        elif prediction <= 100:
            st.warning("Air Quality: Moderate 😐")
        else:
            st.error("Air Quality: Poor 😷")

    except ValueError:
        st.error("❌ Invalid date! Please enter a valid date combination.")

# Footer
st.markdown("---")
st.caption("Built by [Rajiv] | Model: Random Forest | Data: Air Quality India")