import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# -------------------------------
# Load trained model + preprocessing tools
# -------------------------------
model = joblib.load("models/best_regressor_LightGBM.joblib")
imputer = joblib.load("models/imputer.joblib")
scaler = joblib.load("models/scaler.joblib")

st.title("🌡️ Heatwave & Temperature Prediction System")

st.write("Enter weather parameters to get next-day max temperature prediction.")

# -------------------------------
# Input fields
# -------------------------------
latitude = st.number_input("Latitude", value=19.0)
longitude = st.number_input("Longitude", value=72.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=6.5)
cloud_cover = st.number_input("Cloud Cover (%)", value=20.0)
precipitation_probability = st.number_input("Precipitation Probability (%)", value=10.0)
pressure_surface_level = st.number_input("Pressure (hPa)", value=1005.0)
dew_point = st.number_input("Dew Point (°C)", value=18.0)
uv_index = st.number_input("UV Index", value=7.0)
visibility = st.number_input("Visibility (km)", value=5.0)
rainfall = st.number_input("Rainfall (mm)", value=0.2)
min_temperature = st.number_input("Today's Min Temperature (°C)", value=28.0)
max_humidity = st.number_input("Max Humidity (%)", value=60.0)
min_humidity = st.number_input("Min Humidity (%)", value=30.0)

max_temp_lag1 = st.number_input("Yesterday Max Temp (°C)", value=35.0)
max_temp_lag2 = st.number_input("2 Days Back Max Temp (°C)", value=34.0)
max_temp_roll3 = (max_temp_lag1 + max_temp_lag2 + 33) / 3

month = datetime.now().month
dayofyear = datetime.now().timetuple().tm_yday
is_summer = 1 

# -------------------------------
# Create input DataFrame
# -------------------------------
input_df = pd.DataFrame([{
    'latitude': latitude,
    'longitude': longitude,
    'wind_speed': wind_speed,
    'cloud_cover': cloud_cover,
    'precipitation_probability': precipitation_probability,
    'pressure_surface_level': pressure_surface_level,
    'dew_point': dew_point,
    'uv_index': uv_index,
    'visibility': visibility,
    'rainfall': rainfall,
    'min_temperature': min_temperature,
    'max_humidity': max_humidity,
    'min_humidity': min_humidity,
    'max_temp_lag1': max_temp_lag1,
    'max_temp_lag2': max_temp_lag2,
    'max_temp_roll3': max_temp_roll3,
    'month': month,
    'dayofyear': dayofyear,
    'is_summer': is_summer
}])

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔮 Predict Temperature"):
    
    # Impute missing values
    input_imputed = imputer.transform(input_df)

    # No scaling for LightGBM (only Linear Regression needs scaler)
    prediction = model.predict(input_imputed)[0]

    st.subheader(f"🌡️ Predicted Next-Day Max Temperature: **{prediction:.2f} °C**")

    # Heatwave detection
    if prediction > 37:
        st.error("🔥 **Heatwave is POSSIBLE!**")
    else:
        st.success("❄️ **Heatwave NOT expected.**")
