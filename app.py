import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from datetime import datetime

# Load trained Random Forest model
loaded_rf = joblib.load("RandomForest_model.pkl")

# Define feature order
feature_order = ['pm10', 'Rain_PM10', 'Fog_PM10', 'AQI_lag_1', 'Wind_PM10', 'AQI_7_day_avg',
                 'AQI_lag_2', 'AQI_lag_3', 'AQI_lag_4', 'Year', 'AQI_lag_5', 'AQI_lag_7',
                 'AQI_lag_6', 'Humidity_Temp', 'H', 'Temp_SO2', 'Tm', 'Month', 'Temp_NO2', 'Rain_SO2']

# Fetch real-time AQI data

def fetch_realtime_aqi():
    url = "https://api.waqi.info/feed/here/?token=e957378050e921377992405d2c4255ce56922963"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Check if the necessary fields are present in the response
        if "data" in data and "aqi" in data["data"]:
            return data["data"]["aqi"]
    return None
# Fetch real-time weather data for Bengaluru, Karnataka
def fetch_realtime_weather():
    url = "https://www.meteosource.com/api/v1/free/point?place_id=bengaluru&sections=all&timezone=UTC&language=en&units=metric&key=5f3luxisgmil3gtni0njkz5pltvsnd8lm43vpmib"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# Get AQI category and recommendations
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory. No precautions needed."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable, but sensitive individuals may experience minor issues."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Children, elderly, and those with respiratory conditions should limit outdoor activity."
    elif aqi <= 200:
        return "Unhealthy", "Everyone may experience adverse effects; sensitive groups should avoid outdoor exertion."
    elif aqi <= 300:
        return "Very Unhealthy", "Health warnings issued; everyone should limit outdoor activity."
    else:
        return "Hazardous", "Serious health risks for everyone; avoid outdoor exposure."

# Streamlit App
st.title("Air Quality Index (AQI) Prediction")

# Display real-time AQI
real_time_aqi = fetch_realtime_aqi()
if real_time_aqi is not None:
    st.subheader(f"Real-time AQI:  {real_time_aqi}")
    category, suggestion = get_aqi_category(real_time_aqi)
    st.write(f"Category: **{category}**")
    st.write(f"Suggestion: {suggestion}")
    
else:
    st.write("Unable to fetch real-time AQI data.")

# Display real-time weather data
weather_data = fetch_realtime_weather()
if weather_data:
    current_weather = weather_data.get("current", {})
    temperature = current_weather.get("temperature", "N/A")
    humidity = current_weather.get("humidity", "N/A")
    wind_speed = current_weather.get("wind", {}).get("speed", "N/A")
    feels_like = current_weather.get("feels_like", "N/A")
    
    st.subheader("Real-time Weather in Bengaluru, Karnataka")
    st.write(f"Temperature: {temperature}°C")
    st.write(f"Feels Like: {feels_like}°C")
    st.write(f"Humidity: {humidity}%")
    st.write(f"Wind Speed: {wind_speed} m/s")
    
st.subheader("Influence of Pollutants and Weather on AQI")
st.write("Enter air pollution and weather parameters to predict AQI.")

# Layout for input sliders like volume adjusters
col1, col2, col3 = st.columns(3)

with col1:
    pm10 = st.slider("PM10 (µg/m³)", 10.0, 200.0, 50.0)
    temp_min = st.slider("Min Temp (°C)", 10.0, 30.0, 20.0)
    temp_avg = st.slider("Avg Temp (°C)", 15.0, 35.0, 25.0)

with col2:
    rain = st.selectbox("Rainfall (0 or 1)", [0, 1], index=0)
    humidity = st.slider("Humidity (%)", 30.0, 100.0, 70.0)
    so2 = st.slider("SO₂ (µg/m³)", 0.0, 50.0, 10.0)

with col3:
    fog = st.selectbox("Fog Intensity (0 or 1)", [0, 1], index=0)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
    no2 = st.slider("NO₂ (µg/m³)", 0.0, 100.0, 20.0)

# Other inputs
year = st.number_input("Year", min_value=2016, max_value=2100, step=1, value=2025)
month = st.number_input("Month", min_value=1, max_value=12, step=1, value=1)

# AQI Lag values and 7-day average (default 69.3)
aqi_lags = {f"AQI_lag_{i}": st.number_input(f"AQI from {i} Days Ago", min_value=0.0, step=0.1, value=69.3) for i in range(1, 8)}
aq_7_day_avg = st.number_input("Average AQI of Last 7 Days", min_value=0.0, step=0.1, value=69.3)

# Derived features
rain_pm10 = rain * pm10
fog_pm10 = fog * pm10
wind_pm10 = wind_speed * pm10
humidity_temp = humidity * temp_avg
temp_so2 = temp_avg * so2
temp_no2 = temp_avg * no2
rain_so2 = rain * so2

# Create DataFrame
user_input = {"pm10": pm10, "Rain_PM10": rain_pm10, "Fog_PM10": fog_pm10, "Wind_PM10": wind_pm10, "AQI_7_day_avg": aq_7_day_avg, "Year": year, "Humidity_Temp": humidity_temp, "H": humidity, "Temp_SO2": temp_so2, "Tm": temp_min, "Month": month, "Temp_NO2": temp_no2, "Rain_SO2": rain_so2, **aqi_lags}
user_input_df = pd.DataFrame([user_input])
user_input_df = user_input_df[feature_order]  # Ensure column order matches

# Prediction
if st.button("Predict AQI"):
    prediction = loaded_rf.predict(user_input_df)[0]
    category, suggestion = get_aqi_category(prediction)
    
    st.subheader(f"Predicted AQI: {prediction:.2f}")
    st.write(f"Category: **{category}**")
    st.write(f"Suggestion: {suggestion}")
