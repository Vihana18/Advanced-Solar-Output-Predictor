import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import pvlib

# ==========================================
# 1. PAGE SETUP & DESIGN
# ==========================================
st.set_page_config(page_title="Dubai Solar Predictor", layout="centered")
st.title("☀️ Dubai Smart Solar Predictive Platform")
st.write("Real-time AI forecasting mapped with geometric and thermal physics filters.")

# ==========================================
# 2. CACHE THE AI BRAIN (Crucial for Speed!)
# ==========================================
@st.cache_resource
def load_solar_brain():
    # This stops Streamlit from re-loading the heavy file on every single click
    return joblib.load('advanced_solar_brain.pkl')

ai_brain = load_solar_brain()

# ==========================================
# 3. INTERACTIVE SIDEBAR (User Inputs)
# ==========================================
st.sidebar.header("🛠️ Hardware Configuration Profile")

PANEL_MAX_CAPACITY = st.sidebar.number_input("Maximum Panel Capacity (Watts)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
PANEL_SURFACE_TILT = st.sidebar.slider("Panel Tilt Angle (0° = flat, 90° = vertical)", 0, 90, 25)
PANEL_SURFACE_AZIMUTH = st.sidebar.slider("Compass Facing Angle (180° = South, 90° = East)", 0, 360, 180)

# Constants
API_KEY = "81b04a4cf48bfc77710a24cb03e6915c"
LOCATION = "Dubai"
LATITUDE = 25.2048
LONGITUDE = 55.2708
TEMP_COEFFICIENT = -0.0035

# ==========================================
# 4. LIVE WEATHER API FETCH
# ==========================================
url = f"https://api.openweathermap.org/data/2.5/weather?q={LOCATION}&appid={API_KEY}&units=metric"

try:
    response = requests.get(url).json()
    
    if response.get("cod") == 200:
        # Extract live features
        temp = response['main']['temp']
        clouds = response['clouds']['all']
        humidity = response['main']['humidity']
        pressure = response['main']['pressure']
        rain = response.get('rain', {}).get('1h', 0)
        
        # Display Current Weather Metrics in clean columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{temp}°C")
        col2.metric("Cloud Cover", f"{clouds}%")
        col3.metric("Humidity", f"{humidity}%")
        
        # ==========================================
        # 5. CELESTIAL SOLAR GEOMETRY
        # ==========================================
        current_time = pd.Timestamp(datetime.utcnow(), tz='UTC')
        sol_pos = pvlib.solarposition.get_solarposition(current_time, LATITUDE, LONGITUDE)
        live_zenith = sol_pos['apparent_zenith'].iloc[0]
        live_azimuth = sol_pos['azimuth'].iloc[0]
        
        # Calculate Angle of Incidence (AOI)
        aoi = pvlib.irradiance.aoi(PANEL_SURFACE_TILT, PANEL_SURFACE_AZIMUTH, live_zenith, live_azimuth)
        
        # ==========================================
        # 6. RUN AI BRAIN + PHYSICS FILTERS
        # ==========================================
        # Proxy GHI calculation based on cloud layer density
        estimated_ghi = 950 * (1 - (0.0075 * clouds))
        
        # Get raw machine learning baseline prediction
        clues = [[estimated_ghi, clouds, temp, live_zenith, live_azimuth, humidity, pressure, rain, PANEL_MAX_CAPACITY]]
        raw_prediction = ai_brain.predict(clues)[0]
        
        # Apply Geometric Penalty Factor
        geometric_efficiency = np.cos(np.radians(aoi))
        if geometric_efficiency < 0:
            geometric_efficiency = 0
            
        # Apply Thermal Heat Degradation Factor
        heat_loss_multiplier = 1.0
        if temp > 25:
            heat_loss_multiplier = 1.0 + (TEMP_COEFFICIENT * (temp - 25))
            
        # Process composite tailored wattage
        tailored_wattage = raw_prediction * geometric_efficiency * heat_loss_multiplier
        
        # Physical boundary clamp
        if tailored_wattage > PANEL_MAX_CAPACITY:
            tailored_wattage = PANEL_MAX_CAPACITY
        if tailored_wattage < 0:
            tailored_wattage = 0.0

        # ==========================================
        # 7. DISPLAY FINAL PREDICTION RESULTS
        # ==========================================
        st.markdown("---")
        st.subheader("📊 Live Predictive System Output")
        
        st.success(f"### **Estimated Generation: {tailored_wattage:.2f} Watts**")
        
        # Give immediate feedback on alignment quality
        if aoi < 45:
            st.info("🎯 **Excellent Structural Alignment:** The sunbeams are tracking efficiently against your tilt configuration.")
        else:
            st.warning("⚠️ **High Geometric Mismatch:** High reflection/shading losses due to your panel orientation angle.")
            
    else:
        st.error(f"API Error from OpenWeatherMap: {response.get('message')}")

except Exception as e:
    st.error(f"Could not connect to live weather tracking server: {e}")