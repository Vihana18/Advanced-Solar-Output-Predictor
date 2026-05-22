Python
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import pvlib
import datetime

# 1. PAGE SETUP WITH CUSTOM DESIGN CONFIGURATION
st.set_page_config(
    page_title="Smart Solar Energy Output Predictor", 
    layout="centered"
)

# INJECTING CUSTOM CSS FOR PASTEL MINT GREEN THEME
# This styles the main body background and adjusts text colors to compliment it
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #e8f5e9;
    }
    
    /* Global Text Styling for Readability */
    p, span, label {
        color: #2c3e50 !important;
    }
    
    /* Custom Headers to compliment pastel green */
    .main-title {
        color: #1b5e20;
        font-size: 2.3rem;
        font-weight: bold;
        text-align: center;
        white-space: nowrap; /* Forces title onto a single line */
        margin-bottom: 5px;
    }
    
    .section-heading {
        color: #2e7d32;
        margin-top: 20px;
        font-weight: 600;
    }
    
    /* Soft white/mint card wrapper for metrics to pop against the background */
    div[data-testid="stMetricValue"] {
        color: #1b5e20 !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Single line title with no icons
st.markdown("<div class='main-title'>Smart Solar Energy Output Predictor</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; color: #4b6584;'>Empowering Sustainable Micro-Generation via Intelligent AI Analytics</p>", unsafe_allow_html=True)
st.write("This platform pairs a trained machine learning backend with real-time solar tracking and thermal physics constraints.")

# 2. CACHE THE MODEL
@st.cache_resource
def load_solar_brain():
    return joblib.load('advanced_solar_brain.pkl')

ai_brain = load_solar_brain()

# 3. SIDEBAR USER INTERFACE (Kept exactly as it was)
st.sidebar.markdown("<h2 style='color: #1b5e20;'>⚙️ Hardware Profile</h2>", unsafe_allow_html=True)
PANEL_MAX_CAPACITY = st.sidebar.number_input("Maximum Panel Capacity (Watts)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
PANEL_SURFACE_TILT = st.sidebar.slider("Panel Tilt Angle (0° = flat, 90° = vertical wall)", 0, 90, 25)
PANEL_SURFACE_AZIMUTH = st.sidebar.slider("Compass Facing Angle (180° = South, 90° = East)", 0, 360, 180)

# Global Metrics
API_KEY = "81b04a4cf48bfc77710a24cb03e6915c"
LOCATION = "Dubai"
LATITUDE = 25.2048
LONGITUDE = 55.2708
TEMP_COEFFICIENT = -0.0035

# 4. LIVE METEOROLOGICAL DATA RETRIEVAL
url = f"https://api.openweathermap.org/data/2.5/weather?q={LOCATION}&appid={API_KEY}&units=metric"

try:
    response = requests.get(url).json()
    
    if response.get("cod") == 200:
        temp = response['main']['temp']
        clouds = response['clouds']['all']
        humidity = response['main']['humidity']
        pressure = response['main']['pressure']
        rain = response.get('rain', {}).get('1h', 0)
        
        # Display crisp metrics
        st.markdown("<div class='section-heading'>### Live Environmental Vectors</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Ambient Temp", f"{temp}°C")
        col2.metric("Cloud Layer", f"{clouds}%")
        col3.metric("Relative Humidity", f"{humidity}%")
        
        # 5. CELESTIAL POSITIONING MATH
        current_time = pd.Timestamp(datetime.datetime.utcnow(), tz='UTC')
        sol_pos = pvlib.solarposition.get_solarposition(current_time, LATITUDE, LONGITUDE)
        live_zenith = sol_pos['apparent_zenith'].iloc[0]
        live_azimuth = sol_pos['azimuth'].iloc[0]
        
        aoi = pvlib.irradiance.aoi(PANEL_SURFACE_TILT, PANEL_SURFACE_AZIMUTH, live_zenith, live_azimuth)
        
        # 6. HYBRID MACHINE LEARNING & PHYSICS ENGINE
        estimated_ghi = 950 * (1 - (0.0075 * clouds))
        clues = [[estimated_ghi, clouds, temp, live_zenith, live_azimuth, humidity, pressure, rain, PANEL_MAX_CAPACITY]]
        raw_prediction = ai_brain.predict(clues)[0]
        
        # Geometric Alignment Filter
        geometric_efficiency = np.cos(np.radians(aoi))
        if geometric_efficiency < 0:
            geometric_efficiency = 0.0
            
        # Silicon Thermal Degradation Filter
        heat_loss_multiplier = 1.0
        if temp > 25:
            heat_loss_multiplier = 1.0 + (TEMP_COEFFICIENT * (temp - 25))
            
        # Composite Net Output
        tailored_wattage = raw_prediction * geometric_efficiency * heat_loss_multiplier
        
        # Physical boundary validation clamps
        if tailored_wattage > PANEL_MAX_CAPACITY: tailored_wattage = PANEL_MAX_CAPACITY
        if tailored_wattage < 0: tailored_wattage = 0.0

        # 7. RENDERING RENEWABLE OUTPUT GRAPHICS
        st.markdown("---")
        st.markdown("<div class='section-heading'>### Predicted Clean Energy Yield</div>", unsafe_allow_html=True)
        
        # Big clean highlighted output block
        st.success(f"## **Estimated Output: {tailored_wattage:.2f} Watts**")
        
        # Real-time eco-efficiency feedback cards
        if aoi < 45:
            st.info("🎯 **Optimal Angle Optimization:** Your structural layout has excellent alignment with current solar coordinates, maximizing renewable capture.")
        else:
            st.warning("⚠️ **Geometric Shading/Reflection Losses:** High mismatch angle detected. Consider adjusting hardware tilt dynamically to minimize reflection.")
            
    else:
        st.error(f"Weather API Error: {response.get('message')}")
except Exception as e:
    st.error(f"System Tracking Failure: {e}")