
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

# INJECTING CUSTOM CSS FOR ADVANCED TYPOGRAPHY AND SIDEBAR STYLING
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #e8f5e9;
    }
    
    /* Make all standard paragraph and description text bigger */
    p, span, label, .stMarkdown {
        font-size: 1.15rem !important;
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    /* Single line title styling */
    .main-title {
        color: #1b5e20;
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        white-space: nowrap;
        margin-bottom: 8px;
    }
    
    /* Enhanced Subheading Typography */
    .custom-subheading {
        color: #1b5e20 !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 5px;
    }
    
    /* Make metric text display much larger */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #1b5e20 !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetricLabel"] p {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #388e3c !important;
    }
    
    /* FORCE SIDEBAR TEXT TO BE DISTINCT WHITE */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] h2 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #ffffff;
        padding-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Single line title with clear typography
st.markdown("<div class='main-title'>Smart Solar Energy Output Predictor</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.3rem !important; font-style: italic; color: #34495e; font-weight: bold;'>Empowering Sustainable Micro-Generation via Intelligent AI Analytics</p>", unsafe_allow_html=True)
st.write("**This platform pairs a trained machine learning backend with real-time solar tracking and thermal physics constraints to accurately model generation outcomes.**")

# 2. CACHE THE MODEL
@st.cache_resource
def load_solar_brain():
    return joblib.load('advanced_solar_brain.pkl')

ai_brain = load_solar_brain()

# 3. SIDEBAR USER INTERFACE (Text styled white via global CSS injection)
st.sidebar.markdown("## ⚙️ Hardware Profile")
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
        
        # Large Structured Subheading with Icon
        st.markdown("<div class='custom-subheading'>🌍 Live Environmental Vectors</div>", unsafe_allow_html=True)
        
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
        st.markdown("<br><hr>", unsafe_allow_html=True)
        
        # Large Structured Subheading with Icon
        st.markdown("<div class='custom-subheading'>⚡ Predicted Clean Energy Yield</div>", unsafe_allow_html=True)
        
        # Highly visible output box
        st.success(f"### **Estimated Output: {tailored_wattage:.2f} Watts**")
        
        # Real-time bolded eco-efficiency feedback cards
        if aoi < 45:
            st.info("🎯 **Optimal Angle Optimization:** Your structural layout has excellent alignment with current solar coordinates, maximizing renewable capture.")
        else:
            st.warning("⚠️ **Geometric Shading/Reflection Losses:** High mismatch angle detected. Consider adjusting hardware tilt dynamically to minimize reflection.")
            
    else:
        st.error(f"Weather API Error: {response.get('message')}")
except Exception as e:
    st.error(f"System Tracking Failure: {e}")