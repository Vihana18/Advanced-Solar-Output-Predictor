import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import pvlib
import datetime

# 1. PAGE SETUP & DESIGN CONFIGURATION
st.set_page_config(
    page_title="Smart Solar Energy Output Predictor", 
    layout="centered"
)

# INJECTING CSS FOR SPECIFIC COLOR THEMES AND TYPOGRAPHY MATCHING
st.markdown(
    """
    <style>
    /* Main body background color change */
    .stApp {
        background-color: #f4f7f5;
    }
    
    /* Global Page High Contrast Text Rules */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #111111 !important;
        font-weight: 500;
    }
    
    /* Clean Single Line Hero Header Title */
    .hero-title {
        color: #1a331e;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .hero-subtitle {
        text-align: center; 
        font-size: 1.25rem; 
        color: #556b2f; 
        font-weight: 600;
        margin-bottom: 35px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Header Banners for Content Boxes */
    .solvix-header {
        background-color: #1b4332;
        color: #ffffff !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        padding: 12px 20px;
        border-radius: 8px 8px 0px 0px;
        margin-top: 25px;
    }
    
    /* Targets Native Streamlit Containers to style them like custom white cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e1e8e3 !important;
        border-radius: 0px 0px 8px 8px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03);
    }
    
    /* Massive Metric Numbers */
    div[data-testid="stMetricValue"] {
        font-size: 2.6rem !important;
        color: #1b4332 !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetricLabel"] p {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #2d6a4f !important;
    }
    
    /* SIDEBAR CONTROL FOR DARK NAVIGATION LAYOUT */
    [data-testid="stSidebar"] {
        background-color: #11261a !important; 
    }
    
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
        border-bottom: 2px solid #52b788;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Title Presentation Block
st.markdown("<div class='hero-title'>Smart Solar Energy Output Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Powering A Brighter Future</div>", unsafe_allow_html=True)

# 2. BACKEND RESOURCE RETRIEVAL CACHE
@st.cache_resource
def load_solar_brain():
    return joblib.load('advanced_solar_brain.pkl')

ai_brain = load_solar_brain()

# 3. INTERACTIVE CONTROL NAVIGATION
st.sidebar.markdown("## ⚙️ Hardware Profile")
PANEL_MAX_CAPACITY = st.sidebar.number_input("Maximum Panel Capacity (Watts)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
PANEL_SURFACE_TILT = st.sidebar.slider("Panel Tilt Angle (0° = Flat, 90° = Vertical)", 0, 90, 25)
PANEL_SURFACE_AZIMUTH = st.sidebar.slider("Compass Facing Angle (180° = South, 90° = East)", 0, 360, 180)

# Global Constants
API_KEY = "81b04a4cf48bfc77710a24cb03e6915c"
LOCATION = "Dubai"
LATITUDE = 25.2048
LONGITUDE = 55.2708
TEMP_COEFFICIENT = -0.0035

# 4. WEATHER SENSOR READING EXTRACTION
url = f"https://api.openweathermap.org/data/2.5/weather?q={LOCATION}&appid={API_KEY}&units=metric"

try:
    response = requests.get(url).json()
    
    if response.get("cod") == 200:
        temp = response['main']['temp']
        clouds = response['clouds']['all']
        humidity = response['main']['humidity']
        pressure = response['main']['pressure']
        rain = response.get('rain', {}).get('1h', 0)
        
        # --- CARD SECTION 1: DYNAMIC ENVIRONMENTAL GRID ---
        st.markdown("<div class='solvix-header'>🌍 Live Environmental Vectors</div>", unsafe_allow_html=True)
        
        # Using a true native container component creates a solid white card block
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ambient Temp", f"{temp}°C")
            with col2:
                st.metric("Cloud Layer", f"{clouds}%")
            with col3:
                st.metric("Relative Humidity", f"{humidity}%")
        
        # 5. CELESTIAL ANGLE GEOMETRY LAYER
        current_time = pd.Timestamp(datetime.datetime.utcnow(), tz='UTC')
        sol_pos = pvlib.solarposition.get_solarposition(current_time, LATITUDE, LONGITUDE)
        live_zenith = sol_pos['apparent_zenith'].iloc[0]
        live_azimuth = sol_pos['azimuth'].iloc[0]
        
        aoi = pvlib.irradiance.aoi(PANEL_SURFACE_TILT, PANEL_SURFACE_AZIMUTH, live_zenith, live_azimuth)
        
        # 6. CALCULATE COMPOSITE PHYSICS ESTIMATES
        estimated_ghi = 950 * (1 - (0.0075 * clouds))
        clues = [[estimated_ghi, clouds, temp, live_zenith, live_azimuth, humidity, pressure, rain, PANEL_MAX_CAPACITY]]
        raw_prediction = ai_brain.predict(clues)[0]
        
        # Geometric loss modifier
        geometric_efficiency = np.cos(np.radians(aoi))
        if geometric_efficiency < 0:
            geometric_efficiency = 0.0
            
        # Solar panel heating efficiency loss modifier
        heat_loss_multiplier = 1.0
        if temp > 25:
            heat_loss_multiplier = 1.0 + (TEMP_COEFFICIENT * (temp - 25))
            
        # Combined generation answer
        tailored_wattage = raw_prediction * geometric_efficiency * heat_loss_multiplier
        
        if tailored_wattage > PANEL_MAX_CAPACITY: tailored_wattage = PANEL_MAX_CAPACITY
        if tailored_wattage < 0: tailored_wattage = 0.0

        # --- CARD SECTION 2: MACHINE LEARNING FORECASTING YIELD ---
        st.markdown("<div class='solvix-header'>⚡ Predicted Clean Energy Yield</div>", unsafe_allow_html=True)
        
        # This native container holds all values inside its clean white layout borders
        with st.container(border=True):
            # Performance status metric banner
            st.success(f"### **Estimated System Output: {tailored_wattage:.2f} Watts**")
            
            # Parametric values
            st.markdown(f"<p style='margin-top: 12px; margin-bottom: 12px; font-size: 1.15rem; font-weight: bold; color: #111111 !important;'>Calculated Rays Mismatch Angle (AOI): {aoi:.2f}°</p>", unsafe_allow_html=True)
            
            if aoi < 45:
                st.info("🎯 **Optimal Alignment Profile:** Structural deployment metrics coordinate directly with current solar trajectories, maximizing potential clean energy input.")
            else:
                st.error("⚠️ **Geometric Shading Vector Warning:** Mismatch threshold exceeds optimal limits. High reflective shading verified; reposition hardware orientation to recover lost generation.")
            
    else:
        st.error(f"Weather Gateway Error: {response.get('message')}")
except Exception as e:
    st.error(f"Dynamic Integration Error: {e}")
