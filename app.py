import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from datetime import datetime
from fpdf import FPDF

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="EcoCast | Heatwave Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# 2. Custom CSS & UI Styling (Dark Orange Input Fields)
# -------------------------------
st.markdown("""
<style>
    /* Global Background - Warm Gradient */
    .stApp {
        background: linear-gradient(135deg, #FFEFBA 0%, #FFFFFF 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Titles */
    h1 {
        color: #d35400; /* Pumpkin Orange */
        text-align: center;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h3 {
        color: #e67e22;
        border-bottom: 3px solid #f39c12;
        padding-bottom: 10px;
        margin-top: 20px;
        font-weight: 700;
    }

    /* --- TARGETING INPUT FIELDS (ip fields) --- */
    
    /* The Input Container (The box itself) */
    div[data-baseweb="input"] {
        background-color: #ffcc80 !important; /* Dark Orange/Amber Background */
        border: 2px solid #ef6c00 !important; /* Darker Orange Border */
        border-radius: 8px !important;
    }

    /* The Text Inside the Input Fields */
    input[class*="st-"] {
        color: #3e2723 !important; /* Dark Brown text for contrast */
        font-weight: bold !important;
    }
    
    /* Input Label Color */
    .stNumberInput label, .stTextInput label {
        color: #d35400 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* ----------------------------------------- */

    /* Input Cards (Container for the inputs) */
    .css-1r6slb0, .stExpander {
        background-color: #fff8e1; /* Very Light Orange/Yellow */
        border: 1px solid #ffe0b2;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.15);
    }
    
    /* Button Styling - Orange Gradient */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff6f00 0%, #ff8f00 100%); /* Deep Dark Orange */
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 8px;
        transition-duration: 0.4s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff8f00 0%, #ff6f00 100%);
        transform: scale(1.02);
        box-shadow: 0 8px 12px rgba(255, 100, 0, 0.4);
        color: #fff;
    }

    /* Result Boxes */
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        color: white;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        border: 2px solid white;
    }
    .safe { background: linear-gradient(135deg, #56ab2f, #a8e063); }
    .danger { background: linear-gradient(135deg, #cb2d3e, #ef473a); animation: pulse 2s infinite; }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 71, 58, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(239, 71, 58, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 71, 58, 0); }
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 3. Load Models
# -------------------------------
try:
    model = joblib.load("models/best_regressor_LightGBM.joblib")
    imputer = joblib.load("models/imputer.joblib")
    scaler = joblib.load("models/scaler.joblib")
except FileNotFoundError:
    st.error("⚠️ System Error: Model files not found in 'models/' directory.")
    st.stop()

# -------------------------------
# 4. PDF Generator
# -------------------------------
def create_pdf(inputs_dict, prediction_val, heatwave_status):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_text_color(211, 84, 0) # Pumpkin Orange
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, txt="EcoCast Prediction Report", ln=True, align='C')
    pdf.set_draw_color(211, 84, 0)
    pdf.line(10, 30, 200, 30)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    
    pdf.ln(5)
    pdf.set_fill_color(255, 245, 230) 
    pdf.rect(10, 50, 190, 35, 'F')
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 55)
    pdf.cell(0, 10, txt=f"Predicted Max Temperature: {prediction_val:.2f} C", ln=True)
    
    pdf.set_xy(15, 65)
    if "POSSIBLE" in heatwave_status:
        pdf.set_text_color(192, 57, 43) # Red
        status_color = "CRITICAL WARNING"
    else:
        pdf.set_text_color(39, 174, 96) # Green
        status_color = "NORMAL"
        
    pdf.cell(0, 10, txt=f"Status: {heatwave_status} ({status_color})", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(25)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Input Parameters Used:", ln=True)
    pdf.set_font("Arial", size=11)
    
    for key, value in inputs_dict.items():
        clean_key = key.replace("_", " ").title()
        pdf.cell(95, 8, txt=f"{clean_key}: {value}", border=1)
        if list(inputs_dict.keys()).index(key) % 2 != 0:
            pdf.ln()
            
    return pdf.output(dest='S').encode('latin-1')

# -------------------------------
# 5. Initialization
# -------------------------------
defaults = {
    "latitude": 19.0, "longitude": 72.0, "wind_speed": 6.5,
    "cloud_cover": 20.0, "precipitation_probability": 10.0,
    "pressure_surface_level": 1005.0, "dew_point": 18.0,
    "uv_index": 7.0, "visibility": 5.0, "rainfall": 0.2,
    "min_temperature": 28.0, "max_humidity": 60.0, "min_humidity": 30.0,
    "max_temp_lag1": 35.0, "max_temp_lag2": 34.0
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------------------
# 6. Layout
# -------------------------------
st.markdown("<h1>☀️ EcoCast: Heatwave AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #e67e22; font-weight: bold;'>Advanced Temperature Risk Assessment System</p>", unsafe_allow_html=True)
st.markdown("---")

# API Section
st.markdown("### 📡 Live Data Sync")
with st.container():
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        city_name = st.text_input("📍 Search City", value="Mumbai")
    with col_btn:
        st.write("") 
        st.write("") 
        fetch_btn = st.button("🔥 Sync Data")

if fetch_btn:
    api_key = "YOUR_API_KEY"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"
    
    with st.spinner(f"Fetching data for {city_name}..."):
        try:
            response = requests.get(url)
            data = response.json()
            if data.get("cod") == 200:
                st.session_state['latitude'] = float(data["coord"]["lat"])
                st.session_state['longitude'] = float(data["coord"]["lon"])
                st.session_state['wind_speed'] = float(data["wind"]["speed"])
                st.session_state['cloud_cover'] = float(data["clouds"]["all"])
                if "grnd_level" in data["main"]:
                    st.session_state['pressure_surface_level'] = float(data["main"]["grnd_level"])
                else:
                    st.session_state['pressure_surface_level'] = float(data["main"]["pressure"])
                st.session_state['visibility'] = float(data["visibility"]) / 1000.0
                st.session_state['max_humidity'] = float(data["main"]["humidity"])
                st.success(f"✅ Data Synced: {data['name']}")
            else:
                st.error(f"❌ Error: {data.get('message', 'City not found')}")
        except Exception as e:
            st.error(f"❌ Connection Failed: {e}")

# Main Input Form
st.markdown("### 🎛️ Model Parameters")

with st.expander("📝 Manual Inputs (Expand to Edit)", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌍 Location & Air**")
        latitude = st.number_input("Latitude", key="latitude")
        longitude = st.number_input("Longitude", key="longitude")
        wind_speed = st.number_input("Wind Speed (m/s)", key="wind_speed")
        pressure_surface_level = st.number_input("Pressure (hPa)", key="pressure_surface_level")
        visibility = st.number_input("Visibility (km)", key="visibility")
    with col2:
        st.markdown("**💧 Moisture & UV**")
        cloud_cover = st.number_input("Cloud Cover (%)", key="cloud_cover")
        max_humidity = st.number_input("Max Humidity (%)", key="max_humidity")
        min_humidity = st.number_input("Min Humidity (%)", key="min_humidity")
        dew_point = st.number_input("Dew Point (°C)", key="dew_point")
        uv_index = st.number_input("UV Index", key="uv_index")
    with col3:
        st.markdown("**☔ Rain & History**")
        rainfall = st.number_input("Rainfall (mm)", key="rainfall")
        precipitation_probability = st.number_input("Precipitation Prob (%)", key="precipitation_probability")
        min_temperature = st.number_input("Today's Min Temp (°C)", key="min_temperature")
        max_temp_lag1 = st.number_input("Yesterday Max Temp (°C)", key="max_temp_lag1")
        max_temp_lag2 = st.number_input("2 Days Ago Max Temp (°C)", key="max_temp_lag2")

# Internal Calculations
max_temp_roll3 = (max_temp_lag1 + max_temp_lag2 + 33) / 3
month = datetime.now().month
dayofyear = datetime.now().timetuple().tm_yday
is_summer = 1 

input_data = {
    'latitude': latitude, 'longitude': longitude, 'wind_speed': wind_speed,
    'cloud_cover': cloud_cover, 'precipitation_probability': precipitation_probability,
    'pressure_surface_level': pressure_surface_level, 'dew_point': dew_point,
    'uv_index': uv_index, 'visibility': visibility, 'rainfall': rainfall,
    'min_temperature': min_temperature, 'max_humidity': max_humidity,
    'min_humidity': min_humidity, 'max_temp_lag1': max_temp_lag1,
    'max_temp_lag2': max_temp_lag2, 'max_temp_roll3': max_temp_roll3,
    'month': month, 'dayofyear': dayofyear, 'is_summer': is_summer
}
input_df = pd.DataFrame([input_data])

# -------------------------------
# 7. Prediction & Report
# -------------------------------
st.markdown("---")
st.markdown("### 🚀 Run Analysis")

if st.button("🌡️ Predict Next-Day Temperature"):
    with st.spinner("Analyzing atmospheric heat patterns..."):
        input_imputed = imputer.transform(input_df)
        prediction = model.predict(input_imputed)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction > 37:
        status_header = "🔥 HEATWAVE WARNING"
        status_text = "Heatwave is POSSIBLE. Extreme caution advised."
        css_class = "prediction-box danger"
        icon = "🥵"
    else:
        status_header = "🌿 CONDITIONS NORMAL"
        status_text = "No Heatwave expected. Standard weather."
        css_class = "prediction-box safe"
        icon = "😎"

    st.markdown(f"""
        <div class="{css_class}">
            <h2 style='margin:0; color: white;'>{icon} {prediction:.2f} °C</h2>
            <h4 style='margin:0; color: white; opacity: 0.95;'>Predicted Maximum Temperature</h4>
            <hr style='border-color: rgba(255,255,255,0.4); margin: 15px 0;'>
            <h3 style='margin:0; color: white; border: none; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>{status_header}</h3>
            <p style='margin:0; color: white; font-weight: 500;'>{status_text}</p>
        </div>
    """, unsafe_allow_html=True)

    pdf_bytes = create_pdf(input_data, prediction, status_text)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_dwn_1, col_dwn_2, col_dwn_3 = st.columns([1, 2, 1])
    with col_dwn_2:
        st.download_button(
            label="📄 Download Detailed PDF Report",
            data=pdf_bytes,
            file_name=f"EcoCast_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )