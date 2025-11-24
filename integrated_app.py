"""
EcoCast - Integrated Disaster Detection System
==============================================

Combines three AI-powered disaster detection modules:
1. Heatwave Prediction (LightGBM ML Model)
2. Forest Fire Detection (CNN & MobileNetV2)
3. Landslide Detection (U-Net Segmentation)

SETUP INSTRUCTIONS:
-------------------
1. Install dependencies:
   pip install streamlit pandas numpy tensorflow joblib requests pillow fpdf python-dotenv sentinelhub

2. Configure API Keys in .env file:
   - Create a .env file in the project root directory
   - Add your OpenWeatherMap API key:
     OPENWEATHER_API_KEY=your_api_key_here
   - Get a free API key from: https://openweathermap.org/api

3. (Optional) Configure Sentinel Hub for satellite data:
   - Add to the same .env file:
     SENTINELHUB_CLIENT_ID=your_client_id
     SENTINELHUB_CLIENT_SECRET=your_client_secret
   - Create account at: https://www.sentinel-hub.com/

4. Run the application:
   streamlit run integrated_app.py

FEATURES:
---------
- Real-time weather data integration
- Multiple data sources (upload images or fetch satellite data)
- AI-powered risk assessment
- Fire spread likelihood scoring
- Emergency alert generation
- Comprehensive PDF and text report generation
- Historical alert tracking

MODELS REQUIRED:
----------------
- models/best_regressor_LightGBM.joblib
- models/imputer.joblib
- models/scaler.joblib
- fire_detection_cnnn_final.h5
- forestfire/mobilenet_fire_final.h5
- streamlit_app/best_model.keras (or model_save.h5)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import os
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import quote
from PIL import Image
from fpdf import FPDF
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox

# Keras backend for custom metrics
K = keras.backend

# =============================
# LOAD ENVIRONMENT VARIABLES
# =============================
load_dotenv()

# =============================
# API CONFIGURATION
# =============================
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

if not OPENWEATHER_API_KEY:
    st.error("⚠️ OpenWeatherMap API key not found! Please add OPENWEATHER_API_KEY to your .env file")

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(
    page_title="EcoCast | Disaster Detection System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# CUSTOM CSS STYLING
# =============================
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main Title */
    .main-title {
        color: #ffffff;
        text-align: center;
        font-weight: 900;
        font-size: 48px;
        letter-spacing: -1px;
        margin-bottom: 10px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: #e0e7ff;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Section Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Weather Card Specific */
    .weather-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    .metric-box {
        background: rgba(255,255,255,0.15);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #fbbf24;
    }
    
    .metric-label {
        font-size: 14px;
        color: #e0e7ff;
        margin-top: 5px;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #f59e0b 0%, #ef4444 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(239, 68, 68, 0.5);
    }

    /* Input Fields */
    div[data-baseweb="input"], .stTextInput input, .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        color: #1f2937 !important;
    }
    
    /* Labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Alert Boxes */
    .alert-danger {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    .alert-safe {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================
# UTILITY FUNCTIONS FOR LANDSLIDE
# =============================
def _safe_mid(values: np.ndarray, fallback: float = 0.5) -> float:
    """Return a stable mid-point using percentiles."""
    vmax = float(np.nanmax(values))
    if vmax <= 0:
        return fallback
    p75 = float(np.nanpercentile(values, 75))
    return max(min(p75, vmax), 1e-3)

def _invert_channel(chan: np.ndarray, mid: float) -> np.ndarray:
    inv = 1.0 - chan / (mid + 1e-6)
    return np.clip(inv, 0.0, 1.0)

# =============================
# SENTINEL HUB CONFIGURATION
# =============================
def _sentinel_config_from_env():
    """Load Sentinel Hub credentials from environment"""
    cfg = SHConfig()
    env_path = Path("streamlit_app/.env")
    if env_path.exists():
        load_dotenv(env_path)
    cfg.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    cfg.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not cfg.sh_client_id or not cfg.sh_client_secret:
        return None
    return cfg

def fetch_sentinel_rgbnir(lat: float, lon: float, size: int = 128, config: SHConfig = None):
    """Fetch Sentinel-2 satellite data"""
    cfg = config or _sentinel_config_from_env()
    if cfg is None:
        raise RuntimeError("Sentinel Hub credentials not configured")
    
    buffer_m = 128 * 15 / 2
    deg_lon = buffer_m / 111320
    deg_lat = buffer_m / 110540
    
    bbox = BBox([lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat], crs=CRS.WGS84)
    
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B02","B03","B04","B08"],
        output: { bands: 4, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      return [sample.B02, sample.B03, sample.B04, sample.B08];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(DataCollection.SENTINEL2_L2A)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(size, size),
        config=cfg
    )
    
    arr = request.get_data()[0]
    arr = np.clip(arr.astype(np.float32), 0, None) / 10000.0
    return arr

def fetch_dem_and_slope(lat: float, lon: float, size: int = 128, config: SHConfig = None):
    """Fetch DEM and calculate slope"""
    cfg = config or _sentinel_config_from_env()
    if cfg is None:
        raise RuntimeError("Sentinel Hub credentials not configured")
    
    buffer_m = 128 * 15 / 2
    deg_lon = buffer_m / 111320
    deg_lat = buffer_m / 110540
    
    bbox = BBox([lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat], crs=CRS.WGS84)
    
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["DEM"],
        output: { bands: 1, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      return [sample.DEM];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(DataCollection.DEM)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(size, size),
        config=cfg
    )
    
    data = request.get_data()[0]
    dem = data[:,:,0].astype(np.float32) if data.ndim == 3 else data.astype(np.float32)
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx * gx + gy * gy)
    return dem, slope

def fetch_feature_stack(lat: float, lon: float, size: int = 128):
    """Fetch and process satellite data for landslide detection"""
    cfg = _sentinel_config_from_env()
    if cfg is None:
        raise RuntimeError("Sentinel Hub credentials not configured")
    
    rgbnir = fetch_sentinel_rgbnir(lat, lon, size=size, config=cfg)
    dem, slope = fetch_dem_and_slope(lat, lon, size=size, config=cfg)
    
    feature_data = np.zeros((size, size, 14), dtype=np.float32)
    feature_data[:, :, 1] = rgbnir[:, :, 0]  # Blue
    feature_data[:, :, 2] = rgbnir[:, :, 1]  # Green
    feature_data[:, :, 3] = rgbnir[:, :, 2]  # Red
    feature_data[:, :, 7] = rgbnir[:, :, 3]  # NIR
    feature_data[:, :, 12] = slope
    feature_data[:, :, 13] = dem
    
    feature_stack = build_feature_stack(feature_data)
    return feature_stack, rgbnir

# =============================
# LANDSLIDE HELPER FUNCTIONS
# =============================
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def unet_model():
    """U-Net architecture for landslide detection"""
    inputs = tf.keras.layers.Input((128, 128, 6))
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_feature_stack(image_cube: np.ndarray) -> np.ndarray:
    """Feature engineering for landslide detection"""
    data = np.nan_to_num(image_cube, nan=1e-6)
    
    data_blue = data[:, :, 1]
    data_green = data[:, :, 2]
    data_red = data[:, :, 3]
    data_nir = data[:, :, 7]
    
    mid_rgb = _safe_mid(np.stack([data_red, data_green, data_blue], axis=2))
    mid_slope = _safe_mid(data[:, :, 12])
    mid_elevation = _safe_mid(data[:, :, 13])
    
    ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red) + 1e-6)

    feature_stack = np.zeros((128, 128, 6), dtype=np.float32)
    feature_stack[:, :, 0] = _invert_channel(data_red, mid_rgb)
    feature_stack[:, :, 1] = _invert_channel(data_green, mid_rgb)
    feature_stack[:, :, 2] = _invert_channel(data_blue, mid_rgb)
    feature_stack[:, :, 3] = ndvi
    feature_stack[:, :, 4] = _invert_channel(data[:, :, 12], mid_slope)
    feature_stack[:, :, 5] = _invert_channel(data[:, :, 13], mid_elevation)
    feature_stack[:, :, 3] = np.nan_to_num(feature_stack[:, :, 3], nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack

def process_image_landslide(uploaded_file) -> Tuple[np.ndarray, np.ndarray]:
    """Process uploaded image for landslide detection"""
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0

    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]

    nir = (green * 1.2 + red * 0.2).astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    mid_rgb = _safe_mid(np.stack([red, green, blue], axis=2))

    features = np.zeros((128, 128, 6), dtype=np.float32)
    features[:, :, 0] = _invert_channel(red, mid_rgb)
    features[:, :, 1] = _invert_channel(green, mid_rgb)
    features[:, :, 2] = _invert_channel(blue, mid_rgb)
    features[:, :, 3] = ndvi

    gy, gx = np.gradient(np.mean(img_array, axis=2))
    slope_fake = np.sqrt(gx*gx + gy*gy)
    slope_mid = _safe_mid(slope_fake)
    features[:, :, 4] = _invert_channel(slope_fake, slope_mid)

    brightness = np.mean(img_array, axis=2)
    elev_mid = _safe_mid(brightness)
    features[:, :, 5] = _invert_channel(brightness, elev_mid)

    return features, img_array

# =============================
# FOREST FIRE HELPER FUNCTIONS
# =============================
def to_rgb(img_pil):
    """Ensure PIL image is RGB"""
    if img_pil.mode == "RGBA" or img_pil.mode == "LA":
        return Image.alpha_composite(Image.new("RGBA", img_pil.size, (255,255,255)), img_pil).convert("RGB")
    if img_pil.mode != "RGB":
        return img_pil.convert("RGB")
    return img_pil

def preprocess_for_cnn(pil_img, target=128):
    """Preprocess for custom CNN"""
    img = to_rgb(pil_img)
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_mnv2(pil_img, target=224):
    """Preprocess for MobileNetV2"""
    img = to_rgb(pil_img)
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_fire(model, prepped):
    """Run fire prediction"""
    preds = model.predict(prepped, verbose=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0,0])
    elif preds.ndim == 1:
        prob = float(preds[0])
    else:
        prob = float(np.ravel(preds)[0])
    return np.clip(prob, 0.0, 1.0)

def calculate_fire_spread_score(wind_speed, humidity, temperature):
    """Calculate fire spread likelihood score"""
    # Normalize factors (0-100 scale)
    wind_factor = min(wind_speed / 30.0, 1.0) * 40  # Max 40 points
    humidity_factor = (1 - min(humidity / 100.0, 1.0)) * 35  # Max 35 points
    temp_factor = min(max(temperature - 20, 0) / 30.0, 1.0) * 25  # Max 25 points
    
    total_score = wind_factor + humidity_factor + temp_factor
    
    if total_score >= 70:
        risk_level = "CRITICAL"
        color = "danger"
    elif total_score >= 50:
        risk_level = "HIGH"
        color = "warning"
    elif total_score >= 30:
        risk_level = "MODERATE"
        color = "warning"
    else:
        risk_level = "LOW"
        color = "safe"
    
    return total_score, risk_level, color

# =============================
# PDF REPORT GENERATORS
# =============================
def create_heatwave_pdf(inputs_dict, prediction_val, heatwave_status):
    """Generate PDF report for heatwave prediction"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_text_color(211, 84, 0)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, txt="EcoCast Heatwave Prediction Report", ln=True, align='C')
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
        pdf.set_text_color(192, 57, 43)
        status_color = "CRITICAL WARNING"
    else:
        pdf.set_text_color(39, 174, 96)
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

def create_fire_pdf(report_text):
    """Generate PDF report for fire detection"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 8, "Wildfire Detection Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", '', 11)
    for line in report_text.split("\n"):
        pdf.multi_cell(0, 6, line)
    return pdf.output(dest="S").encode("latin-1")

def create_landslide_pdf(record: Dict) -> bytes:
    """Generate PDF report for landslide detection"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, txt="Landslide Detection Report", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, txt=f"Generated on: {record['timestamp']}", ln=True, align='R')
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt=f"Location: {record['location']}", ln=True)
    pdf.cell(0, 10, txt=f"Coordinates: {record['latitude']}, {record['longitude']}", ln=True)
    pdf.cell(0, 10, txt=f"Alert Status: {record['label']}", ln=True)
    pdf.cell(0, 10, txt=f"Coverage: {record['coverage']*100:.2f}%", ln=True)
    pdf.cell(0, 10, txt=f"Severity: {record['severity']}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, f"Notes: {record['operator_notes']}")
    
    return pdf.output(dest='S').encode('latin-1')

# =============================
# WEATHER DATA FETCHING
# =============================
def fetch_weather_data(city_name, api_key):
    """Fetch weather data from OpenWeatherMap API"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.json().get('message', 'Unknown error')}"
    except Exception as e:
        return None, str(e)

# =============================
# MODEL LOADING
# =============================
@st.cache_resource
def load_heatwave_models():
    """Load heatwave prediction models"""
    try:
        model = joblib.load("models/best_regressor_LightGBM.joblib")
        imputer = joblib.load("models/imputer.joblib")
        scaler = joblib.load("models/scaler.joblib")
        return model, imputer, scaler, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_resource
def load_fire_models():
    """Load fire detection models"""
    try:
        cnn = tf.keras.models.load_model("fire_detection_cnnn_final.h5")
        mnv2 = tf.keras.models.load_model("forestfire/mobilenet_fire_final.h5")
        return cnn, mnv2, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_landslide_model():
    """Load landslide detection model"""
    try:
        model_path = Path("streamlit_app/best_model.keras")
        if not model_path.exists():
            model_path = Path("streamlit_app/model_save.h5")
        
        if model_path.exists():
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects={"precision_m": precision_m, "recall_m": recall_m, "f1_m": f1_m},
            )
            return model, None
        else:
            # Build architecture and try loading weights
            model = unet_model()
            model.load_weights(str(model_path))
            return model, None
    except Exception as e:
        return None, str(e)

# =============================
# INITIALIZE SESSION STATE
# =============================
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "fire_last_prediction" not in st.session_state:
    st.session_state.fire_last_prediction = None
if "landslide_alert_log" not in st.session_state:
    st.session_state.landslide_alert_log = []

# =============================
# MAIN APP HEADER
# =============================
st.markdown('<h1 class="main-title">🌍 EcoCast: Integrated Disaster Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered monitoring for Heatwaves, Forest Fires, and Landslides</p>', unsafe_allow_html=True)

# =============================
# CONFIGURATION - Set your API key here
# =============================
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select Module", ["🏠 Home", "☀️ Heatwave Prediction", "🔥 Forest Fire Detection", "⛰️ Landslide Detection"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌤️ Live Weather Data")
city_name = st.sidebar.text_input("City Name", value="Mumbai", help="Enter city name to fetch live weather data")

if st.sidebar.button("🔄 Fetch Weather Data", use_container_width=True):
    if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "YOUR_API_KEY_HERE":
        with st.spinner(f"Fetching weather data for {city_name}..."):
            weather_data, error = fetch_weather_data(city_name, OPENWEATHER_API_KEY)
            if weather_data:
                st.session_state.weather_data = weather_data
                st.sidebar.success(f"✅ Data loaded for {weather_data.get('name', city_name)}")
            else:
                st.sidebar.error(f"❌ {error}")
    else:
        st.sidebar.warning("⚠️ API key not configured. Please set OPENWEATHER_API_KEY in the code.")

# Display current weather if available
if st.session_state.weather_data:
    wd = st.session_state.weather_data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Conditions")
    st.sidebar.metric("🌡️ Temperature", f"{wd['main']['temp']:.1f}°C")
    st.sidebar.metric("💧 Humidity", f"{wd['main']['humidity']}%")
    st.sidebar.metric("🌬️ Wind Speed", f"{wd['wind']['speed']} m/s")
    if 'wind' in wd and 'deg' in wd['wind']:
        st.sidebar.metric("🧭 Wind Direction", f"{wd['wind']['deg']}°")

# =============================
# HOME PAGE
# =============================
if page == "🏠 Home":
    st.markdown("---")
    
    # Hero Section
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); text-align: center; padding: 40px;">
        <h2 style="color: #fbbf24; margin-bottom: 20px;">🌍 Welcome to EcoCast</h2>
        <p style="font-size: 18px; color: #e0e7ff; line-height: 1.8;">
            Advanced AI-powered disaster detection and monitoring system integrating multiple environmental 
            hazard detection capabilities with real-time weather data analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎯 Detection Modules")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card" style="min-height: 280px; transition: transform 0.3s;">
            <h2 style="text-align: center; font-size: 48px; margin: 20px 0;">☀️</h2>
            <h3 style="text-align: center; color: #fbbf24;">Heatwave Prediction</h3>
            <p style="text-align: center; color: #e0e7ff; line-height: 1.6; padding: 10px;">
                Predict maximum temperature and assess heatwave risk using LightGBM regression models 
                with weather data integration.
            </p>
            <div style="text-align: center; margin-top: 20px;">
                <span style="background: rgba(251, 191, 36, 0.2); padding: 8px 16px; border-radius: 20px; font-size: 12px; color: #fbbf24;">
                    ML-Powered
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="min-height: 280px; transition: transform 0.3s;">
            <h2 style="text-align: center; font-size: 48px; margin: 20px 0;">🔥</h2>
            <h3 style="text-align: center; color: #f59e0b;">Forest Fire Detection</h3>
            <p style="text-align: center; color: #e0e7ff; line-height: 1.6; padding: 10px;">
                Detect active fires from satellite and aerial imagery using CNN and MobileNetV2 models 
                with fire spread risk analysis.
            </p>
            <div style="text-align: center; margin-top: 20px;">
                <span style="background: rgba(245, 158, 11, 0.2); padding: 8px 16px; border-radius: 20px; font-size: 12px; color: #f59e0b;">
                    Deep Learning
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card" style="min-height: 280px; transition: transform 0.3s;">
            <h2 style="text-align: center; font-size: 48px; margin: 20px 0;">⛰️</h2>
            <h3 style="text-align: center; color: #10b981;">Landslide Detection</h3>
            <p style="text-align: center; color: #e0e7ff; line-height: 1.6; padding: 10px;">
                Identify landslide-prone areas using U-Net segmentation with satellite imagery 
                and terrain analysis from Sentinel Hub.
            </p>
            <div style="text-align: center; margin-top: 20px;">
                <span style="background: rgba(16, 185, 129, 0.2); padding: 8px 16px; border-radius: 20px; font-size: 12px; color: #10b981;">
                    Satellite Data
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## ⚡ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #fbbf24; margin-bottom: 15px;">🎯 Core Capabilities</h4>
            <ul style="color: #e0e7ff; line-height: 2;">
                <li><strong>Real-time Weather Integration</strong> - Live data from OpenWeatherMap API</li>
                <li><strong>AI-Powered Predictions</strong> - Advanced ML and Deep Learning models</li>
                <li><strong>Multi-Source Data</strong> - Upload images or fetch satellite data</li>
                <li><strong>Emergency Alerts</strong> - Automated risk assessment and warnings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #fbbf24; margin-bottom: 15px;">📊 Analysis & Reporting</h4>
            <ul style="color: #e0e7ff; line-height: 2;">
                <li><strong>Comprehensive Reports</strong> - PDF and text format downloads</li>
                <li><strong>Geographic Mapping</strong> - Coordinate-based location tracking</li>
                <li><strong>Risk Scoring</strong> - Quantitative hazard assessment</li>
                <li><strong>Historical Tracking</strong> - Alert history and trend analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-value">3</div>
            <div class="metric-label">Detection Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">24/7</div>
            <div class="metric-label">Monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">100%</div>
            <div class="metric-label">AI-Powered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        alert_count = len(st.session_state.landslide_alert_log)
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-value">{alert_count}</div>
            <div class="metric-label">Total Alerts</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(245, 158, 11, 0.1));">
        <h3 style="color: #fbbf24; text-align: center; margin-bottom: 20px;">🚀 Getting Started</h3>
        <ol style="color: #e0e7ff; line-height: 2; font-size: 16px;">
            <li><strong>Configure API Key:</strong> Set your OpenWeatherMap API key in the code (OPENWEATHER_API_KEY)</li>
            <li><strong>Select Module:</strong> Choose a disaster detection module from the sidebar</li>
            <li><strong>Fetch Weather Data:</strong> Enter a city name and click "Fetch Weather Data"</li>
            <li><strong>Run Analysis:</strong> Upload an image or use live data to perform detection</li>
            <li><strong>Download Reports:</strong> Generate and download comprehensive PDF/text reports</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.info("👈 **Select a module from the sidebar to begin disaster monitoring and detection**")

# =============================
# HEATWAVE PREDICTION PAGE
# =============================
elif page == "☀️ Heatwave Prediction":
    st.markdown("## ☀️ Heatwave Prediction & Risk Assessment")
    
    # Load models
    model, imputer, scaler, error = load_heatwave_models()
    if error:
        st.error(f"⚠️ Failed to load heatwave models: {error}")
        st.stop()
    
    # Weather data integration
    if st.session_state.weather_data:
        wd = st.session_state.weather_data
        st.markdown(f"""
        <div class="weather-card">
            <h3>🌤️ Current Weather in {wd.get('name', 'Unknown')}</h3>
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <div class="metric-box">
                    <div class="metric-value">{wd['main']['temp']:.1f}°C</div>
                    <div class="metric-label">Temperature</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{wd['main']['humidity']}%</div>
                    <div class="metric-label">Humidity</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{wd['wind']['speed']} m/s</div>
                    <div class="metric-label">Wind Speed</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{wd['clouds']['all']}%</div>
                    <div class="metric-label">Cloud Cover</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize defaults
    defaults = {
        "latitude": wd['coord']['lat'] if st.session_state.weather_data else 19.0,
        "longitude": wd['coord']['lon'] if st.session_state.weather_data else 72.0,
        "wind_speed": wd['wind']['speed'] if st.session_state.weather_data else 6.5,
        "cloud_cover": wd['clouds']['all'] if st.session_state.weather_data else 20.0,
        "precipitation_probability": 10.0,
        "pressure_surface_level": wd['main'].get('grnd_level', wd['main']['pressure']) if st.session_state.weather_data else 1005.0,
        "dew_point": 18.0,
        "uv_index": 7.0,
        "visibility": wd['visibility'] / 1000.0 if st.session_state.weather_data else 5.0,
        "rainfall": 0.2,
        "min_temperature": wd['main']['temp_min'] if st.session_state.weather_data else 28.0,
        "max_humidity": wd['main']['humidity'] if st.session_state.weather_data else 60.0,
        "min_humidity": 30.0,
        "max_temp_lag1": 35.0,
        "max_temp_lag2": 34.0
    }
    
    # Input form
    st.markdown("### 🎛️ Model Parameters")
    with st.expander("📝 Adjust Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌍 Location & Air**")
            latitude = st.number_input("Latitude", value=float(defaults["latitude"]))
            longitude = st.number_input("Longitude", value=float(defaults["longitude"]))
            wind_speed = st.number_input("Wind Speed (m/s)", value=float(defaults["wind_speed"]))
            pressure = st.number_input("Pressure (hPa)", value=float(defaults["pressure_surface_level"]))
            visibility = st.number_input("Visibility (km)", value=float(defaults["visibility"]))
        
        with col2:
            st.markdown("**💧 Moisture & UV**")
            cloud_cover = st.number_input("Cloud Cover (%)", value=float(defaults["cloud_cover"]))
            max_humidity = st.number_input("Max Humidity (%)", value=float(defaults["max_humidity"]))
            min_humidity = st.number_input("Min Humidity (%)", value=float(defaults["min_humidity"]))
            dew_point = st.number_input("Dew Point (°C)", value=float(defaults["dew_point"]))
            uv_index = st.number_input("UV Index", value=float(defaults["uv_index"]))
        
        with col3:
            st.markdown("**☔ Rain & History**")
            rainfall = st.number_input("Rainfall (mm)", value=float(defaults["rainfall"]))
            precip_prob = st.number_input("Precipitation Prob (%)", value=float(defaults["precipitation_probability"]))
            min_temp = st.number_input("Today's Min Temp (°C)", value=float(defaults["min_temperature"]))
            max_temp_lag1 = st.number_input("Yesterday Max Temp (°C)", value=float(defaults["max_temp_lag1"]))
            max_temp_lag2 = st.number_input("2 Days Ago Max Temp (°C)", value=float(defaults["max_temp_lag2"]))
    
    # Calculations
    max_temp_roll3 = (max_temp_lag1 + max_temp_lag2 + 33) / 3
    month = datetime.now().month
    dayofyear = datetime.now().timetuple().tm_yday
    is_summer = 1
    
    input_data = {
        'latitude': latitude, 'longitude': longitude, 'wind_speed': wind_speed,
        'cloud_cover': cloud_cover, 'precipitation_probability': precip_prob,
        'pressure_surface_level': pressure, 'dew_point': dew_point,
        'uv_index': uv_index, 'visibility': visibility, 'rainfall': rainfall,
        'min_temperature': min_temp, 'max_humidity': max_humidity,
        'min_humidity': min_humidity, 'max_temp_lag1': max_temp_lag1,
        'max_temp_lag2': max_temp_lag2, 'max_temp_roll3': max_temp_roll3,
        'month': month, 'dayofyear': dayofyear, 'is_summer': is_summer
    }
    
    input_df = pd.DataFrame([input_data])
    
    st.markdown("---")
    
    if st.button("🌡️ Predict Next-Day Temperature", use_container_width=True):
        with st.spinner("Analyzing atmospheric heat patterns..."):
            input_imputed = imputer.transform(input_df)
            prediction = model.predict(input_imputed)[0]
        
        if prediction > 37:
            st.markdown(f"""
            <div class="alert-danger">
                <h2>🥵 {prediction:.2f} °C</h2>
                <h3>🔥 HEATWAVE WARNING</h3>
                <p>Heatwave is POSSIBLE. Extreme caution advised.</p>
            </div>
            """, unsafe_allow_html=True)
            status_text = "Heatwave is POSSIBLE"
        else:
            st.markdown(f"""
            <div class="alert-safe">
                <h2>😎 {prediction:.2f} °C</h2>
                <h3>🌿 CONDITIONS NORMAL</h3>
                <p>No Heatwave expected. Standard weather.</p>
            </div>
            """, unsafe_allow_html=True)
            status_text = "No Heatwave expected"
        
        # Generate PDF
        pdf_bytes = create_heatwave_pdf(input_data, prediction, status_text)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"Heatwave_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# =============================
# FOREST FIRE DETECTION PAGE
# =============================
elif page == "🔥 Forest Fire Detection":
    st.markdown("## 🔥 Forest Fire Detection & Spread Analysis")
    
    # Load models
    model_cnn, model_mnv2, error = load_fire_models()
    if error:
        st.error(f"⚠️ Failed to load fire detection models: {error}")
        st.stop()
    
    # Weather-based fire spread analysis
    if st.session_state.weather_data:
        wd = st.session_state.weather_data
        temp = wd['main']['temp']
        humidity = wd['main']['humidity']
        wind = wd['wind']['speed']
        
        score, risk_level, color = calculate_fire_spread_score(wind, humidity, temp)
        
        alert_class = f"alert-{color}"
        st.markdown(f"""
        <div class="{alert_class}">
            <h3>🔥 Fire Spread Likelihood Score: {score:.1f}/100</h3>
            <h2>Risk Level: {risk_level}</h2>
            <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                <div>
                    <strong>Wind:</strong> {wind} m/s<br>
                    <small>Higher wind = faster spread</small>
                </div>
                <div>
                    <strong>Humidity:</strong> {humidity}%<br>
                    <small>Lower humidity = easier ignition</small>
                </div>
                <div>
                    <strong>Temperature:</strong> {temp:.1f}°C<br>
                    <small>Higher temp = more risk</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🖼️ Upload Image for Detection")
    with col2:
        model_choice = st.selectbox("Model", ["MobileNetV2 (Satellite)", "Custom CNN (Ground/Aerial)"])
    
    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Detect Fire", use_container_width=True):
            with st.spinner("Analyzing image..."):
                if "MobileNetV2" in model_choice:
                    prepped = preprocess_for_mnv2(img_pil, target=224)
                    selected_model = model_mnv2
                else:
                    prepped = preprocess_for_cnn(img_pil, target=128)
                    selected_model = model_cnn
                
                prob = predict_fire(selected_model, prepped)
                is_fire = prob >= 0.5
                
                # Store prediction
                st.session_state.fire_last_prediction = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model': model_choice,
                    'filename': uploaded.name,
                    'probability': float(prob),
                    'is_fire': bool(is_fire),
                    'weather': st.session_state.weather_data
                }
                
                if is_fire:
                    st.markdown(f"""
                    <div class="alert-danger">
                        <h2>🔥 FIRE DETECTED</h2>
                        <h3>Confidence: {prob:.2%}</h3>
                        <p>Active fire or smoke detected in the image</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-safe">
                        <h2>✅ NO FIRE DETECTED</h2>
                        <h3>Confidence: {(1-prob):.2%}</h3>
                        <p>No fire signatures detected</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Reports section
    if st.session_state.fire_last_prediction:
        st.markdown("---")
        st.markdown("### 📊 Detection Report & Analysis")
        
        lp = st.session_state.fire_last_prediction
        
        # Display results in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <h4 style="color: #fbbf24;">📅 Timestamp</h4>
                <p style="font-size: 14px; color: #e0e7ff;">{}</p>
            </div>
            """.format(lp['timestamp']), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <h4 style="color: #fbbf24;">🤖 Model</h4>
                <p style="font-size: 14px; color: #e0e7ff;">{}</p>
            </div>
            """.format(lp['model']), unsafe_allow_html=True)
        with col3:
            result_color = "#ef4444" if lp['is_fire'] else "#10b981"
            result_icon = "🔥" if lp['is_fire'] else "✅"
            st.markdown("""
            <div class="card" style="text-align: center;">
                <h4 style="color: {};">{} Result</h4>
                <p style="font-size: 18px; font-weight: bold; color: {};">{}</p>
            </div>
            """.format(result_color, result_icon, result_color, 
                      "Fire Detected" if lp['is_fire'] else "No Fire"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed information
        with st.expander("📋 Detailed Analysis Report", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📄 Detection Details**")
                st.write(f"• File: `{lp['filename']}`")
                st.write(f"• Fire Probability: **{lp['probability']:.2%}**")
                st.write(f"• Confidence: **{(lp['probability'] if lp['is_fire'] else 1-lp['probability']):.2%}**")
            
            with col2:
                if lp['weather']:
                    wd = lp['weather']
                    st.markdown("**🌤️ Weather Conditions**")
                    st.write(f"• Location: {wd.get('name', 'Unknown')}")
                    st.write(f"• Temperature: {wd['main']['temp']:.1f}°C")
                    st.write(f"• Humidity: {wd['main']['humidity']}%")
                    st.write(f"• Wind Speed: {wd['wind']['speed']} m/s")
        
        # Generate comprehensive report
        report_lines = [
            "=" * 60,
            "WILDFIRE DETECTION REPORT",
            "=" * 60,
            "",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DETECTION SUMMARY",
            "-" * 60,
            f"Analysis Timestamp: {lp['timestamp']}",
            f"Image File: {lp['filename']}",
            f"AI Model: {lp['model']}",
            f"Detection Result: {'FIRE DETECTED' if lp['is_fire'] else 'NO FIRE DETECTED'}",
            f"Fire Probability: {lp['probability']:.4f} ({lp['probability']*100:.2f}%)",
            f"Confidence Level: {(lp['probability'] if lp['is_fire'] else 1-lp['probability'])*100:.2f}%",
            "",
        ]
        
        if lp['weather']:
            wd = lp['weather']
            score, risk_level, _ = calculate_fire_spread_score(
                wd['wind']['speed'], 
                wd['main']['humidity'], 
                wd['main']['temp']
            )
            
            report_lines.extend([
                "WEATHER CONDITIONS",
                "-" * 60,
                f"Location: {wd.get('name', 'Unknown')}",
                f"Coordinates: {wd['coord']['lat']}, {wd['coord']['lon']}",
                f"Temperature: {wd['main']['temp']:.1f}°C",
                f"Feels Like: {wd['main']['feels_like']:.1f}°C",
                f"Humidity: {wd['main']['humidity']}%",
                f"Wind Speed: {wd['wind']['speed']} m/s",
                f"Wind Direction: {wd['wind'].get('deg', 'N/A')}°",
                f"Cloud Cover: {wd['clouds']['all']}%",
                f"Visibility: {wd.get('visibility', 'N/A')} meters",
                "",
                "FIRE SPREAD RISK ANALYSIS",
                "-" * 60,
                f"Fire Spread Likelihood Score: {score:.1f}/100",
                f"Risk Level: {risk_level}",
                "",
                "Risk Factors:",
                f"  • Wind Factor: {'HIGH' if wd['wind']['speed'] > 15 else 'MODERATE' if wd['wind']['speed'] > 8 else 'LOW'}",
                f"  • Humidity Factor: {'HIGH RISK' if wd['main']['humidity'] < 30 else 'MODERATE' if wd['main']['humidity'] < 50 else 'LOW RISK'}",
                f"  • Temperature Factor: {'HIGH' if wd['main']['temp'] > 35 else 'MODERATE' if wd['main']['temp'] > 25 else 'LOW'}",
                "",
                "INTERPRETATION:",
                "  Wind Speed: Higher winds increase fire spread rate",
                "  Humidity: Lower humidity makes vegetation more flammable",
                "  Temperature: Higher temperatures dry out fuel sources",
                "",
            ])
        
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 60,
        ])
        
        if lp['is_fire']:
            report_lines.extend([
                "⚠️ FIRE DETECTED - IMMEDIATE ACTION REQUIRED:",
                "  1. Alert local fire authorities immediately",
                "  2. Evacuate personnel from affected areas",
                "  3. Activate emergency response protocols",
                "  4. Monitor fire spread direction using wind data",
                "  5. Establish firebreaks if safe to do so",
            ])
            if lp['weather'] and score > 50:
                report_lines.append("  6. HIGH SPREAD RISK - Expect rapid fire progression")
        else:
            report_lines.extend([
                "✅ NO FIRE DETECTED - ROUTINE MONITORING:",
                "  1. Continue regular surveillance",
                "  2. Monitor weather conditions",
                "  3. Maintain fire prevention measures",
            ])
        
        report_lines.extend([
            "",
            "=" * 60,
            "End of Report",
            "=" * 60,
        ])
        
        report_text = "\n".join(report_lines)
        
        # Display text report in expandable section
        with st.expander("📄 View Full Text Report"):
            st.text(report_text)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = create_fire_pdf(report_text)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Fire_Detection_Report_{lp['timestamp'].replace(' ', '_').replace(':', '-')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="📝 Download Text Report",
                data=report_text.encode('utf-8'),
                file_name=f"Fire_Detection_Report_{lp['timestamp'].replace(' ', '_').replace(':', '-')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# =============================
# LANDSLIDE DETECTION PAGE
# =============================
elif page == "⛰️ Landslide Detection":
    st.markdown("## ⛰️ Landslide Detection & Risk Assessment")
    
    # Load model
    landslide_model, error = load_landslide_model()
    if error:
        st.error(f"⚠️ Failed to load landslide model: {error}")
        st.stop()
    
    # Detection settings
    prob_threshold = 0.3
    coverage_threshold = 0.005
    
    # Data source selection
    st.markdown("### 📡 Choose Data Source")
    data_source = st.radio(
        "Select input method:",
        ["📤 Upload Image", "🛰️ Real-time Satellite Data"],
        horizontal=True,
        help="Upload your own image or fetch live satellite data"
    )
    
    st.markdown("---")
    
    # Initialize variables
    uploaded = None
    latitude = 28.3949
    longitude = 84.1240
    location_name = "Unknown Location"
    
    if data_source == "📤 Upload Image":
        st.markdown("### 🖼️ Upload Image for Analysis")
        st.info("📸 Upload a satellite or aerial image of the area to check for landslide risk.")
        
        uploaded = st.file_uploader(
            "Upload image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Supports satellite imagery and aerial photos"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=28.3949, format="%.6f")
            location_name = st.text_input("Location Name", value="Nagaland, India")
        with col2:
            longitude = st.number_input("Longitude", value=84.1240, format="%.6f")
        
        analyze_button = st.button("🔍 Analyze for Landslide Risk", use_container_width=True, disabled=uploaded is None)
        
    else:  # Real-time Satellite Data
        st.markdown("### 🛰️ Real-time Satellite Data")
        st.info("🌍 Enter coordinates to fetch live Sentinel-2 satellite imagery and DEM data")
        
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=30.557367, format="%.6f", help="Example: Joshimath landslide area")
            location_name = st.text_input("Location Name", value="Joshimath, India")
        with col2:
            longitude = st.number_input("Longitude", value=79.565394, format="%.6f")
        
        if st.button("📥 Fetch Satellite Data", use_container_width=True):
            try:
                with st.spinner(f"🛰️ Downloading satellite imagery for {location_name}..."):
                    features, rgbnir_raw = fetch_feature_stack(latitude, longitude, size=128)
                    st.session_state["fetched_features"] = features
                    st.session_state["fetched_rgbnir"] = rgbnir_raw
                    st.session_state["fetched_coords"] = (latitude, longitude, location_name)
                st.success("✅ Satellite data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Failed to fetch satellite data: {str(e)}")
                st.info("💡 Make sure Sentinel Hub credentials are configured in streamlit_app/.env")
        
        analyze_button = st.button(
            "🔍 Analyze for Landslide Risk", 
            use_container_width=True,
            disabled="fetched_features" not in st.session_state
        )
    
    st.markdown("---")
    
    if analyze_button:
        # Determine data source and process accordingly
        if data_source == "📤 Upload Image":
            if uploaded is None:
                st.error("❌ Please upload an image first.")
                st.stop()
            
            with st.spinner("🔬 Analyzing uploaded image..."):
                features, img_display = process_image_landslide(uploaded)
                source_note = "Auto-detected from uploaded image"
        else:
            if "fetched_features" not in st.session_state:
                st.error("❌ Please fetch satellite data first.")
                st.stop()
            
            with st.spinner("🔬 Analyzing satellite data..."):
                features = st.session_state["fetched_features"]
                rgbnir_raw = st.session_state["fetched_rgbnir"]
                latitude, longitude, location_name = st.session_state["fetched_coords"]
                
                # Build true-color RGB from raw bands
                rgb_display = np.stack([
                    rgbnir_raw[:, :, 2],  # Red
                    rgbnir_raw[:, :, 1],  # Green
                    rgbnir_raw[:, :, 0],  # Blue
                ], axis=2)
                img_display = np.clip(rgb_display, 0, 1)
                source_note = "Auto-detected from satellite imagery"
        
        # Run prediction
        input_batch = np.expand_dims(features, axis=0)
        preds = landslide_model.predict(input_batch, verbose=0)
        prob_map = preds[0, :, :, 0]
        
        binary_mask = (prob_map >= prob_threshold).astype(np.uint8)
        coverage = float(binary_mask.sum() / binary_mask.size)
        
        label = "LANDSLIDE ALERT" if coverage >= coverage_threshold else "Stable"
        severity = "Critical" if coverage > 0.05 else "High" if coverage > 0.01 else "Moderate"
        
        # Create alert record
        record = {
            "label": label,
            "coverage": coverage,
            "latitude": latitude,
            "longitude": longitude,
            "location": location_name,
            "incident_type": "Landslide",
            "severity": severity,
            "operator_notes": source_note,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_source": data_source
        }
        
        st.session_state.landslide_alert_log.insert(0, record)
        
        # Display results
        if label == "LANDSLIDE ALERT":
            st.markdown(f"""
            <div class="alert-danger">
                <h2>🚨 LANDSLIDE ALERT</h2>
                <h3>Risk Coverage: {coverage*100:.2f}%</h3>
                <h3>Severity: {severity}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-safe">
                <h2>✅ AREA APPEARS SAFE</h2>
                <h3>Coverage: {coverage*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Display images
        def make_overlay(base_rgb, mask, color=(1.0, 0.0, 0.0), alpha=0.4):
            if base_rgb.ndim == 2:
                base_rgb = np.stack([base_rgb]*3, axis=2)
            mask3 = np.expand_dims(mask.astype(np.float32), 2)
            color_arr = np.array(color, dtype=np.float32).reshape(1,1,3)
            return np.clip(base_rgb*(1 - alpha*mask3) + color_arr*(alpha*mask3), 0, 1)
        
        overlay_image = make_overlay(img_display, binary_mask)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_display, caption="Original Image", use_container_width=True)
        with col2:
            prob_display = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-6)
            st.image(prob_display, caption="Risk Heatmap", use_container_width=True)
        with col3:
            st.image(overlay_image, caption="Detection Overlay", use_container_width=True)
        
        # Debug information
        with st.expander("🔍 Model Debug Info"):
            st.write(f"**Probability map range:** {prob_map.min():.4f} to {prob_map.max():.4f}")
            st.write(f"**Mean probability:** {prob_map.mean():.4f}")
            st.write(f"**Coverage:** {coverage:.4f} ({coverage*100:.2f}%)")
            st.write(f"**Threshold used:** {prob_threshold}")
            st.write(f"**Pixels above threshold:** {np.sum(prob_map >= prob_threshold)}/{prob_map.size}")
        
        # Emergency response for alerts
        if label == "LANDSLIDE ALERT":
            st.markdown("---")
            st.markdown("### 🚨 Emergency Response")
            
            emergency_msg = f"""🚨 LANDSLIDE ALERT 🚨

Location: {location_name}
Coordinates: {latitude}, {longitude}
Severity: {severity}
Coverage: {coverage*100:.1f}%
Time: {record['timestamp']}

Emergency Services Contacts:
📞 Police/Fire: 911 (US) / 112 (EU) / 100 (India)
🚑 Ambulance: 911 (US) / 102 (India)
🆘 Disaster Management: Local authorities

Action Required:
- Evacuate affected areas immediately
- Alert nearby residents
- Contact local disaster response team
"""
            
            st.text_area("Alert Message", emergency_msg, height=300)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📞 Emergency Contacts"):
                    st.info("📞 Emergency Numbers:\n- 911 (US)\n- 112 (EU)\n- 100 (India)")
            with col2:
                if st.button("🗺️ Open in Maps"):
                    maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
                    st.markdown(f"[Open Location in Maps]({maps_url})")
            with col3:
                if st.button("📱 Generate SMS"):
                    sms_text = f"LANDSLIDE ALERT at {location_name} ({latitude}, {longitude}). Coverage: {coverage*100:.1f}%. Evacuate immediately!"
                    st.code(f"SMS: {sms_text}")
        
        # Generate comprehensive PDF report
        st.markdown("---")
        st.markdown("### 📄 Generate Report")
        
        with st.expander("📋 View Full Report Details", expanded=False):
            st.markdown("**Report Contents:**")
            st.write(f"• Location: {location_name}")
            st.write(f"• Coordinates: {latitude}, {longitude}")
            st.write(f"• Alert Status: {label}")
            st.write(f"• Risk Coverage: {coverage*100:.2f}%")
            st.write(f"• Severity Level: {severity}")
            st.write(f"• Data Source: {data_source}")
            st.write(f"• Analysis Time: {record['timestamp']}")
        
        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = create_landslide_pdf(record)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Landslide_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col2:
            # Text report
            text_report = f"""LANDSLIDE DETECTION REPORT
{'='*60}

Location: {location_name}
Coordinates: {latitude}, {longitude}
Alert Status: {label}
Severity: {severity}
Coverage: {coverage*100:.2f}%
Data Source: {data_source}
Timestamp: {record['timestamp']}

Analysis Details:
- Probability threshold: {prob_threshold}
- Coverage threshold: {coverage_threshold}
- Pixels flagged: {np.sum(prob_map >= prob_threshold)}/{prob_map.size}

{'='*60}
"""
            st.download_button(
                label="📝 Download Text Report",
                data=text_report.encode('utf-8'),
                file_name=f"Landslide_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Alert history
    if st.session_state.landslide_alert_log:
        st.markdown("---")
        st.markdown("### 📋 Alert History")
        df = pd.DataFrame(st.session_state.landslide_alert_log)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No alerts recorded yet. Upload an image to begin analysis.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #e0e7ff; padding: 20px;">
    <p>🌍 <strong>EcoCast Disaster Detection System</strong> | Powered by AI & Real-time Data</p>
    <p style="font-size: 12px;">For emergency situations, always contact local authorities immediately.</p>
</div>
""", unsafe_allow_html=True)
