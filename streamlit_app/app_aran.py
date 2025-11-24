import datetime as dt
import io
import json
import webbrowser
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import quote
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox

# Keras backend for custom metrics
K = keras.backend

# ---------------------------------------------------------------------------
# Utility: robust normalization helpers
# ---------------------------------------------------------------------------
def _safe_mid(values: np.ndarray, fallback: float = 0.5) -> float:
    """Return a stable mid-point (approx half of top range) using percentiles."""
    vmax = float(np.nanmax(values))
    if vmax <= 0:
        return fallback
    p75 = float(np.nanpercentile(values, 75))
    return max(min(p75, vmax), 1e-3)

def _invert_channel(chan: np.ndarray, mid: float) -> np.ndarray:
    inv = 1.0 - chan / (mid + 1e-6)
    return np.clip(inv, 0.0, 1.0)

DEFAULT_MODEL_PATH = Path(__file__).parent / "best_model.keras"

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


def _sentinel_config_from_env():
    cfg = SHConfig()
    cfg.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    cfg.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not cfg.sh_client_id or not cfg.sh_client_secret:
        return None
    return cfg

def fetch_sentinel_rgbnir(lat: float, lon: float, size: int = 128, config: SHConfig = None):
    """Return array shape (size,size,4): B02,B03,B04,B08 -> BLUE,GREEN,RED,NIR"""
    cfg = config or _sentinel_config_from_env()
    if cfg is None:
        raise RuntimeError("Sentinel Hub credentials missing (set in .env)")

    # Create bounding box with ~1.92 km buffer (128 pixels * 15m resolution)
    buffer_m = 128 * 15 / 2
    deg_lon = buffer_m / 111320
    deg_lat = buffer_m / 110540

    min_lon = lon - deg_lon
    max_lon = lon + deg_lon
    min_lat = lat - deg_lat
    max_lat = lat + deg_lat

    bbox = BBox([min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)


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

    arr = request.get_data()[0]  # shape (size,size,4)
    # SentinelHub returns bands in range [0, ~10000] depending; normalize to 0..1
    arr = np.clip(arr.astype(np.float32), 0, None)
    # Simple per-band normalization: divide by 10000 (typical reflectance scale)
    arr = arr / 10000.0
    return arr  # order: B02 (blue), B03 (green), B04 (red), B08 (nir)

def fetch_dem_and_slope(lat: float, lon: float, size: int = 128, config: SHConfig = None):
    """Return tuple (dem, slope) where both are (size,size) float arrays (meters, slope magnitude)."""
    cfg = config or _sentinel_config_from_env()
    if cfg is None:
        raise RuntimeError("Sentinel Hub credentials missing (set in .env)")

    # DEM often 30m resolution; use larger buffer to match scale
    buffer_m = 128 * 15 / 2
    deg_lon = buffer_m / 111320
    deg_lat = buffer_m / 110540

    min_lon = lon - deg_lon
    max_lon = lon + deg_lon
    min_lat = lat - deg_lat
    max_lat = lat + deg_lat

    bbox = BBox([min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

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
    # Handle both 2D and 3D array formats
    if data.ndim == 3:
        dem = data[:,:,0].astype(np.float32)
    else:
        dem = data.astype(np.float32)
    # compute slope (simple gradient magnitude)
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx * gx + gy * gy)
    return dem, slope

def fetch_feature_stack(lat: float, lon: float, size: int = 128):
    """Return (feature_stack, rgbnir_raw) where rgbnir_raw has bands B02,B03,B04,B08 normalized 0..1."""
    cfg = _sentinel_config_from_env()
    if cfg is None:
        raise RuntimeError("Sentinel Hub credentials missing (set in .env)")

    rgbnir = fetch_sentinel_rgbnir(lat, lon, size=size, config=cfg)  # shape (size,size,4)
    dem, slope = fetch_dem_and_slope(lat, lon, size=size, config=cfg)  # (size,size)

    # Create properly structured data for feature engineering
    feature_data = np.zeros((size, size, 14), dtype=np.float32)

    # Map to the expected channel positions
    feature_data[:, :, 1] = rgbnir[:, :, 0]  # Blue -> channel 1
    feature_data[:, :, 2] = rgbnir[:, :, 1]  # Green -> channel 2
    feature_data[:, :, 3] = rgbnir[:, :, 2]  # Red -> channel 3
    feature_data[:, :, 7] = rgbnir[:, :, 3]  # NIR -> channel 7
    feature_data[:, :, 12] = slope          # Slope -> channel 12
    feature_data[:, :, 13] = dem            # DEM -> channel 13

    feature_stack = build_feature_stack(feature_data)
    return feature_stack, rgbnir

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

# ---------------------------------------------------------------------------
# U-Net architecture definition (matches training script)
# ---------------------------------------------------------------------------
def unet_model():
    inputs = tf.keras.layers.Input((128, 128, 6))

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

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

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


def load_landslide_model(model_path: str) -> tf.keras.Model:
    """Robust model loader handling .keras (full model), .h5, and weights-only fallback.
    Also compiles model with correct metrics for consistent inference API.
    """
    resolved = Path(model_path)
    tf_version = tf.__version__

    if not resolved.exists():
        st.error(f"Primary model file not found: {resolved}")
        st.stop()

    load_errors = []

    # 1. Try full model load for .keras / .h5
    try:
        model = tf.keras.models.load_model(
            resolved,
            custom_objects={"precision_m": precision_m, "recall_m": recall_m, "f1_m": f1_m},
        )
        st.caption(f"Loaded full model from {resolved.name} (TensorFlow {tf_version}).")
        try:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[precision_m, recall_m, f1_m])
        except Exception:  # pragma: no cover
            pass
        return model
    except Exception as e:
        load_errors.append(f"full:{e}")

    # 2. If a secondary H5 exists, try that
    alt_h5 = Path(__file__).parent / "model_save.h5"
    if alt_h5.exists():
        try:
            model = tf.keras.models.load_model(
                alt_h5,
                custom_objects={"precision_m": precision_m, "recall_m": recall_m, "f1_m": f1_m},
            )
            st.caption(f"Loaded fallback full model from {alt_h5.name} (TensorFlow {tf_version}).")
            return model
        except Exception as e:
            load_errors.append(f"h5:{e}")

    # 3. Build architecture and try loading weights
    try:
        model = unet_model()
        model.load_weights(str(resolved))
        st.caption(f"Loaded weights into fresh U-Net architecture from {resolved.name}.")
        try:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[precision_m, recall_m, f1_m])
        except Exception:  # pragma: no cover
            pass
        return model
    except Exception as e:
        load_errors.append(f"weights:{e}")

    st.error("All load strategies failed. Errors:\n" + "\n".join(load_errors))
    st.stop()


def build_feature_stack(image_cube: np.ndarray) -> np.ndarray:
    """Replicate the feature engineering used during training with more robust scaling."""
    data = np.nan_to_num(image_cube, nan=1e-6)
    
    # Define data variables first
    data_blue = data[:, :, 1]   # channel 1
    data_green = data[:, :, 2]  # channel 2
    data_red = data[:, :, 3]    # channel 3
    data_nir = data[:, :, 7]    # channel 7
    
    # Now calculate normalization values
    mid_rgb = _safe_mid(np.stack([data_red, data_green, data_blue], axis=2))
    mid_slope = _safe_mid(data[:, :, 12])
    mid_elevation = _safe_mid(data[:, :, 13])
    
    ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red) + 1e-6)

    feature_stack = np.zeros((128, 128, 6), dtype=np.float32)
    feature_stack[:, :, 0] = _invert_channel(data_red, mid_rgb)   # RED
    feature_stack[:, :, 1] = _invert_channel(data_green, mid_rgb)  # GREEN
    feature_stack[:, :, 2] = _invert_channel(data_blue, mid_rgb)  # BLUE
    feature_stack[:, :, 3] = ndvi  # NDVI
    feature_stack[:, :, 4] = _invert_channel(data[:, :, 12], mid_slope)  # SLOPE
    feature_stack[:, :, 5] = _invert_channel(data[:, :, 13], mid_elevation)  # ELEVATION
    # NDVI can be [-1,1]; keep as-is but replace infs
    feature_stack[:, :, 3] = np.nan_to_num(feature_stack[:, :, 3], nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack


def process_image(uploaded_file) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0

    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]

    # Use a "pseudo-NIR" so NDVI isn’t completely broken
    nir = (green * 1.2 + red * 0.2).astype(np.float32)  # better approximation

    ndvi = (nir - red) / (nir + red + 1e-6)

    mid_rgb = _safe_mid(np.stack([red, green, blue], axis=2))

    features = np.zeros((128, 128, 6), dtype=np.float32)
    features[:, :, 0] = _invert_channel(red, mid_rgb)
    features[:, :, 1] = _invert_channel(green, mid_rgb)
    features[:, :, 2] = _invert_channel(blue, mid_rgb)
    features[:, :, 3] = ndvi

    # Better fake slope: image gradients
    gy, gx = np.gradient(np.mean(img_array, axis=2))
    slope_fake = np.sqrt(gx*gx + gy*gy)

    slope_mid = _safe_mid(slope_fake)
    features[:, :, 4] = _invert_channel(slope_fake, slope_mid)

    # Better fake elevation: brightness-based
    brightness = np.mean(img_array, axis=2)
    elev_mid = _safe_mid(brightness)
    features[:, :, 5] = _invert_channel(brightness, elev_mid)

    return features, img_array


def infer_mask(
    model: tf.keras.Model,
    features: np.ndarray,
    prob_threshold: float,
    coverage_threshold: float,
) -> Dict:
    """Run the model and derive high-level risk metadata."""
    input_batch = np.expand_dims(features, axis=0)
    preds = model.predict(input_batch, verbose=0)
    prob_map = preds[0, :, :, 0]
    # Auto-suggest threshold (quantile) for debug
    suggested_thr = float(np.quantile(prob_map, 0.98))
    binary_mask = (prob_map >= prob_threshold).astype(np.uint8)
    coverage = float(binary_mask.sum() / binary_mask.size)
    label = "LANDSLIDE ALERT" if coverage >= coverage_threshold else "Stable"
    return {
        "prob_map": prob_map,
        "binary_mask": binary_mask,
        "coverage": coverage,
        "label": label,
        "suggested_threshold": suggested_thr,
    }


def build_alert_record(
    *,
    label: str,
    coverage: float,
    latitude: float,
    longitude: float,
    location_name: str,
    incident_type: str,
    severity: str,
    operator_notes: str,
) -> Dict:
    now = dt.datetime.utcnow()
    return {
        "label": label,
        "coverage": coverage,
        "latitude": latitude,
        "longitude": longitude,
        "location": location_name or "Unknown",
        "incident_type": incident_type,
        "severity": severity,
        "operator_notes": operator_notes,
        "timestamp": now.isoformat() + "Z",
    }


def build_report_dataframe(record: Dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp_utc": record["timestamp"],
                "location": record["location"],
                "latitude": record["latitude"],
                "longitude": record["longitude"],
                "incident_type": record["incident_type"],
                "severity": record["severity"],
                "model_decision": record["label"],
                "coverage_fraction": record["coverage"],
                "operator_notes": record["operator_notes"],
            }
        ]
    )


def dataframe_to_bytes(df: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    buffer.write(df.to_csv(index=False).encode("utf-8"))
    buffer.seek(0)
    return buffer


def alert_emergency_services(record: Dict) -> str:
    """Generate emergency service contact information."""
    message = f"""🚨 LANDSLIDE ALERT 🚨
    
Location: {record['location']}
Coordinates: {record['latitude']}, {record['longitude']}
Severity: {record['severity']}
Coverage: {record['coverage']*100:.1f}%
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
    return message


def render_plan(record: Dict):
    recommendations = [
        "Activate local emergency communication channels",
        "Dispatch reconnaissance drone/crew to validate imagery",
        "Stage evacuation transport near vulnerable communities",
        "Coordinate with hospital triage units and supply depots",
        "Update public dashboard and push mobile notifications",
    ]
    st.markdown("#### Recommended response plan")
    for item in recommendations:
        st.write(f"- {item}")
    st.caption(
        "Tailor these actions with local SOPs and combine with hydrological / rainfall forecasts "
        "for a multi-hazard view."
    )


@st.cache_data(show_spinner=False)
def load_reference_predictions() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / "validation_final_predictions.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Landslide Guardian", page_icon=None, layout="centered")
st.title("Landslide Guardian – Disaster Detection & Response Console")
st.caption(
    "Real-time landslide segmentation + Incident management."
)

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

# with st.sidebar:
#     st.header("Detection settings")
#     model_path = st.text_input("Model file", value=str(DEFAULT_MODEL_PATH))
#     prob_threshold = st.slider("Mask probability threshold", 0.1, 0.9, value=0.5, step=0.05)
#     coverage_threshold = st.slider("Minimum affected fraction", 0.0, 0.3, value=0.01, step=0.005)
#     st.divider()
#     st.subheader("Alert routing")
#     enable_email = st.checkbox("Email local authorities")
#     email_config = {}
#     if enable_email:
#         email_config = {
#             "sender": st.text_input("Sender email"),
#             "password": st.text_input("Sender password", type="password"),
#             "recipient": st.text_input("Recipient email"),
#             "smtp_host": st.text_input("SMTP host", value="smtp.gmail.com"),
#             "smtp_port": st.number_input("SMTP port", value=465, step=1),
#         }
    # st.divider()
    # reference_df = load_reference_predictions()
    # if not reference_df.empty:
    #     st.metric("Reference validation alerts", reference_df["label"].value_counts().get("yes", 0))

# Defaults
model_path = str(DEFAULT_MODEL_PATH)
prob_threshold = 0.3  # Lower threshold for uploaded images
coverage_threshold = 0.005  # More sensitive coverage detection

st.subheader("Choose Data Source")
data_source = st.radio(
    "Select input method:",
    ["Upload Image", "Real-time Satellite Data"],
    horizontal=True
)

st.divider()

uploaded = None
latitude = 28.3949
longitude = 84.1240
location_name = "Unknown Location"

if data_source == "Upload Image":
    st.subheader("Upload Image for Analysis")
    uploaded = st.file_uploader(
        "Upload an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
    )
    st.info("Upload a satellite or aerial image of the area to check for landslide risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=28.3949)
    with col2:
        longitude = st.number_input("Longitude", value=84.1240)
    location_name = st.text_input("Location name", value="Nagaland")
    
else:
    st.subheader("Real-time Satellite Data")
    st.info("Enter coordinates to fetch satellite imagery")
    
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=30.557367)
    with col2:
        longitude = st.number_input("Longitude", value=79.565394)
    location_name = st.text_input("Location name", value="Joshimath")
    
    if st.button("Fetch Satellite Data"):
        try:
            st.info(f"Fetching Sentinel-2 + DEM for: {latitude}, {longitude}")
            with st.spinner("Downloading imagery from Sentinel Hub..."):
                features, rgbnir_raw = fetch_feature_stack(latitude, longitude, size=128)
            st.success("Satellite data fetched and preprocessed.")
            # store features & raw rgbnir in session_state
            st.session_state["fetched_features"] = features
            st.session_state["fetched_rgbnir"] = rgbnir_raw
        except Exception as e:
            st.error(f"Satellite fetch failed: {e}")
            st.exception(e)

st.divider()
run_button = st.button("Analyze for Landslide Risk", type="primary")

if run_button:
    if data_source == "Upload Image":
        # Upload image path: require uploaded image
        if uploaded is None:
            st.error("Please upload an image first.")
        else:
            st.write("Analyzing image...")
            model = load_landslide_model(model_path)
            features, img_display = process_image(uploaded)
            results = infer_mask(model, features, prob_threshold, coverage_threshold)
            
            severity = "Critical" if results["coverage"] > 0.05 else "High" if results["coverage"] > 0.01 else "Moderate"
            
            record = build_alert_record(
                label=results["label"],
                coverage=results["coverage"],
                latitude=latitude,
                longitude=longitude,
                location_name=location_name,
                incident_type="Landslide",
                severity=severity,
                operator_notes="Auto-detected",
            )
            st.session_state.alert_log.insert(0, record)
    else:
        # Satellite data path: use fetched features
        features = st.session_state.get("fetched_features")
        rgbnir_raw = st.session_state.get("fetched_rgbnir")
        if features is None or rgbnir_raw is None:
            st.error("No satellite data fetched yet. Click 'Fetch Satellite Data' first.")
        else:
            st.write("Analyzing satellite data...")
            model = load_landslide_model(model_path)
            results = infer_mask(model, features, prob_threshold, coverage_threshold)
            
            severity = "Critical" if results["coverage"] > 0.05 else "High" if results["coverage"] > 0.01 else "Moderate"
            
            record = build_alert_record(
                label=results["label"],
                coverage=results["coverage"],
                latitude=latitude,
                longitude=longitude,
                location_name=location_name,
                incident_type="Landslide",
                severity=severity,
                operator_notes="Auto-detected from satellite",
            )
            st.session_state.alert_log.insert(0, record)
            
            # Build true-color RGB from raw bands (B04 red, B03 green, B02 blue)
            rgb_display = np.stack([
                rgbnir_raw[:, :, 2],  # Red
                rgbnir_raw[:, :, 1],  # Green
                rgbnir_raw[:, :, 0],  # Blue
            ], axis=2)
            img_display = np.clip(rgb_display, 0, 1)

    # Display results (common for both paths)
    if results["label"] == "LANDSLIDE ALERT":
        st.error(f"🚨 {results['label']} - Risk coverage: {results['coverage']*100:.2f}%")
    else:
        st.success(f"✅ {results['label']} - Area appears safe")
    
    # Replace your current display code with:
    # Helper: overlay mask
    def make_overlay(base_rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha: float = 0.4):
        if base_rgb.ndim == 2:  # grayscale -> stack
            base_rgb = np.stack([base_rgb]*3, axis=2)
        mask3 = np.expand_dims(mask.astype(np.float32), 2)
        color_arr = np.array(color, dtype=np.float32).reshape(1,1,3)
        return np.clip(base_rgb*(1 - alpha*mask3) + color_arr*(alpha*mask3), 0, 1)

    overlay_image = make_overlay(img_display, results["binary_mask"]) if "binary_mask" in results else img_display

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_display, caption="True Color (RGB)", use_container_width=True)
    with col2:
        # Normalize probability map for better visualization
        prob_display = (results["prob_map"] - results["prob_map"].min()) / (results["prob_map"].max() - results["prob_map"].min() + 1e-6)
        st.image(prob_display, caption="Risk Heatmap", use_container_width=True, clamp=True)
    with col3:
        st.image(overlay_image, caption="Mask Overlay", use_container_width=True)
    
    # Debug section to understand model predictions
    with st.expander("🔍 Model Debug Info"):
        st.write(f"**Probability map range:** {results['prob_map'].min():.4f} to {results['prob_map'].max():.4f}")
        st.write(f"**Mean probability:** {results['prob_map'].mean():.4f}")
        st.write(f"**Coverage:** {results['coverage']:.4f} ({results['coverage']*100:.2f}%)")
        st.write(f"**Threshold used:** {prob_threshold}")
        st.write(f"**Pixels above threshold:** {np.sum(results['prob_map'] >= prob_threshold)}/{results['prob_map'].size}")
        # Channel stats (first 6 feature channels) for diagnosing black outputs
        if data_source == "Upload Image":
            source_feats = features
        else:
            source_feats = features if 'features' in locals() else None
        if source_feats is not None:
            ch_stats = {f"ch{i}": (float(source_feats[:, :, i].min()), float(source_feats[:, :, i].max()), float(source_feats[:, :, i].mean())) for i in range(6)}
            st.write("**Feature channel (min, max, mean):**", ch_stats)
        if data_source == "Real-time Satellite Data" and 'rgbnir_raw' in locals():
            raw_stats = {f"raw_band{i}": (float(rgbnir_raw[:, :, i].min()), float(rgbnir_raw[:, :, i].max())) for i in range(rgbnir_raw.shape[2])}
            st.write("**Raw Sentinel band min/max:**", raw_stats)
    
    # Emergency alerts
    if results["label"] == "LANDSLIDE ALERT":
        st.divider()
        st.subheader("🚨 Emergency Response")
        
        emergency_msg = alert_emergency_services(record)
        st.text_area("Alert Message", emergency_msg, height=300)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📞 Call Emergency (911)"):
                st.warning("Dial 911 or your local emergency number immediately!")
                webbrowser.open("tel:911")
        with col2:
            if st.button("📱 Send SMS Alert"):
                sms_text = f"LANDSLIDE ALERT at {location_name} ({latitude}, {longitude}). Coverage: {results['coverage']*100:.1f}%. Evacuate immediately!"
                st.info(f"SMS: {sms_text}")
                st.code(f"sms:?body={quote(sms_text)}")
        with col3:
            if st.button("🗺️ Open in Maps"):
                maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
                webbrowser.open(maps_url)
                st.success(f"Opening location in maps: {maps_url}")

st.divider()
st.subheader("Alert History")
if st.session_state.alert_log:
    st.dataframe(pd.DataFrame(st.session_state.alert_log), use_container_width=True)
else:
    st.info("No alerts yet. Upload an image to begin analysis.")

st.caption(
    "This Streamlit console stays in sync with the provided Keras checkpoints. Adjust thresholds "
    "based on validation statistics for your geography before operational use."
)