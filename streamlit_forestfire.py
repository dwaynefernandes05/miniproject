# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io
import time
import base64
from datetime import datetime

# Optional PDF generator (fpdf). We'll try to import and fall back gracefully.
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Wildfire Detection Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS for styling
# -----------------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg,#071021, #0b1220);
        color: #e6eef8;
    }
    /* Header animation */
    .title {
        font-size:38px;
        font-weight:800;
        letter-spacing: -0.5px;
        background: -webkit-linear-gradient(#ff8a65, #ffd180);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align:center;
    }
    .subtitle { color:#bfcfe6; text-align:center; margin-bottom:18px; }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius:12px;
        padding:18px;
        border: 1px solid rgba(255,255,255,0.04);
        box-shadow: 0 6px 20px rgba(2,6,23,0.6);
    }
    .big-metric { font-size: 28px; font-weight:700; color:#fff; }
    .small { color:#9fb3d1; }
    .btn {
        background: linear-gradient(90deg,#ff6b6b,#ffb86b);
        color: #0b1220;
        font-weight:700;
        padding:10px 18px;
        border-radius:8px;
        border: none;
    }
    .footer { color:#8ea2c3; font-size:12px; text-align:center; margin-top:14px; }
    /* Hover effects */
    .card:hover { transform: translateY(-6px); transition: all 0.25s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource
def load_models(cnn_path="fire_detection_cnnn_final.h5", mnv2_path="mobilenet_fire_final.h5"):
    """Load the Keras models once and cache them."""
    # Use allow_pickle False by default; these are regular saved models
    cnn = tf.keras.models.load_model(cnn_path)
    mnv2 = tf.keras.models.load_model(mnv2_path)
    return cnn, mnv2

def to_rgb(img_pil):
    """Ensure PIL image is RGB (convert RGBA/LA/CMYK to RGB)."""
    if img_pil.mode == "RGBA" or img_pil.mode == "LA":
        return Image.alpha_composite(Image.new("RGBA", img_pil.size, (255,255,255)), img_pil).convert("RGB")
    if img_pil.mode != "RGB":
        return img_pil.convert("RGB")
    return img_pil

def preprocess_for_cnn(pil_img, target=128):
    """Preprocess for custom CNN (128x128, normalized 0-1)."""
    img = to_rgb(pil_img)
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_mnv2(pil_img, target=224):
    """Preprocess for MobileNetV2 (224x224) using MobileNet preprocessing."""
    img = to_rgb(pil_img)
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img).astype("float32")
    # MobileNetV2 preprocess: scale to [-1,1]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_and_format(model, prepped):
    """Run prediction and return confidence (0-1). Assumes model outputs a single sigmoid."""
    preds = model.predict(prepped)
    # Handle different output shapes
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0,0])
    elif preds.ndim == 1:
        prob = float(preds[0])
    else:
        # fallback: take first scalar-like value
        prob = float(np.ravel(preds)[0])
    return np.clip(prob, 0.0, 1.0)

def make_pdf_report_bytes(report_text, thumbnail_bytes=None):
    """Return bytes of a PDF report using fpdf if available, else returns None."""
    if not FPDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 8, "Wildfire Detection Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", '', 11)
    # add report text (wrap)
    for line in report_text.split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.ln(6)
    if thumbnail_bytes:
        # insert thumbnail below text (resize to reasonable)
        try:
            pdf.image(thumbnail_bytes, x=pdf.l_margin, w=80)
        except Exception:
            pass
    return pdf.output(dest="S").encode("latin-1")

def make_txt_report_bytes(report_text):
    """Return bytes for a plain text report."""
    return report_text.encode("utf-8")

# -----------------------------
# Load models
# -----------------------------
with st.spinner("Thinking...."):
    try:
        model_cnn, model_mnv2 = load_models()
    except Exception as e:
        st.error("Failed to load models. Make sure .h5 files are in app directory and valid Keras models.")
        st.exception(e)
        st.stop()

# -----------------------------
# Sidebar: navigation & settings (multi-page)
# -----------------------------
st.sidebar.markdown("## 🛰️ Wildfire Detection")
page = st.sidebar.radio("Navigate", ("Detect", "Reports"))

# Global: model choice (toggle)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔁 Model Selection")
model_choice = st.sidebar.radio("", ("MobileNetV2 (Satellite)", "Custom CNN (Ground/Aerial)"))

# -----------------------------
# Page: DETECT
# -----------------------------
if page == "Detect":
    st.markdown('<div class="title">Image-based Wildfire Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an image, select model, and run detection.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    st.markdown("**Or use sample images** (place sample images in `samples/` folder to display here).")
    run_button = st.button("Detect Fire", use_container_width=True, type="primary")
    st.write("")  # spacing
    st.markdown("**Model selected:** " + ( "MobileNetV2 (Satellite)" if "MobileNetV2" in model_choice else "Custom CNN (Ground/Aerial)"))

    if uploaded is not None:
        try:
            img_pil = Image.open(uploaded)
        except Exception as e:
            st.error("Could not open the uploaded file as an image.")
            st.exception(e)
            st.stop()

        # show preview
        st.image(img_pil, caption="Preview: Uploaded Image", use_container_width=False)

        # select model-specific preprocess
        if "MobileNetV2" in model_choice:
            prepped = preprocess_for_mnv2(img_pil, target=224)
            selected_model = model_mnv2
            expected_shape = (1, 224, 224, 3)
        else:
            prepped = preprocess_for_cnn(img_pil, target=128)
            selected_model = model_cnn
            expected_shape = (1, 128, 128, 3)

        # run prediction when user clicks Detect
        if run_button:
            # small animation
            with st.spinner("Running model..."):
                # quick sleep to show spinner nicely
                time.sleep(0.6)
                try:
                    prob = predict_and_format(selected_model, prepped)
                except Exception as e:
                    st.error("Model prediction failed. Check model input shapes and model file compatibility.")
                    st.exception(e)
                    st.stop()

            is_fire = prob >= 0.5
            confidence = prob if is_fire else (1 - prob)

            # Save last prediction in session state for Reports page
            st.session_state['last_prediction'] = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': "MobileNetV2 (224x224)" if "MobileNetV2" in model_choice else "Custom CNN (128x128)",
                'filename': uploaded.name,
                'probability': float(prob),
                'is_fire': bool(is_fire)
            }

            # display results card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if is_fire:
                st.markdown("## 🔥 Fire Detected")
                st.markdown(f"**Confidence:** {prob:.4f}")
                st.success("Detected: Fire (model believes this image contains active fire/smoke)")
            else:
                st.markdown("## 🌿 No Fire Detected")
                st.markdown(f"**Confidence (No fire):** {(1-prob):.4f}")
                st.info("Detected: No fire (model does not detect clear fire signatures)")
            st.markdown("</div>", unsafe_allow_html=True)

            # quick progress bar to feel interactive
            p = st.progress(0)
            for i in range(30):
                time.sleep(0.01)
                p.progress((i+1)/30)
            p.empty()


    else:
        st.info("Upload an image on the left and click Detect.")

# -----------------------------
# Page: REPORTS
# -----------------------------
elif page == "Reports":
    st.markdown('<div class="title">Reports & Downloads</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Generate a concise PDF/TXT report of your last prediction for sharing with stakeholders.</div>', unsafe_allow_html=True)

    if 'last_prediction' not in st.session_state:
        st.warning("No prediction available yet. Run a detection on the 'Detect' page first.")
    else:
        lp = st.session_state['last_prediction']
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"**Last Run:** {lp['timestamp']}")
        st.write(f"**Model:** {lp['model']}")
        st.write(f"**File:** {lp['filename']}")
        st.write(f"**Probability (fire):** {lp['probability']:.4f}")
        st.write(f"**Decision:** {'Fire' if lp['is_fire'] else 'No Fire'}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Prepare report text
        report_lines = [
            "Wildfire Detection Report",
            "=========================",
            f"Timestamp: {lp['timestamp']}",
            f"Uploaded file: {lp['filename']}",
            f"Model used: {lp['model']}",
            f"Predicted: {'Fire' if lp['is_fire'] else 'No Fire'}",
            f"Model probability (fire): {lp['probability']:.4f}",
            "",
        ]
        report_text = "\n".join(report_lines)

        # Try to include the thumbnail if the file still exists in session state (we kept the uploaded object on Detect)
        thumbnail_bytes = None
        try:
            # get uploaded file bytes by re-opening file (if user uploaded earlier)
            # Note: in some Streamlit sessions the original UploadFile may not persist across reruns;
            # in that case the thumbnail will be omitted and TXT fallback still works.
            uploaded_file_obj = st.session_state.get('uploaded_file_obj', None)
        except Exception:
            uploaded_file_obj = None

        # create PDF if possible
        pdf_bytes = None
        if FPDF_AVAILABLE:
            try:
                pdf_bytes = make_pdf_report_bytes(report_text, thumbnail_bytes=None)
            except Exception:
                pdf_bytes = None

        if pdf_bytes:
            st.success("PDF report generated.")
            st.download_button(
                label="📥 Download PDF report",
                data=pdf_bytes,
                file_name=f"wildfire_report_{lp['timestamp'].replace(' ','_').replace(':','-')}.pdf",
                mime="application/pdf"
            )
        else:
            # TXT fallback
            txt_bytes = make_txt_report_bytes(report_text)
            st.info("PDF generator not available — downloading simple TXT report.")
            st.download_button(
                label="📥 Download TXT report",
                data=txt_bytes,
                file_name=f"wildfire_report_{lp['timestamp'].replace(' ','_').replace(':','-')}.txt",
                mime="text/plain"
            )

# -----------------------------
# End of app
# -----------------------------
