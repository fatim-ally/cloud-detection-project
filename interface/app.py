"""
app.py
------
Streamlit UI skeleton for the "AI-Based Cloud Detection and
Localization in Satellite Images" project — Interface & Integration
component.

STATUS: UI skeleton only. The AI model and preprocessing pipeline
are NOT implemented yet — this file uses placeholder/mock outputs
so the interface can be built, tested, and demoed independently of
the model work. Every point marked "MOCK" below is where real
model output will later be plugged in.

Author: Samra Syed
"""

import streamlit as st
from PIL import Image

from modules import preprocessing
from modules import prediction
from modules import visualization

# ---------------------------------------------------------------------
# 1. PAGE CONFIG + PROJECT TITLE / DESCRIPTION
# ---------------------------------------------------------------------
st.set_page_config(page_title="Cloud Detection & Localization", layout="wide")

st.title("☁️ AI-Based Cloud Detection and Localization in Satellite Images")
st.markdown(
    """
    This application allows a user to upload a satellite image and view
    the model's cloud detection and localization results.

    **Pipeline:** Upload Image → Preprocessing → AI Detection → Results → Cloud Highlighting

    > ⚠️ **Note:** This is currently a UI skeleton. The preprocessing
    > pipeline and the trained U-Net model are not connected yet —
    > all results shown below are placeholder/mock outputs.
    """
)

st.divider()

# ---------------------------------------------------------------------
# 2. SATELLITE IMAGE UPLOAD WIDGET
# ---------------------------------------------------------------------
st.header("1. Upload Satellite Image")
uploaded_file = st.file_uploader(
    "Upload a satellite image (JPG, JPEG, PNG, TIFF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

original_image = None
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

# ---------------------------------------------------------------------
# 3. DISPLAY OF UPLOADED / ORIGINAL IMAGE
# ---------------------------------------------------------------------
if original_image is not None:
    st.subheader("Original Image")
    st.image(original_image, caption="Uploaded satellite image", use_container_width=True)
else:
    st.info("Please upload a satellite image to continue.")

st.divider()

# ---------------------------------------------------------------------
# 8. "RUN DETECTION" BUTTON
# ---------------------------------------------------------------------
st.header("2. Run Cloud Detection")
run_clicked = st.button("🚀 Run Detection", disabled=(original_image is None))

# Session state so results persist after the button click
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if run_clicked and original_image is not None:
    with st.spinner("Running preprocessing and detection..."):
        # --- MOCK PIPELINE CALLS ---
        # These already call the real module functions (preprocessing.py,
        # prediction.py, visualization.py). Right now those functions
        # internally return mock/placeholder data. Once the real
        # pipeline and U-Net model are ready, NO CHANGES are needed
        # here in app.py — only inside the module functions themselves.
        processed_image = preprocessing.preprocess_image(original_image)
        predicted_mask = prediction.predict_mask(processed_image)
        highlighted_image = visualization.highlight_clouds(original_image, predicted_mask)
        metrics = prediction.compute_metrics(predicted_mask)

        st.session_state.processed_image = processed_image
        st.session_state.predicted_mask = predicted_mask
        st.session_state.highlighted_image = highlighted_image
        st.session_state.metrics = metrics
        st.session_state.results_ready = True

st.divider()

# ---------------------------------------------------------------------
# 4, 5, 6. RESULT PLACEHOLDER SECTIONS
# ---------------------------------------------------------------------
st.header("3. Results")

if st.session_state.results_ready:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Preprocessed Image")
        st.image(st.session_state.processed_image, use_container_width=True)
        st.caption("MOCK — will show real preprocessed output once pipeline is connected.")

    with col2:
        st.subheader("Predicted Cloud Mask")
        st.image(st.session_state.predicted_mask, use_container_width=True, clamp=True)
        st.caption("MOCK — will show real U-Net predicted mask.")

    with col3:
        st.subheader("Cloud Regions Highlighted")
        st.image(st.session_state.highlighted_image, use_container_width=True)
        st.caption("MOCK — will show real highlighted detection output.")

    st.divider()

    # -------------------------------------------------------------
    # 7. PLACEHOLDER METRICS
    # -------------------------------------------------------------
    st.header("4. Evaluation Metrics")
    m = st.session_state.metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("IoU", f"{m['IoU']:.2f}")
    metric_col2.metric("Dice", f"{m['Dice']:.2f}")
    metric_col3.metric("Precision", f"{m['Precision']:.2f}")
    metric_col4.metric("Recall", f"{m['Recall']:.2f}")
    st.caption("MOCK — will be calculated against ground-truth masks once the model is trained.")
else:
    st.info("Results will appear here after you upload an image and click **Run Detection**.")

st.divider()
st.caption("Interface & Integration component — Samra Syed | AI-Based Cloud Detection and Localization in Satellite Images")
