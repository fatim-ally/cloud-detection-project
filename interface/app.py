"""
app.py
------
AI-Based Cloud Detection and Localization in Satellite Images
Interface & Integration — Streamlit UI Skeleton (single-file version)

STATUS: UI skeleton only. Mock/placeholder outputs for preprocessing,
prediction, and metrics. This version also fixes the metrics scope
problem: real IoU/Dice/Precision/Recall need a KNOWN ground-truth
mask to compare against, which is not available for an arbitrary
user upload — only for our labeled sample/test-set images.

Author: Samra Syed
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# =====================================================================
# SECTION A — PREPROCESSING (placeholder)
# =====================================================================
def preprocess_image(image: Image.Image):
    """TODO: real preprocessing pipeline. Currently returns image unchanged (MOCK)."""
    return image.copy()


# =====================================================================
# SECTION B — PREDICTION (U-Net goes here)
# =====================================================================
def load_model():
    """TODO: load trained U-Net from models/ folder. Currently returns None (MOCK)."""
    return None


MODEL = load_model()


def predict_mask(processed_image):
    """TODO: real U-Net inference. Currently returns a random mock mask."""
    width, height = processed_image.size if hasattr(processed_image, "size") else (256, 256)
    mock_mask = np.random.randint(0, 2, size=(height, width)) * 255
    return mock_mask.astype(np.uint8)


def compute_metrics(predicted_mask, ground_truth_mask):
    """
    Real IoU / Dice / Precision / Recall require BOTH a predicted mask
    AND a known ground-truth mask to compare against. If no ground
    truth is available (e.g. user uploaded an arbitrary image), metrics
    cannot be computed — this returns None in that case, and the UI
    hides the metrics section accordingly.

    TODO: once implemented, calculate real IoU/Dice/Precision/Recall
    between `predicted_mask` and `ground_truth_mask`.
    """
    if ground_truth_mask is None:
        return None

    # --- MOCK BEHAVIOR (remove once real evaluation is implemented) ---
    return {
        "IoU": 0.00,
        "Dice": 0.00,
        "Precision": 0.00,
        "Recall": 0.00,
    }


# =====================================================================
# SECTION C — VISUALIZATION
# =====================================================================
def highlight_clouds(original_image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.4):
    """TODO: real overlay logic once mask format is known. Currently mock red overlay."""
    original_rgba = original_image.convert("RGBA").resize((mask.shape[1], mask.shape[0]))
    overlay = Image.new("RGBA", original_rgba.size, color + (0,))
    overlay_pixels = overlay.load()
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] > 0:
                overlay_pixels[x, y] = color + (int(255 * alpha),)
    highlighted = Image.alpha_composite(original_rgba, overlay)
    return highlighted.convert("RGB")


# =====================================================================
# SECTION D — LABELED SAMPLE TEST SET (only source with ground truth)
# =====================================================================
SAMPLE_CSV_PATH = "val_patches.csv"  # validation split — has real gt_path, unlike test set


@st.cache_data
def load_sample_patches():
    """
    TODO: This CSV currently only has red/green/blue/nir band paths
    (no ground-truth mask column). Once a `gt_path` column is added
    for these labeled patches, load it here too.
    """
    try:
        return pd.read_csv(SAMPLE_CSV_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["patch_key", "scene_id"])


def load_sample_image_and_mask(patch_key: str):
    """
    TODO: Replace with real loading of the RGB bands for this patch
    (combined into a displayable image) plus its real ground-truth
    cloud mask, once available.

    Currently: returns a mock image + mock ground-truth mask so the
    "sample image" flow works end-to-end (MOCK behavior).
    """
    mock_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    mock_ground_truth = (np.random.randint(0, 2, size=(256, 256)) * 255).astype(np.uint8)
    return mock_image, mock_ground_truth


# =====================================================================
# SECTION E — STREAMLIT UI
# =====================================================================
st.set_page_config(page_title="Cloud Detection & Localization", layout="wide")

st.title("☁️ AI-Based Cloud Detection and Localization in Satellite Images")
st.markdown(
    """
    Upload a satellite image, or select a labeled sample image, to view
    the model's cloud detection and localization results.

    **Pipeline:** Choose Image → Preprocessing → AI Detection → Results → Cloud Highlighting → Metrics (if ground truth available)

    > ⚠️ **Note:** UI skeleton only — preprocessing pipeline and trained
    > U-Net model are not connected yet. All outputs below are mock.
    """
)
st.divider()

# ---- 1. Choose image source ----
st.header("1. Choose Satellite Image")
source_choice = st.radio(
    "Image source",
    ["Upload my own image (no ground truth)", "Select a labeled sample image (has ground truth)"],
    horizontal=True,
)

original_image = None
ground_truth_mask = None

if source_choice == "Upload my own image (no ground truth)":
    uploaded_file = st.file_uploader(
        "Upload a satellite image (JPG, JPEG, PNG, TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        ground_truth_mask = None  # arbitrary upload → no known ground truth

else:
    samples_df = load_sample_patches()
    if samples_df.empty:
        st.warning("Sample test set (test_patches.csv) not found — switch to upload mode.")
    else:
        selected_patch = st.selectbox("Select a sample patch", samples_df["patch_key"].tolist())
        original_image, ground_truth_mask = load_sample_image_and_mask(selected_patch)

# ---- 2. Display original image ----
if original_image is not None:
    st.subheader("Original Image")
    st.image(original_image, caption="Selected satellite image", use_container_width=True)
    if ground_truth_mask is not None:
        st.caption("✅ Ground truth available — metrics will be shown after detection.")
    else:
        st.caption("ℹ️ No ground truth for this image — prediction & highlighting shown, metrics unavailable.")
else:
    st.info("Please upload or select a satellite image to continue.")

st.divider()

# ---- 3. Run Detection ----
st.header("2. Run Cloud Detection")
run_clicked = st.button("🚀 Run Detection", disabled=(original_image is None))

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if run_clicked and original_image is not None:
    with st.spinner("Running preprocessing and detection..."):
        processed_image = preprocess_image(original_image)
        predicted_mask = predict_mask(processed_image)
        highlighted_image = highlight_clouds(original_image, predicted_mask)
        metrics = compute_metrics(predicted_mask, ground_truth_mask)  # None if no GT

        st.session_state.processed_image = processed_image
        st.session_state.predicted_mask = predicted_mask
        st.session_state.highlighted_image = highlighted_image
        st.session_state.metrics = metrics
        st.session_state.results_ready = True

st.divider()

# ---- 4. Results ----
st.header("3. Results")

if st.session_state.results_ready:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Preprocessed Image")
        st.image(st.session_state.processed_image, use_container_width=True)
        st.caption("MOCK — real preprocessed output once pipeline is connected.")

    with col2:
        st.subheader("Predicted Cloud Mask")
        st.image(st.session_state.predicted_mask, use_container_width=True, clamp=True)
        st.caption("MOCK — real U-Net predicted mask.")

    with col3:
        st.subheader("Cloud Regions Highlighted")
        st.image(st.session_state.highlighted_image, use_container_width=True)
        st.caption("MOCK — real highlighted detection output.")

    st.divider()

    # ---- 5. Metrics (only if ground truth exists) ----
    st.header("4. Evaluation Metrics")
    m = st.session_state.metrics
    if m is None:
        st.info("📊 Metrics unavailable — no ground truth for this image. Select a labeled sample image to see IoU / Dice / Precision / Recall.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IoU", f"{m['IoU']:.2f}")
        c2.metric("Dice", f"{m['Dice']:.2f}")
        c3.metric("Precision", f"{m['Precision']:.2f}")
        c4.metric("Recall", f"{m['Recall']:.2f}")
        st.caption("MOCK — real values once model + evaluation are connected.")
else:
    st.info("Results will appear here after you choose an image and click **Run Detection**.")

st.divider()
st.caption("Interface & Integration component — Samra Syed | AI-Based Cloud Detection and Localization in Satellite Images")
