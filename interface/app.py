"""
app.py
------
AI-Based Cloud Detection and Localization in Satellite Images
Interface & Integration — Streamlit app

STATUS: Fully integrated with real model (full_best.keras), dataset stats,
validation patches, and ground-truth mask evaluation.

Author: Samra Syed (Integrations by Fatima)
"""

import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from pathlib import Path


from PIL import Image

from segmentation_metrics import compute_all_metrics  # must sit next to app.py

# =====================================================================
# CONFIG — Production settings
# =====================================================================
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR.parent / "model" / "checkpoints_full" / "full_best.keras"
BAND_STATS_PATH = BASE_DIR.parent / "dataset_split" / "stats" / "band_stats_full_train.json"
INPUT_SIZE = 384            # model expects (384, 384, 4)
NUM_BANDS = 4               # R, G, B, NIR (same band order as training)

# Frozen threshold from validation sweep (default set to 0.50 until exact sweep is locked in)
FROZEN_THRESHOLD = 0.89

# Stats were computed on raw uint16 satellite patch values
STATS_PIXEL_RANGE = 65535.0


def _get_threshold() -> float:
    if FROZEN_THRESHOLD is None:
        raise ValueError(
            "FROZEN_THRESHOLD is not set in app.py. Add the frozen value "
            "from the validation sweep before running detection."
        )
    return float(FROZEN_THRESHOLD)


# =====================================================================
# SECTION A — PREPROCESSING
# =====================================================================
def preprocess_image(image):
    """Pass-through image handler for the pipeline UI."""
    return image


# =====================================================================
# SECTION B — PREDICTION (real U-Net)
# =====================================================================
@st.cache_resource
def load_model():
    """Load the U-Net model once and reuse it across Streamlit reruns."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"U-Net checkpoint not found at: {MODEL_PATH}"
        )

    import tensorflow as tf

    with st.spinner("Loading U-Net model into memory (one-time setup)..."):
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )

    return model
 

#@st.cache_resource
def load_band_stats():
    """
    Loads per-band mean and std from band_stats_full_train.json or npz.
    Handles nested band keys: {"bands": {"red": {"mean":..., "std":...}, ...}}
    """
    if not BAND_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Band statistics file not found at: {BAND_STATS_PATH}\n"
            "Place band_stats_full_train.json inside interface/models/."
        )

    if BAND_STATS_PATH.suffix.lower() == ".json":
        with open(BAND_STATS_PATH, "r") as f:
            data = json.load(f)

        # Support nested structure from band_stats_full_train.json
        if "bands" in data:
            bands_order = ["red", "green", "blue", "nir"]
            mean = [data["bands"][b]["mean"] for b in bands_order]
            std = [data["bands"][b]["std"] for b in bands_order]
        else:
            mean = data["mean"]
            std = data["std"]

        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
    else:
        data = np.load(BAND_STATS_PATH)
        mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(data["std"], dtype=np.float32).reshape(-1)

    if mean.shape != (NUM_BANDS,) or std.shape != (NUM_BANDS,):
        raise ValueError(
            f"Band stats must have {NUM_BANDS} values each; "
            f"got mean{mean.shape}, std{std.shape}."
        )
    return mean, std


#MODEL = load_model()



def _to_model_input(image) -> np.ndarray:
    """Converts input image/patch into normalized (384, 384, 4) tensor."""
    mean, std = load_band_stats()

    if isinstance(image, Image.Image):
        rgb = (
            np.asarray(image.convert("RGB"), dtype=np.float32)
            / 255.0 * STATS_PIXEL_RANGE
        )
        arr = rgb
    else:
        arr = np.asarray(image, dtype=np.float32)

    if arr.shape[-1] == 3:
        nir = np.full(
            arr.shape[:2] + (1,),
            mean[3],
            dtype=np.float32
        )
        arr = np.concatenate([arr, nir], axis=-1)

    # Resize while preserving all four bands
    y_indices = np.linspace(
        0, arr.shape[0] - 1, INPUT_SIZE
    ).astype(int)
    x_indices = np.linspace(
        0, arr.shape[1] - 1, INPUT_SIZE
    ).astype(int)

    img_resized = arr[y_indices[:, None], x_indices[None, :], :]
    img_resized = img_resized.astype(np.float32)

    return (
        (img_resized - mean) / np.maximum(std, 1e-7)
    ).astype(np.float32)




def predict_mask(processed_image):
    """Runs U-Net prediction using the cached model instance."""
    model = load_model()  # Retrieves instantly from memory!
    threshold = _get_threshold()
    x = _to_model_input(processed_image)

    probs = model.predict(x[np.newaxis, ...], verbose=0)[0]
    probs = np.squeeze(probs)

    if probs.shape != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError(
            f"Unexpected model output shape {probs.shape}; "
            f"expected ({INPUT_SIZE}, {INPUT_SIZE}) after squeezing."
        )

    binary = probs > threshold
    return (binary.astype(np.uint8) * 255)


def compute_metrics(predicted_mask, ground_truth_mask):
    """Calculates IoU, Dice, Precision, and Recall when ground truth exists."""
    if ground_truth_mask is None:
        return None

    y_true = (np.asarray(ground_truth_mask) > 0).astype(np.uint8)
    y_pred = (np.asarray(predicted_mask) > 0).astype(np.uint8)

    if y_true.shape != y_pred.shape:
        # Auto-resize ground truth if patch dimensions mismatch model output
        y_true = np.array(
            Image.fromarray(y_true).resize((INPUT_SIZE, INPUT_SIZE), resample=Image.NEAREST)
        )

    raw = compute_all_metrics(y_true, y_pred, threshold=_get_threshold())

    lowered = {str(k).lower(): v for k, v in dict(raw).items()}
    try:
        return {
            "IoU": float(lowered["iou"]),
            "Dice": float(lowered["dice"]),
            "Precision": float(lowered["precision"]),
            "Recall": float(lowered["recall"]),
        }
    except KeyError as e:
        raise KeyError(
            f"compute_all_metrics did not return key {e}. Keys: {list(lowered.keys())}"
        )


# =====================================================================
# SECTION C — VISUALIZATION
# =====================================================================
def highlight_clouds(original_image, mask: np.ndarray, color=(255, 0, 0), alpha=0.4):
    """Blends a semi-transparent red overlay on predicted cloud regions."""
    if isinstance(original_image, np.ndarray):
        # Scale 4-band array to 8-bit RGB for visual preview
        rgb = original_image[..., :3]
        if rgb.max() > 255.0:
            rgb = (rgb / STATS_PIXEL_RANGE * 255.0).clip(0, 255)
        img_pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    else:
        img_pil = original_image.convert("RGB")

    original_rgba = img_pil.convert("RGBA").resize((mask.shape[1], mask.shape[0]))
    overlay = Image.new("RGBA", original_rgba.size, color + (0,))
    overlay_pixels = overlay.load()

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] > 0:
                overlay_pixels[x, y] = color + (int(255 * alpha),)

    highlighted = Image.alpha_composite(original_rgba, overlay)
    return highlighted.convert("RGB")


# =====================================================================
# SECTION D — LABELED SAMPLE DATASET (Validation Split)
# =====================================================================
# Look for val_patches.csv in root or relative paths
SAMPLES_DIR = BASE_DIR / "samples"


def load_sample_patches():
    """Finds all pre-extracted .npz sample patch files in interface/samples/."""
    if not SAMPLES_DIR.exists():
        return []
    return sorted([f.name for f in SAMPLES_DIR.glob("*.npz")])


def load_sample_image_and_mask(npz_filename):
    """Loads 4-band satellite patch array and ground truth mask directly from .npz file."""
    file_path = SAMPLES_DIR / npz_filename
    try:
        data = np.load(file_path)
        four_band_image = data["image"]  # shape (384, 384, 4)
        ground_truth_mask = data["gt"]  # shape (384, 384)
        return four_band_image, ground_truth_mask
    except Exception as e:
        st.error(f"Error loading sample .npz patch ({npz_filename}): {e}")
        return np.zeros((INPUT_SIZE, INPUT_SIZE, NUM_BANDS), dtype=np.float32), np.zeros(
            (INPUT_SIZE, INPUT_SIZE), dtype=np.uint8
        )

# =====================================================================
# SECTION E — STREAMLIT UI
# =====================================================================
st.set_page_config(page_title="Cloud Detection & Localization", layout="wide")

st.title("☁️ AI-Based Cloud Detection and Localization in Satellite Images")
st.markdown(
    """
    Upload a satellite image, or select a labeled validation patch, to view
    the model's cloud detection and localization results.

    **Pipeline:** Choose Image → Preprocessing → AI Detection → Results → Cloud Highlighting → Metrics (if ground truth available)
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
        ground_truth_mask = None

else:
    sample_files = load_sample_patches()
    if not sample_files:
        st.warning(
            "No sample patches found in interface/samples/ — switch to upload mode."
        )
    else:
        selected_file = st.selectbox("Select a validation sample patch", sample_files)
        original_image, ground_truth_mask = load_sample_image_and_mask(selected_file)

# ---- 2. Display original image ----
if original_image is not None:
    st.subheader("Original Image / Patch Preview")
    if isinstance(original_image, np.ndarray):
        rgb_preview = (original_image[..., :3] / STATS_PIXEL_RANGE * 255.0).clip(0, 255).astype(np.uint8)
        st.image(rgb_preview, caption=f"Selected satellite patch ({selected_file})", width="stretch")
    else:
        st.image(original_image, caption="Uploaded satellite image", width="stretch")

    if ground_truth_mask is not None:
        st.caption("✅ Ground truth available — evaluation metrics will be computed.")
    else:
        st.caption("ℹ️ Arbitrary upload — prediction & highlighting active, metrics unavailable.")
else:
    st.info("Please upload or select a satellite image to continue.")

st.divider()

# ---- 3. Run Detection ----
st.header("2. Run Cloud Detection")
run_clicked = st.button("🚀 Run Detection", disabled=(original_image is None))

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if run_clicked and original_image is not None:
    with st.spinner("Running inference and mask post-processing..."):
        processed_image = preprocess_image(original_image)
        predicted_mask = predict_mask(processed_image)
        highlighted_image = highlight_clouds(original_image, predicted_mask)
        metrics = compute_metrics(predicted_mask, ground_truth_mask)

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
        st.subheader("Preprocessed Input")
        if isinstance(st.session_state.processed_image, np.ndarray):
            rgb_preview = (st.session_state.processed_image[..., :3] / STATS_PIXEL_RANGE * 255.0).clip(0, 255).astype(np.uint8)
            st.image(rgb_preview, width="stretch")
        else:
            st.image(st.session_state.processed_image, width="stretch")

    with col2:
        st.subheader("Predicted Cloud Mask")
        st.image(st.session_state.predicted_mask, width="stretch", clamp=True)

    with col3:
        st.subheader("Cloud Regions Highlighted")
        st.image(st.session_state.highlighted_image, width="stretch")

    st.divider()

    # ---- 5. Metrics ----
    st.header("4. Evaluation Metrics")
    m = st.session_state.metrics
    if m is None:
        st.info("📊 Metrics unavailable — no ground truth for this image. Select a validation sample patch to see IoU / Dice / Precision / Recall.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IoU", f"{m['IoU']:.4f}")
        c2.metric("Dice", f"{m['Dice']:.4f}")
        c3.metric("Precision", f"{m['Precision']:.4f}")
        c4.metric("Recall", f"{m['Recall']:.4f}")
else:
    st.info("Results will appear here after you choose an image and click **Run Detection**.")

st.divider()
st.caption("Interface & Integration component — Samra Syed and Fatima Fahad | AI-Based Cloud Detection and Localization in Satellite Images")