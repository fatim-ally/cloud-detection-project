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
import tensorflow as tf
from PIL import Image

from segmentation_metrics import compute_all_metrics  # must sit next to app.py

# =====================================================================
# CONFIG — Production settings
# =====================================================================
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "full_best.keras"
BAND_STATS_PATH = BASE_DIR / "models" / "band_stats_full_train.json"

INPUT_SIZE = 384            # model expects (384, 384, 4)
NUM_BANDS = 4               # R, G, B, NIR (same band order as training)

# Frozen threshold from validation sweep (default set to 0.50 until exact sweep is locked in)
FROZEN_THRESHOLD = 0.50

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
    """Loads the trained U-Net checkpoint once and caches it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"U-Net checkpoint not found at: {MODEL_PATH}\n"
            "Place full_best.keras inside interface/models/."
        )
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


@st.cache_resource
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


MODEL = load_model()


def _to_model_input(image) -> np.ndarray:
    """Converts input image/patch into normalized (384, 384, 4) tensor."""
    mean, std = load_band_stats()

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 * STATS_PIXEL_RANGE
        arr = rgb
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected an (H, W, C) image, got shape {arr.shape}.")

    if arr.shape[-1] == 3:
        nir = np.full(arr.shape[:2] + (1,), mean[3], dtype=np.float32)
        arr = np.concatenate([arr, nir], axis=-1)
    elif arr.shape[-1] != NUM_BANDS:
        raise ValueError(f"Expected 3 or {NUM_BANDS} channels, got {arr.shape[-1]}.")

    # Resize to input dimensions and perform band standardization
    arr = tf.image.resize(arr, (INPUT_SIZE, INPUT_SIZE), method="bilinear").numpy()
    arr = (arr - mean) / np.maximum(std, 1e-7)
    return arr.astype(np.float32)


def predict_mask(processed_image):
    """Runs U-Net prediction and applies probability thresholding."""
    threshold = _get_threshold()
    x = _to_model_input(processed_image)

    probs = MODEL.predict(x[np.newaxis, ...], verbose=0)[0]
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
SAMPLE_CSV_PATH = BASE_DIR.parent / "dataset_split" / "val_patches.csv"
if not SAMPLE_CSV_PATH.exists():
    SAMPLE_CSV_PATH = Path("val_patches.csv")


@st.cache_data
def load_sample_patches():
    """Loads validation patch index containing ground-truth references."""
    try:
        return pd.read_csv(SAMPLE_CSV_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["patch_key", "scene_id", "red_path", "green_path", "blue_path", "nir_path", "gt_path"])


def load_sample_image_and_mask(patch_row):
    """Loads actual 4-band patch images and ground truth cloud masks."""
    try:
        red = np.array(Image.open(patch_row["red_path"]), dtype=np.float32)
        green = np.array(Image.open(patch_row["green_path"]), dtype=np.float32)
        blue = np.array(Image.open(patch_row["blue_path"]), dtype=np.float32)
        nir = np.array(Image.open(patch_row["nir_path"]), dtype=np.float32)

        four_band_image = np.stack([red, green, blue, nir], axis=-1)
        ground_truth_mask = np.array(Image.open(patch_row["gt_path"]), dtype=np.uint8)

        return four_band_image, ground_truth_mask
    except Exception as e:
        st.error(f"Error loading patch files: {e}")
        # Fallback empty arrays if file paths on disk are unreachable
        return np.zeros((384, 384, 4), dtype=np.float32), np.zeros((384, 384), dtype=np.uint8)


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
    samples_df = load_sample_patches()
    if samples_df.empty:
        st.warning("Validation dataset index (val_patches.csv) not found — switch to upload mode.")
    else:
        patch_keys = samples_df["patch_key"].tolist()
        selected_key = st.selectbox("Select a validation sample patch", patch_keys)
        selected_row = samples_df[samples_df["patch_key"] == selected_key].iloc[0]

        original_image, ground_truth_mask = load_sample_image_and_mask(selected_row)

# ---- 2. Display original image ----
if original_image is not None:
    st.subheader("Original Image / Patch Preview")
    if isinstance(original_image, np.ndarray):
        rgb_preview = (original_image[..., :3] / STATS_PIXEL_RANGE * 255.0).clip(0, 255).astype(np.uint8)
        st.image(rgb_preview, caption=f"Selected satellite patch ({selected_key})", use_container_width=True)
    else:
        st.image(original_image, caption="Uploaded satellite image", use_container_width=True)

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
            st.image(rgb_preview, use_container_width=True)
        else:
            st.image(st.session_state.processed_image, use_container_width=True)

    with col2:
        st.subheader("Predicted Cloud Mask")
        st.image(st.session_state.predicted_mask, use_container_width=True, clamp=True)

    with col3:
        st.subheader("Cloud Regions Highlighted")
        st.image(st.session_state.highlighted_image, use_container_width=True)

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
st.caption("Interface & Integration component — Samra Syed | AI-Based Cloud Detection and Localization in Satellite Images")