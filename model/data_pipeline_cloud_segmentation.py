"""
Preprocessing / Data-Loading Pipeline — Cloud Segmentation
============================================================
Project: AI-Based Cloud Detection and Localization in Satellite Images
Component: AI Model Development (bridges Ayesha's dataset split -> the U-Net model)

Input:  train_patches.csv / val_patches.csv / test_patches.csv
        (columns: patch_key, scene_id, red_path, green_path, blue_path,
         nir_path, gt_path — test_patches.csv has no gt_path)

Output: tf.data.Dataset objects yielding (image, mask) batches shaped
        (batch, 384, 384, 4) and (batch, 384, 384, 1), ready to feed
        straight into build_unet() from unet_cloud_segmentation_skeleton.py

Decisions made here (see chat for reasoning):
  - Normalization: per-band mean/std, computed once from the TRAIN split only
  - Near-all-black / low-info patches: kept, not filtered (can revisit later)
  - Augmentation: NONE for now (baseline first, augment later if needed)

STATUS: pipeline + normalization stats only. No model.fit() call here.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

PATCH_SIZE = 384          # matches 38-Cloud's fixed patch size
BANDS = ["red", "green", "blue", "nir"]   # channel order used everywhere below
BATCH_SIZE = 8             # conservative default for Colab's free-tier GPU memory
SHUFFLE_BUFFER = 500


# ---------------------------------------------------------------------------
# 2. LOW-LEVEL FILE LOADING (numpy / PIL — used inside tf.py_function)
# ---------------------------------------------------------------------------

def _load_tif(path):
    """Load a single-band 38-Cloud TIFF file as a float32 numpy array.
    38-Cloud band TIFFs are typically 16-bit; we keep native values here and
    normalize later with the stats computed in section 3 — do NOT rescale to
    0-255 here, that would throw away dynamic range."""
    arr = np.array(Image.open(path)).astype(np.float32)
    return arr


def _load_patch(row_dict, has_mask):
    """Load one patch's 4 bands (+ mask if available) as numpy arrays.
    Returns:
        image: (H, W, 4) float32, band order = red, green, blue, nir
        mask:  (H, W, 1) float32 in {0., 1.}, or None if has_mask is False
    """
    bands = [_load_tif(row_dict[f"{b}_path"]) for b in BANDS]
    image = np.stack(bands, axis=-1)  # (H, W, 4)

    mask = None
    if has_mask:
        m = _load_tif(row_dict["gt_path"])
        # 38-Cloud masks are stored as 0/255 (or already 0/1 on some mirrors).
        # Normalize to a strict {0., 1.} regardless of which convention this
        # particular download used.
        mask = (m > 0).astype(np.float32)[..., np.newaxis]  # (H, W, 1)

    return image, mask


# ---------------------------------------------------------------------------
# 3. PER-BAND NORMALIZATION STATISTICS (computed once, from TRAIN split only)
# ---------------------------------------------------------------------------

def compute_band_stats(train_df, sample_size=300, seed=42):
    """
    Estimate per-band mean/std from a random sample of the TRAINING split
    only (never from val/test — that would leak information about those
    splits into how training data is scaled).

    A sample (not the full ~6,500+ train patches) is used purely for speed;
    300 patches is generally enough to get a stable per-band estimate for
    this kind of imagery. Increase sample_size if you want a more precise
    estimate and have time to spare.

    Returns:
        means: np.array shape (4,), stds: np.array shape (4,) — order = BANDS
    """
    sample_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=seed)

    sums = np.zeros(len(BANDS), dtype=np.float64)
    sums_sq = np.zeros(len(BANDS), dtype=np.float64)
    n_pixels = 0

    for _, row in sample_df.iterrows():
        image, _ = _load_patch(row.to_dict(), has_mask=False)
        n_pixels += image.shape[0] * image.shape[1]
        sums += image.sum(axis=(0, 1))
        sums_sq += (image.astype(np.float64) ** 2).sum(axis=(0, 1))

    means = sums / n_pixels
    variances = (sums_sq / n_pixels) - means ** 2
    stds = np.sqrt(np.maximum(variances, 1e-8))  # guard against tiny negative from float error

    return means.astype(np.float32), stds.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. TF.DATA PIPELINE
# ---------------------------------------------------------------------------

def _make_tf_loader(has_mask, means, stds):
    """
    Builds the function tf.data will call per-row. Wraps the numpy-based
    loader in tf.py_function since TIFF decoding isn't natively supported
    by tf.io for this format, then applies normalization and fixes shapes
    (tf.py_function erases static shape info, so we set it back explicitly).
    """
    means_t = tf.constant(means, dtype=tf.float32)
    stds_t = tf.constant(stds, dtype=tf.float32)

    def _py_load(*path_tensors):
        row_dict = {
            "red_path": path_tensors[0].numpy().decode("utf-8"),
            "green_path": path_tensors[1].numpy().decode("utf-8"),
            "blue_path": path_tensors[2].numpy().decode("utf-8"),
            "nir_path": path_tensors[3].numpy().decode("utf-8"),
        }
        if has_mask:
            row_dict["gt_path"] = path_tensors[4].numpy().decode("utf-8")
        image, mask = _load_patch(row_dict, has_mask=has_mask)
        if has_mask:
            return image, mask
        return image

    def loader(paths):
        if has_mask:
            image, mask = tf.py_function(
                func=_py_load,
                inp=[paths["red_path"], paths["green_path"], paths["blue_path"],
                     paths["nir_path"], paths["gt_path"]],
                Tout=[tf.float32, tf.float32],
            )
            mask.set_shape([PATCH_SIZE, PATCH_SIZE, 1])
        else:
            image = tf.py_function(
                func=_py_load,
                inp=[paths["red_path"], paths["green_path"], paths["blue_path"],
                     paths["nir_path"]],
                Tout=tf.float32,
            )
        image.set_shape([PATCH_SIZE, PATCH_SIZE, len(BANDS)])

        # Per-band standardization: (x - mean) / std
        image = (image - means_t) / stds_t

        if has_mask:
            return image, mask
        return image

    return loader


def build_dataset(csv_path, means, stds, has_mask=True,
                   batch_size=BATCH_SIZE, shuffle=False, cache=False):
    """
    Builds a tf.data.Dataset from one of the split CSVs.

    Args:
        csv_path: path to train_patches.csv / val_patches.csv / test_patches.csv
        means, stds: per-band normalization stats from compute_band_stats()
                     (always compute these from TRAIN, reuse the same values
                     for val/test — never recompute per-split)
        has_mask: True for train/val (have gt_path), False for test_patches.csv
                  unless you're doing a final labeled evaluation on a subset
        shuffle: True for training, False for val/test
        cache: set True once patches comfortably fit in memory/disk cache to
               speed up repeated epochs; leave False on first run to confirm
               correctness before optimizing speed

    Returns: a batched, prefetched tf.data.Dataset yielding
        (image, mask) tuples if has_mask else image tensors alone.
    """
    df = pd.read_csv(csv_path)

    needed_cols = ["red_path", "green_path", "blue_path", "nir_path"]
    if has_mask:
        needed_cols.append("gt_path")
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected column(s) {missing}. "
            f"has_mask={has_mask} — pass has_mask=False for test_patches.csv."
        )

    path_dict = {col: df[col].tolist() for col in needed_cols}
    ds = tf.data.Dataset.from_tensor_slices(path_dict)

    loader = _make_tf_loader(has_mask, means, stds)
    ds = ds.map(loader, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# 5. SANITY CHECK — build everything, load ONE batch, print shapes/stats only
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check the data pipeline.")
    parser.add_argument("--split_dir", required=True,
                         help="Folder containing train_patches.csv, val_patches.csv, test_patches.csv")
    args = parser.parse_args()

    train_csv = os.path.join(args.split_dir, "train_patches.csv")
    val_csv = os.path.join(args.split_dir, "val_patches.csv")
    test_csv = os.path.join(args.split_dir, "test_patches.csv")

    print("Computing per-band normalization stats from TRAIN split (sampled)...")
    train_df_full = pd.read_csv(train_csv)
    means, stds = compute_band_stats(train_df_full)
    print("Per-band means:", means)
    print("Per-band stds :", stds)

    print("\nBuilding datasets...")
    train_ds = build_dataset(train_csv, means, stds, has_mask=True, shuffle=True)
    val_ds = build_dataset(val_csv, means, stds, has_mask=True, shuffle=False)
    test_ds = build_dataset(test_csv, means, stds, has_mask=False, shuffle=False)

    print("\nPulling one batch from train_ds to confirm shapes...")
    images, masks = next(iter(train_ds))
    print("Image batch shape:", images.shape, "dtype:", images.dtype)
    print("Mask batch shape :", masks.shape, "dtype:", masks.dtype)
    print("Mask unique values in this batch:", np.unique(masks.numpy()))

    print("\nPipeline built successfully. No training performed — this script only "
          "verifies the loader works and produces correctly-shaped, normalized batches.")
