"""
Compute per-band (Red, Green, Blue, NIR) normalization statistics — mean and
standard deviation — across the FULL 38-Cloud training set listed in
train_patches.csv.

This does NOT train a model and does NOT estimate or invent any numbers.
Every value is computed by actually reading every referenced band file and
accumulating exact running sums (streaming, not sampled, not loaded fully
into memory at once).

Usage (as a script):
    python compute_full_band_stats.py

Or paste the CONFIG block + the rest directly into a Colab / Jupyter cell —
it runs top to bottom with no command-line arguments required.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

# =============================================================================
# CONFIG — adjust these two paths for your environment, then run.
# =============================================================================
TRAIN_CSV_PATH = "dataset_split/train_patches.csv"          # from your data-splitting notebook
OUTPUT_JSON_PATH = "dataset_split/stats/band_stats_full_train.json"

# Bands to compute stats for, and the CSV column that holds each band's file path.
BAND_COLUMNS = {
    "red":   "red_path",
    "green": "green_path",
    "blue":  "blue_path",
    "nir":   "nir_path",
}

PROGRESS_EVERY = 500  # print a progress line every N patches
# =============================================================================


def load_band_array(path):
    """Load one single-band image file as a float64 numpy array."""
    arr = np.array(Image.open(path)).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-band (2-D) image, got shape {arr.shape} for: {path}")
    return arr


def verify_all_files_exist(df, band_columns):
    """Fail loudly and completely, listing every missing file, rather than
    silently computing statistics from a partial/incorrect set of patches."""
    missing = []
    for col in band_columns.values():
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' not found in the CSV. "
                f"Columns present: {list(df.columns)}"
            )
        missing_mask = ~df[col].apply(os.path.isfile)
        if missing_mask.any():
            missing.extend(df.loc[missing_mask, col].tolist())
    if missing:
        preview = "\n  ".join(missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} band file(s) referenced in {TRAIN_CSV_PATH} do not exist on disk:\n"
            f"  {preview}{more}\n"
            f"Fix the paths (or re-download/re-mount the dataset) before computing statistics — "
            f"do not proceed with a partial file set."
        )


def compute_streaming_band_stats(df, band_columns, progress_every=PROGRESS_EVERY):
    """
    Streams through every patch exactly once. For each band, maintains only
    three running scalars (pixel count, sum, sum of squares) — never holds
    more than one image in memory at a time, regardless of dataset size.

    Returns a dict: {band_name: {"mean": float, "std": float, "n_pixels": int}}
    """
    running = {band: {"n": 0, "sum": 0.0, "sumsq": 0.0} for band in band_columns}

    n_patches = len(df)
    start_time = time.time()

    for i, row in enumerate(df.itertuples(index=False), start=1):
        for band, col in band_columns.items():
            path = getattr(row, col)
            arr = load_band_array(path)

            if not np.isfinite(arr).all():
                raise ValueError(
                    f"Non-finite pixel value(s) (NaN/inf) found in band '{band}' file: {path}. "
                    f"Investigate this file before trusting any statistics computed from it."
                )

            r = running[band]
            r["n"] += arr.size
            r["sum"] += arr.sum()
            r["sumsq"] += np.square(arr).sum()

        if progress_every and (i % progress_every == 0 or i == n_patches):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else float("inf")
            eta = (n_patches - i) / rate if rate > 0 else float("inf")
            print(f"  processed {i}/{n_patches} patches "
                  f"({elapsed:6.1f}s elapsed, ~{eta:6.1f}s remaining)")

    results = {}
    for band, r in running.items():
        n, s, sq = r["n"], r["sum"], r["sumsq"]
        mean = s / n
        # Population variance (ddof=0) — standard convention for normalization stats.
        variance = (sq / n) - (mean ** 2)
        variance = max(variance, 0.0)  # guards against a tiny negative from floating-point error
        std = float(np.sqrt(variance))
        results[band] = {"mean": float(mean), "std": std, "n_pixels": int(n)}

    return results


def main():
    if not os.path.isfile(TRAIN_CSV_PATH):
        raise FileNotFoundError(
            f"Could not find '{TRAIN_CSV_PATH}'. Set TRAIN_CSV_PATH at the top of this "
            f"script to the actual path of your data-splitting notebook's train_patches.csv."
        )

    df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {TRAIN_CSV_PATH}: {len(df)} training patches listed.")

    print("Verifying every referenced band file exists on disk before computing anything...")
    verify_all_files_exist(df, BAND_COLUMNS)
    print("All referenced band files found. Proceeding with the full computation.\n")

    print(f"Computing streaming per-band statistics over all {len(df)} patches "
          f"(this reads every file exactly once; no sampling)...")
    stats = compute_streaming_band_stats(df, BAND_COLUMNS)

    print("\n" + "=" * 60)
    print("FULL TRAINING SET — PER-BAND NORMALIZATION STATISTICS")
    print("=" * 60)
    for band in BAND_COLUMNS:
        s = stats[band]
        print(f"  {band.upper():6s}  mean = {s['mean']:12.4f}   std = {s['std']:12.4f}   "
              f"(n_pixels = {s['n_pixels']:,})")
    print("=" * 60)

    output = {
        "dataset": "38-Cloud",
        "source_csv": os.path.abspath(TRAIN_CSV_PATH),
        "computed_from": "FULL training set (no sampling)",
        "n_patches": int(len(df)),
        "bands": stats,
        "method": "streaming sum / sum-of-squares, float64 accumulation, population std (ddof=0)",
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version,
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {os.path.abspath(OUTPUT_JSON_PATH)}")


if __name__ == "__main__":
    main()
