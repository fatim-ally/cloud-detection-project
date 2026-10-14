"""
threshold_localization_keras.py
-------------------------------
Keras/TensorFlow version of Meerab's threshold-sweep + cloud-localization code.

Why this exists: Meerab's original script was written for PyTorch (torch,
DataLoader, logits). Our U-Net is Keras with a sigmoid output, so it already
outputs probabilities. The logic below (coarse+fine sweep, frozen threshold,
coverage table, sample visualisations) is hers; only the model/data handling
and the sweep-result parsing were adapted.

What it does (VALIDATION data only):
  1. Runs the model once over val_patches.csv -> probability maps.
  2. Sweeps thresholds with segmentation_metrics.threshold_sweep (macro IoU).
  3. Freezes the best threshold to frozen_threshold.json.
  4. Per-image cloud coverage (predicted vs true %) + scatter plot.
  5. Saves sample figures: random patches and the worst-IoU patches.

Usage (Colab, from the folder with all the .py files):
    python threshold_localization_keras.py \
        --split_dir /content/drive/MyDrive/cloud-detection-project/dataset_split \
        --stats_json /content/drive/MyDrive/cloud-detection-project/dataset_split/stats/band_stats_full_train.json \
        --checkpoint /content/drive/MyDrive/cloud-detection-project/dataset_split/stats/checkpoints_full/full_best.keras \
        --out_dir /content/drive/MyDrive/cloud-detection-project/dataset_split/stats/threshold_results
"""

import os
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files; no display needed
import matplotlib.pyplot as plt
import tensorflow as tf

import segmentation_metrics as sm
from data_pipeline_cloud_segmentation import build_dataset
from train_baseline import load_band_stats

METRICS = ("iou", "dice", "precision", "recall")
RGB_CHANNELS = (0, 1, 2)   # R, G, B of our (H, W, 4) input
N_SAMPLES = 6
SEED = 42


# ----------------------------------------------------------------------------
# 1. PREDICTIONS
# ----------------------------------------------------------------------------
def collect_predictions(model, val_ds, max_patches=None):
    """Run the model once. Returns rgb (N,H,W,3 float16, display only),
    gt (N,H,W uint8), probs (N,H,W float32)."""
    rgbs, gts, probs = [], [], []
    n = 0
    for x, y in val_ds:
        p = model.predict(x, verbose=0)[..., 0]
        rgbs.append(x.numpy()[..., list(RGB_CHANNELS)].astype(np.float16))
        gts.append((y.numpy()[..., 0] > 0.5).astype(np.uint8))
        probs.append(p.astype(np.float32))
        n += len(p)
        if n % 400 < len(p):
            print(f"  predicted {n} patches...")
        if max_patches and n >= max_patches:
            break
    rgb, gt, pr = map(np.concatenate, (rgbs, gts, probs))
    if max_patches:
        rgb, gt, pr = rgb[:max_patches], gt[:max_patches], pr[:max_patches]

    assert set(np.unique(gt)) <= {0, 1}, "Ground truth is not binary."
    assert 0.0 <= pr.min() and pr.max() <= 1.0, "Probabilities outside [0, 1]."
    return rgb, gt, pr


# ----------------------------------------------------------------------------
# 2. THRESHOLD SWEEP (validation only)
# ----------------------------------------------------------------------------
def _sweep(gt, probs, thresholds):
    """segmentation_metrics.threshold_sweep returns (best_t, {t: metrics_dict}).
    Turn that into a tidy DataFrame sorted by threshold."""
    _, table = sm.threshold_sweep(gt, probs, thresholds=list(thresholds))
    rows = []
    for t, m in table.items():
        row = {"threshold": float(t)}
        row.update({k: float(m[k]) for k in METRICS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def find_best_threshold(gt, probs):
    """Coarse sweep, then a fine sweep around the coarse optimum."""
    coarse = _sweep(gt, probs, np.round(np.arange(0.05, 0.96, 0.05), 3))
    centre = coarse.loc[coarse["iou"].idxmax(), "threshold"]
    fine_t = np.round(np.arange(centre - 0.05, centre + 0.0501, 0.01), 3).clip(0.01, 0.99)
    fine = _sweep(gt, probs, np.unique(fine_t))
    sweep = (pd.concat([coarse, fine]).drop_duplicates("threshold")
               .sort_values("threshold").reset_index(drop=True))
    return sweep, sweep.loc[sweep["iou"].idxmax()]


def plot_sweep(sweep, best_threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in METRICS:
        ax.plot(sweep["threshold"], sweep[m], label=m.capitalize())
    ax.axvline(best_threshold, ls="--", c="k", lw=1, label=f"best IoU @ {best_threshold:.2f}")
    ax.set(xlabel="Threshold", ylabel="Score", title="Validation threshold sweep")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def freeze_threshold(best_row, row_at_0p5, n_images, checkpoint, path):
    record = {
        "frozen_threshold": float(best_row["threshold"]),
        "selected_on": "validation set ONLY (metric: macro per-patch IoU)",
        "checkpoint": str(checkpoint),
        "validation_scores_at_threshold": {m: float(best_row[m]) for m in METRICS},
        "validation_scores_at_0.5": {m: float(row_at_0p5[m]) for m in METRICS},
        "n_validation_images": int(n_images),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "rule": "Reuse unchanged on the test set / in the app. Do NOT re-sweep on test data.",
    }
    Path(path).write_text(json.dumps(record, indent=2))
    return record


# ----------------------------------------------------------------------------
# 3. CLOUD COVERAGE
# ----------------------------------------------------------------------------
def binarize(probs, threshold):
    return (probs >= threshold).astype(np.uint8)   # >= matches segmentation_metrics


def coverage_table(gt, probs, threshold):
    pred = binarize(probs, threshold)
    flat = lambda a: a.reshape(len(a), -1)
    inter = flat(pred & gt).sum(1)
    union = flat(pred | gt).sum(1)
    pred_pct, gt_pct = flat(pred).mean(1) * 100, flat(gt).mean(1) * 100
    return pd.DataFrame({
        "image_id": np.arange(len(gt)),
        "pred_cloud_pct": pred_pct,
        "gt_cloud_pct": gt_pct,
        "abs_error_pct_pts": np.abs(pred_pct - gt_pct),
        "image_iou": np.where(union > 0, inter / np.maximum(union, 1), 1.0),
    })


def plot_coverage_scatter(table, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(table["gt_cloud_pct"], table["pred_cloud_pct"], s=8, alpha=0.5)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(xlabel="Ground-truth cloud (%)", ylabel="Predicted cloud (%)",
           title="Per-image coverage (validation)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# 4. VISUALIZATION
# ----------------------------------------------------------------------------
def to_rgb(img_hw3):
    """(H,W,3) -> (H,W,3) in [0,1], 2-98 percentile stretch. Display only."""
    rgb = img_hw3.astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay_mask(rgb, mask, color=(1.0, 0.0, 0.0), alpha=0.45):
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out


def visualize_samples(rgb_all, gt, probs, threshold, indices, path, title):
    """One row per sample: original | ground truth | prediction | overlay."""
    pred = binarize(probs, threshold)
    fig, axes = plt.subplots(len(indices), 4, figsize=(14, 3.3 * len(indices)), squeeze=False)
    for row, i in zip(axes, indices):
        rgb = to_rgb(rgb_all[i])
        union = (pred[i] | gt[i]).sum()
        iou = (pred[i] & gt[i]).sum() / union if union else 1.0
        panels = [
            (rgb, f"[{i}] Original"),
            (gt[i], f"Ground truth ({gt[i].mean() * 100:.1f}%)"),
            (pred[i], f"Predicted ({pred[i].mean() * 100:.1f}%)"),
            (overlay_mask(rgb, pred[i]), f"Overlay (IoU {iou:.3f})"),
        ]
        for ax, (img, label) in zip(row, panels):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(label, fontsize=9); ax.axis("off")
    fig.suptitle(f"{title} | threshold = {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Threshold sweep + localization (Keras).")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--stats_json", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="threshold_results")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_patches", type=int, default=None,
                    help="Optional: only use the first N val patches (quick test)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    means, stds = load_band_stats(args.stats_json)
    val_csv = os.path.join(args.split_dir, "val_patches.csv")
    val_ds = build_dataset(val_csv, means, stds, has_mask=True,
                           batch_size=args.batch_size, shuffle=False)

    print(f"\nLoading model: {args.checkpoint}")
    model = tf.keras.models.load_model(args.checkpoint, compile=False)

    print("Collecting validation predictions...")
    rgb, gt, probs = collect_predictions(model, val_ds, args.max_patches)
    print(f"Done: {len(gt)} patches, probs shape {probs.shape}")

    print("\nSweeping thresholds (validation only)...")
    sweep, best = find_best_threshold(gt, probs)
    row_0p5 = sweep.loc[np.isclose(sweep["threshold"], 0.5)].iloc[0]
    threshold = float(best["threshold"])

    sweep.to_csv(out / "validation_threshold_sweep.csv", index=False)
    plot_sweep(sweep, threshold, out / "validation_threshold_sweep.png")
    freeze_threshold(best, row_0p5, len(gt), args.checkpoint, out / "frozen_threshold.json")

    print(sweep.round(4).to_string(index=False))
    print(f"\nFROZEN THRESHOLD: {threshold:.2f}  | val IoU {best['iou']:.4f} "
          f"(at 0.5: {row_0p5['iou']:.4f})")

    coverage = coverage_table(gt, probs, threshold)
    coverage.to_csv(out / "validation_cloud_coverage.csv", index=False)
    plot_coverage_scatter(coverage, out / "coverage_scatter.png")
    print("\nCoverage summary:")
    print(coverage.describe().round(3))

    n = min(N_SAMPLES, len(gt))
    random_idx = np.random.default_rng(SEED).choice(len(gt), size=n, replace=False)
    worst_idx = coverage.nsmallest(n, "image_iou").index.to_numpy()
    visualize_samples(rgb, gt, probs, threshold, random_idx,
                      out / "samples_random.png", "Random validation samples")
    visualize_samples(rgb, gt, probs, threshold, worst_idx,
                      out / "samples_worst_iou.png", "Lowest-IoU validation samples")

    print(f"\nAll outputs saved in: {out.resolve()}")
    print("Files: validation_threshold_sweep.csv/.png, frozen_threshold.json, "
          "validation_cloud_coverage.csv, coverage_scatter.png, "
          "samples_random.png, samples_worst_iou.png")


if __name__ == "__main__":
    main()
