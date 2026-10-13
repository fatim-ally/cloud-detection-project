"""
Cloud Detection - Threshold Sweep (validation only) and Cloud Localization.

Usage (Colab): `model`, `val_loader` and segmentation_metrics.py must be available.
    1. Paste this file into a cell (or import it).
    2. Run the block under "RUN" at the bottom.

Assumptions (marked ADJUST):
    - model returns logits; val_loader yields (image, mask)
    - threshold_sweep(y_true, y_prob, thresholds) is the call shape
"""

import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from segmentation_metrics import threshold_sweep

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("outputs")
RGB_CHANNELS = (0, 1, 2)          # ADJUST: channels used for display
APPLY_SIGMOID = True             # ADJUST: False if model already outputs probabilities
N_SAMPLES = 6
SEED = 42
REPORTED_AT_0P5 = {"iou": 0.675, "dice": 0.731, "precision": 0.803, "recall": 0.750}
METRICS = ("iou", "dice", "precision", "recall")


# ----------------------------------------------------------------------------
# 1. PREDICTIONS
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader):
    """Run the model once; return images (N,C,H,W), gt (N,H,W), probs (N,H,W)."""
    model.eval().to(DEVICE)
    images, gts, probs = [], [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        p = torch.sigmoid(out) if APPLY_SIGMOID else out
        images.append(x.numpy().astype(np.float32))
        gts.append(y.squeeze(1).numpy().astype(np.uint8))
        probs.append(p.squeeze(1).cpu().numpy().astype(np.float32))
    images, gts, probs = map(np.concatenate, (images, gts, probs))

    assert set(np.unique(gts)) <= {0, 1}, "Ground truth is not binary."
    assert 0.0 <= probs.min() and probs.max() <= 1.0, "Probabilities outside [0, 1]."
    return images, gts, probs


# ----------------------------------------------------------------------------
# 2. THRESHOLD SWEEP (validation only)
# ----------------------------------------------------------------------------
def _sweep(gt, probs, thresholds):
    """Call threshold_sweep() and return a tidy DataFrame sorted by threshold."""
    res = threshold_sweep(gt, probs, thresholds=thresholds)   # ADJUST if needed
    df = res.copy() if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"thresholds": "threshold"})
    if "threshold" not in df:
        df.insert(0, "threshold", list(thresholds)[: len(df)])
    return df.sort_values("threshold").reset_index(drop=True)


def find_best_threshold(gt, probs):
    """Coarse sweep, then a fine sweep around the coarse optimum. Returns (sweep_df, best_row)."""
    coarse = _sweep(gt, probs, np.round(np.arange(0.05, 0.96, 0.05), 3))
    centre = coarse.loc[coarse["iou"].idxmax(), "threshold"]
    fine = _sweep(gt, probs, np.round(np.arange(centre - 0.05, centre + 0.0501, 0.01), 3).clip(0.01, 0.99))

    sweep = (pd.concat([coarse, fine]).drop_duplicates("threshold")
               .sort_values("threshold").reset_index(drop=True))
    return sweep, sweep.loc[sweep["iou"].idxmax()]


def check_reproduces_reported(sweep, tol=0.005):
    """The sweep at 0.5 must match the previously reported validation scores."""
    row = sweep.loc[np.isclose(sweep["threshold"], 0.5)].iloc[0]
    for metric, expected in REPORTED_AT_0P5.items():
        assert abs(row[metric] - expected) < tol, (
            f"{metric} at 0.5 = {row[metric]:.4f}, expected {expected}. "
            "Pipeline differs from the earlier run.")
    return row


def plot_sweep(sweep, best_threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in METRICS:
        ax.plot(sweep["threshold"], sweep[m], label=m.capitalize())
    ax.axvline(best_threshold, ls="--", c="k", lw=1, label=f"best IoU @ {best_threshold:.2f}")
    ax.set(xlabel="Threshold", ylabel="Score", title="Validation threshold sweep")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.show()


def freeze_threshold(best_row, row_at_0p5, n_images, path):
    """Write the chosen threshold and its provenance to JSON."""
    record = {
        "frozen_threshold": float(best_row["threshold"]),
        "selected_on": "validation set ONLY (metric: IoU)",
        "validation_scores_at_threshold": {m: float(best_row[m]) for m in METRICS},
        "validation_iou_at_0.5": float(row_at_0p5["iou"]),
        "n_validation_images": int(n_images),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "rule": "Reuse unchanged on the test set. Do NOT re-sweep on test data.",
    }
    Path(path).write_text(json.dumps(record, indent=2))
    return record


# ----------------------------------------------------------------------------
# 3. CLOUD COVERAGE
# ----------------------------------------------------------------------------
def binarize(probs, threshold):
    return (probs >= threshold).astype(np.uint8)


def coverage_table(gt, probs, threshold):
    """Per-image predicted/true cloud coverage (%) and per-image IoU."""
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
        "image_iou": np.where(union > 0, inter / np.maximum(union, 1), 1.0),  # empty/empty = 1
    })


def plot_coverage_scatter(table, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(table["gt_cloud_pct"], table["pred_cloud_pct"], s=8, alpha=0.5)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(xlabel="Ground-truth cloud (%)", ylabel="Predicted cloud (%)",
           title="Per-image coverage (validation)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.show()


# ----------------------------------------------------------------------------
# 4. VISUALIZATION
# ----------------------------------------------------------------------------
def to_rgb(img_chw):
    """(C,H,W) -> (H,W,3) in [0,1], 2-98 percentile stretch. Display only."""
    rgb = np.stack([img_chw[c] for c in RGB_CHANNELS], axis=-1)
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay_mask(rgb, mask, color=(1.0, 0.0, 0.0), alpha=0.45):
    """Blend a semi-transparent colour over the masked pixels."""
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out


def visualize_samples(images, gt, probs, threshold, indices, path, title):
    """One row per sample: original | ground truth | prediction | overlay."""
    pred = binarize(probs, threshold)
    fig, axes = plt.subplots(len(indices), 4, figsize=(14, 3.3 * len(indices)), squeeze=False)
    for row, i in zip(axes, indices):
        rgb = to_rgb(images[i])
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
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
OUT_DIR.mkdir(exist_ok=True)

# Task 1 - threshold sweep on validation predictions
val_images, val_gt, val_probs = collect_predictions(model, val_loader)

sweep, best = find_best_threshold(val_gt, val_probs)
row_0p5 = check_reproduces_reported(sweep)
THRESHOLD = float(best["threshold"])

sweep.to_csv(OUT_DIR / "validation_threshold_sweep.csv", index=False)
plot_sweep(sweep, THRESHOLD, OUT_DIR / "validation_threshold_sweep.png")
frozen = freeze_threshold(best, row_0p5, len(val_gt), OUT_DIR / "frozen_threshold.json")

print(sweep.round(4).to_string(index=False))
print(f"\nFrozen threshold: {THRESHOLD:.2f} | val IoU {best['iou']:.4f} (0.5 -> {row_0p5['iou']:.4f})")

# Task 2 - coverage and visualization (uses the frozen threshold)
coverage = coverage_table(val_gt, val_probs, THRESHOLD)
coverage.to_csv(OUT_DIR / "validation_cloud_coverage.csv", index=False)
plot_coverage_scatter(coverage, OUT_DIR / "coverage_scatter.png")
print(coverage.describe().round(3))

n = min(N_SAMPLES, len(val_gt))
random_idx = np.random.default_rng(SEED).choice(len(val_gt), size=n, replace=False)
worst_idx = coverage.nsmallest(n, "image_iou").index.to_numpy()

visualize_samples(val_images, val_gt, val_probs, THRESHOLD, random_idx,
                  OUT_DIR / "samples_random.png", "Random validation samples")
visualize_samples(val_images, val_gt, val_probs, THRESHOLD, worst_idx,
                  OUT_DIR / "samples_worst_iou.png", "Lowest-IoU validation samples")
