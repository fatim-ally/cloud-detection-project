"""
evaluate_real_model.py
------------------------
Run the REAL trained U-Net checkpoint's predictions through Meerab's
segmentation_metrics.py, and compare against the trivial "predict nothing"
baseline already established in baseline_reference.json.

Project: AI-Based Cloud Detection and Localization in Satellite Images
Component: AI Model Development + Evaluation (joint task)

This is the script that closes the loop between:
  - unet_cloud_segmentation_skeleton.py / train_baseline.py  (the model)
  - data_pipeline_cloud_segmentation.py                      (the data loader)
  - segmentation_metrics.py                                  (the real metrics)

STATUS: evaluation only. Does not train or modify the model.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from data_pipeline_cloud_segmentation import build_dataset, BANDS
from segmentation_metrics import (
    iou_score, dice_coefficient, precision_score, recall_score,
    MetricsAccumulator,
)


def load_band_stats(stats_json_path):
    with open(stats_json_path) as f:
        stats = json.load(f)
    means = np.array([stats["bands"][b]["mean"] for b in BANDS], dtype=np.float32)
    stds = np.array([stats["bands"][b]["std"] for b in BANDS], dtype=np.float32)
    return means, stds


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained U-Net checkpoint on val_patches.csv with real metrics."
    )
    parser.add_argument("--split_dir", required=True,
                         help="Folder with val_patches.csv")
    parser.add_argument("--stats_json", required=True,
                         help="Path to band_stats_full_train.json")
    parser.add_argument("--checkpoint", required=True,
                         help="Path to the trained .keras model (e.g. baseline_best.keras)")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Probability threshold for binarizing predictions")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--baseline_json", default=None,
                         help="Optional: path to baseline_reference.json, to print a comparison")
    parser.add_argument("--out", default="real_model_evaluation.json")
    parser.add_argument("--details", default="real_model_per_patch.csv")
    args = parser.parse_args()

    means, stds = load_band_stats(args.stats_json)
    val_csv = os.path.join(args.split_dir, "val_patches.csv")

    df = pd.read_csv(val_csv)
    patch_keys = df["patch_key"].tolist()
    print(f"Validation patches: {len(patch_keys)}")

    print(f"\nLoading trained model from: {args.checkpoint}")
    # compile=False: we only need inference here, not the training-time
    # loss/optimizer state, so no need to re-register custom loss functions.
    model = tf.keras.models.load_model(args.checkpoint, compile=False)

    print("Building validation dataset...")
    val_ds = build_dataset(val_csv, means, stds, has_mask=True,
                            batch_size=args.batch_size, shuffle=False)

    acc = MetricsAccumulator(threshold=args.threshold)
    per_patch_rows = []
    idx = 0

    print(f"\nRunning inference + scoring (threshold={args.threshold})...")
    for images, masks in val_ds:
        preds = model.predict(images, verbose=0)  # (batch, H, W, 1) probabilities
        masks_np = masks.numpy()

        acc.update(masks_np, preds)

        batch_iou = np.atleast_1d(iou_score(masks_np, preds, threshold=args.threshold, reduction="none"))
        batch_dice = np.atleast_1d(dice_coefficient(masks_np, preds, threshold=args.threshold, reduction="none"))
        batch_prec = np.atleast_1d(precision_score(masks_np, preds, threshold=args.threshold, reduction="none"))
        batch_rec = np.atleast_1d(recall_score(masks_np, preds, threshold=args.threshold, reduction="none"))

        for j in range(len(batch_iou)):
            per_patch_rows.append({
                "patch_key": patch_keys[idx + j],
                "IoU": float(batch_iou[j]),
                "Dice": float(batch_dice[j]),
                "Precision": float(batch_prec[j]),
                "Recall": float(batch_rec[j]),
            })
        idx += len(batch_iou)

        if idx % 200 < args.batch_size:
            print(f"  Evaluated {idx} patches...")

    results_df = pd.DataFrame(per_patch_rows)
    metric_names = ["IoU", "Dice", "Precision", "Recall"]
    macro_mean = {m: float(results_df[m].mean()) for m in metric_names}
    macro_std = {m: float(results_df[m].std()) for m in metric_names}
    micro = acc.result()

    summary = {
        "model_type": "trained U-Net (real predictions)",
        "checkpoint": os.path.abspath(args.checkpoint),
        "threshold": args.threshold,
        "dataset": "38-Cloud",
        "source_csv": os.path.abspath(val_csv),
        "n_patches_evaluated": int(len(results_df)),
        "macro_mean": macro_mean,
        "macro_std": macro_std,
        "micro": {"iou": micro["micro_iou"], "dice": micro["micro_dice"]},
        "confusion_counts": {"tp": micro["tp"], "fp": micro["fp"],
                              "fn": micro["fn"], "tn": micro["tn"]},
    }

    print("\n" + "=" * 65)
    print("REAL TRAINED MODEL — EVALUATION")
    print("=" * 65)
    for m in metric_names:
        print(f"  {m:<10} mean = {macro_mean[m]:.6f}   std = {macro_std[m]:.6f}")
    print(f"  micro IoU  = {micro['micro_iou']:.6f}")
    print(f"  micro Dice = {micro['micro_dice']:.6f}")
    print("=" * 65)

    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json) as f:
            baseline = json.load(f)
        print("\nComparison vs. trivial 'predict nothing' baseline:")
        for m in metric_names:
            base_val = baseline["mean"][m]
            real_val = macro_mean[m]
            delta = real_val - base_val
            arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "=")
            print(f"  {m:<10} baseline = {base_val:.4f}   real model = {real_val:.4f}   "
                  f"delta = {delta:+.4f} ({arrow})")
        summary["baseline_comparison"] = {
            m: {"baseline": baseline["mean"][m],
                "real_model": macro_mean[m],
                "delta": macro_mean[m] - baseline["mean"][m]}
            for m in metric_names
        }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    results_df.to_csv(args.details, index=False)

    print(f"\nSummary saved to: {args.out}")
    print(f"Per-patch results saved to: {args.details}")
    print("\nNote: these numbers reflect only the short baseline training run "
          "(3 epochs) completed so far — a starting point for comparison, not "
          "a claim about final model quality.")


if __name__ == "__main__":
    main()
