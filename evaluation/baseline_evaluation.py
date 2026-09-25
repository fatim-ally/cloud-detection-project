"""
Meerab Baseline Evaluation
AI-Based Cloud Detection and Localization in Satellite Images

Purpose:
    Establish a reference baseline before evaluating the real U-Net model.

Baseline:
    Predict "no cloud anywhere" for every validation patch.

Metrics:
    IoU, Dice, Precision, Recall

Input:
    val_patches.csv
    Ground-truth masks referenced by the CSV's gt_path column

Output:
    baseline_reference.json
    baseline_per_patch.csv

IMPORTANT:
    This script does not train a model and does not invent results.
    All reported metrics are calculated from the real validation masks
    using the team's existing segmentation_metrics.py module.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================
# Load team's existing metrics module
# ============================================================

try:
    import segmentation_metrics as sm
except ImportError:
    print(
        "\nERROR: segmentation_metrics.py was not found.\n"
        "Place segmentation_metrics.py in the same folder as this script.\n"
    )
    sys.exit(1)


# ============================================================
# Load and binarize ground-truth mask
# ============================================================

def load_mask(gt_path):
    """Load a ground-truth mask and convert it to binary 0/1."""

    gt_path = Path(gt_path)

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask not found: {gt_path}"
        )

    suffix = gt_path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        if tifffile is None:
            raise ImportError(
                "tifffile is required for TIFF masks. "
                "Install it with: pip install tifffile"
            )
        mask = tifffile.imread(str(gt_path))

    else:
        if Image is None:
            raise ImportError(
                "Pillow is required for image masks. "
                "Install it with: pip install pillow"
            )
        mask = np.array(Image.open(gt_path))

    mask = np.asarray(mask)

    # Remove an accidental channel dimension.
    if mask.ndim == 3:
        mask = mask[..., 0]

    # Convert mask to binary:
    # 0 = background, 1 = cloud
    return (mask > 0).astype(np.uint8)


# ============================================================
# Resolve paths stored in val_patches.csv
# ============================================================

def resolve_gt_path(gt_path, csv_path):
    """
    Resolve a ground-truth path.

    Absolute paths are used directly.
    Relative paths are checked relative to:
      1. current working directory
      2. directory containing val_patches.csv
    """

    path = Path(str(gt_path))

    if path.is_absolute():
        return path

    if path.exists():
        return path

    csv_dir = Path(csv_path).resolve().parent
    candidate = csv_dir / path

    if candidate.exists():
        return candidate

    return candidate


# ============================================================
# Run team's existing metric functions
# ============================================================

def calculate_metrics(pred_mask, gt_mask):
    """Calculate the four required segmentation metrics."""

    if hasattr(sm, "compute_all_metrics"):
        raw = sm.compute_all_metrics(gt_mask, pred_mask)

    else:
        required = [
            "iou_score",
            "dice_coefficient",
            "precision_score",
            "recall_score",
        ]

        missing = [
            name for name in required
            if not hasattr(sm, name)
        ]

        if missing:
            raise AttributeError(
                "segmentation_metrics.py is missing: "
                + ", ".join(missing)
            )

        raw = {
            "iou": sm.iou_score(pred_mask, gt_mask),
            "dice": sm.dice_coefficient(pred_mask, gt_mask),
            "precision": sm.precision_score(pred_mask, gt_mask),
            "recall": sm.recall_score(pred_mask, gt_mask),
        }

    # Normalize possible naming differences.
    result = {}

    for key, value in raw.items():
        name = str(key).lower()

        if name in ("iou", "iou_score"):
            result["IoU"] = float(value)

        elif name in ("dice", "dice_coefficient"):
            result["Dice"] = float(value)

        elif name in ("precision", "precision_score"):
            result["Precision"] = float(value)

        elif name in ("recall", "recall_score"):
            result["Recall"] = float(value)

    required_metrics = ["IoU", "Dice", "Precision", "Recall"]

    missing = [
        metric for metric in required_metrics
        if metric not in result
    ]

    if missing:
        raise ValueError(
            "The metrics module did not return: "
            + ", ".join(missing)
        )

    return result


# ============================================================
# Evaluate trivial baseline
# ============================================================

def evaluate_baseline(csv_path, n=None):
    """Evaluate the all-zero/no-cloud baseline."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "gt_path" not in df.columns:
        raise ValueError(
            "'gt_path' column was not found in val_patches.csv.\n"
            f"Available columns: {list(df.columns)}"
        )

    total_patches = len(df)

    if n is not None:
        if n <= 0:
            raise ValueError("--n must be greater than 0.")
        df = df.head(min(n, total_patches))

    print(f"Validation patches listed: {total_patches}")
    print(f"Patches selected: {len(df)}")
    print("Baseline: predict no cloud anywhere (all-zero mask)\n")

    patch_results = []
    skipped = 0

    for index, row in df.iterrows():

        try:
            gt_path = resolve_gt_path(
                row["gt_path"],
                csv_path
            )

            gt_mask = load_mask(gt_path)

            # Trivial baseline:
            # predict background everywhere.
            pred_mask = np.zeros_like(
                gt_mask,
                dtype=np.uint8
            )

            metrics = calculate_metrics(
                pred_mask,
                gt_mask
            )

            patch_results.append({
                "patch_key": row.get("patch_key", index),
                "IoU": metrics["IoU"],
                "Dice": metrics["Dice"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
            })

            if len(patch_results) % 50 == 0:
                print(
                    f"  Evaluated {len(patch_results)} patches..."
                )

        except Exception as error:
            skipped += 1
            print(
                f"  [SKIPPED] row {index}: {error}",
                file=sys.stderr
            )

    if not patch_results:
        raise RuntimeError(
            "No validation masks were successfully evaluated. "
            "Check the gt_path values and dataset location."
        )

    results_df = pd.DataFrame(patch_results)

    metric_names = [
        "IoU",
        "Dice",
        "Precision",
        "Recall",
    ]

    mean = {
        metric: float(results_df[metric].mean())
        for metric in metric_names
    }

    std = {
        metric: float(results_df[metric].std())
        for metric in metric_names
    }

    summary = {
        "baseline_type": "trivial baseline (predict nothing)",
        "dataset": "38-Cloud",
        "source_csv": str(csv_path.resolve()),
        "n_patches_selected": int(len(df)),
        "n_patches_evaluated": int(len(results_df)),
        "n_patches_skipped": int(skipped),
        "mean": mean,
        "std": std,
    }

    return summary, results_df


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trivial no-cloud baseline "
            "on the 38-Cloud validation set."
        )
    )

    parser.add_argument(
        "--csv",
        default="val_patches.csv",
        help="Path to val_patches.csv",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of validation patches to evaluate.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate the complete validation set.",
    )

    parser.add_argument(
        "--out",
        default="baseline_reference.json",
        help="Output JSON file.",
    )

    parser.add_argument(
        "--details",
        default="baseline_per_patch.csv",
        help="Output per-patch CSV file.",
    )

    args = parser.parse_args()

    n = None if args.all else args.n

    print("\n" + "=" * 65)
    print("CLOUD SEGMENTATION - TRIVIAL BASELINE EVALUATION")
    print("=" * 65)

    summary, results_df = evaluate_baseline(
        args.csv,
        n
    )

    print("\n" + "=" * 65)
    print("TRIVIAL BASELINE (PREDICT NOTHING)")
    print("=" * 65)

    print(
        f"Patches evaluated : "
        f"{summary['n_patches_evaluated']}"
    )

    print(
        f"Patches skipped   : "
        f"{summary['n_patches_skipped']}"
    )

    print("\nReference metrics:")

    for metric in [
        "IoU",
        "Dice",
        "Precision",
        "Recall",
    ]:
        print(
            f"  {metric:<10} "
            f"mean = {summary['mean'][metric]:.6f}   "
            f"std = {summary['std'][metric]:.6f}"
        )

    print("=" * 65)

    print(
        "\nThese values are the reference baseline obtained by "
        "predicting no cloud for every evaluated validation patch."
    )

    with open(args.out, "w") as file:
        json.dump(summary, file, indent=4)

    results_df.to_csv(
        args.details,
        index=False
    )

    print(f"\nSummary saved to: {args.out}")
    print(f"Per-patch results saved to: {args.details}")
    print("\nBaseline evaluation completed successfully.")


if __name__ == "__main__":
    main()
