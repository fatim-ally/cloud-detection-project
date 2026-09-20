"""
train_baseline.py
-------------------
First real baseline training run for cloud segmentation.

Project: AI-Based Cloud Detection and Localization in Satellite Images
Component: AI Model Development — Week 3 task

This connects:
  - unet_cloud_segmentation_skeleton.py  (the model architecture)
  - data_pipeline_cloud_segmentation.py  (the data loader)
  - band_stats_full_train.json           (Ayesha's full-dataset normalization stats)

into one working training loop, and runs a SHORT baseline pass (few epochs)
to confirm everything trains end-to-end without errors. This is NOT the
final full training run — it's the "does this actually work" checkpoint
before committing to a long training session.

Usage (in Colab, once dataset_split/ and band_stats_full_train.json exist):
    python train_baseline.py \\
        --split_dir /content/drive/MyDrive/cloud-detection-project/dataset_split \\
        --stats_json /content/drive/MyDrive/cloud-detection-project/band_stats_full_train.json \\
        --epochs 3
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from unet_cloud_segmentation_skeleton import build_unet
from data_pipeline_cloud_segmentation import build_dataset, BANDS


# ---------------------------------------------------------------------------
# 1. LOSS + METRIC — BCE + Dice, per the team's agreed training config
# ---------------------------------------------------------------------------

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Soft Dice, used as a monitoring metric during training (not the raw,
    unsmoothed Dice used for final reported evaluation — that's Meerab's
    segmentation_metrics.py, kept deliberately separate, see project notes)."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Combined loss: binary cross-entropy + Dice, as specified in the
    team's U-Net implementation plan."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)


# ---------------------------------------------------------------------------
# 2. LOAD PRECOMPUTED BAND STATS (Ayesha's full-dataset numbers — no
#    recomputation here, this script trusts and reuses her output)
# ---------------------------------------------------------------------------

def load_band_stats(stats_json_path):
    with open(stats_json_path) as f:
        stats = json.load(f)

    means = np.array([stats["bands"][b]["mean"] for b in BANDS], dtype=np.float32)
    stds = np.array([stats["bands"][b]["std"] for b in BANDS], dtype=np.float32)

    print(f"Loaded band stats from: {stats_json_path}")
    print(f"  Computed from: {stats.get('computed_from', 'unknown')} "
          f"({stats.get('n_patches', '?')} patches)")
    print(f"  Means: {means}")
    print(f"  Stds : {stds}")
    return means, stds


# ---------------------------------------------------------------------------
# 3. MAIN — build everything, run a short baseline training pass
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the first baseline U-Net training pass.")
    parser.add_argument("--split_dir", required=True,
                         help="Folder with train_patches.csv / val_patches.csv")
    parser.add_argument("--stats_json", required=True,
                         help="Path to band_stats_full_train.json (from Ayesha's script)")
    parser.add_argument("--epochs", type=int, default=3,
                         help="Small number for this baseline check — NOT full training")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    args = parser.parse_args()

    means, stds = load_band_stats(args.stats_json)

    train_csv = os.path.join(args.split_dir, "train_patches.csv")
    val_csv = os.path.join(args.split_dir, "val_patches.csv")

    print("\nBuilding train/val datasets...")
    train_ds = build_dataset(train_csv, means, stds, has_mask=True,
                              batch_size=args.batch_size, shuffle=True)
    val_ds = build_dataset(val_csv, means, stds, has_mask=True,
                            batch_size=args.batch_size, shuffle=False)

    print("\nBuilding model...")
    model = build_unet()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, "accuracy"],
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, "baseline_best.keras"),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nStarting baseline training run — {args.epochs} epoch(s), "
          f"batch size {args.batch_size}...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    history_path = os.path.join(args.checkpoint_dir, "baseline_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"\nSaved training history to: {history_path}")
    print("This is a BASELINE check only — no claim is made here about final "
          "model quality. Full training, tuning, and Meerab's evaluation "
          "metrics come next.")


if __name__ == "__main__":
    main()
