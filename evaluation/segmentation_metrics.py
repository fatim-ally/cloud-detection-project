"""
segmentation_metrics.py
------------------------
Evaluation Metrics for Binary Cloud Segmentation (IoU / Dice / Precision / Recall)

Project: AI-Based Cloud Detection and Localization in Satellite Images
Component: Evaluation and Metrics

Implemented from the team's segmentation-metrics specification document.
Pure NumPy — no dataset, no model, no training. Framework-independent:
accepts NumPy arrays, Python lists, torch.Tensor or tf.Tensor (converted
internally by duck-typing; neither PyTorch nor TensorFlow needs to be
installed for this module to import).

Positive class convention: 1 = cloud pixel, 0 = clear/background pixel.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 0. INPUT HANDLING — duck-typed conversion + shape canonicalization
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert NumPy arrays, Python lists, torch.Tensor or tf.Tensor to a
    NumPy array, without requiring torch/tf to be installed."""
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor duck-typing (avoids a hard torch dependency)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    # tf.Tensor duck-typing
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.asarray(x)


def _canonicalize_shape(arr):
    """
    Normalize accepted shapes down to (N, H, W):
        (H, W)          -> (1, H, W)                 single tile
        (N, H, W)       -> (N, H, W)                  already canonical
        (N, H, W, 1)    -> (N, H, W)                  TF/Keras channels-last
        (N, 1, H, W)    -> (N, H, W)                  PyTorch channels-first
        (N, H, W, C>1)  -> raises ValueError           multi-class, out of scope
    A singleton channel axis is squeezed ONLY for 4-D input, so a batch of
    one image is never silently collapsed into a single mask.
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            return arr[..., 0]          # channels-last, e.g. (N, H, W, 1)
        if arr.shape[1] == 1:
            return arr[:, 0, ...]       # channels-first, e.g. (N, 1, H, W)
        raise ValueError(
            f"4-D input with shape {arr.shape} has more than one channel "
            "(multi-class output) — out of scope for this binary module."
        )

    raise ValueError(f"Unsupported input shape {arr.shape} (expected 2-, 3-, or 4-D).")


def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def binarize(y_pred, threshold=0.5, from_logits=False):
    """Convert probabilities (or logits, with from_logits=True) to a {0,1} mask.

    mask[i, j] = 1 if probability[i, j] >= threshold, else 0
    (comparison is >=, so a pixel at exactly `threshold` is classified positive)
    """
    probs = _to_numpy(y_pred).astype(np.float64)
    if from_logits:
        probs = sigmoid(probs)
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError(
            "y_pred contains values outside [0, 1]. If this is a raw logit "
            "output, pass from_logits=True."
        )
    return (probs >= threshold).astype(np.uint8)


def _prepare(y_true, y_pred, threshold, from_logits):
    """Validate y_true, binarize y_pred, and canonicalize both to (N, H, W)."""
    t_raw = _to_numpy(y_true)
    unique_vals = np.unique(t_raw)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"y_true must contain only binary labels (0 or 1); found values "
            f"{unique_vals.tolist()}. y_true should already be a hard decision, "
            "not a probability map."
        )
    t = _canonicalize_shape(t_raw).astype(np.uint8)
    p = _canonicalize_shape(binarize(y_pred, threshold=threshold, from_logits=from_logits))

    if t.shape != p.shape:
        raise ValueError(
            f"y_true and y_pred describe different pixel grids after "
            f"canonicalization: {t.shape} vs {p.shape}."
        )
    return t, p


# ---------------------------------------------------------------------------
# 1. CONFUSION COUNTS
# ---------------------------------------------------------------------------

def confusion_counts(y_true, y_pred, threshold=0.5, from_logits=False, per_image=False):
    """Pixel-level TP / FP / FN / TN for the positive (cloud) class."""
    t, p = _prepare(y_true, y_pred, threshold, from_logits)
    axis = tuple(range(1, t.ndim))  # every axis except the batch axis

    tp = np.sum((t == 1) & (p == 1), axis=axis, dtype=np.float64)
    fp = np.sum((t == 0) & (p == 1), axis=axis, dtype=np.float64)
    fn = np.sum((t == 1) & (p == 0), axis=axis, dtype=np.float64)
    tn = np.sum((t == 0) & (p == 0), axis=axis, dtype=np.float64)

    if not per_image:
        tp, fp, fn, tn = tp.sum(), fp.sum(), fn.sum(), tn.sum()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# 2. ZERO-DENOMINATOR POLICY + SHARED METRIC ENGINE
# ---------------------------------------------------------------------------

def _safe_ratio(numerator, denominator, both_empty, empty_score):
    """Every zero-denominator case has a documented, tested answer — no NaN
    can ever escape. both_empty (TP+FP+FN == 0) -> empty_score (default 1.0,
    perfect agreement on a scene with no cloud and none predicted). Otherwise
    a zero denominator means one side is empty and the other isn't -> 0.0."""
    denominator = np.asarray(denominator, dtype=np.float64)
    numerator = np.asarray(numerator, dtype=np.float64)
    out = np.divide(numerator, denominator,
                     out=np.zeros_like(denominator, dtype=np.float64),
                     where=denominator > 0)
    return np.where(denominator > 0, out, np.where(both_empty, empty_score, 0.0))


def _reduce(values, reduction):
    values = np.atleast_1d(values)
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown reduction '{reduction}' (expected 'mean' or 'none').")


def _metric(y_true, y_pred, kind, threshold, from_logits, reduction, empty_score):
    micro = reduction == "micro"
    counts = confusion_counts(y_true, y_pred, threshold=threshold,
                               from_logits=from_logits, per_image=not micro)
    tp = np.atleast_1d(counts["tp"])
    fp = np.atleast_1d(counts["fp"])
    fn = np.atleast_1d(counts["fn"])
    both_empty = (tp + fp + fn) == 0

    def ratio(num, den):
        return _safe_ratio(num, den, both_empty, empty_score)

    if kind == "iou":
        values = ratio(tp, tp + fp + fn)
    elif kind == "dice":
        values = ratio(2.0 * tp, 2.0 * tp + fp + fn)
    elif kind == "precision":
        values = ratio(tp, tp + fp)
    elif kind == "recall":
        values = ratio(tp, tp + fn)
    else:
        raise ValueError(f"Unknown metric kind '{kind}'.")

    if micro:
        return float(values[0])
    return _reduce(values, "mean" if reduction == "mean" else reduction)


# ---------------------------------------------------------------------------
# 3. PUBLIC METRIC FUNCTIONS
# ---------------------------------------------------------------------------

def iou_score(y_true, y_pred, threshold=0.5, from_logits=False,
              reduction="mean", empty_score=1.0):
    """IoU = TP / (TP + FP + FN) for the cloud class. Primary metric of record."""
    return _metric(y_true, y_pred, "iou", threshold, from_logits, reduction, empty_score)


def dice_coefficient(y_true, y_pred, threshold=0.5, from_logits=False,
                      reduction="mean", empty_score=1.0):
    """Dice = 2*TP / (2*TP + FP + FN). Identical to pixel-wise F1."""
    return _metric(y_true, y_pred, "dice", threshold, from_logits, reduction, empty_score)


def precision_score(y_true, y_pred, threshold=0.5, from_logits=False,
                     reduction="mean", empty_score=1.0):
    """Precision = TP / (TP + FP). Purity of the predicted mask."""
    return _metric(y_true, y_pred, "precision", threshold, from_logits, reduction, empty_score)


def recall_score(y_true, y_pred, threshold=0.5, from_logits=False,
                  reduction="mean", empty_score=1.0):
    """Recall = TP / (TP + FN). Coverage of the true mask."""
    return _metric(y_true, y_pred, "recall", threshold, from_logits, reduction, empty_score)


def f1_score(y_true, y_pred, threshold=0.5, from_logits=False,
             reduction="mean", empty_score=1.0):
    """Alias of dice_coefficient (Dice is mathematically the harmonic mean
    of precision and recall)."""
    return dice_coefficient(y_true, y_pred, threshold, from_logits, reduction, empty_score)


def accuracy_score(y_true, y_pred, threshold=0.5, from_logits=False):
    """(TP + TN) / all pixels. SANITY CHECK ONLY — inflated by class imbalance
    on satellite scenes (TN usually dominates). Never the headline metric."""
    counts = confusion_counts(y_true, y_pred, threshold=threshold, from_logits=from_logits)
    total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    if total == 0:
        return 1.0
    return float((counts["tp"] + counts["tn"]) / total)


def specificity_score(y_true, y_pred, threshold=0.5, from_logits=False):
    """TN / (TN + FP). Is clear sky preserved? Quantifies data lost to over-masking."""
    counts = confusion_counts(y_true, y_pred, threshold=threshold, from_logits=from_logits)
    denom = counts["tn"] + counts["fp"]
    if denom == 0:
        return 1.0
    return float(counts["tn"] / denom)


def compute_all_metrics(y_true, y_pred, threshold=0.5, from_logits=False,
                         reduction="mean", empty_score=1.0, include_counts=True):
    """Full suite (IoU, Dice, Precision, Recall, Accuracy, Specificity) plus
    TP/FP/FN/TN in one pass. Log this straight to CSV or TensorBoard."""
    common = dict(threshold=threshold, from_logits=from_logits)
    macro = dict(reduction=reduction, empty_score=empty_score, **common)

    results = {
        "iou": iou_score(y_true, y_pred, **macro),
        "dice": dice_coefficient(y_true, y_pred, **macro),
        "precision": precision_score(y_true, y_pred, **macro),
        "recall": recall_score(y_true, y_pred, **macro),
        "accuracy": accuracy_score(y_true, y_pred, **common),
        "specificity": specificity_score(y_true, y_pred, **common),
    }
    if include_counts:
        results.update({k: float(v) for k, v in
                         confusion_counts(y_true, y_pred, **common).items()})
    return results


# ---------------------------------------------------------------------------
# 4. THRESHOLD SWEEP
# ---------------------------------------------------------------------------

def threshold_sweep(y_true, y_pred, thresholds=None, optimize="iou",
                     from_logits=False, reduction="mean", empty_score=1.0):
    """Sweep a set of candidate thresholds and return the best one plus the
    full sweep table. Sweep once on VALIDATION data, freeze the winner, and
    reuse that frozen value on the test split — tuning on test data inflates
    the reported score and invalidates the comparison."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    table = {}
    best_t, best_val = None, -np.inf
    for t in thresholds:
        metrics = compute_all_metrics(y_true, y_pred, threshold=t, from_logits=from_logits,
                                       reduction=reduction, empty_score=empty_score,
                                       include_counts=False)
        table[t] = metrics
        if metrics[optimize] > best_val:
            best_val = metrics[optimize]
            best_t = t

    return best_t, table


# ---------------------------------------------------------------------------
# 5. STREAMING ACCUMULATOR — keeps macro AND micro at once, over mini-batches
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """Streams .update() calls over mini-batches (e.g. a tf.data or PyTorch
    DataLoader loop) and keeps both macro and micro running results, without
    holding every prediction in memory at once."""

    def __init__(self, threshold=0.5, from_logits=False, empty_score=1.0):
        self.threshold = threshold
        self.from_logits = from_logits
        self.empty_score = empty_score
        self._per_image_iou = []
        self._per_image_dice = []
        self._per_image_precision = []
        self._per_image_recall = []
        self._tp_total = 0.0
        self._fp_total = 0.0
        self._fn_total = 0.0
        self._tn_total = 0.0
        self._n_images = 0

    def update(self, y_true, y_pred):
        counts = confusion_counts(y_true, y_pred, threshold=self.threshold,
                                   from_logits=self.from_logits, per_image=True)
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        both_empty = (tp + fp + fn) == 0

        self._per_image_iou.extend(_safe_ratio(tp, tp + fp + fn, both_empty, self.empty_score).tolist())
        self._per_image_dice.extend(_safe_ratio(2 * tp, 2 * tp + fp + fn, both_empty, self.empty_score).tolist())
        self._per_image_precision.extend(_safe_ratio(tp, tp + fp, both_empty, self.empty_score).tolist())
        self._per_image_recall.extend(_safe_ratio(tp, tp + fn, both_empty, self.empty_score).tolist())

        self._tp_total += tp.sum()
        self._fp_total += fp.sum()
        self._fn_total += fn.sum()
        self._tn_total += tn.sum()
        self._n_images += len(np.atleast_1d(tp))

    def result(self):
        tp, fp, fn, tn = self._tp_total, self._fp_total, self._fn_total, self._tn_total
        micro_both_empty = np.array([(tp + fp + fn) == 0])

        return {
            "macro_iou": float(np.mean(self._per_image_iou)) if self._per_image_iou else float("nan"),
            "macro_dice": float(np.mean(self._per_image_dice)) if self._per_image_dice else float("nan"),
            "macro_precision": float(np.mean(self._per_image_precision)) if self._per_image_precision else float("nan"),
            "macro_recall": float(np.mean(self._per_image_recall)) if self._per_image_recall else float("nan"),
            "micro_iou": float(_safe_ratio(tp, tp + fp + fn, micro_both_empty, self.empty_score)[0]),
            "micro_dice": float(_safe_ratio(2 * tp, 2 * tp + fp + fn, micro_both_empty, self.empty_score)[0]),
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
            "n_images": self._n_images,
        }


# ---------------------------------------------------------------------------
# 6. PYTORCH TRAINING LOSS (lazily imported — torch is NOT a hard dependency)
# ---------------------------------------------------------------------------

def soft_dice_loss_torch(logits, targets, smooth=1.0):
    """Differentiable soft-Dice training loss (PyTorch). Smoothing belongs in
    the LOSS, not the reported metric — this is the only place in the module
    a smoothing constant is applied."""
    import torch

    probs = torch.sigmoid(logits)
    probs_flat = probs.reshape(probs.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1).float()

    intersection = (probs_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# 7. STANDALONE DEMO — matches the worked examples in the spec document
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("10.1 — 4x4 hand-checkable example")
    print("=" * 60)

    y_true = np.array([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]])
    y_prob = np.array([[0.01, 0.02, 0.03, 0.04],
                        [0.10, 0.90, 0.95, 0.80],
                        [0.05, 0.85, 0.40, 0.02],
                        [0.00, 0.01, 0.02, 0.03]])

    counts = confusion_counts(y_true, y_prob)
    print(f"counts -> TP={counts['tp']:.0f} FP={counts['fp']:.0f} "
          f"FN={counts['fn']:.0f} TN={counts['tn']:.0f}")
    print(f"iou = {iou_score(y_true, y_prob):.4f}   "
          f"dice = {dice_coefficient(y_true, y_prob):.4f}")
    print(f"precision = {precision_score(y_true, y_prob):.4f}   "
          f"recall = {recall_score(y_true, y_prob):.4f}")
    print(f"accuracy = {accuracy_score(y_true, y_prob):.4f}   "
          f"specificity = {specificity_score(y_true, y_prob):.4f}")

    print("\n" + "=" * 60)
    print("10.2 — batch of three tiles, channels-last (3, 32, 32, 1)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    y_true_b = np.zeros((3, 32, 32, 1), dtype=np.uint8)
    y_true_b[0, 8:20, 8:20, 0] = 1          # tile 1: compact block
    y_true_b[1, 15:17, :, 0] = 1            # tile 2: thin band
    # tile 3 stays entirely clear

    y_prob_b = np.clip(y_true_b.astype(np.float64) * 0.8 + 0.1
                        + rng.normal(0, 0.15, size=y_true_b.shape), 0, 1)

    per_image_iou = iou_score(y_true_b, y_prob_b, reduction="none")
    print(f"per-image IoU : {np.round(per_image_iou, 3).tolist()}")
    print(f"macro IoU (mean)  : {iou_score(y_true_b, y_prob_b):.4f}")
    print(f"micro IoU (pooled): {iou_score(y_true_b, y_prob_b, reduction='micro'):.4f}")

    summary = compute_all_metrics(y_true_b, y_prob_b)
    print(f"iou = {summary['iou']:.4f}  dice = {summary['dice']:.4f}  "
          f"precision = {summary['precision']:.4f}  recall = {summary['recall']:.4f}")
    print(f"accuracy = {summary['accuracy']:.4f}  specificity = {summary['specificity']:.4f}")
    print(f"tp = {summary['tp']:.0f} fp = {summary['fp']:.0f} "
          f"fn = {summary['fn']:.0f} tn = {summary['tn']:.0f}")

    print("\n" + "=" * 60)
    print("10.3 — edge cases (no NaN should ever appear)")
    print("=" * 60)

    clear = np.zeros((8, 8))
    print("clear scene, nothing predicted  -> IoU =",
          iou_score(clear, np.zeros((8, 8))))
    print("clear scene, everything predicted -> IoU =",
          iou_score(clear, np.ones((8, 8))))
    cloudy = np.ones((8, 8))
    print("cloudy scene, nothing predicted -> recall =",
          recall_score(cloudy, np.zeros((8, 8))))

    print("\nDone — no NaN produced in any case above.")