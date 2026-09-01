"""Probability calibration analysis."""

import numpy as np


def calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Expected calibration error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Find samples in this probability bin
        mask = (y_proba.max(axis=1) >= lower) & (y_proba.max(axis=1) < upper)
        if mask.sum() == 0:
            continue

        bin_proba = y_proba[mask].max(axis=1)
        bin_true = y_true[mask] == np.argmax(y_proba[mask], axis=1)

        avg_confidence = bin_proba.mean()
        avg_accuracy = bin_true.mean()

        ece += mask.sum() / len(y_true) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def reliability_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Compute reliability diagram data.

    Returns list of dicts with bin_lower, bin_upper, avg_confidence, avg_accuracy, count.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    result = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        mask = (y_proba.max(axis=1) >= lower) & (y_proba.max(axis=1) < upper)
        if mask.sum() == 0:
            result.append({
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "avg_confidence": float((lower + upper) / 2),
                "avg_accuracy": 0.0,
                "count": 0,
            })
            continue

        bin_proba = y_proba[mask].max(axis=1)
        bin_true = (y_true[mask] == np.argmax(y_proba[mask], axis=1)).astype(float)

        result.append({
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "avg_confidence": float(bin_proba.mean()),
            "avg_accuracy": float(bin_true.mean()),
            "count": int(mask.sum()),
        })

    return result


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict:
    """
    Compute confusion matrix as a dict.

    Returns:
        {label: {predicted_label: count}}
    """
    n_labels = len(labels)
    matrix = np.zeros((n_labels, n_labels), dtype=int)

    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1

    result = {}
    for i, true_label in enumerate(labels):
        result[true_label] = {}
        for j, pred_label in enumerate(labels):
            result[true_label][pred_label] = int(matrix[i, j])

    return result
