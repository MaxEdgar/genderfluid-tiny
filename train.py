#!/usr/bin/env python3
"""Training script for genderfluid-tiny model."""

import json
import os
import sys
import time
import random

import numpy as np
import yaml

from genderfluid.preprocessing import normalize_name
from genderfluid.features import FeatureExtractor
from genderfluid.classifier import NameClassifier, LABELS
from genderfluid.calibration import calibration_error, confusion_matrix
from genderfluid.model_io import save_model


def load_split(path: str) -> tuple:
    """Load a dataset split. Returns (names, labels, weights)."""
    names = []
    labels = []
    weights = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Split not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            name = entry.get("name", "").strip()
            label = entry.get("label", "").strip()
            weight = entry.get("weight", 1.0)

            if not name or label not in LABELS:
                continue

            normalized = normalize_name(name)
            if not normalized:
                continue

            names.append(normalized)
            labels.append(LABELS.index(label))
            weights.append(float(weight))

    return names, labels, weights


def evaluate(fe, clf, names, labels, weights):
    """Evaluate model on a dataset split."""
    features = fe.extract_batch(names)
    y_true = np.array(labels)

    class_indices, probas = clf.predict(features)

    accuracy = float(np.mean(class_indices == y_true))

    cm = confusion_matrix(y_true, class_indices, LABELS)

    precisions = []
    recalls = []
    f1s = []
    for i in range(len(LABELS)):
        tp = cm[LABELS[i]][LABELS[i]]
        fp = sum(cm[LABELS[j]][LABELS[i]] for j in range(len(LABELS)) if j != i)
        fn = sum(cm[LABELS[i]][LABELS[j]] for j in range(len(LABELS)) if j != i)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    ece = calibration_error(y_true, probas)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "per_class_f1": {LABELS[i]: float(f1s[i]) for i in range(len(LABELS))},
        "confusion_matrix": cm,
        "calibration_error": float(ece),
        "n_samples": len(names),
    }


def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def sweep_features(train_names, train_labels, train_weights,
                   val_names, val_labels, val_weights,
                   config, feature_dims):
    """Sweep over feature dimensions and select best."""
    min_f1 = config.get("quality", {}).get("minimum_macro_f1", 0.80)
    best = None

    for dims in feature_dims:
        fe = FeatureExtractor(
            min_ngram=config.get("model", {}).get("min_ngram", 2),
            max_ngram=config.get("model", {}).get("max_ngram", 5),
            dimensions=dims,
        )

        X_tr = fe.extract_batch(train_names)
        X_v = fe.extract_batch(val_names)

        clf = NameClassifier(min_confidence=config.get("model", {}).get("min_confidence", 0.70))
        clf.train(X_tr, np.array(train_labels), sample_weight=np.array(train_weights))

        metrics = evaluate(fe, clf, val_names, val_labels, val_weights)

        # Save temporarily to check size
        temp_path = f"/tmp/gft_{dims}.bin"
        save_model(fe, clf, {}, temp_path)
        size_mb = model_size_mb(temp_path)
        os.remove(temp_path)

        print(f"  Features: {dims}  F1: {metrics['macro_f1']:.4f}  Acc: {metrics['accuracy']:.4f}  Size: {size_mb:.2f} MB")

        candidate = {
            "dimensions": dims,
            "feature_extractor": fe,
            "classifier": clf,
            "metrics": metrics,
            "size_mb": size_mb,
        }

        if best is None:
            best = candidate
        elif metrics["macro_f1"] >= min_f1 and size_mb < best["size_mb"]:
            best = candidate
        elif metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = candidate

    return best


def main():
    """Main training pipeline."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 50)
    print("GENDERFLUID-TINY TRAINING")
    print("=" * 50)
    print()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "validation.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    if not os.path.exists(train_path):
        print("No training data found. Run: python prepare_data.py")
        sys.exit(1)

    print("Loading datasets...")
    train_names, train_labels, train_weights = load_split(train_path)
    val_names, val_labels, val_weights = load_split(val_path)
    test_names, test_labels, test_weights = load_split(test_path)

    print(f"  Train: {len(train_names)} examples")
    print(f"  Validation: {len(val_names)} examples")
    print(f"  Test: {len(test_names)} examples")
    print()

    # Feature dimension sweep
    feature_dims = config.get("training", {}).get("max_features_sweep", [1024, 2048, 4096, 8192])
    print("Sweeping feature dimensions...")
    best = sweep_features(
        train_names, train_labels, train_weights,
        val_names, val_labels, val_weights,
        config, feature_dims,
    )

    fe = best["feature_extractor"]
    clf = best["classifier"]

    print(f"\nBest: {best['dimensions']} features, F1={best['metrics']['macro_f1']:.4f}, Size={best['size_mb']:.2f} MB")
    print()

    # Final evaluation on test set
    print("Final test set evaluation...")
    test_metrics = evaluate(fe, clf, test_names, test_labels, test_weights)
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Test ECE: {test_metrics['calibration_error']:.4f}")
    print(f"  Confusion matrix:")
    for true_label in LABELS:
        row = "    "
        for pred_label in LABELS:
            row += f"{test_metrics['confusion_matrix'][true_label][pred_label]:>6}"
        print(f"  {true_label:>20}: {row}")
    print()

    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "genderfluid-tiny.bin")

    metadata = {
        "model_name": config.get("model", {}).get("name", "genderfluid-tiny"),
        "version": config.get("model", {}).get("version", "1.0.0"),
        "seed": seed,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(train_names),
        "val_size": len(val_names),
        "test_size": len(test_names),
        "feature_dimensions": best["dimensions"],
        "validation_f1": best["metrics"]["macro_f1"],
        "test_f1": test_metrics["macro_f1"],
    }

    size = save_model(fe, clf, metadata, bin_path)
    size_mb = size / (1024 * 1024)
    print(f"Model saved to {bin_path}")
    print(f"Model size: {size_mb:.2f} MB")
    print()

    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Model: {bin_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Test F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print()

    size_check = size_mb * 1024 * 1024
    print(f"Model size: {size_mb:.2f} MB")
    print(f"50 MB limit: {'PASS' if size_mb < 50 else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
