#!/usr/bin/env python3
"""Evaluation script for genderfluid-tiny model."""

import json
import os
import sys

import numpy as np

from genderfluid.preprocessing import normalize_name
from genderfluid.classifier import LABELS
from genderfluid.calibration import calibration_error, confusion_matrix
from genderfluid.model_io import load_model


def main():
    model_path = os.path.join(os.path.dirname(__file__), "models", "genderfluid-tiny.bin")
    if not os.path.exists(model_path):
        print("Error: no model found at", model_path)
        print("Run: python train.py")
        sys.exit(1)

    print("Loading model...")
    fe, clf, metadata = load_model(model_path)

    for split_name in ["validation.jsonl", "test.jsonl"]:
        split_path = os.path.join(os.path.dirname(__file__), "data", split_name)
        if not os.path.exists(split_path):
            print(f"Skipping {split_name} (not found)")
            continue

        names = []
        labels = []
        with open(split_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                name = entry.get("name", "").strip()
                label = entry.get("label", "").strip()
                if name and label in LABELS:
                    normalized = normalize_name(name)
                    if normalized:
                        names.append(normalized)
                        labels.append(LABELS.index(label))

        if not names:
            continue

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

        print(f"\n{'='*40}")
        print(f"Evaluation on {split_name}")
        print(f"{'='*40}")
        print(f"Samples: {len(names)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {np.mean(f1s):.4f}")
        print(f"Macro Precision: {np.mean(precisions):.4f}")
        print(f"Macro Recall: {np.mean(recalls):.4f}")
        print(f"Calibration Error (ECE): {ece:.4f}")
        print(f"\nPer-class F1:")
        for i, label in enumerate(LABELS):
            print(f"  {label}: {f1s[i]:.4f}")
        print(f"\nConfusion matrix:")
        print(f"  {'':>20} {'pred_girl':>12} {'pred_boy':>12} {'pred_unc':>12}")
        for true_label in LABELS:
            row = f"  {true_label:>20}"
            for pred_label in LABELS:
                row += f" {cm[true_label][pred_label]:>12}"
            print(row)


if __name__ == "__main__":
    main()
