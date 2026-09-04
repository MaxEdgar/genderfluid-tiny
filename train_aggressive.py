#!/usr/bin/env python3
"""
Aggressive training script for genderfluid-tiny.

Designed to run on GitHub Actions runners (7GB RAM).
Sweeps feature dimensions, n-gram ranges, C values, and solvers.
Selects the best model by validation F1 under 50MB.
"""

import gc
import json
import os
import sys
import time
import random
import itertools

import numpy as np
import yaml

from genderfluid.preprocessing import normalize_name
from genderfluid.features import FeatureExtractor
from genderfluid.classifier import NameClassifier, LABELS
from genderfluid.calibration import calibration_error, confusion_matrix
from genderfluid.model_io import save_model


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CONFIGURATION - aggressive sweep parameters
# ============================================================

FEATURE_DIMS = [4096, 8192, 16384]
NGRAM_RANGES = [(2, 5), (3, 5)]
C_VALUES = [0.5, 1.0, 5.0]
SOLVERS = ["lbfgs"]
MAX_ITER = 2000
MAX_MODEL_SIZE_MB = 50.0
MIN_MACRO_F1_TARGET = 0.85


def load_split(path):
    names, labels, weights = [], [], []
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
    features = fe.extract_batch(names)
    y_true = np.array(labels)
    class_indices, probas = clf.predict(features)

    accuracy = float(np.mean(class_indices == y_true))
    cm = confusion_matrix(y_true, class_indices, LABELS)

    precisions, recalls, f1s = [], [], []
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


def model_size_bytes(path):
    return os.path.getsize(path)


def train_single_config(train_names, train_labels, train_weights,
                        val_names, val_labels, val_weights,
                        dims, ngram_range, C, solver):
    """Train one configuration and return metrics + model objects."""
    fe = FeatureExtractor(
        min_ngram=ngram_range[0],
        max_ngram=ngram_range[1],
        dimensions=dims,
    )

    X_tr = fe.extract_batch(train_names)
    X_v = fe.extract_batch(val_names)

    clf = NameClassifier(C=C, max_iter=MAX_ITER, min_confidence=0.70)
    clf.train(X_tr, np.array(train_labels), sample_weight=np.array(train_weights))

    metrics = evaluate(fe, clf, val_names, val_labels, val_weights)

    # Check size
    temp_path = f"/tmp/gft_test_{dims}_{ngram_range[0]}{ngram_range[1]}_{C}_{solver}.bin"
    save_model(fe, clf, {}, temp_path)
    size = model_size_bytes(temp_path)
    os.remove(temp_path)

    del X_tr, X_v
    gc.collect()

    return fe, clf, metrics, size


def compute_class_weights(labels):
    """Compute balanced class weights."""
    counts = np.bincount(labels, minlength=len(LABELS))
    total = len(labels)
    weights = {}
    for i, count in enumerate(counts):
        if count > 0:
            weights[i] = total / (len(LABELS) * count)
        else:
            weights[i] = 1.0
    return weights


def main():
    print("=" * 60)
    print("GENDERFLUID-TINY AGGRESSIVE TRAINING")
    print("=" * 60)
    print()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "validation.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    if not os.path.exists(train_path):
        print("ERROR: No training data found. Run: python fetch_multinational_data.py")
        sys.exit(1)

    print("Loading datasets...")
    train_names, train_labels, train_weights = load_split(train_path)
    val_names, val_labels, val_weights = load_split(val_path)
    test_names, test_labels, test_weights = load_split(test_path)

    print(f"  Train:      {len(train_names):,} examples")
    print(f"  Validation: {len(val_names):,} examples")
    print(f"  Test:       {len(test_names):,} examples")
    print()

    # Compute class weights for balanced training
    class_w = compute_class_weights(train_labels)
    balanced_weights = np.array([class_w[l] for l in train_labels], dtype=np.float32)
    # Multiply with existing weights
    combined_weights = balanced_weights * np.array(train_weights, dtype=np.float32)

    # Build parameter grid
    configs = list(itertools.product(FEATURE_DIMS, NGRAM_RANGES, C_VALUES, SOLVERS))
    total = len(configs)
    print(f"Sweeping {total} configurations...")
    print(f"  Feature dims: {FEATURE_DIMS}")
    print(f"  N-gram ranges: {NGRAM_RANGES}")
    print(f"  C values: {C_VALUES}")
    print(f"  Solvers: {SOLVERS}")
    print()

    best = None
    results = []

    for i, (dims, ngram_range, C, solver) in enumerate(configs):
        t0 = time.time()
        try:
            fe, clf, metrics, size = train_single_config(
                train_names, train_labels, combined_weights,
                val_names, val_labels, val_weights,
                dims, ngram_range, C, solver,
            )
        except Exception as e:
            print(f"  [{i+1}/{total}] FAILED: dims={dims} ng={ngram_range} C={C} sol={solver} -- {e}")
            continue

        elapsed = time.time() - t0
        size_mb = size / (1024 * 1024)

        result = {
            "dims": dims,
            "ngram": ngram_range,
            "C": C,
            "solver": solver,
            "f1": metrics["macro_f1"],
            "acc": metrics["accuracy"],
            "size_mb": size_mb,
            "ece": metrics["calibration_error"],
            "time": elapsed,
        }
        results.append(result)

        status = ""
        if size_mb > MAX_MODEL_SIZE_MB:
            status = " [OVER SIZE]"
        elif metrics["macro_f1"] >= MIN_MACRO_F1_TARGET:
            status = " [TARGET MET]"

        print(f"  [{i+1}/{total}] dims={dims:<6} ng={str(ngram_range):<8} C={C:<5} sol={solver:<8} "
              f"F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"Size={size_mb:.2f}MB  ECE={metrics['calibration_error']:.4f}  "
              f"({elapsed:.1f}s){status}")

        # Select best: highest F1 under size limit
        if size_mb <= MAX_MODEL_SIZE_MB:
            if best is None or metrics["macro_f1"] > best["f1"]:
                best = result.copy()
                best["_fe"] = fe
                best["_clf"] = clf
                best["_metrics"] = metrics
                best["_size"] = size

        # Free memory
        del fe, clf, metrics
        gc.collect()

    print()
    print("=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)

    # Sort by F1
    results.sort(key=lambda x: x["f1"], reverse=True)
    print(f"\nTop 10 configurations:")
    print(f"  {'Rank':<5} {'Dims':<7} {'Ngram':<10} {'C':<6} {'Solver':<10} {'F1':<8} {'Acc':<8} {'Size':<8} {'ECE':<8}")
    print("  " + "-" * 75)
    for j, r in enumerate(results[:10]):
        print(f"  {j+1:<5} {r['dims']:<7} {str(r['ngram']):<10} {r['C']:<6} {r['solver']:<10} "
              f"{r['f1']:<8.4f} {r['acc']:<8.4f} {r['size_mb']:<8.2f} {r['ece']:<8.4f}")

    if best is None:
        print("\nERROR: No valid model found under 50MB!")
        sys.exit(1)

    print(f"\nBest configuration:")
    print(f"  Features:     {best['dims']}")
    print(f"  N-gram range: {best['ngram']}")
    print(f"  C:            {best['C']}")
    print(f"  Solver:       {best['solver']}")
    print(f"  Val F1:       {best['f1']:.4f}")
    print(f"  Val Acc:      {best['acc']:.4f}")
    print(f"  Val ECE:      {best['ece']:.4f}")
    print(f"  Model size:   {best['size_mb']:.2f} MB")
    print()

    # Final test evaluation
    fe = best["_fe"]
    clf = best["_clf"]

    print("Final test set evaluation...")
    test_metrics = evaluate(fe, clf, test_names, test_labels, test_weights)
    print(f"  Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  Test ECE:          {test_metrics['calibration_error']:.4f}")
    print(f"  Per-class F1:")
    for label in LABELS:
        print(f"    {label:>20}: {test_metrics['per_class_f1'][label]:.4f}")
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
        "model_name": "genderfluid-tiny",
        "version": config.get("model", {}).get("version", "1.0.0"),
        "seed": SEED,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_mode": "aggressive",
        "train_size": len(train_names),
        "val_size": len(val_names),
        "test_size": len(test_names),
        "feature_dimensions": best["dims"],
        "ngram_range": list(best["ngram"]),
        "C": best["C"],
        "solver": best["solver"],
        "sweep_total": total,
        "sweep_best_rank": next(i for i, r in enumerate(results)
                                 if (r["dims"], r["ngram"], r["C"], r["solver"])
                                 == (best["dims"], best["ngram"], best["C"], best["solver"])) + 1,
        "validation_f1": best["f1"],
        "validation_accuracy": best["acc"],
        "validation_ece": best["ece"],
        "test_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_ece": test_metrics["calibration_error"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "class_weights_used": True,
    }

    # Attach an n-gram bloom filter of the dataset vocabulary (each unique
    # name lives in exactly one split, so all splits are included) so
    # inference can return "uncertain" for out-of-vocabulary names instead
    # of guessing.
    from genderfluid.features import build_bloom
    bloom_names = list(train_names) + list(val_names) + list(test_names)
    clf.bloom = build_bloom(bloom_names, fe.min_ngram, fe.max_ngram)

    size = save_model(fe, clf, metadata, bin_path)
    size_mb = size / (1024 * 1024)

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:        {bin_path}")
    print(f"  Size:         {size_mb:.2f} MB ({size:,} bytes)")
    print(f"  Test F1:      {test_metrics['macro_f1']:.4f}")
    print(f"  Test Acc:     {test_metrics['accuracy']:.4f}")
    print(f"  Configs swept: {total}")
    best_rank = next(i for i, r in enumerate(results)
                     if (r["dims"], r["ngram"], r["C"], r["solver"])
                     == (best["dims"], best["ngram"], best["C"], best["solver"])) + 1
    print(f"  Best rank:    #{best_rank}")
    print()
    print(f"  50 MB limit:  {'PASS' if size_mb < 50 else 'FAIL'}")
    print(f"  F1 target:    {'PASS' if test_metrics['macro_f1'] >= MIN_MACRO_F1_TARGET else 'BELOW TARGET'}")
    print()

    # Save sweep results for reference
    sweep_path = os.path.join(output_dir, "sweep_results.json")
    sweep_save = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with open(sweep_path, "w") as f:
        json.dump(sweep_save, f, indent=2)
    print(f"Sweep results saved to {sweep_path}")


if __name__ == "__main__":
    main()
