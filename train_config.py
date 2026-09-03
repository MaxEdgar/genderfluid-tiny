#!/usr/bin/env python3
"""Train a single configuration. Used by the parallel GitHub Actions matrix.

Each matrix job trains exactly one (dims, ngram, C, solver) combination and
uploads its model plus a result JSON. A separate finalize step picks the best.

Env overrides (used by CI and for low-memory local smoke tests):
  GFT_DATA_DIR  where train.jsonl / validation.jsonl live (default: ./data)
  GFT_OUT_DIR   where the model and result JSON are written (default: ./out)
"""

import json
import os
import sys
import time

import numpy as np

from train_aggressive import (
    MAX_MODEL_SIZE_MB,
    compute_class_weights,
    load_split,
    train_single_config,
)
from genderfluid.model_io import save_model


def main():
    if len(sys.argv) != 5:
        print("usage: train_config.py <dims> <ngram_min-ngram_max> <C> <solver>")
        sys.exit(2)

    dims = int(sys.argv[1])
    ngram_lo, ngram_hi = (int(v) for v in sys.argv[2].split("-"))
    ngram_range = (ngram_lo, ngram_hi)
    c_value = float(sys.argv[3])
    solver = sys.argv[4]

    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get("GFT_DATA_DIR", os.path.join(base, "data"))
    out_dir = os.environ.get("GFT_OUT_DIR", os.path.join(base, "out"))

    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "validation.jsonl")
    for p in (train_path, val_path):
        if not os.path.exists(p):
            print(f"ERROR: split not found: {p}")
            sys.exit(1)

    print(f"Config: dims={dims} ngram={ngram_range} C={c_value} solver={solver}")
    train_names, train_labels, train_weights = load_split(train_path)
    val_names, val_labels, val_weights = load_split(val_path)
    print(f"  Train: {len(train_names):,}  Validation: {len(val_names):,}")

    class_w = compute_class_weights(train_labels)
    balanced = np.array([class_w[l] for l in train_labels], dtype=np.float32)
    combined = balanced * np.array(train_weights, dtype=np.float32)

    t0 = time.time()
    fe, clf, metrics, size = train_single_config(
        train_names, train_labels, combined,
        val_names, val_labels, val_weights,
        dims, ngram_range, c_value, solver,
    )
    elapsed = time.time() - t0

    tag = f"{dims}_{ngram_lo}-{ngram_hi}_{c_value}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"model_{tag}.bin")
    save_model(fe, clf, {"config": tag}, model_path)

    result = {
        "dims": dims,
        "ngram": list(ngram_range),
        "C": c_value,
        "solver": solver,
        "f1": metrics["macro_f1"],
        "acc": metrics["accuracy"],
        "ece": metrics["calibration_error"],
        "size_mb": size / (1024 * 1024),
        "time": elapsed,
        "model": os.path.basename(model_path),
    }
    with open(os.path.join(out_dir, f"result_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)

    over = " [OVER SIZE]" if result["size_mb"] > MAX_MODEL_SIZE_MB else ""
    print(f"  F1={result['f1']:.4f} Acc={result['acc']:.4f} "
          f"Size={result['size_mb']:.3f}MB ({elapsed:.1f}s){over}")
    print(json.dumps(result))
    if result["size_mb"] > MAX_MODEL_SIZE_MB:
        sys.exit(1)


if __name__ == "__main__":
    main()
