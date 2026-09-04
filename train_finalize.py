#!/usr/bin/env python3
"""Pick the best configuration from a parallel sweep and produce the final model.

Reads result_*.json / model_*.bin files produced by train_config.py (matrix
jobs), selects the best config by validation F1 under the size limit,
evaluates it on the held-out test set, and writes the final model to
models/genderfluid-tiny.bin plus the packaged copy genderfluid/models/.

Env overrides (used by CI and for low-memory local smoke tests):
  GFT_DATA_DIR    where train/validation/test jsonl live (default: ./data)
  GFT_SWEEP_DIR   where result_*.json and model_*.bin live (default: ./sweep)
  GFT_MODEL_OUT   write the final model to this single path instead of the
                  repo model locations (used for smoke tests; also skips the
                  sweep_results.json write)
"""

import glob
import json
import os
import sys
import time

from train_aggressive import MAX_MODEL_SIZE_MB, load_split, evaluate
from genderfluid.classifier import LABELS
from genderfluid.model_io import load_model, save_model


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get("GFT_DATA_DIR", os.path.join(base, "data"))
    sweep_dir = os.environ.get("GFT_SWEEP_DIR", os.path.join(base, "sweep"))
    model_out = os.environ.get("GFT_MODEL_OUT")

    results = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "result_*.json"))):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        print(f"ERROR: no result_*.json files found in {sweep_dir}")
        sys.exit(1)

    print("=" * 60)
    print("FINALIZE: choosing best config from parallel sweep")
    print("=" * 60)
    print(f"  Configs available: {len(results)}")

    results.sort(key=lambda r: r["f1"], reverse=True)
    candidates = [r for r in results if r["size_mb"] <= MAX_MODEL_SIZE_MB]
    if not candidates:
        print("ERROR: every configuration exceeds the size limit")
        sys.exit(1)
    best = max(candidates, key=lambda r: r["f1"])
    best_rank = next(i for i, r in enumerate(results)
                     if (r["dims"], r["ngram"], r["C"], r["solver"])
                     == (best["dims"], best["ngram"], best["C"], best["solver"])) + 1

    print(f"  Best config (#{best_rank}): dims={best['dims']} ngram={best['ngram']} "
          f"C={best['C']} solver={best['solver']}")
    print(f"  Val F1:  {best['f1']:.4f}   Val Acc: {best['acc']:.4f}   "
          f"Size: {best['size_mb']:.3f} MB")

    # Load the winner and evaluate on the held-out test set.
    model_file = os.path.join(sweep_dir, best["model"])
    if not os.path.exists(model_file):
        print(f"ERROR: winner model missing: {model_file}")
        sys.exit(1)
    fe, clf, _ = load_model(model_file)

    test_path = os.path.join(data_dir, "test.jsonl")
    test_names, test_labels, test_weights = load_split(test_path)
    print(f"  Test set: {len(test_names):,} names")

    test_metrics = evaluate(fe, clf, test_names, test_labels, test_weights)
    print("\nFinal test set evaluation:")
    print(f"  Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  Test ECE:          {test_metrics['calibration_error']:.4f}")
    for label in LABELS:
        print(f"    {label:>20}: {test_metrics['per_class_f1'][label]:.4f}")

    # Rebuild the final model with full metadata.
    train_size = len(load_split(os.path.join(data_dir, "train.jsonl"))[0])
    val_size = len(load_split(os.path.join(data_dir, "validation.jsonl"))[0])
    # Read the configured version so model metadata tracks the package.
    # Plain-text parse to avoid a yaml dependency in CI.
    model_version = "1.0.0"
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "config.yaml"), encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("version:"):
                    model_version = line.split(":", 1)[1].strip().strip("\"'")
                    break
    except Exception:
        pass
    metadata = {
        "model_name": "genderfluid-tiny",
        "version": model_version,
        "seed": 42,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_mode": "matrix_sweep",
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_names),
        "feature_dimensions": best["dims"],
        "ngram_range": best["ngram"],
        "C": best["C"],
        "solver": best["solver"],
        "sweep_total": len(results),
        "sweep_best_rank": best_rank,
        "validation_f1": best["f1"],
        "validation_accuracy": best["acc"],
        "validation_ece": best["ece"],
        "test_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_ece": test_metrics["calibration_error"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "class_weights_used": True,
    }

    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        save_model(fe, clf, metadata, model_out)
        print(f"\nFinal model written to: {model_out} (smoke mode)")
        return

    # Build an n-gram bloom filter over the whole dataset (each unique name
    # lives in exactly one split, so training alone would miss valid names in
    # the other splits) so inference can return "uncertain" for genuinely
    # out-of-vocabulary names instead of guessing.
    from genderfluid.features import build_bloom
    all_names = []
    for split in ("train", "validation", "test"):
        split_names, _, _ = load_split(os.path.join(data_dir, f"{split}.jsonl"))
        all_names.extend(split_names)
    clf.bloom = build_bloom(all_names, fe.min_ngram, fe.max_ngram)

    model_dir = os.path.join(base, "models")
    os.makedirs(model_dir, exist_ok=True)
    bin_path = os.path.join(model_dir, "genderfluid-tiny.bin")
    packaged_path = os.path.join(base, "genderfluid", "models",
                                 "genderfluid-tiny.bin")
    save_model(fe, clf, metadata, bin_path)
    save_model(fe, clf, metadata, packaged_path)

    size = os.path.getsize(bin_path)
    size_mb = size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:        {bin_path}")
    print(f"  Packaged:     {packaged_path}")
    print(f"  Size:         {size_mb:.2f} MB ({size:,} bytes)")
    print(f"  Test F1:      {test_metrics['macro_f1']:.4f}")
    print(f"  Test Acc:     {test_metrics['accuracy']:.4f}")
    print(f"  Configs swept: {len(results)}")
    print(f"  Best rank:    #{best_rank}")
    print(f"  50 MB limit:  {'PASS' if size_mb < 50 else 'FAIL'}")

    with open(os.path.join(model_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Sweep results saved to {model_dir}/sweep_results.json")


if __name__ == "__main__":
    main()
