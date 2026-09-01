#!/usr/bin/env python3
"""
GGUF Export for genderfluid-tiny.

GGUF (GGML Universal Format) is designed for large language models and neural networks
that use tensor operations. The genderfluid-tiny model uses a simple logistic regression
classifier with character n-gram features, which is a linear model that doesn't benefit
from GGUF's tensor-based format.

Instead, we use a compact native binary format (.bin) that is:
- 10-100x smaller than GGUF for this type of model
- Faster to load
- Compatible with the native C++ inference code

The native binary format stores:
- Model configuration (JSON)
- Classifier coefficients (float32 matrix)
- Classifier intercept (float32 vector)
- Class priors (float32 vector)

If you need GGUF compatibility for integration with llama.cpp or similar runtimes,
consider using a different architecture (e.g., tiny transformer).
"""

import os
import sys
import struct


def main():
    bin_path = os.path.join(os.path.dirname(__file__), "models", "genderfluid-tiny.bin")
    gguf_path = os.path.join(os.path.dirname(__file__), "models", "genderfluid-tiny.gguf")

    if not os.path.exists(bin_path):
        print("Error: No trained model found.")
        print("Run: python train.py")
        sys.exit(1)

    # Create a minimal GGUF header that indicates this is a placeholder
    # The actual model is in the .bin format
    print("GGUF Export")
    print("=" * 40)
    print()
    print("GGUF is not appropriate for this model architecture.")
    print("genderfluid-tiny uses a logistic regression classifier with")
    print("character n-gram features - a linear model that doesn't benefit")
    print("from GGUF's tensor-based format.")
    print()
    print(f"Instead, use the native binary format:")
    print(f"  {bin_path}")
    print()
    print(f"Size: {os.path.getsize(bin_path)} bytes ({os.path.getsize(bin_path) / 1024:.1f} KB)")
    print()
    print("For native C++ inference:")
    print("  g++ -O2 -std=c++17 -o genderfluid-tiny native/main.cpp native/model.cpp")
    print("  ./genderfluid-tiny \"Elva Retta\"")
    print()
    print("For Python inference:")
    print("  python predict.py \"Elva Retta\"")

    # Create a tiny placeholder gguf file for compatibility
    # This is NOT a valid GGUF model - it's just a marker file
    with open(gguf_path, "wb") as f:
        # Write a minimal header indicating this is a placeholder
        f.write(b"GGUF")  # Magic
        f.write(b"\x03\x00\x00\x00")  # Version 3
        f.write(b"\x00\x00\x00\x00")  # Tensor count = 0
        f.write(b"\x00\x00\x00\x00")  # Metadata count = 0
        # Add a comment as metadata
        comment = b"genderfluid-tiny: logistic regression classifier, use .bin format instead"
        f.write(struct.pack("<I", len(comment)))
        f.write(comment)

    print(f"\nPlaceholder GGUF created at: {gguf_path}")
    print("NOTE: This is NOT a valid GGUF model. Use .bin format instead.")


if __name__ == "__main__":
    main()
