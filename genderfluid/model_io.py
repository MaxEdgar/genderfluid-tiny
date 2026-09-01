"""Model save/load in compact binary format."""

import json
import struct
import os
import numpy as np

from genderfluid.features import FeatureExtractor
from genderfluid.classifier import NameClassifier


# Binary format:
# Header: 4 bytes magic "GFT\0", 4 bytes version
# Config JSON length (4 bytes), Config JSON bytes
# Feature extractor weights shape (2x int32), weights as float32
# Classifier coef shape (2x int32), coef as float32
# Classifier intercept shape (1x int32), intercept as float32
# Calibration: class_prior probabilities (3x float32)

MAGIC = b"GFT\x00"
FORMAT_VERSION = 1


def save_model(
    feature_extractor: FeatureExtractor,
    classifier: NameClassifier,
    metadata: dict,
    output_path: str,
) -> int:
    """
    Save model in compact binary format.

    Returns file size in bytes.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    config = {
        "feature_extractor": feature_extractor.get_config(),
        "classifier": classifier.get_config(),
        "metadata": metadata,
    }
    config_json = json.dumps(config).encode("utf-8")

    with open(output_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))

        # Config
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)

        # Feature extractor is stateless (hashing), nothing to save for weights

        # Classifier coefficients
        if classifier.model is not None:
            coef = classifier.model.coef_.astype(np.float32)
            intercept = classifier.model.intercept_.astype(np.float32)

            f.write(struct.pack("<ii", *coef.shape))
            f.write(coef.tobytes())

            f.write(struct.pack("<i", intercept.shape[0]))
            f.write(intercept.tobytes())
        else:
            # Save empty arrays
            f.write(struct.pack("<ii", 0, 0))
            f.write(struct.pack("<i", 0))

        # Calibrated classifier class priors if available
        if classifier.calibrated is not None and hasattr(classifier.calibrated, "calibrated_classifiers_"):
            # Save class prior from base estimator
            if hasattr(classifier.model, "class_prior_"):
                priors = classifier.model.class_prior_.astype(np.float32)
            else:
                priors = np.zeros(3, dtype=np.float32)
            f.write(priors.tobytes())
        else:
            f.write(np.zeros(3, dtype=np.float32).tobytes())

    return os.path.getsize(output_path)


def load_model(
    model_path: str,
) -> tuple[FeatureExtractor, NameClassifier, dict]:
    """
    Load model from compact binary format.

    Returns:
        (feature_extractor, classifier, metadata)
    """
    with open(model_path, "rb") as f:
        # Header
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid model file: bad magic bytes")

        version = struct.unpack("<I", f.read(4))[0]
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported model version: {version}")

        # Config
        config_len = struct.unpack("<I", f.read(4))[0]
        config_json = f.read(config_len)
        config = json.loads(config_json)

        # Feature extractor (stateless)
        fe_config = config.get("feature_extractor", {})
        feature_extractor = FeatureExtractor.from_config(fe_config)

        # Classifier
        coef_shape = struct.unpack("<ii", f.read(8))
        if coef_shape[0] > 0 and coef_shape[1] > 0:
            coef = np.frombuffer(f.read(coef_shape[0] * coef_shape[1] * 4), dtype=np.float32).reshape(coef_shape)

            intercept_len = struct.unpack("<i", f.read(4))[0]
            intercept = np.frombuffer(f.read(intercept_len * 4), dtype=np.float32)
        else:
            coef = np.zeros((3, fe_config.get("dimensions", 4096)), dtype=np.float32)
            f.read(4)  # intercept_len = 0
            intercept = np.zeros(3, dtype=np.float32)

        # Class priors
        priors = np.frombuffer(f.read(12), dtype=np.float32)

        # Reconstruct classifier
        clf_config = config.get("classifier", {})
        classifier = NameClassifier.from_config(clf_config)

        # Create model directly (bypass training)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            C=clf_config.get("C", 1.0),
            max_iter=clf_config.get("max_iter", 1000),
            solver="lbfgs",
            random_state=42,
        )
        model.coef_ = coef
        model.intercept_ = intercept
        model.classes_ = np.array([0, 1, 2])
        if len(priors) == 3:
            model.class_prior_ = priors
        classifier.model = model
        # Mark calibrated as None since we saved the base model
        classifier.calibrated = None

        metadata = config.get("metadata", {})

    return feature_extractor, classifier, metadata


def save_native_bin(
    feature_extractor: FeatureExtractor,
    classifier: NameClassifier,
    metadata: dict,
    output_path: str,
) -> int:
    """Save model in a compact native binary format for C++ inference."""
    # This is the same as save_model but with a more explicit layout
    return save_model(feature_extractor, classifier, metadata, output_path)
