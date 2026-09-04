"""Model save/load in compact binary format."""

import json
import struct
import os
import numpy as np

from genderfluid.features import FeatureExtractor
from genderfluid.classifier import NameClassifier, NUM_CLASSES


# Binary format v2:
# Header: 4 bytes magic "GFT\0", 4 bytes version (2)
# Config JSON length (4 bytes), Config JSON bytes
# Classifier coef shape (2x int32), coef as float32
# Classifier intercept shape (1x int32), intercept as float32
# Class prior (3x float32)
# Calibration A (3x float32), Calibration B (3x float32)

MAGIC = b"GFT\x00"
FORMAT_VERSION = 2


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

        # Classifier coefficients
        if classifier.model is not None:
            coef = classifier.model.coef_.astype(np.float32)
            intercept = classifier.model.intercept_.astype(np.float32)
            prior = getattr(classifier.model, "class_prior_", None)
        elif classifier.coef_ is not None:
            coef = np.asarray(classifier.coef_, dtype=np.float32)
            intercept = np.asarray(classifier.intercept_, dtype=np.float32)
            prior = classifier.class_prior_
        else:
            coef = None
            intercept = None
            prior = None

        if coef is not None:
            f.write(struct.pack("<ii", *coef.shape))
            f.write(coef.tobytes())
            f.write(struct.pack("<i", intercept.shape[0]))
            f.write(intercept.tobytes())
        else:
            f.write(struct.pack("<ii", 0, 0))
            f.write(struct.pack("<i", 0))

        # Class prior
        if prior is not None:
            priors = np.asarray(prior, dtype=np.float32)
        else:
            priors = np.zeros(NUM_CLASSES, dtype=np.float32)
        f.write(priors.tobytes())

        # Calibration parameters (sigmoid A, B per class)
        calib_A = classifier.calib_A if classifier.calib_A is not None else np.ones(NUM_CLASSES, dtype=np.float32)
        calib_B = classifier.calib_B if classifier.calib_B is not None else np.zeros(NUM_CLASSES, dtype=np.float32)
        f.write(calib_A.astype(np.float32).tobytes())
        f.write(calib_B.astype(np.float32).tobytes())

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
            raise ValueError("Invalid model file: bad magic bytes")

        version = struct.unpack("<I", f.read(4))[0]
        if version not in (1, FORMAT_VERSION):
            raise ValueError(f"Unsupported model version: {version}")

        # Config
        config_len = struct.unpack("<I", f.read(4))[0]
        config_json = f.read(config_len)
        config = json.loads(config_json)

        # Feature extractor (stateless)
        fe_config = config.get("feature_extractor", {})
        feature_extractor = FeatureExtractor.from_config(fe_config)

        # Classifier coefficients
        coef_shape = struct.unpack("<ii", f.read(8))
        if coef_shape[0] > 0 and coef_shape[1] > 0:
            coef = np.frombuffer(
                f.read(coef_shape[0] * coef_shape[1] * 4), dtype=np.float32
            ).reshape(coef_shape)
            intercept_len = struct.unpack("<i", f.read(4))[0]
            intercept = np.frombuffer(f.read(intercept_len * 4), dtype=np.float32)
        else:
            coef = np.zeros((3, fe_config.get("dimensions", 4096)), dtype=np.float32)
            f.read(4)  # intercept_len = 0
            intercept = np.zeros(3, dtype=np.float32)

        # Class priors
        priors = np.frombuffer(f.read(12), dtype=np.float32)

        # Calibration parameters
        if version >= 2:
            calib_A = np.frombuffer(f.read(12), dtype=np.float32).copy()
            calib_B = np.frombuffer(f.read(12), dtype=np.float32).copy()
        else:
            # v1 format: skip old calibration data, use identity
            calib_A = np.ones(NUM_CLASSES, dtype=np.float32)
            calib_B = np.zeros(NUM_CLASSES, dtype=np.float32)

        # Reconstruct classifier with numpy weights (no sklearn needed).
        clf_config = config.get("classifier", {})
        classifier = NameClassifier.from_config(clf_config)
        classifier.coef_ = coef
        classifier.intercept_ = intercept
        classifier.classes_ = np.array([0, 1, 2])
        classifier.class_prior_ = priors if len(priors) == 3 else None
        classifier.calib_A = calib_A
        classifier.calib_B = calib_B

        metadata = config.get("metadata", {})

    return feature_extractor, classifier, metadata


def save_native_bin(
    feature_extractor: FeatureExtractor,
    classifier: NameClassifier,
    metadata: dict,
    output_path: str,
) -> int:
    """Save model in compact binary format for C++ inference."""
    return save_model(feature_extractor, classifier, metadata, output_path)
