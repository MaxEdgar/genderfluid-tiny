"""Inference module - high-level API for predictions."""

import os
from typing import Optional

from genderfluid.preprocessing import normalize_name
from genderfluid.features import FeatureExtractor
from genderfluid.classifier import NameClassifier, LABELS
from genderfluid.model_io import load_model


# Default model path
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "genderfluid-tiny.bin"
)

# Singleton model cache
_model_cache: Optional[tuple[FeatureExtractor, NameClassifier, dict]] = None
_model_path_cache: Optional[str] = None


def _load_or_cache(model_path: Optional[str] = None) -> tuple[FeatureExtractor, NameClassifier, dict]:
    """Load model from disk or use cached version."""
    global _model_cache, _model_path_cache

    path = model_path or DEFAULT_MODEL_PATH

    if _model_cache is not None and _model_path_cache == path:
        return _model_cache

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at {path}\n\n"
            "Run: python train.py"
        )

    _model_cache = load_model(path)
    _model_path_cache = path
    return _model_cache


def predict_name(
    name: str,
    model_path: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """
    Predict gender association for a single name.

    Args:
        name: Full name string
        model_path: Path to model file (optional, uses default)
        country: Optional country context
        language: Optional language context

    Returns:
        Dictionary with prediction results
    """
    feature_extractor, classifier, metadata = _load_or_cache(model_path)

    normalized = normalize_name(name)
    if not normalized:
        return {
            "name": name,
            "girl_associated_probability": 0.33,
            "boy_associated_probability": 0.33,
            "uncertain_probability": 0.34,
            "classification": "uncertain",
            "confidence": "low",
            "warning": "Empty or invalid name",
        }

    features = feature_extractor.extract(normalized)
    features_2d = features.reshape(1, -1)

    class_idx, proba = classifier.predict(features_2d)
    class_idx = class_idx[0]
    proba = proba[0]

    classification = LABELS[class_idx]
    max_prob = float(max(proba))

    if max_prob >= 0.90:
        confidence = "high"
    elif max_prob >= 0.70:
        confidence = "medium"
    else:
        confidence = "low"

    result = {
        "name": name,
        "girl_associated_probability": round(float(proba[0]), 4),
        "boy_associated_probability": round(float(proba[1]), 4),
        "uncertain_probability": round(float(proba[2]), 4),
        "classification": classification,
        "confidence": confidence,
    }

    # Add context warnings if requested but data is insufficient
    if country or language:
        result["context_warning"] = (
            "Model was not trained with regional/language context. "
            "Results reflect general training data associations only."
        )

    return result


def predict_names(
    names: list[str],
    model_path: Optional[str] = None,
) -> list[dict]:
    """
    Predict gender association for multiple names (batch).

    Optimized to not reload the model for each prediction.
    """
    feature_extractor, classifier, metadata = _load_or_cache(model_path)

    results = []
    normalized_names = [normalize_name(n) for n in names]

    # Batch feature extraction
    valid_indices = [i for i, n in enumerate(normalized_names) if n]
    if valid_indices:
        valid_features = feature_extractor.extract_batch(
            [normalized_names[i] for i in valid_indices]
        )
        class_indices, probas = classifier.predict(valid_features)

    for i, name in enumerate(names):
        norm = normalized_names[i]
        if not norm:
            results.append({
                "name": name,
                "girl_associated_probability": 0.33,
                "boy_associated_probability": 0.33,
                "uncertain_probability": 0.34,
                "classification": "uncertain",
                "confidence": "low",
                "warning": "Empty or invalid name",
            })
        else:
            idx = valid_indices.index(i)
            class_idx = class_indices[idx]
            proba = probas[idx]
            classification = LABELS[class_idx]
            max_prob = float(max(proba))

            if max_prob >= 0.90:
                confidence = "high"
            elif max_prob >= 0.70:
                confidence = "medium"
            else:
                confidence = "low"

            results.append({
                "name": name,
                "girl_associated_probability": round(float(proba[0]), 4),
                "boy_associated_probability": round(float(proba[1]), 4),
                "uncertain_probability": round(float(proba[2]), 4),
                "classification": classification,
                "confidence": confidence,
            })

    return results
