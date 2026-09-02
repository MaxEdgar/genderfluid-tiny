"""Inference API for genderfluid-tiny."""

import os
from typing import Optional

from genderfluid.preprocessing import normalize_name
from genderfluid.classifier import LABELS
from genderfluid.model_io import load_model


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "genderfluid-tiny.bin"
)


class GenderfluidModel:
    """
    Name-gender association classifier.

    Usage::

        model = GenderfluidModel()              # loads default model
        model = GenderfluidModel("path/to.bin") # loads custom model

        result = model.predict("Elva Retta")
        print(result["classification"])  # "girl-associated"

        results = model.predict_batch(["Emma", "James", "Alex"])
    """

    def __init__(self, model_path: Optional[str] = None):
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No model found at {path}\n\n"
                "Train a model first:\n"
                "  python process_real_data.py\n"
                "  python prepare_data.py\n"
                "  python train.py"
            )
        self._fe, self._clf, self._metadata = load_model(path)
        self._model_path = path

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def model_path(self) -> str:
        return self._model_path

    def _format_result(self, name: str, class_idx: int, proba) -> dict:
        proba = proba[0] if proba.ndim > 1 else proba
        max_prob = float(max(proba))

        if max_prob >= 0.90:
            confidence = "high"
        elif max_prob >= 0.70:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "name": name,
            "girl_associated_probability": round(float(proba[0]), 4),
            "boy_associated_probability": round(float(proba[1]), 4),
            "uncertain_probability": round(float(proba[2]), 4),
            "classification": LABELS[class_idx],
            "confidence": confidence,
        }

    def predict(self, name: str) -> dict:
        """
        Predict gender association for a single name.

        Returns dict with keys: name, girl_associated_probability,
        boy_associated_probability, uncertain_probability,
        classification, confidence.
        """
        normalized = normalize_name(name)
        if not normalized:
            return {
                "name": name,
                "girl_associated_probability": 0.33,
                "boy_associated_probability": 0.33,
                "uncertain_probability": 0.34,
                "classification": "uncertain",
                "confidence": "low",
            }

        features = self._fe.extract(normalized).reshape(1, -1)
        class_idx, proba = self._clf.predict(features)
        return self._format_result(name, int(class_idx[0]), proba)

    def predict_batch(self, names: list[str]) -> list[dict]:
        """
        Predict gender association for multiple names.

        More efficient than calling predict() in a loop
        because the model is loaded once and features are batched.
        """
        normalized = [normalize_name(n) for n in names]

        results = [None] * len(names)
        valid = [(i, normalized[i]) for i in range(len(names)) if normalized[i]]

        if valid:
            valid_names = [n for _, n in valid]
            features = self._fe.extract_batch(valid_names)
            class_indices, probas = self._clf.predict(features)

            for j, (orig_idx, _) in enumerate(valid):
                results[orig_idx] = self._format_result(
                    names[orig_idx], int(class_indices[j]), probas[j:j+1]
                )

        for i, name in enumerate(names):
            if results[i] is None:
                results[i] = {
                    "name": name,
                    "girl_associated_probability": 0.33,
                    "boy_associated_probability": 0.33,
                    "uncertain_probability": 0.34,
                    "classification": "uncertain",
                    "confidence": "low",
                }

        return results


# ---------------------------------------------------------------------------
# Module-level convenience functions (use singleton model for repeated calls)
# ---------------------------------------------------------------------------

_default_model: Optional[GenderfluidModel] = None


def _get_default_model() -> GenderfluidModel:
    global _default_model
    if _default_model is None:
        _default_model = GenderfluidModel()
    return _default_model


def predict_name(
    name: str,
    model_path: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """
    Predict gender association for a single name.

    Args:
        name: Full name string (e.g. "Elva Retta", "Alex", "Michelle Renatta Chan")
        model_path: Path to model file (optional, uses default)
        country: Optional country context (informational only)
        language: Optional language context (informational only)

    Returns:
        Dictionary with keys: name, girl_associated_probability,
        boy_associated_probability, uncertain_probability,
        classification, confidence.

    Example::

        result = predict_name("Elva Retta")
        print(result["classification"])  # "girl-associated"
    """
    if model_path:
        model = GenderfluidModel(model_path)
    else:
        model = _get_default_model()

    result = model.predict(name)

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

    More efficient than calling predict_name() in a loop.

    Args:
        names: List of name strings
        model_path: Path to model file (optional, uses default)

    Returns:
        List of prediction dictionaries.

    Example::

        results = predict_names(["Emma", "James", "Alex"])
        for r in results:
            print(f"{r['name']}: {r['classification']}")
    """
    if model_path:
        model = GenderfluidModel(model_path)
    else:
        model = _get_default_model()

    return model.predict_batch(names)
