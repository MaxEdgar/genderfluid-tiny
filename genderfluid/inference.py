"""Inference API for genderfluid-tiny."""

import os
from typing import Optional

from genderfluid.preprocessing import (
    fold_accents,
    has_supported_script,
    normalize_name,
)
from genderfluid.classifier import LABELS
from genderfluid.features import iter_ngram_strings
from genderfluid.model_io import load_model


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "genderfluid-tiny.bin"
)

# A normalized name whose discriminative character n-grams (pure-letter
# 4-grams and longer) were mostly NOT seen during training is treated as
# out-of-vocabulary: the model has no learned signal for it and should say
# ``uncertain`` rather than guess. Absent a bloom filter (older model
# files), every name passes.
OOV_MIN_COVERAGE = 0.50

# Probability split used for inputs the model cannot judge at all: empty
# strings, names in scripts never seen in training, or out-of-vocabulary
# patterns. Kept deliberately flat so callers cannot mistake it for a guess.
_UNKNOWN_PROB = (0.05, 0.05, 0.90)


class GenderfluidModel:
    """
    Name-gender association classifier.

    Usage::

        model = GenderfluidModel()              # loads default model
        model = GenderfluidModel("path/to.bin") # loads custom model

        result = model.predict("Olivia")
        print(result["classification"])  # "girl-associated"

        results = model.predict_batch(["Emma", "James", "Alex"])
    """

    def __init__(self, model_path: Optional[str] = None):
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No model found at {path}\n\n"
                "Build the model first:\n"
                "  python fetch_multinational_data.py\n"
                "  python train_aggressive.py\n"
                "(or trigger the Train Model GitHub Actions workflow)"
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

    # -- guard helpers ---------------------------------------------------

    @staticmethod
    def _unknown_result(display_name: str) -> dict:
        """Flat uncertain result for inputs the model cannot judge."""
        return {
            "name": display_name,
            "girl_associated_probability": _UNKNOWN_PROB[0],
            "boy_associated_probability": _UNKNOWN_PROB[1],
            "uncertain_probability": _UNKNOWN_PROB[2],
            "classification": "uncertain",
            "confidence": "low",
        }

    def _is_oov(self, normalized: str) -> bool:
        """True when most of the name's discriminative n-grams were unseen in
        training.

        Uses internal (non-padded) n-grams of each whitespace token: padded
        boundary grams appear inside almost every real name and cannot
        separate real patterns from gibberish. Latin tokens need 4+ grams
        ("fjkwo" shares "jkwo" with the real name "jkwon", but not "fjkw"
        or "fjkwo"); CJK tokens are 1-3 characters, so they fall back to
        2+ gram evidence ("甜兔" is never seen, "若汐" always is). Names
        too short to yield any such gram bypass the check.
        """
        bloom = getattr(self._clf, "bloom", None)
        if bloom is None:
            return False
        grams = []
        for token in normalized.split():
            if len(token) < 2:
                continue
            if any(ord(c) > 0x2FFF for c in token):
                min_len = 2  # CJK / kana / hangul token
            else:
                min_len = 4  # Latin token
            top = min(len(token), self._fe.max_ngram)
            for n in range(min_len, top + 1):
                for i in range(len(token) - n + 1):
                    grams.append(token[i : i + n])
        if not grams:
            return False
        return bloom.coverage(grams) < OOV_MIN_COVERAGE

    @staticmethod
    def _primary_candidate(name: str, normalized: str) -> Optional[str]:
        """Given-name candidate for multi-word names.

        Returns the normalized first word when the raw input has 2+ words
        separated by spaces AND that first word is not hyphen/apostrophe
        joined ("Jean-Luc" / "O'Brien" are single given names, not given +
        surname), and the candidate differs from the full normalized name.
        """
        words = name.strip().split()
        if len(words) < 2:
            return None
        first = words[0]
        if any(c in first for c in ("-", "'", "\u2019")):
            return None
        primary = normalize_name(first)
        if not primary or primary == normalized:
            return None
        return primary

    def _classify_candidate(self, normalized: str) -> Optional[dict]:
        """Classify an already-normalized name. Returns None when the name is
        out-of-vocabulary (no bloom -> never None)."""
        if self._is_oov(normalized):
            return None
        features = self._fe.extract(normalized).reshape(1, -1)
        class_idx, proba = self._clf.predict(features)
        return self._format_result(normalized, int(class_idx[0]), proba)

    def _decide(self, name: str) -> dict:
        """Full decision pipeline for one input name (name not yet attached)."""
        normalized = normalize_name(name)
        if not normalized:
            return self._unknown_result(name)
        if not has_supported_script(normalized):
            return self._unknown_result(name)

        result = self._classify_candidate(normalized)

        primary = self._primary_candidate(name, normalized)
        if primary is not None:
            primary_result = self._classify_candidate(primary)
            if primary_result is not None:
                if result is None:
                    # Surname/full-string pattern unseen, given name is known:
                    # fall back to the given name instead of guessing.
                    result = primary_result
                elif (
                    result["classification"] == "uncertain"
                    and result["confidence"] in ("low", "medium")
                    and primary_result["confidence"] != "low"
                ):
                    # The whole string left the model genuinely undecided, but
                    # the given name alone is decisive: use the given name.
                    # Deliberately conservative - in the training distribution
                    # two-word names are usually compound GIVEN names
                    # ("si mohamed", "alex anne"), so a confident whole-string
                    # verdict is kept even when it disagrees with the first
                    # token ("Emma Watson" full names lean on the surname; the
                    # README recommends passing the given name for those).
                    result = primary_result

        if result is None:
            # Accent-fold fallback: the merge stored many accented names under
            # their plain spelling (Mar\u00eda -> "maria"), so accented input can
            # look out-of-vocabulary while the plain form was trained. Retry
            # the diacritic-free spelling before giving up.
            folded = fold_accents(normalized)
            if folded and folded != normalized:
                folded_result = self._classify_candidate(folded)
                if folded_result is not None:
                    result = folded_result

        if result is None:
            return self._unknown_result(name)
        return result

    def predict(self, name: str) -> dict:
        """
        Predict gender association for a single name.

        Returns dict with keys: name, girl_associated_probability,
        boy_associated_probability, uncertain_probability,
        classification, confidence.

        Full names are classified as a whole (the training data's two-word
        names are compound given names, so a confident whole-string verdict
        is kept); if the whole string leaves the model undecided but the
        given name alone is decisive, the given-name result is used.
        Names in scripts never seen in training (Cyrillic, Arabic, ...),
        strings without letters, and out-of-vocabulary patterns return
        ``uncertain`` instead of a confident guess.
        """
        result = self._decide(name)
        result["name"] = name
        return result

    def predict_batch(self, names: list[str]) -> list[dict]:
        """
        Predict gender association for multiple names.

        More efficient than calling predict() in a loop
        because the model is loaded once and features are batched.

        Raises TypeError if names is not a list of strings.
        """
        if isinstance(names, str):
            raise TypeError(
                "names must be a list of strings, got a single string. "
                "Use predict_name() for one name, or pass [name]."
            )
        normalized = [normalize_name(n) for n in names]

        # Bucket inputs: simple names go through the fast CSR batch path;
        # empty / unsupported-script / OOV / given-name-override cases are
        # handled per name afterwards.
        batch_idx = []
        batch_norm = []
        per_name = []  # (index, name) handled individually

        for i, name in enumerate(names):
            n = normalized[i]
            if not n or not has_supported_script(n):
                per_name.append(i)
                continue
            # Multi-word names eligible for given-name override (or whose
            # full string may be OOV while the given name is known) are
            # decided per name for clarity.
            if self._primary_candidate(name, n) is not None:
                per_name.append(i)
                continue
            if self._is_oov(n):
                per_name.append(i)
                continue
            batch_idx.append(i)
            batch_norm.append(n)

        results: list = [None] * len(names)

        if batch_norm:
            features = self._fe.extract_batch_arrays(batch_norm)
            class_indices, probas = self._clf.predict(features)
            for j, orig_idx in enumerate(batch_idx):
                results[orig_idx] = self._format_result(
                    names[orig_idx], int(class_indices[j]), probas[j:j + 1]
                )

        for i in per_name:
            results[i] = self.predict(names[i])

        for i, name in enumerate(names):
            if results[i] is None:
                results[i] = self._unknown_result(name)

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
        name: Full name string (e.g. "Olivia", "Alex", "Isabella")
        model_path: Path to model file (optional, uses default)
        country: Optional country context (informational only)
        language: Optional language context (informational only)

    Returns:
        Dictionary with keys: name, girl_associated_probability,
        boy_associated_probability, uncertain_probability,
        classification, confidence.

    Example::

        result = predict_name("Olivia")
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
