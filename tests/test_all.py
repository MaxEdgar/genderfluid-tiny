#!/usr/bin/env python3
"""Tests for genderfluid-tiny."""

import os
import sys
import json
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from genderfluid.preprocessing import normalize_name, extract_given_names, get_primary_name
from genderfluid.features import FeatureExtractor
from genderfluid.classifier import NameClassifier, LABELS
from genderfluid.calibration import calibration_error
from genderfluid.model_io import save_model, load_model
from genderfluid.inference import predict_name, predict_names


# ==================== PREPROCESSING TESTS ====================

class TestPreprocessing:
    def test_basic_normalization(self):
        assert normalize_name("Emma") == "emma"
        assert normalize_name("ELVA RETTA") == "elva retta"
        assert normalize_name("elva retta") == "elva retta"

    def test_extra_spaces(self):
        assert normalize_name("Elva   Retta") == "elva retta"
        assert normalize_name("  Emma  ") == "emma"

    def test_hyphens(self):
        assert normalize_name("Elva-Retta") == "elva retta"
        assert normalize_name("Mary-Jane") == "mary jane"

    def test_apostrophes(self):
        assert normalize_name("O'Brien") == "o brien"
        assert normalize_name("D'Angelo") == "d angelo"

    def test_unicode_preservation(self):
        result = normalize_name("Ñoño")
        assert "ñ" in result
        result = normalize_name("Søren")
        assert "ø" in result

    def test_empty_input(self):
        assert normalize_name("") == ""
        assert normalize_name("   ") == ""

    def test_punctuation_removal(self):
        assert normalize_name("Emma!") == "emma"
        assert normalize_name("Emma.") == "emma"
        assert normalize_name("Emma, Smith") == "emma smith"

    def test_extract_given_names(self):
        assert extract_given_names("Michelle Renatta Chan") == ["michelle", "renatta", "chan"]
        assert extract_given_names("Max") == ["max"]

    def test_get_primary_name(self):
        assert get_primary_name("Michelle Renatta Chan") == "michelle"
        assert get_primary_name("Max") == "max"
        assert get_primary_name("") == ""


# ==================== FEATURE TESTS ====================

class TestFeatures:
    def test_feature_shape(self):
        fe = FeatureExtractor(min_ngram=2, max_ngram=5, dimensions=1024)
        features = fe.extract("emma")
        assert features.shape == (1024,)

    def test_feature_normalization(self):
        fe = FeatureExtractor(min_ngram=2, max_ngram=5, dimensions=1024)
        features = fe.extract("emma")
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 1e-5

    def test_batch_extraction(self):
        fe = FeatureExtractor(min_ngram=2, max_ngram=5, dimensions=1024)
        features = fe.extract_batch(["emma", "james", "alex"])
        assert features.shape == (3, 1024)

    def test_different_names_different_features(self):
        fe = FeatureExtractor(min_ngram=2, max_ngram=5, dimensions=1024)
        f1 = fe.extract("emma")
        f2 = fe.extract("james")
        assert not np.allclose(f1, f2)

    def test_same_name_same_features(self):
        fe = FeatureExtractor(min_ngram=2, max_ngram=5, dimensions=1024)
        f1 = fe.extract("emma")
        f2 = fe.extract("emma")
        assert np.allclose(f1, f2)

    def test_config_roundtrip(self):
        fe = FeatureExtractor(min_ngram=3, max_ngram=6, dimensions=2048)
        config = fe.get_config()
        fe2 = FeatureExtractor.from_config(config)
        assert fe2.min_ngram == 3
        assert fe2.max_ngram == 6
        assert fe2.dimensions == 2048


# ==================== CLASSIFIER TESTS ====================

def _make_3class_data(n_per_class=50, dims=128):
    """Create test data with all 3 classes."""
    np.random.seed(42)
    X0 = np.random.randn(n_per_class, dims).astype(np.float32)
    X0[:, 0] += 3  # Girl feature
    X1 = np.random.randn(n_per_class, dims).astype(np.float32)
    X1[:, 1] += 3  # Boy feature
    X2 = np.random.randn(n_per_class, dims).astype(np.float32)
    X2[:, 0] += 1  # Slightly girl
    X2[:, 1] += 1  # Slightly boy

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
    return X, y


class TestClassifier:
    def test_train_predict(self):
        X, y = _make_3class_data()
        clf = NameClassifier(min_confidence=0.50)
        clf.train(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape[1] == len(LABELS)

    def test_probabilities_sum_to_one(self):
        X, y = _make_3class_data()
        clf = NameClassifier(min_confidence=0.50)
        clf.train(X, y)
        proba = clf.predict_proba(X)
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-3)

    def test_probabilities_valid_range(self):
        X, y = _make_3class_data()
        clf = NameClassifier(min_confidence=0.50)
        clf.train(X, y)
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_uncertain_classification(self):
        X, y = _make_3class_data()
        clf = NameClassifier(min_confidence=0.99)
        clf.train(X, y)
        class_idx, proba = clf.predict(X)
        uncertain_count = np.sum(class_idx == 2)
        assert uncertain_count > 0


# ==================== CALIBRATION TESTS ====================

class TestCalibration:
    def test_calibration_error_range(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
        ])
        ece = calibration_error(y_true, y_proba, n_bins=5)
        assert 0 <= ece <= 1


# ==================== MODEL I/O TESTS ====================

class TestModelIO:
    def test_save_load_roundtrip(self):
        fe = FeatureExtractor(min_ngram=2, max_ngram=4, dimensions=256)
        clf = NameClassifier(min_confidence=0.70)

        X, y = _make_3class_data(n_per_class=30, dims=256)
        clf.train(X, y)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            save_model(fe, clf, {"test": True}, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            fe2, clf2, meta = load_model(path)
            assert fe2.min_ngram == fe.min_ngram
            assert fe2.dimensions == fe.dimensions
            assert meta.get("test") is True

            # Verify predictions match
            test_name = "emma"
            f1 = fe.extract(test_name).reshape(1, -1)
            f2 = fe2.extract(test_name).reshape(1, -1)

            p1 = clf.model.predict_proba(f1)[0]
            p2 = clf2.model.predict_proba(f2)[0]
            np.testing.assert_allclose(p1, p2, atol=1e-4)
        finally:
            os.unlink(path)


# ==================== INFERENCE TESTS ====================

class TestInference:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Train and save a model for inference tests."""
        fe = FeatureExtractor(min_ngram=2, max_ngram=4, dimensions=256)
        clf = NameClassifier(min_confidence=0.70)

        X, y = _make_3class_data(n_per_class=30, dims=256)
        clf.train(X, y)

        self.model_path = tempfile.mktemp(suffix=".bin")
        save_model(fe, clf, {}, self.model_path)

        import genderfluid.inference as inf
        inf._model_cache = None
        inf._model_path_cache = None
        self._orig_default = inf.DEFAULT_MODEL_PATH
        inf.DEFAULT_MODEL_PATH = self.model_path

        yield

        inf.DEFAULT_MODEL_PATH = self._orig_default
        inf._model_cache = None
        inf._model_path_cache = None
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    def test_predict_returns_dict(self):
        result = predict_name("Emma")
        assert isinstance(result, dict)
        assert "name" in result
        assert "girl_associated_probability" in result
        assert "boy_associated_probability" in result
        assert "uncertain_probability" in result
        assert "classification" in result
        assert "confidence" in result

    def test_predict_probabilities_valid(self):
        result = predict_name("Emma")
        assert 0 <= result["girl_associated_probability"] <= 1
        assert 0 <= result["boy_associated_probability"] <= 1
        assert 0 <= result["uncertain_probability"] <= 1
        total = (result["girl_associated_probability"] +
                 result["boy_associated_probability"] +
                 result["uncertain_probability"])
        assert abs(total - 1.0) < 0.01

    def test_predict_valid_classification(self):
        result = predict_name("Emma")
        assert result["classification"] in LABELS

    def test_predict_empty_name(self):
        result = predict_name("")
        assert result["classification"] == "uncertain"

    def test_predict_unicode_name(self):
        result = predict_name("Ñoño")
        assert isinstance(result, dict)
        assert "classification" in result

    def test_predict_multiple_word_name(self):
        result = predict_name("Michelle Renatta Chan")
        assert isinstance(result, dict)
        assert "classification" in result

    def test_batch_predict(self):
        results = predict_names(["Emma", "James", "Alex"])
        assert len(results) == 3
        for r in results:
            assert "classification" in r
            total = (r["girl_associated_probability"] +
                     r["boy_associated_probability"] +
                     r["uncertain_probability"])
            assert abs(total - 1.0) < 0.01

    def test_country_context_warning(self):
        result = predict_name("Alex", country="US")
        assert "context_warning" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
