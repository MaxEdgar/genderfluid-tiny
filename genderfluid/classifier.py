"""Logistic regression classifier with probability calibration."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from typing import Optional


LABELS = ["girl-associated", "boy-associated", "uncertain"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = 3
MIN_EXAMPLES_PER_CLASS = 5  # Enough for cv=3 plus margin


class NameClassifier:
    """
    Logistic regression classifier for name-gender association.

    Always trains with all 3 classes to ensure predict_proba returns shape (n, 3).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        min_confidence: float = 0.70,
    ):
        self.C = C
        self.max_iter = max_iter
        self.min_confidence = min_confidence
        self.model: Optional[LogisticRegression] = None
        self.calibrated: Optional[CalibratedClassifierCV] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Train the classifier with all 3 classes guaranteed."""
        rng = np.random.RandomState(42)
        unique_classes = set(y.tolist())
        missing = set(range(NUM_CLASSES)) - unique_classes

        X_train = X
        y_train = y.copy()
        sw_train = sample_weight.copy() if sample_weight is not None else None

        # Add enough synthetic examples per missing class for cv=3
        for cls in sorted(missing):
            n_pad = MIN_EXAMPLES_PER_CLASS
            X_pad = rng.randn(n_pad, X.shape[1]).astype(np.float32) * 0.001
            y_pad = np.full(n_pad, cls, dtype=int)
            X_train = np.vstack([X_train, X_pad])
            y_train = np.concatenate([y_train, y_pad])
            if sw_train is not None:
                w_pad = np.full(n_pad, 0.001)
                sw_train = np.concatenate([sw_train, w_pad])
            else:
                sw_train = np.concatenate([
                    np.ones(len(y), dtype=np.float32),
                    np.full(n_pad, 0.001, dtype=np.float32),
                ])

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42,
        )
        self.model.fit(X_train, y_train, sample_weight=sw_train)

        # Calibrate probabilities
        self.calibrated = CalibratedClassifierCV(
            self.model, cv=3, method="sigmoid"
        )
        self.calibrated.fit(X_train, y_train, sample_weight=sw_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities for each class. Always returns shape (n, 3)."""
        if self.calibrated is not None:
            proba = self.calibrated.predict_proba(X)
        elif self.model is not None:
            proba = self.model.predict_proba(X)
        else:
            raise RuntimeError("Model not trained. Call train() first.")

        # Ensure 3-column output
        if proba.shape[1] < NUM_CLASSES:
            full_proba = np.zeros((proba.shape[0], NUM_CLASSES), dtype=np.float32)
            model_classes = self.model.classes_
            for i, c in enumerate(model_classes):
                full_proba[:, c] = proba[:, i]
            for c in range(NUM_CLASSES):
                if c not in model_classes:
                    full_proba[:, c] = 1.0 / NUM_CLASSES
            row_sums = full_proba.sum(axis=1, keepdims=True)
            full_proba /= row_sums
            return full_proba

        return proba

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict class and probabilities.
        Returns (class_indices, probabilities) where probabilities shape is (n_samples, 3)
        """
        proba = self.predict_proba(X)

        max_proba = np.max(proba, axis=1)
        below_threshold = max_proba < self.min_confidence

        class_indices = np.argmax(proba, axis=1)
        class_indices[below_threshold] = LABEL_TO_IDX["uncertain"]

        return class_indices, proba

    def get_config(self) -> dict:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_config(cls, config: dict) -> "NameClassifier":
        return cls(
            C=config.get("C", 1.0),
            max_iter=config.get("max_iter", 1000),
            min_confidence=config.get("min_confidence", 0.70),
        )
