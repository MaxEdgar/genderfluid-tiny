"""Logistic regression classifier with lightweight probability calibration."""

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from typing import Optional


LABELS = ["girl-associated", "boy-associated", "uncertain"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = 3
MIN_EXAMPLES_PER_CLASS = 5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NameClassifier:
    """
    Logistic regression classifier for name-gender association.

    Uses lightweight sigmoid (Platt) calibration instead of
    CalibratedClassifierCV to keep memory usage minimal.
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
        # Per-class sigmoid calibration parameters: A, B for sigmoid(A*raw + B)
        self.calib_A: Optional[np.ndarray] = None
        self.calib_B: Optional[np.ndarray] = None

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

        for cls in sorted(missing):
            n_pad = MIN_EXAMPLES_PER_CLASS
            y_pad = np.full(n_pad, cls, dtype=int)
            if sparse.issparse(X_train):
                # Keep the matrix sparse so large training sets stay in memory.
                X_pad = sparse.csr_matrix(
                    (rng.randn(n_pad, X.shape[1]) * 0.001).astype(np.float32)
                )
                X_train = sparse.vstack([X_train, X_pad])
            else:
                X_pad = rng.randn(n_pad, X.shape[1]).astype(np.float32) * 0.001
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

        # Lightweight Platt scaling calibration on a holdout from training data
        n = len(y_train)
        cal_size = min(2000, n // 5)
        idx = rng.choice(n, cal_size, replace=False)
        X_cal, y_cal = X_train[idx], y_train[idx]

        raw_proba = self.model.predict_proba(X_cal)
        # Fit per-class sigmoid: P(y=c|raw) = sigmoid(A * logit(raw) + B)
        self.calib_A = np.ones(NUM_CLASSES, dtype=np.float32)
        self.calib_B = np.zeros(NUM_CLASSES, dtype=np.float32)

        for c in range(NUM_CLASSES):
            targets = (y_cal == c).astype(np.float32)
            if targets.sum() < 5 or (1 - targets).sum() < 5:
                continue
            raw = raw_proba[:, c].clip(1e-7, 1 - 1e-7)
            logit = np.log(raw / (1 - raw))
            # Simple linear fit: A, B via least squares
            # Platt scaling: minimize targets * log(sigmoid(A*l+B)) + (1-targets) * log(1-sigmoid(A*l+B))
            # Use a robust 2-parameter fit
            best_A, best_B = 1.0, 0.0
            best_loss = float("inf")
            for A_try in [0.5, 1.0, 1.5, 2.0]:
                for B_try in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    s = _sigmoid(A_try * logit + B_try)
                    s = s.clip(1e-7, 1 - 1e-7)
                    loss = -np.mean(targets * np.log(s) + (1 - targets) * np.log(1 - s))
                    if loss < best_loss:
                        best_loss = loss
                        best_A, best_B = A_try, B_try
            self.calib_A[c] = best_A
            self.calib_B[c] = best_B

        del X_cal, y_cal, raw_proba
        import gc
        gc.collect()

    def _calibrate_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply sigmoid calibration to raw probabilities."""
        if self.calib_A is None:
            return raw_proba

        calibrated = np.zeros_like(raw_proba)
        for c in range(raw_proba.shape[1]):
            raw = raw_proba[:, c].clip(1e-7, 1 - 1e-7)
            logit = np.log(raw / (1 - raw))
            calibrated[:, c] = _sigmoid(self.calib_A[c] * logit + self.calib_B[c])

        # Renormalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        calibrated /= row_sums
        return calibrated

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities. Always returns shape (n, 3)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        raw_proba = self.model.predict_proba(X)

        # Ensure 3-column output
        if raw_proba.shape[1] < NUM_CLASSES:
            full = np.zeros((raw_proba.shape[0], NUM_CLASSES), dtype=np.float32)
            for i, c in enumerate(self.model.classes_):
                full[:, c] = raw_proba[:, i]
            for c in range(NUM_CLASSES):
                if c not in self.model.classes_:
                    full[:, c] = 1.0 / NUM_CLASSES
            row_sums = full.sum(axis=1, keepdims=True)
            full /= np.maximum(row_sums, 1e-10)
            raw_proba = full

        return self._calibrate_proba(raw_proba.astype(np.float32))

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
