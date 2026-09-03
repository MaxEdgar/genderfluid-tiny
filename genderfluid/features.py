"""Character n-gram feature extraction with hashing."""

import numpy as np
from scipy import sparse


class FeatureExtractor:
    """
    Extract character n-gram features using hashing trick.

    Uses a fixed-size feature vector via hashing to avoid large vocabularies.
    Supports unigrams through 5-grams of characters.
    """

    def __init__(
        self,
        min_ngram: int = 2,
        max_ngram: int = 5,
        dimensions: int = 4096,
        seed: int = 42,
    ):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.dimensions = dimensions
        self.seed = seed

    def _hash_ngram(self, ngram: str) -> int:
        """Hash an n-gram to a feature index using a stable hash."""
        h = 0
        for ch in ngram:
            h = h * 31 + ord(ch)
        return abs(h) % self.dimensions

    def _sign_hash(self, ngram: str) -> int:
        """Return +1 or -1 for sign hashing to reduce collisions."""
        h = 0
        for ch in ngram:
            h = h * 131 + ord(ch)
        return 1 if h % 2 == 0 else -1

    def _extract_raw(self, name: str) -> dict:
        """Return {feature_index: signed_value} for a normalized name."""
        features: dict = {}
        for n in range(self.min_ngram, self.max_ngram + 1):
            padded = f"{' ' * (n - 1)}{name}{' ' * (n - 1)}"
            for i in range(len(padded) - n + 1):
                ngram = padded[i : i + n]
                idx = self._hash_ngram(ngram)
                sign = self._sign_hash(ngram)
                features[idx] = features.get(idx, 0.0) + sign
        return features

    def extract(self, name: str) -> np.ndarray:
        """
        Extract feature vector for a single normalized name.

        Returns a dense vector of shape (dimensions,).
        """
        features = np.zeros(self.dimensions, dtype=np.float32)
        for idx, val in self._extract_raw(name).items():
            features[idx] = val

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features

    def extract_batch(self, names: list[str]) -> np.ndarray:
        """
        Extract features for a batch of names.

        Returns a scipy sparse CSR matrix of shape (len(names), dimensions)
        so that datasets with millions of names fit in memory.
        """
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for i, name in enumerate(names):
            raw = self._extract_raw(name)
            vals = list(raw.values())
            norm = np.sqrt(sum(v * v for v in vals))
            if norm <= 0:
                continue
            inv = 1.0 / norm
            for idx, val in raw.items():
                rows.append(i)
                cols.append(idx)
                data.append(val * inv)

        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(names), self.dimensions),
            dtype=np.float32,
        )

    def get_config(self) -> dict:
        """Return configuration dict."""
        return {
            "min_ngram": self.min_ngram,
            "max_ngram": self.max_ngram,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_config(cls, config: dict) -> "FeatureExtractor":
        """Create from configuration dict."""
        return cls(
            min_ngram=config.get("min_ngram", 2),
            max_ngram=config.get("max_ngram", 5),
            dimensions=config.get("dimensions", 4096),
        )
