"""Character n-gram feature extraction with hashing."""

import numpy as np


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

    def extract(self, name: str) -> np.ndarray:
        """
        Extract feature vector for a single normalized name.

        Returns a sparse-like dense vector of shape (dimensions,).
        """
        features = np.zeros(self.dimensions, dtype=np.float32)

        for n in range(self.min_ngram, self.max_ngram + 1):
            # Pad with spaces for boundary n-grams
            padded = f"{' ' * (n - 1)}{name}{' ' * (n - 1)}"
            for i in range(len(padded) - n + 1):
                ngram = padded[i : i + n]
                idx = self._hash_ngram(ngram)
                sign = self._sign_hash(ngram)
                features[idx] += sign

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features

    def extract_batch(self, names: list[str]) -> np.ndarray:
        """Extract features for a batch of names."""
        return np.array([self.extract(name) for name in names], dtype=np.float32)

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
