"""Character n-gram feature extraction with hashing."""

import numpy as np


def iter_ngram_strings(name: str, min_ngram: int, max_ngram: int):
    """Yield every character n-gram string of a normalized name, matching the
    padded windowing used by ``FeatureExtractor`` (space padding on both sides)."""
    for n in range(min_ngram, max_ngram + 1):
        padded = f"{' ' * (n - 1)}{name}{' ' * (n - 1)}"
        for i in range(len(padded) - n + 1):
            yield padded[i : i + n]


class BloomFilter:
    """Stable in-memory bloom filter over character n-grams.

    Used to detect whether a name's patterns were seen during training, so
    out-of-vocabulary gibberish can return ``uncertain`` instead of a
    confident guess. Pure Python (no numpy/scipy) so it stays off the heavy
    import path; the filter is a single big integer.

    Hash functions are two stable 64-bit hashes (same inputs always produce
    the same indexes across processes/versions -- unlike ``hash()`` on str).
    """

    def __init__(self, nbits: int, k: int = 6, seed: int = 42, bits: int = 0):
        self.nbits = int(nbits)
        self.k = int(k)
        self.seed = seed
        self._ba = bytearray((self.nbits + 7) // 8)
        if bits:
            raw = int(bits).to_bytes((self.nbits + 7) // 8, "little")
            self._ba = bytearray(raw)

    # -- hashing -------------------------------------------------------

    _MASK = 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _mix64(x: int) -> int:
        """SplitMix64 finalizer: avalanches a hash so nearby inputs map to
        unrelated 64-bit values (raw FNV hashes cluster for similar strings,
        which ruins bloom bit spreading)."""
        x &= BloomFilter._MASK
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & BloomFilter._MASK
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & BloomFilter._MASK
        x ^= x >> 31
        return x

    @classmethod
    def _hashes(cls, text: str) -> tuple:
        """Return two stable, well-mixed 64-bit hashes for a string."""
        h1 = 0xCBF29CE484222325
        h2 = 0x9E3779B97F4A7C15
        for ch in text:
            o = ord(ch)
            h1 ^= o
            h1 = (h1 * 0x100000001B3) & cls._MASK
            h2 ^= o
            h2 = (h2 * 0xBF58476D1CE4E5B9) & cls._MASK
        return cls._mix64(h1), cls._mix64(h2)

    def _indexes(self, text: str):
        h1, h2 = self._hashes(text)
        if h2 == 0:
            h2 = 1
        out = []
        for i in range(self.k):
            out.append((h1 + i * h2) % self.nbits)
        return out

    # -- mutations -----------------------------------------------------

    def add(self, text: str) -> None:
        ba = self._ba
        for idx in self._indexes(text):
            ba[idx >> 3] |= 1 << (idx & 7)

    def contains(self, text: str) -> bool:
        ba = self._ba
        for idx in self._indexes(text):
            if not (ba[idx >> 3] >> (idx & 7)) & 1:
                return False
        return True

    def coverage(self, texts) -> float:
        """Fraction of texts present in the filter (1.0 = all seen)."""
        texts = list(texts)
        if not texts:
            return 0.0
        return sum(1 for t in texts if self.contains(t)) / len(texts)

    # -- serialization -------------------------------------------------

    def to_bytes(self) -> bytes:
        return bytes(self._ba)

    @classmethod
    def from_bytes(cls, nbits: int, k: int, seed: int, raw: bytes) -> "BloomFilter":
        bf = cls(nbits=nbits, k=k, seed=seed)
        bf._ba = bytearray(raw[: (nbits + 7) // 8])
        return bf

    def get_config(self) -> dict:
        return {"nbits": self.nbits, "k": self.k, "seed": self.seed}


def build_bloom(
    names,
    min_ngram: int,
    max_ngram: int,
    nbits: int = 1 << 26,
    k: int = 6,
) -> BloomFilter:
    """Build a bloom filter over every character n-gram of normalized names.

    ``names`` must already be normalized (lowercased, spaces collapsed), as
    produced by the dataset splits. The filter lets inference detect names
    whose patterns never appeared in training.
    """
    bloom = BloomFilter(nbits=nbits, k=k)
    for name in names:
        if not name:
            continue
        for gram in iter_ngram_strings(name, min_ngram, max_ngram):
            bloom.add(gram)
    return bloom


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

    def _extract_batch_entries(self, names: list[str]) -> tuple:
        """Return (rows, cols, vals) arrays of L2-normalized nonzero entries."""
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

        return (
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(data, dtype=np.float32),
        )

    def extract_batch(self, names: list[str]) -> np.ndarray:
        """
        Extract features for a batch of names.

        Returns a scipy sparse CSR matrix of shape (len(names), dimensions)
        so that datasets with millions of names fit in memory. Used by the
        training/evaluation pipeline (scipy is imported lazily here so the
        inference path never loads it).
        """
        from scipy import sparse

        rows, cols, data = self._extract_batch_entries(names)
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(names), self.dimensions),
            dtype=np.float32,
        )

    def extract_batch_arrays(self, names: list[str]) -> tuple:
        """
        Numpy-only CSR extraction for inference: no scipy import required.

        Returns (data, cols, indptr, n_rows) matching scipy CSR semantics so
        ``NameClassifier`` can do the sparse matrix multiply with pure numpy.
        """
        rows, cols, data = self._extract_batch_entries(names)
        n_rows = len(names)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr)
        return data, cols, indptr, n_rows

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
