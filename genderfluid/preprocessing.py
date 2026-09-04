"""Text preprocessing for name normalization."""

import re
import unicodedata


def normalize_name(name: str) -> str:
    """
    Normalize a name for feature extraction.

    Handles:
    - Unicode normalization (NFC)
    - Case normalization
    - Whitespace normalization
    - Punctuation handling (preserves hyphens and apostrophes as separators)
    - Extra spaces

    Preserves meaningful Unicode characters (e.g., accented letters).

    Raises TypeError if name is not a string.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"name must be a string, got {type(name).__name__} "
            f"({name!r}). Pass a name string such as 'Olivia'."
        )
    if not name or not name.strip():
        return ""

    # Unicode normalize. NFKC also folds fullwidth Latin (Ａ -> a) and
    # compatibility characters that CJK input methods commonly produce,
    # while preserving meaningful hanzi, kana, and accented letters.
    name = unicodedata.normalize("NFKC", name.strip())

    # Lowercase
    name = name.lower()

    # Replace hyphens and apostrophes with spaces (they separate name parts)
    name = name.replace("-", " ").replace("'", " ").replace("'", " ")

    # Remove other punctuation except spaces
    name = re.sub(r"[^\w\s]", "", name)

    # Collapse multiple whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name


def extract_given_names(name: str) -> list[str]:
    """
    Extract individual name parts from a full name.

    The model primarily uses given-name information.
    Returns all name parts (first, middle, last).
    """
    normalized = normalize_name(name)
    if not normalized:
        return []
    return normalized.split()


def get_primary_name(full_name: str) -> str:
    """Get the primary (first) given name from a full name."""
    parts = extract_given_names(full_name)
    return parts[0] if parts else ""


# Character ranges the model was trained on: basic/extended Latin letters,
# CJK ideographs, Japanese kana, and Hangul. Accented Latin (e.g. e, o)
# is inside Latin-1 Supplement / Latin Extended and covered by the ranges
# below; the model also sees punctuation folded to spaces.
_SUPPORTED_LETTER_RANGES = (
    # basic latin (a-z after lowercasing; ranges used only for classification)
    (0x0041, 0x007A),
    # latin-1 supplement (accents, oe, eth, thorn, etc.)
    (0x00C0, 0x024F),
    # latin extended additional + phonetic-ish letters used in names
    (0x1E00, 0x1EFF),
    # CJK unified ideographs + extension A
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    # hiragana / katakana
    (0x3040, 0x30FF),
    # hangul syllables + jamo
    (0xAC00, 0xD7AF),
    (0x1100, 0x11FF),
)


def has_supported_script(name: str) -> bool:
    """Return True if the (normalized) name contains at least one character in
    a script the model was trained on.

    Guards against confidently classifying digits, symbols, or names written
    in scripts never seen in training (Cyrillic, Arabic, Greek, Devanagari,
    ...), which currently can only collide with unrelated features.
    """
    for ch in name:
        o = ord(ch)
        for lo, hi in _SUPPORTED_LETTER_RANGES:
            if lo <= o <= hi:
                return True
    return False
