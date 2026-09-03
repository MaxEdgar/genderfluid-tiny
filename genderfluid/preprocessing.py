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

    # Unicode normalize
    name = unicodedata.normalize("NFC", name.strip())

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
