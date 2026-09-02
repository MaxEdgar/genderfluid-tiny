"""genderfluid-tiny: Ultra-tiny name-gender association classifier."""

from genderfluid.inference import (
    predict_name,
    predict_names,
    GenderfluidModel,
)


def classify_name(name: str) -> str:
    """
    Return just the classification string.

    Returns one of: "girl-associated", "boy-associated", "uncertain"

    Example::

        from genderfluid import classify_name

        classify_name("Emma")    # "girl-associated"
        classify_name("James")   # "boy-associated"
        classify_name("Alex")    # "uncertain"
    """
    return predict_name(name)["classification"]


def is_girl_name(name: str) -> bool:
    """
    Return True if the name is classified as girl-associated.

    Example::

        from genderfluid import is_girl_name

        if is_girl_name("Emma"):
            print("girl name")
    """
    return predict_name(name)["classification"] == "girl-associated"


def is_boy_name(name: str) -> bool:
    """
    Return True if the name is classified as boy-associated.

    Example::

        from genderfluid import is_boy_name

        if is_boy_name("James"):
            print("boy name")
    """
    return predict_name(name)["classification"] == "boy-associated"


def name_probability(name: str) -> float:
    """
    Return the girl-associated probability as a float between 0 and 1.

    Example::

        from genderfluid import name_probability

        p = name_probability("Emma")  # 0.97
        p = name_probability("Alex")  # 0.27
    """
    return predict_name(name)["girl_associated_probability"]


__version__ = "1.0.1"
__all__ = [
    "predict_name",
    "predict_names",
    "GenderfluidModel",
    "classify_name",
    "is_girl_name",
    "is_boy_name",
    "name_probability",
]
