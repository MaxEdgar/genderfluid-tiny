"""genderfluid-tiny: Ultra-tiny name-gender association classifier."""

__version__ = "2.0.1"


def classify_name(name: str) -> str:
    """Return the classification string for a name."""
    return predict_name(name)["classification"]


def is_girl_name(name: str) -> bool:
    """Return True if the name is classified as girl-associated."""
    return predict_name(name)["classification"] == "girl-associated"


def is_boy_name(name: str) -> bool:
    """Return True if the name is classified as boy-associated."""
    return predict_name(name)["classification"] == "boy-associated"


def name_probability(name: str) -> float:
    """Return the girl-associated probability as a float between 0 and 1."""
    return predict_name(name)["girl_associated_probability"]


def predict_name(name: str, **kwargs):
    """Predict gender association for a single name."""
    from genderfluid.inference import predict_name as _predict_name
    return _predict_name(name, **kwargs)


def predict_names(names: list, **kwargs):
    """Predict gender association for multiple names."""
    from genderfluid.inference import predict_names as _predict_names
    return _predict_names(names, **kwargs)


class GenderfluidModel:
    """Name-gender association classifier."""

    def __new__(cls, *args, **kwargs):
        from genderfluid.inference import GenderfluidModel as _Model
        return _Model(*args, **kwargs)


__all__ = [
    "classify_name",
    "is_girl_name",
    "is_boy_name",
    "name_probability",
    "predict_name",
    "predict_names",
    "GenderfluidModel",
]
