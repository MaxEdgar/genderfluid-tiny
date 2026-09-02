"""genderfluid-tiny: Ultra-tiny name-gender association classifier."""

from genderfluid.inference import (
    predict_name,
    predict_names,
    GenderfluidModel,
)

__version__ = "1.0.0"
__all__ = ["predict_name", "predict_names", "GenderfluidModel"]
