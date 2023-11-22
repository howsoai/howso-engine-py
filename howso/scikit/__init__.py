"""The Python API for the Howso Scikit Client."""

try:
    import sklearn  # noqa
except ImportError:
    raise ImportError(
        "scikit-learn must be installed to use the howso.scikit module. Please run `pip install howso-engine[scikit]`"
    )

from .scikit import (  # noqa: F401
    ACTION,
    CLASSIFICATION,
    DEFAULT_TTL,
    FEATURE,
    HowsoClassifier,
    HowsoEstimator,
    HowsoRegressor,
    REGRESSION,
)

__all__ = [
    "HowsoEstimator",
    "HowsoRegressor",
    "HowsoClassifier",
    "CLASSIFICATION",
    "REGRESSION",
    "FEATURE",
    "ACTION",
    "DEFAULT_TTL",
]
