"""Deprecated module. Import from `howso.client.schemas` instead."""
import warnings

from howso.client.schemas.reaction import Reaction

__all__ = [
    "Reaction"
]

warnings.warn(
    "The module `howso.utilities.reaction` is deprecated. Use `howso.client.schemas` instead.",
    DeprecationWarning, stacklevel=2)
