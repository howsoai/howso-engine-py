from typing import (
    Any,
    Dict,
    List,
    Tuple,
    TypedDict,
)

from pandas import DataFrame

__all__ = [
    "Reaction",
    "ReactionSeries",
    "Distances"
]


class Reaction(TypedDict):
    """React response format."""

    action: DataFrame
    explanation: Dict[str, Any]


class ReactionSeries(TypedDict):
    """React Series response format."""

    series: DataFrame
    explanation: Dict[str, Any]


class Distances(TypedDict):
    """Distances response format."""

    session_indices: List[Tuple[str, int]]
    distances: DataFrame
