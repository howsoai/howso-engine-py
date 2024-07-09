from __future__ import annotations

import os
from typing import Any, Literal

from numpy import float64
from numpy.typing import NDArray
from pandas import DataFrame
from typing_extensions import Sequence, TypeAlias, TypedDict


class Cases(TypedDict):
    """Representation of a table of cases."""

    cases: list[list[Any]]
    """Matrix of row and column values."""

    features: list[str]
    """The feature column names."""


class Distances(TypedDict):
    """Representation of a case distances result."""

    case_indices: CaseIndices
    """The corresponding distances case indices."""

    distances: NDArray[float64]
    """The matrix of computed distances."""


class Evaluation(TypedDict):
    """Representation of an Evaluate result."""

    aggregated: Any
    """The aggregated evaluation output."""

    evaluated: dict[str, list[Any]]
    """A mapping of feature names to lists of values."""


CaseIndices: TypeAlias = Sequence[tuple[str, int]]
"""Sequence of ``case_indices`` tuples."""

GenerateNewCases: TypeAlias = Literal["always", "attempt", "no"]
"""Valid values for ``generate_new_cases`` parameters."""

LibraryType: TypeAlias = Literal["st", "mt"]
"""Valid values for ``library_type`` parameters."""

Mode: TypeAlias = Literal["robust", "full"]
"""Valid values for ``mode`` parameters."""

NewCaseThreshold: TypeAlias = Literal["max", "min", "most_similar"]
"""Valid values for ``new_case_threshold`` parameters."""

NormalizeMethod: TypeAlias = Literal["feature_count", "fractional", "relative"]
"""Valid values for ``normalize_method`` parameters."""

PathLike: TypeAlias = str | bytes | os.PathLike
"""Objects which can be interpreted as paths."""

Persistence: TypeAlias = Literal["allow", "always", "never"]
"""Valid values for ``persistence`` parameters."""

Precision: TypeAlias = Literal["exact", "similar"]
"""Valid values for ``precision`` parameters."""

SeriesIDTracking: TypeAlias = Literal["fixed", "dynamic", "no"]
"""Valid values for ``series_id_tracking`` parameters."""

TabularData2D: TypeAlias = DataFrame | list[list[Any]]
"""2-dimensional tabular data."""

TabularData3D: TypeAlias = list[DataFrame] | list[list[list[Any]]]
"""3-dimensional tabular (i.e., time-series) data."""

TargetedModel: TypeAlias = Literal["single_targeted", "omni_targeted", "targetless"]
"""Valid values for ``targeted_model`` parameters."""
