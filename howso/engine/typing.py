"""Types and type aliases for objects used in Howso Engine."""

from __future__ import annotations

import os
from typing import (
    Iterable,
    List,
    Literal,
    Tuple,
    Union,
)

from pandas import DataFrame
from typing_extensions import TypeAlias

#: Type alias for case indices.
CaseIndices: TypeAlias = Iterable[Tuple[str, int]]
#: Type alias for the valid values for ``generate_new_cases`` parameters.
GenerateNewCases: TypeAlias = Literal["always", "attempt", "no"]
#: Type alias for the valid values for ``library`` parameters.
Library: TypeAlias = Literal["st", "mt"]
#: Type alias for the valid values for ``mode`` parameters.
Mode: TypeAlias = Literal["robust", "full"]
#: Type alias for the valid values for ``new_case_threshold`` parameters.
NewCaseThreshold: TypeAlias = Literal["max", "min", "most_similar"]
#: Type alias for the valid values for ``normalize_method`` parameters.
NormalizeMethod: TypeAlias = Literal["feature_count", "fractional", "relative"]
#: Type alias for objects which can be interpreted as paths.
PathLike: TypeAlias = Union[str, bytes, os.PathLike]
#: Type alias for the valid values for ``persistence`` parameters.
Persistence: TypeAlias = Literal["allow", "always", "never"]
#: Type alias for the valid values for ``precision`` parameters.
Precision: TypeAlias = Literal["exact", "similar"]
#: Type alias for the valid values for ``series_id_tracking`` parameters.
SeriesIDTracking: TypeAlias = Literal["fixed", "dynamic", "no"]
#: Type alias for 2-dimensional tabular data.
TabularData2D = Union[DataFrame, List[List[object]]]
#: Type alias for 3-dimensional tabular (i.e., time-series) data.
TabularData3D = Union[List[DataFrame], List[List[List[object]]]]
#: Type alias for the valid values for ``targeted_model`` parameters.
TargetedModel: TypeAlias = Literal["single_targeted", "omni_targeted", "targetless"]
