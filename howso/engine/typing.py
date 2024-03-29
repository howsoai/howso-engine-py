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

CaseIndices: TypeAlias = Iterable[Tuple[str, int]]
GenerateNewCases: TypeAlias = Literal["always", "attempt", "no"]
Library: TypeAlias = Literal["st", "mt"]
Mode: TypeAlias = Literal["robust", "full"]
NewCaseThreshold: TypeAlias = Literal["max", "min", "most_similar"]
NormalizeMethod: TypeAlias = Literal["feature_count", "fractional", "relative"]
PathLike: TypeAlias = Union[str, bytes, os.PathLike]
Persistence: TypeAlias = Literal["allow", "always", "never"]
Precision: TypeAlias = Literal["exact", "similar"]
SeriesIDTracking: TypeAlias = Literal["fixed", "dynamic", "no"]
TabularData2D = Union[DataFrame, List[List[object]]]
TabularData3D = Union[List[DataFrame], List[List[List[object]]]]
TargetedModel: TypeAlias = Literal["single_targeted", "omni_targeted", "targetless"]
