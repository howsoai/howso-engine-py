# Pylance is unaware that we are importing the requisite annotations future
# to use the 3.10+ union syntax so we disable it for this file.

# type: ignore
from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Tuple,
)

from howso.utilities.feature_attributes.base import FeatureAttributesBase

CaseIndices = Iterable[List[str | int] | Tuple[str | int]]
FeatureAttributesType = Dict[str, Dict] | FeatureAttributesBase
GenerateNewCasesType = Literal["always", "attempt", "no"]
LibraryType = Literal["st", "mt"]
Mode = Literal["robust", "full"]
NewCaseThresholdType = Literal["max", "min", "most_similar"]
NormalizeMethod = Literal["feature_count", "fractional", "relative"]
PersistenceType = Literal["allow", "always", "never"]
Precision = Literal["exact", "similar"]
SeriesIDTrackingType = Literal["fixed", "dynamic", "no"]
TargetedModelType = Literal["single_targeted", "omni_targeted", "targetless"]