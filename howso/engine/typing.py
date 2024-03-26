from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Tuple

from howso.utilities.feature_attributes.base import FeatureAttributesBase

PersistenceType = Literal["allow", "always", "never"]
LibraryType = Literal["st", "mt"]
FeatureAttributesType = Dict[str, Dict] | FeatureAttributesBase
TargetedModelType = Literal["single_targeted", "omni_targeted", "targetless"]
GenerateNewCasesType = Literal["always", "attempt", "no"]
NewCaseThresholdType = Literal["max", "min", "most_similar"]
SeriesIDTrackingType = Literal["fixed", "dynamic", "no"]
CaseIndices = Iterable[List[str | int] | Tuple[str | int]]
Precision = Literal["exact", "similar"]
Mode = Literal["robust", "full"]
NormalizeMethod = Literal["feature_count", "fractional", "relative"]