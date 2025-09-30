from collections.abc import Collection, Mapping, Iterator
from copy import deepcopy
from pprint import pformat
from typing import Any, Literal, overload, TypeAlias, TypeVar

import pandas as pd

__all__ = [
    "GroupReaction"
]

_VT = TypeVar("_VT")

GroupAction: TypeAlias = Literal["action"]

GroupDetail: TypeAlias = Literal[
    "categorical_action_probabilities",  # A dict of feature name to dict of class to probability for each group
    "influential_cases", # a list of dicts for each group
    "feature_full_residuals", # a map of residuals for each action feature for each group
]
"""Metric output keys of react group that do not translate directly into a DataFrame alongside other metrics."""

GroupMetric: TypeAlias = Literal[
    "familiarity_conviction_addition",
    "familiarity_conviction_removal",
    "distance_contribution",
    "base_model_average_distance_contribution",
    "combined_model_average_distance_contribution",
    "familiarity_conviction_addition",
    "familiarity_conviction_removal",
    "kl_divergence_addition",
    "kl_divergence_removal",
    "p_value_of_addition",
    "p_value_of_removal",
    "similarity_conviction",
]
"""Metric output keys of react group that can be combined together into a DataFrame."""

GroupTableProperty: TypeAlias = Literal["action", "metrics"]
"""All output properties that can format into a dataframe."""

GroupComplexProperty: TypeAlias = Literal["details"]
"""All output properties that can format into a dataframe."""

GroupProperty: TypeAlias = Literal[GroupTableProperty, GroupComplexProperty]
"""All output properties of the group reaction."""

PropertyValue: TypeAlias = pd.DataFrame | dict[str, dict[Any, float]] | list[list[dict]]
"""The value variants of all properties."""


class GroupReaction(Mapping[GroupProperty, PropertyValue]):
    """
    An implementation of a Mapping to contain react group outputs.

    Parameters
    ----------
    data : Mapping, default None
        The response object of react_group.
    """

    __slots__ = ("_metrics", "_action", "_details")

    def __init__(self, data: Mapping[str, Any] | None) -> None:
        self._action: pd.DataFrame = pd.DataFrame()
        self._metrics: pd.DataFrame = pd.DataFrame()
        self._details: dict[GroupDetail, Any] = {}
        if data is not None:
            action_data = data.get("action", [])
            action_features = data.get("action_features", [])
            if len(action_data) and len(action_features):
                self._action = pd.DataFrame(action_data)

            computed_metrics = set(GroupMetric.__args__).intersection(data)
            if len(computed_metrics):
                self._metrics = pd.DataFrame(
                    {computed_metric: data[computed_metric] for computed_metric in computed_metrics}
                )

            computed_details = set(GroupDetail.__args__).intersection(data)
            for computed_metric in computed_details:
                self._details.update({computed_metric: data[computed_metric]})


    @overload
    def __getitem__(self, key: GroupComplexProperty) -> list[dict[str, float]] | list[dict[str, dict[Any, float]]] | list[dict[str, Any]]: ...
    @overload
    def __getitem__(self, key: GroupTableProperty) -> pd.DataFrame: ...

    def __getitem__(self, key: GroupProperty) -> Any:
        if key == "action":
            return self._action
        elif key == "details":
            return self._details
        elif key == "metrics":
            return self._metrics
        raise ValueError('Invalid key. Should be one of: "action", "details", or "metrics".')

    @overload
    def get(self, key: GroupComplexProperty, /) -> pd.DataFrame | None: ...
    @overload
    def get(self, key: GroupComplexProperty, /, default: _VT) -> pd.DataFrame | _VT: ...

    @overload
    def get(self, key: GroupTableProperty, /) -> pd.DataFrame | None: ...
    @overload
    def get(self, key: GroupTableProperty, /, default: _VT) -> pd.DataFrame | _VT: ...

    def get(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: GroupProperty, /, default: _VT | None = None
    ) -> PropertyValue | _VT | None:
        return super().get(key, default=default)

    def __iter__(self) -> Iterator[GroupProperty]:
        """Iterate over the keys."""
        return iter(GroupProperty.__args__)

    def __len__(self) -> int:
        """Return the number of items."""
        return len(GroupProperty.__args__)

    def __contains__(self, key: object) -> bool:
        """Check if key exists."""
        return GroupProperty.__args__.__contains__(key)

    def __repr__(self) -> str:
        """Return printable representation."""
        return pformat({
            "action": self._action,
            "details": self._details,
            "metrics": self._metrics
        })

    def to_dict(self) -> dict[str, Any]:
        """Get a copy of the reaction as plain dictionaries."""
        return dict(deepcopy({
            "action": self._action.to_dict(),
            "metrics": self._metrics.to_dict(),
            "details": self._details,
        }).items())
