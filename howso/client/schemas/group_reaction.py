from collections.abc import Collection, Mapping, Iterator
from copy import deepcopy
from pprint import pformat
from typing import Any, Literal, overload, TypeAlias, TypeVar

import pandas as pd

__all__ = [
    "GroupReaction"
]

_VT = TypeVar("_VT")

ComplexMetric: TypeAlias = Literal[
    "categorical_action_probabilities",  # A dict of feature name to dict of class to probability for each group
    "influential_cases", # a list of dicts for each group
]
"""Metric output keys of react group that do not translate directly into a DataFrame alongside other metrics."""

TableMetric: TypeAlias = Literal[
    "action",
    "feature_full_residuals",
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

Metric: TypeAlias = Literal[ComplexMetric, TableMetric]
"""All metric output keys of react group."""

MetricValue: TypeAlias = pd.DataFrame | dict[str, dict]
"""The value variants of all metrics."""


class GroupReaction(Mapping[Metric, MetricValue]):
    """
    An implementation of a Mapping to contain react aggregate metric outputs.

    Parameters
    ----------
    data : Mapping, default None
        The response object of react_aggregate.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[Metric, Mapping[str, Any]] | None) -> None:
        self._data: dict[Metric, Any] = {}
        if data is not None:
            self._data.update(data)

    @overload
    def __getitem__(self, key: Literal["influential_cases"]) -> list[list[dict]]: ...
    @overload
    def __getitem__(self, key: Literal["influential_cases"]) -> list[dict[str, dict[Any, float]]]: ...
    @overload
    def __getitem__(self, key: TableMetric) -> pd.DataFrame: ...

    def __getitem__(self, key: Metric) -> Any:
        value = self._data[key]
        if key == "action":
            return pd.DataFrame(value)
        elif key in TableMetric.__args__:
            if key == "feature_full_residuals":
                return pd.DataFrame(value)
            return pd.DataFrame(value, columns=[key])
        return value

    @overload
    def get(self, key: TableMetric, /) -> pd.DataFrame | None: ...
    @overload
    def get(self, key: TableMetric, /, default: _VT) -> pd.DataFrame | _VT: ...

    def get(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: Metric, /, default: _VT | None = None
    ) -> MetricValue | _VT | None:
        return super().get(key, default=default)

    def __iter__(self) -> Iterator[Metric]:
        """Iterate over the keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        """Check if key exists."""
        return self._data.__contains__(key)

    def __eq__(self, value: object) -> bool:
        """Check object equals this object."""
        return self._data.__eq__(value)

    def __repr__(self) -> str:
        """Return printable representation."""
        if set(ComplexMetric.__args__).intersection(self._data):
            return pformat(self._data)
        return repr(self.to_dataframe())

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get the reaction as a DataFrame.

        .. NOTE::
            Complex metrics will be excluded from the returned DataFrame.
            e.g. ``confusion_matrix`` and ``feature_robust_accuracy_contributions``

        Returns
        -------
        DataFrame
            The DataFrame representation of the reaction.
        """
        data = {k: v for k, v in self._data.items() if k not in ComplexMetric.__args__}
        return pd.DataFrame(data).T.sort_index()

    def to_dict(self) -> dict[str, Any]:
        """Get a copy of the reaction as plain dictionaries."""
        return dict(deepcopy(self._data).items())
