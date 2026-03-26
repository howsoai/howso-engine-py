from collections.abc import Mapping, Iterator
from copy import deepcopy
from pprint import pformat
from typing import Any, Literal, overload, TypeAlias, TypeVar, get_args

import pandas as pd

from ..typing import ConfusionMatrix

__all__ = [
    "AggregateReaction"
]

_VT = TypeVar("_VT")

ComplexMetric: TypeAlias = Literal[
    # A matrix of feature name to feature name
    "feature_robust_accuracy_contributions",
    # A dict containing "features", "feature_values", and some subset of "ac_values", "pc_values", and "pc_directional_values"
    "value_robust_contributions",
    # A dict containing "features", "feature_values", and "surprisal_asymmetries"
    "value_robust_surprisal_asymmetry",
    # Features mapped to confusion matrix schemas
    "confusion_matrix",
]
"""Metric output keys of react aggregate that do not translate directly into a DataFrame alongside other metrics."""

TableMetric: TypeAlias = Literal[
    "estimated_residual_lower_bound",
    "feature_full_residuals",
    "feature_robust_residuals",
    "feature_deviations",
    "feature_full_prediction_contributions",
    "feature_full_directional_prediction_contributions",
    "feature_robust_prediction_contributions",
    "feature_robust_directional_prediction_contributions",
    "feature_full_accuracy_contributions",
    "feature_full_accuracy_contributions_permutation",
    "feature_robust_accuracy_contributions",
    "feature_robust_accuracy_contributions_permutation",
    "value_robust_contributions",
    "value_robust_surprisal_asymmetry",
    "adjusted_smape",
    "smape",
    "mae",
    "recall",
    "precision",
    "accuracy",
    "r2",
    "rmse",
    "spearman_coeff",
    "mcc",
    "missing_value_accuracy",
    "missing_information"
]
"""Metric output keys of react aggregate that can be combined together into a DataFrame."""

ResidualTypes: TypeAlias = Literal[
    "feature_full_residuals",
    "feature_robust_residuals",
    "feature_deviations",
]

FeatureContributionTypes: TypeAlias = Literal[
    "feature_full_prediction_contributions",
    "feature_full_directional_prediction_contributions",
    "feature_robust_prediction_contributions",
    "feature_robust_directional_prediction_contributions",
    "feature_full_accuracy_contributions",
    "feature_full_accuracy_contributions_permutation",
    "feature_robust_accuracy_contributions",
    "feature_robust_accuracy_contributions_permutation",
]

Metric: TypeAlias = Literal[ComplexMetric, TableMetric]
"""All metric output keys of react aggregate."""

MetricValue: TypeAlias = pd.DataFrame | dict[str, ConfusionMatrix]
"""The value variants of all metrics."""


class AggregateReaction(Mapping[Metric, MetricValue]):
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
    def __getitem__(self, key: Literal["confusion_matrix"]) -> dict[str, ConfusionMatrix]: ...
    @overload
    def __getitem__(self, key: TableMetric) -> pd.DataFrame: ...

    def __getitem__(self, key: Metric) -> Any | None:
        value = self._data[key]
        if isinstance(value, Mapping):
            if key == "confusion_matrix":
                return {
                    k: {**v, "matrix": pd.DataFrame(v["matrix"])}
                    if isinstance(v, Mapping) and "matrix" in v
                    else v
                    for k, v in value.items()
                }
            elif key == "feature_robust_accuracy_contributions":
                return pd.DataFrame(value)
            elif key == "value_robust_contributions":
                df = pd.DataFrame(data=value['feature_values'], columns=value['features'])
                # Any of these keys *could* be in value.
                if "ac_values" in value:
                    df['ac_values'] = value['ac_values']
                if "pc_values" in value:
                    df['pc_values'] = value['pc_values']
                if "pc_directional_values" in value:
                    df['pc_directional_values'] = value['pc_directional_values']
                return df
            elif key == "value_robust_surprisal_asymmetry":
                df = pd.DataFrame(data=value['feature_values'], columns=value['features'])
                # This key should always be in value
                df['surprisal_asymmetries'] = value['surprisal_asymmetries']
                return df
            return pd.DataFrame({key: value}).T
        return value

    @overload
    def get(self, key: Literal["confusion_matrix"], /) -> dict[str, ConfusionMatrix] | None: ...
    @overload
    def get(self, key: Literal["confusion_matrix"], /, default: _VT) -> dict[str, ConfusionMatrix] | _VT: ...
    @overload
    def get(self, key: TableMetric, /) -> pd.DataFrame | None: ...
    @overload
    def get(self, key: TableMetric, /, default: _VT) -> pd.DataFrame | _VT: ...

    def get(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: Metric, /, default: _VT | None = None,
    ) -> MetricValue | _VT | None:
        return super().get(key, default=default)

    def get_feature_residuals(
        self,
        null_residuals: bool = False,
    ) -> pd.DataFrame:
        """
        Get the computed feature residuals as a DataFrame.

        Parameters
        ----------
        null_residuals : bool, default False
            A flag indicating if the residuals for the nullness of features should be returned rather than
            the residual for the values of non-null cases.

        Returns
        -------
        DataFrame
            The DataFrame representation of the computed feature residuals.
        """
        data = {x: self._data[x] for x in get_args(ResidualTypes) if x in self._data}
        if null_residuals:
            def map_func(x):
                return x[1] if isinstance(x, list) else None
        else:
            def map_func(x):
                return x[0] if isinstance(x, list) else x

        return pd.DataFrame(data).map(map_func)

    def get_feature_contributions(
        self, key: FeatureContributionTypes,
        null_contributions=False,
    ) -> pd.DataFrame:
        """
        Get the computed feature contributions as a DataFrame.

        Parameters
        ----------
        null_contributions : bool, default False
            A flag indicating if the contributions for the nullness of features should be returned rather than
            the residual for the values of non-null cases.

        Returns
        -------
        DataFrame
            The DataFrame representation of the computed feature contributions.
        """
        value = self._data[key]
        if null_contributions:
            def map_func(x):
                return x[1] if isinstance(x, list) else None
        else:
            def map_func(x):
                return x[0] if isinstance(x, list) else x
        return pd.DataFrame(value).map(map_func)

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
