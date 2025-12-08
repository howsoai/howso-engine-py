from __future__ import annotations

from collections.abc import MutableMapping
from pprint import pformat
from typing import Any, cast, Iterator, Literal, Mapping, overload, Sequence, TypeAlias, TypedDict

import pandas as pd

from howso.utilities import deserialize_cases, format_column, format_dataframe
from howso.utilities.internals import update_caps_maps


__all__ = [
    "ReactDetails",
    "Reaction",
]

# Details with case data that need to be deserialized
DETAILS_WITH_CASE_DATA = {
    "boundary_cases",
    "boundary_values",
    "hypothetical_values",
    "influential_cases",
    "most_similar_cases",
    "outlying_feature_values",
    "predicted_values_for_case",
    "relevant_values",
}

ReactionKey: TypeAlias = Literal[
    "action",
    "details"
]


class ReactDetails(TypedDict, total=False):
    """The details returned by `react`."""

    action_features: list[str]
    """A list of feature names that correspond to the predicted action values."""

    boundary_cases: list[pd.DataFrame]
    """A list of DataFrames mapping boundary cases for each given case."""

    boundary_values: list[dict[str, list[Any]]]
    """A list of dict feature names to the boundary values computed for them for each context."""

    case_full_accuracy_contributions: list[pd.DataFrame]
    """A list of DataFrames mapping the case index and full MDA for each influential case of each given case."""

    case_full_prediction_contributions: list[pd.DataFrame]
    """
    A list of DataFrames mapping the case index and the full contribution to the action feature for each
    influential case of each given case.
    """

    case_robust_accuracy_contributions: list[pd.DataFrame]
    """A list of DataFrames mapping the case index and robust MDA for each influential case of each given case."""

    case_robust_prediction_contributions: list[pd.DataFrame]
    """
    A list of DataFrames mapping the case index and robust contribution to the action feature for each influential
    case of each given case.
    """

    categorical_action_probabilities: list[dict[str, dict[Any, float]]]
    """A list of dicts of feature names to their estimated probabilities of each class for the given cases."""

    context_features: list[str]
    """A list of feature names that correspond to the provided contexts."""

    context_values: pd.DataFrame
    """A DataFrame mapping context feature names to the provided context values."""

    derivation_parameters: pd.DataFrame
    """A DataFrame mapping the parameters used in the `react` call to their values."""

    distance_contribution: list[float]
    """A list of the computed distance contribution for each given case."""

    distance_ratio: list[float]
    """A list of the computed distance ratio for each given case."""

    distance_ratio_parts: pd.DataFrame
    """A DataFrame of the parts that are used to compute the distance ratio for each case."""

    feature_deviations: pd.DataFrame
    """
    A DataFrame of the mean absolute error of predicting each feature using the full set of context features and
    the feature being predicted as context.
    """

    feature_full_accuracy_contributions_ex_post: pd.DataFrame
    """
    A DataFrame defining the local feature full MDA of the action feature for each feature in the query given the
    prediction was already made as the given action value.
    """

    feature_full_accuracy_contributions: pd.DataFrame
    """A DataFrame defining the local feature full MDA of the action feature for each feature in the query."""

    feature_full_directional_prediction_contributions: pd.DataFrame
    """
    A DataFrame defining the local feature robust directional contributions of the action feature for each
    feature in the query.
    """

    feature_full_directional_prediction_contributions_for_case: pd.DataFrame
    """
    A DataFrame defining the local feature robust directional contributions of the action feature for each
    feature each given case.
    """

    feature_full_prediction_contributions_for_case: pd.DataFrame
    """A DataFrame mapping the case index and full contribution to the action feature each given case."""

    feature_full_prediction_contributions: pd.DataFrame
    """
    A DataFrame defining the local feature full contributions of the action feature for each feature in the
    query.
    """

    feature_full_residual_convictions_for_case: pd.DataFrame
    """A DataFrame mapping the feature name to feature full residual conviction for each given case."""

    feature_full_residuals_for_case: pd.DataFrame
    """A DataFrame mapping the feature name to the full prediction residual for each given cas."""

    feature_full_residuals: pd.DataFrame
    """A DataFrame of the local feature full residuals for each feature in the query."""

    feature_robust_accuracy_contributions_ex_post: pd.DataFrame
    """
    A DataFrame defining the local feature robust MDA of the action feature for each feature in the query given the
    prediction was already made as the given action value.
    """

    feature_robust_accuracy_contributions: pd.DataFrame
    """A DataFrame defining the local feature robust MDA of the action feature for each feature in the query."""

    feature_robust_directional_prediction_contributions: pd.DataFrame
    """
    A DataFrame defining the local feature robust directional contributions of the action feature for each feature
    in the query.
    """

    feature_robust_directional_prediction_contributions_for_case: pd.DataFrame
    """
    A DataFrame defining the local feature robust directional contributions of the action feature each given
    case.
    """

    feature_robust_prediction_contributions_for_case: pd.DataFrame
    """A DataFrame defining the local feature robust contributions of the action feature each given case."""

    feature_robust_prediction_contributions: pd.DataFrame
    """
    A DataFrame defining the local feature robust contributions of the action feature for each feature in the
    query.
    """

    feature_robust_residuals_for_case: pd.DataFrame
    """A DataFrame mapping feature name to the robust prediction residual for each given case."""

    feature_robust_residuals: pd.DataFrame
    """A DataFrame defining the local feature robust residuals for each feature in the query."""

    generate_attempts: list[float]
    """A list of the number of generation attempts taken for each synthesized case."""

    hypothetical_values: pd.DataFrame
    """
    A DataFrame mapping feature name to feature values indicating how feature values would be predicted if the given
    hypothetical values were true.
    """

    influential_cases: list[pd.DataFrame]
    """A list of DataFrames defining the influential cases for each given case."""

    most_similar_case_indices: list[pd.DataFrame]
    """A list of DataFrames defining the most similar case indices and their distance from each given case."""

    most_similar_cases: list[pd.DataFrame]
    """A list of DataFrames defining the most similar cases to each given case."""

    non_clustered_distance_contribution: list[float]
    """A list of the computed distance contribution for each given case without its cluster ID."""

    non_clustered_similarity_conviction: list[float]
    """A list of the computed similarity conviction for each given case without its cluster ID."""

    observational_errors: pd.DataFrame
    """A DataFrame of the observational errors for each feature defined in the feature attributes."""

    outlying_feature_values: list[dict[str, dict[str, Any]]]
    """
    A list of dicts mapping feature names to a description of the outlying values and the extremes observed among
    similar cases.
    """

    predicted_values_for_case: pd.DataFrame
    """A DataFrame mapping feature name to predicted value for each given case."""

    prediction_stats: dict[str, Any]
    """A dict mapping the resulting prediction stats for the region of cases nearest to each given case."""

    relevant_values: list[dict[str, list[Any]]]
    """A list of dict mapping feature name to the list of relevant values for each context."""

    similarity_conviction: list[float]
    """A list of the average similarity conviction of cases in each group."""

    # Below are time-series only

    aggregated_categorical_action_probabilities: list[pd.DataFrame]
    """
    A list of DataFrames defining the aggregated categorical action probabilities for each nominal feature across
    all of the cases of each series.
    """

    series_generate_attempts: list[float]
    """A list of generation attempts for each series as a whole."""

    series_residuals: list[pd.DataFrame]
    """A list of DataFrames of estimated uncertainties of continuous features for each time step of the series."""


class Reaction(Mapping[ReactionKey, pd.DataFrame | ReactDetails]):
    """
    An implementation of a MutableMapping to hold a collection of react outputs.

    This is useful where the results need to be aggregated together from a
    collection of single results or batched results to act as a single react output.

    Additional Reactions can be aggregated by using the `accumulate` method. This will
    coalesce the new details into the correct places within any existing reactions.

    All details with case data are deserialized to their original types, and all details
    that can be represented in DataFrame format will be converted.

    Parameters
    ----------
    action : pandas.DataFrame or list or dict, default None
        A DataFrame with columns representing the requested
        features of ``react`` or ``react_series`` cases.
    details : MutableMapping, default None
        The details of results from ``react`` or ``react_series``
        when providing a ``details`` parameter.
    attributes : MutableMapping, default none
        The feature attributes of the data.
    process_details : boolean, default True
        Set to `False` if details have already been formatted and deserialized and
        processed (for example, if details come from another `Reaction` object).
    """

    __slots__ = ("_action", "_details")

    def __init__(
        self,
        action: pd.DataFrame | list[Mapping[str, Any]],
        details: Mapping[str, Any],
        attributes: Mapping[str, Any],
        *,
        process_details: bool = True,
    ):
        """Initialize the dictionary with the allowed keys."""
        if isinstance(action, pd.DataFrame):
            self._action = action
        else:
            if "action_features" not in details:
                raise ValueError("If `action` is not a DataFrame, `action_features` must be present in `details`.")
            self._action = deserialize_cases(action, details["action_features"], attributes)
        if process_details:
            self._details = self.format_react_details(details, attributes)
        else:
            self._details = cast(ReactDetails, details)

    @overload
    def __getitem__(self, key: Literal["action"]) -> pd.DataFrame:
        """Get the action values from the Reaction."""
        ...

    @overload
    def __getitem__(self, key: Literal["details"]) -> ReactDetails:
        """Get the details from the Reaction."""
        ...

    def __getitem__(self, key: ReactionKey) -> pd.DataFrame | ReactDetails:
        """Get an item by key if the key is allowed."""
        if key == "action":
            return self._action
        elif key == "details":
            return self._details
        raise ValueError(f"Invalid key: {key}. Valid keys are 'action' or 'details'.")

    @overload
    def __setitem__(self, key: Literal["action"], value: pd.DataFrame):
        """Set the action value."""
        ...

    @overload
    def __setitem__(self, key: Literal["details"], value: ReactDetails):
        """Set the details value."""
        ...

    def __setitem__(self, key: ReactionKey, value: pd.DataFrame | ReactDetails):
        """Set an item by key if the key is allowed."""
        if key == "action":
            # If not provided a DataFrame, data is likely raw from HSE and needs to be deserialized
            if isinstance(value, pd.DataFrame):
                self._action = value
            else:
                raise ValueError("Value being set for `action` must be a Pandas DataFrame.")
        elif key == "details":
            if isinstance(value, Mapping):
                self._details = cast(ReactDetails, value)
            else:
                raise ValueError("Value being set for `details` must be a Mapping.")
        else:
            raise ValueError(f"Invalid key: {key}. Valid keys are 'action' or 'details'.")

    def __iter__(self) -> Iterator[ReactionKey]:
        """Iterate over the two valid keys."""
        return iter(ReactionKey.__args__)

    def __len__(self) -> int:
        """Get the length of this Reaction."""
        return len(ReactionKey.__args__)

    def __repr__(self) -> str:
        """Return a printable representation of this Reaction."""
        return f"{repr(self._action)}\n{pformat(self._details)}"

    @staticmethod
    def format_react_details(details: MutableMapping[str, Any], attributes: Mapping[str, Any]) -> ReactDetails:
        """
        Converts any valid details from a react call to a DataFrame and deserializes them.

        Note that some details may not be suitable for a DataFrame and will remain unchanged.

        Parameters
        ----------
        details : MutableMapping[str, Any]
            A MutableMapping (dict-like) with keys that are members of `ReactDetails`.

        Returns
        -------
        ReactDetails
            A dictionary mapping detail names to their values.

        Raises
        ------
        ValueError
            If any of the provided keys are not valid.
        """
        formatted_details = {}

        def _convert(detail_name, detail: Any) -> Any:
            """Recursively format and deserialize details."""
            # If the detail is not a list, return as-is
            if not isinstance(detail, list):
                return detail
            # List of dict --> DataFrame
            elif all(isinstance(v, dict) for v in detail):
                # Special case: categorical action probabilities
                if detail_name == "categorical_action_probabilities":
                    return update_caps_maps(detail, attributes)
                # Special case: prediction stats
                elif detail_name == "prediction_stats":
                    for group in detail:
                        if "confusion_matrix" in group.keys():
                            group["confusion_matrix"].update({
                                k: {**v, "matrix": pd.DataFrame(v["matrix"])}
                                if isinstance(v, Mapping) and "matrix" in v
                                else v
                                for k, v in group["confusion_matrix"].items()
                            })
                    return pd.DataFrame(detail).T
                # Special case: details to be deserialized
                elif detail_name in DETAILS_WITH_CASE_DATA:
                    return format_dataframe(
                        pd.DataFrame(detail),
                        features=attributes
                    )
                # All other details
                else:
                    return pd.DataFrame(detail)

            # Recurse
            return [_convert(detail_name, v) for v in detail]

        for detail_name, detail in details.items():
            # Special case: context_values
            if detail_name == "context_values":
                context_columns = details.get('context_features')
                formatted_details.update({detail_name: deserialize_cases(detail, context_columns, attributes)})
            # Special cases: non-DataFrames that need deserialization
            elif detail_name in ["relevant_values", "boundary_values"]:
                deserialized_cases = []
                for case in detail:
                    deserialized_case = {}
                    for k, v in case.items():
                        deserialized_case[k] = format_column(pd.Series(v), feature=attributes[k])
                    deserialized_cases.append(deserialized_case)
                formatted_details.update({detail_name: deserialized_cases})
            # Specail case: outlying_feature_values (leave as-is)
            elif detail_name == "outlying_feature_values":
                formatted_details.update({detail_name: detail})
            # Other valid details
            elif detail_name in ReactDetails.__annotations__.keys():
                formatted_details.update({detail_name: _convert(detail_name, detail)})
            # Unknown detail
            else:
                raise ValueError(f"Unknown Reaction detail name: {detail_name}")

        return ReactDetails(**formatted_details)

    def accumulate(self, reactions: Reaction | Sequence[Reaction]):
        """
        Merge one or more other Reaction objects into this Reaction.

        Parameters
        ----------
        reactions : list of Reaction
            One or more Reaction objects to accumulate to this Reaction.accumulate
        """
        if not isinstance(reactions, Sequence):
            reactions = [reactions]
        for reaction in reactions:
            if not isinstance(reaction, Reaction):
                raise TypeError(f"All items in `reactions` must be of type `Reaction` (found type: {type(reaction)}).")
            if reaction["action"] is not None:
                if self._action is not None:
                    self._action = pd.concat([self._action, reaction["action"]])
                else:
                    self._action = reaction["action"]

            if reaction["details"] is not None:
                for key, detail in reaction["details"].items():
                    if detail is None:
                        continue
                    elif key not in self._details:
                        self._details[key] = detail
                    elif key in ["action_features", "context_features"]:
                        # Special case: avoid duplicate entries in feature name lists
                        self._details[key] = list(set(self._details[key] + detail))
                    elif hasattr(detail, "extend") and callable(detail.extend):
                        self._details[key].extend(detail)
                    elif isinstance(detail, pd.DataFrame):
                        self._details[key] = pd.concat([self._details[key], detail])
                    else:
                        raise TypeError(
                            f"The value under the key {key} was expected to be a list (or another "
                            f"MutableSequence) or DataFrame but it is of type "
                            f"{type(self._details[key])} instead."
                        )
