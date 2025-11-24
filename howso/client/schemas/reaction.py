from __future__ import annotations

from collections import abc
from functools import singledispatchmethod
from pprint import pformat
from typing import Any, get_args, Literal, TypeAlias, TypedDict, Mapping
import warnings

import numpy as np
import pandas as pd

from howso.internals import (
    deserialize_to_dataframe,
    update_caps_maps,
)
from howso.utilities import format_dataframe
from howso.utilities.constants import _RENAMED_DETAIL_KEYS  # type: ignore reportPrivateUsage


__all__ = [
    "Reaction"
]


SingleDataFrameDetail: TypeAlias = Literal[
    "derivation_parameters",
    "distance_ratio_parts",
    "feature_deviations",
    "feature_full_accuracy_contributions_ex_post",
    "feature_full_accuracy_contributions",
    "feature_full_directional_prediction_contributions",
    "feature_full_directional_prediction_contributions_for_case",
    "feature_full_prediction_contributions_for_case",
    "feature_full_prediction_contributions",
    "feature_full_residual_convictions_for_case",
    "feature_full_residuals_for_case",
    "predicted_values_for_case",
    "feature_full_residuals",
    "feature_robust_accuracy_contributions_ex_post",
    "feature_robust_accuracy_contributions",
    "feature_robust_directional_prediction_contributions",
    "feature_robust_directional_prediction_contributions_for_case",
    "feature_robust_prediction_contributions_for_case",
    "feature_robust_prediction_contributions",
    "feature_robust_residuals_for_case",
    "feature_robust_residuals",
    "hypothetical_values",
    "observational_errors",
    "outlying_feature_values",
    "relevant_values",
]

GroupedDataFrameDetail: TypeAlias = Literal[
    "boundary_cases",
    "boundary_values",
    "case_full_accuracy_contributions",
    "case_full_prediction_contributions",
    "case_robust_accuracy_contributions",
    "case_robust_prediction_contributions",
    "influential_cases",
    "most_similar_case_indices",
    "most_similar_cases",
]

OtherDetail: TypeAlias = Literal[
    "categorical_action_probabilities",
    "context_values",
    "distance_contribution",
    "generate_attempts",
    "non_clustered_distance_contribution",
    "non_clustered_similarity_conviction",
    "prediction_stats",
    "similarity_conviction",
]


class ReactDetails(TypedDict, total=False):
    """The details returned by `react`."""

    boundary_cases: list[pd.DataFrame]

    boundary_values: list[pd.DataFrame]

    case_full_accuracy_contributions: list[pd.DataFrame]

    case_full_prediction_contributions: list[pd.DataFrame]

    case_robust_accuracy_contributions: list[pd.DataFrame]

    case_robust_prediction_contributions: list[pd.DataFrame]

    categorical_action_probabilities: list[dict[str, dict[Any, float]]]
    """The categorical action probabilities for each nominal action feature for each group."""

    context_values: pd.DataFrame

    derivation_parameters: pd.DataFrame

    distance_contribution: list[float]

    distance_ratio_parts: pd.DataFrame

    feature_deviations: pd.DataFrame

    feature_full_accuracy_contributions_ex_post: pd.DataFrame

    feature_full_accuracy_contributions: pd.DataFrame

    feature_full_directional_prediction_contributions: pd.DataFrame

    feature_full_directional_prediction_contributions_for_case: pd.DataFrame

    feature_full_prediction_contributions_for_case: pd.DataFrame

    feature_full_prediction_contributions: pd.DataFrame

    feature_full_residual_convictions_for_case: pd.DataFrame

    feature_full_residuals_for_case: pd.DataFrame

    predicted_values_for_case: pd.DataFrame

    feature_full_residuals: pd.DataFrame
    """The full residuals for each action feature for each group."""

    feature_robust_accuracy_contributions_ex_post: pd.DataFrame

    feature_robust_accuracy_contributions: pd.DataFrame

    feature_robust_directional_prediction_contributions: pd.Dataframe

    feature_robust_directional_prediction_contributions_for_case: pd.DataFrame

    feature_robust_prediction_contributions_for_case: pd.DataFrame

    feature_robust_prediction_contributions: pd.DataFrame

    feature_robust_residuals_for_case: pd.DataFrame

    feature_robust_residuals: pd.DataFrame

    generate_attempts: list[float]

    hypothetical_values: pd.DataFrame

    influential_cases: list[pd.DataFrame]
    """The collection of influential cases to each group."""

    most_similar_case_indices: list[pd.DataFrame]

    most_similar_cases: list[pd.DataFrame]

    non_clustered_distance_contribution: list[float]

    non_clustered_similarity_conviction: list[float]

    observational_errors: pd.DataFrame

    outlying_feature_values: pd.DataFrame

    prediction_stats: dict[str, Any]

    relevant_values: pd.DataFrame

    similarity_conviction: list[float]

    # Below are time-series ONLY

    aggregated_categorical_action_probabilities: list[pd.DataFrame]  # TODO verify this

    series_generate_attempts: list[float]


class Reaction(abc.MutableMapping):
    """
    An implementation of a MutableMapping to hold a collection of react outputs.

    This is useful where the results need to be aggregated together from a
    collection of single results or batched results to act as a single react outpu

    Additional Reactions can be aggregated by using the `add_reaction()`
    method. This will coalesce the new details into the correct places within
    any existing reactions.

    All individual action items (cases) can be returned with their
    corresponding details via the ``gen_cases`` generator. The returned pair of
    values will be returned as a Reaction.

    Parameters
    ----------
    action : pandas.DataFrame or list or dict, default None
        (Optional) A DataFrame with columns representing the requested
        features of ``react`` or ``react_series`` cases.
    details : MutableMapping, default None
        (Optional) The details of results from ``react`` or ``react_series``
        when providing a ``details`` parameter.
    attributes : MutableMapping, default none
        (Optional) The feature attributes of the data.
    """

    __slots__ = ("_action", "_details", "_attributes")

    # TODO can these be TypeAliases instead and serve a dual purpose? (See GroupReaction)
    SPECIAL_KEYS = {"action_features", }
    KNOWN_KEYS = {  # NOTE: for react_series, these will all be the same except wrapped in an extra list (since they will change per series)
        # These are dict[list] if not otherwise specified (ex. {'action_features': ['play'], 'similarity_conviction': [1024]})
        "boundary_cases",  # list[DataFrame]
        "boundary_cases_familiarity_convictions",  # N/A
        "boundary_values",  # list[DataFrame]
        "case_full_accuracy_contributions",  # list[DataFrame]
        "case_full_prediction_contributions",  # list [DataFrame]
        "case_robust_accuracy_contributions",  # list[DataFrame]
        "case_robust_prediction_contributions",  # list[DataFrame]
        "categorical_action_probabilities",  # SPECIAL CASE: Caps Maps update
        "context_values",
        "derivation_parameters",  # DataFrame
        "distance_contribution",
        "distance_ratio_parts",
        "distance_ratio",
        "feature_deviations",
        "feature_full_accuracy_contributions_ex_post",
        "feature_full_accuracy_contributions",
        "feature_full_directional_prediction_contributions",
        "feature_full_directional_prediction_contributions_for_case",
        "feature_full_prediction_contributions_for_case",
        "feature_full_prediction_contributions",
        "feature_full_residual_convictions_for_case",
        "feature_full_residuals_for_case",
        "predicted_values_for_case",  # Insights is deserializing; do here and remove there # TODO
        "feature_full_residuals",
        "feature_robust_accuracy_contributions_ex_post",
        "feature_robust_accuracy_contributions",
        "feature_robust_directional_prediction_contributions",
        "feature_robust_directional_prediction_contributions_for_case",
        "feature_robust_prediction_contributions_for_case",
        "feature_robust_prediction_contributions",
        "feature_robust_residuals_for_case",
        "feature_robust_residuals",
        "generate_attempts",  # Just a list of numbers
        "hypothetical_values",
        "influential_cases_familiarity_convictions",
        "influential_cases_raw_weights",
        "influential_cases",  # list of DFs
        "most_similar_case_indices",
        "most_similar_cases",  # List of DFs
        "non_clustered_distance_contribution",
        "non_clustered_similarity_conviction",
        "observational_errors",
        "outlying_feature_values",
        "prediction_stats",  # Confusion matrix thing
        "relevant_values",
        "similarity_conviction",
        # react_series-only details
        "aggregated_categorical_action_probabilities",
        "series_generate_attempts",  # List of numbers (react_series ONLY)
    }

    # These detail keys are deprecated, but should be treated as KNOWN_KEYs
    # during the deprecation period.
    KNOWN_KEYS |= set(_RENAMED_DETAIL_KEYS.keys())

    def __init__(self,
                 action: Optional[pd.DataFrame | list | dict] = None,
                 details: Optional[abc.MutableMapping[str, Any]] = None,
                 attributes: Optional[abc.MutableMapping[str, Any]] = None,
                 ):
        """Initialize the dictionary with the allowed keys."""
        self._attributes = attributes
        self._action = action
        self._details = self.format_react_details(details) if details else {}

    def __getitem__(self, key: str):
        """Get an item by key if the key is allowed."""
        raise ValueError(f"Invalid key: {key}. Valid keys are 'action' or 'details'.")

    @overload
    def __getitem__(self, key: Literal["action"]) -> pd.DataFrame:
        return self._action

    @overload
    def __getitem__(self, key: Literal["details"]) -> ReactDetails:
        return self._details or {}

    def __repr__(self) -> str:
        """Return printable representation."""
        return f"{repr(self._action)}\n{pformat(self._details)}"

    def format_react_details(self, details: abc.MutableMapping[str, Any]) -> ReactDetails:
        """
        Converts any valid details from a react call to a DataFrame and deserializes them.

        Note that some details may not be suitable for a DataFrame and will remain unchanged.

        Parameters
        ----------
        details : MutableMapping[str, Any]
            A MutableMapping (dict-like) with keys that are members of `Reaction.KNOWN_KEYS`.

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
        for detail_name, detail in details.items():
            # Special case: action_features
            if detail_name == "action_features" or detail_name == "context_features":
                continue
            elif detail_name == "context_values":
                pass  # TODO
            # Special case: categorical_action_probabilities
            elif detail_name == "categorical_action_probabilities":
                formatted_details.update(
                    {detail_name: update_caps_maps(detail, self._attributes)}
                )
            # Special case: prediction_stats
            elif detail_name == "prediction_stats":
                for group in detail:
                    if "confusion_matrix" in group.keys():  # TODO make sure this actually updates
                        group["confusion_matrix"].update({
                            k: {**v, "matrix": pd.DataFrame(v["matrix"])}
                            if isinstance(v, Mapping) and "matrix" in v
                            else v
                            for k, v in detail.items()
                        })
                formatted_details.update({detail_name: detail})
            # Details that are to be a list of DataFrames
            elif detail_name in get_args(GroupedDataFrameDetail):
                # Must deserialize each collection of cases for each group
                # (List of DataFrames)
                deserialized_cases = []
                for group_cases in detail:
                    deserialized_cases.append(
                        format_dataframe(
                            pd.DataFrame(group_cases),
                            features=self._attributes
                        )
                    )
                formatted_details.update({detail_name: deserialized_cases})
            # Details that are to be a DataFrame
            elif detail_name in get_args(SingleDataFrameDetail):
                # TODO: make sure all details with case data are deserialized -- there might be more!
                formatted_details.update({detail_name: pd.DataFrame(detail)})
            # Details that are something else, and should stay as-is
            elif detail_name in get_args(OtherDetail):
                formatted_details.update({detail_name: detail})
            # Unknown detail
            else:
                raise ValueError(f"Unknown detail name: {detail_name}")

        return formatted_details


    def accumulate(self, reactions: Reaction | list[Reaction]):
        """
        Merge one or more other Reaction objects into this Reaction.

        Parameters
        ----------
        reactions : list of Reaction
            One or more Reaction objects to accumulate to this Reaction.accumulate
        """
        if isinstance(reactions, Reaction):
            reactions = [reactions]
        for reaction in reactions:
            if not isinstance(reaction, Reaction):
                raise TypeError(f"All items in `reactions` must be of type `Reaction` (found type: {type(reaction)}).")
            # Existing code below:
            if reaction["action"] is not None:
                if self._action is not None:
                    self._action = pd.concat([self._action, reaction["action"]])
                else:
                    self._action = reaction["action"]

            if self._details is not None:
                for key, detail in reaction["details"].items():
                    if key not in self._details:
                        self._details[key] = detail
                    if key in self.SPECIAL_KEYS or key not in self.KNOWN_KEYS or (
                        key == "context_values" and detail is None
                    ):
                        continue
                    if hasattr(detail, "extend") and callable(detail.extend):
                        self._details[key].extend(detail)
                    elif isinstance(detail, pd.DataFrame):
                        self._details[key] = pd.concat([self._details[key], detail])
                    else:
                        raise TypeError(
                            f"The value under the key {key} was expected to be a list (or another "
                            f"MutableSequence) or DataFrame but it is of type "
                            f"{type(self._data['details'][key])} instead."
                        )
