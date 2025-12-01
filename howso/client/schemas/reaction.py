from __future__ import annotations

from collections.abc import MutableMapping
from pprint import pformat
from typing import Any, Literal, Mapping, Optional, overload, Sequence, TypeAlias, TypedDict

import pandas as pd

from howso.utilities import deserialize_cases, format_dataframe
from howso.utilities.internals import update_caps_maps


__all__ = [
    "Reaction"
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

    boundary_cases: list[pd.DataFrame]

    boundary_values: pd.DataFrame

    case_full_accuracy_contributions: list[pd.DataFrame]

    case_full_prediction_contributions: list[pd.DataFrame]

    case_robust_accuracy_contributions: list[pd.DataFrame]

    case_robust_prediction_contributions: list[pd.DataFrame]

    categorical_action_probabilities: list[dict[str, dict[Any, float]]]

    context_features: list[str]

    context_values: pd.DataFrame

    derivation_parameters: pd.DataFrame

    distance_contribution: list[float]

    distance_ratio: list[float]

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

    feature_full_residuals: pd.DataFrame

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

    most_similar_case_indices: list[pd.DataFrame]

    most_similar_cases: list[pd.DataFrame]

    non_clustered_distance_contribution: list[float]

    non_clustered_similarity_conviction: list[float]

    observational_errors: pd.DataFrame

    outlying_feature_values: pd.DataFrame

    predicted_values_for_case: pd.DataFrame

    prediction_stats: dict[str, Any]

    relevant_values: pd.DataFrame

    similarity_conviction: list[float]

    # Below are time-series only

    aggregated_categorical_action_probabilities: list[pd.DataFrame]

    series_generate_attempts: list[float]


class Reaction(MutableMapping[ReactionKey, ReactDetails | list[ReactDetails]]):
    """
    An implementation of a MutableMapping to hold a collection of react outputs.

    This is useful where the results need to be aggregated together from a
    collection of single results or batched results to act as a single react output.

    Additional Reactions can be aggregated by using the `add_reaction()`
    method. This will coalesce the new details into the correct places within
    any existing reactions.

    All individual action items (cases) can be returned with their
    corresponding details via the ``gen_cases`` generator. The returned pair of
    values will be returned as a Reaction.

    Parameters
    ----------
    action : pandas.DataFrame or list or dict, default None
        A DataFrame with columns representing the requested
        features of ``react`` or ``react_series`` cases.
    details : MutableMapping, default None
        The details of results from ``react`` or ``react_series``
        when providing a ``details`` parameter.
    attributes : MutableMapping, default none
        (Optional) The feature attributes of the data.
    """

    __slots__ = ("_action", "_details", "_attributes")

    def __init__(self,
                 action: pd.DataFrame | list[MutableMapping[str, Any] | pd.DataFrame] | MutableMapping[str, Any],
                 details: MutableMapping[str, Any] | list[MutableMapping[str, Any]],
                 attributes: Optional[MutableMapping[str, Any]] = None,
                 ):
        """Initialize the dictionary with the allowed keys."""
        self._attributes = attributes
        self._action = deserialize_cases(action, details["action_features"], attributes)
        self._details = self.format_react_details(details)

    @overload
    def __getitem__(self, key: Literal["action"]) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the action values from the Reaction."""
        ...

    @overload
    def __getitem__(self, key: Literal["details"]) -> ReactDetails | list[ReactDetails]:
        """Get the details from the Reaction."""
        ...

    def __getitem__(self, key: str):
        """Get an item by key if the key is allowed."""
        if key == "action":
            return self._action
        elif key == "details":
            return self._details
        raise ValueError(f"Invalid key: {key}. Valid keys are 'action' or 'details'.")

    def __setitem__(self, key: str, value: pd.DataFrame | list[Any] | MutableMapping[Any]):
        """Set an item by key if the key is allowed."""
        if key == "action":
            self._action = value
        elif key == "details":
            self._details = value
        else:
            raise ValueError(f"Invalid key: {key}. Valid keys are 'action' or 'details'.")

    def __iter__(self):
        """Iterate over the two valid keys."""
        return iter({"action": self._action, "details": self._details})

    def __len__(self):
        """Get the length of this Reaction."""
        return sum([self._action is not None, self._details is not None])

    def __delitem__(self, key: str, value: pd.DataFrame | list[Any] | MutableMapping[Any]):
        """Delete an item by key if the key is allowed."""
        if key == "action":
            self._action = None
        elif key == "details":
            self._details = None
        else:
            raise ValueError(f"Invalid key: {key}. Valid keys are 'action' or 'details'.")

    def __repr__(self) -> str:
        """Return a printable representation of this Reaction."""
        return f"{repr(self._action)}\n{pformat(self._details)}"

    def format_react_details(self, details: MutableMapping[str, Any]) -> ReactDetails:
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
                    return update_caps_maps(detail, self._attributes)
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
                        features=self._attributes
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
                formatted_details.update({detail_name: deserialize_cases(detail, context_columns, self._attributes)})
            # Other valid details
            elif detail_name in ReactDetails.__annotations__.keys():
                formatted_details.update({detail_name: _convert(detail_name, detail)})
            # Unknown detail
            else:
                raise ValueError(f"Unknown Reaction detail name: {detail_name}")

        return formatted_details

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

            if self._details is not None:
                for key, detail in reaction["details"].items():
                    if detail is None:
                        continue
                    elif key not in self._details:
                        self._details[key] = detail
                    elif hasattr(detail, "extend") and callable(detail.extend):
                        self._details[key].extend(detail)
                    elif isinstance(detail, pd.DataFrame):
                        self._details[key] = pd.concat([self._details[key], detail])
                    else:
                        raise TypeError(
                            f"The value under the key {key} was expected to be a list (or another "
                            f"MutableSequence) or DataFrame but it is of type "
                            f"{type(self._data['details'][key])} instead."
                        )
