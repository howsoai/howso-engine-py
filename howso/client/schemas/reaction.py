from __future__ import annotations

from collections import abc
from functools import singledispatchmethod
from pprint import pformat
import typing as t
import warnings

import numpy as np
import pandas as pd

from howso.utilities.constants import _RENAMED_DETAIL_KEYS  # type: ignore reportPrivateUsage


__all__ = [
    "Reaction"
]


class Reaction(abc.MutableMapping):
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
        (Optional) A DataFrame with columns representing the requested
        features of ``react`` or ``react_series`` cases.

    details : list or None
        (Optional) The details of results from ``react`` or ``react_series``
        when providing a ``details`` parameter.
    """

    __slots__ = ("_data", "_reorganized_details")

    SPECIAL_KEYS = {"action_features", }
    KNOWN_KEYS = {
        "boundary_cases",
        "boundary_cases_familiarity_convictions",
        "boundary_values",
        "case_full_accuracy_contributions",
        "case_full_prediction_contributions",
        "case_robust_accuracy_contributions",
        "case_robust_prediction_contributions",
        "categorical_action_probabilities",
        "context_values",
        "derivation_parameters",
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
        "generate_attempts",
        "hypothetical_values",
        "influential_cases_familiarity_convictions",
        "influential_cases_raw_weights",
        "influential_cases",
        "most_similar_case_indices",
        "most_similar_cases",
        "observational_errors",
        "outlying_feature_values",
        "prediction_stats",
        "robust_influences",
        "similarity_conviction",
        # react_series-only details
        "aggregated_categorical_action_probabilities",
        "series_generate_attempts",
    }

    # These detail keys are deprecated, but should be treated as KNOWN_KEYs
    # during the deprecation period.
    KNOWN_KEYS |= set(_RENAMED_DETAIL_KEYS.keys())

    def __init__(self,
                 action: t.Optional[pd.DataFrame | list | dict] = None,
                 details: t.Optional[abc.MutableMapping[str, t.Any]] = None
                 ):
        """Initialize the dictionary with the allowed keys."""
        self._data = {
            'action': None,
            'details': {}
        }

        if details is None:
            details = {}

        if action is not None:
            self.add_reaction(action, details)
        elif details:
            self._data['details'] = details

        self._reorganized_details = None

    def _validate_key(self, key: str) -> str:
        """
        Raise KeyError if key is not one of the allowed keys.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        str
            The validated key.

        Raises
        ------
        KeyError
            If the given `key` is not accepted.
        """
        # These replacements are for convenience as we switch from "series" to
        # "action" and "explanation" to "details".
        if key in ("series", "explanation"):
            warnings.warn(
                "The keys 'series' and 'explanation' are deprecated and will be "
                "removed in a future release. Please use 'action' and 'details'.",
                DeprecationWarning
            )
        if key == "series":
            key = "action"
        if key == "explanation":
            key = "details"
        if key not in ('action', 'details'):
            raise KeyError(f"Invalid Key: {key}. Valid keys are 'action' or 'details'.")

        return key

    def __getitem__(self, key: str):
        """Get an item by key if the key is allowed."""
        key = self._validate_key(key)
        return self._data[key]

    def __setitem__(self, key: str, value: t.Any):
        """Set an item by key if the key is allowed."""
        key = self._validate_key(key)
        self._reorganized_details = None
        self._data[key] = value

    def __delitem__(self, key: str):
        """Delete an item by key if the key is allowed."""
        key = self._validate_key(key)
        self._reorganized_details = None
        del self._data[key]

    def __iter__(self):
        """Iterate over the keys."""
        return iter(self._data)

    def __len__(self):
        """Return the number of items."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return printable representation."""
        action = self._data.get("action")
        details = self._data.get("details")
        return f'{repr(action)}\n{pformat(details)}'

    @singledispatchmethod
    def add_reaction(self, action: pd.DataFrame,
                     details: abc.MutableMapping[str, t.Any]):
        """
        Add more data to the instance.

        The class instance is used for scenarios where the results it
        aggregates contain details as well as scenarios where the results do
        not. But, if details are ever aggregated, we expect their length to
        match the aggregated cases.

        Parameters
        ----------
        action : pandas.DataFrame, default None
            (Optional) A dataFrame with columns representing the requested
            features of ``react`` or ``react_series`` cases.

        details : Mapping or None
            (Optional) The details of results from ``react`` or
            ``react_series`` when providing a ``details`` parameter.
        """
        if self._data.get("action") is not None:
            self._data["action"] = pd.concat([self._data["action"], action])
        else:
            self._data["action"] = action

        expected_length = len(action)

        if self._data.get("details"):
            for key, cases in details.items():
                if key in self.SPECIAL_KEYS or key not in self.KNOWN_KEYS or (
                    key == "context_values" and cases is None
                ):
                    continue
                if hasattr(cases, "extend") and callable(cases.extend):
                    if self._data["details"] and len(cases) != expected_length:
                        raise TypeError(
                            f"The length of the case values, {len(cases)} "
                            f"given for the key {key} does not match the "
                            f"expected length of {expected_length}."
                        )
                    self._data["details"][key].extend(cases)
                else:
                    raise TypeError(
                        f"The value under the key {key} was expected to be a "
                        f"list (or another MutableSequence) but it is of type "
                        f"{type(self._data['details'][key])} instead."
                    )
        else:
            self._data["details"] = details

        # Invalidate the reorganized_details cache.
        self._reorganized_details = None

    @add_reaction.register
    def _(self, action: dict, details: abc.MutableMapping[str, t.Any]):
        """Add dict[list, dict] to Reaction."""
        action_df = pd.DataFrame.from_dict(action)
        return self.add_reaction(action_df, details)

    @add_reaction.register
    def _(self, action: list, details: abc.MutableMapping[str, t.Any]):
        """Add list[dict] to Reaction."""
        action_df = pd.DataFrame(action)
        return self.add_reaction(action_df, details)

    def gen_cases(self) -> t.Generator[dict, None, None]:
        """
        Yield dict containing DetailedCase items for a single case.

        The `action` value is a single row of a DataFrame not only to preserve
        the type `DataFrame`, but also the dtypes of the columns which are not
        guaranteed to be preserved if we instead returned a `series` here.
        """
        details = self.reorganized_details
        for idx in range(self._data["action"].shape[0]):
            yield {
                "action": self._data["action"][idx:idx + 1],
                "details": details[idx] if details else dict()
            }

    @property
    def reorganized_details(self):
        """
        Lazily compute re-organized details.

        See _reorganize_details() for more information.
        """
        if self._reorganized_details is None:
            self._reorganized_details = self._reorganize_details(
                self._data["details"])
        return self._reorganized_details

    @classmethod
    def _reorganize_details(cls, details: abc.MutableMapping[str, list]) -> list[dict]:
        """
        Re-organize `details` to be a list of dicts. One dict per case.

        Example input:
            {
                k1: [v11, v12, ... v1m],
                k2: [v21, v22, ... v2m],
                ...
                kn: [vn1, vn2, ... vnm]
            }

        Example output:
            [
                {k1: v11, k2: v21, ... kn: vn1},
                {k1: v12, k2: v22, ... kn: vn2},
                ...
                {k1: v1m, k2: v2m, ... kn: vnm}
            ]

        Parameters
        ----------
        details : dict of list
            The reaction details.

        Returns
        -------
        List of dicts
            One dict per case.
        """
        if isinstance(details, list):
            return details

        if details in [None, np.nan]:
            return []

        # Ensure only "known" keys are present in the details.
        if extra_keys := set(details.keys()) - cls.KNOWN_KEYS - cls.SPECIAL_KEYS:
            warnings.warn(
                f'Unrecognized detail keys found: [{", ".join(extra_keys)}] '
                f'and ignored.')
        cleaned_details = {
            k: v for k, v in details.items()
            if k in cls.KNOWN_KEYS and v
        }
        # Transform dict[list] -> list[dict]
        per_case_details = [
            dict(zip([key for key in cleaned_details.keys()], values))
            for values in zip(*cleaned_details.values())
        ]
        # Re-insert the constant, special_keys and their values.
        for case in per_case_details:
            case.update({key: details[key] for key in cls.SPECIAL_KEYS})

        return per_case_details


# Doesn't work if this is inside the `Reaction` class.
@Reaction.add_reaction.register
def _(self, reaction: "Reaction"):
    """Add another `Reaction` to Reaction."""
    return self.add_reaction(reaction.get("action"), reaction.get("details", {}))
