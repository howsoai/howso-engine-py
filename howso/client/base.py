from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping, MutableMapping
import typing as t
from uuid import UUID
import warnings

import numpy as np
from pandas import DataFrame, Index

from howso.client.schemas import HowsoVersion, Project, Reaction, Session, Trainee
from howso.utilities import internals
from howso.utilities import utilities as util
from howso.utilities.features import serialize_cases
from .exceptions import HowsoError
from .typing import (
    CaseIndices,
    Cases,
    Distances,
    Evaluation,
    Mode,
    Persistence,
    Precision,
    TabularData2D,
)

if t.TYPE_CHECKING:
    from .cache import TraineeCache
    from .configuration import HowsoConfiguration


class AbstractHowsoClient(ABC):
    """The base definition of the Howso client interface."""

    configuration: "HowsoConfiguration"
    """The client configuration options."""

    ERROR_MESSAGES = {
        "missing_session": "There is currently no active session. Begin a new session to continue."
    }
    """Mapping of error code to default error message."""

    WARNING_MESSAGES = {
        "invalid_precision": (
            'Supported values for `precision` are "exact" and "similar". The operation will be completed as '
            'if the value of `precision` is "exact".')
    }
    """Mapping of warning type to default warning message."""

    SUPPORTED_PRECISION_VALUES = ["exact", "similar"]
    """Allowed values for precision."""

    @property
    @abstractmethod
    def trainee_cache(self) -> TraineeCache:
        """Return the Trainee cache."""

    @property
    @abstractmethod
    def active_session(self) -> Session | None:
        """Return the active Session."""

    @property
    @abstractmethod
    def train_initial_batch_size(self) -> int:
        """The default number of cases in the first train batch."""

    @property
    @abstractmethod
    def react_initial_batch_size(self) -> int:
        """The default number of cases in the first react batch."""

    @abstractmethod
    def _resolve_trainee(self, trainee_id: str, **kwargs) -> str:
        """
        Resolve a Trainee resource.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to resolve.

        Returns
        -------
        str
            The normalized Trainee unique identifier.
        """

    @abstractmethod
    def _auto_persist_trainee(self, trainee_id: str):
        """
        Automatically persists the Trainee if the persistence state allows.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to persist.
        """

    @abstractmethod
    def _execute(self, trainee_id: str, label: str, payload: t.Any, **kwargs) -> t.Any:
        """
        Execute a label in Howso engine.

        Parameters
        ----------
        trainee_id : str
            The entity handle of the Trainee.
        label : str
            The label to execute.
        payload : Any
            The payload to send to label.

        Returns
        -------
        Any
            The label's response.
        """

    @abstractmethod
    def _execute_sized(self, trainee_id: str, label: str, payload: t.Any, **kwargs) -> tuple[t.Any, int, int]:
        """
        Execute a label in Howso engine and return the request and response sizes.

        Parameters
        ----------
        trainee_id : str
            The entity handle of the Trainee.
        label : str
            The label to execute.
        payload : Any
            The payload to send to label.

        Returns
        -------
        Any
            The label's response.
        int
            The request payload size.
        int
            The response payload size.
        """

    @abstractmethod
    def get_version(self) -> HowsoVersion:
        """Get Howso version."""

    @abstractmethod
    def create_trainee(
        self,
        name: t.Optional[str] = None,
        features: t.Optional[Mapping[str, Mapping]] = None,
        *,
        id: t.Optional[str | UUID] = None,
        library_type: t.Optional[t.Literal["st", "mt"]] = None,
        max_wait_time: t.Optional[int | float] = None,
        metadata: t.Optional[MutableMapping[str, t.Any]] = None,
        overwrite_trainee: bool = False,
        persistence: Persistence = "allow",
        project: t.Optional[str | Project] = None,
        resources: t.Optional[Mapping[str, t.Any]] = None
    ) -> Trainee:
        """Create a Trainee in the Howso service."""

    @abstractmethod
    def update_trainee(self, trainee: Trainee) -> Trainee:
        """Update an existing trainee in the Howso service."""

    @abstractmethod
    def get_trainee(self, trainee_id: str) -> Trainee:
        """Get an existing trainee from the Howso service."""

    @abstractmethod
    def get_trainee_information(self, trainee_id: str) -> dict:
        """Get information about the trainee."""

    @abstractmethod
    def get_trainees(self, search_terms=None) -> list[dict]:
        """Return a list of all accessible trainees."""

    @abstractmethod
    def delete_trainee(self, trainee_id, file_path=None):
        """Delete a trainee in the Howso service."""

    @abstractmethod
    def copy_trainee(
        self, trainee_id, new_trainee_name=None, *,
        library_type=None,
        resources=None,
    ) -> Trainee:
        """Copy a trainee in the Howso service."""

    @abstractmethod
    def copy_subtrainee(
        self, trainee_id, new_trainee_name, *,
        target_name_path=None, target_id=None,
        source_name_path=None, source_id=None
    ):
        """Copy a subtrainee in trainee's hierarchy."""

    @abstractmethod
    def acquire_trainee_resources(self, trainee_id: str, *, max_wait_time: t.Optional[int | float] = None):
        """Acquire resources for a Trainee in the Howso service."""

    @abstractmethod
    def release_trainee_resources(self, trainee_id: str):
        """Release a Trainee's resources from the Howso service."""

    @abstractmethod
    def persist_trainee(self, trainee_id: str):
        """Persist a trainee in the Howso service."""

    def set_random_seed(self, trainee_id: str, seed: int | float | str):
        """
        Sets the random seed for the Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set the random seed for.
        seed : int or float or str
            The random seed.
            Ex: ``7998``, ``"myrandomseed"``
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print(f'Setting random seed for Trainee with id: {trainee_id}')
        self._execute(trainee_id, "set_random_seed", {"seed": seed})
        self._auto_persist_trainee(trainee_id)

    def train(
        self, trainee_id, cases, features=None, *,
        accumulate_weight_feature=None,
        batch_size=None,
        derived_features=None,
        initial_batch_size=None,
        input_is_substituted=False,
        progress_callback=None,
        series=None,
        train_weights_only=False,
        validate=True,
    ):
        """Train a trainee with sessions containing training cases."""

    def impute(
        self,
        trainee_id: str,
        features: t.Optional[Collection[str]] = None,
        features_to_impute: t.Optional[Collection[str]] = None,
        batch_size: int = 1
    ):
        """
        Impute, or fill in the missing values, for the specified features.

        If no 'features' are specified, will use all features in the trainee
        for imputation. If no 'features_to_impute' are specified, will impute
        all features specified by 'features'.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to impute.
        features : Collection of str, optional
            A list of feature names to use for imputation.
            If not specified, all features will be used imputed.
        features_to_impute : Collection of str, optional
            A list of feature names to impute.
            If not specified, features will be used (see above)
        batch_size : int, default 1
            Larger batch size will increase accuracy and decrease speed.
            Batch size indicates how many rows to fill before recomputing
            conviction.

            The default value (which is 1) should return the best accuracy but
            might be slower. Higher values should improve performance but may
            decrease accuracy of results.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")
        util.validate_list_shape(features, 1, "features", "str")
        util.validate_list_shape(features_to_impute, 1, "features_to_impute", "str")

        if self.configuration.verbose:
            print(f'Imputing Trainee with id: {trainee_id}')
        self._execute(trainee_id, "impute", {
            "features": features,
            "features_to_impute": features_to_impute,
            "session": self.active_session.id,
            "batch_size": batch_size,
        })
        self._auto_persist_trainee(trainee_id)

    def remove_cases(
        self,
        trainee_id: str,
        num_cases: int,
        *,
        case_indices: t.Optional[CaseIndices] = None,
        condition: t.Optional[Mapping] = None,
        condition_session: t.Optional[str] = None,
        distribute_weight_feature: t.Optional[str] = None,
        precision: t.Optional[Precision] = None,
    ) -> int:
        """
        Removes training cases from a Trainee.

        The training cases will be completely purged from the model and
        the model will behave as if it had never been trained with them.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to remove cases from.
        num_cases : int
            The number of cases to remove; minimum 1 case must be removed.
            Ignored if case_indices is specified.
        case_indices : Sequence of tuple of {str, int}, optional
            A list of tuples containing session ID and session training index
            for each case to be removed.
        condition : dict of str to object, optional
            The condition map to select the cases to remove that meet all the
            provided conditions. Ignored if case_indices is specified.

            .. NOTE::
                The dictionary keys are the feature name and values are one of:

                    - None
                    - A value, must match exactly.
                    - An array of two numeric values, specifying an inclusive
                      range. Only applicable to continuous and numeric ordinal
                      features.
                    - An array of string values, must match any of these values
                      exactly. Only applicable to nominal and string ordinal
                      features.

            .. TIP::
                Example 1 - Remove all values belonging to `feature_name`::

                    criteria = {"feature_name": None}

                Example 2 - Remove cases that have the value 10::

                    criteria = {"feature_name": 10}

                Example 3 - Remove cases that have a value in range [10, 20]::

                    criteria = {"feature_name": [10, 20]}

                Example 4 - Remove cases that match one of ['a', 'c', 'e']::

                    condition = {"feature_name": ['a', 'c', 'e']}

        condition_session : str, optional
            If specified, ignores the condition and operates on cases for
            the specified session id. Ignored if case_indices is specified.
        distribute_weight_feature : str, optional
            When specified, will distribute the removed cases' weights
            from this feature into their neighbors.
        precision : {"exact", "similar"}, optional
            The precision to use when moving the cases, defaults to "exact".
            Ignored if case_indices is specified.

        Returns
        -------
        int
            The number of cases removed.

        Raises
        ------
        ValueError
            If `num_cases` is not at least 1.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if num_cases < 1:
            raise ValueError('num_cases must be a value greater than 0')

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'])

        # Convert session instance to id
        if (
            isinstance(condition, dict) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Removing case(s) from Trainee with id: {trainee_id}')

        result = self._execute(trainee_id, "remove_cases", {
            "case_indices": case_indices,
            "condition": condition,
            "condition_session": condition_session,
            "precision": precision,
            "num_cases": num_cases,
            "distribute_weight_feature": distribute_weight_feature,
        })
        self._auto_persist_trainee(trainee_id)
        if not result:
            return 0
        return result.get('count', 0)

    def move_cases(
        self,
        trainee_id: str,
        num_cases: int,
        *,
        case_indices: t.Optional[CaseIndices] = None,
        condition: t.Optional[Mapping] = None,
        condition_session: t.Optional[str] = None,
        precision: t.Optional[Precision] = None,
        preserve_session_data: bool = False,
        source_id: t.Optional[str] = None,
        source_name_path: t.Optional[Collection[str]] = None,
        target_name_path: t.Optional[Collection[str]] = None,
        target_id: t.Optional[str] = None
    ) -> int:
        """
        Moves training cases from one Trainee to another in the hierarchy.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee doing the moving.
        num_cases : int
            The number of cases to move; minimum 1 case must be moved.
            Ignored if case_indices is specified.
        case_indices : Sequence of tuple of {str, int}, optional
            A list of tuples containing session ID and session training index
            for each case to be removed.
        condition : dict, optional
            The condition map to select the cases to move that meet all the
            provided conditions. Ignored if case_indices is specified.

            .. NOTE::
                The dictionary keys are the feature name and values are one of:

                    - None
                    - A value, must match exactly.
                    - An array of two numeric values, specifying an inclusive
                      range. Only applicable to continuous and numeric ordinal
                      features.
                    - An array of string values, must match any of these values
                      exactly. Only applicable to nominal and string ordinal
                      features.

            .. TIP::
                Example 1 - Move all values belonging to `feature_name`::

                    criteria = {"feature_name": None}

                Example 2 - Move cases that have the value 10::

                    criteria = {"feature_name": 10}

                Example 3 - Move cases that have a value in range [10, 20]::

                    criteria = {"feature_name": [10, 20]}

                Example 4 - Remove cases that match one of ['a', 'c', 'e']::

                    condition = {"feature_name": ['a', 'c', 'e']}

                Example 5 - Move cases using session name and index::

                    criteria = {'.session':'your_session_name',
                                '.session_index': 1}

        condition_session : str, optional
            If specified, ignores the condition and operates on cases for
            the specified session id. Ignored if case_indices is specified.
        precision : {"exact", "similar"}, optional
            The precision to use when moving the cases. Options are 'exact'
            or 'similar'. If not specified, "exact" will be used.
            Ignored if case_indices is specified.
        preserve_session_data : bool, default False
            When True, will move cases without cleaning up session data.
        source_id : str, optional
            The source trainee unique id from which to move cases. Ignored
            if source_name_path is specified. If neither source_name_path nor
            source_id are specified, moves cases from the trainee itself.
        source_name_path : list of str, optional
            List of strings specifying the user-friendly path of the child
            subtrainee from which to move cases.
        target_name_path : list of str, optional
            List of strings specifying the user-friendly path of the child
            subtrainee to move cases to.
        target_id : str, optional
            The target trainee id to move the cases to. Ignored if
            target_name_path is specified. If neither target_name_path nor
            target_id are specified, moves cases to the trainee itself.

        Returns
        -------
        int
            The number of cases moved.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if num_cases < 1:
            raise ValueError('num_cases must be a value greater than 0')

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'])

        # Convert session instance to id
        if (
            isinstance(condition, dict) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Moving case(s) from Trainee with id: {trainee_id}')

        result = self._execute(trainee_id, "move_cases", {
            "target_id": target_id,
            "case_indices": case_indices,
            "condition": condition,
            "condition_session": condition_session,
            "precision": precision,
            "num_cases": num_cases,
            "preserve_session_data": preserve_session_data,
            "session": self.active_session.id,
            "source_id": source_id,
            "source_name_path": source_name_path,
            "target_name_path": target_name_path
        })
        self._auto_persist_trainee(trainee_id)
        if not result:
            return 0
        return result.get('count', 0)

    def edit_cases(
        self,
        trainee_id: str,
        feature_values: Iterable[t.Any] | DataFrame,
        *,
        case_indices: t.Optional[CaseIndices] = None,
        condition: t.Optional[Mapping] = None,
        condition_session: t.Optional[str] = None,
        features: t.Optional[Collection[str]] = None,
        num_cases: t.Optional[int] = None,
        precision: t.Optional[Precision] = None,
    ) -> int:
        """
        Edit feature values for the specified cases.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to edit the cases of.
        feature_values : list of object or pandas.DataFrame
            The feature values to edit the case(s) with. If specified as a list,
            the order corresponds with the order of the `features` parameter.
            If specified as a DataFrame, only the first row will be used.
        case_indices : Sequence of tuple of {str, int}, optional
            Iterable of Sequences containing the session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. This explicitly specifies the cases to edit. When
            specified, `condition` and `condition_session` are ignored.
        condition : dict, optional
            A condition map to select which cases to edit. Ignored when
            `case_indices` are specified.

            .. NOTE::
                The dictionary keys are the feature name and values are one of:

                    - None
                    - A value, must match exactly.
                    - An array of two numeric values, specifying an inclusive
                      range. Only applicable to continuous and numeric ordinal
                      features.
                    - An array of string values, must match any of these values
                      exactly. Only applicable to nominal and string ordinal
                      features.

        condition_session : str, optional
            If specified, ignores the condition and operates on all cases for
            the specified session.
        features : iterable of str, optional
            The names of the features to edit. Required when `feature_values`
            is not specified as a DataFrame.
        num_cases : int, default None
            The maximum amount of cases to edit. If not specified, the limit
            will be k cases if precision is "similar", or no limit if precision
            is "exact".
        precision : {"exact", "similar"}, optional
            The precision to use when moving the cases, defaults to "exact".

        Returns
        -------
        int
            The number of cases modified.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'])

        if case_indices is not None:
            util.validate_case_indices(case_indices)

        cached_trainee = self.trainee_cache.get(trainee_id)

        # Serialize feature_values
        serialized_feature_values = None
        if feature_values is not None:
            if features is None:
                features = internals.get_features_from_data(feature_values, data_parameter='feature_values')
            serialized_feature_values = serialize_cases(feature_values, features, cached_trainee.features)
            if serialized_feature_values:
                # Only a single case should be provided
                serialized_feature_values = serialized_feature_values[0]

        # Convert session instance to id
        if (
            isinstance(condition, dict) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Editing case(s) in Trainee with id: {trainee_id}')

        result = self._execute(trainee_id, "edit_cases", {
            "case_indices": case_indices,
            "condition": condition,
            "condition_session": condition_session,
            "features": features,
            "feature_values": serialized_feature_values,
            "precision": precision,
            "num_cases": num_cases,
            "session": self.active_session.id,
        })
        self._auto_persist_trainee(trainee_id)
        if not result:
            return 0
        return result.get('count', 0)

    @abstractmethod
    def remove_series_store(self, trainee_id, series=None):
        """Clear stored series from trainee."""

    @abstractmethod
    def append_to_series_store(
        self,
        trainee_id,
        series,
        contexts,
        *,
        context_features=None
    ):
        """Append the specified contexts to a series store."""

    def set_substitute_feature_values(self, trainee_id: str, substitution_value_map: Mapping):
        """
        Set a Trainee's substitution map for use in extended nominal generation.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set substitute feature values for.
        substitution_value_map : dict
            A dictionary of feature name to a dictionary of feature value to
            substitute feature value.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print(f'Setting substitute feature values for Trainee with id: {trainee_id}')
        self._execute(trainee_id, "set_substitute_feature_values", {
            "substitution_value_map": substitution_value_map
        })
        self._auto_persist_trainee(trainee_id)

    def get_substitute_feature_values(self, trainee_id: str, clear_on_get: bool = True) -> dict:
        """
        Gets a substitution map for use in extended nominal generation.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to get the substitution feature values from.
        clear_on_get : bool, default True
            Clears the substitution values map in the Trainee upon retrieving
            them. This is done if it is desired to prevent the substitution map
            from being persisted. If set to False the model will not be cleared
            which preserves substitution mappings if the model is saved;
            representing a potential privacy leak should the substitution map
            be made public.

        Returns
        -------
        dict of dict
            A dictionary of feature name to a dictionary of feature value to
            substitute feature value.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print(f'Getting substitute feature values from Trainee with id: {trainee_id}')
        result = self._execute(trainee_id, "get_substitute_feature_values", {})
        if clear_on_get:
            self._execute(trainee_id, "set_substitute_feature_values", {
                "substitution_value_map": {}
            })
            self._auto_persist_trainee(trainee_id)
        if result is None:
            return dict()
        return result

    def set_feature_attributes(self, trainee_id: str, feature_attributes: dict[str, dict]):
        """
        Sets feature attributes for a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        feature_attributes : dict of str to dict
            A dict of dicts of feature attributes. Each key is the feature
            'name' and each value is a dict of feature-specific parameters.

            Example::

                {
                    "length": { "type" : "continuous", "decimal_places": 1 },
                    "width": { "type" : "continuous", "significant_digits": 4 },
                    "degrees": { "type" : "continuous", "cycle_length": 360 },
                    "class": { "type" : "nominal" }
                }
        """
        self._resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        if not isinstance(feature_attributes, dict):
            raise ValueError("`feature_attributes` must be a dict")
        if self.configuration.verbose:
            print(f'Setting feature attributes for Trainee with id: {trainee_id}')

        self._execute(trainee_id, "set_feature_attributes", {
            "feature_attributes": internals.preprocess_feature_attributes(feature_attributes),
        })
        self._auto_persist_trainee(trainee_id)

        # Update trainee in cache
        updated_feature_attributes = self._execute(trainee_id, "get_feature_attributes", {})
        cached_trainee.features = internals.postprocess_feature_attributes(updated_feature_attributes)

    def get_feature_attributes(self, trainee_id: str) -> dict[str, dict]:
        """
        Get stored feature attributes.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.

        Returns
        -------
        dict
            A dictionary of feature name to dictionary of feature attributes.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print('Getting feature attributes from Trainee with id: {trainee_id}')
        feature_attributes = self._execute(trainee_id, "get_feature_attributes", {})
        return internals.postprocess_feature_attributes(feature_attributes)

    @abstractmethod
    def get_sessions(self, search_terms=None) -> list[Session]:
        """Get list of all accessible sessions."""

    @abstractmethod
    def get_session(self, session_id) -> Session:
        """Get session details."""

    @abstractmethod
    def update_session(self, session_id, *, metadata=None) -> Session:
        """Update a session."""

    @abstractmethod
    def begin_session(self, name='default', metadata=None) -> Session:
        """Begin a new session."""

    @abstractmethod
    def get_trainee_sessions(self, trainee_id) -> list[dict[str, str]]:
        """Get the session ids of a trainee."""

    @abstractmethod
    def delete_trainee_session(self, trainee_id, session):
        """Delete a session from a trainee."""

    @abstractmethod
    def get_trainee_session_indices(self, trainee_id, session) -> Index | list[int]:
        """Get list of all session indices for a specified session."""

    @abstractmethod
    def get_trainee_session_training_indices(self, trainee_id, session) -> Index | list[int]:
        """Get list of all session training indices for a specified session."""

    @abstractmethod
    def get_hierarchy(self, trainee_id) -> dict:
        """Output the hierarchy for a trainee."""

    @abstractmethod
    def rename_subtrainee(
        self,
        trainee_id,
        new_name,
        *,
        child_name_path=None,
        child_id=None
    ) -> None:
        """Renames a contained child trainee in the hierarchy."""

    def get_marginal_stats(
        self,
        trainee_id: str,
        *,
        condition: t.Optional[Mapping] = None,
        num_cases: t.Optional[int] = None,
        precision: t.Optional[Precision] = None,
        weight_feature: t.Optional[str] = None
    ) -> dict[str, dict[str, float]]:
        """
        Get marginal stats for all features.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to retrieve marginal stats for.
        condition : dict or None, optional
            A condition map to select which cases to compute marginal stats
            for.

            .. NOTE::
                The dictionary keys are the feature name and values are one of:

                    - None
                    - A value, must match exactly.
                    - An array of two numeric values, specifying an inclusive
                      range. Only applicable to continuous and numeric ordinal
                      features.
                    - An array of string values, must match any of these values
                      exactly. Only applicable to nominal and string ordinal
                      features.
        num_cases : int, default None
            The maximum amount of cases to use to calculate marginal stats.
            If not specified, the limit will be k cases if precision is
            "similar". Only used if `condition` is not None.
        precision : str, default None
            The precision to use when selecting cases with the condition.
            Options are 'exact' or 'similar'. If not specified "exact" will be
            used. Only used if `condition` is not None.
        weight_feature : str, optional
            When specified, will attempt to return stats that were computed
            using this weight_feature.

        Returns
        -------
        dict of str to dict of str to float
            A map of feature names to map of stat type to stat values.
        """
        trainee_id = self._resolve_trainee(trainee_id)

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'])

        if self.configuration.verbose:
            print(f'Getting feature marginal stats for trainee with id: {trainee_id}')

        stats = self._execute(trainee_id, "get_marginal_stats", {
            "condition": condition,
            "num_cases": num_cases,
            "precision": precision,
            "weight_feature": weight_feature,
        })
        if stats is None:
            return dict()
        return stats

    @abstractmethod
    def react_series(
        self, trainee_id, *,
        action_features=None,
        actions=None,
        batch_size=None,
        case_indices=None,
        contexts=None,
        context_features=None,
        continue_series=False,
        continue_series_features=None,
        continue_series_values=None,
        derived_action_features=None,
        derived_context_features=None,
        desired_conviction=None,
        details=None,
        exclude_novel_nominals_from_uniqueness_check=False,
        feature_bounds_map=None,
        final_time_steps=None,
        generate_new_cases="no",
        init_time_steps=None,
        initial_batch_size=None,
        initial_features=None,
        initial_values=None,
        input_is_substituted=False,
        leave_case_out=None,
        max_series_lengths=None,
        new_case_threshold="min",
        num_series_to_generate=1,
        ordered_by_specified_features=False,
        output_new_series_ids=True,
        preserve_feature_values=None,
        progress_callback=None,
        series_context_features=None,
        series_context_values=None,
        series_id_tracking="fixed",
        series_stop_maps=None,
        series_index=None,
        substitute_output=True,
        suppress_warning=False,
        use_case_weights=False,
        use_regional_model_residuals=True,
        weight_feature=None
    ) -> Reaction:
        """React in a series until a stop condition is met."""

    @abstractmethod
    def react_into_features(
        self, trainee_id, *,
        distance_contribution: bool | str = False,
        familiarity_conviction_addition: bool | str = False,
        familiarity_conviction_removal: bool | str = False,
        features=None,
        influence_weight_entropy: bool | str = False,
        p_value_of_addition: bool | str = False,
        p_value_of_removal: bool | str = False,
        similarity_conviction: bool | str = False,
        use_case_weights: bool | str = False,
        weight_feature=None
    ):
        """Calculate conviction and other data for the specified feature(s)."""

    @abstractmethod
    def react_aggregate(
        self, trainee_id, *,
        action_feature=None,
        confusion_matrix_min_count=None,
        context_features=None,
        details=None,
        feature_influences_action_feature=None,
        hyperparameter_param_path=None,
        num_samples=None,
        num_robust_influence_samples=None,
        num_robust_residual_samples=None,
        num_robust_influence_samples_per_case=None,
        prediction_stats_action_feature=None,
        residuals_hyperparameter_feature=None,
        robust_hyperparameters=None,
        sample_model_fraction=None,
        sub_model_size=None,
        use_case_weights=None,
        weight_feature=None,
    ) -> DataFrame | dict:
        """Computes, caches, and/or returns specified feature interpretations."""

    @abstractmethod
    def react_group(
        self, trainee_id, new_cases, *,
        distance_contributions=False,
        familiarity_conviction_addition=True,
        familiarity_conviction_removal=False,
        features=None,
        kl_divergence_addition=False,
        kl_divergence_removal=False,
        p_value_of_addition=False,
        p_value_of_removal=False,
        use_case_weights=False,
        weight_feature=None
    ) -> DataFrame | dict:
        """Compute specified data for a **set** of cases."""

    @abstractmethod
    def react(
        self, trainee_id, *,
        action_features=None,
        actions=None,
        allow_nulls=False,
        batch_size=None,
        case_indices=None,
        contexts=None,
        context_features=None,
        derived_action_features=None,
        derived_context_features=None,
        desired_conviction=None,
        details=None,
        exclude_novel_nominals_from_uniqueness_check=False,
        feature_bounds_map=None,
        generate_new_cases="no",
        initial_batch_size=None,
        input_is_substituted=False,
        into_series_store=None,
        leave_case_out=None,
        new_case_threshold="min",
        num_cases_to_generate=1,
        ordered_by_specified_features=False,
        post_process_features=None,
        post_process_values=None,
        preserve_feature_values=None,
        progress_callback=None,
        substitute_output=True,
        suppress_warning=False,
        use_case_weights=False,
        use_regional_model_residuals=True,
        weight_feature=None
    ) -> Reaction:
        """Send a `react` to the Howso engine."""

    def evaluate(
        self,
        trainee_id: str,
        features_to_code_map: Mapping[str, str],
        *,
        aggregation_code: t.Optional[str] = None
    ) -> Evaluation:
        """
        Evaluate custom code on feature values of all cases in the trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        features_to_code_map : dict of str to str
            A dictionary with feature name keys and custom Amalgam code string values.

            The custom code can use "#feature_name 0" to reference the value
            of that feature for each case.
        aggregation_code : str, optional
            A string of custom Amalgam code that can access the list of values
            derived form the custom code in features_to_code_map.
            The custom code can use "#feature_name 0" to reference the list of
            values derived from using the custom code in features_to_code_map.

        Returns
        -------
        dict
            A dictionary with keys: 'evaluated' and 'aggregated'

            'evaluated' is a dictionary with feature name
            keys and lists of values derived from the features_to_code_map
            custom code.

            'aggregated' is None if no aggregation_code is given, it otherwise
            holds the output of the custom 'aggregation_code'
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print(f'Evaluating on Trainee with id: {trainee_id}')
        return self._execute(trainee_id, "evaluate", {
            "features_to_code_map": features_to_code_map,
            "aggregation_code": aggregation_code
        })

    @abstractmethod
    def analyze(
        self,
        trainee_id,
        context_features=None,
        action_features=None,
        *,
        bypass_calculate_feature_residuals=None,
        bypass_calculate_feature_weights=None,
        bypass_hyperparameter_analysis=None,
        dt_values=None,
        inverse_residuals_as_weights=None,
        k_folds=None,
        k_values=None,
        num_analysis_samples=None,
        num_samples=None,
        analysis_sub_model_size=None,
        p_values=None,
        targeted_model=None,
        use_case_weights=None,
        use_deviations=None,
        weight_feature=None,
        **kwargs
    ):
        """Analyzes a trainee."""

    @abstractmethod
    def auto_analyze(self, trainee_id):
        """Auto-analyze the trainee model."""

    @abstractmethod
    def get_auto_ablation_params(
        self,
        trainee_id
    ):
        """Get trainee parameters for auto-ablation set by :meth:`set_auto_ablation_params`."""

    @abstractmethod
    def set_auto_ablation_params(
        self,
        trainee_id,
        auto_ablation_enabled=False,
        *,
        auto_ablation_weight_feature=".case_weight",
        conviction_lower_threshold=None,
        conviction_upper_threshold=None,
        exact_prediction_features=None,
        influence_weight_entropy_threshold=0.6,
        minimum_model_size=1_000,
        relative_prediction_threshold_map=None,
        residual_prediction_features=None,
        tolerance_prediction_threshold_map=None,
        **kwargs
    ):
        """Set trainee parameters for auto-ablation."""

    @abstractmethod
    def reduce_data(
        self,
        trainee_id,
        features=None,
        distribute_weight_feature=None,
        influence_weight_entropy_threshold=None,
        skip_auto_analyze=False,
        **kwargs
    ):
        """Smartly reduce the amount of trained cases while accumulating case weights."""

    @abstractmethod
    def set_auto_analyze_params(
        self,
        trainee_id,
        auto_analyze_enabled=False,
        analyze_threshold=None,
        *,
        auto_analyze_limit_size=None,
        analyze_growth_factor=None,
        **kwargs
    ):
        """Set trainee parameters for auto analysis."""

    def get_cases(
        self,
        trainee_id: str,
        session: t.Optional[str] = None,
        case_indices: t.Optional[CaseIndices] = None,
        indicate_imputed: bool = False,
        features: t.Optional[Collection[str]] = None,
        condition: t.Optional[Mapping] = None,
        num_cases: t.Optional[int] = None,
        precision: t.Optional[t.Literal["exact", "similar"]] = None
    ) -> Cases:
        """
        Retrieve cases from a model given a Trainee id.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee retrieve cases from.
        session : str, optional
            The session ID to retrieve cases for, in their trained order.

            .. NOTE::
                If a session is not provided, retrieves all feature values
                for cases for all (unordered) sessions in the order they
                were trained within each session.

        case_indices : Sequence of tuple of {str, int}, optional
            Iterable of Sequences, of session id and index, where index is the
            original 0-based index of the case as it was trained into the
            session. If specified, returns only these cases and ignores the
            session parameter.
        indicate_imputed : bool, default False
            If set, an additional value will be appended to the cases
            indicating if the case was imputed.
        features : Collection of str, optional
            A list of feature names to return values for in leu of all
            default features.

            Built-in features that are available for retrieval:

                | **.session** - The session id the case was trained under.
                | **.session_training_index** - 0-based original index of the
                  case, ordered by training during the session; is never
                  changed.
        condition : dict, optional
            The condition map to select the cases to retrieve that meet all the
            provided conditions.

            .. NOTE::
                The dictionary keys are the feature name and values are one of:

                    - None
                    - A value, must match exactly.
                    - An array of two numeric values, specifying an inclusive
                      range. Only applicable to continuous and numeric ordinal
                      features.
                    - An array of string values, must match any of these values
                      exactly. Only applicable to nominal and string ordinal
                      features.

            .. TIP::
                Example 1 - Retrieve all values belonging to `feature_name`::

                    criteria = {"feature_name": None}

                Example 2 - Retrieve cases that have the value 10::

                    criteria = {"feature_name": 10}

                Example 3 - Retrieve cases that have a value in range [10, 20]::

                    criteria = {"feature_name": [10, 20]}

                Example 4 - Retrieve cases that match one of ['a', 'c', 'e']::

                    condition = {"feature_name": ['a', 'c', 'e']}

                Example 5 - Retrieve cases using session name and index::

                    criteria = {'.session':'your_session_name',
                                '.session_training_index': 1}

        num_cases : int, default None
            The maximum amount of cases to retrieve. If not specified, the limit
            will be k cases if precision is "similar", or no limit if precision
            is "exact".
        precision : {"exact", "similar}, optional
            The precision to use when retrieving the cases via condition.
            Options are "exact" or "similar". If not provided, "exact" will
            be used.

        Returns
        -------
        dict
            A cases object containing the feature names and cases.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if case_indices is not None:
            util.validate_case_indices(case_indices)

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'])

        util.validate_list_shape(features, 1, "features", "str")
        if session is None and case_indices is None:
            warnings.warn("Calling get_cases without a session id does not guarantee case order.")
        if self.configuration.verbose:
            print(f'Retrieving cases for Trainee with id {trainee_id}.')

        result = self._execute(trainee_id, "get_cases", {
            "features": features,
            "session": session,
            "case_indices": case_indices,
            "indicate_imputed": indicate_imputed,
            "condition": condition,
            "num_cases": num_cases,
            "precision": precision,
        })
        if result is None:
            result = dict()
        return Cases(
            features=result.get('features') or [],
            cases=result.get('cases') or [],
        )

    def get_extreme_cases(
        self,
        trainee_id: str,
        num: int,
        sort_feature: str,
        features: t.Optional[Collection[str]] = None
    ) -> Cases:
        """
        Gets the extreme cases of a Trainee for the given feature(s).

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to retrieve extreme cases from.
        num : int
            The number of cases to get.
        sort_feature : str
            The feature name by which extreme cases are sorted by.
        features: Collection of str, optional
            The feature names to use when getting extreme cases.

        Returns
        -------
        dict
            A cases object containing the feature names and extreme cases.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        util.validate_list_shape(features, 1, "features", "str")

        if self.configuration.verbose:
            print(f'Getting extreme cases for trainee with id: {trainee_id}')
        result = self._execute(trainee_id, "get_extreme_cases", {
            "features": features,
            "sort_feature": sort_feature,
            "num": num,
        })
        if result is None:
            result = dict()
        return Cases(
            features=result.get('features') or [],
            cases=result.get('cases') or [],
        )

    def get_num_training_cases(self, trainee_id: str) -> int:
        """
        Return the number of trained cases in the Trainee.

        Parameters
        ----------
        trainee_id : str
            The Id of the Trainee to retrieve the number of training cases from.

        Returns
        -------
        int
            The number of cases in the model
        """
        trainee_id = self._resolve_trainee(trainee_id)
        ret = self._execute(trainee_id, "get_num_training_cases", {})
        if isinstance(ret, dict):
            return ret.get('count', 0)
        return 0

    def get_feature_conviction(
        self,
        trainee_id: str,
        *,
        action_features: t.Optional[Collection[str]] = None,
        features: t.Optional[Collection[str]] = None,
        familiarity_conviction_addition: bool = True,
        familiarity_conviction_removal: bool = False,
        use_case_weights: bool = False,
        weight_feature: t.Optional[str] = None
    ) -> dict:
        """
        Get familiarity conviction for features in the model.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        features : Collection of str, optional
            An iterable of feature names to calculate convictions. At least 2
            features are required to get familiarity conviction. If not
            specified all features will be used.
        action_features : Collection of str, optional
            An iterable of feature names to be treated as action features
            during conviction calculation in order to determine the conviction
            of each feature against the set of action_features. If not
            specified, conviction is computed for each feature against the rest
            of the features as a whole.
        familiarity_conviction_addition : bool, default True
            Calculate and output familiarity conviction of adding the
            specified features in the output.
        familiarity_conviction_removal : bool, default False
            Calculate and output familiarity conviction of removing
            the specified features in the output.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        use_case_weights : bool, default False
            If set to True will scale influence weights by each
            case's weight_feature weight.

        Returns
        -------
        dict
            A dict with familiarity_conviction_addition or
            familiarity_conviction_removal
        """
        trainee_id = self._resolve_trainee(trainee_id)
        util.validate_list_shape(features, 1, "features", "str")
        util.validate_list_shape(action_features, 1, "action_features", "str")
        if self.configuration.verbose:
            print(f'Getting conviction of features for Trainee with id: {trainee_id}')
        return self._execute(trainee_id, "get_feature_conviction", {
            "features": features,
            "action_features": action_features,
            "familiarity_conviction_addition": familiarity_conviction_addition,
            "familiarity_conviction_removal": familiarity_conviction_removal,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })

    def add_feature(
        self,
        trainee_id: str,
        feature: str,
        feature_value: t.Optional[int | float | str] = None,
        *,
        condition: t.Optional[Mapping] = None,
        condition_session: t.Optional[str] = None,
        feature_attributes: t.Optional[Mapping] = None,
        overwrite: bool = False
    ):
        """
        Adds a feature to a trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee add the feature to.
        feature : str
            The name of the feature.
        feature_attributes : dict, optional
            The dict of feature specific attributes for this feature. If
            unspecified and conditions are not specified, will assume feature
            type as 'continuous'.
        feature_value : int or float or str, optional
            The value to populate the feature with.
            By default, populates the new feature with None.
        condition : Mapping, optional
            A condition map where feature values will only be added when
            certain criteria is met.

            If None, the feature will be added to all cases in the model
            and feature metadata will be updated to include it. If specified
            as an empty dict, the feature will still be added to all cases in
            the model but the feature metadata will not be updated.

            .. NOTE::
                The dictionary keys are the feature name and values are one of:

                    - None
                    - A value, must match exactly.
                    - An array of two numeric values, specifying an inclusive
                      range. Only applicable to continuous and numeric ordinal
                      features.
                    - An array of string values, must match any of these values
                      exactly. Only applicable to nominal and string ordinal
                      features.

            .. TIP::
                For instance to add the `feature_value` only when the
                `length` and `width` features are equal to 10::

                    condition = {"length": 10, "width": 10}

        condition_session : str, optional
            If specified, ignores the condition and operates on cases for the
            specified session id.
        overwrite : bool, default False
            If True, the feature will be over-written if it exists.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if feature_attributes is not None:
            updated_attributes = internals.preprocess_feature_attributes({feature: feature_attributes})
            if updated_attributes is None:
                raise AssertionError("Failed to preprocess feature attributes for new feature.")
            feature_attributes = updated_attributes[feature]

        if self.configuration.verbose:
            print(f'Adding feature "{feature}" to Trainee with id {trainee_id}.')
        self._execute(trainee_id, "add_feature", {
            "feature": feature,
            "feature_value": feature_value,
            "overwrite": overwrite,
            "condition": condition,
            "feature_attributes": feature_attributes,
            "session": self.active_session.id,
            "condition_session": condition_session,
        })
        self._auto_persist_trainee(trainee_id)

        # Update trainee in cache
        cached_trainee.features = self.get_feature_attributes(trainee_id)

    def remove_feature(
        self,
        trainee_id: str,
        feature: str,
        *,
        condition: t.Optional[Mapping] = None,
        condition_session: t.Optional[str] = None
    ):
        """
        Removes a feature from a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee remove the feature from.
        feature : str
            The name of the feature to remove.
        condition : dict, optional
            A condition map where features will only be removed when certain
            criteria is met.

            If None, the feature will be removed from all cases in the model
            and feature metadata will be updated to exclude it. If specified
            as an empty dict, the feature will still be removed from all cases
            in the model but the feature metadata will not be updated.

            .. NOTE::
                The dictionary keys are the feature name and values are one of:

                    - None
                    - A value, must match exactly.
                    - An array of two numeric values, specifying an inclusive
                      range. Only applicable to continuous and numeric ordinal
                      features.
                    - An array of string values, must match any of these values
                      exactly. Only applicable to nominal and string ordinal
                      features.

            .. TIP::
                For instance to remove the `length` feature only when the
                value is between 1 and 5::

                    condition = {"length": [1, 5]}

        condition_session : str, optional
            If specified, ignores the condition and operates on cases for the
            specified session id.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if self.configuration.verbose:
            print(f'Removing feature "{feature}" from Trainee with id: {trainee_id}')
        self._execute(trainee_id, "remove_feature", {
            "feature": feature,
            "condition": condition,
            "session": self.active_session.id,
            "condition_session": condition_session,
        })
        self._auto_persist_trainee(trainee_id)

        # Update trainee in cache
        cached_trainee.features = self.get_feature_attributes(trainee_id)

    def get_pairwise_distances(
        self,
        trainee_id: str,
        features: t.Optional[Collection[str]] = None,
        *,
        action_feature: t.Optional[str] = None,
        from_case_indices: t.Optional[CaseIndices] = None,
        from_values: t.Optional[TabularData2D] = None,
        to_case_indices: t.Optional[CaseIndices] = None,
        to_values: t.Optional[TabularData2D] = None,
        use_case_weights: bool = False,
        weight_feature: t.Optional[str] = None
    ) -> list[float]:
        """
        Compute pairwise distances between specified cases.

        Returns a list of computed distances between each respective pair of
        cases specified in either `from_values` or `from_case_indices` to
        `to_values` or `to_case_indices`. If only one case is specified in any
        of the lists, all respective distances are computed to/from that one
        case.

        .. NOTE::
            - One of `from_values` or `from_case_indices` must be specified,
              not both.
            - One of `to_values` or `to_case_indices` must be specified,
              not both.

        Parameters
        ----------
        trainee_id : str
            The trainee ID.
        features : iterable of str, optional
            List of feature names to use when computing pairwise distances.
            If unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this `action_feature`, otherwise uses targetless
            hyperparameters.
        from_case_indices : Iterable of Sequence[Union[str, int]], optional
            An iterable of sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of `to_values` or `to_case_indices`.
        from_values : list of list of object or pandas.DataFrame, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of `to_values` or `to_case_indices`.
        to_case_indices : Iterable of Sequence[Union[str, int]], optional
            An Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of `from_values` or `from_case_indices`.
        to_values : list of list of object or pandas.DataFrame, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of `from_values` or `from_case_indices`.
        use_case_weights : bool, default False
            If set to True, will scale influence weights by each case's
            `weight_feature` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        list
            A list of computed pairwise distances between each corresponding
            pair of cases in `from_case_indices` and `to_case_indices`.
        """
        trainee_id = self._resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features

        util.validate_list_shape(from_values, 2, 'from_values', 'list of list of object')
        util.validate_list_shape(to_values, 2, 'to_values', 'list of list of object')

        if from_case_indices is None and from_values is None:
            raise ValueError("One of `from_case_indices` or `from_values` "
                             "must be specified.")
        elif from_case_indices is not None and from_values is not None:
            raise ValueError("Only one of `from_case_indices` or `from_values` "
                             "may be specified, not both.")

        if to_case_indices is None and to_values is None:
            raise ValueError("One of `to_case_indices` or `to_values` "
                             "must be specified.")
        elif to_case_indices is not None and to_values is not None:
            raise ValueError("Only one of `to_case_indices` or `to_values` "
                             "may be specified, not both.")

        # Validate case_indices if provided
        if from_case_indices:
            util.validate_case_indices(from_case_indices)
        if to_case_indices:
            util.validate_case_indices(to_case_indices)

        # Serialize values if defined
        if from_values is not None:
            if features is None:
                features = internals.get_features_from_data(
                    from_values, data_parameter='from_values')
            from_values = serialize_cases(from_values, features, feature_attributes)
        if to_values is not None:
            if features is None:
                features = internals.get_features_from_data(
                    to_values, data_parameter='to_values')
            to_values = serialize_cases(to_values, features, feature_attributes)

        if self.configuration.verbose:
            print(f'Getting pairwise distances for Trainee with id: {trainee_id}')

        result = self._execute(trainee_id, "get_pairwise_distances", {
            "features": features,
            "action_feature": action_feature,
            "from_case_indices": from_case_indices,
            "from_values": from_values,
            "to_case_indices": to_case_indices,
            "to_values": to_values,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })
        if result is None:
            return []
        return result

    def get_distances(  # noqa: C901
        self,
        trainee_id: str,
        features: t.Optional[Iterable[str]] = None,
        *,
        action_feature: t.Optional[str] = None,
        case_indices: t.Optional[CaseIndices] = None,
        feature_values: t.Optional[list[t.Any] | DataFrame] = None,
        use_case_weights: bool = False,
        weight_feature: t.Optional[str] = None
    ) -> Distances:
        """
        Compute distances matrix for specified cases.

        Returns a dict with computed distances between all cases
        specified in `case_indices` or from all cases in local model as defined
        by `feature_values`. If neither `case_indices` nor `feature_values` is
        specified, returns computed distances for the entire dataset.

        Parameters
        ----------
        trainee_id : str
            The trainee ID.
        features : iterable of str, optional
            List of feature names to use when computing distances. If
            unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this `action_feature`, otherwise uses targetless
            hyperparameters.
        case_indices : Iterable of Sequence[Union[str, int]], optional
            An Iterable of Sequences, of session id and index, where index is
            the original 0-based index of the case as it was trained into the
            session. If specified, returns distances for all of these
            cases. Ignored if `feature_values` is provided. If neither
            `feature_values` nor `case_indices` is specified, uses full dataset.
        feature_values : list of object or DataFrame, optional
            If specified, returns distances of the local model relative to
            these values, ignores `case_indices` parameter. If provided a
            DataFrame, only the first row will be used.
        use_case_weights : bool, default False
            If set to True, will scale influence weights by each case's
            `weight_feature` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        dict
            A dict containing a matrix of computed distances and the list of
            corresponding case indices in the following format::

                {
                    'case_indices': [ session-indices ],
                    'distances': [ [ distances ] ]
                }
        """
        trainee_id = self._resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features

        # Validate case_indices if provided
        if case_indices is not None:
            util.validate_case_indices(case_indices)

        if feature_values is not None:
            if (
                isinstance(feature_values, Iterable)
                and len(np.array(feature_values).shape) == 1
                and len(feature_values) > 0
            ):
                # Convert 1d list to 2d list for serialization
                feature_values = [feature_values]

            if features is None:
                features = internals.get_features_from_data(feature_values, data_parameter='feature_values')
            feature_values = serialize_cases(feature_values, features, feature_attributes)
            if feature_values:
                # Only a single case should be provided
                feature_values = feature_values[0]
            # Ignored when feature_values specified
            case_indices = None

        if case_indices is not None and len(case_indices) < 2:
            raise ValueError("If providing `case_indices`, must provide at "
                             "least 2 cases for computation.")

        if self.configuration.verbose:
            print(f'Getting distances between cases for Trainee with id: {trainee_id}')

        preallocate = True  # If matrix should be preallocated in memory
        matrix_ndarray = None  # Used when preallocating
        matrix_list = []  # Used if we cannot preallocate
        page_size = 2000
        indices = []
        total_rows = 0
        total_cols = 0
        mismatch_msg = (
            "Received mismatched distance value pairs. It is likely some "
            "cases were either deleted or trained during the computation of "
            "get_distances. Rerunning this operation may resolve this error."
        )

        if feature_values is not None:
            # When specifying feature values, only distances closest to this
            # case will be returned. The largest matrix size that could be
            # expected is 144x144, so we can request the entire matrix at once.
            # Set num_cases to 1 so we only page once.
            num_cases = 1
            preallocate = False  # won't know the actual size beforehand
        elif case_indices is not None:
            num_cases = len(case_indices)
        else:
            num_cases = self.get_num_training_cases(trainee_id)

        # Preallocate matrix (This will raise a numpy MemoryError if too large)
        if preallocate:
            matrix_ndarray = np.zeros((num_cases, num_cases), dtype='float64')

        for row_offset in range(0, num_cases, page_size):
            for column_offset in range(0, num_cases, page_size):
                result = self._execute(trainee_id, "get_distances", {
                    "features": features,
                    "action_feature": action_feature,
                    "case_indices": case_indices,
                    "feature_values": feature_values,
                    "weight_feature": weight_feature,
                    "use_case_weights": use_case_weights,
                    "row_offset": row_offset,
                    "row_count": page_size,
                    "column_offset": column_offset,
                    "column_count": page_size,
                })

                column_case_indices = result['column_case_indices']
                row_case_indices = result['row_case_indices']
                distances = result['distances']

                if preallocate and matrix_ndarray is not None:
                    # Fill in allocated matrix
                    try:
                        matrix_ndarray[
                            row_offset:row_offset + len(row_case_indices),
                            column_offset:column_offset + len(column_case_indices)
                        ] = distances
                    except ValueError as err:
                        # Unexpected shape when populating array
                        raise HowsoError(mismatch_msg) from err
                else:
                    if column_offset == 0:
                        # Append new rows
                        matrix_list += distances
                    else:
                        # Extend existing columns
                        try:
                            for i, cols in enumerate(distances):
                                matrix_list[row_offset + i].extend(cols)
                        except (AttributeError, IndexError):
                            # Unexpected shape when populating array
                            raise HowsoError(mismatch_msg)

                if column_offset == 0:
                    total_rows += len(row_case_indices)
                if row_offset == 0:
                    total_cols += len(column_case_indices)
                    # Collect the axis indices. Both axis will be the same,
                    # so we only need to collect them the first time we page
                    # through the columns when row offset is 0.
                    indices += column_case_indices

        if preallocate:
            if total_cols != num_cases or total_rows != num_cases:
                # Received unexpected number of distances
                raise HowsoError(mismatch_msg)
            distances_matrix = matrix_ndarray if matrix_ndarray is not None else np.ndarray(0, dtype="float64")
        else:
            if matrix_list:
                # Validate matrix shape
                if (
                    total_cols != total_rows or
                    not all(len(r) == total_cols for r in matrix_list)
                ):
                    raise HowsoError(mismatch_msg)
            # Convert matrix to numpy array
            distances_matrix = np.array(matrix_list, dtype='float64')

        return {
            'case_indices': indices,
            'distances': distances_matrix
        }

    def get_params(
        self,
        trainee_id: str,
        *,
        action_feature: t.Optional[str] = None,
        context_features: t.Optional[Iterable[str]] = None,
        mode: t.Optional[Mode] = None,
        weight_feature: t.Optional[str] = None,
    ) -> dict[str, t.Any]:
        """
        Get the parameters used by the Trainee.

        If 'action_feature',
        'context_features', 'mode', or 'weight_feature' are specified, then
        the best hyperparameters analyzed in the Trainee are the value of the
        'hyperparameter_map' key, otherwise this value will be the dictionary
        containing all the hyperparameter sets in the Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee get parameters from.
        action_feature : str, optional
            If specified will return the best analyzed hyperparameters to
            target this feature.
        context_features : str, optional
            If specified, will find and return the best analyzed hyperparameters
            to use with these context features.
        mode : str, optional
            If specified, will find and return the best analyzed hyperparameters
            that were computed in this mode.
        weight_feature : str, optional
            If specified, will find and return the best analyzed hyperparameters
            that were analyzed using this weight feature.

        Returns
        -------
        dict
            A dict including the either all of the Trainee's internal
            parameters or only the best hyperparameters selected using the
            passed parameters.
        """
        self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print(f'Getting model attributes from Trainee with id: {trainee_id}')
        return self._execute(trainee_id, "get_params", {
            "action_feature": action_feature,
            "context_features": context_features,
            "mode": mode,
            "weight_feature": weight_feature,
        })

    def set_params(self, trainee_id: str, params: Mapping):
        """
        Sets specific hyperparameters in the Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee set hyperparameters.

        params : dict
            A dictionary in the following format containing the hyperparameter
            information, which is required, and other parameters which are
            all optional.

            Example::

                {
                    "hyperparameter_map": {
                        ".targetless": {
                            "robust": {
                                ".none": {
                                    "dt": -1, "p": .1, "k": 8
                                }
                            }
                        }
                    },
                }
        """
        trainee_id = self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print(f'Setting model attributes for Trainee with id: {trainee_id}')

        parameters = dict(params)
        deprecated_params = {
            'auto_optimize_enabled': 'auto_analyze_enabled',
            'optimize_threshold': 'analyze_threshold',
            'optimize_growth_factor': 'analyze_growth_factor',
            'auto_optimize_limit_size': 'auto_analyze_limit_size',
        }

        # replace any old params with new params and remove old param
        for old_param, new_param in deprecated_params.items():
            if old_param in parameters:
                parameters[new_param] = parameters[old_param]
                del parameters[old_param]
                warnings.warn(
                    f'The `{old_param}` parameter has been renamed to '
                    f'`{new_param}`, please use the new parameter '
                    'instead.', UserWarning)

        self._execute(trainee_id, "set_params", parameters)
        self._auto_persist_trainee(trainee_id)
