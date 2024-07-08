from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, MutableMapping
import typing as t
from uuid import UUID
import warnings

from pandas import DataFrame, Index

from howso.utilities import internals
from howso.utilities import utilities as util
from howso.utilities.features import serialize_cases
from .exceptions import HowsoError

if t.TYPE_CHECKING:
    from howso.client.schemas import HowsoVersion, Project, Reaction, Session, Trainee, TraineePersistence
    from .cache import TraineeCache
    from .configuration import HowsoConfiguration
    from .typing import CaseIndices, Cases, Precision


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
    def _auto_resolve_trainee(self, trainee_id: str):
        """
        Resolve a Trainee and acquire its resources.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to persist.
        """

    @abstractmethod
    def _auto_persist_trainee(self, trainee_id: str):
        """
        Automatically persists the Trainee if the persistence state allows.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to persist.
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

    def _resolve_trainee_id(self, trainee_id: str, *args, **kwargs):
        """Resolve trainee identifier."""
        return trainee_id

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
        persistence: TraineePersistence = "allow",
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
    def get_trainee_metrics(self, trainee_id: str) -> dict:
        """Get metric information for a trainee."""

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
        seed: int or float or str
            The random seed.
            Ex: ``7998``, ``"myrandomseed"``
        """
        self._auto_resolve_trainee(trainee_id)
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
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")
        util.validate_list_shape(features, 1, "features", "str")
        util.validate_list_shape(features_to_impute, 1, "features_to_impute", "str")

        if self.configuration.verbose:
            print(f'Imputing Trainee with id: {trainee_id}')
        self._auto_resolve_trainee(trainee_id)
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

        self._auto_resolve_trainee(trainee_id)
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

        self._auto_resolve_trainee(trainee_id)
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
        feature_values: list[t.Any] | DataFrame,
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
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'])

        if case_indices is not None:
            util.validate_case_indices(case_indices)

        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        # Serialize feature_values
        if feature_values is not None:
            if features is None:
                features = internals.get_features_from_data(feature_values, data_parameter='feature_values')
            serialized_feature_values = serialize_cases(feature_values, features, cached_trainee.features)
            if serialized_feature_values:
                # Only a single case should be provided
                serialized_feature_values = serialized_feature_values[0]
        else:
            serialized_feature_values = None

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

    @abstractmethod
    def set_substitute_feature_values(self, trainee_id, substitution_value_map):
        """Set a substitution map for use in extended nominal generation."""

    @abstractmethod
    def get_substitute_feature_values(self, trainee_id, clear_on_get=True) -> dict:
        """Get a substitution map for use in extended nominal generation."""

    @abstractmethod
    def set_feature_attributes(self, trainee_id, feature_attributes):
        """Set feature attributes for a trainee."""

    @abstractmethod
    def get_feature_attributes(self, trainee_id):
        """Get a dict of feature attributes."""

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

    @abstractmethod
    def get_marginal_stats(
        self, trainee_id, *,
        condition=None,
        num_cases=None,
        precision=None,
        weight_feature=None,
    ) -> DataFrame | dict:
        """Get marginal stats for all features."""

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

    @abstractmethod
    def evaluate(self, trainee_id, features_to_code_map, *, aggregation_code=None) -> dict:
        """Evaluate custom code on case values within the trainee."""

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
        if case_indices is not None:
            util.validate_case_indices(case_indices)

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'])

        util.validate_list_shape(features, 1, "features", "str")
        if session is None and case_indices is None:
            warnings.warn("Calling get_cases without a session id does not guarantee case order.")
        if self.configuration.verbose:
            print(f'Retrieving cases for Trainee with id {trainee_id}.')

        self._auto_resolve_trainee(trainee_id)
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
        util.validate_list_shape(features, 1, "features", "str")

        self._auto_resolve_trainee(trainee_id)
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
        self._auto_resolve_trainee(trainee_id)
        ret = self._execute(trainee_id, "get_num_training_cases", {})
        if isinstance(ret, dict):
            return ret.get('count', 0)
        return 0

    @abstractmethod
    def get_feature_conviction(
        self,
        trainee_id,
        *,

        familiarity_conviction_addition: bool | str = True,
        familiarity_conviction_removal: bool | str = False,
        use_case_weights: bool = False,
        features=None,
        action_features=None,
        weight_feature=None
    ) -> dict | DataFrame:
        """Get familiarity conviction for features in the model."""

    @abstractmethod
    def add_feature(self, trainee_id, feature, feature_value=None, *,
                    condition=None, condition_session=None,
                    feature_attributes=None, overwrite=False):
        """Add a feature to a trainee's model."""

    @abstractmethod
    def remove_feature(self, trainee_id, feature, *, condition=None,
                       condition_session=None):
        """Remove a feature from a trainee."""

    @abstractmethod
    def get_pairwise_distances(self, trainee_id, features=None, *,
                               action_feature=None, from_case_indices=None,
                               from_values=None, to_case_indices=None,
                               to_values=None, use_case_weights=False,
                               weight_feature=None) -> list[float]:
        """Compute pairwise distances between specified cases."""

    @abstractmethod
    def get_distances(self, trainee_id, features=None, *,
                      action_feature=None, case_indices=None,
                      feature_values=None, use_case_weights=False,
                      weight_feature=None) -> dict:
        """Compute distances matrix for specified cases."""

    @abstractmethod
    def get_params(self, trainee_id, *, action_feature=None,
                   context_features=None, mode=None, weight_feature=None) -> dict[str, t.Any]:
        """Get parameters used by the system."""

    @abstractmethod
    def set_params(self, trainee_id, params):
        """Set specific hyperparameters in the trainee."""
