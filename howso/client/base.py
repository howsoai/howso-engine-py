from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping, MutableMapping, Sized
from pathlib import Path
import typing as t
from uuid import UUID
import warnings

import numpy as np
from pandas import DataFrame

from howso.utilities import internals
from howso.utilities import utilities as util
from howso.utilities.feature_attributes.base import MultiTableFeatureAttributes, SingleTableFeatureAttributes
from howso.utilities.features import serialize_cases
from howso.utilities.monitors import ProgressTimer
from .exceptions import HowsoError, UnsupportedArgumentWarning
from .schemas import HowsoVersion, Project, Reaction, Session, Trainee, TraineeRuntime
from .typing import (
    CaseIndices,
    Cases,
    Distances,
    Evaluation,
    GenerateNewCases,
    LibraryType,
    Mode,
    NewCaseThreshold,
    Persistence,
    Precision,
    SeriesIDTracking,
    TabularData2D,
    TabularData3D,
    TargetedModel,
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
            'if the value of `%s` is "exact".')
    }
    """Mapping of warning type to default warning message."""

    SUPPORTED_PRECISION_VALUES = ["exact", "similar"]
    """Allowed values for precision."""

    @property
    def batch_scaler_class(self):
        """The batch scaling manager class used by operations that batch requests."""
        return internals.BatchScalingManager

    @property
    def verbose(self) -> bool:
        """Get verbose flag."""
        # Backwards compatible reference to the verbose flag
        return self.configuration.verbose

    @property
    @abstractmethod
    def trainee_cache(self) -> TraineeCache:
        """Return the Trainee cache."""

    @property
    @abstractmethod
    def active_session(self) -> Session:
        """Return the active Session."""

    @property
    @abstractmethod
    def train_initial_batch_size(self) -> int:
        """The default number of cases in the first train batch."""

    @property
    @abstractmethod
    def react_initial_batch_size(self) -> int:
        """The default number of cases in the first react batch."""

    @staticmethod
    def sanitize_for_json(payload: t.Any, *, exclude_null: bool = False) -> t.Any:
        """
        Prepare payload for json serialization.

        Parameters
        ----------
        payload : Any
            The payload to sanitize.
        exclude_null : bool, default False
            If top level Mapping keys should be filtered out if they are null.

        Returns
        -------
        Any
            The sanitized payload.
        """
        payload = internals.sanitize_for_json(payload)
        if exclude_null and isinstance(payload, Mapping):
            payload = dict((k, v) for k, v in payload.items() if v is not None)
        return payload

    def resolve_feature_attributes(self, trainee_id: str) -> dict[str, dict]:
        """
        Resolve a Trainee's feature attributes.

        Returns cached feature attributes if available. Otherwise
        resolves the Trainee and cache's its feature attributes.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        dict
            The Trainee feature attributes.
        """
        cached = self.trainee_cache.get_item(trainee_id, None)
        if cached is None:
            # Trainee not yet cached, resolve it first
            trainee_id = self._resolve_trainee(trainee_id).id
            cached = self.trainee_cache.get_item(trainee_id)

        if cached["feature_attributes"] is None:
            # Feature attributes not yet cached, get them
            cached["feature_attributes"] = self.get_feature_attributes(trainee_id)
        return cached["feature_attributes"]

    @abstractmethod
    def _resolve_trainee(self, trainee_id: str, **kwargs) -> Trainee:
        """
        Resolve a Trainee resource.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to resolve.

        Returns
        -------
        Trainee
            The Trainee object.
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
    def _store_session(self, trainee_id: str, session: Session):
        """
        Store a session for a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        session : Session
            The session to store.
        """

    @abstractmethod
    def _should_react_batch(self, params: dict, total_size: int) -> bool:
        """
        Determine if given react should be batched.

        Parameters
        ----------
        params : dict
            The react request parameters.
        total_size : int
            The size of the cases being reacted to.

        Returns
        -------
        bool
            Whether a react should be batched.
        """

    @abstractmethod
    def _get_trainee_thread_count(self, trainee_id: str) -> int:
        """
        Get the number of available cpu threads a Trainee has access to.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        int
            The allocated number of cpu threads for a Trainee.
        """

    @abstractmethod
    def execute(self, trainee_id: str, label: str, payload: t.Any, **kwargs) -> t.Any:
        """
        Execute a label in Howso engine.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
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
    def execute_sized(self, trainee_id: str, label: str, payload: t.Any, **kwargs) -> tuple[t.Any, int, int]:
        """
        Execute a label in Howso engine and return the request and response sizes.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
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
    def is_tracing_enabled(self, trainee_id: str) -> bool:
        """
        Get if tracing is enabled for Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        bool
            True, if tracing is enabled for provided Trainee.
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
        library_type: t.Optional[LibraryType] = None,
        max_wait_time: t.Optional[int | float] = None,
        metadata: t.Optional[MutableMapping[str, t.Any]] = None,
        overwrite_trainee: bool = False,
        persistence: Persistence = "allow",
        project: t.Optional[str | Project] = None,
        resources: t.Optional[Mapping[str, t.Any]] = None
    ) -> Trainee:
        """Create a Trainee in the Howso service."""

    @abstractmethod
    def update_trainee(self, trainee: Mapping | Trainee) -> Trainee:
        """Update an existing trainee in the Howso service."""

    @abstractmethod
    def get_trainee(self, trainee_id: str) -> Trainee:
        """Get an existing trainee from the Howso service."""

    @abstractmethod
    def get_trainee_runtime(self, trainee_id: str) -> TraineeRuntime:
        """
        Get runtime details of a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        TraineeRuntime
            The Trainee runtime details. Including Trainee version and
            configuration parameters.
        """

    @abstractmethod
    def query_trainees(self, search_terms: t.Optional[str] = None) -> list[dict]:
        """Query accessible Trainees."""

    @abstractmethod
    def delete_trainee(self, trainee_id: str, *, file_path: t.Optional[Path | str] = None):
        """Delete a Trainee from the Howso service."""

    @abstractmethod
    def copy_trainee(
        self, trainee_id, new_trainee_name=None, *,
        library_type=None,
        resources=None,
    ) -> Trainee:
        """Copy a trainee in the Howso service."""

    @abstractmethod
    def acquire_trainee_resources(self, trainee_id: str, *, max_wait_time: t.Optional[int | float] = None):
        """Acquire resources for a Trainee in the Howso service."""

    @abstractmethod
    def release_trainee_resources(self, trainee_id: str):
        """Release a Trainee's resources from the Howso service."""

    @abstractmethod
    def persist_trainee(self, trainee_id: str):
        """Persist a trainee in the Howso service."""

    @abstractmethod
    def begin_session(self, name: str | None = 'default', metadata: t.Optional[Mapping] = None) -> Session:
        """Begin a new session."""

    @abstractmethod
    def query_sessions(
        self,
        search_terms: t.Optional[str] = None,
        *,
        trainee: t.Optional[str | Trainee] = None,
        **kwargs
    ) -> list[Session]:
        """Query all accessible sessions."""

    @abstractmethod
    def get_session(self, session_id: str) -> Session:
        """Get session details."""

    @abstractmethod
    def update_session(self, session_id: str, *, metadata: t.Optional[Mapping] = None) -> Session:
        """Update a session."""

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
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Setting random seed for Trainee with id: {trainee_id}')
        self.execute(trainee_id, "set_random_seed", {"seed": seed})
        self._auto_persist_trainee(trainee_id)

    def train(  # noqa: C901
        self,
        trainee_id: str,
        cases: TabularData2D,
        features: t.Optional[Collection[str]] = None,
        *,
        accumulate_weight_feature: t.Optional[str] = None,
        batch_size: t.Optional[int] = None,
        derived_features: t.Optional[Collection[str]] = None,
        initial_batch_size: t.Optional[int] = None,
        input_is_substituted: bool = False,
        progress_callback: t.Optional[Callable] = None,
        series: t.Optional[str] = None,
        skip_auto_analyze: bool = False,
        train_weights_only: bool = False,
        validate: bool = True,
    ):
        """
        Train one or more cases into a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the target Trainee.
        cases : list of list of object or pandas.DataFrame
            One or more cases to train into the model.
        features : Collection of str, optional
            The feature names corresponding to the case values.
            This parameter should be provided in the following scenarios:

                a. When cases are not in the format of a DataFrame, or
                   the DataFrame does not define named columns.
                b. You want to train only a subset of columns defined in your
                   cases DataFrame.
                c. You want to re-order the columns that are trained.

        accumulate_weight_feature : str, optional
            Name of feature into which to accumulate neighbors'
            influences as weight for ablated cases. If unspecified, will not
            accumulate weights.
        batch_size : int, optional
            Define the number of cases to train at once. If left unspecified,
            the batch size will be determined automatically.
        derived_features : Collection of str, optional
            Feature names for which values should be derived in the specified
            order. If this list is not provided, features with the
            'auto_derive_on_train' feature attribute set to True will be
            auto-derived. If provided an empty list, no features are derived.
            Any derived_features that are already in the 'features' list will
            not be derived since their values are being explicitly provided.
        initial_batch_size : int, optional
            Define the number of cases to train in the first batch. If
            unspecified, the value of the ``train_initial_batch_size`` property
            is used. The number of cases in following batches will be
            automatically adjusted. This value is ignored if ``batch_size`` is
            specified.
        input_is_substituted : bool, default False
            if True assumes provided nominal feature values have
            already been substituted.
        progress_callback : callable, optional
            A callback method that will be called before each
            batched call to train and at the end of training. The method is
            given a ProgressTimer containing metrics on the progress and timing
            of the train operation.
        series : str, optional
            Name of the series to pull features and case values
            from internal series storage. If specified, trains on all cases
            that are stored in the internal series store for the specified
            series. The trained feature set is the combined features from
            storage and the passed in features. If cases is of length one,
            the value(s) of this case are appended to all cases in the series.
            If cases is the same length as the series, the value of each case
            in cases is applied in order to each of the cases in the series.
        skip_auto_analyze : bool, default False
            When true, the Trainee will not auto-analyze when appropriate.
            Instead, the boolean response will be True if an analyze is needed.
        train_weights_only : bool, default False
            When true, and accumulate_weight_feature is provided,
            will accumulate all of the cases' neighbor weights instead of
            training the cases into the model.
        validate : bool, default True
            Whether to validate the data against the provided feature
            attributes. Issues warnings if there are any discrepancies between
            the data and the features dictionary.

        Returns
        -------
        bool
            Flag indicating if the Trainee needs to analyze. Only true if
            auto-analyze is enabled and the conditions are met.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)

        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        # Make sure single table dicts are wrapped by SingleTableFeatureAttributes
        if (
            isinstance(feature_attributes, Mapping) and
            not isinstance(feature_attributes, MultiTableFeatureAttributes)
        ):
            feature_attributes = SingleTableFeatureAttributes(feature_attributes, {})

        # Check to see if the feature attributes still generally describe
        # the data, and warn the user if they do not
        if isinstance(cases, DataFrame) and validate:
            try:
                feature_attributes.validate(cases)
            except NotImplementedError:
                # MultiTableFeatureAttributes does not yet support DataFrame validation
                pass

        # See if any features were inferred to have data that is unsupported by the OS.
        # Issue a warning and drop the feature before training, if so.
        unsupported_features = []
        if isinstance(feature_attributes, MultiTableFeatureAttributes):
            for stfa in feature_attributes.values():
                unsupported_features = [feat for feat in stfa.keys() if stfa.has_unsupported_data(feat)]
        elif isinstance(feature_attributes, SingleTableFeatureAttributes):
            unsupported_features = [feat for feat in feature_attributes.keys()
                                    if feature_attributes.has_unsupported_data(feat)]
        if isinstance(cases, DataFrame):
            for feature in unsupported_features:
                warnings.warn(
                    f'Ignoring feature {feature} as it contains values that are too '
                    'large or small for your operating system. Please evaluate the '
                    'bounds for this feature.')
                cases.drop(feature, axis=1, inplace=True)

        util.validate_list_shape(features, 1, "features", "str")
        util.validate_list_shape(cases, 2, "cases", "list", allow_none=False)
        if features is None:
            features = internals.get_features_from_data(cases)
        serialized_cases = serialize_cases(cases, features, feature_attributes, warn=True) or []

        needs_analyze = False

        if self.configuration.verbose:
            print(f'Training session(s) on Trainee with id: {trainee_id}')

        with ProgressTimer(len(serialized_cases)) as progress:
            gen_batch_size = None
            batch_scaler = None
            if series is not None:
                # If training series, always send full size
                batch_size = len(serialized_cases)
            if not batch_size:
                # Scale the batch size automatically
                start_batch_size = initial_batch_size or self.train_initial_batch_size
                batch_scaler = self.batch_scaler_class(start_batch_size, progress)
                gen_batch_size = batch_scaler.gen_batch_size()
                batch_size = next(gen_batch_size, None)

            while not progress.is_complete and batch_size:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress)
                start = progress.current_tick
                end = progress.current_tick + batch_size
                response, in_size, out_size = self.execute_sized(trainee_id, "train", {
                    "cases": serialized_cases[start:end],
                    "accumulate_weight_feature": accumulate_weight_feature,
                    "derived_features": derived_features,
                    "features": features,
                    "input_is_substituted": input_is_substituted,
                    "series": series,
                    "session": self.active_session.id,
                    "skip_auto_analyze": skip_auto_analyze,
                    "train_weights_only": train_weights_only,
                })
                if response and response.get('status') == 'analyze':
                    needs_analyze = True
                if batch_scaler is None or gen_batch_size is None:
                    progress.update(batch_size)
                else:
                    batch_size = batch_scaler.send(
                        gen_batch_size,
                        batch_scaler.SendOptions(None, (in_size, out_size)))

        # Final call to batch callback on completion
        if isinstance(progress_callback, Callable):
            progress_callback(progress)

        self._store_session(trainee_id, self.active_session)
        self._auto_persist_trainee(trainee_id)

        return needs_analyze

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
        trainee_id = self._resolve_trainee(trainee_id).id
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")
        util.validate_list_shape(features, 1, "features", "str")
        util.validate_list_shape(features_to_impute, 1, "features_to_impute", "str")

        if self.configuration.verbose:
            print(f'Imputing Trainee with id: {trainee_id}')
        self.execute(trainee_id, "impute", {
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
        condition : Mapping of str to object, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if num_cases < 1:
            raise ValueError('num_cases must be a value greater than 0')

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'].format("precision"))

        # Convert session instance to id
        if (
            isinstance(condition, MutableMapping) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Removing case(s) from Trainee with id: {trainee_id}')

        result = self.execute(trainee_id, "remove_cases", {
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
        condition : Mapping, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if num_cases < 1:
            raise ValueError('num_cases must be a value greater than 0')

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'].format("precision"))

        # Convert session instance to id
        if (
            isinstance(condition, MutableMapping) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Moving case(s) from Trainee with id: {trainee_id}')

        result = self.execute(trainee_id, "move_cases", {
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
        feature_values: Collection[t.Any] | DataFrame,
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

        Updates the accumulated data mass for the model proportional to the
        number of cases and features modified.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to edit the cases of.
        feature_values : list of object or pandas.DataFrame
            The feature values to edit the case(s) with. If specified as a list,
            the order corresponds with the order of the `features` parameter.
            If specified as a DataFrame, only the first row will be used.
        case_indices : Sequence of tuple of {str, int}, optional
            Sequence of tuples containing the session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. This explicitly specifies the cases to edit. When
            specified, `condition` and `condition_session` are ignored.
        condition : Mapping, optional
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
        features : Collection of str, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'].format("precision"))

        if case_indices is not None:
            util.validate_case_indices(case_indices)

        # Serialize feature_values
        serialized_feature_values = None
        if feature_values is not None:
            feature_attributes = self.resolve_feature_attributes(trainee_id)
            if features is None:
                features = internals.get_features_from_data(feature_values, data_parameter='feature_values')
            serialized_feature_values = serialize_cases(feature_values, features, feature_attributes)
            if serialized_feature_values:
                # Only a single case should be provided
                serialized_feature_values = serialized_feature_values[0]

        # Convert session instance to id
        if (
            isinstance(condition, MutableMapping) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Editing case(s) in Trainee with id: {trainee_id}')

        result = self.execute(trainee_id, "edit_cases", {
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

    def remove_series_store(self, trainee_id: str, series: t.Optional[str] = None):
        """
        Clear any stored series from the Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to remove the series store from.
        series : str, optional
            The ID of the series to clear.
            If None, the Trainee's entire series store will be cleared.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Removing stored series from Trainee with id: {trainee_id} and series: {series}')
        self.execute(trainee_id, "remove_series_store", {"series": series})

    def append_to_series_store(
        self,
        trainee_id: str,
        series: str,
        contexts: TabularData2D,
        *,
        context_features: t.Optional[Collection[str]] = None
    ):
        """
        Append the specified contexts to a series store.

        For use with train series.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to append to.
        series : str
            The name of the series store to append to.
        contexts : list of list of object or pandas.DataFrame
            The list of list of context values to append to the series.
        context_features : Collection of str, optional
            The feature names corresponding to context values.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)

        util.validate_list_shape(contexts, 2, "contexts", "list of object", allow_none=False)

        if context_features is None:
            context_features = internals.get_features_from_data(
                contexts,
                data_parameter='contexts',
                features_parameter='context_features'
            )

        # Preprocess contexts
        serialized_contexts = serialize_cases(contexts, context_features, feature_attributes)

        if self.configuration.verbose:
            print(f'Appending to series store for Trainee with id: {trainee_id}, and series: {series}')

        self.execute(trainee_id, "append_to_series_store", {
            "context_features": context_features,
            "context_values": serialized_contexts,
            "series": series,
        })

    def set_substitute_feature_values(self, trainee_id: str, substitution_value_map: Mapping):
        """
        Set a Trainee's substitution map for use in extended nominal generation.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set substitute feature values for.
        substitution_value_map : Mapping
            A dictionary of feature name to a dictionary of feature value to
            substitute feature value.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Setting substitute feature values for Trainee with id: {trainee_id}')
        self.execute(trainee_id, "set_substitute_feature_values", {
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Getting substitute feature values from Trainee with id: {trainee_id}')
        result = self.execute(trainee_id, "get_substitute_feature_values", {})
        if clear_on_get:
            self.execute(trainee_id, "set_substitute_feature_values", {
                "substitution_value_map": {}
            })
            self._auto_persist_trainee(trainee_id)
        if result is None:
            return dict()
        return result

    def set_feature_attributes(
        self,
        trainee_id: str,
        feature_attributes: Mapping[str, Mapping]
    ) -> dict[str, dict]:
        """
        Sets feature attributes for a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        feature_attributes : Mapping of str to Mapping
            A dict of dicts of feature attributes. Each key is the feature
            'name' and each value is a dict of feature-specific parameters.

            Example::

                {
                    "length": { "type" : "continuous", "decimal_places": 1 },
                    "width": { "type" : "continuous", "significant_digits": 4 },
                    "degrees": { "type" : "continuous", "cycle_length": 360 },
                    "class": { "type" : "nominal" }
                }

        Returns
        -------
        dict
            The updated dictionary of feature name to dictionary of feature attributes.
        """
        trainee_id = self._resolve_trainee(trainee_id).id

        if not isinstance(feature_attributes, Mapping):
            raise ValueError("`feature_attributes` must be a dict")
        if self.configuration.verbose:
            print(f'Setting feature attributes for Trainee with id: {trainee_id}')

        updated_feature_attributes = self.execute(trainee_id, "set_feature_attributes", {
            "feature_attributes": internals.preprocess_feature_attributes(feature_attributes),
        })
        updated_feature_attributes = internals.postprocess_feature_attributes(updated_feature_attributes)
        self._auto_persist_trainee(trainee_id)

        # Update trainee in cache
        cached_trainee = self.trainee_cache.get_item(trainee_id)
        cached_trainee["feature_attributes"] = updated_feature_attributes
        return updated_feature_attributes

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
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Getting feature attributes from Trainee with id: {trainee_id}')
        feature_attributes = self.execute(trainee_id, "get_feature_attributes", {})
        return internals.postprocess_feature_attributes(feature_attributes)

    def get_sessions(self, trainee_id: str) -> list[dict[str, str]]:
        """
        Get all sessions in a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to get the list of sessions from.

        Returns
        -------
        list of dict of str to str
            A list of dicts with keys "id" and "name" for each session
            in the Trainee.

        Examples
        --------
        >>> print(cl.get_sessions(trainee.id))
        [{'id': '6c35e481-fb49-4178-a96f-fe4b5afe7af4', 'name': 'default'}]
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Getting sessions from Trainee with id: {trainee_id}')
        result = self.execute(trainee_id, "get_sessions", {"attributes": ['name', ]})
        if not result:
            return []
        return result

    def delete_session(self, trainee_id: str, session: str | Session):
        """
        Delete a session from a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to delete the session from.
        session : str
            The session or session identifier to delete.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if isinstance(session, Session):
            session = session.id
        if self.configuration.verbose:
            print(f'Deleting session {session} from Trainee with id: {trainee_id}')
        self.execute(trainee_id, "delete_session", {"session": session})
        self._auto_persist_trainee(trainee_id)

    def get_session_indices(self, trainee_id: str, session: str | Session) -> list[int]:
        """
        Get list of all session indices for a specified session.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee get parameters from.
        session : str
            The session or session identifier to retrieve indices of.

        Returns
        -------
        list of int
            A list of the session indices for the session.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if isinstance(session, Session):
            session = session.id
        if self.configuration.verbose:
            print(f'Getting session {session} indices from Trainee with id: {trainee_id}')
        result = self.execute(trainee_id, "get_session_indices", {"session": session})
        if result is None:
            return []
        return result

    def get_session_training_indices(self, trainee_id: str, session: str | Session) -> list[int]:
        """
        Get list of all session training indices for a specified session.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee get parameters from.
        session : str or Session
            The session or session identifier to retrieve indices of.

        Returns
        -------
        list of int
            A list of the session training indices for the session.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if isinstance(session, Session):
            session = session.id
        if self.configuration.verbose:
            print(f'Getting session {session} training indices from Trainee with id: {trainee_id}')
        result = self.execute(trainee_id, "get_session_training_indices", {"session": session})
        if result is None:
            return []
        return result

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
        condition : Mapping or None, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'].format("precision"))

        # Convert session instance to id
        if (
            isinstance(condition, MutableMapping) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Getting feature marginal stats for trainee with id: {trainee_id}')

        stats = self.execute(trainee_id, "get_marginal_stats", {
            "condition": condition,
            "num_cases": num_cases,
            "precision": precision,
            "weight_feature": weight_feature,
        })
        if stats is None:
            return dict()
        return stats

    def react(  # noqa: C901
        self,
        trainee_id: str,
        *,
        action_features: t.Optional[Collection[str]] = None,
        actions: t.Optional[TabularData2D] = None,
        allow_nulls: bool = False,
        batch_size: t.Optional[int] = None,
        case_indices: t.Optional[CaseIndices] = None,
        contexts: t.Optional[TabularData2D] = None,
        context_features: t.Optional[Collection[str]] = None,
        derived_action_features: t.Optional[Collection[str]] = None,
        derived_context_features: t.Optional[Collection[str]] = None,
        desired_conviction: t.Optional[float] = None,
        details: t.Optional[Mapping] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        feature_bounds_map: t.Optional[Mapping] = None,
        generate_new_cases: GenerateNewCases = "no",
        initial_batch_size: t.Optional[int] = None,
        input_is_substituted: bool = False,
        into_series_store: t.Optional[str] = None,
        leave_case_out: bool = False,
        new_case_threshold: NewCaseThreshold = "min",
        num_cases_to_generate: int = 1,
        ordered_by_specified_features: bool = False,
        post_process_features: t.Optional[Collection[str]] = None,
        post_process_values: t.Optional[TabularData2D] = None,
        preserve_feature_values: t.Optional[Collection[str]] = None,
        progress_callback: t.Optional[Callable] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: t.Optional[bool] = None,
        use_regional_model_residuals: bool = True,
        weight_feature: t.Optional[str] = None,
    ) -> Reaction:
        r"""
        React to supplied values and cases contained within the Trainee.

        If desired_conviction is not specified, executes a discriminative
        react: provided a list of context values, the trainee reacts to the
        model and produces predictions for the specified actions. If
        desired_conviction is specified, executes a generative react,
        produces action_values for the specified action_features conditioned
        on the optionally provided contexts.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.

        contexts : list of list of object or DataFrame, optional
            A 2d list of context values to react to.
            If None for discriminative react, it is assumed that `session`
            and `session_id` keys are set in the `details`.

            >>> contexts = [[1, 2, 3], [4, 5, 6]]

        action_features : iterable of str, optional
            An iterable of feature names to treat as action features during
            react.

            >>> action_features = ['rain_chance', 'is_sunny']

        actions : list of list of object or DataFrame, optional
            One or more action values to use for action features.
            If specified, will only return the specified explanation
            details for the given actions. (Discriminative reacts only)

            >>> actions = [[1, 2, 3], [4, 5, 6]]

        allow_nulls : bool, default False
            When true will allow return of null values if there
            are nulls in the local model for the action features, applicable
            only to discriminative reacts.

        batch_size: int, optional
            Define the number of cases to react to at once. If left unspecified,
            the batch size will be determined automatically.

        context_features : iterable of str, optional
            An iterable of feature names to treat as context features during
            react.

            >>> context_features = ['temperature', 'humidity', 'dew_point',
            ...                     'barometric_pressure']
        derived_context_features : iterable of str, optional
            An iterable of feature names whose values should be computed
            from the provided context in the specified order. Must be different
            than context_features.
        derived_action_features : iterable of str, optional
            An iterable of feature names whose values should be computed
            after generation from the generated case prior to output, in the
            specified order. Must be a subset of action_features.

            .. note::

                Both of these derived feature lists rely on the features'
                "derived_feature_code" attribute to compute the values. If
                'derived_feature_code' attribute is undefined or references
                non-0 feature indices, the derived value will be null.

        input_is_substituted : bool, default False
            if True assumes provided categorical (nominal or
            ordinal) feature values have already been substituted.
        substitute_output : bool, default True
            If False, will not substitute categorical feature
            values. Only applicable if a substitution value map has been set.
        details : dict, optional
            If details are specified, the response will contain the requested
            explanation data along with the reaction. Below are the valid keys
            and data types for the different audit details. Omitted keys,
            values set to None, or False values for Booleans will not be
            included in the audit data returned.

            - boundary_cases : bool, optional
                If True, outputs an automatically determined (when
                'num_boundary_cases' is not specified) relevant number of
                boundary cases. Uses both context and action features of the
                reacted case to determine the counterfactual boundary based on
                action features, which maximize the dissimilarity of action
                features while maximizing the similarity of context features.
                If action features aren't specified, uses familiarity conviction
                to determine the boundary instead.
            - boundary_cases_familiarity_convictions : bool, optional
                If True, outputs familiarity conviction of addition for each of
                the boundary cases.
            - case_contributions_full : bool, optional
                If true outputs each influential case's differences between the
                predicted action feature value and the predicted action feature
                value if each individual case were not included. Uses only the
                context features of the reacted case to determine that area.
                Uses full calculations, which uses leave-one-out for cases for
                computations.
            - case_contributions_robust : bool, optional
                If true outputs each influential case's differences between the
                predicted action feature value and the predicted action feature
                value if each individual case were not included. Uses only the
                context features of the reacted case to determine that area.
                Uses robust calculations, which uses uniform sampling from
                the power set of all combinations of cases.
            - case_feature_residuals_full : bool, optional
                If True, outputs feature residuals for all (context and action)
                features for just the specified case. Uses leave-one-out for
                each feature, while using the others to predict the left out
                feature with their corresponding values from this case. Uses
                full calculations, which uses leave-one-out for cases for
                computations.
            - case_feature_residuals_robust : bool, optional
                If True, outputs feature residuals for all (context and action)
                features for just the specified case. Uses leave-one-out for
                each feature, while using the others to predict the left out
                feature with their corresponding values from this case. Uses
                robust calculations, which uses uniform sampling from the power
                set of features as the contexts for predictions.
            - case_mda_robust : bool, optional
                If True, outputs each influential case's mean decrease in
                accuracy of predicting the action feature in the local model
                area, as if each individual case were included versus not
                included. Uses only the context features of the reacted case to
                determine that area. Uses robust calculations, which uses
                uniform sampling from the power set of all combinations of cases.
            - case_mda_full : bool, optional
                If True, outputs each influential case's mean decrease in
                accuracy of predicting the action feature in the local model
                area, as if each individual case were included versus not
                included. Uses only the context features of the reacted case to
                determine that area. Uses full calculations, which uses
                leave-one-out for cases for  computations.
            - categorical_action_probabilities : bool, optional
                If True, outputs probabilities for each class for the action.
                Applicable only to categorical action features.
            - derivation_parameters : bool, optional
                If True, outputs a dictionary of the parameters used in the
                react call. These include k, p, distance_transform,
                feature_weights, feature_deviations, nominal_class_counts,
                and use_irw.

                - k: the number of cases used for the local model.
                - p: the parameter for the Lebesgue space.
                - distance_transform: the distance transform used as an
                  exponent to convert distances to raw influence weights.
                - feature_weights: the weight for each feature used in the
                  distance metric.
                - feature_deviations: the deviation for each feature used in
                  the distance metric.
                - nominal_class_counts: the number of unique values for each
                  nominal feature. This is used in the distance metric.
                - use_irw: a flag indicating if feature weights were
                  derived using inverse residual weighting.
            - distance_contribution : bool, optional
                If True, outputs the distance contribution (expected total
                surprisal contribution) for the reacted case. Uses both context
                and action feature values.
            - distance_ratio : bool, optional
                If True, outputs the ratio of distance (relative surprisal)
                between this reacted case and its nearest case to the minimum
                distance (relative surprisal) in between the closest two cases
                in the local area. All distances are computed using only the
                specified context features.
            - feature_contributions_robust : bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context were not in the
                model for all context features in the local model area Uses
                robust calculations, which uses uniform sampling from the power
                set of features as the contexts for predictions. Directional feature
                contributions are returned under the key
                'directional_feature_contributions_robust'.
            - feature_contributions_full : bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context were not in the
                model for all context features in the local model area. Uses
                full calculations, which uses leave-one-out for cases for
                computations. Directional feature contributions are returned
                under the key 'directional_feature_contributions_full'.
            - case_feature_contributions_robust: bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context feature were not
                in the model for all context features in this case, using only
                the values from this specific case. Uses
                robust calculations, which uses uniform sampling from the power
                set of features as the contexts for predictions.
                Directional case feature contributions are returned under the
                'case_directional_feature_contributions_robust' key.
            - case_feature_contributions_full: bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context feature were not
                in the model for all context features in this case, using only
                the values from this specific case. Uses
                full calculations, which uses leave-one-out for cases for
                computations. Directional case feature
                contributions are returned under the
                'case_directional_feature_contributions_full' key.
            - feature_mda_robust : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature given the context.
                Uses only the context features of the reacted case to determine
                that area. Uses robust calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - feature_mda_full : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature given the context.
                Uses only the context features of the reacted case to determine
                that area. Uses full calculations, which uses leave-one-out
                for cases for computations.
            - feature_mda_ex_post_robust : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature as an explanation detail
                given that the specified prediction was already made as
                specified by the action value. Uses both context and action
                features of the reacted case to determine that area. Uses
                robust calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - feature_mda_ex_post_full : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature as an explanation detail
                given that the specified prediction was already made as
                specified by the action value. Uses both context and action
                features of the reacted case to determine that area. Uses
                full calculations, which uses leave-one-out for cases for
                computations.
            - features : list of str, optional
                A list of feature names that specifies for what features will
                per-feature details be computed (residuals, contributions,
                mda, etc.). This should generally preserve compute, but will
                not when computing details robustly. Details will be computed
                for all context and action features if this value is not
                specified.
            - feature_residual_robust : bool, optional
                If True, outputs feature residuals for all (context and action)
                features locally around the prediction. Uses only the context
                features of the reacted case to determine that area. Uses robust
                calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - feature_residuals_full : bool, optional
                If True, outputs feature residuals for all (context and action)
                features locally around the prediction. Uses only the context
                features of the reacted case to determine that area. Uses
                full calculations, which uses leave-one-out for cases for computations.
            - hypothetical_values : dict, optional
                A dictionary of feature name to feature value. If specified,
                shows how a prediction could change in a what-if scenario where
                the influential cases' context feature values are replaced with
                the specified values.  Iterates over all influential cases,
                predicting the action features each one using the updated
                hypothetical values. Outputs the predicted arithmetic over the
                influential cases for each action feature.
            - influential_cases : bool, optional
                If True, outputs the most influential cases and their influence
                weights based on the surprisal of each case relative to the
                context being predicted among the cases. Uses only the context
                features of the reacted case.
            - influential_cases_familiarity_convictions :  bool, optional
                If True, outputs familiarity conviction of addition for each of
                the influential cases.
            - influential_cases_raw_weights : bool, optional
                If True, outputs the surprisal for each of the influential
                cases.
            - case_feature_residual_convictions_robust : bool, optional
                If True, outputs this case's feature residual convictions for
                the region around the prediction. Uses only the context
                features of the reacted case to determine that region.
                Computed as: region feature residual divided by case feature
                residual. Uses robust calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - case_feature_residual_convictions_full : bool, optional
                If True, outputs this case's feature residual convictions for
                the region around the prediction. Uses only the context
                features of the reacted case to determine that region.
                Computed as: region feature residual divided by case feature
                residual. Uses full calculations, which uses leave-one-out
                for cases for computations.
            - most_similar_cases : bool, optional
                If True, outputs an automatically determined (when
                'num_most_similar_cases' is not specified) relevant number of
                similar cases, which will first include the influential cases.
                Uses only the context features of the reacted case.
            - num_boundary_cases : int, optional
                Outputs this manually specified number of boundary cases.
            - num_most_similar_cases : int, optional
                Outputs this manually specified number of most similar cases,
                which will first include the influential cases.
            - num_most_similar_case_indices : int, optional
                Outputs this specified number of most similar case indices when
                'distance_ratio' is also set to True.
            - num_robust_influence_samples_per_case : int, optional
                Specifies the number of robust samples to use for each case.
                Applicable only for computing robust feature contributions or
                robust case feature contributions. Defaults to 2000. Higher
                values will take longer but provide more stable results.
            - observational_errors : bool, optional
                If True, outputs observational errors for all features as
                defined in feature attributes.
            - outlying_feature_values : bool, optional
                If True, outputs the reacted case's context feature values that
                are outside the min or max of the corresponding feature values
                of all the cases in the local model area. Uses only the context
                features of the reacted case to determine that area.
            - prediction_stats : bool, optional
                When true outputs feature prediction stats for all (context
                and action) features locally around the prediction. The stats
                returned  are ("r2", "rmse", "spearman_coeff", "precision",
                "recall", "accuracy", "mcc", "confusion_matrix", "missing_value_accuracy").
                Uses only the context features of the reacted case to determine that area.
                Uses full calculations, which uses leave-one-out context features for
                computations.
            - selected_prediction_stats : list, optional. List of stats to output. When unspecified,
                returns all except the confusion matrix. Allowed values:

                - all : Returns all the the available prediction stats, including the confusion matrix.
                - accuracy : The number of correct predictions divided by the
                  total number of predictions.
                - confusion_matrix : A sparse map of actual feature value to a map of
                  predicted feature value to counts.
                - mae : Mean absolute error. For continuous features, this is
                  calculated as the mean of absolute values of the difference
                  between the actual and predicted values. For nominal features,
                  this is 1 - the average categorical action probability of each case's
                  correct classes. Categorical action probabilities are the probabilities
                  for each class for the action feature.
                - mda : Mean decrease in accuracy when each feature is dropped
                  from the model, applies to all features.
                - feature_mda_permutation_full : Mean decrease in accuracy that used
                  scrambling of feature values instead of dropping each
                  feature, applies to all features.
                - precision : Precision (positive predictive) value for nominal
                  features only.
                - r2 : The r-squared coefficient of determination, for
                  continuous features only.
                - recall : Recall (sensitivity) value for nominal features only.
                - rmse : Root mean squared error, for continuous features only.
                - spearman_coeff : Spearman's rank correlation coefficient,
                  for continuous features only.
                - mcc : Matthews correlation coefficient, for nominal features only.
            - similarity_conviction : bool, optional
                If True, outputs similarity conviction for the reacted case.
                Uses both context and action feature values as the case values
                for all computations. This is defined as expected (local)
                distance contribution divided by reacted case distance
                contribution.
            - generate_attempts : bool, optional
                If True outputs the number of attempts taken to generate each
                case. Only applicable when 'generate_new_cases' is "always" or
                "attempt".

            >>> details = {'num_most_similar_cases': 5,
            ...            'feature_residuals_full': True}

        desired_conviction : float
            If specified will execute a generative react. If not
            specified will executed a discriminative react. Conviction is the
            ratio of expected surprisal to generated surprisal for each
            feature generated, valid values are in the range of
            :math:`(0, \\infty)`.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.
        case_indices : Iterable of Sequence[Union[str, int]], defaults to None
            An Iterable of Sequences, of session id and index, where
            index is the original 0-based index of the case as it was trained
            into the session. If this case does not exist, discriminative react
            outputs null, generative react ignores it.
        preserve_feature_values : iterable of str
            List of features that will preserve their values from
            the case specified by case_indices, appending and overwriting the
            specified contexts as necessary.  For generative reacts, if
            case_indices isn't specified will preserve feature values of a
            random case.
        leave_case_out : bool, default False
            If set to True and specified along with case_indices,
            each individual react will respectively ignore the corresponding
            case specified by case_indices by leaving it out.
        initial_batch_size: int, optional
            Define the number of cases to react to in the first batch. If
            unspecified, the value of the ``react_initial_batch_size`` property
            is used. The number of cases in following batches will be
            automatically adjusted. This value is ignored if ``batch_size`` is
            specified.
        into_series_store : str, optional
            The name of a series store. If specified, will store an internal
            record of all react contexts for this session and series to be used
            later with train series.
        use_regional_model_residuals : bool
            If false uses model feature residuals, if True
            recalculates regional model residuals.
        feature_bounds_map : dict of dict
            A mapping of feature names to the bounds for the
            feature values to be generated in. For continuous features this
            should be a numeric value, for datetimes this should be a datetime
            string. Min bounds should be equal to or smaller than max bounds,
            except when setting the bounds around the cycle length of a cyclic
            feature.(e.g., to allow 0 +/- 60 degrees, set min=300 and max=60).

            .. code-block::
                :caption: Example feature bounds map:

                {
                    "feature_a": {"min": 0},
                    "feature_b" : {"min": 1, "max": 5},
                    "feature_c": {"max": 1}
                }

        generate_new_cases : {"always", "attempt", "no"}, default "no"
            (Optional) Whether to generate new cases.

            This parameter takes in a string equal to one of the following:

            a. "attempt"

                `Synthesizer` attempts to generate new cases and
                if its not possible to generate a new case, it might
                generate cases in "no" mode (see point c.)
            b. "always"

                `Synthesizer` always generates new cases and
                if its not possible to generate a new case, it returns
                `None`.
            c. "no"

                `Synthesizer` generates data based on the
                `desired_conviction` specified and the generated data is
                not guaranteed to be a new case (that is, a case not found
                in original dataset.)

        ordered_by_specified_features : bool, default False
            If True order of generated feature values will match
            the order of specified features.
        num_cases_to_generate : int, default 1
            The number of cases to generate.
        suppress_warning : bool, defaults to False
            If True, warnings will not be displayed.
        post_process_features : iterable of str, optional
            List of feature names that will be made available during the
            execution of post_process feature attributes.
        post_process_values : list of list of object or DataFrame, optional
            A 2d list of values corresponding to post_process_features that
            will be made available during the execution of post_process feature
            attributes.
        progress_callback : callable, optional
            A callback method that will be called before each
            batched call to react and at the end of reacting. The method is
            given a ProgressTimer containing metrics on the progress and timing
            of the react operation, and the batch result.
        new_case_threshold : str, optional
            Distance to determine the privacy cutoff. If None,
            will default to "min".

            Possible values:

                - min: minimum distance in the original local space.
                - max: maximum distance in the original local space.
                - most_similar: distance between the nearest neighbor to the
                  nearest neighbor in the original space.
        exclude_novel_nominals_from_uniqueness_check : bool, default False
            If True, will exclude features which have a subtype defined in their feature
            attributes from the uniqueness check that happens when ``generate_new_cases``
            is True. Only applies to generative reacts.

        Returns
        -------
        Reaction:
            A MutableMapping (dict-like) with these keys -> values:
                action -> pandas.DataFrame
                    A data frame of action values.

                details -> Dict or List
                    An aggregated list of any requested details.

        Raises
        ------
        ValueError
            If `derived_action_features` is not a subset of `action_features`.

            If `new_case_threshold` is not one of {"max", "min", "most_similar"}.

            If the number of context values does not match the number of context features.
        HowsoError
            If `num_cases_to_generate` is not an integer greater than 0.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)

        action_features, actions, context_features, contexts = (
            self._preprocess_react_parameters(
                action_features=action_features,
                action_values=actions,
                case_indices=case_indices,
                context_features=context_features,
                context_values=contexts,
                desired_conviction=desired_conviction,
                preserve_feature_values=preserve_feature_values,
                trainee_id=trainee_id
            )
        )

        if post_process_values is not None and post_process_features is None:
            post_process_features = internals.get_features_from_data(
                post_process_values,
                data_parameter='post_process_values',
                features_parameter='post_process_features')
        post_process_values = serialize_cases(post_process_values, post_process_features, feature_attributes)

        if post_process_values is not None and contexts is not None:
            if (len(contexts) > 1 and len(post_process_values) > 1 and
                    len(contexts) != len(post_process_values)):
                raise ValueError(
                    "If more than one value is provided for 'contexts' "
                    "and 'post_process_values', then they must be of the same "
                    "length."
                )

        if action_features is not None and derived_action_features is not None:
            if not set(derived_action_features).issubset(set(action_features)):
                raise ValueError(
                    'Specified `derived_action_features` must be a subset of '
                    '`action_features`.')

        if new_case_threshold not in [None, "min", "max", "most_similar"]:
            raise ValueError(
                f"The value '{new_case_threshold}' specified for the parameter "
                "`new_case_threshold` is not valid. It accepts one of the"
                " following values - ['min', 'max', 'most_similar',]"
            )

        if details is not None and 'robust_computation' in details:
            details = dict(details)
            details['robust_influences'] = details['robust_computation']
            details['robust_residuals'] = details['robust_computation']
            del details['robust_computation']
            warnings.warn(
                'The detail "robust_computation" is deprecated and will be '
                'removed in a future release. Please use "robust_residuals" '
                'and/or "robust_influences" instead.', DeprecationWarning)

        if details is not None and 'local_case_feature_residual_conviction_robust' in details:
            details = dict(details)
            details['case_feature_residual_conviction_robust'] = details.pop(
                'local_case_feature_residual_conviction_robust')
            warnings.warn(
                'The detail "local_case_feature_residual_conviction_robust" is deprecated and will be '
                'removed in a future release. Please use "case_feature_residual_conviction_robust" instead',
                DeprecationWarning)

        if details is not None and 'local_case_feature_residual_conviction_full' in details:
            details = dict(details)
            details['case_feature_residual_conviction_full'] = details.pop(
                'local_case_feature_residual_conviction_full')
            warnings.warn(
                'The detail "local_case_feature_residual_conviction_full" is deprecated and will be '
                'removed in a future release. Please use "case_feature_residual_conviction_full" instead',
                DeprecationWarning)

        if desired_conviction is None:
            if contexts is not None:
                for context in contexts:
                    if context is not None and len(context) != len(context_features or []):
                        raise ValueError(
                            "The number of provided context values in "
                            "`contexts` does not match the number of features "
                            "in `context_features`."
                        )
                total_size = len(contexts)
            elif case_indices is not None:
                total_size = len(case_indices)
            else:
                total_size = 1

            # discriminative react parameters
            react_params = {
                "action_values": actions,
                "context_features": context_features,
                "context_values": contexts,
                "action_features": action_features,
                "derived_context_features": derived_context_features,
                "derived_action_features": derived_action_features,
                "post_process_features": post_process_features,
                "post_process_values": post_process_values,
                "case_indices": case_indices,
                "allow_nulls": allow_nulls,
                "input_is_substituted": input_is_substituted,
                "substitute_output": substitute_output,
                "weight_feature": weight_feature,
                "use_case_weights": use_case_weights,
                "leave_case_out": leave_case_out,
                "preserve_feature_values": preserve_feature_values,
                "new_case_threshold": new_case_threshold,
                "details": details,
            }
        else:
            if (
                not isinstance(num_cases_to_generate, int) or
                num_cases_to_generate <= 0
            ):
                raise HowsoError("`num_cases_to_generate` must be an integer greater than 0.")
            total_size = num_cases_to_generate

            context_features, contexts = \
                self._preprocess_generate_parameters(
                    trainee_id,
                    desired_conviction=desired_conviction,
                    num_cases_to_generate=num_cases_to_generate,
                    action_features=action_features,
                    context_features=context_features,
                    context_values=contexts,
                    case_indices=case_indices
                )

            # generative react parameters
            react_params = {
                "num_cases_to_generate": num_cases_to_generate,
                "context_features": context_features,
                "context_values": contexts,
                "action_features": action_features,
                "derived_context_features": derived_context_features,
                "derived_action_features": derived_action_features,
                "post_process_features": post_process_features,
                "post_process_values": post_process_values,
                "use_regional_model_residuals": use_regional_model_residuals,
                "desired_conviction": desired_conviction,
                "feature_bounds_map": feature_bounds_map,
                "generate_new_cases": generate_new_cases,
                "ordered_by_specified_features": ordered_by_specified_features,
                "preserve_feature_values": preserve_feature_values,
                "new_case_threshold": new_case_threshold,
                "into_series_store": into_series_store,
                "input_is_substituted": input_is_substituted,
                "substitute_output": substitute_output,
                "weight_feature": weight_feature,
                "use_case_weights": use_case_weights,
                "case_indices": case_indices,
                "leave_case_out": leave_case_out,
                "details": details,
                "exclude_novel_nominals_from_uniqueness_check": exclude_novel_nominals_from_uniqueness_check,
            }

        if batch_size or self._should_react_batch(react_params, total_size):
            # Run in batch
            if self.configuration.verbose:
                print(f'Batch reacting to context on Trainee with id: {trainee_id}')
            response = self._batch_react(
                trainee_id,
                react_params,
                batch_size=batch_size,
                initial_batch_size=initial_batch_size,
                total_size=total_size,
                progress_callback=progress_callback
            )
        else:
            # Run as a single react request
            if self.configuration.verbose:
                print(f'Reacting to context on Trainee with id: {trainee_id}')
            with ProgressTimer(total_size) as progress:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress, None)
                response, _, _ = self._react(trainee_id, react_params)
                progress.update(total_size)

            if isinstance(progress_callback, Callable):
                progress_callback(progress, response)

        response = internals.format_react_response(response)

        if desired_conviction is not None:
            # If the number of cases generated is less then requested, raise
            # warning (only for generative reacts)
            internals.insufficient_generation_check(
                num_cases_to_generate, len(response['action']),
                suppress_warning=suppress_warning
            )

        return Reaction(response.get('action'), response.get('details'))

    def _batch_react(
        self,
        trainee_id: str,
        params: dict,
        *,
        batch_size: t.Optional[int] = None,
        initial_batch_size: t.Optional[int] = None,
        total_size: int,
        progress_callback: t.Optional[Callable] = None
    ):
        """
        Make react requests in batch.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
        params : dict
            The engine react parameters.
        batch_size : int, optional
            Define the number of cases to react to at once. If left unspecified,
            the batch size will be determined automatically.
        initial_batch_size : int, optional
            Define the number of cases to react to in the first batch. If
            unspecified, the value of the ``react_initial_batch_size`` property
            is used. The number of cases in following batches will be
            automatically adjusted. This value is ignored if ``batch_size`` is
            specified.
        total_size : int
            The total size of the data that will be batched.
        progress_callback : callable, optional
            A method to be called during batching to retrieve the progress
            metrics.

        Returns
        -------
        dict
            The react response.
        """
        temp_result = None
        accumulated_result = {'action_values': []}

        actions = params.get('action_values')
        contexts = params.get('context_values')
        case_indices = params.get('case_indices')
        post_process_values = params.get('post_process_values')

        with ProgressTimer(total_size) as progress:
            gen_batch_size = None
            batch_scaler = None
            if not batch_size:
                # Scale the batch size automatically
                start_batch_size = initial_batch_size or self.react_initial_batch_size
                batch_scaler = self.batch_scaler_class(start_batch_size, progress)
                gen_batch_size = batch_scaler.gen_batch_size()
                batch_size = next(gen_batch_size, None)

            while not progress.is_complete and batch_size is not None:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress, temp_result)
                batch_start = progress.current_tick
                batch_end = progress.current_tick + batch_size

                if actions is not None and len(actions) > 1:
                    params['action_values'] = actions[batch_start:batch_end]
                if contexts is not None and len(contexts) > 1:
                    params['context_values'] = contexts[batch_start:batch_end]
                if case_indices is not None and len(case_indices) > 1:
                    params['case_indices'] = case_indices[batch_start:batch_end]
                if post_process_values is not None and len(post_process_values) > 1:
                    params['post_process_values'] = post_process_values[batch_start:batch_end]

                if params.get('desired_conviction') is not None:
                    params['num_cases_to_generate'] = batch_size
                temp_result, in_size, out_size = self._react(trainee_id, params)

                internals.accumulate_react_result(accumulated_result, temp_result)
                if batch_scaler is None or gen_batch_size is None:
                    progress.update(batch_size)
                else:
                    batch_size = batch_scaler.send(
                        gen_batch_size,
                        batch_scaler.SendOptions(None, (in_size, out_size)))

        # Final call to callback on completion
        if isinstance(progress_callback, Callable):
            progress_callback(progress, temp_result)

        return accumulated_result

    def _react(self, trainee_id: str, params: dict) -> tuple[dict, int, int]:
        """
        Make a single react request.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
        params : dict
            The engine react parameters.

        Returns
        -------
        dict
            The react response.
        int
            The request payload size.
        int
            The response payload size.
        """
        ret, in_size, out_size = self.execute_sized(trainee_id, 'react', params)

        # Action values and features should always be defined
        if not ret.get('action_values'):
            ret['action_values'] = []
        if not ret.get('action_features'):
            ret['action_features'] = []

        return ret, in_size, out_size

    def _preprocess_react_parameters(
        self,
        trainee_id: str,
        *,
        action_features: t.Optional[Collection[str]],
        action_values: t.Optional[TabularData2D],
        case_indices: t.Optional[CaseIndices],
        context_features: t.Optional[Collection[str]],
        context_values: t.Optional[TabularData2D],
        desired_conviction: t.Optional[float],
        preserve_feature_values: t.Optional[Collection[str]],
        continue_series: bool = False,
    ) -> tuple[Collection[str] | None, list[t.Any] | None, Collection[str] | None, list[t.Any] | None]:
        """
        Preprocess parameters for `react_` methods.

        Helper method to pre-process the parameters. Updates the passed-in
        parameters as necessary.

        Parameters
        ----------
        trainee_id : str
            See :meth:`AbstractHowsoClient.react()`.
        action_features : Collection of str, optional
            See :meth:`AbstractHowsoClient.react()`.
        action_values : Collection of object, optional
            See :meth:`AbstractHowsoClient.react()`.
        context_features : Collection of str, optional
            See :meth:`AbstractHowsoClient.react()`.
        context_values : Collection of object, optional
            See :meth:`AbstractHowsoClient.react()`.
        desired_conviction : float, optional
            See :meth:`AbstractHowsoClient.react()`.
        case_indices : Sequence of tuple of {str, int}, optional
            See :meth:`AbstractHowsoClient.react()`.
        preserve_feature_values : Collection of str, optional
            See :meth:`AbstractHowsoClient.react()`.
        continue_series : bool
            See :meth:`AbstractHowsoClient.react_series()`.

        Returns
        -------
        tuple
           Updated action_features, actions, context_features, contexts
        """
        feature_attributes = self.resolve_feature_attributes(trainee_id)

        # Validate case_indices if provided
        if case_indices is not None:
            util.validate_case_indices(case_indices)

        # validate discriminative-react only parameters
        if desired_conviction is None:
            util.validate_list_shape(context_values, 2, "contexts", "list of object")
            util.validate_list_shape(action_features, 1, "action_features", "str")
            util.validate_list_shape(context_features, 1, "context_features", "str")

            if context_values is None:
                # case_indices/preserve_feature_values are not necessary
                # when using continue_series, as initial_feature/values may be used
                if not continue_series and action_values is None and (
                    case_indices is None or preserve_feature_values is None
                ):
                    raise ValueError(
                        "If `contexts` are not specified, both `case_indices` "
                        "and `preserve_feature_values` must be specified."
                    )

        # Preprocess context values
        if context_values is not None and context_features is None:
            context_features = internals.get_features_from_data(
                context_values,
                data_parameter='contexts',
                features_parameter='context_features')
        serialized_contexts = serialize_cases(context_values, context_features, feature_attributes)

        # Preprocess action values
        if action_values is not None and action_features is None:
            util.validate_list_shape(action_values, 2, "actions", "object")
            action_features = internals.get_features_from_data(
                action_values,
                data_parameter='actions',
                features_parameter='action_features')
        serialized_actions = serialize_cases(action_values, action_features, feature_attributes)

        return action_features, serialized_actions, context_features, serialized_contexts

    def _preprocess_generate_parameters(  # noqa: C901
        self,
        trainee_id: str,
        desired_conviction: float,
        num_cases_to_generate: int,
        *,
        action_features: t.Optional[Collection[str]] = None,
        case_indices: t.Optional[CaseIndices] = None,
        context_features: t.Optional[Collection[str]] = None,
        context_values: t.Optional[TabularData2D] = None,
    ) -> tuple[Collection[str] | None, TabularData2D | None]:
        """
        Helper method to pre-process generative react parameters.

        Updates the passed-in parameters as necessary.

        Parameters
        ----------
        trainee_id : str
            See :meth:`AbstractHowsoClient.react`.
        desired_conviction : float, optional
            See :meth:`AbstractHowsoClient.react`.
        num_cases_to_generate : int, optional
            See :meth:`AbstractHowsoClient.react`.
        action_features : Collection of str, optional
            See :meth:`AbstractHowsoClient.react`.
        case_indices : Sequence of tuple of {str, int}, optional
            See :meth:`AbstractHowsoClient.react`.
        context_features : Collection of str, optional
            See :meth:`AbstractHowsoClient.react`.
        context_values : list of object, optional
            See :meth:`AbstractHowsoClient.react`.

        Returns
        -------
        tuple
            The prepared context_features and context_values.
        """
        if case_indices is not None:
            if len(case_indices) != 1 and \
                    len(case_indices) != num_cases_to_generate:
                raise HowsoError(
                    "The number of `case_indices` provided does not match "
                    "the number of cases to generate."
                )

        # Validate case_indices if provided
        if case_indices is not None:
            util.validate_case_indices(case_indices)

        util.validate_list_shape(action_features, 1, "action_features", "str")

        if context_values is not None:
            if len(context_values) != 1 and len(context_values) != num_cases_to_generate:
                raise HowsoError(
                    "The number of case `contexts` provided does not match the "
                    "number of cases to generate."
                )

        if context_features and not context_values:
            raise HowsoError(
                "For generative reacts, when `context_features` are provided, "
                "`contexts` values must also be provided."
            )

        if context_features is not None or context_values is not None:
            if (
                context_features is None
                or context_values is None
                or not isinstance(context_values[0], Sized)
                or len(context_features) != len(context_values[0])
            ):
                raise HowsoError(
                    "The number of provided context values in `contexts` "
                    "does not match the number of features in "
                    "`context_features`."
                )

        if desired_conviction <= 0:
            raise HowsoError("Desired conviction must be greater than 0.")

        if self.configuration.verbose:
            print(f'Generating case from trainee with id: {trainee_id}')

        for i in range(num_cases_to_generate):
            target_context_values = None
            if context_values is not None:
                if len(context_values) == 1:
                    target_context_values = context_values[0]
                elif len(context_values) == num_cases_to_generate:
                    target_context_values = context_values[i]

            if context_features and (
                not target_context_values or
                not isinstance(target_context_values, Sized) or
                len(target_context_values) != len(context_features)
            ):
                raise HowsoError(
                    f"The number of provided context values in `contexts[{i}]` "
                    "does not match the number of features in `context_features`."
                )

        return context_features, context_values

    def react_series(  # noqa: C901
        self,
        trainee_id: str,
        *,
        action_features: t.Optional[Collection[str]] = None,
        actions: t.Optional[TabularData2D] = None,
        batch_size: t.Optional[int] = None,
        case_indices: t.Optional[CaseIndices] = None,
        contexts: t.Optional[TabularData2D] = None,
        context_features: t.Optional[Collection[str]] = None,
        continue_series: bool = False,
        continue_series_features: t.Optional[Collection[str]] = None,
        continue_series_values: t.Optional[list[t.Any] | list[list[t.Any]]] = None,
        derived_action_features: t.Optional[Collection[str]] = None,
        derived_context_features: t.Optional[Collection[str]] = None,
        desired_conviction: t.Optional[float] = None,
        details: t.Optional[Mapping] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        feature_bounds_map: t.Optional[Mapping[str, Mapping[str, t.Any]]] = None,
        final_time_steps: t.Optional[list[t.Any]] = None,
        generate_new_cases: GenerateNewCases = "no",
        init_time_steps: t.Optional[list[t.Any]] = None,
        initial_batch_size: t.Optional[int] = None,
        initial_features: t.Optional[Collection[str]] = None,
        initial_values: t.Optional[TabularData2D] = None,
        input_is_substituted: bool = False,
        leave_case_out: bool = False,
        max_series_lengths: t.Optional[list[int]] = None,
        new_case_threshold: NewCaseThreshold = "min",
        num_series_to_generate: int = 1,
        ordered_by_specified_features: bool = False,
        output_new_series_ids: bool = True,
        preserve_feature_values: t.Optional[Collection[str]] = None,
        progress_callback: t.Optional[Callable] = None,
        series_context_features: t.Optional[Collection[str]] = None,
        series_context_values: t.Optional[TabularData3D] = None,
        series_id_tracking: SeriesIDTracking = "fixed",
        series_index: t.Optional[str] = None,
        series_stop_maps: t.Optional[list[Mapping[str, Mapping[str, t.Any]]]] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: t.Optional[bool] = None,
        use_regional_model_residuals: bool = True,
        weight_feature: t.Optional[str] = None
    ) -> Reaction:
        """
        React in a series until a series_stop_map condition is met.

        Aggregates rows of data corresponding to the specified context, action,
        derived_context and derived_action features, utilizing previous rows to
        derive values as necessary. Outputs a dict of "action_features" and
        corresponding "action" where "action" is the completed 'matrix' for the
        corresponding `action_features` and `derived_action_features`.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
        num_series_to_generate : int, default 1
            The number of series to generate.
        final_time_steps : list of object, optional
            The time steps at which to end synthesis. Time-series
            only. Must provide either one for all series, or exactly one per
            series.
        init_time_steps : list of object, optional
            The time steps at which to begin synthesis. Time-series
            only. Must provide either one for all series, or exactly one per
            series.
        initial_features : iterable of str, optional
            List of features to condition just the first case in a
            series, overwrites context_features and derived_context_features
            for that first case. All specified initial features must be in one
            of: context_features, action_features, derived_context_features or
            derived_action_features. If provided a value that isn't in one of
            those lists, it will be ignored.
        initial_values : list of list of object, optional
            2d list of values corresponding to the initial_features,
            used to condition just the first case in each series. Must provide
            either one for all series, or exactly one per series.
        series_stop_maps : list of dict of dict, optional
            A dictionary of feature name to stop conditions. Must provide either
            one for all series, or exactly one per series.

            .. TIP::
                Stop series when value exceeds max or is smaller than min::

                    {"feature_name":  {"min" : 1, "max": 2}}

                Stop series when feature value matches any of the values
                listed::

                    {"feature_name":  {"values": ["val1", "val2"]}}

        max_series_lengths : list of int, optional
            maximum size a series is allowed to be.  Default is
            3 * model_size, a 0 or less is no limit. If forecasting
            with ``continue_series``, this defines the maximum length of the
            forecast. Must provide either one for all series, or exactly
            one per series.
        continue_series : bool, default False
            When True will attempt to continue existing series instead of
            starting new series. If `initial_values` provide series IDs, it
            will continue those explicitly specified IDs, otherwise it will
            randomly select series to continue.
            .. note::

                Terminated series with terminators cannot be continued and
                will result in null output.
        continue_series_features : list of str, optional
            The list of feature names corresponding to the values in each row of
            `continue_series_values`. This value is ignored if
            `continue_series_values` is None.
        continue_series_values : list of list of list of object or list of pandas.DataFrame, default None
            The set of series data to be forecasted with feature values in the
            same order defined by `continue_series_values`. The value of
            `continue_series` will be ignored and treated as true if this value
            is specified.
        derived_context_features : iterable of str, optional
            List of context features whose values should be computed
            from the entire series in the specified order. Must be
            different than context_features.
        derived_action_features : iterable of str, optional
            List of action features whose values should be computed
            from the resulting last row in series, in the specified order.
            Must be a subset of action_features.

            .. note::

                Both of these derived feature lists rely on the features'
                "derived_feature_code" attribute to compute the values. If
                "derived_feature_code" attribute references non-existing
                feature indices, the derived value will be null.

        exclude_novel_nominals_from_uniqueness_check : bool, default False
            If True, will exclude features which have a subtype defined in their feature
            attributes from the uniqueness check that happens when ``generate_new_cases``
            is True. Only applies to generative reacts.
        series_context_features : iterable of str, optional
            List of context features corresponding to
            series_context_values, if specified must not overlap with any
            initial_features or context_features.
        series_context_values : list of list of list of object or list of DataFrame, optional
            3d-list of context values, one for each feature for each
            row for each series. If specified, max_series_lengths are ignored.
        output_new_series_ids : bool, default True
            If True, series ids are replaced with unique values on output.
            If False, will maintain or replace ids with existing trained values,
            but also allows output of series with duplicate existing ids.
        series_id_tracking : {"dynamic", "fixed", "no"}, default "fixed"
            Controls how closely generated series should follow existing series (plural).

            Choices are: "fixed" , "dynamic" or "no":

            - If "fixed", tracks the particular relevant series ID.
            - If "dynamic", tracks the particular relevant series ID, but is
              allowed to change the series ID that it tracks based on its
              current context.
            - If "no", does not track any particular series ID.
        series_index : str, Optional
            When set to a string, will include the series index as a
            column in the returned DataFrame using the column name given.
            If set to None, no column will be added.
        progress_callback : callable, optional
            A callback method that will be called before each
            batched call to react series and at the end of reacting. The method
            is given a ProgressTimer containing metrics on the progress and
            timing of the react series operation, and the batch result.
        batch_size: int, optional
            Define the number of series to react to at once. If left
            unspecified, the batch size will be determined automatically.
        initial_batch_size: int, optional
            The number of series to react to in the first batch. If unspecified,
            the number will be determined automatically. The number of series
            in following batches will be automatically adjusted. This value is
            ignored if ``batch_size`` is specified.
        contexts: list of list of object or DataFrame
            See parameter ``contexts`` in :meth:`AbstractHowsoClient.react`.
        action_features: iterable of str
            See parameter ``action_features`` in :meth:`AbstractHowsoClient.react`.
        actions: list of list of object or DataFrame
            See parameter ``actions`` in :meth:`AbstractHowsoClient.react`.
        context_features: iterable of str
            See parameter ``context_features`` in :meth:`AbstractHowsoClient.react`.
        input_is_substituted : bool, default False
            See parameter ``input_is_substituted`` in :meth:`AbstractHowsoClient.react`.
        substitute_output : bool
            See parameter ``substitute_output`` in :meth:`AbstractHowsoClient.react`.
        details: dict, optional
            See parameter ``details`` in :meth:`AbstractHowsoClient.react`.
        desired_conviction: float
            See parameter ``desired_conviction`` in :meth:`AbstractHowsoClient.react`.
        weight_feature : str
            See parameter ``weight_feature`` in :meth:`AbstractHowsoClient.react`.
        use_case_weights : bool, optional
            See parameter ``use_case_weights`` in :meth:`AbstractHowsoClient.react`.
        case_indices: iterable of sequence of str, int
            See parameter ``case_indices`` in :meth:`AbstractHowsoClient.react`.
        preserve_feature_values : iterable of str
            See parameter ``preserve_feature_values`` in :meth:`AbstractHowsoClient.react`.
        new_case_threshold : str
            See parameter ``new_case_threshold`` in :meth:`AbstractHowsoClient.react`.
        leave_case_out : bool
            See parameter ``leave_case_out`` in :meth:`AbstractHowsoClient.react`.
        use_regional_model_residuals : bool
            See parameter ``use_regional_model_residuals`` in :meth:`AbstractHowsoClient.react`.
        feature_bounds_map: dict of dict
            See parameter ``feature_bounds_map`` in :meth:`AbstractHowsoClient.react`.
        generate_new_cases : {"always", "attempt", "no"}
            See parameter ``generate_new_cases`` in :meth:`AbstractHowsoClient.react`.
        ordered_by_specified_features : bool
            See parameter ``ordered_by_specified_features`` in :meth:`AbstractHowsoClient.react`.
        suppress_warning : bool
            See parameter ``suppress_warning`` in :meth:`AbstractHowsoClient.react`.

        Returns
        -------
        Reaction:
            A MutableMapping (dict-like) with these keys -> values:
                action -> pandas.DataFrame
                    A data frame of action values.

                details -> Dict or List
                    An aggregated list of any requested details.

        Raises
        ------
        ValueError
            If the number of provided context values does not match the length of
            context features.

            If `series_context_values` is not a 3d list of objects.

            If `series_continue_values` is not a 3d list of objects.

            If `derived_action_features` is not a subset of `action_features`.

            If `new_case_threshold` is not one of {"max", "min", "most_similar"}.
        HowsoError
            If `num_series_to_generate` is not an integer greater than 0.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)

        util.validate_list_shape(initial_features, 1, "initial_features", "str")
        util.validate_list_shape(initial_values, 2, "initial_values", "list of object")
        util.validate_list_shape(initial_features, 1, "max_series_lengths", "num")
        util.validate_list_shape(series_stop_maps, 1, "series_stop_maps", "dict")
        util.validate_list_shape(series_context_features, 1, "series_context_features", "str")

        if continue_series_values and util.num_list_dimensions(continue_series_values) != 3:
            raise ValueError(
                "Improper shape of `continue_series_values` values passed. "
                "`continue_series_values` must be a 3d list of object.")
        if series_context_values and util.num_list_dimensions(series_context_values) != 3:
            raise ValueError(
                "Improper shape of `series_context_values` values passed. "
                "`series_context_values` must be a 3d list of object.")

        if continue_series_values is not None:
            continue_series = True

        action_features, actions, context_features, contexts = (
            self._preprocess_react_parameters(
                action_features=action_features,
                action_values=actions,
                case_indices=case_indices,
                context_features=context_features,
                context_values=contexts,
                desired_conviction=desired_conviction,
                preserve_feature_values=preserve_feature_values,
                trainee_id=trainee_id,
                continue_series=continue_series,
            )
        )

        if action_features is not None and derived_action_features is not None:
            if not set(derived_action_features).issubset(set(action_features)):
                raise ValueError(
                    'Specified `derived_action_features` must be a subset of `action_features`.')

        serialized_series_context_values = None
        if series_context_values:
            serialized_series_context_values = []
            for series in series_context_values:
                if series_context_features is None:
                    series_context_features = internals.get_features_from_data(
                        data=series,
                        data_parameter="series_context_values",
                        features_parameter="series_context_features")
                serialized_series_context_values.append(
                    serialize_cases(series, series_context_features, feature_attributes))

        serialized_continue_series_values = None
        if continue_series_values:
            serialized_continue_series_values = []
            for series in continue_series_values:
                if continue_series_features is None:
                    continue_series_features = internals.get_features_from_data(
                        data=series,
                        data_parameter="continue_series_values",
                        features_parameter="continue_series_features")
                serialized_continue_series_values.append(
                    serialize_cases(series, continue_series_features, feature_attributes))

        if new_case_threshold not in [None, "min", "max", "most_similar"]:
            raise ValueError(
                f"The value '{new_case_threshold}' specified for the parameter "
                "`new_case_threshold` is not valid. It accepts one of the"
                " following values - ['min', 'max', 'most_similar',]"
            )

        if initial_values is not None and initial_features is None:
            initial_features = internals.get_features_from_data(
                initial_values,
                data_parameter='initial_values',
                features_parameter='initial_features')
        initial_values = serialize_cases(initial_values, initial_features, feature_attributes)

        # All of these params must be of length 1 or N
        # where N is the length of the largest
        one_or_more_params = [
            contexts,
            initial_values,
            serialized_continue_series_values,
            serialized_series_context_values,
            case_indices,
            actions,
            max_series_lengths,
            series_stop_maps,
        ]
        if any(one_or_more_params):
            param_lengths = set([len(x) for x in one_or_more_params if x])
            if len(param_lengths - {1}) > 1:
                # Raise error if any of the params have different lengths
                # greater than 1
                raise ValueError(
                    'When providing any of `contexts`, `actions`, '
                    '`series_context_values`, `continue_series_values`, '
                    '`case_indices`, `initial_values`, `max_series_lengths`, '
                    'or `series_stop_maps`, each must be of length 1 or the same '
                    'length as each other.')
        else:
            param_lengths = {1}

        if desired_conviction is None:
            if case_indices and not preserve_feature_values:
                raise ValueError(
                    "For discriminative reacts, `preserve_feature_values` "
                    "is required when `case_indices` is specified.")
            else:
                total_size = max(param_lengths)

            react_params = {
                "action_features": action_features,
                "action_values": actions,
                "context_features": context_features,
                "context_values": contexts,
                "continue_series": continue_series,
                "continue_series_features": continue_series_features,
                "continue_series_values": serialized_continue_series_values,
                "initial_features": initial_features,
                "initial_values": initial_values,
                "final_time_steps": final_time_steps,
                "init_time_steps": init_time_steps,
                "series_stop_maps": series_stop_maps,
                "max_series_lengths": max_series_lengths,
                "derived_context_features": derived_context_features,
                "derived_action_features": derived_action_features,
                "series_context_features": series_context_features,
                "series_context_values": serialized_series_context_values,
                "case_indices": case_indices,
                "preserve_feature_values": preserve_feature_values,
                "new_case_threshold": new_case_threshold,
                "input_is_substituted": input_is_substituted,
                "substitute_output": substitute_output,
                "weight_feature": weight_feature,
                "use_case_weights": use_case_weights,
                "leave_case_out": leave_case_out,
                "details": details,
                "exclude_novel_nominals_from_uniqueness_check": exclude_novel_nominals_from_uniqueness_check,
                "series_id_tracking": series_id_tracking,
                "output_new_series_ids": output_new_series_ids,
            }

        else:
            if (
                not isinstance(num_series_to_generate, int) or
                num_series_to_generate <= 0
            ):
                raise HowsoError("`num_series_to_generate` must be an integer "
                                 "greater than 0.")
            if max(param_lengths) not in [1, num_series_to_generate]:
                raise ValueError(
                    'For generative reacts, when specifying parameters with '
                    'values for each series they must be of length 1 or the '
                    'value specified by `num_series_to_generate`.')
            total_size = num_series_to_generate

            context_features, contexts = \
                self._preprocess_generate_parameters(
                    trainee_id,
                    action_features=action_features,
                    context_features=context_features,
                    context_values=contexts,
                    desired_conviction=desired_conviction,
                    num_cases_to_generate=num_series_to_generate,
                    case_indices=case_indices
                )

            react_params = {
                "num_series_to_generate": num_series_to_generate,
                "action_features": action_features,
                "context_features": context_features,
                "context_values": contexts,
                "continue_series": continue_series,
                "continue_series_features": continue_series_features,
                "continue_series_values": continue_series_values,
                "initial_features": initial_features,
                "initial_values": initial_values,
                "final_time_steps": final_time_steps,
                "init_time_steps": init_time_steps,
                "series_stop_maps": series_stop_maps,
                "max_series_lengths": max_series_lengths,
                "derived_context_features": derived_context_features,
                "derived_action_features": derived_action_features,
                "exclude_novel_nominals_from_uniqueness_check": exclude_novel_nominals_from_uniqueness_check,
                "series_context_features": series_context_features,
                "series_context_values": serialized_series_context_values,
                "use_regional_model_residuals": use_regional_model_residuals,
                "desired_conviction": desired_conviction,
                "feature_bounds_map": feature_bounds_map,
                "generate_new_cases": generate_new_cases,
                "ordered_by_specified_features": ordered_by_specified_features,
                "input_is_substituted": input_is_substituted,
                "substitute_output": substitute_output,
                "weight_feature": weight_feature,
                "use_case_weights": use_case_weights,
                "preserve_feature_values": preserve_feature_values,
                "new_case_threshold": new_case_threshold,
                "case_indices": case_indices,
                "leave_case_out": leave_case_out,
                "details": details,
                "series_id_tracking": series_id_tracking,
                "output_new_series_ids": output_new_series_ids,
            }

        if batch_size or self._should_react_batch(react_params, total_size):
            if self.configuration.verbose:
                print(f'Batch series reacting on trainee with id: {trainee_id}')
            response = self._batch_react_series(
                trainee_id,
                react_params,
                total_size=total_size,
                batch_size=batch_size,
                initial_batch_size=initial_batch_size,
                progress_callback=progress_callback)
        else:
            if self.configuration.verbose:
                print(f'Series reacting on trainee with id: {trainee_id}')
            with ProgressTimer(total_size) as progress:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress, None)
                response, _, _ = self._react_series(trainee_id, react_params)
                progress.update(total_size)

            if isinstance(progress_callback, Callable):
                progress_callback(progress, response)

        # put all details under the 'details' key
        action = response.pop('action')
        response = {'action': action, 'details': response}

        # If the number of series generated is less then requested, raise
        # warning, for generative reacts
        if desired_conviction is not None:
            len_action = len(response['action'])
            internals.insufficient_generation_check(
                num_series_to_generate, len_action,
                suppress_warning=suppress_warning
            )

        series_df = util.build_react_series_df(response, series_index=series_index)
        return Reaction(series_df, response.get('details'))

    def _batch_react_series(  # noqa: C901
        self,
        trainee_id: str,
        params: dict,
        *,
        batch_size: t.Optional[int] = None,
        initial_batch_size: t.Optional[int] = None,
        total_size: int,
        progress_callback: t.Optional[Callable] = None
    ):
        """
        Make react series requests in batch.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
        params : dict
            The engine react series parameters.
        batch_size: int, optional
            Define the number of series to react to at once. If left
            unspecified, the batch size will be determined automatically.
        initial_batch_size: int, optional
            The number of series to react to in the first batch. If unspecified,
            the number will be determined automatically. The number of series
            in following batches will be automatically adjusted. This value is
            ignored if ``batch_size`` is specified.
        total_size : int
            The total size of the data that will be batched.
        progress_callback : callable, optional
            A function to be called during batching to retrieve or
            report progress metrics.

        Returns
        -------
        dict
            The `react_series` response.
        """
        temp_result = None
        accumulated_result = {'action_values': []}

        actions = params.get('action_values')
        contexts = params.get('context_values')
        case_indices = params.get('case_indices')
        initial_values = params.get('initial_values')
        max_series_lengths = params.get('max_series_lengths')
        series_context_values = params.get('series_context_values')
        series_stop_maps = params.get('series_stop_maps')
        continue_values = params.get('continue_series_values')

        with ProgressTimer(total_size) as progress:
            batch_scaler = None
            gen_batch_size = None
            if not batch_size:
                if not initial_batch_size:
                    start_batch_size = max(self._get_trainee_thread_count(trainee_id), 1)
                else:
                    start_batch_size = initial_batch_size
                batch_scaler = self.batch_scaler_class(start_batch_size, progress)
                gen_batch_size = batch_scaler.gen_batch_size()
                batch_size = next(gen_batch_size, None)

            while not progress.is_complete and batch_size is not None:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress, temp_result)
                batch_start = progress.current_tick
                batch_end = progress.current_tick + batch_size

                if actions is not None and len(actions) > 1:
                    params['action_values'] = actions[batch_start:batch_end]
                if contexts is not None and len(contexts) > 1:
                    params['context_values'] = contexts[batch_start:batch_end]
                if case_indices is not None and len(case_indices) > 1:
                    params['case_indices'] = case_indices[batch_start:batch_end]
                if initial_values is not None and len(initial_values) > 1:
                    params['initial_values'] = initial_values[batch_start:batch_end]
                if max_series_lengths is not None and len(max_series_lengths) > 1:
                    params['max_series_lengths'] = max_series_lengths[batch_start:batch_end]
                if series_context_values is not None and len(series_context_values) > 1:
                    params['series_context_values'] = series_context_values[batch_start:batch_end]
                if series_stop_maps is not None and len(series_stop_maps) > 1:
                    params['series_stop_maps'] = series_stop_maps[batch_start:batch_end]
                if continue_values is not None and len(continue_values) > 1:
                    params['continue_series_values'] = continue_values[batch_start:batch_end]

                if params.get('desired_conviction') is not None:
                    params['num_series_to_generate'] = batch_size
                temp_result, in_size, out_size = self._react_series(trainee_id, params)

                internals.accumulate_react_result(accumulated_result, temp_result)
                if batch_scaler is None or gen_batch_size is None:
                    progress.update(batch_size)
                else:
                    batch_size = batch_scaler.send(
                        gen_batch_size,
                        batch_scaler.SendOptions(None, (in_size, out_size)))

        # Final call to callback on completion
        if isinstance(progress_callback, Callable):
            progress_callback(progress, temp_result)

        return accumulated_result

    def _react_series(self, trainee_id: str, params: dict):
        """
        Make a single react series request.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        params : dict
            The engine react series parameters.

        Returns
        -------
        dict
            The react series response.
        int
            The request payload size.
        int
            The response payload size.
        """
        batch_result, in_size, out_size = self.execute_sized(trainee_id, "react_series", params)

        if batch_result is None or batch_result.get('action_values') is None:
            raise ValueError('Invalid parameters passed to react_series.')

        ret = dict()
        batch_result = util.replace_doublemax_with_infinity(batch_result)

        # batch_result always has action_features and action_values
        ret['action_features'] = batch_result.pop('action_features') or []
        ret['action'] = batch_result.pop('action_values')

        # ensure all the details items are output as well
        for k, v in batch_result.items():
            ret[k] = [] if v is None else v

        return ret, in_size, out_size

    def react_into_features(
        self,
        trainee_id: str,
        *,
        distance_contribution: bool | str = False,
        familiarity_conviction_addition: bool | str = False,
        familiarity_conviction_removal: bool | str = False,
        features: t.Optional[Collection[str]] = None,
        influence_weight_entropy: bool | str = False,
        p_value_of_addition: bool | str = False,
        p_value_of_removal: bool | str = False,
        similarity_conviction: bool | str = False,
        use_case_weights: t.Optional[bool] = None,
        weight_feature: t.Optional[str] = None,
    ):
        """
        Calculate and cache conviction and other statistics.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to calculate and store conviction for.
        features : iterable of str, optional
            An iterable of features to calculate convictions.
        familiarity_conviction_addition : bool or str, default False
            The name of the feature to store conviction of addition
            values. If set to True the values will be stored to the feature
            'familiarity_conviction_addition'.
        familiarity_conviction_removal : bool or str, default False
            The name of the feature to store conviction of removal
            values. If set to True the values will be stored to the feature
            'familiarity_conviction_removal'.
        influence_weight_entropy : bool or str, default False
            The name of the feature to store influence weight entropy values in.
            If set to True, the values will be stored in the feature
            'influence_weight_entropy'.
        p_value_of_addition : bool or str, default False
            The name of the feature to store p value of addition
            values. If set to True the values will be stored to the feature
            'p_value_of_addition'.
        p_value_of_removal : bool or str, default False
            The name of the feature to store p value of removal
            values. If set to True the values will be stored to the feature
            'p_value_of_removal'.
        similarity_conviction : bool or str, default False
            The name of the feature to store similarity conviction
            values. If set to True the values will be stored to the feature
            'similarity_conviction'.
        distance_contribution : bool or str, default False
            The name of the feature to store distance contribution.
            If set to True the values will be stored to the
            feature 'distance_contribution'.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        util.validate_list_shape(features, 1, "features", "str")
        if self.configuration.verbose:
            print(f'Reacting into features on Trainee with id: {trainee_id}')
        self.execute(trainee_id, "react_into_features", {
            "features": features,
            "familiarity_conviction_addition": familiarity_conviction_addition,
            "familiarity_conviction_removal": familiarity_conviction_removal,
            "influence_weight_entropy": influence_weight_entropy,
            "p_value_of_addition": p_value_of_addition,
            "p_value_of_removal": p_value_of_removal,
            "similarity_conviction": similarity_conviction,
            "distance_contribution": distance_contribution,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })
        self._auto_persist_trainee(trainee_id)

    def react_aggregate(
        self,
        trainee_id: str,
        *,
        action_feature: t.Optional[str] = None,
        action_features: t.Optional[Collection[str]] = None,
        confusion_matrix_min_count: t.Optional[int] = None,
        context_features: t.Optional[Collection[str]] = None,
        details: t.Optional[dict] = None,
        feature_influences_action_feature: t.Optional[str] = None,
        hyperparameter_param_path: t.Optional[Collection[str]] = None,
        num_robust_influence_samples: t.Optional[int] = None,
        num_robust_residual_samples: t.Optional[int] = None,
        num_robust_influence_samples_per_case: t.Optional[int] = None,
        num_samples: t.Optional[int] = None,
        prediction_stats_action_feature: t.Optional[str] = None,
        residuals_hyperparameter_feature: t.Optional[str] = None,
        robust_hyperparameters: t.Optional[bool] = None,
        sample_model_fraction: t.Optional[float] = None,
        sub_model_size: t.Optional[int] = None,
        use_case_weights: t.Optional[bool] = None,
        weight_feature: t.Optional[str] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Reacts into the aggregate trained cases in the Trainee.

        Calculates, caches, and/or returns the requested influences and prediction stats.

        Parameters
        ----------
        action_feature : str, optional
            Name of target feature for which to do computations. If ``prediction_stats_action_feature``
            and ``feature_influences_action_feature`` are not provided, they will default to this value.
            If ``feature_influences_action_feature`` is not provided and feature influences ``details`` are
            selected, this feature must be provided.
        action_features : iterable of str, optional
            List of feature names to compute any requested residuals or prediction statistics for. If unspecified,
            the value used for context features will be used.
        confusion_matrix_min_count : int, optional
            The number of predictions a class should have (value of a cell in the
            matrix) for it to remain in the confusion matrix. If the count is
            less than this value, it will be accumulated into a single value of
            all insignificant predictions for the class and removed from the
            confusion matrix. Defaults to 10, applicable only to confusion
            matrices when computing residuals.
        context_features : iterable of str, optional
            List of features names to use as contexts for
            computations. Default is all trained non-unique features if
            unspecified.
        details : map of str -> object, optional
            If details are specified, the response will contain the requested
            explanation data.. Below are the valid keys and data types for the
            different audit details. Omitted keys, values set to None, or False
            values for Booleans will not be included in the data returned.

            - prediction_stats : bool, optional
                If True outputs full feature prediction stats for all (context and action) features.
                The prediction stats returned are set by the "selected_prediction_stats" parameter
                in the `details` parameter. Uses full calculations, which uses leave-one-out for
                features for computations.
            - feature_residuals_full : bool, optional
                For each context_feature, use the full set of all other context_features to predict
                the feature. When ``prediction_stats`` in the ``details`` parameter is true, the
                Trainee will also calculate the full feature residuals.
            - feature_residuals_robust : bool, optional
                For each context_feature, use the robust (power set/permutations) set of all other
                context_features to predict the feature.
            - feature_contributions_full : bool, optional
                For each context_feature, use the full set of all other
                context_features to compute the mean absolute delta between
                prediction of action feature with and without the context features
                in the model. Returns the mean absolute delta
                under the key 'feature_contributions_full' and returns the mean
                delta under the key 'directional_feature_contributions_full'.
            - feature_contributions_robust : bool, optional
                For each context_feature, use the robust (power set/permutation)
                set of all other context_features to compute the mean absolute
                delta between prediction of the action feature with and without the
                context features in the model. Returns the mean absolute delta
                under the key 'feature_contributions_robust' and returns the mean
                delta under the key 'directional_feature_contributions_robust'.
            - feature_mda_full : bool, optional
                When True will compute Mean Decrease in Accuracy (MDA)
                for each context feature at predicting the action feature. Drop
                each feature and use the full set of remaining context features
                for each prediction.
            - feature_mda_robust : bool, optional
                Compute Mean Decrease in Accuracy MDA by dropping each feature and using the
                robust (power set/permutations) set of remaining context features
                for each prediction.
            - feature_feature_mda_permutation_full : bool, optional
                Compute MDA by scrambling each feature and using the
                full set of remaining context features for each prediction.
            - feature_feature_mda_permutation_robust : bool, optional
                Compute MDA by scrambling each feature and using the
                robust (power set/permutations) set of remaining context features
                for each prediction.
            - action_condition : map of str -> any, optional
                A condition map to select the action set, which is the dataset for which
                the prediction stats are for. If both ``action_condition`` and ``context_condition``
                are provided, then all of the action cases selected by the ``action_condition``
                will be excluded from the context set, which is the set being queried to make to
                make predictions on the action set, effectively holding them out.
                If only ``action_condition`` is specified, then only the single predicted case
                will be left out.

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
            - action_num_cases : int, optional
                The maximum amount of cases to use to calculate prediction stats.
                If not specified, the limit will be k cases if precision is
                "similar", or 1000 cases if precision is "exact". Works with or
                without ``action_condition``.
                -If ``action_condition`` is set:
                    If None, will be set to k if precision is "similar" or no limit if precision is "exact".
                - If ``action_condition`` is not set:
                    If None, will be set to the Howso default limit of 2000.
            - action_condition_precision : {"exact", "similar"}, optional
                The precision to use when selecting cases with the ``action_condition``.
                If not specified "exact" will be used. Only used if ``action_condition``
                is not None.
            - context_condition : map of str -> any, optional
                A condition map to select the context set, which is the set being queried to make
                to make predictions on the action set. If both ``action_condition`` and ``context_condition``
                are provided,  then all of the cases from the action set, which is the dataset for which the
                prediction stats are for, will be excluded from the context set, effectively holding them out.
                If only ``action_condition`` is specified,  then only the single predicted case will be left out.

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
            - context_precision_num_cases : int, optional
                Limit on the number of context cases when ``context_condition_precision`` is set to "similar".
                If None, will be set to k.
            - context_condition_precision : {"exact", "similar"}, optional
                The precision to use when selecting cases with the ``context_condition``.
                If not specified "exact" will be used. Only used if ``context_condition``
                is not None.
            - prediction_stats_features : list, optional
                List of features to use when calculating conditional prediction stats. Should contain all
                action and context features desired. If ``action_feature`` is also provided, that feature will
                automatically be appended to this list if it is not already in the list.
                    stats : list of str, optional
            - selected_prediction_stats : list, optional
                List of stats to output. When unspecified, returns all except the confusion matrix. Allowed values:

                - all : Returns all the the available prediction stats, including the confusion matrix.
                - accuracy : The number of correct predictions divided by the
                  total number of predictions.
                - confusion_matrix : A sparse map of actual feature value to a map of
                  predicted feature value to counts.
                - mae : Mean absolute error. For continuous features, this is
                  calculated as the mean of absolute values of the difference
                  between the actual and predicted values. For nominal features,
                  this is 1 - the average categorical action probability of each case's
                  correct classes. Categorical action probabilities are the probabilities
                  for each class for the action feature.
                - mda : Mean decrease in accuracy when each feature is dropped
                  from the model, applies to all features.
                - feature_mda_permutation_full : Mean decrease in accuracy that used
                  scrambling of feature values instead of dropping each
                  feature, applies to all features.
                - precision : Precision (positive predictive) value for nominal
                  features only.
                - r2 : The r-squared coefficient of determination, for
                  continuous features only.
                - recall : Recall (sensitivity) value for nominal features only.
                - rmse : Root mean squared error, for continuous features only.
                - spearman_coeff : Spearman's rank correlation coefficient,
                  for continuous features only.
                - mcc : Matthews correlation coefficient, for nominal features only.
        feature_influences_action_feature : str, optional
            When feature influences such as contributions and mda, use this feature as
            the action feature.  If not provided, will default to the ``action_feature`` if provided.
            If ``action_feature`` is not provided and feature influences ``details`` are
            selected, this feature must be provided.
        hyperparameter_param_path : iterable of str, optional.
            Full path for hyperparameters to use for computation. If specified
            for any residual computations, takes precedence over action_feature
            parameter.  Can be set to a 'paramPath' value from the results of
            'get_params()' for a specific set of hyperparameters.
        num_robust_influence_samples : int, optional
            Total sample size of model to use (using sampling with replacement)
            for robust contribution computation. Defaults to 300.
        num_robust_residual_samples : int, optional
            Total sample size of model to use (using sampling with replacement)
            for robust mda and residual computation.
            Defaults to 1000 * (1 + log(number of features)).  Note: robust mda
            will be updated to use num_robust_influence_samples in a future release.
        num_robust_influence_samples_per_case : int, optional
            Specifies the number of robust samples to use for each case for
            robust contribution computations.
            Defaults to 300 + 2 * (number of features).
        num_samples : int, optional
            Total sample size of model to use (using sampling with replacement)
            for all non-robust computation. Defaults to 1000.
            If specified overrides sample_model_fraction.```
        residuals_hyperparameter_feature : str, optional
            When calculating residuals and prediction stats, uses this target
            features's hyperparameters. The trainee must have been analyzed with
            this feature as the action feature first. If not provided, by default
            residuals and prediction stats uses ".targetless" hyperparameters.
        robust_hyperparameters : bool, optional
            When specified, will attempt to return residuals that were
            computed using hyperparameters with the specified robust or
            non-robust type.
        prediction_stats_action_feature : str, optional
            When calculating residuals and prediction stats, uses this target features's
            hyperparameters. The trainee must have been analyzed with this feature as the
            action feature first. If both ``prediction_stats_action_feature`` and
            ``action_feature`` are not provided, by default residuals and prediction
            stats uses ".targetless" hyperparameters. If "action_feature" is provided,
            and this value is not provided, will default to ``action_feature``.
        sample_model_fraction : float, optional
            A value between 0.0 - 1.0, percent of model to use in sampling
            (using sampling without replacement). Applicable only to non-robust
            computation. Ignored if num_samples is specified.
            Higher values provide better accuracy at the cost of compute time.
        sub_model_size : int, optional
            Subset of model to use for calculations. Applicable only
            to models > 1000 cases.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.
        weight_feature : str, optional
            The name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        dict of str to dict of str to float
            If specified, a map of feature to map of stat type to stat values is returned.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        util.validate_list_shape(context_features, 1, "context_features", "str")

        if isinstance(details, dict):
            if action_condition_precision := details.get("action_condition_precision"):
                if action_condition_precision not in self.SUPPORTED_PRECISION_VALUES:
                    warnings.warn(self.WARNING_MESSAGES['invalid_precision'].format("action_condition_precision"))

            if context_condition_precision := details.get("context_condition_precision"):
                if context_condition_precision not in self.SUPPORTED_PRECISION_VALUES:
                    warnings.warn(self.WARNING_MESSAGES['invalid_precision'].format("context_condition_precision"))

        if self.configuration.verbose:
            print(f'Reacting into aggregate trained cases of Trainee with id: {trainee_id}')

        stats = self.execute(trainee_id, "react_aggregate", {
            "action_feature": action_feature,
            "action_features": action_features,
            "residuals_hyperparameter_feature": residuals_hyperparameter_feature,
            "context_features": context_features,
            "confusion_matrix_min_count": confusion_matrix_min_count,
            "details": details,
            "feature_influences_action_feature": feature_influences_action_feature,
            "hyperparameter_param_path": hyperparameter_param_path,
            "num_robust_influence_samples": num_robust_influence_samples,
            "num_robust_residual_samples": num_robust_residual_samples,
            "num_robust_influence_samples_per_case": num_robust_influence_samples_per_case,
            "num_samples": num_samples,
            "prediction_stats_action_feature": prediction_stats_action_feature,
            "robust_hyperparameters": robust_hyperparameters,
            "sample_model_fraction": sample_model_fraction,
            "sub_model_size": sub_model_size,
            "use_case_weights": use_case_weights,
            "weight_feature": weight_feature,
        })
        if stats is None:
            stats = dict()

        self._auto_persist_trainee(trainee_id)
        return stats

    def react_group(
        self,
        trainee_id: str,
        new_cases: TabularData3D,
        *,
        features: t.Optional[Collection[str]] = None,
        distance_contributions: bool = False,
        familiarity_conviction_addition: bool = True,
        familiarity_conviction_removal: bool = False,
        kl_divergence_addition: bool = False,
        kl_divergence_removal: bool = False,
        p_value_of_addition: bool = False,
        p_value_of_removal: bool = False,
        weight_feature: t.Optional[str] = None,
        use_case_weights: t.Optional[bool] = None,
    ) -> dict:
        """
        Computes specified data for a **set** of cases.

        Return the list of familiarity convictions (and optionally, distance
        contributions or p values) for each set.

        Parameters
        ----------
        trainee_id : str
            The trainee id.

        new_cases : list of list of list of object or list of DataFrame
            Specify a **set** using a list of cases to compute the conviction of
            groups of cases as shown in the following example.

            >>> [ [[1, 2, 3], [4, 5, 6], [7, 8, 9]], # Group 1
            >>>   [[1, 2, 3]] ] # Group 2

        features : Collection of str, optional
            The feature names to consider while calculating convictions.
        distance_contributions : bool, default False
            Calculate and output distance contribution ratios in
            the output dict for each case.
        familiarity_conviction_addition : bool, default True
            Calculate and output familiarity conviction of adding the
            specified cases.
        familiarity_conviction_removal : bool, default False
            Calculate and output familiarity conviction of removing
            the specified cases.
        kl_divergence_addition : bool, default False
            Calculate and output KL divergence of adding the
            specified cases.
        kl_divergence_removal : bool, default False
            Calculate and output KL divergence of removing the
            specified cases.
        p_value_of_addition : bool, default False
            If true will output p value of addition.
        p_value_of_removal : bool, default False
            If true will output p value of removal.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.

        Returns
        -------
        dict
            The react group response.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)
        serialized_cases = None

        if util.num_list_dimensions(new_cases) != 3:
            raise ValueError(
                "Improper shape of `new_cases` values passed. "
                "`new_cases` must be a 3d list of object.")

        serialized_cases = []
        for group in new_cases:
            if features is None:
                features = internals.get_features_from_data(group)
            serialized_cases.append(serialize_cases(group, features, feature_attributes))

        if self.configuration.verbose:
            print(f'Reacting to a set of cases on Trainee with id: {trainee_id}')
        result = self.execute(trainee_id, "react_group", {
            "features": features,
            "new_cases": serialized_cases,
            "distance_contributions": distance_contributions,
            "familiarity_conviction_addition": familiarity_conviction_addition,
            "familiarity_conviction_removal": familiarity_conviction_removal,
            "kl_divergence_addition": kl_divergence_addition,
            "kl_divergence_removal": kl_divergence_removal,
            "p_value_of_addition": p_value_of_addition,
            "p_value_of_removal": p_value_of_removal,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })
        if result is None:
            result = dict()
        return result

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
        features_to_code_map : Mapping of str to str
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Evaluating on Trainee with id: {trainee_id}')
        return self.execute(trainee_id, "evaluate", {
            "features_to_code_map": features_to_code_map,
            "aggregation_code": aggregation_code
        })

    def analyze(
        self,
        trainee_id: str,
        context_features: t.Optional[Collection[str]] = None,
        action_features: t.Optional[Collection[str]] = None,
        *,
        analysis_sub_model_size: t.Optional[int] = None,
        bypass_calculate_feature_residuals: t.Optional[bool] = None,
        bypass_calculate_feature_weights: t.Optional[bool] = None,
        bypass_hyperparameter_analysis: t.Optional[bool] = None,
        dt_values: t.Optional[Collection[float]] = None,
        inverse_residuals_as_weights: t.Optional[bool] = None,
        k_folds: t.Optional[int] = None,
        k_values: t.Optional[Collection[int]] = None,
        num_analysis_samples: t.Optional[int] = None,
        num_samples: t.Optional[int] = None,
        p_values: t.Optional[Collection[float]] = None,
        targeted_model: t.Optional[TargetedModel] = None,
        use_case_weights: t.Optional[bool] = None,
        use_deviations: t.Optional[bool] = None,
        weight_feature: t.Optional[str] = None,
        **kwargs
    ):
        """
        Analyzes a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        context_features : Collection of str, optional
            The context features to analyze for.
        action_features : Collection of str, optional
            The action features to analyze for.
        analysis_sub_model_size : int or Node, optional
            Number of samples to use for analysis. The rest will be randomly
            held-out and not included in calculations.
        bypass_calculate_feature_residuals : bool, optional
            When True, bypasses calculation of feature residuals.
        bypass_calculate_feature_weights : bool, optional
            When True, bypasses calculation of feature weights.
        bypass_hyperparameter_analysis : bool, optional
            When True, bypasses hyperparameter analysis.
        dt_values : Collection of float, optional
            The dt value hyperparameters to analyze with.
        inverse_residuals_as_weights : bool, default is False
            When True, will compute and use inverse of residuals as
            feature weights.
        k_folds : int, default 6
            The number of cross validation folds to do.
        k_values : Collection of int, optional
            The number of cross validation folds to do. A value of 1 does
            hold-one-out instead of k-fold.
        num_analysis_samples : int, optional
            If the dataset size to too large, analyze on (randomly sampled)
            subset of data. The `num_analysis_samples` specifies the number of
            observations to be considered for analysis.
        num_samples : int, optional
            The number of samples used in calculating feature residuals.
        p_values : Collection of float, optional
            The p value hyperparameters to analyze with.
        targeted_model : {"omni_targeted", "single_targeted", "targetless"}, optional
            Type of hyperparameter targeting.
            Valid options include:

                - **single_targeted**: Analyze hyperparameters for the
                  specified action_features.
                - **omni_targeted**: Analyze hyperparameters for each context
                  feature as an action feature, ignores action_features
                  parameter.
                - **targetless**: Analyze hyperparameters for all context
                  features as possible action features, ignores
                  action_features parameter.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.
        use_deviations : bool, optional
            When True, uses deviations for LK metric in queries.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        kwargs
            Additional experimental analyze parameters.
        """
        trainee_id = self._resolve_trainee(trainee_id).id

        util.validate_list_shape(context_features, 1, "context_features", "str")
        util.validate_list_shape(action_features, 1, "action_features", "str")
        util.validate_list_shape(p_values, 1, "p_values", "int")
        util.validate_list_shape(k_values, 1, "k_values", "float")
        util.validate_list_shape(dt_values, 1, "dt_values", "float")

        if targeted_model not in ['single_targeted', 'omni_targeted', 'targetless', None]:
            raise ValueError(
                f'Invalid value "{targeted_model}" for targeted_model. '
                'Valid values include single_targeted, omni_targeted, '
                'and targetless.')

        deprecated_params = {
            'bypass_hyperparameter_optimization': 'bypass_hyperparameter_analysis',
            'num_optimization_samples': 'num_analysis_samples',
            'optimization_sub_model_size': 'analysis_sub_model_size',
            'dwe_values': 'dt_values'
        }
        # explicitly update parameters if old names are provided
        if kwargs:
            for old_param, new_param in deprecated_params.items():
                if old_param in kwargs:
                    if old_param == 'bypass_hyperparameter_optimization':
                        bypass_hyperparameter_analysis = kwargs[old_param]
                    elif old_param == 'num_optimization_samples':
                        num_analysis_samples = kwargs[old_param]
                    elif old_param == 'optimization_sub_model_size':
                        analysis_sub_model_size = kwargs[old_param]
                    elif old_param == 'dwe_values':
                        dt_values = kwargs[old_param]

                    del kwargs[old_param]
                    warnings.warn(
                        f'The `{old_param}` parameter has been renamed to '
                        f'`{new_param}`, please use the new parameter '
                        'instead.', UserWarning)

        analyze_params = dict(
            action_features=action_features,
            context_features=context_features,
            bypass_calculate_feature_residuals=bypass_calculate_feature_residuals,  # noqa: #E501
            bypass_calculate_feature_weights=bypass_calculate_feature_weights,
            bypass_hyperparameter_analysis=bypass_hyperparameter_analysis,  # noqa: #E501
            dt_values=dt_values,
            use_case_weights=use_case_weights,
            inverse_residuals_as_weights=inverse_residuals_as_weights,
            k_folds=k_folds,
            k_values=k_values,
            num_analysis_samples=num_analysis_samples,
            num_samples=num_samples,
            analysis_sub_model_size=analysis_sub_model_size,
            p_values=p_values,
            targeted_model=targeted_model,
            use_deviations=use_deviations,
            weight_feature=weight_feature,
        )

        # Add experimental options
        analyze_params.update(kwargs)

        if kwargs:
            warn_params = ', '.join(kwargs)
            warnings.warn(
                'The following parameter(s) are not officially supported by `analyze` and '
                f'may or may not have an effect: {warn_params}',
                UnsupportedArgumentWarning)

        if self.configuration.verbose:
            print(f'Analyzing Trainee with id: {trainee_id}')
            print('Analyzing Trainee with parameters: ' + str({
                k: v for k, v in analyze_params.items() if v is not None
            }))

        self.execute(trainee_id, "analyze", analyze_params)
        self._auto_persist_trainee(trainee_id)

    def auto_analyze(self, trainee_id: str):
        """
        Auto-analyze the Trainee model.

        Re-uses all parameters from the previous analyze or
        set_auto_analyze_params call. If analyze or set_auto_analyze_params
        has not been previously called, auto_analyze will default to a robust
        and versatile analysis.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to auto-analyze.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f"Auto-analyzing Trainee with id: {trainee_id}")

        self.execute(trainee_id, "auto_analyze", {})
        self._auto_persist_trainee(trainee_id)
        if self.is_tracing_enabled(trainee_id):
            # When trace is enabled, output the auto-analyzed parameters into the trace file
            self.execute(trainee_id, "get_params", {})

    def set_auto_analyze_params(
        self,
        trainee_id: str,
        auto_analyze_enabled: bool = False,
        analyze_threshold: t.Optional[int] = None,
        *,
        analysis_sub_model_size: t.Optional[int] = None,
        analyze_growth_factor: t.Optional[float] = None,
        action_features: t.Optional[Collection[str]] = None,
        bypass_calculate_feature_residuals: t.Optional[bool] = None,
        bypass_calculate_feature_weights: t.Optional[bool] = None,
        bypass_hyperparameter_analysis: t.Optional[bool] = None,
        context_features: t.Optional[Collection[str]] = None,
        dt_values: t.Optional[Collection[float]] = None,
        inverse_residuals_as_weights: t.Optional[bool] = None,
        k_folds: t.Optional[int] = None,
        k_values: t.Optional[Collection[int]] = None,
        num_analysis_samples: t.Optional[int] = None,
        num_samples: t.Optional[int] = None,
        p_values: t.Optional[Collection[float]] = None,
        targeted_model: t.Optional[TargetedModel] = None,
        use_deviations: t.Optional[bool] = None,
        use_case_weights: t.Optional[bool] = None,
        weight_feature: t.Optional[str] = None,
        **kwargs
    ):
        """
        Set Trainee parameters for auto analysis.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set auto analysis parameters for.
        auto_analyze_enabled : bool, default False
            When True, the :func:`train` method will trigger an analyze when
            it's time for the model to be analyzed again.
        analyze_threshold : int, optional
            The threshold for the number of cases at which the model should be
            re-analyzed.
        analyze_growth_factor : float, optional
            The factor by which to increase the analyze threshold every time
            the model grows to the current threshold size.
        action_features : Collection of str, optional
            The action features to analyze for.
        context_features : Collection of str, optional
            The context features to analyze for.
        k_folds : int, optional
            The number of cross validation folds to do. A value of 1 does
            hold-one-out instead of k-fold.
        num_samples : int, optional
            The number of samples used in calculating feature residuals.
        dt_values : Collection of float, optional
            The dt value hyperparameters to analyze with.
        k_values : Collection of int, optional
            The number of cross validation folds to do. A value of 1 does
            hold-one-out instead of k-fold.
        p_values : Collection of float, optional
            The p value hyperparameters to analyze with.
        bypass_calculate_feature_residuals : bool, optional
            When True, bypasses calculation of feature residuals.
        bypass_calculate_feature_weights : bool, optional
            When True, bypasses calculation of feature weights.
        bypass_hyperparameter_analysis : bool, optional
            When True, bypasses hyperparameter analysis.
        targeted_model : Literal["omni_targeted", "single_targeted", "targetless"], optional
            Type of hyperparameter targeting.
            Valid options include:

                - **single_targeted**: Analyze hyperparameters for the
                  specified action_features.
                - **omni_targeted**: Analyze hyperparameters for each context
                  feature as an action feature, ignores action_features
                  parameter.
                - **targetless**: Analyze hyperparameters for all context
                  features as possible action features, ignores
                  action_features parameter.
        num_analysis_samples : int, optional
            If the dataset size to too large, analyze on (randomly sampled)
            subset of data. The `num_analysis_samples` specifies the number of
            observations to be considered for analysis.
        analysis_sub_model_size : int, optional
            Number of samples to use for analysis. The rest will be
            randomly held-out and not included in calculations.
        use_deviations : bool, optional
            When True, uses deviations for LK metric in queries.
        inverse_residuals_as_weights : bool, optional
            When True, will compute and use inverse of residuals as feature
            weights.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.
        weight_feature : str
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        kwargs : dict, optional
            Parameters specific for analyze() may be passed in via kwargs, and
            will be cached and used during future auto-analysis.
        """
        trainee_id = self._resolve_trainee(trainee_id).id

        deprecated_params = {
            'auto_optimize_enabled': 'auto_analyze_enabled',
            'optimize_threshold': 'analyze_threshold',
            'optimize_growth_factor': 'analyze_growth_factor',
        }
        analyze_deprecated_params = {
            'bypass_hyperparameter_optimization': 'bypass_hyperparameter_analysis',
            'num_optimization_samples': 'num_analysis_samples',
            'optimization_sub_model_size': 'analysis_sub_model_size',
            'dwe_values': 'dt_values'
        }

        # explicitly update parameters if old names are provided
        if kwargs:
            for old_param, new_param in deprecated_params.items():
                if old_param in kwargs:
                    if old_param == 'auto_optimize_enabled':
                        auto_analyze_enabled = kwargs[old_param]
                    elif old_param == 'optimize_threshold':
                        analyze_threshold = kwargs[old_param]
                    elif old_param == 'optimize_growth_factor':
                        analyze_growth_factor = kwargs[old_param]

                    del kwargs[old_param]
                    warnings.warn(
                        f'The `{old_param}` parameter has been renamed to '
                        f'`{new_param}`, please use the new parameter '
                        'instead.', UserWarning)

            # replace any old kwarg param with new param and remove old param
            for old_param, new_param in analyze_deprecated_params.items():
                if old_param in kwargs:
                    kwargs[new_param] = kwargs[old_param]
                    del kwargs[old_param]
                    warnings.warn(
                        f'The `{old_param}` parameter has been renamed to '
                        f'`{new_param}`, please use the new parameter '
                        'instead.', UserWarning)

        if 'targeted_model' in kwargs:
            targeted_model = kwargs['targeted_model']
            if targeted_model not in ['single_targeted', 'omni_targeted', 'targetless']:
                raise ValueError(
                    f'Invalid value "{targeted_model}" for targeted_model. '
                    'Valid values include single_targeted, omni_targeted, '
                    'and targetless.')

        # Collect valid parameters
        if kwargs:
            warn_params = ", ".join(kwargs)
            warnings.warn(
                f"The following parameter(s) are not officially supported by `auto_analyze` and "
                f"may or may not have an effect: {warn_params}",
                UnsupportedArgumentWarning)

        if self.configuration.verbose:
            print(f'Setting auto analyze parameters for Trainee with id: {trainee_id}')

        self.execute(trainee_id, "set_auto_analyze_params", {
            "auto_analyze_enabled": auto_analyze_enabled,
            "analyze_threshold": analyze_threshold,
            "analyze_growth_factor": analyze_growth_factor,
            "action_features": action_features,
            "context_features": context_features,
            "k_folds": k_folds,
            "num_samples": num_samples,
            "dt_values": dt_values,
            "k_values": k_values,
            "p_values": p_values,
            "bypass_hyperparameter_analysis": bypass_hyperparameter_analysis,
            "bypass_calculate_feature_residuals": bypass_calculate_feature_residuals,
            "bypass_calculate_feature_weights": bypass_calculate_feature_weights,
            "targeted_model": targeted_model,
            "num_analysis_samples": num_analysis_samples,
            "analysis_sub_model_size": analysis_sub_model_size,
            "use_deviations": use_deviations,
            "inverse_residuals_as_weights": inverse_residuals_as_weights,
            "use_case_weights": use_case_weights,
            "weight_feature": weight_feature,
            **kwargs,
        })
        self._auto_persist_trainee(trainee_id)

    def get_auto_ablation_params(self, trainee_id: str) -> dict[str, t.Any]:
        """
        Get Trainee parameters for auto-ablation set by :meth:`set_auto_ablation_params`.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to get auto ablation parameters for.

        Returns
        -------
        dict
            The auto-ablation parameters.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        return self.execute(trainee_id, "get_auto_ablation_params", {})

    def set_auto_ablation_params(
        self,
        trainee_id: str,
        auto_ablation_enabled: bool = False,
        *,
        ablated_cases_distribution_batch_size: int = 100,
        auto_ablation_weight_feature: str = ".case_weight",
        batch_size: int = 2_000,
        conviction_lower_threshold: t.Optional[float] = None,
        conviction_upper_threshold: t.Optional[float] = None,
        exact_prediction_features: t.Optional[Collection[str]] = None,
        influence_weight_entropy_threshold: float = 0.6,
        minimum_model_size: int = 1_000,
        relative_prediction_threshold_map: t.Optional[Mapping[str, float]] = None,
        residual_prediction_features: t.Optional[Collection[str]] = None,
        tolerance_prediction_threshold_map: t.Optional[Mapping[str, tuple[float, float]]] = None,
        **kwargs
    ):
        """
        Set trainee parameters for auto-ablation.

        .. note::
            All ablation endpoints, including :meth:`set_auto_ablation_params` are experimental and may
            have their API changed without deprecation.

        .. seealso::
            The params ``influence_weight_entropy_threshold`` and ``auto_ablation_weight_feature`` that are
            set using this endpoint are used as defaults by :meth:`reduce_data`.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set auto ablation parameters for.
        auto_ablation_enabled : bool, default False
            When True, the :meth:`train` method will ablate cases that meet the set criteria.
        ablated_cases_distribution_batch_size: int, default 100
            Number of cases in a batch to distribute ablated cases' influence weights.
        auto_ablation_weight_feature : str, default ".case_weight"
            The weight feature that should be accumulated to when cases are ablated.
        batch_size: number, default 2,000
            Number of cases in a batch to consider for ablation prior to training and
            to recompute influence weight entropy.
        minimum_model_size : int, default 1,000
            The threshold of the minimum number of cases at which the model should auto-ablate.
        influence_weight_entropy_threshold : float, default 0.6
            The influence weight entropy quantile that a case must be beneath in order to be trained.
        exact_prediction_features : Optional[List[str]], optional
            For each of the features specified, will ablate a case if the prediction matches exactly.
        residual_prediction_features : Optional[List[str]], optional
            For each of the features specified, will ablate a case if
            abs(prediction - case value) / prediction <= feature residual.
        tolerance_prediction_threshold_map : Optional[Dict[str, Tuple[float, float]]], optional
            For each of the features specified, will ablate a case if the prediction >= (case value - MIN)
            and the prediction <= (case value + MAX).
        relative_prediction_threshold_map : Optional[Dict[str, float]], optional
            For each of the features specified, will ablate a case if
            abs(prediction - case value) / prediction <= relative threshold
        conviction_lower_threshold : Optional[float], optional
            The conviction value above which cases will be ablated.
        conviction_upper_threshold : Optional[float], optional
            The conviction value below which cases will be ablated.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        params = dict(
            ablated_cases_distribution_batch_size=ablated_cases_distribution_batch_size,
            auto_ablation_enabled=auto_ablation_enabled,
            auto_ablation_weight_feature=auto_ablation_weight_feature,
            batch_size=batch_size,
            minimum_model_size=minimum_model_size,
            influence_weight_entropy_threshold=influence_weight_entropy_threshold,
            exact_prediction_features=exact_prediction_features,
            residual_prediction_features=residual_prediction_features,
            tolerance_prediction_threshold_map=tolerance_prediction_threshold_map,
            relative_prediction_threshold_map=relative_prediction_threshold_map,
            conviction_lower_threshold=conviction_lower_threshold,
            conviction_upper_threshold=conviction_upper_threshold,
        )
        params.update(kwargs)
        if kwargs:
            warn_params = ", ".join(kwargs)
            warnings.warn(
                f"The following parameter(s) are not officially supported by `auto_ablation` and "
                f"may or may not have an effect: {warn_params}",
                UnsupportedArgumentWarning)
        if self.configuration.verbose:
            print(f'Setting auto ablation parameters for Trainee with id: {trainee_id}')
        self.execute(trainee_id, "set_auto_ablation_params", params)

    def reduce_data(
        self,
        trainee_id: str,
        features: t.Optional[Collection[str]] = None,
        distribute_weight_feature: t.Optional[str] = None,
        influence_weight_entropy_threshold: t.Optional[float] = None,
        skip_auto_analyze: bool = False,
        **kwargs,
    ) -> dict:
        """
        Smartly reduce the amount of trained cases while accumulating case weights.

        Determines which cases to remove by comparing the influence weight entropy of each trained
        case to the ``influence_weight_entropy_threshold`` quantile of existing influence weight
        entropies.

        .. note::
            All ablation endpoints, including :meth:`reduce_data` are experimental and may have their
            API changed without deprecation.

        .. seealso::
            The default ``distribute_weight_feature`` and ``influence_weight_entropy_threshold`` are
            pulled from the auto-ablation parameters, which can be set or retrieved with
            :meth:`set_auto_ablation_params` and :meth:`get_auto_ablation_params`, respectively.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee for which to reduce data.
        features : Collection of str, optional
            The features which should be used to determine which cases to remove. This defaults to all of
            the trained features (excluding internal features).
        distribute_weight_feature : str, optional
            The name of the weight feature to accumulate case weights to as cases are removed. This
            defaults to the value of ``auto_ablation_weight_feature`` from :meth:`set_auto_ablation_params`,
            which defaults to ".case_weight".
        influence_weight_entropy_threshold : float, optional
            The quantile of influence weight entropy above which cases will be removed. This defaults
            to the value of ``influence_weight_entropy_threshold`` from :meth:`set_auto_ablation_params`,
            which defaults to 0.6.
        skip_auto_analyze : bool, default False
            Whether to skip auto-analyzing as cases are removed.

        Returns
        -------
        dict
            A dictionary for reporting experimental outputs of reduce data. Currently, the default
            non-experimental output is an empty dictionary.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        params = dict(
            features=features,
            distribute_weight_feature=distribute_weight_feature,
            influence_weight_entropy_threshold=influence_weight_entropy_threshold,
            skip_auto_analyze=skip_auto_analyze,
        )
        params.update(kwargs)
        if kwargs:
            warn_params = ", ".join(kwargs)
            warnings.warn(
                f"The following parameter(s) are not officially supported by `reduce_data` and "
                f"may or may not have an effect: {warn_params}",
                UnsupportedArgumentWarning)
        if self.configuration.verbose:
            print(f'Reducing data on Trainee with id: {trainee_id}')
        result = self.execute(trainee_id, "reduce_data", params)

        if result is None:
            return dict()

        return result

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
            Sequence of tuples, of session id and index, where index is the
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
        condition : Mapping, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if case_indices is not None:
            util.validate_case_indices(case_indices)

        if precision is not None and precision not in self.SUPPORTED_PRECISION_VALUES:
            warnings.warn(self.WARNING_MESSAGES['invalid_precision'].format("precision"))

        util.validate_list_shape(features, 1, "features", "str")
        if session is None and case_indices is None:
            warnings.warn("Calling get_cases without a session id does not guarantee case order.")

        # Convert session instance to id
        if (
            isinstance(condition, MutableMapping) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Retrieving cases for Trainee with id {trainee_id}.')
        result = self.execute(trainee_id, "get_cases", {
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
        trainee_id = self._resolve_trainee(trainee_id).id
        util.validate_list_shape(features, 1, "features", "str")

        if self.configuration.verbose:
            print(f'Getting extreme cases for trainee with id: {trainee_id}')
        result = self.execute(trainee_id, "get_extreme_cases", {
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
        trainee_id = self._resolve_trainee(trainee_id).id
        ret = self.execute(trainee_id, "get_num_training_cases", {})
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
        use_case_weights: t.Optional[bool] = None,
        weight_feature: t.Optional[str] = None
    ) -> dict:
        """
        Get familiarity conviction for features in the model.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        features : Collection of str, optional
            A collection of feature names to calculate convictions. At least 2
            features are required to get familiarity conviction. If not
            specified all features will be used.
        action_features : Collection of str, optional
            A collection of feature names to be treated as action features
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
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.

        Returns
        -------
        dict
            A dict with familiarity_conviction_addition or
            familiarity_conviction_removal
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        util.validate_list_shape(features, 1, "features", "str")
        util.validate_list_shape(action_features, 1, "action_features", "str")
        if self.configuration.verbose:
            print(f'Getting conviction of features for Trainee with id: {trainee_id}')
        return self.execute(trainee_id, "get_feature_conviction", {
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

        Updates the accumulated data mass for the model proportional to the
        number of cases modified.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee add the feature to.
        feature : str
            The name of the feature.
        feature_attributes : Mapping, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id

        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        if feature_attributes is not None:
            updated_attributes = internals.preprocess_feature_attributes({feature: feature_attributes})
            if updated_attributes is None:
                raise AssertionError("Failed to preprocess feature attributes for new feature.")
            feature_attributes = updated_attributes[feature]

        # Convert session instance to id
        if (
            isinstance(condition, MutableMapping) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Adding feature "{feature}" to Trainee with id {trainee_id}.')
        self.execute(trainee_id, "add_feature", {
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
        cached_trainee = self.trainee_cache.get_item(trainee_id)
        cached_trainee["feature_attributes"] = self.get_feature_attributes(trainee_id)

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

        Updates the accumulated data mass for the model proportional to the
        number of cases modified.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee remove the feature from.
        feature : str
            The name of the feature to remove.
        condition : Mapping, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id

        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        # Convert session instance to id
        if (
            isinstance(condition, MutableMapping) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.configuration.verbose:
            print(f'Removing feature "{feature}" from Trainee with id: {trainee_id}')
        self.execute(trainee_id, "remove_feature", {
            "feature": feature,
            "condition": condition,
            "session": self.active_session.id,
            "condition_session": condition_session,
        })
        self._auto_persist_trainee(trainee_id)

        # Update trainee in cache
        cached_trainee = self.trainee_cache.get_item(trainee_id)
        cached_trainee["feature_attributes"] = self.get_feature_attributes(trainee_id)

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
        use_case_weights: t.Optional[bool] = None,
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
        features : Collection of str, optional
            List of feature names to use when computing pairwise distances.
            If unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this `action_feature`, otherwise uses targetless
            hyperparameters.
        from_case_indices : Sequence of tuple of {str, int}, optional
            A sequence of tuples, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of `to_values` or `to_case_indices`.
        from_values : list of list of object or pandas.DataFrame, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of `to_values` or `to_case_indices`.
        to_case_indices : Sequence of tuple of {str, int}, optional
            A sequence of tuples, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of `from_values` or `from_case_indices`.
        to_values : list of list of object or pandas.DataFrame, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of `from_values` or `from_case_indices`.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        list
            A list of computed pairwise distances between each corresponding
            pair of cases in `from_case_indices` and `to_case_indices`.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)

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

        result = self.execute(trainee_id, "get_pairwise_distances", {
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
        features: t.Optional[Collection[str]] = None,
        *,
        action_feature: t.Optional[str] = None,
        case_indices: t.Optional[CaseIndices] = None,
        feature_values: t.Optional[Collection[t.Any] | DataFrame] = None,
        use_case_weights: t.Optional[bool] = None,
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
        features : Collection of str, optional
            List of feature names to use when computing distances. If
            unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this `action_feature`, otherwise uses targetless
            hyperparameters.
        case_indices : Sequence of tuple of {str, int}, optional
            A sequence of tuples, of session id and index, where index is
            the original 0-based index of the case as it was trained into the
            session. If specified, returns distances for all of these
            cases. Ignored if `feature_values` is provided. If neither
            `feature_values` nor `case_indices` is specified, uses full dataset.
        feature_values : Collection of object or DataFrame, optional
            If specified, returns distances of the local model relative to
            these values, ignores `case_indices` parameter. If provided a
            DataFrame, only the first row will be used.
        use_case_weights : bool, optional
            If set to True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights
            will be used if the Trainee has them.
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
                    'distances': DataFrame[ distances ]
                }
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)

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
                result = self.execute(trainee_id, "get_distances", {
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
            'distances': internals.deserialize_to_dataframe(distances_matrix)
        }

    def get_params(
        self,
        trainee_id: str,
        *,
        action_feature: t.Optional[str] = None,
        context_features: t.Optional[Collection[str]] = None,
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
            The ID of the Trainee.
        action_feature : str, optional
            If specified will return the best analyzed hyperparameters to
            target this feature.
        context_features : Collection of str, optional
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Getting model attributes from Trainee with id: {trainee_id}')
        return self.execute(trainee_id, "get_params", {
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
            The ID of the Trainee.

        params : Mapping
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
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Setting model attributes for Trainee with id: {trainee_id}')

        parameters = dict(params)
        deprecated_params = {
            'auto_optimize_enabled': 'auto_analyze_enabled',
            'optimize_threshold': 'analyze_threshold',
            'optimize_growth_factor': 'analyze_growth_factor'
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

        self.execute(trainee_id, "set_params", parameters)
        self._auto_persist_trainee(trainee_id)

    def clear_imputed_data(
        self,
        trainee_id: str,
        impute_session: t.Optional[str | Session] = None
    ):
        """
        Clears values that were imputed during a specified session.

        Won't clear values that were manually set by the user after the impute.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        impute_session : str or Session, optional
            Session or session identifier of the impute for which to clear the data.
            If none is provided, will clear all imputed.
        """
        trainee_id = self._resolve_trainee(trainee_id).id

        if not self.active_session:
            raise HowsoError(self.ERROR_MESSAGES["missing_session"], code="missing_session")

        self.execute(trainee_id, "clear_imputed_data", {
            "session": self.active_session.id,
            "impute_session": impute_session,
        })
