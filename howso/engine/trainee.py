from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    Union,
)
import uuid
import warnings

from howso.client import AbstractHowsoClient
from howso.client.exceptions import HowsoApiError, HowsoError
from howso.client.pandas import HowsoPandasClientMixin
from howso.client.protocols import ProjectClient
from howso.direct import HowsoDirectClient
from howso.engine.client import get_client
from howso.engine.project import Project
from howso.engine.session import Session
from howso.openapi.models import (
    Project as BaseProject,
    Session as BaseSession,
    Trainee as BaseTrainee,
)
from howso.utilities import CaseIndices
from howso.utilities.feature_attributes.base import SingleTableFeatureAttributes

if TYPE_CHECKING:
    from howso.openapi.models import (
        Metrics,
        TraineeIdentity,
        TraineeInformation,
        TraineeResources,
    )
    from pandas import DataFrame, Index

    class Reaction(TypedDict):
        """React response format."""

        action: DataFrame
        explanation: Dict[str, Any]

    class Distances(TypedDict):
        """Distances response format."""

        session_indices: List[Tuple[str, int]]
        distances: DataFrame

__all__ = [
    "Trainee",
    "list_trainees",
    "get_trainee",
    "delete_trainee",
]


class Trainee(BaseTrainee):
    """
    A Howso Trainee.

    A Trainee is most closely related to what would normally be called a
    'model' in Machine Learning. It contains feature information,
    training cases, session data, parameters, and other metadata. A Trainee is
    actually a little more abstract than a model which is why we don't use
    the terms interchangeably.

    Parameters
    ----------
    name : str, optional
        The name of the trainee.
    features : dict of {str: dict}
        The feature attributes of the trainee. Where feature `name` is the key
        and a sub dictionary of feature attributes is the value.
    default_action_features : list of str, optional
        The default action feature names of the trainee.
    default_context_features : list of str, optional
        The default context feature names of the trainee.
    library_type : str, optional
        The library type of the Trainee. Valid options include:

            - "st": use single-threaded library.
            - "mt": use multi-threaded library.
    max_wait_time : int or float, default 30
        The number of seconds to wait for a trainee to be created and become
        available before aborting gracefully. Set to `0` (or None) to wait as
        long as the system-configured maximum for sufficient resources to
        become available, which is typically 20 minutes.
    persistence : str, default "allow"
        The requested persistence state of the trainee. Allowed values include
        "allow", "always", and "never".
    project : str or Project, optional
        The instance or id of the project to use for the trainee.
    metadata : dict, optional
        Any key-value pair to store as custom metadata for the trainee.
    resources : TraineeResources or dict, optional
        Customize the resources provisioned for the Trainee instance.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.
    overwrite_existing : bool, default False
        Overwrite existing trainee with the same name (if exists).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        features: Optional[Dict[str, Dict]] = None,
        *,
        default_action_features: Optional[Iterable[str]] = None,
        default_context_features: Optional[Iterable[str]] = None,
        id: Optional[str] = None,
        library_type: Optional[str] = None,
        max_wait_time: Optional[Union[int, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persistence: Optional[str] = "allow",
        project: Optional[Union[str, BaseProject]] = None,
        resources: Optional[Union["TraineeResources", Dict[str, Any]]] = None,
        client: Optional[AbstractHowsoClient] = None,
        overwrite_existing: Optional[bool] = False,
    ) -> None:
        """Implement the constructor."""
        self._created: bool = False
        self._updating: bool = False
        self._was_saved: bool = False
        self.client = client or get_client()

        # Set the trainee properties
        self._features = features
        self._metadata = metadata
        self._id = id
        self.name = name
        self.persistence = persistence
        self.set_default_features(
            action_features=default_action_features,
            context_features=default_context_features,
        )

        # Allow passing project id or the project instance
        if isinstance(project, BaseProject):
            self._project_id = project.id
            self._project_instance = Project.from_openapi(
                project, client=self.client)
        else:
            self._project_id = project
            self._project_instance = None  # lazy loaded

        # Create the trainee at the API
        self._create(
            library_type=library_type,
            max_wait_time=max_wait_time,
            overwrite=overwrite_existing,
            resources=resources,
        )

    @property
    def id(self) -> str:
        """
        The unique identifier of the trainee.

        Returns
        -------
        str
            The trainee's ID.
        """
        return self._id

    @property
    def project_id(self) -> Optional[str]:
        """
        The unique identifier of the trainee's project.

        Returns
        -------
        str or None
            The trainee's project ID.
        """
        return self._project_id

    @property
    def project(self) -> Optional[Project]:
        """
        The trainee's project.

        Returns
        -------
        Project or None
            The trainee's project.
        """
        if (
            not self.project_id or
            not isinstance(self.client, ProjectClient)
        ):
            return None

        if (
            self._project_instance is None or
            self._project_instance.id != self.project_id
        ):
            project = self.client.get_project(self.project_id)
            self._project_instance = Project.from_openapi(
                project, client=self.client)

        return self._project_instance

    @property
    def name(self) -> Optional[str]:
        """
        The name of the trainee.

        Returns
        -------
        str or None
            The name.
        """
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        """
        Set the name of the trainee.

        Parameters
        ----------
        name : str or None
            The name.

        Returns
        -------
        None
        """
        if name is not None and len(name) > 128:
            raise ValueError(
                "Invalid value for `name`, length must be less "
                "than or equal to `128`"
            )
        self._name = name
        self.update()

    @property
    def persistence(self) -> str:
        """
        The persistence state of the trainee.

        Returns
        -------
        str
            The trainee's persistence value.
        """
        return self._persistence

    @persistence.setter
    def persistence(self, persistence) -> None:
        """
        Set the persistence state of the trainee.

        Parameters
        ----------
        persistence : str
            The persistence value. Allowed values include: "allow", "always",
            and "never".

        Returns
        -------
        None
        """
        allowed_values = ["allow", "always", "never"]
        if persistence not in allowed_values:
            raise ValueError(
                f"Invalid value for `persistence` ({persistence}), must be"
                f"one of {allowed_values}"
            )
        self._persistence = persistence
        self.update()

    @property
    def features(self) -> SingleTableFeatureAttributes:
        """
        The trainee feature attributes.

        .. WARNING::
            This returns a deep `copy` of the feature attributes. To update
            features attributes of the trainee, use the method
            :func:`set_feature_attributes`.

        Returns
        -------
        FeatureAttributesBase
            The feature attributes of the trainee.
        """
        return SingleTableFeatureAttributes(deepcopy(self._features))

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """
        The trainee metadata.

        .. WARNING::
            This returns a deep `copy` of the metadata. To update the
            metadata of the trainee, use the method :func:`set_metadata`.

        Returns
        -------
        dict
            The metadata of the trainee.
        """
        return deepcopy(self._metadata)

    @property
    def default_action_features(self) -> Optional[List[str]]:
        """
        The default action features of the trainee.

        .. WARNING::
            This returns a deep `copy` of the default action features. To
            update them, use the method :func:`set_default_features`.

        Returns
        -------
        list of str or None
            The default action feature names for the trainee.
        """
        return deepcopy(self._default_action_features)

    @property
    def default_context_features(self) -> Optional[List[str]]:
        """
        The default context features of the trainee.

        .. WARNING::
            This returns a deep `copy` of the default context features. To
            update them, use the method :func:`set_default_features`.

        Returns
        -------
        list of str or None
            The default context feature names for the trainee.
        """
        return deepcopy(self._default_context_features)

    @property
    def active_session(self) -> Session:
        """
        The active session.

        Returns
        -------
        Session
            The session instance.
        """
        return Session.from_openapi(self.client.active_session, client=self.client)

    def save(self, file_path: Optional[Union[Path, str]] = None) -> None:
        """
        Save a Trainee to disk.

        Parameters
        ----------
        file_path : Path or str, optional
            The path of the file to save the Trainee to. This path can contain
            an absolute path, a relative path or simply a file name. If no filepath
            is provided, the default filepath will be the CWD.

            If `file_path` is a relative path (with or without a file name),
            the absolute path will be computed appending the `file_path` to the
            CWD.

            If `file_path` is an absolute path, this is the absolute path that
            will be used.

            If `file_path` does not contain a filename, then the natural
            trainee name will be used `<uuid>.caml`.
        """
        if not isinstance(self.client, HowsoDirectClient):
            raise HowsoError("To save, `client` must be HowsoDirectClient.")

        if file_path:
            if not isinstance(file_path, Path):
                file_path = Path(file_path)

            file_path = file_path.expanduser().resolve()

            # It is decided that if the file contains a suffix then it contains a
            # file name.
            if file_path.suffix:
                # If the final suffix is NOT ".caml", then add the suffix ".caml".
                if file_path.suffix.lower() != '.caml':
                    file_path = file_path.parent.joinpath(f"{file_path.stem}.caml")
                    warnings.warn(
                        'Filepath with a non `.caml` extension was provided. Extension will be '
                        'ignored and the file be will be saved as a `.caml` file', UserWarning)
            else:
                # Add the natural name to the file_path
                file_path = file_path.joinpath(f"{self.id}.caml")

            # If path is not absolute, append it to the default directory.
            if not file_path.is_absolute():
                file_path = self.client.howso.default_save_path.joinpath(file_path)

            # Ensure the parent path exists.
            if not file_path.parents[0].exists():
                file_path.parents[0].mkdir(parents=True, exist_ok=True)

            file_name = file_path.stem
            file_path = f"{file_path.parents[0]}/"
        else:
            file_name = None

        self.client.howso.persist(
            trainee_id=self.id,
            filename=file_name,
            filepath=file_path
        )

    def set_feature_attributes(self, feature_attributes: Dict[str, Dict]) -> None:
        """
        Update the trainee feature attributes.

        Parameters
        ----------
        feature_attributes : dict of {str: dict}
            The feature attributes of the trainee. Where feature `name` is the
            key and a sub dictionary of feature attributes is the value.

        Returns
        -------
        None
        """
        self.client.set_feature_attributes(
            trainee_id=self.id, feature_attributes=feature_attributes
        )
        self._features = self.client.trainee_cache.get(self.id).features

    def set_default_features(
        self,
        *,
        action_features: Optional[Iterable[str]],
        context_features: Optional[Iterable[str]],
    ) -> None:
        """
        Update the trainee default features.

        Parameters
        ----------
        action_features : list of str or None
            The default action feature names.
        context_features : list of str or None
            The default context feature names.

        Returns
        -------
        None
        """
        if action_features is not None:
            self._default_action_features = list(action_features)
        else:
            self._default_action_features = None
        if context_features is not None:
            self._default_context_features = list(context_features)
        else:
            self._default_context_features = None
        self.update()

    def set_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        """
        Update the trainee metadata.

        Parameters
        ----------
        metadata : dict or None
            Any key-value pair to store as custom metadata for the trainee.
            Providing `None` will remove the current metadata.

        Returns
        -------
        None
        """
        self._metadata = metadata
        self.update()

    def copy(
        self,
        name: Optional[str] = None,
        *,
        library_type: Optional[str] = None,
        project: Optional[Union[str, BaseProject]] = None,
        resources: Optional[Union["TraineeResources", Dict[str, Any]]] = None
    ) -> "Trainee":
        """
        Copy the trainee to another trainee.

        Parameters
        ----------
        name : str, optional
            The name of the new trainee.
        library_type : str, optional
            The library type of the Trainee. If not specified, the new trainee
            will inherit the value from the original. Valid options include:

                - "st": use single-threaded library.
                - "mt": use multi-threaded library.
        project : str or Project, optional
            The instance or id of the project to use for the new trainee.
        resources : TraineeResources or dict, optional
            Customize the resources provisioned for the Trainee instance. If
            not specified, the new trainee will inherit the value from the
            original.

        Returns
        -------
        Trainee
            The new trainee copy.
        """
        if isinstance(project, BaseProject):
            project_id = project.id
        else:
            project_id = project or self.project_id

        params = {
            "trainee_id": self.id,
            "new_trainee_name": name,
            "library_type": library_type,
            "resources": resources,
        }

        # Only pass project_id for platform clients
        if isinstance(self.client, ProjectClient):
            params["project_id"] = project_id

        copy = self.client.copy_trainee(**params)

        return Trainee.from_openapi(copy, client=self.client)

    def persist(self) -> None:
        """
        Persist the trainee.

        Returns
        -------
        None
        """
        self.client.persist_trainee(self.id)
        self._was_saved = True

    def delete(self) -> None:
        """
        Delete the trainee.

        Returns
        -------
        None
        """
        if not self.id:
            return
        self.client.delete_trainee(self.id)
        self._created = False
        self._id = None

    def unload(self) -> None:
        """
        Unload the trainee.

        .. deprecated:: 1.0.0
            Use :meth:`Trainee.release_resources` instead.

        Returns
        -------
        None
        """
        warnings.warn(
            'The method `unload()` is deprecated and will be removed '
            'in a future release. Please use `release_resources()` '
            'instead.', DeprecationWarning)
        self.release_resources()

    def acquire_resources(self, *, max_wait_time=None) -> None:
        """
        Acquire resources for a trainee in the Howso service.

        Parameters
        ----------
        max_wait_time : int or float, default 60
            The number of seconds to wait for trainee resources to be acquired
            before aborting gracefully. Set to `0` (or None) to wait as long as
            the system-configured maximum for sufficient resources to become
            available, which is typically 20 minutes.

        Returns
        -------
        None
        """
        self.client.acquire_trainee_resources(
            self.id, max_wait_time=max_wait_time)

    def release_resources(self) -> None:
        """
        Release a trainee's resources from the Howso service.

        Returns
        -------
        None
        """
        if not self.id:
            return
        self.client.release_trainee_resources(self.id)

    def information(self) -> "TraineeInformation":
        """
        Get detail information about the trainee.

        Returns
        -------
        TraineeInformation
            The trainee detail information. Including trainee version and
            configuration parameters.
        """
        return self.client.get_trainee_information(self.id)

    def metrics(self) -> "Metrics":
        """
        Get metric information of the trainee.

        Returns
        -------
        Metrics
            The trainee metric information. Including cpu and memory.
        """
        return self.client.get_trainee_metrics(self.id)

    def set_random_seed(self, seed: Union[int, float, str]) -> None:
        """
        Set the random seed for the trainee.

        Parameters
        ----------
        seed : int or float or str
            The random seed.

        Returns
        -------
        None
        """
        self.client.set_random_seed(trainee_id=self.id, seed=seed)

    def train(
        self,
        cases: Union[List[List[object]], "DataFrame"],
        *,
        ablatement_params: Optional[Dict[str, List[object]]] = None,
        accumulate_weight_feature: Optional[str] = None,
        batch_size: Optional[int] = None,
        derived_features: Optional[Iterable[str]] = None,
        features: Optional[Iterable[str]] = None,
        input_is_substituted: Optional[bool] = False,
        progress_callback: Optional[Callable] = None,
        series: Optional[str] = None,
        train_weights_only: Optional[bool] = False,
        validate: bool = True,
    ) -> None:
        """
        Train one or more cases into the trainee (model).

        Parameters
        ----------
        cases : list of list of object or pandas.DataFrame
            One or more cases to train into the model.
        ablatement_params : dict [str, list of obj], optional
            A dict of feature name to threshold type.
            Valid thresholds include:

                - ['exact']: Don't train if prediction matches exactly
                - ['tolerance', MIN, MAX]: Don't train if ``prediction
                  >= (case value - MIN) & prediction <= (case value + MAX)``
                - ['relative', PERCENT]: Don't train if
                  ``abs(prediction - case value) / prediction <= PERCENT``
                - ['residual']: Don't train if
                  ``abs(prediction - case value) <= feature residual``

        accumulate_weight_feature : str, default None
            Name of feature into which to accumulate neighbors'
            influences as weight for ablated cases. If unspecified, will not
            accumulate weights.
        batch_size : int or None, optional
            Define the number of cases to train at once. If left unspecified,
            the batch size will be determined automatically.
        derived_features : list of str, optional
            List of feature names for which values should be derived
            in the specified order. If this list is not provided, features with
            the 'auto_derive_on_train' feature attribute set to True will be
            auto-derived. If provided an empty list, no features are derived.
            Any derived_features that are already in the 'features' list will
            not be derived since their values are being explicitly provided.
        features : list of str, optional
            A list of feature names.
            This parameter should be provided in the following scenarios:

                a. When cases are not in the format of a DataFrame, or
                   the DataFrame does not define named columns.
                b. You want to train only a subset of columns defined in your
                   cases DataFrame.
                c. You want to re-order the columns that are trained.

        input_is_substituted : bool, default False
            If True assumes provided nominal feature values have already
            been substituted.
        progress_callback : callable or None, optional
            (Optional) A callback method that will be called before each
            batched call to train and at the end of training. The method is
            given a ProgressTimer containing metrics on the progress and timing
            of the train operation.
        series : str, optional
            The name of the series to pull features and case values from
            internal series storage. If specified, trains on all cases that
            are stored in the internal series store for the specified series.
            The trained feature set is the combined features from storage and
            the passed in features. If cases is of length one, the value(s) of
            this case are appended to all cases in the series. If cases is the
            same length as the series, the value of each case in cases is
            applied in order to each of the cases in the series.
        train_weights_only:  bool, default False
            When true, and accumulate_weight_feature is provided,
            will accumulate all of the cases' neighbor weights instead of
            training the cases into the model.
        validate : bool, default True
            Whether to validate the data against the provided feature
            attributes. Issues warnings if there are any discrepancies between
            the data and the features dictionary.

        Returns
        -------
        None
        """
        self.client.train(
            trainee_id=self.id,
            ablatement_params=ablatement_params,
            accumulate_weight_feature=accumulate_weight_feature,
            batch_size=batch_size,
            cases=cases,
            derived_features=derived_features,
            features=features,
            input_is_substituted=input_is_substituted,
            progress_callback=progress_callback,
            series=series,
            train_weights_only=train_weights_only,
            validate=validate,
        )

    def optimize(self, *args, **kwargs):
        """
        Optimizes a trainee.

        .. deprecated:: 6.0.0
            Use :meth:`Trainee.analyze` instead.

        Parameters
        ----------
        context_features : iterable of str, optional
            The context features to optimize for.
        action_features : iterable of str, optional
            The action features to optimize for.
        k_folds : int
            optional, (default 6) number of cross validation folds to do
        bypass_hyperparameter_optimization : bool
            optional, bypasses hyperparameter optimization
        bypass_calculate_feature_residuals : bool
            optional, bypasses feature residual calculation
        bypass_calculate_feature_weights : bool
            optional, bypasses calculation of feature weights
        use_deviations : bool
            optional, uses deviations for LK metric in queries
        num_samples : int
            used in calculating feature residuals
        k_values : list of int
            optional list used in hyperparameter search
        p_values : list of float
            optional list used in hyperparameter search
        dwe_values : list of float
            optional list used in hyperparameter search
        optimize_level : int
            optional value, if specified, will optimize for the following
            flows:

                1. predictions/accuracy (hyperparameters)
                2. data synth (cache: global residuals)
                3. standard explanations
                4. full analysis
        targeted_model : {"omni_targeted", "single_targeted", "targetless"}
            optional, valid values as follows:

                "single_targeted" = optimize hyperparameters for the
                    specified action_features
                "omni_targeted" = optimize hyperparameters for each context
                    feature as an action feature, ignores action_features
                    parameter
                "targetless" = optimize hyperparameters for all context
                    features as possible action features, ignores
                    action_features parameter
        num_optimization_samples : int, optional
            If the dataset size to too large, optimize on
            (randomly sampled) subset of data. The
            `num_optimization_samples` specifies the number of
            observations to be considered for optimization.
        optimization_sub_model_size : int or Node, optional
            Number of samples to use for optimization. The rest
            will be randomly held-out and not included in calculations.
        inverse_residuals_as_weights : bool, default is False
            When True will compute and use inverse of residuals
            as feature weights
        use_case_weights : bool, default False
            When True will scale influence weights by each
            case's weight_feature weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        kwargs
            Additional experimental optimize parameters.
        """
        warnings.warn(
            'The method `optimize()` is deprecated and will be'
            'removed in a future release. Please use `analyze()` '
            'instead.', DeprecationWarning)

        self.analyze(*args, **kwargs)

    def auto_optimize(self):
        """
        Auto-optimize the trainee model.

        Re-uses all parameters from the previous optimize or
        set_auto_optimize_params call. If optimize or set_auto_optimize_params
        has not been previously called, auto_optimize will default to a robust
        and versatile optimization.

        .. deprecated:: 6.0.0
            Use :meth:`Trainee.auto_analyze` instead.
        """
        warnings.warn(
            'The method `auto_optimize()` is deprecated and will be'
            'removed in a future release. Please use `auto_analyze()` '
            'instead.', DeprecationWarning)

        return self.auto_analyze()

    def set_auto_optimize_params(self, *args, **kwargs):
        """
        Set trainee parameters for auto optimization.

        .. deprecated:: 6.0.0
            Use :meth:`Trainee.set_auto_analyze_params` instead.

        Parameters
        ----------
        auto_optimize_enabled : bool, default False
            When True, the :func:`train` method will trigger an optimize when
            it's time for the model to be optimized again.
        optimize_threshold : int, optional
            The threshold for the number of cases at which the model should be
            re-optimized.
        auto_optimize_limit_size : int, optional
            The size of of the model at which to stop doing auto-optimization.
            Value of 0 means no limit.
        optimize_growth_factor : float, optional
            The factor by which to increase the optimize threshold every time
            the model grows to the current threshold size.
        kwargs : dict, optional
            Parameters specific for optimize() may be passed in via kwargs, and
            will be cached and used during future auto-optimizations.
        """
        warnings.warn(
            'The method `set_auto_optimize_params()` is deprecated and will be'
            'removed in a future release. Please use `set_auto_analyze_params()` '
            'instead.', DeprecationWarning)

        self.set_auto_analyze_params(*args, **kwargs)

    def auto_analyze(self) -> None:
        """
        Auto-analyze the trainee.

        Re-use all parameters from the previous analyze call, assuming that
        the user has called 'analyze' before. If not, it will default to a
        robust and versatile analysis.

        Returns
        -------
        None
        """
        self.client.auto_analyze(self.id)

    def set_auto_analyze_params(
        self,
        auto_analyze_enabled: bool = False,
        analyze_threshold: Optional[int] = None,
        *,
        auto_analyze_limit_size: Optional[int] = None,
        analyze_growth_factor: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Set parameters for auto analysis.

        Auto-analysis is disabled if this is called without specifying an
        analyze_threshold.

        Parameters
        ----------
        auto_analyze_enabled : bool, default False
            When True, the :func:`train` method will trigger an analyze when
            it's time for the model to be analyzed again.
        analyze_threshold : int, optional
            The threshold for the number of cases at which the model should be
            re-analyzed.
        auto_analyze_limit_size : int, optional
            The size of the model at which to stop doing auto-analysis.
            Value of 0 means no limit.
        analyze_growth_factor : float, optional
            The factor by which to increase the analysis threshold every
            time the model grows to the current threshold size.
        kwargs : dict, optional
            See parameters in :func:`analyze`.

        Returns
        -------
        None
        """
        self.client.set_auto_analyze_params(
            trainee_id=self.id,
            auto_analyze_enabled=auto_analyze_enabled,
            auto_analyze_limit_size=auto_analyze_limit_size,
            analyze_growth_factor=analyze_growth_factor,
            analyze_threshold=analyze_threshold,
            **kwargs,
        )

    def analyze(
        self,
        context_features: Optional[Iterable[str]] = None,
        action_features: Optional[Iterable[str]] = None,
        *,
        bypass_calculate_feature_residuals: Optional[bool] = None,
        bypass_calculate_feature_weights: Optional[bool] = None,
        bypass_hyperparameter_analysis: Optional[bool] = None,
        dt_values: Optional[List[float]] = None,
        inverse_residuals_as_weights: Optional[bool] = None,
        k_folds: Optional[int] = None,
        k_values: Optional[List[int]] = None,
        num_analysis_samples: Optional[int] = None,
        num_samples: Optional[int] = None,
        analysis_sub_model_size: Optional[int] = None,
        analyze_level: Optional[int] = None,
        p_values: Optional[List[float]] = None,
        targeted_model: Optional[str] = None,
        use_case_weights: Optional[bool] = None,
        use_deviations: Optional[bool] = None,
        weight_feature: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Analyzes the trainee.

        Parameters
        ----------
        context_features : list of str, optional
            The context features to analyze for.
        action_features : list of str, optional
            The action features to analyze for.
        bypass_calculate_feature_residuals : bool, default False
            When True, bypasses calculation of feature residuals.
        bypass_calculate_feature_weights : bool, default False
            When True, bypasses calculation of feature weights.
        bypass_hyperparameter_analysis : bool, default False
            When True, bypasses hyperparameter analysis.
        dt_values : list of float, optional
            The dt value hyperparameters to analyze with.
        inverse_residuals_as_weights : bool, default is False
            When True, will compute and use inverse of residuals as feature
            weights.
        k_folds : int, optional
            The number of cross validation folds to do. A value of 1 does
            hold-one-out instead of k-fold.
        k_values : list of int, optional
            The k value hyperparameters to analyze with.
        num_analysis_samples : int, optional
            Specifies the number of observations to be considered for
            analysis.
        num_samples : int, optional
            Number of samples used in calculating feature residuals.
        analysis_sub_model_size : int, optional
            Number of samples to use for analysis. The rest will be
            randomly held-out and not included in calculations.
        analyze_level : int, optional
            If specified, will analyze for the following flows:

                1. Predictions/accuracy (hyperparameters)
                2. Data synth (cache: global residuals)
                3. Standard explanations
                4. Full analysis

        p_values : list of float, optional
            The p value hyperparameters to analyze with.
        targeted_model : str or None
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

        use_case_weights : bool, default False
            (Optional) When True will scale influence weights by each
            case's weight_feature weight.
        use_deviations : bool, default False
            When True, uses deviations for LK metric in queries.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        kwargs
            Additional experimental analyze parameters.

        Returns
        -------
        None
        """
        self.client.analyze(
            trainee_id=self.id,
            action_features=action_features,
            context_features=context_features,
            bypass_calculate_feature_residuals=bypass_calculate_feature_residuals,  # noqa: E501
            bypass_calculate_feature_weights=bypass_calculate_feature_weights,
            bypass_hyperparameter_analysis=bypass_hyperparameter_analysis,  # noqa: E501
            dt_values=dt_values,
            use_case_weights=use_case_weights,
            inverse_residuals_as_weights=inverse_residuals_as_weights,
            k_folds=k_folds,
            k_values=k_values,
            num_analysis_samples=num_analysis_samples,
            num_samples=num_samples,
            analysis_sub_model_size=analysis_sub_model_size,
            analyze_level=analyze_level,
            p_values=p_values,
            targeted_model=targeted_model,
            use_deviations=use_deviations,
            weight_feature=weight_feature,
            **kwargs
        )

    def predict(
        self,
        contexts: Optional[Union[List[List[object]], "DataFrame"]] = None,
        *,
        action_features: Optional[Iterable[str]] = None,
        allow_nulls: Optional[bool] = False,
        case_indices: Optional[CaseIndices] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        leave_case_out: Optional[bool] = None,
        suppress_warning: Optional[bool] = False,
        use_case_weights: Optional[bool] = False,
        weight_feature: Optional[str] = None,
    ) -> list:
        """
        Wrapper around :func:`react`.

        Performs a discriminative react to predict the action feature values based on the
        given contexts. Returns only the predicted action values.

        .. seealso::
            :func:`react`

        Parameters
        ----------
        contexts : list of list of object or pandas.DataFrame, optional
            (Optional) The context values to react to. If context values are not specified,
            then `case_indices` must be specified.
        action_features : list of str, optional
            (Optional) Feature names to treat as action features during react. If no
            `action_features` is specified, the Trainee `default_action_features`
            is used.
        allow_nulls : bool, default False, optional
            See parameter ``allow_nulls`` in :func:`react`.
        case_indices : Iterable of Sequence[str, int], default None, optional
            (Optional) Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If this case does not exist, discriminative react
            outputs null.
        context_features : list of str, optional
            (Optional) Feature names to treat as context features during react. If no
            `context_features` is specified, then the Trainee's `default_action_features`
            are used. If the Trainee has no `default_action_features`, then
            `context_features` will be all of the `features` excluding the
            `action_features`.
        derived_action_features : list of str, optional
            See parameter ``derived_action_features`` in :func:`react`.
        derived_context_features : list of str, optional
            See parameter ``derived_context_features`` in :func:`react`.
        leave_case_out : bool, default False
            See parameter ``leave_case_out`` in :func:`react`.
        suppress_warning : bool, default False
            See parameter ``suppress_warning`` in :func:`react`.
        use_case_weights : bool, default False
            See parameter ``use_case_weights`` in :func:`react`.
        weight_feature : str, optional
            See parameter ``weight_feature`` in :func:`react`.

        Returns
        -------
        pandas.DataFrame
            DataFrame consisting of the discriminative predicted results.
        """
        if action_features is None:
            if self.default_action_features is None:
                raise HowsoError(
                    "No action features specified and no default action features are present. "
                    "Please specify the action feature or add default action features "
                    "to the Trainee."
                )
            else:
                action_features = self.default_action_features

        if context_features is None:
            if self.default_context_features is None:
                context_features = [key for key in self.features.keys() if key not in action_features]
            else:
                context_features = self.default_context_features

        results = self.react(
            action_features=action_features,
            allow_nulls=allow_nulls,
            case_indices=case_indices,
            contexts=contexts,
            context_features=context_features,
            derived_action_features=derived_action_features,
            derived_context_features=derived_context_features,
            leave_case_out=leave_case_out,
            suppress_warning=suppress_warning,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )

        return results['action']

    def react(
        self,
        contexts: Optional[Union[List[List[object]], "DataFrame"]] = None,
        *,
        action_features: Optional[Iterable[str]] = None,
        actions: Optional[Union[List[List[object]], "DataFrame"]] = None,
        allow_nulls: Optional[bool] = False,
        case_indices: Optional[CaseIndices] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict[str, object]] = None,
        feature_bounds_map: Optional[Dict[str, Dict[str, object]]] = None,
        generate_new_cases: Optional[str] = "no",
        input_is_substituted: Optional[bool] = False,
        into_series_store: Optional[str] = None,
        leave_case_out: Optional[bool] = None,
        new_case_threshold: Optional[str] = "min",
        num_cases_to_generate: Optional[int] = 1,
        ordered_by_specified_features: Optional[bool] = False,
        preserve_feature_values: Optional[Iterable[str]] = None,
        progress_callback: Optional[Callable] = None,
        substitute_output: Optional[bool] = True,
        suppress_warning: Optional[bool] = False,
        use_case_weights: Optional[bool] = False,
        use_regional_model_residuals: Optional[bool] = True,
        weight_feature: Optional[str] = None,
    ) -> "Reaction":
        """
        React to the trainee.

        If `desired_conviction` is specified, executes a generative react,
        producing `action_values` for the specified `action_features`
        conditioned on the optionally provided `contexts`.

        If `desired_conviction` is **not** specified, executes a discriminative
        react. Provided a list of `contexts`, the trainee reacts to the model
        and produces predictions for the specified actions.

        Parameters
        ----------
        contexts : list of list of object or pandas.DataFrame, optional
            The context values to react to.
        action_features : list of str, optional
            Feature names to treat as action features during react.
        actions : list of list of object or pandas.DataFrame, optional
            One or more action values to use for action features.
            If specified, will only return the specified explanation
            details for the given actions. (Discriminative reacts only)
        allow_nulls : bool, default False
            (Optional) When true will allow return of null values if there
            are nulls in the local model for the action features, applicable
            only to discriminative reacts.
        case_indices : Iterable of Sequence[str, int], defaults to None
            (Optional) Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If this case does not exist, discriminative react
            outputs null, generative react ignores it.
        context_features : list of str, optional
            Feature names to treat as context features during react.
        derived_action_features : list of str, optional
            Features whose values should be computed after reaction from
            the resulting case prior to output, in the specified order.
            Must be a subset of action_features.

            .. NOTE::
                Both of these derived feature lists rely on the features'
                "derived_feature_code" attribute to compute the values. If the
                "derived_feature_code" attribute is undefined or references a
                non-0 feature indices, the derived value will be null.

        derived_context_features : list of str, optional
            Features whose values should be computed from the provided
            context in the specified order.
        desired_conviction : float, optional
            If specified will execute a generative react. If not
            specified will execute a discriminative react. Conviction is the
            ratio of expected surprisal to generated surprisal for each
            feature generated, valid values are in the range of (0,infinity].
        details : dict of {str: object}
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
            - case_contributions : bool, optional
                If True, outputs each influential case's differences between the
                predicted action feature value and the predicted action feature
                value if each individual case were not included. Uses only the
                context features of the reacted case to determine that area.
                Relies on 'robust_computation' parameter to determine whether
                to do standard or robust computation.
            - case_feature_residuals : bool, optional
                If True, outputs feature residuals for all (context and action)
                features for just the specified case. Uses leave-one-out for
                each feature, while using the others to predict the left out
                feature with their corresponding values from this case. Relies
                on 'robust_computation' parameter to determine whether to do
                standard or robust computation.
            - case_mda : bool, optional
                If True, outputs each influential case's mean decrease in
                accuracy of predicting the action feature in the local model
                area, as if each individual case were included versus not
                included. Uses only the context features of the reacted case to
                determine that area. Relies on 'robust_computation' parameter
                to determine whether to do standard or robust computation.
            - categorical_action_probabilities : bool, optional
                If True, outputs probabilities for each class for the action.
                Applicable only to categorical action features.
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
            - feature_contributions : bool, optional
                If True, outputs each context feature's differences between the
                predicted action feature value and the predicted action feature
                value if each context were not in the model for all context
                features in the local model area. Relies on 'robust_computation'
                parameter to determine whether to do standard or robust
                computation.
            - case_feature_contributions: bool, optional
                If True outputs each context feature's differences between the
                predicted action feature value and the predicted action feature
                value if each context feature were not in the model for all
                context features in this case, using only the values from this
                specific case. Relies on 'robust_computation' parameter to
                determine whether to do standard or robust computation.
            - feature_mda : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature given the context.
                Uses only the context features of the reacted case to determine
                that area. Relies on 'robust_computation' parameter to
                determine whether to do standard or robust computation.
            - feature_mda_ex_post : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature as an explanation
                given that the specified prediction was already made as
                specified by the action value. Uses both context and action
                features of the reacted case to determine that area. Relies on
                'robust_computation' parameter to determine whether to do
                standard or robust computation.
            - feature_residuals : bool, optional
                If True, outputs feature residuals for all (context and action)
                features locally around the prediction. Uses only the context
                features of the reacted case to determine that area. Relies on
                'robust_computation' parameter to determine whether to do
                standard or robust computation.
            - global_case_feature_residual_convictions : bool, optional
                If True, outputs this case's feature residual convictions for
                the global model. Computed as: global model feature residual
                divided by case feature residual. Relies on
                'robust_computation' parameter to determine whether to do
                standard or robust computation.
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
            - local_case_feature_residual_convictions : bool, optional
                If True, outputs this case's feature residual convictions for
                the region around the prediction. Uses only the context
                features of the reacted case to determine that region.
                Computed as: region feature residual divided by case feature
                residual. Relies on 'robust_computation' parameter to determine
                whether to do standard or robust computation.
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
            - prediction_residual_conviction: bool, optional
                If True, outputs residual conviction for the reacted case's
                action features by computing the prediction residual for the
                action features in the local model area. Uses both context and
                action features to determine that area. This is defined as the
                expected (global) model residual divided by computed local
                residual.
            - similarity_conviction : bool, optional
                If True, outputs similarity conviction for the reacted case.
                Uses both context and action feature values as the case values
                for all computations. This is defined as expected (local)
                distance contribution divided by reacted case distance
                contribution.
            - robust_computation : bool, optional
                Default is False, uses leave-one-out for features (or cases,
                as needed) for all relevant computations. If True, uses
                uniform sampling from the power set of all combinations of
                features (or cases, as needed) instead.

        feature_bounds_map : dict of {str: dict of {str: object}}, optional
            A mapping of feature names to the bounds for the feature values to
            be generated in. For continuous features this should be a numeric
            value, for datetimes this should be a datetime string or a numeric
            epoch value. Min bounds should be equal to or smaller than max
            bounds, except when setting the bounds around the cycle length of
            a cyclic feature. (e.g., to allow 0 +/- 60 degrees, set min=300
            and max=60).

            Example::

                {
                    "feature_a": {"min": 0},
                    "feature_b" : {"min": 1, "max": 5},
                    "feature_c": {"max": 1}
                }

        generate_new_cases : str, default "no"
            This parameter takes in a string that may be one of the following:

                - **attempt**: `Synthesizer` attempts to generate new cases and
                  if its not possible to generate a new case, it might
                  generate cases in "no" mode (see point c.)
                - **always**: `Synthesizer` always generates new cases and
                  if its not possible to generate a new case, it returns
                  `None`.
                - **no**: `Synthesizer` generates data based on the
                  `desired_conviction` specified and the generated data is
                  not guaranteed to be a new case (that is, a case not found
                  in original dataset.)

        input_is_substituted : bool, default False
            When True, assumes provided categorical (nominal or ordinal)
            feature values have already been substituted.
        into_series_store : str, optional
            The name of a series store. If specified, will store an internal
            record of all react contexts for this session and series to be used
            later with train series.
        leave_case_out : bool, default False
            When True and specified along with `case_indices`, each individual
            react will respectively ignore the corresponding case specified
            by `case_indices` by leaving it out.
        new_case_threshold : str, default None
            (Optional) Distance to determine the privacy cutoff. If None,
            will default to "min".

            Possible values:

                - min: minimum distance in the original local space.
                - max: maximum distance in the original local space.
                - most_similar: distance between the nearest neighbor to the nearest
                  neighbor in the original space.
        num_cases_to_generate : int, default 1
            The number of cases to generate.
        ordered_by_specified_features : bool, default False
            When True, the order of generated feature values will match
            the order of specified features.
        preserve_feature_values : list of str, optional
            Features that will preserve their values from the case specified
            by `case_indices`, appending and overwriting the specified
            contexts as necessary. For generative reacts, if `case_indices`
            isn't specified will preserve feature values of a random case.
        progress_callback : callable or None, optional
            (Optional) A callback method that will be called before each
            batched call to react and at the end of reacting. The method is
            given a ProgressTimer containing metrics on the progress and timing
            of the react operation, and the batch result.
        substitute_output : bool, default True
            When False, will not substitute categorical feature values. Only
            applicable if a substitution value map has been set.
        suppress_warning : bool, default False
            When True, warnings will not be displayed.
        use_case_weights : bool, default False
            When True, will scale influence weights by each case's
            `weight_feature` weight.
        use_regional_model_residuals : bool, default True
            When false, uses model feature residuals. When True, recalculates
            regional model residuals.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        dict of {action: pandas.DataFrame, explanation: dict}
            The action values and explanations.
        """
        return self.client.react(
            trainee_id=self.id,
            action_features=action_features,
            actions=actions,
            allow_nulls=allow_nulls,
            case_indices=case_indices,
            contexts=contexts,
            context_features=context_features,
            derived_action_features=derived_action_features,
            derived_context_features=derived_context_features,
            desired_conviction=desired_conviction,
            details=details,
            feature_bounds_map=feature_bounds_map,
            generate_new_cases=generate_new_cases,
            input_is_substituted=input_is_substituted,
            into_series_store=into_series_store,
            leave_case_out=leave_case_out,
            new_case_threshold=new_case_threshold,
            num_cases_to_generate=num_cases_to_generate,
            ordered_by_specified_features=ordered_by_specified_features,
            preserve_feature_values=preserve_feature_values,
            progress_callback=progress_callback,
            substitute_output=substitute_output,
            suppress_warning=suppress_warning,
            use_case_weights=use_case_weights,
            use_regional_model_residuals=use_regional_model_residuals,
            weight_feature=weight_feature,
        )

    def react_series(
        self,
        contexts: Optional[Union[List[List[object]], "DataFrame"]] = None,
        *,
        action_features: Optional[Iterable[str]] = None,
        actions: Optional[Union[List[List[object]], "DataFrame"]] = None,
        case_indices: Optional[CaseIndices] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict[str, object]] = None,
        feature_bounds_map: Optional[Dict[str, Dict[str, object]]] = None,
        final_time_steps: Optional[List[object]] = None,
        generate_new_cases: Optional[str] = "no",
        series_index: Optional[str] = ".series",
        init_time_steps: Optional[List[object]] = None,
        initial_features: Optional[Iterable[str]] = None,
        initial_values: Optional[Union[List[List[object]], "DataFrame"]] = None,
        input_is_substituted: Optional[bool] = False,
        leave_case_out: Optional[bool] = None,
        max_series_lengths: Optional[List[int]] = None,
        new_case_threshold: Optional[str] = "min",
        num_series_to_generate: Optional[int] = 1,
        ordered_by_specified_features: Optional[bool] = False,
        output_new_series_ids: Optional[bool] = True,
        preserve_feature_values: Optional[Iterable[str]] = None,
        progress_callback: Optional[Callable] = None,
        series_context_features: Optional[Iterable[str]] = None,
        series_context_values: Optional[
            Union[List[List[List[object]]], List["DataFrame"]]
        ] = None,
        series_id_tracking: Optional[Literal["fixed", "dynamic", "no"]] = "fixed",
        series_stop_maps: Optional[List[Dict[str, Dict[str, object]]]] = None,
        substitute_output: Optional[bool] = True,
        suppress_warning: Optional[bool] = False,
        use_case_weights: Optional[bool] = False,
        use_regional_model_residuals: Optional[bool] = True,
        weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        React to the trainee in a series until a stop condition is met.

        Aggregates rows of data corresponding to the specified context, action,
        derived_context and derived_action features, utilizing previous rows to
        derive values as necessary. Outputs an dict of "action_features" and
        corresponding "series" where "series" is the completed 'matrix' for
        the corresponding action_features and derived_action_features.

        Parameters
        ----------
        contexts : list of list of object or pandas.DataFrame, optional
            The context values to react to.
        action_features : list of str, optional
            See parameter ``action_features`` in :func:`react`.
        actions : list of list of object or pandas.DataFrame, optional
            See parameter ``actions`` in :func:`react`.
        case_indices : Iterable of Sequence[str, int]
            See parameter ``case_indices`` in :func:`react`.
        context_features : list of str, optional
            See parameter ``context_features`` in :func:`react`.
        derived_action_features : list of str, optional
            See parameter ``derived_action_features`` in :func:`react`.
        derived_context_features : list of str, optional
            See parameter ``derived_context_features`` in :func:`react`.
        desired_conviction : float, optional
            See parameter ``desired_conviction`` in :func:`react`.
        details : dict of {str: object}
            See parameter ``details`` in :func:`react`.
        feature_bounds_map : dict of {str: dict of {str: object}}, optional
            See parameter ``feature_bounds_map`` in :func:`react`.
        final_time_steps: list of object, optional
            The time steps at which to end synthesis. Time-series only.
            Time-series only. Must provide either one for all series, or
            exactly one per series.
        generate_new_cases : str, default "no"
            See parameter ``generate_new_cases`` in :func:`react`.
        series_index : str, default ".series"
            When set to a string, will include the series index as a
            column in the returned DataFrame using the column name given.
            If set to None, no column will be added.
        init_time_steps: list of object, optional
            The time steps at which to begin synthesis. Time-series only.
            Time-series only. Must provide either one for all series, or
            exactly one per series.
        initial_features : list of str, optional
            Features to condition just the first case in a series,
            overwrites context_features and derived_context_features for that
            first case. All specified initial features must be in one of:
            context_features, action_features, derived_context_features or
            derived_action_features. If provided a value that isn't in one of
            those lists, it will be ignored.
        initial_values : list of list of object or pandas.DataFrame, optional
            Values corresponding to the initial_features, used to condition
            just the first case in each series. Must provide either exactly one
            value to use for all series, or one per series.
        input_is_substituted : bool, default False
            See parameter ``input_is_substituted`` in :func:`react`.
        leave_case_out : bool, default False
            See parameter ``leave_case_out`` in :func:`react`.
        max_series_lengths : list of int, optional
            Maximum size a series is allowed to be. A 0 or less is no limit.
            Must provide either exactly one to use for all series, or one per
            series. Default is ``3 * model_size``
        new_case_threshold : str or None, optional
            (Optional) See parameter ``new_case_threshold`` in :func:`react`.
        num_series_to_generate : int, default 1
            The number of series to generate.
        ordered_by_specified_features : bool, default False
            See parameter ``ordered_by_specified_features`` in :func:`react`.
        output_new_series_ids : bool, default True
            If True, series ids are replaced with unique values on output.
            If False, will maintain or replace ids with existing trained values,
            but also allows output of series with duplicate existing ids.
        preserve_feature_values : list of str, optional
            See parameter ``preserve_feature_values`` in :func:`react`.
        progress_callback : callable or None, optional
            (Optional) A callback method that will be called before each
            batched call to react series and at the end of reacting. The method
            is given a ProgressTimer containing metrics on the progress and
            timing of the react series operation, and the batch result.
        series_context_features :  list of str, default None
            (Optional) list of context features corresponding to
            series_context_values, if specified must not overlap with any
            initial_features or context_features.
        series_context_values : list of list of list of object or list of pandas.DataFrame, default None
            (Optional) 3d-list of context values, one for each feature for each
            row for each series. If specified, batch_size and
            max_series_lengths are ignored.
        series_id_tracking : {"fixed", "dynamic", "no"}, default "fixed"
            Controls how closely generated series should follow existing series (plural).

            - If "fixed", tracks the particular relevant series ID.
            - If "dynamic", tracks the particular relevant series ID, but is allowed to
              change the series ID that it tracks based on its current context.
            - If "no", does not track any particular series ID.
        series_stop_maps : list of dict of {str: dict}, optional
            Map of series stop conditions. Must provide either exactly one to
            use for all series, or one per series.

            .. TIP::
                Stop series when value exceeds max or is smaller than min::

                    {"feature_name":  {"min" : 1, "max": 2}}

                Stop series when feature value matches any of the values
                listed::

                    {"feature_name":  {"values": ["val1", "val2"]}}

        substitute_output : bool, default True
            See parameter ``substitute_output`` in :func:`react`.
        suppress_warning : bool, default False
            See parameter ``suppress_warning`` in :func:`react`.
        use_case_weights : bool, default False
            See parameter ``use_case_weights`` in :func:`react`.
        use_regional_model_residuals : bool, default True
            See parameter ``use_regional_model_residuals`` in :func:`react`.
        weight_feature : str, optional
            See parameter ``weight_feature`` in :func:`react`.

        Returns
        -------
        pandas.DataFrame
            The series action values.
        """
        return self.client.react_series(
            trainee_id=self.id,
            action_features=action_features,
            actions=actions,
            case_indices=case_indices,
            contexts=contexts,
            context_features=context_features,
            derived_action_features=derived_action_features,
            derived_context_features=derived_context_features,
            desired_conviction=desired_conviction,
            details=details,
            feature_bounds_map=feature_bounds_map,
            final_time_steps=final_time_steps,
            generate_new_cases=generate_new_cases,
            series_index=series_index,
            init_time_steps=init_time_steps,
            initial_features=initial_features,
            initial_values=initial_values,
            input_is_substituted=input_is_substituted,
            leave_case_out=leave_case_out,
            max_series_lengths=max_series_lengths,
            new_case_threshold=new_case_threshold,
            num_series_to_generate=num_series_to_generate,
            ordered_by_specified_features=ordered_by_specified_features,
            output_new_series_ids=output_new_series_ids,
            preserve_feature_values=preserve_feature_values,
            progress_callback=progress_callback,
            series_context_features=series_context_features,
            series_context_values=series_context_values,
            series_id_tracking=series_id_tracking,
            series_stop_maps=series_stop_maps,
            substitute_output=substitute_output,
            suppress_warning=suppress_warning,
            use_case_weights=use_case_weights,
            use_regional_model_residuals=use_regional_model_residuals,
            weight_feature=weight_feature,
        )

    def impute(
        self,
        *,
        batch_size: Optional[int] = 1,
        features: Optional[Iterable[str]] = None,
        features_to_impute: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Impute (fill) the missing values for the specified features_to_impute.

        If no 'features' are specified, will use all features in the trainee
        for imputation. If no 'features_to_impute' are specified, will impute
        all features specified by 'features'.

        Parameters
        ----------
        batch_size : int, default 1
            Larger batch size will increase speed but decrease
            accuracy. Batch size indicates how many rows to fill before
            recomputing conviction.

            The default value (which is 1) should return the best accuracy but
            might be slower. Higher values should improve performance but may
            decrease accuracy of results.
        features : list of str, optional
            A list of feature names to use for imputation. If not specified,
            all features will be used.
        features_to_impute : list of str, optional
            A list of feature names to impute. If not specified, features
            will be used.

        Returns
        -------
        None
        """
        self.client.impute(
            trainee_id=self.id,
            batch_size=batch_size,
            features=features,
            features_to_impute=features_to_impute,
        )

    def remove_cases(
        self,
        num_cases: int,
        *,
        case_indices: Optional[Iterable[Tuple[str, int]]] = None,
        condition: Optional[Dict[str, object]] = None,
        condition_session: Optional[Union[str, BaseSession]] = None,
        distribute_weight_feature: Optional[str] = None,
        precision: Optional[str] = None,
        preserve_session_data: bool = False
    ) -> int:
        """
        Remove training cases from the trainee.

        The training cases will be completely purged from the model and
        the model will behave as if it had never been trained with them.

        Parameters
        ----------
        num_cases : int
            The number of cases to remove; minimum 1 case must be removed.
            Ignored if case_indices is specified.
        case_indices : list of tuples
            A list of tuples containing session ID and session training index
            for each case to be removed.
        condition : dict, optional
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

                    condition = {"feature_name": None}

                Example 2 - Remove cases that have the value 10::

                    condition = {"feature_name": 10}

                Example 3 - Remove cases that have a value in range [10, 20]::

                    condition = {"feature_name": [10, 20]}

                Example 4 - Remove cases that match one of ['a', 'c', 'e']::

                    condition = {"feature_name": ['a', 'c', 'e']}

        condition_session : str or BaseSession, optional
            If specified, ignores the condition and operates on cases for
            the specified session id or BaseSession instance. Ignored if
            case_indices is specified.
        distribute_weight_feature : str, default None
            When specified, will distribute the removed cases' weights
            from this feature into their neighbors.
        precision : str, default None
            The precision to use when removing the cases. Options are 'exact'
            or 'similar'. If not specified "exact" will be used. Ignored if
            case_indices is specified.
        preserve_session_data : bool, default False
            When True, will remove cases without cleaning up session data.

        Returns
        -------
        int
            The number of cases removed.
        """
        if isinstance(condition_session, BaseSession):
            condition_session_id = condition_session.id
        else:
            condition_session_id = condition_session
        return self.client.remove_cases(
            trainee_id=self.id,
            num_cases=num_cases,
            case_indices=case_indices,
            condition=condition,
            condition_session=condition_session_id,
            distribute_weight_feature=distribute_weight_feature,
            precision=precision,
            preserve_session_data=preserve_session_data,
        )

    def edit_cases(
        self,
        feature_values: Union[List[object], "DataFrame"],
        *,
        case_indices: Optional[CaseIndices] = None,
        condition: Optional[Dict[str, object]] = None,
        condition_session: Optional[Union[str, BaseSession]] = None,
        features: Optional[Iterable[str]] = None,
        num_cases: Optional[int] = None,
        precision: Optional[str] = None
    ) -> int:
        """
        Edit feature values for the specified cases.

        Parameters
        ----------
        feature_values : list of object or pandas.DataFrame
            The feature values to edit the case(s) with. If specified as a list,
            the order corresponds with the order of the `features` parameter.
            If specified as a DataFrame, only the first row will be used.
        case_indices : Iterable of Sequence[str, int], optional
            An Iterable of Sequences containing the session id and index, where
            index is the original 0-based index of the case as it was trained
            into the session. This explicitly specifies the cases to edit. When
            specified, `condition` and `condition_session` are ignored.
        condition : dict or None, optional
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

        condition_session : str or BaseSession, optional
            If specified, ignores the condition and operates on all cases for
            the specified session id or BaseSession instance.
        features : list of str, optional
            The names of the features to edit. Required when `feature_values`
            is not specified as a DataFrame.
        num_cases : int, default None
            The maximum amount of cases to edit. If not specified, the limit
            will be k cases if precision is "similar", or no limit if precision
            is "exact".
        precision : str, default None
            The precision to use when removing the cases. Options are 'exact'
            or 'similar'. If not specified "exact" will be used.

        Returns
        -------
        int
            The number of cases modified.
        """
        if isinstance(condition_session, BaseSession):
            condition_session_id = condition_session.id
        else:
            condition_session_id = condition_session
        return self.client.edit_cases(
            trainee_id=self.id,
            case_indices=case_indices,
            condition=condition,
            condition_session=condition_session_id,
            features=features,
            feature_values=feature_values,
            num_cases=num_cases,
            precision=precision,
        )

    def get_sessions(self) -> List[Dict[str, str]]:
        """
        Get all session ids of the trainee.

        Returns
        -------
        list of dict
            A list of dicts with keys "id" and "name" for each session
            in the model.
        """
        return self.client.get_trainee_sessions(self.id)

    def delete_session(self, session: Union[str, BaseSession]) -> None:
        """
        Delete a session from the trainee.

        Parameters
        ----------
        session : str or BaseSession
            The id or instance of the session to remove from the model.

        Returns
        -------
        None
        """
        if isinstance(session, BaseSession):
            session_id = session.id
        else:
            session_id = session
        self.client.delete_trainee_session(trainee_id=self.id, session=session_id)

    def get_session_indices(self, session: Union[str, BaseSession]) -> "Index":
        """
        Get all session indices for a specified session.

        Parameters
        ----------
        session : str or BaseSession
            The id or instance of the session to retrieve indices for from
            the model.

        Returns
        -------
        pandas.Index
            An index of the session indices for the requested session.
        """
        if isinstance(session, BaseSession):
            session_id = session.id
        else:
            session_id = session
        return self.client.get_trainee_session_indices(
            trainee_id=self.id,
            session=session_id,
        )

    def get_session_training_indices(self, session: Union[str, BaseSession]) -> "Index":
        """
        Get all session training indices for a specified session.

        Parameters
        ----------
        session : str or BaseSession
            The id or instance of the session to retrieve training indices for
            from the model.

        Returns
        -------
        pandas.Index
            An index of the session training indices for the requested session.
        """
        if isinstance(session, BaseSession):
            session_id = session.id
        else:
            session_id = session
        return self.client.get_trainee_session_training_indices(
            trainee_id=self.id,
            session=session_id,
        )

    def get_cases(
        self,
        *,
        case_indices: Optional[CaseIndices] = None,
        features: Optional[Iterable[str]] = None,
        indicate_imputed: Optional[bool] = False,
        session: Optional[Union[str, BaseSession]] = None,
        condition: Optional[Dict] = None,
        num_cases: Optional[int] = None,
        precision: Optional[str] = None
    ) -> "DataFrame":
        """
        Get the trainee's cases.

        Parameters
        ----------
        case_indices : Iterable of Sequence[str, int], optional
            List of tuples, of session id and index, where index is the
            original 0-based index of the case as it was trained into the
            session. If specified, returns only these cases and ignores the
            session parameter.

            .. NOTE::
                If case_indices are provided, condition (and precision)
                are ignored.

        features : list of str, optional
            A list of feature names to return values for in leu of all
            default features.

            Built-in features that are available for retrieval:

                | **.session** - The session id the case was trained under.
                | **.session_training_index** - 0-based original index of the
                  case, ordered by training during the session; is never
                  changed.

        indicate_imputed : bool, default False
            If True, an additional value will be appended to the cases
            indicating if the case was imputed.

        session : str or BaseSession, optional
            The id or instance of the session to retrieve training indices for
            from the model.

            .. NOTE::
                If a session is not provided, the order of the cases is not
                guaranteed to be the same as the order they were trained into
                the model.

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

            .. NOTE::
                This option will be ignored if case_indices is supplied.

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
        precision : str, default None
            The precision to use when retrieving the cases via condition.
            Options are 'exact' or 'similar'. If not specified, "exact" will
            be used.

        Returns
        -------
        pandas.DataFrame
            The trainee's cases.
        """
        if isinstance(session, BaseSession):
            session_id = session.id
        else:
            session_id = session
        return self.client.get_cases(
            trainee_id=self.id,
            features=features,
            case_indices=case_indices,
            indicate_imputed=indicate_imputed,
            session=session_id,
            condition=condition,
            num_cases=num_cases,
            precision=precision,
        )

    def get_extreme_cases(
        self,
        *,
        features: Optional[Iterable[str]] = None,
        num: int,
        sort_feature: str,
    ) -> "DataFrame":
        """
        Get the trainee's extreme cases.

        Parameters
        ----------
        features : list of str, optional
            The features to include in the case data.
        num : int
            The number of cases to get.
        sort_feature
            The name of the feature by which extreme cases are sorted.

        Returns
        -------
        pandas.DataFrame
            The trainee's extreme cases.
        """
        return self.client.get_extreme_cases(
            trainee_id=self.id, features=features, num=num, sort_feature=sort_feature
        )

    def get_num_training_cases(self) -> int:
        """
        Return the number of trained cases for the trainee.

        Returns
        -------
        int
            The number of trained cases.
        """
        return self.client.get_num_training_cases(self.id)

    def add_feature(
        self,
        feature: str,
        feature_value: Optional[object] = None,
        *,
        condition: Optional[Dict[str, object]] = None,
        condition_session: Optional[Union[str, BaseSession]] = None,
        feature_attributes: Optional[Dict] = None,
        overwrite: Optional[bool] = False,
    ) -> None:
        """
        Add a feature to the model.

        Parameters
        ----------
        feature : str
            The name of the feature.
        feature_attributes : dict, optional
            The dict of feature specific attributes for this feature. If
            unspecified and conditions are not specified, will assume feature
            type as 'continuous'.
        feature_value : int or float or str, optional
            The value to populate the feature with.
            By default, populates the new feature with None.
        condition : dict, optional
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

        condition_session : str or BaseSession, optional
            If specified, ignores the condition and operates on cases for the
            specified session id or BaseSession instance.
        overwrite : bool, default False
            If True, the feature will be over-written if it exists.
        """
        if isinstance(condition_session, BaseSession):
            condition_session_id = condition_session.id
        else:
            condition_session_id = condition_session
        self.client.add_feature(
            trainee_id=self.id,
            condition=condition,
            condition_session=condition_session_id,
            feature=feature,
            feature_value=feature_value,
            feature_attributes=feature_attributes,
            overwrite=overwrite,
        )
        self._features = self.client.trainee_cache.get(self.id).features

    def remove_feature(
        self,
        feature: str,
        *,
        condition: Optional[Dict[str, object]] = None,
        condition_session: Optional[Union[str, BaseSession]] = None,
    ) -> None:
        """
        Remove a feature from the trainee.

        Parameters
        ----------
        feature : str
            The name of the feature to remove.
        condition : dict, default None
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

        condition_session : str or BaseSession, optional
            If specified, ignores the condition and operates on cases for the
            specified session id or BaseSession instance.
        """
        if isinstance(condition_session, BaseSession):
            condition_session_id = condition_session.id
        else:
            condition_session_id = condition_session
        self.client.remove_feature(
            trainee_id=self.id,
            condition=condition,
            condition_session=condition_session_id,
            feature=feature,
        )
        self._features = self.client.trainee_cache.get(self.id).features

    def remove_series_store(self, series: Optional[str] = None) -> None:
        """
        Clear stored series from trainee.

        Parameters
        ----------
        series : str, optional
            Series id to clear. If not provided, clears the entire
            series store for the trainee.

        Returns
        -------
        None
        """
        self.client.remove_series_store(trainee_id=self.id, series=series)

    def append_to_series_store(
        self,
        series: str,
        contexts: Union[List[List[object]], "DataFrame"],
        *,
        context_features: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Append the specified contexts to a series store.

        For use with train series.

        Parameters
        ----------
        series : str
            The name of the series store to append to.
        contexts : list of list of object or pandas.DataFrame
            The list of context values to append to the series.
        context_features : list of str, optional
            The list of feature names for contexts.

        Returns
        -------
        None
        """
        self.client.append_to_series_store(
            trainee_id=self.id,
            series=series,
            contexts=contexts,
            context_features=context_features,
        )

    def set_substitute_feature_values(
        self, substitution_value_map: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Set a substitution map for use in extended nominal generation.

        Parameters
        ----------
        substitution_value_map : dict
            A dictionary of feature name to a dictionary of feature value to
            substitute feature value.

            If this dict is None, all substitutions will be disabled and
            cleared. If any feature in the `substitution_value_map` has
            features mapping to `None` or `{}`, substitution values will
            immediately be generated.

        Returns
        -------
        None
        """
        self.client.set_substitute_feature_values(
            trainee_id=self.id, substitution_value_map=substitution_value_map
        )

    def get_substitute_feature_values(
        self, *, clear_on_get: Optional[bool] = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get a substitution map for use in extended nominal generation.

        Parameters
        ----------
        clear_on_get : bool, default True
            Clears the substitution values map in the trainee upon retrieving
            them. This is done if it is desired to prevent the substitution
            map from being persisted. If set to False, the model will not be
            cleared which preserves substitution mappings if the model is
            saved; representing a potential privacy leak should the
            substitution map be made public.

        Returns
        -------
        dict of {str: dict}
            A dictionary of feature name to a dictionary of feature value to
            substitute feature value.
        """
        return self.client.get_substitute_feature_values(
            trainee_id=self.id, clear_on_get=clear_on_get
        )

    def react_group(
        self,
        *,
        distance_contributions: Optional[bool] = False,
        familiarity_conviction_addition: Optional[bool] = True,
        familiarity_conviction_removal: Optional[bool] = False,
        features: Optional[Iterable[str]] = None,
        new_cases: Optional[
            Union[List["DataFrame"], List[List[List[object]]]]] = None,
        kl_divergence_addition: Optional[bool] = False,
        kl_divergence_removal: Optional[bool] = False,
        p_value_of_addition: Optional[bool] = False,
        p_value_of_removal: Optional[bool] = False,
        trainees_to_compare: Optional[Iterable[Union["Trainee", str]]] = None,
        use_case_weights: Optional[bool] = False,
        weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        Computes specified data for a **set** of cases.

        Return the list of familiarity convictions (and optionally, distance
        contributions or p values) for each set.

        Parameters
        ----------
        distance_contributions : bool, default False
            (Optional) Calculate and output distance contribution ratios in
            the output dict for each case.
        familiarity_conviction_addition : bool, default True
            (Optional) Calculate and output familiarity conviction of adding the
            specified cases.
        familiarity_conviction_removal : bool, default False
            (Optional) Calculate and output familiarity conviction of removing
            the specified cases.
        features : list of str or None, optional
            A list of feature names to consider while calculating convictions.
        kl_divergence_addition : bool, default False
            (Optional) Calculate and output KL divergence of adding the
            specified cases.
        kl_divergence_removal : bool, default False
            (Optional) Calculate and output KL divergence of removing the
            specified cases.
        new_cases : list of list of list of object or list of pandas.DataFrame, optional
            Specify a **set** using a list of cases to compute the conviction
            of groups of cases as shown in the following example.

            Example::

                new_cases = [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], # Group 1
                    [[1, 2, 3]], # Group 2
                ]

        p_value_of_addition : bool, default False
            (Optional) If true will output p value of addition.
        p_value_of_removal : bool, default False
            (Optional) If true will output p value of removal.
        trainees_to_compare : list of (str or Trainee), optional
            (Optional) If specified ignores the 'new_cases' parameter and uses
            cases from the specified trainee(s) instead. Values should be either
            the trainee object or its ID (trainee name is not supported).
        use_case_weights : bool, default False
            When True, will scale influence weights by each case's
            `weight_feature` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        pandas.DataFrame
            The conviction of grouped cases.
        """
        if trainees_to_compare:
            serialized_trainees_to_compare = [
                t.id if isinstance(t, BaseTrainee) else t
                for t in trainees_to_compare
            ]
        else:
            serialized_trainees_to_compare = None

        return self.client.react_group(
            trainee_id=self.id,
            new_cases=new_cases,
            features=features,
            trainees_to_compare=serialized_trainees_to_compare,
            familiarity_conviction_addition=familiarity_conviction_addition,
            familiarity_conviction_removal=familiarity_conviction_removal,
            kl_divergence_addition=kl_divergence_addition,
            kl_divergence_removal=kl_divergence_removal,
            p_value_of_addition=p_value_of_addition,
            p_value_of_removal=p_value_of_removal,
            distance_contributions=distance_contributions,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )

    def get_feature_conviction(
        self,
        *,
        action_features: Optional[Iterable[str]] = None,
        familiarity_conviction_addition: Optional[bool] = True,
        familiarity_conviction_removal: Optional[bool] = False,
        features: Optional[Iterable[str]] = None,
        use_case_weights: Optional[bool] = False,
        weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        Get familiarity conviction for features in the model.

        Parameters
        ----------
        action_features : list of str, optional
            The feature names to be treated as action features during
            conviction calculation in order to determine the conviction
            of each feature against the set of action_features. If not
            specified, conviction is computed for each feature against the
            rest of the features as a whole.
        familiarity_conviction_addition : bool, default True
            (Optional) Calculate and output familiarity conviction of adding the
            specified cases.
        familiarity_conviction_removal : bool, default False
            (Optional) Calculate and output familiarity conviction of removing
            the specified cases.
        features : list of str, optional
            The feature names to calculate convictions for. At least 2 features
            are required to get familiarity conviction. If not specified all
            features will be used.
        use_case_weights : bool, default False
            When True, will scale influence weights by each case's
            `weight_feature` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the familiarity conviction rows to feature
            columns.
        """
        return self.client.get_feature_conviction(
            trainee_id=self.id,
            action_features=action_features,
            familiarity_conviction_addition=familiarity_conviction_addition,
            familiarity_conviction_removal=familiarity_conviction_removal,
            features=features,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )

    def get_feature_residuals(
        self,
        *,
        action_feature: Optional[str] = None,
        robust: Optional[bool] = None,
        robust_hyperparameters: Optional[bool] = None,
        weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        Get cached feature residuals.

        All keyword arguments are optional, when not specified will auto-select
        cached residuals for output, when specified will attempt to
        output the cached residuals best matching the requested parameters,
        or None if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`Trainee.get_prediction_stats` instead.

        Parameters
        ----------
        action_feature : str, optional
            When specified, will attempt to return residuals that
            were computed for this specified action_feature.
            Note: ".targetless" is the action feature used during targetless
            analysis.
        robust : bool, optional
            When specified, will attempt to return residuals that
            were computed with the specified robust or non-robust type.
        robust_hyperparameters : bool, optional
            When specified, will attempt to return residuals that were
            computed using hyperpparameters with the specified robust or
            non-robust type.
        weight_feature : str, optional
            When specified, will attempt to return residuals that
            were computed using this weight_feature.

        Returns
        -------
        pandas.DataFrame
            The feature residuals.
        """
        return self.client.get_feature_residuals(
            trainee_id=self.id,
            action_feature=action_feature,
            robust=robust,
            robust_hyperparameters=robust_hyperparameters,
            weight_feature=weight_feature,
        )

    def get_prediction_stats(
        self,
        *,
        action_feature: Optional[str] = None,
        condition: Optional[Dict[str, Any]] = None,
        num_cases: Optional[int] = None,
        num_robust_influence_samples_per_case: Optional[int] = None,
        precision: Optional[str] = None,
        robust: Optional[bool] = None,
        robust_hyperparameters: Optional[bool] = None,
        stats: Optional[Iterable[str]] = None,
        weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        Get feature prediction stats.

        Gets cached stats when condition is None.
        If condition is not None, then uses the condition to select cases and
        computes prediction stats for that set of cases.

        All keyword arguments are optional, when not specified will auto-select
        all cached stats for output, when specified will attempt to
        output the cached stats best matching the requested parameters,
        or None if no cached match is found.

        Parameters
        ----------
        action_feature : str, optional
            When specified, will attempt to return stats that
            were computed for this specified action_feature.
            Note: ".targetless" is the action feature used during targetless
            analysis.

            .. NOTE::
                If get_prediction_stats is being used with time series analysis,
                the action feature for which the prediction statistics information
                is desired must be specified.
        condition : dict or None, optional
            A condition map to select which cases to compute prediction stats
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
            The maximum amount of cases to use to calculate prediction stats.
            If not specified, the limit will be k cases if precision is
            "similar", or 1000 cases if precision is "exact". Only used if
            `condition` is not None.
        num_robust_influence_samples_per_case : int, optional
            Specifies the number of robust samples to use for each case for
            robust contribution computations.
            Defaults to 300 + 2 * (number of features).
        precision : str, default None
            The precision to use when selecting cases with the condition.
            Options are 'exact' or 'similar'. If not specified "exact" will be
            used. Only used if `condition` is not None.
        robust : bool, optional
            When specified, will attempt to return stats that
            were computed with the specified robust or non-robust type.
        robust_hyperparameters : bool, optional
            When specified, will attempt to return stats that were
            computed using hyperparameters with the specified robust or
            non-robust type.
        stats : list of str, optional
            List of stats to output. When unspecified, returns all.
            Allowed values:

                - accuracy : The number of correct predictions divided by the
                  total number of predictions.
                - confusion_matrix : A map of actual feature value to a map of
                  predicted feature value to counts.
                - contribution : Feature contributions to predicted value when
                  each feature is dropped from the model, applies to all
                  features.
                - mae : Mean absolute error. For continuous features, this is
                  calculated as the mean of absolute values of the difference
                  between the actual and predicted values. For nominal features,
                  this is 1 - the average categorical action probability of each case's
                  correct classes. Categorical action probabilities are the probabilities
                  for each class for the action feature.
                - mda : Mean decrease in accuracy when each feature is dropped
                  from the model, applies to all features.
                - mda_permutation : Mean decrease in accuracy that used
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

        weight_feature : str, optional
            When specified, will attempt to return stats that
            were computed using this weight_feature.

        Returns
        -------
        DataFrame
            A DataFrame of feature name columns to stat value rows. Indexed
            by the stat type.
        """
        return self.client.get_prediction_stats(
            trainee_id=self.id,
            action_feature=action_feature,
            robust=robust,
            robust_hyperparameters=robust_hyperparameters,
            stats=stats,
            weight_feature=weight_feature,
            condition=condition,
            precision=precision,
            num_cases=num_cases,
            num_robust_influence_samples_per_case=num_robust_influence_samples_per_case,
        )

    def get_marginal_stats(
        self, *, weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        Get marginal stats for all features.

        Parameters
        ----------
        weight_feature : str, optional
            When specified, will attempt to return stats that were computed
            using this weight_feature.

        Returns
        -------
        DataFrame
            A DataFrame of feature name columns to stat value rows. Indexed
            by the stat type.
        """
        return self.client.get_marginal_stats(
            trainee_id=self.id,
            weight_feature=weight_feature
        )

    def react_into_features(
        self,
        *,
        distance_contribution: Optional[Union[str, bool]] = False,
        familiarity_conviction_addition: Optional[Union[str, bool]] = False,
        familiarity_conviction_removal: Optional[Union[str, bool]] = False,
        features: Optional[Iterable[str]] = None,
        p_value_of_addition: Optional[Union[str, bool]] = False,
        p_value_of_removal: Optional[Union[str, bool]] = False,
        similarity_conviction: Optional[Union[str, bool]] = False,
        use_case_weights: Optional[bool] = False,
        weight_feature: Optional[str] = None,
    ) -> None:
        """
        Calculate conviction and other data and stores them into features.

        Parameters
        ----------
        distance_contribution : bool or str, default False
            The name of the feature to store distance contribution.
            If set to True the values will be stored to the feature
            'distance_contribution'.
        familiarity_conviction_addition : bool or str, default False
            (Optional) The name of the feature to store conviction of addition
            values. If set to True the values will be stored to the feature
            'familiarity_conviction_addition'.
        familiarity_conviction_removal : bool or str, default False
            (Optional) The name of the feature to store conviction of removal
            values. If set to True the values will be stored to the feature
            'familiarity_conviction_removal'.
        features : list of str, optional
            A list of features to calculate convictions.
        p_value_of_addition : bool or str, default False
            (Optional) The name of the feature to store p value of addition
            values. If set to True the values will be stored to the feature
            'p_value_of_addition'.
        p_value_of_removal : bool or str, default False
            (Optional) The name of the feature to store p value of removal
            values. If set to True the values will be stored to the feature
            'p_value_of_removal'.
        similarity_conviction : bool or str, default False
            (Optional) The name of the feature to store similarity conviction
            values. If set to True the values will be stored to the feature
            'similarity_conviction'.
        use_case_weights : bool, default False
            When True, will scale influence weights by each case's
            `weight_feature` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        None
        """
        self.client.react_into_features(
            trainee_id=self.id,
            distance_contribution=distance_contribution,
            familiarity_conviction_addition=familiarity_conviction_addition,
            familiarity_conviction_removal=familiarity_conviction_removal,
            p_value_of_addition=p_value_of_addition,
            p_value_of_removal=p_value_of_removal,
            similarity_conviction=similarity_conviction,
            features=features,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )

    def react_into_trainee(
        self,
        *,
        action_feature: Optional[str] = None,
        context_features: Optional[Iterable[str]] = None,
        contributions: Optional[bool] = None,
        contributions_robust: Optional[bool] = None,
        hyperparameter_param_path: Optional[Iterable[str]] = None,
        mda: Optional[bool] = None,
        mda_permutation: Optional[bool] = None,
        mda_robust: Optional[bool] = None,
        mda_robust_permutation: Optional[bool] = None,
        num_robust_influence_samples=None,
        num_robust_residual_samples=None,
        num_robust_influence_samples_per_case=None,
        num_samples: Optional[int] = None,
        residuals: Optional[bool] = None,
        residuals_robust: Optional[bool] = None,
        sample_model_fraction: Optional[float] = None,
        sub_model_size: Optional[int] = None,
        use_case_weights: Optional[bool] = False,
        weight_feature: Optional[str] = None,
    ) -> None:
        """
        Compute and cache specified feature interpretations.

        Parameters
        ----------
        action_feature : str, optional
            Name of target feature whose hyperparameters to use
            for computations.  Default is whatever the model was analyzed for,
            or the mda_action_features for MDA, or ".targetless" if analyzed
            for targetless.
        context_features : list of str, optional
            List of features names to use as contexts for
            computations. Default is all trained non-unique features if
            unspecified.
        contributions : bool, optional
            For each context_feature, use the full set of all other
            context_features to compute the mean absolute delta between
            prediction of action_feature with and without the context_feature
            in the model. False removes cached values.
        contributions_robust : bool, optional
            For each context_feature, use the robust (power set/permutation)
            set of all other context_features to compute the mean absolute
            delta between prediction of action_feature with and without the
            context_feature in the model. False removes cached values.
        hyperparameter_param_path : list of str, optional.
            Full path for hyperparameters to use for computation. If specified
            for any residual computations, takes precendence over action_feature
            parameter.  Can be set to a 'paramPath' value from the results of
            'get_params()' for a specific set of hyperparameters.
        mda : bool, optional
            When True will compute Mean Decrease in Accuracy (MDA)
            for each context feature at predicting mda_action_features. Drop
            each feature and use the full set of remaining context features
            for each prediction. False removes cached values.
        mda_permutation : bool, optional
            Compute MDA by scrambling each feature and using the
            full set of remaining context features for each prediction.
            False removes cached values.
        mda_robust : bool, optional
            Compute MDA by dropping each feature and using the
            robust (power set/permutations) set of remaining context features
            for each prediction. False removes cached values.
        mda_robust_permutation : bool, optional
            Compute MDA by scrambling each feature and using the
            robust (power set/permutations) set of remaining context features
            for each prediction. False removes cached values.
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
        residuals : bool, optional
            For each context_feature, use the full
            set of all other context_features to predict the feature.
            False removes cached values.
        residuals_robust : bool, optional
            For each context_feature, use the robust (power
            set/permutations) set of all other context_features to predict the
            feature.  False removes cached values.
        sample_model_fraction : float, optional
            A value between 0.0 - 1.0, percent of model to use in sampling
            (using sampling without replacement). Applicable only to non-robust
            computation. Ignored if num_samples is specified.
            Higher values provide better accuracy at the cost of compute time.
        sub_model_size : int, optional
            Subset of model to use for calculations. Applicable only
            to models > 1000 cases.
        use_case_weights : bool, default False
            If set to True will scale influence weights by each case's
            weight_feature weight.
        weight_feature : str, optional
            The name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        None
        """
        self.client.react_into_trainee(
            trainee_id=self.id,
            action_feature=action_feature,
            context_features=context_features,
            contributions=contributions,
            contributions_robust=contributions_robust,
            hyperparameter_param_path=hyperparameter_param_path,
            mda=mda,
            mda_permutation=mda_permutation,
            mda_robust=mda_robust,
            mda_robust_permutation=mda_robust_permutation,
            num_robust_influence_samples=num_robust_influence_samples,
            num_robust_residual_samples=num_robust_residual_samples,
            num_robust_influence_samples_per_case=num_robust_influence_samples_per_case,
            num_samples=num_samples,
            residuals=residuals,
            residuals_robust=residuals_robust,
            sample_model_fraction=sample_model_fraction,
            sub_model_size=sub_model_size,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )

    def get_feature_mda(
        self,
        action_feature: str,
        *,
        permutation: Optional[bool] = None,
        robust: Optional[bool] = None,
        weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        Get cached feature Mean Decrease In Accuracy (MDA).

        All keyword arguments are optional, when not specified will auto-select
        cached MDA for output, when specified will attempt to
        output the cached MDA best matching the requested parameters,
        or None if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`Trainee.get_prediction_stats` instead.

        Parameters
        ----------
        action_feature : str
            Will attempt to return MDA that was
            computed for this specified action_feature.
        permutation : bool, optional
            When False, will attempt to return MDA that was computed
            by dropping each feature. When True will attempt to return MDA that
            was computed with permutations by scrambling each feature.
        robust : bool, optional
            When specified, will attempt to return MDA that was
            computed with the specified robust or non-robust type.
        weight_feature : str, optional
            When specified, will attempt to return MDA that was
            computed using this weight_feature.

        Returns
        -------
        pandas.DataFrame
            The mean decrease in accuracy values for context features.
        """
        return self.client.get_feature_mda(
            trainee_id=self.id,
            action_feature=action_feature,
            permutation=permutation,
            robust=robust,
            weight_feature=weight_feature,
        )

    def get_feature_contributions(
        self,
        action_feature: str,
        *,
        robust: Optional[bool] = None,
        directional: bool = False,
        weight_feature: Optional[str] = None,
    ) -> "DataFrame":
        """
        Get cached feature contributions.

        All keyword arguments are optional, when not specified will auto-select
        cached contributions for output, when specified will attempt to
        output the cached contributions best matching the requested parameters,
        or None if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`Trainee.get_prediction_stats` instead.

        Parameters
        ----------
        action_feature : str
            Will attempt to return contributions that were
            computed for this specified action_feature.
        robust : bool, optional
            When specified, will attempt to return contributions that were
            computed with the specified robust or non-robust type.
        directional : bool, default False
            If false returns absolute feature contributions. If true, returns
            directional feature contributions.
        weight_feature : str, optional
            When specified, will attempt to return contributions that were
            computed using this weight_feature.

        Returns
        -------
        pandas.DataFrame
            The contribution values for context features.
        """
        return self.client.get_feature_contributions(
            trainee_id=self.id,
            action_feature=action_feature,
            robust=robust,
            directional=directional,
            weight_feature=weight_feature,
        )

    def get_params(
        self,
        *,
        action_feature: Optional[str] = None,
        context_features: Optional[Iterable[str]] = None,
        mode: Optional[Literal["robust", "full"]] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get the parameters used by the Trainee. If 'action_feature',
        'context_features', 'mode', or 'weight_feature' are specified, then
        the best hyperparameters analyzed in the Trainee are the value of the
        'hyperparameter_map' key, otherwise this value will be the dictionary
        containing all the hyperparameter sets in the Trainee.


        Parameters
        ----------
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
            that were analyzed using this weight feaure.

        Returns
        -------
        dict
            A dict including the either all of the Trainee's internal
            parameters or only the best hyperparameters selected using the
            passed parameters.
        """
        return self.client.get_params(
            self.id,
            action_feature=action_feature,
            context_features=context_features,
            mode=mode,
            weight_feature=weight_feature,
        )

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the workflow attributes for the trainee.

        Parameters
        ----------
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
                    "auto_analyze_enabled": False,
                    "analyze_threshold": 100,
                    "analyze_growth_factor": 7.389,
                    "auto_analyze_limit_size": 100000
                }
        """
        self.client.set_params(self.id, params=params)

    @property
    def client(self) -> Union[AbstractHowsoClient, HowsoPandasClientMixin]:
        """
        The client instance used by the trainee.

        Returns
        -------
        AbstractHowsoClient
            The client instance.
        """
        return self._client

    @client.setter
    def client(self, client: AbstractHowsoClient) -> None:
        """
        Set the client instance used by the trainee.

        Parameters
        ----------
        client : AbstractHowsoClient
            The client instance. Must be a subclass of AbstractHowsoClient
            and HowsoPandasClientMixin.

        Returns
        -------
        None
        """
        if not isinstance(client, AbstractHowsoClient):
            raise HowsoError(
                "`client` must be a subclass of AbstractHowsoClient"
            )
        if not isinstance(client, HowsoPandasClientMixin):
            raise HowsoError("`client` must be a HowsoPandasClient")
        self._client = client

    def _update_attributes(self, trainee: BaseTrainee) -> None:
        """
        Update the protected attributes of the trainee.

        Parameters
        ----------
        trainee : BaseTrainee
            The base trainee instance.

        Returns
        -------
        None
        """
        for key in self.attribute_map.keys():
            # Update the protected attributes directly since the values
            # have already been validated by the "BaseTrainee" instance
            # and to prevent triggering an API update call
            setattr(self, f"_{key}", getattr(trainee, key))

    def update(self) -> None:
        """
        Update the remote trainee with local state.

        Returns
        -------
        None
        """
        if (
            getattr(self, "id", None)
            and getattr(self, "_created", False)
            and not getattr(self, "_updating", False)
        ):
            # Only update for trainees that have been created
            try:
                self._updating = True
                trainee = BaseTrainee(**self.to_dict())
                updated_trainee = self.client.update_trainee(trainee)
                self._update_attributes(updated_trainee)
            finally:
                self._updating = False

    def get_pairwise_distances(
        self,
        features: Optional[Dict[str, Dict]] = None,
        *,
        action_feature: Optional[str] = None,
        from_case_indices: Optional[CaseIndices] = None,
        from_values: Optional[Union[List[List[object]], "DataFrame"]] = None,
        to_case_indices: Optional[CaseIndices] = None,
        to_values: Optional[Union[List[List[object]], "DataFrame"]] = None,
        use_case_weights: Optional[bool] = False,
        weight_feature: Optional[str] = None,
    ) -> List[float]:
        """
        Computes pairwise distances between specified cases.

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
        features : list of str, optional
            List of feature names to use when computing pairwise distances.
            If unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this `action_feature`, otherwise uses targetless
            hyperparameters.
        from_case_indices : Iterable of Sequence[str, int], optional
            An Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of `to_values` or `to_case_indices`.
        from_values : list of list of object or pandas.DataFrame, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of `to_values` or `to_case_indices`.
        to_case_indices : Iterable of Sequence[str, int], optional
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
        return self.client.get_pairwise_distances(
            self.id,
            features=features,
            action_feature=action_feature,
            from_case_indices=from_case_indices,
            from_values=from_values,
            to_case_indices=to_case_indices,
            to_values=to_values,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature
        )

    def get_distances(
        self,
        features: Optional[Dict[str, Dict]] = None,
        *,
        action_feature: Optional[str] = None,
        case_indices: Optional[CaseIndices] = None,
        feature_values: Optional[Union[List[object], "DataFrame"]] = None,
        use_case_weights: Optional[bool] = False,
        weight_feature: Optional[str] = None
    ) -> "Distances":
        """
        Computes distances matrix for specified cases.

        Returns a dict with computed distances between all cases
        specified in `case_indices` or from all cases in local model as defined
        by `feature_values`.

        Parameters
        ----------
        features : list of str, optional
            List of feature names to use when computing distances. If
            unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this `action_feature`, otherwise uses targetless
            hyperparameters.
        case_indices : Iterable of Sequence[str, int], optional
            List of tuples, of session id and index, where index is the
            original 0-based index of the case as it was trained into the
            session. If specified, returns distances for all of these
            cases. Ignored if `feature_values` is provided. If neither
            `feature_values` nor `case_indices` is specified, uses full dataset.
        feature_values : list of object or pandas.DataFrame, optional
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
                    'distances': DataFrame( distances )
                }
        """
        return self.client.get_distances(
            self.id,
            features=features,
            action_feature=action_feature,
            case_indices=case_indices,
            feature_values=feature_values,
            weight_feature=weight_feature,
            use_case_weights=use_case_weights
        )

    def evaluate(
        self,
        features_to_code_map,
        *,
        aggregation_code=None,
    ) -> dict:
        r"""
        Evaluates custom code on feature values of all cases in the trainee.

        Parameters
        ----------
        features_to_code_map : dict[str, str]
            A dictionary with feature name keys and custom Amalgam code string
            values.

            The custom code can use \"#feature_name 0\" to reference the value
            of that feature for each case.
        aggregation_code : str, optional
            A string of custom Amalgam code that can access the list of values
            derived form the custom code in features_to_code_map.

            The custom code can use \"#feature_name 0\" to reference the list of
            values derived from using the custom code in features_to_code_map.

        Returns
        -------
        dict
            A dictionary with keys: 'evaluated' and 'aggregated'.

            'evaluated' is a dictionary with feature name
            keys and lists of values derived from the features_to_code_map
            custom code.

            'aggregated' is None if no aggregation_code is given, it otherwise
            holds the output of the custom 'aggregation_code'
        """
        return self.client.evaluate(
            self.id,
            features_to_code_map=features_to_code_map,
            aggregation_code=aggregation_code,
        )

    def _create(
        self, *,
        library_type: Optional[str] = None,
        max_wait_time: Optional[Union[int, float]] = None,
        overwrite: Optional[bool] = False,
        resources: Optional[Union["TraineeResources", Dict[str, Any]]] = None,
    ) -> None:
        """
        Create the trainee at the API.

        Parameters
        ----------
        library_type : str, optional
            The library type of the Trainee.
        max_wait_time : int or float, optional
            The maximum time to wait for the trainee to be created.
        overwrite : bool, optional
            If True, will overwrite an existing trainee with the same name.
        resources : TraineeResources or dict, optional
            The resources to provision for the trainee.

        Returns
        -------
        None
        """
        if not self.id:
            trainee = BaseTrainee(**self.to_dict())
            new_trainee = self.client.create_trainee(
                trainee=trainee,
                library_type=library_type,
                max_wait_time=max_wait_time,
                overwrite_trainee=overwrite,
                resources=resources,
            )
            self._update_attributes(new_trainee)
        self._created = True

    @classmethod
    def from_openapi(
        cls, trainee: BaseTrainee, *, client: Optional[AbstractHowsoClient] = None
    ) -> "Trainee":
        """
        Create Trainee from base class.

        Parameters
        ----------
        trainee : BaseTrainee
            The base trainee instance.
        client : AbstractHowsoClient, optional
            The Howso client instance to use.

        Returns
        -------
        Trainee
            The trainee instance.
        """
        trainee_dict = trainee.to_dict()
        trainee_dict["client"] = client
        return cls.from_dict(trainee_dict)

    @classmethod
    def from_dict(cls, trainee_dict: dict) -> "Trainee":
        """
        Create Trainee from dict.

        Parameters
        ----------
        trainee_dict : Dict
            The Trainee parameters.

        Returns
        -------
        Trainee
            The trainee instance.
        """
        if not isinstance(trainee_dict, dict):
            raise ValueError("`trainee_dict` parameter is not a dict")
        parameters = {"client": trainee_dict.get("client")}
        for key in cls.attribute_map.keys():
            if key in trainee_dict:
                if key == "project_id":
                    parameters["project"] = trainee_dict[key]
                else:
                    parameters[key] = trainee_dict[key]
        return cls(**parameters)

    def __enter__(self) -> "Trainee":
        """Support context managers."""
        self.acquire_resources()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup trainee during exit of context manager."""
        try:
            if self.persistence != "never" and (self.name or self._was_saved):
                self.release_resources()
            else:
                self.delete()
        except HowsoApiError as error:
            if error.status != 404:
                raise


def delete_trainee(
    name_or_id: str, *, client: Optional[AbstractHowsoClient] = None
) -> None:
    """
    Delete an existing trainee.

    Parameters
    ----------
    name_or_id : str
        The name or id of the trainee.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.

    Returns
    -------
    None
    """
    client = client or get_client()
    client.delete_trainee(str(name_or_id))


def load_trainee(
    file_path: Union[Path, str],
    client: Optional[HowsoDirectClient] = None
) -> "Trainee":
    """
    Load an existing trainee from disk.

    Parameters
    ----------
    file_path : Path or str
        The path of the file to load the Trainee from. This path can contain
        an absolute path, a relative path or simply a file name. A `.caml` file name
        must be always be provided if file paths are provided.

        If `file_path` is a relative path the absolute path will be computed
        appending the `file_path` to the CWD.

        If `file_path` is an absolute path, this is the absolute path that
        will be used.

        If `file_path` is just a filename, then the absolute path will be computed
        appending the filename to the CWD.

    client : HowsoDirectClient, optional
        The Howso client instance to use.

    Returns
    -------
    Trainee
        The trainee instance.
    """
    client = client or get_client()

    if not isinstance(client, HowsoDirectClient):
        raise HowsoError("To save, `client` must be HowsoDirectClient.")

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    trainee_id = str(uuid.uuid4())

    file_path = file_path.expanduser().resolve()

    # It is decided that if the file contains a suffix then it contains a
    # file name.
    if file_path.suffix:
        # Check to make sure sure `.caml` file is provided
        if file_path.suffix.lower() != '.caml':
            raise HowsoError(
                'Filepath with a non `.caml` extension was provided.'
            )
    else:
        # Add the extension to the file_path
        raise HowsoError(
            'A `.caml` file must be provided.'
        )
    # If path is not absolute, append it to the default directory.
    if not file_path.is_absolute():
        file_path = client.howso.default_save_path.joinpath(file_path)

    # Ensure the parent path exists.
    if not file_path.parents[0].exists(): # Noqa
        raise HowsoError(
            f'The specified directory "{file_path.parents[0]}" does not exist.')

    ret = client.howso.load(trainee_id, file_path.stem, f"{file_path.parents[0]}/")

    if ret is None:
        raise HowsoError(f"Trainee from file '{file_path}' not found.")

    trainee = client._get_trainee_from_core(trainee_id)
    client.trainee_cache.set(trainee, entity_id=client.howso.handle)

    return Trainee.from_openapi(trainee, client=client)


def get_trainee(
    name_or_id: str,
    *,
    client: Optional[AbstractHowsoClient] = None
) -> "Trainee":
    """
    Get an existing trainee from Howso Services.

    Parameters
    ----------
    name_or_id : str
        The name or id of the trainee.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.

    Returns
    -------
    Trainee
        The trainee instance.
    """
    client = client or get_client()
    trainee = client.get_trainee(str(name_or_id))
    return Trainee.from_openapi(trainee, client=client)


def list_trainees(
    search_terms: Optional[str] = None,
    *,
    client: Optional[AbstractHowsoClient] = None,
    project: Optional[Union[str, BaseProject]] = None,
) -> List["TraineeIdentity"]:
    """
    Get listing of available trainees.

    This method only returns a simplified informational listing of available
    trainees, not full engine Trainee instances. To get a Trainee instance
    that can be used with the engine API call `get_trainee`.

    Parameters
    ----------
    search_terms : str
        Terms to filter results by.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.
    project : str or Project, optional
        The instance or id of a project to filter by.

    Returns
    -------
    list of TraineeIdentity
        The list of available trainees.
    """
    client = client or get_client()

    params = {'search_terms': search_terms}

    # Only pass project_id for platform clients
    if project is not None and isinstance(client, ProjectClient):
        if isinstance(project, BaseProject):
            params["project_id"] = project.id
        else:
            params["project_id"] = project

    return client.get_trainees(**params)
