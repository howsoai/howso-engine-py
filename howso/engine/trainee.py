from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)
import uuid
import warnings

from pandas import (
    concat,
    DataFrame,
    Index,
)

from howso.client import AbstractHowsoClient
from howso.client.cache import TraineeCache
from howso.client.exceptions import (
    HowsoApiError,
    HowsoError,
    HowsoWarning,
)
from howso.client.pandas import HowsoPandasClientMixin
from howso.client.protocols import (
    LocalSaveableProtocol,
    ProjectClient,
)
from howso.engine.client import get_client
from howso.engine.project import Project
from howso.engine.session import Session
from howso.openapi.models import (
    Cases,
    Metrics,
)
from howso.openapi.models import Project as BaseProject
from howso.openapi.models import Session as BaseSession
from howso.openapi.models import Trainee as BaseTrainee
from howso.openapi.models import (
    TraineeIdentity,
    TraineeInformation,
    TraineeResources,
)
from howso.utilities import matrix_processing
from howso.utilities.feature_attributes.base import SingleTableFeatureAttributes
from howso.utilities.reaction import Reaction

from .typing import (
    CaseIndices,
    GenerateNewCases,
    Library,
    Mode,
    NewCaseThreshold,
    NormalizeMethod,
    PathLike,
    Persistence,
    Precision,
    SeriesIDTracking,
    TabularData2D,
    TabularData3D,
    TargetedModel,
)

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
    features : SingleTableFeatureAttributes, optional
        The feature attributes of the trainee. Where feature ``name`` is the key
        and a sub dictionary of feature attributes is the value. If this is not
        specified in the constructor, it must be set during or before :meth:`train`.
    default_action_features : list of str, optional
        The default action feature names of the trainee.
    default_context_features : list of str, optional
        The default context feature names of the trainee.
    id : str, optional
        The unique identifier of the Trainee. The client automatically completes
        this field and the user should NOT manually use this parameter. Please use
        the ``name`` parameter to manually specify a Trainee name.
    library_type : {"st", "mt"}, optional
        The library type of the Trainee. "st" will use the single-threaded library,
        while "mt" will use the multi-threaded library.
    max_wait_time : int or float, default 30
        The number of seconds to wait for a trainee to be created and become
        available before aborting gracefully. Set to ``0`` (or None) to wait as
        long as the system-configured maximum for sufficient resources to
        become available, which is typically 20 minutes.
    persistence : {"allow", "always", "never"}, default "allow"
        The requested persistence state of the trainee.
    project : str or Project, optional
        The instance or id of the project to use for the trainee.
    metadata : dict, optional
        Any key-value pair to store as custom metadata for the trainee.
    resources : TraineeResources or map, optional
        Customize the resources provisioned for the Trainee instance.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.
    overwrite_existing : bool, default False
        Overwrite existing trainee with the same name (if exists).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        features: Optional[SingleTableFeatureAttributes] = None,
        *,
        overwrite_existing: bool = False,
        persistence: Persistence = "allow",
        default_action_features: Optional[Iterable[str]] = None,
        default_context_features: Optional[Iterable[str]] = None,
        id: Optional[str] = None,
        library_type: Optional[Library] = None,
        max_wait_time: Optional[Union[int, float]] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
        project: Optional[Union[str, BaseProject]] = None,
        resources: Optional[Union["TraineeResources", MutableMapping[str, Any]]] = None,
        client: Optional[AbstractHowsoClient] = None,
    ):
        self._created: bool = False
        self._updating: bool = False
        self._was_saved: bool = False
        self.client = client or get_client()

        # Set the trainee properties
        self._features = features
        self._metadata = metadata
        self.name = name
        self._id = id
        self._custom_save_path = None
        self._calculated_matrices = {}
        self._needs_analyze: bool = False

        self.persistence = persistence
        self.set_default_features(
            action_features=default_action_features,
            context_features=default_context_features,
        )

        # Allow passing project id or the project instance
        if isinstance(project, BaseProject):
            self._project_id = project.id
            if isinstance(self.client, ProjectClient):
                self._project_instance = Project.from_openapi(
                    project, client=self.client)  # type:ignore
            else:
                self._project_instance = None
        else:
            self._project_id = project
            self._project_instance = None  # lazy loaded

        # Create the trainee at the API
        self._create(
            library_type=library_type,
            max_wait_time=max_wait_time,
            overwrite=overwrite_existing,
            resources=resources
        )

    @property
    def id(self) -> str | None:
        """
        The unique identifier of the trainee.

        If a identifier is not provided and a name is provided , the identifier
        will be the name.

        Returns
        -------
        str or None
            The trainee's ID.
        """
        return self._id

    @property
    def project_id(self) -> str | None:
        """
        The unique identifier of the trainee's project.

        Returns
        -------
        str or None
            The trainee's project ID.
        """
        return self._project_id

    @property
    def project(self) -> Project | None:
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
    def save_location(self) -> PathLike:
        """
        The current storage location of the trainee.

        Returns
        -------
        str or bytes or os.PathLike
            The current storage location of the trainee based on the last saved location or the location
            from which the trainee was loaded from. If not saved or loaded from a custom location, then
            the default save location will be returned.
        """
        if self._custom_save_path:
            return self._custom_save_path
        else:
            if isinstance(self.client, LocalSaveableProtocol):
                return self.client.howso.default_save_path
            else:
                return None

    @property
    def name(self) -> str | None:
        """
        The name of the trainee.

        Returns
        -------
        str or None
            The name.
        """
        return self._name

    @name.setter
    def name(self, name: str | None):
        """
        Set the name of the trainee.

        Parameters
        ----------
        name : str or None
            The name.
        """
        if name is not None and len(name) > 128:
            raise ValueError(
                "Invalid value for `name`, length must be less "
                "than or equal to 128"
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
    def persistence(self, persistence: Persistence):
        """
        Set the persistence state of the trainee.

        Parameters
        ----------
        persistence : {"allow", "always", "never"}
            The persistence value.
        """
        allowed_values = {"allow", "always", "never"}
        if persistence not in allowed_values:
            raise ValueError(
                f"Invalid value for ``persistence`` ({persistence}), must be"
                f"one of {allowed_values}"
            )
        self._persistence = persistence
        self.update()

    @property
    def features(self) -> SingleTableFeatureAttributes:
        """
        The trainee feature attributes.

        .. WARNING::
            This returns a deep copy of the feature attributes. To update
            features attributes of the trainee, use the method
            :meth:`set_feature_attributes`.

        Returns
        -------
        SingleTableFeatureAttributes
            The feature attributes of the trainee.
        """
        if self._features:
            return SingleTableFeatureAttributes(deepcopy(self._features))
        else:
            return SingleTableFeatureAttributes({})

    @property
    def metadata(self) -> Dict[str, Any] | None:
        """
        The trainee metadata.

        .. WARNING::
            This returns a deep copy of the metadata. To update the
            metadata of the trainee, use the method :func:`set_metadata`.

        Returns
        -------
        dict or None
            The metadata of the trainee.
        """
        return deepcopy(self._metadata)

    @property
    def needs_analyze(self) -> bool:
        """
        The flag indicating if the Trainee needs to analyze.

        Returns
        -------
        bool
            A flag indicating if the Trainee needs to analyze.
        """
        return self._needs_analyze

    @property
    def calculated_matrices(self) -> Optional[Dict[str, DataFrame]]:
        """
        The calculated matrices.

        Returns
        -------
        None or dict of str -> DataFrame
            The calculated matrices.
        """
        return self._calculated_matrices

    @property
    def default_action_features(self) -> List[str] | None:
        """
        The default action features of the trainee.

        .. WARNING::
            This returns a deep copy of the default action features. To
            update them, use the method :meth:`set_default_features`.

        Returns
        -------
        None or list of str
            The default action feature names for the trainee.
        """
        return deepcopy(self._default_action_features)

    @property
    def default_context_features(self) -> None | List[str]:
        """
        The default context features of the trainee.

        .. WARNING::
            This returns a deep copy of the default context features. To
            update them, use the method :meth:`set_default_features`.

        Returns
        -------
        None or list of str
            The default context feature names for the trainee.
        """
        return deepcopy(self._default_context_features)

    @property
    def active_session(self) -> Session | None:
        """
        The active session.

        Returns
        -------
        Session or None
            The session instance, if it exists.
        """
        if isinstance(self.client, AbstractHowsoClient) and self.client.active_session:
            return Session.from_openapi(self.client.active_session, client=self.client)

    def save(self, file_path: Optional[PathLike] = None):
        """
        Save a Trainee to disk.

        Parameters
        ----------
        file_path : str | bytes | os.PathLike, optional
            The path of the file to save the Trainee to. This path can contain
            an absolute path, a relative path or simply a file name. If no filepath
            is provided, the default filepath will be the CWD. If ``file_path`` is a
            relative path (with or without a file name), the absolute path will be
            computed appending the ``file_path`` to the CWD. If ``file_path`` is an
            absolute path, this is the absolute path that will be used. If ``file_path``
            does not contain a filename, then the natural trainee name will be used ``<uuid>.caml``.
        """
        if not isinstance(self.client, LocalSaveableProtocol):
            raise HowsoError("To save, ``client`` type must have local disk access.")

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

            self._custom_save_path = file_path
            file_name = file_path.stem
            file_path = f"{file_path.parents[0]}/"
        else:
            file_name = None

        if self.id:
            self.client.howso.persist(
                trainee_id=self.id,
                filename=file_name,
                filepath=file_path
            )
        else:
            raise ValueError("Trainee ID is needed for saving.")

    def set_feature_attributes(self, feature_attributes: SingleTableFeatureAttributes):
        """
        Update the trainee feature attributes.

        Parameters
        ----------
        feature_attributes : SingleTableFeatureAttributes
            The feature attributes of the trainee. Where feature ``name`` is the
            key and a sub dictionary of feature attributes is the value.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.set_feature_attributes(
                trainee_id=self.id, feature_attributes=feature_attributes
            )
            if self.id:
                if self.client.trainee_cache:
                    self._features = self.client.trainee_cache.get(self.id).features
                else:
                    raise ValueError("Trainee cache is empty, Trainee features are not added.")
            else:
                raise ValueError("Trainee ID is needed for setting feature attributes.")
        else:
            raise ValueError("Client must have the 'set_feature_attributes' method.")

    def set_default_features(
        self,
        *,
        action_features: Optional[Iterable[str]] = None,
        context_features: Optional[Iterable[str]] = None,
    ):
        """
        Update the trainee default features.

        Parameters
        ----------
        action_features : iterable of str, optional
            The default action feature names.
        context_features : iterable of str, optional
            The default context feature names.
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

    def set_metadata(self, metadata: Optional[MutableMapping[str, Any]]):
        """
        Update the trainee metadata.

        Parameters
        ----------
        metadata : map of str -> any, optional
            Any key-value pair to store as custom metadata for the trainee.
            Providing ``None`` will remove the current metadata.
        """
        self._metadata = metadata
        self.update()

    def copy(
        self,
        name: Optional[str] = None,
        *,
        library_type: Optional[Library] = None,
        project: Optional[str | BaseProject] = None,
        resources: Optional["TraineeResources" | MutableMapping[str, Any]] = None,
    ) -> "Trainee":
        """
        Copy the trainee to another trainee.

        Parameters
        ----------
        name : str, optional
            The name of the new trainee.
        library_type : {"st", "mt"}, optional
            The library type of the Trainee. "st" will use the single-threaded library,
            while "mt" will use the multi-threaded library.
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

        if isinstance(self.client, AbstractHowsoClient):
            copy = self.client.copy_trainee(**params)
        else:
            copy = None
        if copy:
            if isinstance(self.client, AbstractHowsoClient):
                return Trainee.from_openapi(copy, client=self.client)
            raise ValueError("Client must be an instance of 'AbstractHowsoClient'")
        else:
            raise ValueError('Trainee not correctly copied')

    def copy_subtrainee(
        self,
        new_trainee_name: str,
        *,
        source_id: Optional[str] = None,
        source_name_path: Optional[List[str]] = None,
        target_id: Optional[str] = None,
        target_name_path: Optional[List[str]] = None,
    ):
        """
        Copy a subtrainee in trainee's hierarchy.

        Parameters
        ----------
        new_trainee_name: str
            The name of the new Trainee.
        source_id: str, optional
            Id of source trainee to copy. Ignored if source_name_path is
            specified. If neither source_name_path nor source_id are specified,
            copies the trainee itself.
        source_name_path: list of str, optional
            list of strings specifying the user-friendly path of the child
            subtrainee to copy.
        target_id: str, optional
            Id of target trainee to copy trainee into.  Ignored if
            target_name_path is specified. If neither target_name_path nor
            target_id are specified, copies as a direct child of trainee.
        target_name_path: list of str, optional
            List of strings specifying the user-friendly path of the child
            subtrainee to copy trainee into.
        """
        self.client.copy_subtrainee(
            self.id,
            new_trainee_name,
            source_id=source_id,
            source_name_path=source_name_path,
            target_id=target_id,
            target_name_path=target_name_path
        )

    def persist(self) -> None:
        """
        Persist the trainee.

        Returns
        -------
        None
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.persist_trainee(self.id)
            self._was_saved = True

    def delete(self):
        """
        Delete the trainee from the last loaded or saved location.

        If trying to delete a trainee from another location, see :func:`delete_trainee`.
        """
        if isinstance(self.client, AbstractHowsoClient):
            if isinstance(self.client, LocalSaveableProtocol) and self._custom_save_path is not None:
                self.client.delete_trainee(trainee_id=self.id, file_path=self._custom_save_path)
            else:
                if not self.id:
                    raise ValueError("Trainee not deleted, id doesn't exist.")
                self.client.delete_trainee(trainee_id=self.id)
        else:
            raise ValueError("Client must have the 'delete_trainee' method.")

        self._created = False
        self._id = None

    def unload(self):
        """
        Unload the trainee.

        .. deprecated:: 1.0.0
            Use :meth:`release_resources` instead.
        """
        warnings.warn(
            'The method ``unload()`` is deprecated and will be removed '
            'in a future release. Please use ``release_resources()`` '
            'instead.', DeprecationWarning)
        self.release_resources()

    def acquire_resources(self, *, max_wait_time: Optional[int | float] = None):
        """
        Acquire resources for a trainee in the Howso service.

        Parameters
        ----------
        max_wait_time : int or float, optional
            The number of seconds to wait for trainee resources to be acquired
            before aborting gracefully. Set to 0 (or None) to wait as long as
            the system-configured maximum for sufficient resources to become
            available, which is typically 20 minutes.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.acquire_trainee_resources(self.id, max_wait_time=max_wait_time)
        else:
            raise ValueError("Client must have the 'acquire_trainee_resources' method.")

    def release_resources(self):
        """Release a trainee's resources from the Howso service."""
        if not self.id:
            return
        if isinstance(self.client, AbstractHowsoClient):
            self.client.release_trainee_resources(self.id)
        else:
            raise ValueError("Client must have the 'release_trainee_resources' method.")

    def information(self) -> "TraineeInformation":
        """
        Get detail information about the trainee.

        Returns
        -------
        TraineeInformation
            The trainee detail information. Including trainee version and
            configuration parameters.
        """
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.get_trainee_information(self.id)
        else:
            raise ValueError("Client must have 'get_trainee_information' method")

    def metrics(self) -> "Metrics":
        """
        Get metric information of the trainee.

        Returns
        -------
        Metrics
            The trainee metric information. Including cpu and memory.
        """
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.get_trainee_metrics(self.id)
        else:
            raise ValueError("Client must have 'get_trainee_metrics' method")

    def set_random_seed(self, seed: int | float | str):
        """
        Set the random seed for the trainee.

        Parameters
        ----------
        seed : int or float or str
            The random seed.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.set_random_seed(trainee_id=self.id, seed=seed)

    def train(
        self,
        cases: TabularData2D,
        *,
        accumulate_weight_feature: Optional[str] = None,
        batch_size: Optional[int] = None,
        derived_features: Optional[Iterable[str]] = None,
        features: Optional[Iterable[str]] = None,
        initial_batch_size: Optional[int] = None,
        input_is_substituted: bool = False,
        progress_callback: Optional[Callable] = None,
        series: Optional[str] = None,
        skip_auto_analyze: bool = False,
        train_weights_only: bool = False,
        validate: bool = True,
    ):
        """
        Train one or more cases into the trainee (model).

        Parameters
        ----------
        cases : DataFrame or 2-dimensional list of object
            One or more cases to train into the model.
        accumulate_weight_feature : str, optional
            Name of feature into which to accumulate neighbors'
            influences as weight for ablated cases. If unspecified, will not
            accumulate weights.
        batch_size : int, optional
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
            A list of feature names. This parameter must be provided when
            ``cases`` is not a DataFrame with named columns. Otherwise, this parameter
            can be provided when you do not want to train on all of the features
            in ``cases`` or you want to re-order the features in ``cases``.
        initial_batch_size : int, optional
            Define the number of cases to train in the first batch. If
            unspecified, a default defined by the ``train_initial_batch_size``
            property of the selected client will be used.
            The number of cases in following batches will be automatically
            adjusted. This value is ignored if ``batch_size`` is specified.
        input_is_substituted : bool, default False
            If True assumes provided nominal feature values have already
            been substituted.
        progress_callback : callable, optional
            A callback method that will be called before each
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
        skip_auto_analyze : bool, default False
            When true, the Trainee will not auto-analyze when appropriate.
            Instead, the 'needs_analyze' property of the Trainee will be
            updated.
        train_weights_only:  bool, default False
            When true, and accumulate_weight_feature is provided,
            will accumulate all of the cases' neighbor weights instead of
            training the cases into the model.
        validate : bool, default True
            Whether to validate the data against the provided feature
            attributes. Issues warnings if there are any discrepancies between
            the data and the features dictionary.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self._needs_analyze = False
            needs_analyze = self.client.train(
                trainee_id=self.id,
                accumulate_weight_feature=accumulate_weight_feature,
                batch_size=batch_size,
                cases=cases,
                derived_features=derived_features,
                features=features,
                initial_batch_size=initial_batch_size,
                input_is_substituted=input_is_substituted,
                progress_callback=progress_callback,
                series=series,
                skip_auto_analyze=skip_auto_analyze,
                train_weights_only=train_weights_only,
                validate=validate,
            )
            self._needs_analyze = needs_analyze
        else:
            raise ValueError("Client must have the 'train' method.")

    def auto_analyze(self) -> None:
        """
        Auto-analyze the trainee.

        Re-use all parameters from the previous :meth:`analyze` call, assuming that
        the user has called :meth:`analyze` before. If not, it will default to a
        robust and versatile analysis.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.auto_analyze(self.id)
        else:
            raise ValueError("Client must have the 'auto_analyze' method.")

    def get_auto_ablation_params(self) -> Dict[str, Any]:
        """
        Get trainee parameters for auto ablation set by :meth:`set_auto_ablation_params`.

        Returns
        -------
        dict of str -> any
            A dictionary mapping parameter names to parameter values.
        """
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.get_auto_ablation_params(self.id)
        else:
            raise ValueError("Client must have the 'get_auto_ablation_params' method.")

    def set_auto_ablation_params(
        self,
        auto_ablation_enabled: bool = False,
        *,
        auto_ablation_weight_feature: str = ".case_weight",
        conviction_lower_threshold: Optional[float] = None,
        conviction_upper_threshold: Optional[float] = None,
        exact_prediction_features: Optional[List[str]] = None,
        influence_weight_entropy_threshold: float = 0.6,
        minimum_model_size: int = 1_000,
        relative_prediction_threshold_map: Optional[MutableMapping[str, float]] = None,
        residual_prediction_features: Optional[List[str]] = None,
        tolerance_prediction_threshold_map: Optional[MutableMapping[str, Tuple[float, float]]] = None,
        **kwargs
    ):
        """
        Set trainee parameters for auto ablation.

        .. note::
            Auto-ablation is experimental and the API may change without deprecation.

        Parameters
        ----------
        auto_ablation_enabled : bool, default False
            When True, the :meth:`train` method will ablate cases that meet the set criteria.
        auto_ablation_weight_feature : str, default ".case_weight"
            The weight feature that should be accumulated to when cases are ablated.
        minimum_model_size : int, default 1,000
            The threshold ofr the minimum number of cases at which the model should auto-ablate.
        influence_weight_entropy_threshold : float, default 0.6
            The influence weight entropy quantile that a case must be beneath in order to be trained.
        exact_prediction_features : list of str, optional
            For each of the features specified, will ablate a case if the prediction matches exactly.
        residual_prediction_features : list of str, optional
            For each of the features specified, will ablate a case if
            abs(prediction - case value) / prediction <= feature residual.
        tolerance_prediction_threshold_map : map of str to tuple of float, optional
            For each of the features specified, will ablate a case if the prediction >= (case value - MIN)
            and the prediction <= (case value + MAX).
        relative_prediction_threshold_map : map of str -> (float, float), optional
            For each of the features specified, will ablate a case if
            abs(prediction - case value) / prediction <= relative threshold
        conviction_lower_threshold : float, optional
            The conviction value above which cases will be ablated.
        conviction_upper_threshold : float, optional
            The conviction value below which cases will be ablated.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.set_auto_ablation_params(
                trainee_id=self.id,
                auto_ablation_enabled=auto_ablation_enabled,
                auto_ablation_weight_feature=auto_ablation_weight_feature,
                minimum_model_size=minimum_model_size,
                influence_weight_entropy_threshold=influence_weight_entropy_threshold,
                exact_prediction_features=exact_prediction_features,
                residual_prediction_features=residual_prediction_features,
                tolerance_prediction_threshold_map=tolerance_prediction_threshold_map,
                relative_prediction_threshold_map=relative_prediction_threshold_map,
                conviction_lower_threshold=conviction_lower_threshold,
                conviction_upper_threshold=conviction_upper_threshold,
            )
        else:
            raise ValueError("Client must have the 'set_auto_ablation_params' method.")

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

        .. seealso::
            The keyword arguments of :meth:`analyze`.

        Parameters
        ----------
        auto_analyze_enabled : bool, default False
            When True, the :meth:`train` method will trigger an analyze when
            it's time for the model to be analyzed again.
        analyze_threshold : int, optional
            The threshold for the number of cases at which the model should be
            re-analyzed.
        auto_analyze_limit_size : int, optional
            The size of the model at which to stop doing auto-analysis. Value of
            0 means no limit.
        analyze_growth_factor : float, optional
            The factor by which to increase the analysis threshold every
            time the model grows to the current threshold size.
        **kwargs: dict, optional
            Accepts any of the keyword arguments in :meth:`analyze`.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.set_auto_analyze_params(
                trainee_id=self.id,
                auto_analyze_enabled=auto_analyze_enabled,
                auto_analyze_limit_size=auto_analyze_limit_size,
                analyze_growth_factor=analyze_growth_factor,
                analyze_threshold=analyze_threshold,
                **kwargs,
            )
        else:
            raise ValueError("Client must have the 'react_into_trainee' method.")

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
        targeted_model: Optional[TargetedModel] = None,
        use_case_weights: Optional[bool] = None,
        use_deviations: Optional[bool] = None,
        weight_feature: Optional[str] = None,
        **kwargs
    ):
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
        inverse_residuals_as_weights : bool, default False
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
                3. Standard details
                4. Full analysis

        p_values : list of float, optional
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

        use_case_weights : bool, default False
            When True will scale influence weights by each
            case's weight_feature weight.
        use_deviations : bool, default False
            When True, uses deviations for LK metric in queries.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        **kwargs : dict, optional
            Additional experimental analyze parameters.
        """
        if isinstance(self.client, AbstractHowsoClient):
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
        contexts: Optional[TabularData2D] = None,
        *,
        action_features: Optional[Iterable[str]] = None,
        allow_nulls: bool = False,
        case_indices: Optional[CaseIndices] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        leave_case_out: Optional[bool] = None,
        suppress_warning: bool = False,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None,
    ) -> DataFrame:
        """
        Wrapper around :meth:`react`.

        Performs a discriminative react to predict the action feature values based on the
        given contexts. Returns only the predicted action values.

        Parameters
        ----------
        contexts : DataFrame or 2-dimensional list of object, optional
            The context values to react to. If neither this nor ``context_values`` are
            specified then ``case_indices`` must be specified.
        action_features : list of str, optional
            Feature names to treat as action features during react. If no
            ``action_features`` are specified, the ``default_action_features``
            is used.
        allow_nulls : bool, default False, optional
            See parameter ``allow_nulls`` in :meth:`react`.
        case_indices : iterable of (str, int), optional
            Case indices to react to in lieu of ``contexts`` or ``context_values``.
            If these are not specified, one of ``contexts`` or ``context_values``
            must be specified.
        context_features : list of str, optional
            Feature names to treat as context features during react. If no
            ``context_features`` are specified, then the ``default_context_features``
            are used. If the Trainee has no ``default_context_features``, then
            this will be all of the ``features`` excluding the ``action_features``.
        derived_action_features : list of str, optional
            See parameter ``derived_action_features`` in :meth:`react`.
        derived_context_features : list of str, optional
            See parameter ``derived_context_features`` in :meth:`react`.
        leave_case_out : bool, default False
            See parameter ``leave_case_out`` in :meth:`react`.
        suppress_warning : bool, default False
            See parameter ``suppress_warning`` in :meth:`react`.
        use_case_weights : bool, default False
            See parameter ``use_case_weights`` in :meth:`react`.
        weight_feature : str, optional
            See parameter ``weight_feature`` in :meth:`react`.

        Returns
        -------
        DataFrame
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
        contexts: Optional[TabularData2D] = None,
        *,
        action_features: Optional[Iterable[str]] = None,
        actions: Optional[TabularData2D] = None,
        allow_nulls: bool = False,
        batch_size: Optional[int] = None,
        case_indices: Optional[CaseIndices] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        post_process_features: Optional[Iterable[str]] = None,
        post_process_values: Optional[TabularData2D] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[MutableMapping[str, object]] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        feature_bounds_map: Optional[MutableMapping[str, MutableMapping[str, object]]] = None,
        generate_new_cases: GenerateNewCases = "no",
        initial_batch_size: Optional[int] = None,
        input_is_substituted: bool = False,
        into_series_store: Optional[str] = None,
        leave_case_out: Optional[bool] = None,
        new_case_threshold: NewCaseThreshold = "min",
        num_cases_to_generate: int = 1,
        ordered_by_specified_features: bool = False,
        preserve_feature_values: Optional[Iterable[str]] = None,
        progress_callback: Optional[Callable] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None,
    ) -> Reaction:
        r"""
        React to the provided contexts.

        If ``desired_conviction`` is specified, executes a generative react,
        producing ``action_values`` for the specified ``action_features``
        conditioned on the optionally provided ``contexts``.

        If ``desired_conviction`` is **not** specified, executes a discriminative
        react. Provided a list of ``contexts``, the trainee reacts to the model
        and produces predictions for the specified actions.

        Parameters
        ----------
        contexts : DataFrame or 2-dimensional list of object, optional
            The context values to react to.
        action_features : list of str, optional
            Feature names to treat as action features during react.
        actions : DataFrame or 2-dimensional list of object, optional
            One or more action values to use for action features.
            If specified, will only return the specified explanation
            details for the given actions. (Discriminative reacts only)
        allow_nulls : bool, default False
            When true will allow return of null values if there
            are nulls in the local model for the action features, applicable
            only to discriminative reacts.
        batch_size: int, optional
            Define the number of cases to react to at once. If left unspecified,
            the batch size will be determined automatically.
        case_indices : iterable of (str, int), optional
            Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If this case does not exist, discriminative react
            outputs null, generative react ignores it.
        context_features : list of str, optional
            Feature names to treat as context features during react.
        derived_action_features : list of str, optional
            Features whose values should be computed after reaction from
            the resulting case prior to output, in the specified order.
            Must be a subset of ``action_features``.

            .. NOTE::
                Both of these derived feature lists rely on the features'
                "derived_feature_code" attribute to compute the values. If the
                "derived_feature_code" attribute is undefined or references a
                non-0 feature indices, the derived value will be null.

        derived_context_features : list of str, optional
            Features whose values should be computed from the provided
            context in the specified order.
        post_process_features : iterable of str, optional
            List of feature names that will be made available during the
            execution of post_process feature attributes.
        post_process_values : DataFrame or 2-dimensional list of object, optional
            A 2d list of values corresponding to post_process_features that
            will be made available during the execution of post_process feature
            attributes.
        desired_conviction : float, optional
            If specified will execute a generative react. If not
            specified will execute a discriminative react. Conviction is the
            ratio of expected surprisal to generated surprisal for each
            feature generated, valid values are in the range of :math:`(0,\infty]`.
        details : map of str -> object, optional
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
                Relies on 'robust_influences' parameter to determine whether
                to do standard or robust computation.
            - case_feature_residuals : bool, optional
                If True, outputs feature residuals for all (context and action)
                features for just the specified case. Uses leave-one-out for
                each feature, while using the others to predict the left out
                feature with their corresponding values from this case. Relies
                on 'robust_residuals' parameter to determine whether to do
                standard or robust computation.
            - case_mda : bool, optional
                If True, outputs each influential case's mean decrease in
                accuracy of predicting the action feature in the local model
                area, as if each individual case were included versus not
                included. Uses only the context features of the reacted case to
                determine that area. Relies on 'robust_influences' parameter
                to determine whether to do standard or robust computation.
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
            - feature_contributions : bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context were not in the
                model for all context features in the local model area. Relies
                on 'robust_influences' parameter to determine whether to do
                standard or robust computation. Directional feature
                contributions are returned under the key
                'directional_feature_contributions'.
            - case_feature_contributions: bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context feature were not
                in the model for all context features in this case, using only
                the values from this specific case. Relies on
                'robust_influences' parameter to determine whether to do
                standard or robust computation. Directional case feature
                contributions are returned under the
                'case_directional_feature_contributions' key.
            - feature_mda : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature given the context.
                Uses only the context features of the reacted case to determine
                that area. Relies on 'robust_influences' parameter to
                determine whether to do standard or robust computation.
            - feature_mda_ex_post : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature as an explanation detail
                given that the specified prediction was already made as
                specified by the action value. Uses both context and action
                features of the reacted case to determine that area. Relies on
                'robust_influences' parameter to determine whether to do
                standard or robust computation.
            - features : list of str, optional
                A list of feature names that specifies for what features will
                per-feature details be computed (residuals, contributions,
                mda, etc.). This should generally preserve compute, but will
                not when computing details robustly. Details will be computed
                for all context and action features if this value is not
                specified.
            - feature_residuals : bool, optional
                If True, outputs feature residuals for all (context and action)
                features locally around the prediction. Uses only the context
                features of the reacted case to determine that area. Relies on
                'robust_residuals' parameter to determine whether to do
                standard or robust computation.
            - global_case_feature_residual_convictions : bool, optional
                If True, outputs this case's feature residual convictions for
                the global model. Computed as: global model feature residual
                divided by case feature residual. Relies on
                'robust_residuals' parameter to determine whether to do
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
                residual. Relies on 'robust_residuals' parameter to determine
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
            - similarity_conviction : bool, optional
                If True, outputs similarity conviction for the reacted case.
                Uses both context and action feature values as the case values
                for all computations. This is defined as expected (local)
                distance contribution divided by reacted case distance
                contribution.
            - robust_computation: bool, optional
                Deprecated. If specified, will overwrite the value of both
                'robust_residuals' and 'robust_influences'.
            - robust_residuals: bool, optional
                Default is false, uses leave-one-out for features (or cases, as
                needed) for all residual computations. When true, uses uniform
                sampling from the power set of all combinations of features (or
                cases, as needed) instead.
            - robust_influences: bool, optional
                Default is true, uses leave-one-out for features (or cases, as
                needed) for all MDA and contribution computations. When true,
                uses uniform sampling from the power set of all combinations of
                features (or cases, as needed) instead.
            - generate_attempts : bool, optional
                If True outputs the number of attempts taken to generate each
                case. Only applicable when 'generate_new_cases' is "always" or
                "attempt".

        exclude_novel_nominals_from_uniqueness_check : bool, default False
            If True, will exclude features which have a subtype defined in their feature
            attributes from the uniqueness check that happens when ``generate_new_cases``
            is True. Only applies to generative reacts.
        feature_bounds_map : map of str -> map of str -> object, optional
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

        generate_new_cases : {"always", "attempt", "no"}, default "no"
            This parameter takes in a string that may be one of the following:

                - **attempt**: ``Synthesizer`` attempts to generate new cases and
                  if its not possible to generate a new case, it might
                  generate cases in "no" mode (see point c.)
                - **always**: ``Synthesizer`` always generates new cases and
                  if its not possible to generate a new case, it returns
                  ``None``.
                - **no**: ``Synthesizer`` generates data based on the
                  ``desired_conviction`` specified and the generated data is
                  not guaranteed to be a new case (that is, a case not found
                  in original dataset.)

        initial_batch_size: int, optional
            Define the number of cases to react to in the first batch. If
            unspecified, a default defined by the ``react_initial_batch_size``
            property of the selected client will be used.
            The number of cases in following batches will be automatically
            adjusted. This value is ignored if ``batch_size`` is specified.
        input_is_substituted : bool, default False
            When True, assumes provided categorical (nominal or ordinal)
            feature values have already been substituted.
        into_series_store : str, optional
            The name of a series store. If specified, will store an internal
            record of all react contexts for this session and series to be used
            later with train series.
        leave_case_out : bool, default False
            When True and specified along with ``case_indices``, each individual
            react will respectively ignore the corresponding case specified
            by ``case_indices`` by leaving it out.
        new_case_threshold : {"max", "min", "most_similar"}, default "min"
            Distance to determine the privacy cutoff.

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
            by ``case_indices``, appending and overwriting the specified
            contexts as necessary. For generative reacts, if ``case_indices``
            isn't specified will preserve feature values of a random case.
        progress_callback : callable, optional
            A callback method that will be called before each
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
            ``weight_feature`` weight.
        use_regional_model_residuals : bool, default True
            When false, uses model feature residuals. When True, recalculates
            regional model residuals.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        Reaction
            A MutableMapping (dict-like) with these keys -> values:
                action -> DataFrame
                    A data frame of action values.

                details -> dict or list
                    An aggregated list of any requested details.
        """
        return self.client.react(
            trainee_id=self.id,
            action_features=action_features,
            actions=actions,
            allow_nulls=allow_nulls,
            batch_size=batch_size,
            case_indices=case_indices,
            contexts=contexts,
            context_features=context_features,
            derived_action_features=derived_action_features,
            derived_context_features=derived_context_features,
            desired_conviction=desired_conviction,
            details=details,
            exclude_novel_nominals_from_uniqueness_check=exclude_novel_nominals_from_uniqueness_check,
            feature_bounds_map=feature_bounds_map,
            generate_new_cases=generate_new_cases,
            initial_batch_size=initial_batch_size,
            input_is_substituted=input_is_substituted,
            into_series_store=into_series_store,
            leave_case_out=leave_case_out,
            new_case_threshold=new_case_threshold,
            num_cases_to_generate=num_cases_to_generate,
            ordered_by_specified_features=ordered_by_specified_features,
            post_process_features=post_process_features,
            post_process_values=post_process_values,
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
        contexts: Optional[TabularData2D] = None,
        *,
        action_features: Optional[Iterable[str]] = None,
        actions: Optional[TabularData2D] = None,
        batch_size: Optional[int] = None,
        case_indices: Optional[CaseIndices] = None,
        context_features: Optional[Iterable[str]] = None,
        continue_series: bool = False,
        continue_series_features: Optional[Iterable[str]] = None,
        continue_series_values: Optional[TabularData3D] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[MutableMapping[str, object]] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        feature_bounds_map: Optional[MutableMapping[str, MutableMapping[str, object]]] = None,
        final_time_steps: Optional[List[object]] = None,
        generate_new_cases: GenerateNewCases = "no",
        series_index: str = ".series",
        init_time_steps: Optional[List[object]] = None,
        initial_batch_size: Optional[int] = None,
        initial_features: Optional[Iterable[str]] = None,
        initial_values: Optional[TabularData2D] = None,
        input_is_substituted: bool = False,
        leave_case_out: Optional[bool] = None,
        max_series_lengths: Optional[List[int]] = None,
        new_case_threshold: NewCaseThreshold = "min",
        num_series_to_generate: int = 1,
        ordered_by_specified_features: bool = False,
        output_new_series_ids: bool = True,
        preserve_feature_values: Optional[Iterable[str]] = None,
        progress_callback: Optional[Callable] = None,
        series_context_features: Optional[Iterable[str]] = None,
        series_context_values: Optional[TabularData3D] = None,
        series_id_tracking: SeriesIDTracking = "fixed",
        series_stop_maps: Optional[List[MutableMapping[str, MutableMapping[str, object]]]] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None,
    ) -> Reaction:
        """
        React to the trainee in a series until a stop condition is met.

        Aggregates rows of data corresponding to the specified context, action,
        derived_context and derived_action features, utilizing previous rows to
        derive values as necessary. Outputs a dict of "action_features" and
        corresponding "action" where "action" is the completed 'matrix' for
        the corresponding action_features and derived_action_features.

        Parameters
        ----------
        contexts : DataFrame or 2-dimensional list of object, optional
            The context values to react to.
        action_features : list of str, optional
            See parameter ``action_features`` in :meth:`react`.
        actions : DataFrame or 2-dimensional list of object, optional
            See parameter ``actions`` in :meth:`react`.
        batch_size: int, optional
            Define the number of series to react to at once. If left
            unspecified, the batch size will be determined automatically.
        case_indices : CaseIndices
            See parameter ``case_indices`` in :meth:`react`.
        context_features : list of str, optional
            See parameter ``context_features`` in :meth:`react`.
        continue_series : bool, default False
            When True will attempt to continue existing series instead of
            starting new series. If ``initial_values`` provide series IDs, it
            will continue those explicitly specified IDs, otherwise it will
            randomly select series to continue.
            .. note::

                Terminated series with terminators cannot be continued and
                will result in null output.
        continue_series_features : list of str, optional
            The list of feature names corresponding to the values in each row of
            ``continue_series_values``. This value is ignored if
            ``continue_series_values`` is None.
        continue_series_values : list of DataFrame or 3-dimensional list of object, optional
            The set of series data to be forecasted with feature values in the
            same order defined by ``continue_series_values``. The value of
            ``continue_series`` will be ignored and treated as true if this value
            is specified.
        derived_action_features : list of str, optional
            See parameter ``derived_action_features`` in :meth:`react`.
        derived_context_features : list of str, optional
            See parameter ``derived_context_features`` in :meth:`react`.
        desired_conviction : float, optional
            See parameter ``desired_conviction`` in :meth:`react`.
        details : map of str to object
            See parameter ``details`` in :meth:`react`.
        exclude_novel_nominals_from_uniqueness_check : bool, default False
            If True, will exclude features which have a subtype defined in their feature
            attributes from the uniqueness check that happens when ``generate_new_cases``
            is True. Only applies to generative reacts.
        feature_bounds_map : map of str -> map of str -> object, optional
            See parameter ``feature_bounds_map`` in :meth:`react`.
        final_time_steps: list of object, optional
            The time steps at which to end synthesis. Time-series only.
            Time-series only. Must provide either one for all series, or
            exactly one per series.
        generate_new_cases : {"always", "attempt", "no"}, default "no"
            See parameter ``generate_new_cases`` in :meth:`react`.
        series_index : str, default ".series"
            When set to a string, will include the series index as a
            column in the returned DataFrame using the column name given.
            If set to None, no column will be added.
        init_time_steps: list of object, optional
            The time steps at which to begin synthesis. Time-series only.
            Time-series only. Must provide either one for all series, or
            exactly one per series.
        initial_batch_size: int, optional
            The number of series to react to in the first batch. If unspecified,
            the number will be determined automatically by the client. The
            number of series in following batches will be automatically
            adjusted. This value is ignored if ``batch_size`` is specified.
        initial_features : list of str, optional
            Features to condition just the first case in a series,
            overwrites context_features and derived_context_features for that
            first case. All specified initial features must be in one of:
            context_features, action_features, derived_context_features or
            derived_action_features. If provided a value that isn't in one of
            those lists, it will be ignored.
        initial_values : DataFrame or 2-dimensional list of object, optional
            Values corresponding to the initial_features, used to condition
            just the first case in each series. Must provide either exactly one
            value to use for all series, or one per series.
        input_is_substituted : bool, default False
            See parameter ``input_is_substituted`` in :meth:`react`.
        leave_case_out : bool, default False
            See parameter ``leave_case_out`` in :meth:`react`.
        max_series_lengths : list of int, optional
            maximum size a series is allowed to be.  Default is
            3 * model_size, a 0 or less is no limit. If forecasting
            with ``continue_series``, this defines the maximum length of the
            forecast. Must provide either one for all series, or exactly
            one per series.
        new_case_threshold : str, optional
            See parameter ``new_case_threshold`` in :meth:`react`.
        num_series_to_generate : int, default 1
            The number of series to generate.
        ordered_by_specified_features : bool, default False
            See parameter ``ordered_by_specified_features`` in :meth:`react`.
        output_new_series_ids : bool, default True
            If True, series ids are replaced with unique values on output.
            If False, will maintain or replace ids with existing trained values,
            but also allows output of series with duplicate existing ids.
        preserve_feature_values : list of str, optional
            See parameter ``preserve_feature_values`` in :meth:`react`.
        progress_callback : callable, optional
            A callback method that will be called before each
            batched call to react series and at the end of reacting. The method
            is given a ProgressTimer containing metrics on the progress and
            timing of the react series operation, and the batch result.
        series_context_features : list of str, optional
            List of context features corresponding to series_context_values, if
            specified must not overlap with any initial_features or context_features.
        series_context_values : list of list of list of object or list of DataFrame, optional
            3d list of context values, one for each feature for each row for each
            series. If specified, batch_size and max_series_lengths are ignored.
        series_id_tracking : {"fixed", "dynamic", "no"}, default "fixed"
            Controls how closely generated series should follow existing series (plural).

            - If "fixed", tracks the particular relevant series ID.
            - If "dynamic", tracks the particular relevant series ID, but is allowed to
              change the series ID that it tracks based on its current context.
            - If "no", does not track any particular series ID.
        series_stop_maps : list of map of str -> dict, optional
            Map of series stop conditions. Must provide either exactly one to
            use for all series, or one per series.

            .. TIP::
                Stop series when value exceeds max or is smaller than min::

                    {"feature_name":  {"min" : 1, "max": 2}}

                Stop series when feature value matches any of the values
                listed::

                    {"feature_name":  {"values": ["val1", "val2"]}}

        substitute_output : bool, default True
            See parameter ``substitute_output`` in :meth:`react`.
        suppress_warning : bool, default False
            See parameter ``suppress_warning`` in :meth:`react`.
        use_case_weights : bool, default False
            See parameter ``use_case_weights`` in :meth:`react`.
        use_regional_model_residuals : bool, default True
            See parameter ``use_regional_model_residuals`` in :meth:`react`.
        weight_feature : str, optional
            See parameter ``weight_feature`` in :meth:`react`.

        Returns
        -------
        Reaction
            A MutableMapping (dict-like) with these keys -> values:
                action -> DataFrame
                    A data frame of action values.

                details -> dict or list
                    An aggregated list of any requested details.
        """
        if self.id:
            return self.client.react_series(
                trainee_id=self.id,
                action_features=action_features,
                actions=actions,
                batch_size=batch_size,
                case_indices=case_indices,
                contexts=contexts,
                context_features=context_features,
                continue_series=continue_series,
                continue_series_features=continue_series_features,
                continue_series_values=continue_series_values,
                derived_action_features=derived_action_features,
                derived_context_features=derived_context_features,
                desired_conviction=desired_conviction,
                details=details,
                exclude_novel_nominals_from_uniqueness_check=exclude_novel_nominals_from_uniqueness_check,
                feature_bounds_map=feature_bounds_map,
                final_time_steps=final_time_steps,
                generate_new_cases=generate_new_cases,
                series_index=series_index,
                init_time_steps=init_time_steps,
                initial_batch_size=initial_batch_size,
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
        else:
            raise ValueError("Trainee ID is needed for react_series.")

    def impute(
        self,
        *,
        batch_size: int = 1,
        features: Optional[Iterable[str]] = None,
        features_to_impute: Optional[Iterable[str]] = None,
    ):
        """
        Impute (fill) the missing values for the specified features_to_impute.

        If no ``features`` are specified, will use all features in the trainee
        for imputation. If no ``features_to_impute`` are specified, will impute
        all features specified by ``features``.

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
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.impute(
                trainee_id=self.id,
                batch_size=batch_size,
                features=features,
                features_to_impute=features_to_impute,
            )
        else:
            raise ValueError("Client must have 'impute' method")

    def remove_cases(
        self,
        num_cases: int,
        *,
        case_indices: Optional[CaseIndices] = None,
        condition: Optional[MutableMapping[str, object]] = None,
        condition_session: Optional[str | BaseSession] = None,
        distribute_weight_feature: Optional[str] = None,
        precision: Optional[Precision] = None,
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
                Example 1 - Remove all values belonging to ``feature_name``::

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
        distribute_weight_feature : str, optional
            When specified, will distribute the removed cases' weights
            from this feature into their neighbors.
        precision : {"exact", "similar"}, optional
            The precision to use when removing the cases.If not specified
            "exact" will be used. Ignored if case_indices is specified.
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
        if isinstance(self.client, AbstractHowsoClient):
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
        else:
            raise ValueError("Client must have 'remove_cases' method")

    def edit_cases(
        self,
        feature_values: TabularData2D,
        *,
        case_indices: Optional[CaseIndices] = None,
        condition: Optional[MutableMapping[str, object]] = None,
        condition_session: Optional[str | BaseSession] = None,
        features: Optional[Iterable[str]] = None,
        num_cases: Optional[int] = None,
        precision: Optional[str] = None
    ) -> int:
        """
        Edit feature values for the specified cases.

        Parameters
        ----------
        feature_values : DataFrame or 2-dimensional list of object
            The feature values to edit the case(s) with. If specified as a list,
            the order corresponds with the order of the ``features`` parameter.
            If specified as a DataFrame, only the first row will be used.
        case_indices : iterable of (str, int), optional
            An iterable of Sequences containing the session id and index, where
            index is the original 0-based index of the case as it was trained
            into the session. This explicitly specifies the cases to edit. When
            specified, ``condition`` and ``condition_session`` are ignored.
        condition : map of str -> object, optional
            A condition map to select which cases to edit. Ignored when
            ``case_indices`` are specified.

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
            The names of the features to edit. Required when ``feature_values``
            is not specified as a DataFrame.
        num_cases : int, optional
            The maximum amount of cases to edit. If not specified, the limit
            will be k cases if precision is "similar", or no limit if precision
            is "exact".
        precision : str, optional
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
        if isinstance(self.client, AbstractHowsoClient):
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
        else:
            raise ValueError("Client must have the 'edit_cases' method.")

    def get_sessions(self) -> List[Dict[str, str]]:
        """
        Get all session ids of the trainee.

        Returns
        -------
        list of dict of str -> str
            A list of dicts with keys "id" and "name" for each session
            in the model.
        """
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.get_trainee_sessions(self.id)
        else:
            raise ValueError("Client must have the 'get_sessions' method.")

    def delete_session(self, session: Union[str, BaseSession]):
        """
        Delete a session from the trainee.

        Parameters
        ----------
        session : str or BaseSession
            The id or instance of the session to remove from the model.
        """
        if isinstance(session, BaseSession):
            session_id = session.id
        else:
            session_id = session
        if isinstance(self.client, AbstractHowsoClient):
            self.client.delete_trainee_session(trainee_id=self.id, session=session_id)
        else:
            raise ValueError("Client must have the 'delete_trainee_session' method.")

    def get_session_indices(self, session: Union[str, BaseSession]) -> Index | List[int]:
        """
        Get all session indices for a specified session.

        Parameters
        ----------
        session : str or BaseSession
            The id or instance of the session to retrieve indices for from
            the model.

        Returns
        -------
        Index or list of int
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

    def get_session_training_indices(self, session: Union[str, BaseSession]) -> Index | List[int]:
        """
        Get all session training indices for a specified session.

        Parameters
        ----------
        session : str or BaseSession
            The id or instance of the session to retrieve training indices for
            from the model.

        Returns
        -------
        Index or list of int
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
        indicate_imputed: bool = False,
        case_indices: Optional[CaseIndices] = None,
        features: Optional[Iterable[str]] = None,
        session: Optional[str | BaseSession] = None,
        condition: Optional[MutableMapping] = None,
        num_cases: Optional[int] = None,
        precision: Optional[str] = None
    ) -> Cases | DataFrame:
        """
        Get the trainee's cases.

        Parameters
        ----------
        case_indices : iterable of (str, int), optional
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
                Example 1 - Retrieve all values belonging to ``feature_name``::

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
        Cases or DataFrame
            The trainee's cases.
        """
        if isinstance(session, BaseSession):
            session_id = session.id
        else:
            session_id = session
        if self.id:
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
        else:
            raise ValueError("Trainee ID is needed for 'get_cases'.")

    def get_extreme_cases(
        self,
        *,
        features: Optional[Iterable[str]] = None,
        num: int,
        sort_feature: str,
    ) -> Cases | DataFrame:
        """
        Get the trainee's extreme cases.

        Parameters
        ----------
        features : list of str, optional
            The features to include in the case data.
        num : int
            The number of cases to get.
        sort_feature : str
            The name of the feature by which extreme cases are sorted.

        Returns
        -------
        Cases or DataFrame
            The trainee's extreme cases.
        """
        if self.id:
            return self.client.get_extreme_cases(
                trainee_id=self.id,
                features=features,
                num=num,
                sort_feature=sort_feature
            )
        else:
            raise ValueError("Trainee ID is needed for 'get_extreme_cases'.")

    def get_num_training_cases(self) -> int:
        """
        Return the number of trained cases for the trainee.

        Returns
        -------
        int
            The number of trained cases.
        """
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.get_num_training_cases(self.id)
        else:
            raise ValueError("Client must have the 'get_num_training_cases' method.")

    def add_feature(
        self,
        feature: str,
        feature_value: Optional[int | float | str] = None,
        *,
        overwrite: bool = False,
        condition: Optional[MutableMapping[str, object]] = None,
        condition_session: Optional[str | BaseSession] = None,
        feature_attributes: Optional[MutableMapping] = None,
    ):
        """
        Add a feature to the model.

        Parameters
        ----------
        feature : str
            The name of the feature.
        feature_attributes : map, optional
            The dict of feature specific attributes for this feature. If
            unspecified and conditions are not specified, will assume feature
            type as 'continuous'.
        feature_value : int or float or str, optional
            The value to populate the feature with.
            By default, populates the new feature with None.
        condition : map of str -> object, optional
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
                For instance to add the ``feature_value`` only when the
                ``length`` and ``width`` features are equal to 10::

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
        if isinstance(self.client, AbstractHowsoClient):
            if self.id:
                self.client.add_feature(
                    trainee_id=self.id,
                    condition=condition,
                    condition_session=condition_session_id,
                    feature=feature,
                    feature_value=feature_value,
                    feature_attributes=feature_attributes,
                    overwrite=overwrite,
                )
                if self.client.trainee_cache:
                    self._features = self.client.trainee_cache.get(self.id).features
                else:
                    raise ValueError("Trainee Cache is empty, Trainee features are not set.")
            else:
                raise ValueError("Trainee ID is needed for 'add_feature'.")
        else:
            raise ValueError("Client must have the 'add_feature' method.")

    def remove_feature(
        self,
        feature: str,
        *,
        condition: Optional[MutableMapping[str, object]] = None,
        condition_session: Optional[str | BaseSession] = None,
    ):
        """
        Remove a feature from the trainee.

        Parameters
        ----------
        feature : str
            The name of the feature to remove.
        condition : map of str -> object, optional
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
                For instance to remove the ``length`` feature only when the
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
        if isinstance(self.client, AbstractHowsoClient):
            if self.id:
                self.client.remove_feature(
                    trainee_id=self.id,
                    condition=condition,
                    condition_session=condition_session_id,
                    feature=feature,
                )
                if self.client.trainee_cache:
                    self._features = self.client.trainee_cache.get(self.id).features
                else:
                    raise ValueError("Trainee cache is empty, Trainee features are not removed.")
            else:
                raise ValueError("Trainee ID is needed for 'get_extreme_cases'.")
        else:
            raise ValueError("Client must have the 'remove_feature' method.")

    def remove_series_store(self, series: Optional[str] = None):
        """
        Clear stored series from trainee.

        Parameters
        ----------
        series : str, optional
            Series id to clear. If not provided, clears the entire
            series store for the trainee.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.remove_series_store(trainee_id=self.id, series=series)
        else:
            raise ValueError("Client must have the 'remove_series_store' method.")

    def append_to_series_store(
        self,
        series: str,
        contexts: TabularData2D,
        *,
        context_features: Optional[Iterable[str]] = None,
    ):
        """
        Append the specified contexts to a series store.

        For use with train series.

        Parameters
        ----------
        series : str
            The name of the series store to append to.
        contexts : DataFrame or 2-dimensional list of object
            The list of context values to append to the series.
        context_features : iterable of str, optional
            The list of feature names for contexts.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.append_to_series_store(
                trainee_id=self.id,
                series=series,
                contexts=contexts,
                context_features=context_features,
            )
        else:
            raise ValueError("Client must have the 'append_to_series_store' method.")

    def set_substitute_feature_values(
        self, substitution_value_map: MutableMapping[str, MutableMapping[str, Any]]
    ):
        """
        Set a substitution map for use in extended nominal generation.

        Parameters
        ----------
        substitution_value_map : map of str -> map of str -> any
            A dictionary of feature name to a dictionary of feature value to
            substitute feature value.

            If this dict is None, all substitutions will be disabled and
            cleared. If any feature in the ``substitution_value_map`` has
            features mapping to ``None`` or ``{}``, substitution values will
            immediately be generated.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.set_substitute_feature_values(
                trainee_id=self.id, substitution_value_map=substitution_value_map
            )
        else:
            raise ValueError("Client must have the 'set_substitute_feature_values' method.")

    def get_substitute_feature_values(
        self,
        *,
        clear_on_get: bool = True,
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
        dict of str -> dict of str -> any
            A dictionary of feature name to a dictionary of feature value to
            substitute feature value.
        """
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.get_substitute_feature_values(
                trainee_id=self.id, clear_on_get=clear_on_get
            )
        else:
            raise ValueError("Client must have the 'get_substitute_feature_values' method.")

    def react_group(
        self,
        new_cases: Union[List["DataFrame"], List[List[List[object]]]],
        *,
        distance_contributions: bool = False,
        familiarity_conviction_addition: bool = True,
        familiarity_conviction_removal: bool = False,
        kl_divergence_addition: bool = False,
        kl_divergence_removal: bool = False,
        p_value_of_addition: bool = False,
        p_value_of_removal: bool = False,
        use_case_weights: bool = False,
        features: Optional[Iterable[str]] = None,
        weight_feature: Optional[str] = None,
    ) -> DataFrame | dict:
        """
        Computes specified data for a **set** of cases.

        Return the list of familiarity convictions (and optionally, distance
        contributions or :math:`p` values) for each set.

        Parameters
        ----------
        distance_contributions : bool, default False
            Calculate and output distance contribution ratios in
            the output dict for each case.
        familiarity_conviction_addition : bool, default True
            Calculate and output familiarity conviction of adding the
            specified cases.
        familiarity_conviction_removal : bool, default False
            Calculate and output familiarity conviction of removing
            the specified cases.
        features : Iterable of str, optional
            A list of feature names to consider while calculating convictions.
        kl_divergence_addition : bool, default False
            Calculate and output KL divergence of adding the
            specified cases.
        kl_divergence_removal : bool, default False
            Calculate and output KL divergence of removing the
            specified cases.
        new_cases : list of DataFrame or 3-dimensional list of object
            Specify a **set** using a list of cases to compute the conviction
            of groups of cases as shown in the following example.

            Example::

                new_cases = [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], # Group 1
                    [[1, 2, 3]], # Group 2
                ]

        p_value_of_addition : bool, default False
            If true will output :math:`p` value of addition.
        p_value_of_removal : bool, default False
            If true will output :math:`p` value of removal.
        use_case_weights : bool, default False
            When True, will scale influence weights by each case's
            ``weight_feature`` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        DataFrame or dict
            The conviction of grouped cases.
        """
        return self.client.react_group(
            trainee_id=self.id,
            new_cases=new_cases,
            features=features,
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
        familiarity_conviction_addition: bool | str = True,
        familiarity_conviction_removal: bool | str = False,
        use_case_weights: bool = False,
        action_features: Optional[Iterable[str]] = None,
        features: Optional[Iterable[str]] = None,
        weight_feature: Optional[str] = None,
    ) -> DataFrame | dict:
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
            Calculate and output familiarity conviction of adding the
            specified cases.
        familiarity_conviction_removal : bool, default False
            Calculate and output familiarity conviction of removing
            the specified cases.
        features : list of str, optional
            The feature names to calculate convictions for. At least 2 features
            are required to get familiarity conviction. If not specified all
            features will be used.
        use_case_weights : bool, default False
            When True, will scale influence weights by each case's
            ``weight_feature`` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        DataFrame or dict
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
    ) -> DataFrame | None:
        """
        Get cached feature residuals.

        All keyword arguments are optional, when not specified will auto-select
        cached residuals for output, when specified will attempt to
        output the cached residuals best matching the requested parameters,
        or None if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`get_prediction_stats` instead.

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
        DataFrame or None
            The feature residuals or None if no cached values are found.
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
        condition: Optional[MutableMapping[str, Any]] = None,
        num_cases: Optional[int] = None,
        num_robust_influence_samples_per_case: Optional[int] = None,
        precision: Optional[Precision] = None,
        robust: Optional[bool] = None,
        robust_hyperparameters: Optional[bool] = None,
        stats: Optional[Iterable[str]] = None,
        weight_feature: Optional[str] = None,
    ) -> DataFrame | dict:
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
        condition : map of str -> any, optional
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
            ``condition`` is not None.
        num_robust_influence_samples_per_case : int, optional
            Specifies the number of robust samples to use for each case for
            robust contribution computations.
            Defaults to 300 + 2 * (number of features).
        precision : {"exact", "similar"}, optional
            The precision to use when selecting cases with the condition.
            If not specified "exact" will be used. Only used if ``condition``
            is not None.
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
                - confusion_matrix : A sparse map of actual feature value to a map of
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
                - missing_value_accuracy : The number of cases with missing
                  values predicted to have missing values divided by the number
                  of cases with missing values, applies to all features that
                  contain missing values.
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
        DataFrame or dict
            A DataFrame of feature name columns to stat value rows. Indexed
            by the stat type. The return type depends on the underlying client.
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
        self, *,
        condition: Optional[MutableMapping[str, Any]] = None,
        num_cases: Optional[int] = None,
        precision: Optional[Precision] = None,
        weight_feature: Optional[str] = None,
    ) -> Union["DataFrame", Dict]:
        """
        Get marginal stats for all features.

        Parameters
        ----------
        condition : map of str -> any, optional
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
            "similar". Only used if ``condition`` is not None.
        precision : {"exact", "similar"}, optional
            The precision to use when selecting cases with the condition.
            Options are 'exact' or 'similar'. If not specified "exact" will be
            used. Only used if ``condition`` is not None.
        weight_feature : str, optional
            When specified, will attempt to return stats that were computed
            using this weight_feature.

        Returns
        -------
        DataFrame or dict
            A DataFrame of feature name columns to stat value rows. Indexed
            by the stat type. The return type depends on the underlying client.
        """
        return self.client.get_marginal_stats(
            trainee_id=self.id,
            condition=condition,
            num_cases=num_cases,
            precision=precision,
            weight_feature=weight_feature
        )

    def react_into_features(
        self,
        *,
        distance_contribution: str | bool = False,
        familiarity_conviction_addition: str | bool = False,
        familiarity_conviction_removal: str | bool = False,
        features: Optional[Iterable[str]] = None,
        influence_weight_entropy: str | bool = False,
        p_value_of_addition: str | bool = False,
        p_value_of_removal: str | bool = False,
        similarity_conviction: str | bool = False,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None,
    ):
        """
        Calculate conviction and other data and stores them into features.

        Parameters
        ----------
        distance_contribution : bool or str, default False
            The name of the feature to store distance contribution.
            If set to True the values will be stored to the feature
            'distance_contribution'.
        familiarity_conviction_addition : bool or str, default False
            The name of the feature to store conviction of addition
            values. If set to True the values will be stored to the feature
            'familiarity_conviction_addition'.
        familiarity_conviction_removal : bool or str, default False
            The name of the feature to store conviction of removal
            values. If set to True the values will be stored to the feature
            'familiarity_conviction_removal'.
        features : iterable of str, optional
            A list of features to calculate convictions.
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
        use_case_weights : bool, default False
            When True, will scale influence weights by each case's
            ``weight_feature`` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        """
        if isinstance(self.client, AbstractHowsoClient):
            self.client.react_into_features(
                trainee_id=self.id,
                distance_contribution=distance_contribution,
                familiarity_conviction_addition=familiarity_conviction_addition,
                familiarity_conviction_removal=familiarity_conviction_removal,
                influence_weight_entropy=influence_weight_entropy,
                p_value_of_addition=p_value_of_addition,
                p_value_of_removal=p_value_of_removal,
                similarity_conviction=similarity_conviction,
                features=features,
                use_case_weights=use_case_weights,
                weight_feature=weight_feature,
            )
        else:
            raise ValueError("Client must have the 'react_into_features' method.")

    def react_into_trainee(
        self,
        *,
        use_case_weights: bool = False,
        action_feature: Optional[str] = None,
        context_features: Optional[Iterable[str]] = None,
        contributions: Optional[bool] = None,
        contributions_robust: Optional[bool] = None,
        hyperparameter_param_path: Optional[Iterable[str]] = None,
        mda: Optional[bool] = None,
        mda_permutation: Optional[bool] = None,
        mda_robust: Optional[bool] = None,
        mda_robust_permutation: Optional[bool] = None,
        num_robust_influence_samples: Optional[int] = None,
        num_robust_residual_samples: Optional[int] = None,
        num_robust_influence_samples_per_case: Optional[int] = None,
        num_samples: Optional[int] = None,
        residuals: Optional[bool] = None,
        residuals_robust: Optional[bool] = None,
        sample_model_fraction: Optional[float] = None,
        sub_model_size: Optional[int] = None,
        weight_feature: Optional[str] = None,
    ):
        """
        Compute and cache specified feature interpretations.

        Parameters
        ----------
        action_feature : str, optional
            Name of target feature for which to do computations. Default is
            whatever the model was analyzed for, e.g., action feature for MDA
            and contributions, or ".targetless" if analyzed for targetless.
            This parameter is required for MDA or contributions computations.
        context_features : iterable of str, optional
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
        hyperparameter_param_path : iterable of str, optional.
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
        """
        if isinstance(self.client, AbstractHowsoClient):
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
        else:
            raise ValueError("Client must have the 'react_into_trainee' method.")

    def get_feature_mda(
        self,
        action_feature: str,
        *,
        permutation: Optional[bool] = None,
        robust: Optional[bool] = None,
        weight_feature: Optional[str] = None,
    ) -> DataFrame:
        """
        Get cached feature Mean Decrease In Accuracy (MDA).

        All keyword arguments are optional, when not specified will auto-select
        cached MDA for output, when specified will attempt to
        output the cached MDA best matching the requested parameters,
        or None if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`get_prediction_stats` instead.

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
        DataFrame
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
    ) -> DataFrame:
        """
        Get cached feature contributions.

        All keyword arguments are optional, when not specified will auto-select
        cached contributions for output, when specified will attempt to
        output the cached contributions best matching the requested parameters,
        or None if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`get_prediction_stats` instead.

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
        DataFrame
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
        mode: Optional[Mode] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get the parameters used by the Trainee.

        If ``action_feature``, ``context_features``, ``mode``, or ``weight_feature``
        are specified, then the best hyperparameters analyzed in the Trainee are the
        value of the "hyperparameter_map" key, otherwise this value will be the
        dictionary containing all the hyperparameter sets in the Trainee.

        Parameters
        ----------
        action_feature : str, optional
            If specified will return the best analyzed hyperparameters to
            target this feature.
        context_features : iterable of str, optional
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
        dict of str -> any
            A dict including the either all of the Trainee's internal
            parameters or only the best hyperparameters selected using the
            passed parameters.
        """
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.get_params(
                self.id,
                action_feature=action_feature,
                context_features=context_features,
                mode=mode,
                weight_feature=weight_feature,
            )
        else:
            raise ValueError("Client must have the 'get_params' method.")

    def set_params(self, params: MutableMapping[str, Any]):
        """
        Set the workflow attributes for the trainee.

        Parameters
        ----------
        params : map of str -> any
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
        if isinstance(self.client, AbstractHowsoClient):
            self.client.set_params(self.id, params=params)
        else:
            raise ValueError("Client must have the 'set_params' method.")

    @property
    def client(self) -> AbstractHowsoClient | HowsoPandasClientMixin:
        """
        The client instance used by the trainee.

        Returns
        -------
        AbstractHowsoClient
            The client instance.
        """
        return self._client

    @client.setter
    def client(self, client: AbstractHowsoClient):
        """
        Set the client instance used by the trainee.

        Parameters
        ----------
        client : AbstractHowsoClient
            The client instance. Must be a subclass of :class:`AbstractHowsoClient`
            and :class:`HowsoPandasClientMixin`.
        """
        if not isinstance(client, AbstractHowsoClient):
            raise HowsoError(
                "``client`` must be a subclass of AbstractHowsoClient"
            )
        if not isinstance(client, HowsoPandasClientMixin):
            raise HowsoError("``client`` must be a HowsoPandasClient")
        self._client = client

    def _update_attributes(self, trainee: BaseTrainee):
        """
        Update the protected attributes of the trainee.

        Parameters
        ----------
        trainee : BaseTrainee
            The base trainee instance.
        """
        for key in self.attribute_map.keys():
            # Update the protected attributes directly since the values
            # have already been validated by the "BaseTrainee" instance
            # and to prevent triggering an API update call
            setattr(self, f"_{key}", getattr(trainee, key))

    def update(self):
        """Update the remote trainee with local state."""
        if (
            getattr(self, "id", None)
            and getattr(self, "_created", False)
            and not getattr(self, "_updating", False)
        ):
            # Only update for trainees that have been created
            try:
                self._updating = True
                trainee = BaseTrainee(**self.to_dict())
                if isinstance(self.client, AbstractHowsoClient):
                    updated_trainee = self.client.update_trainee(trainee)
                else:
                    raise ValueError("Client must have the 'update_trainee' method.")
                if updated_trainee:
                    self._update_attributes(updated_trainee)
            finally:
                self._updating = False

    def get_pairwise_distances(
        self,
        features: Optional[MutableMapping[str, MutableMapping]] = None,
        *,
        use_case_weights: bool = False,
        action_feature: Optional[str] = None,
        from_case_indices: Optional[CaseIndices] = None,
        from_values: Optional[TabularData2D] = None,
        to_case_indices: Optional[CaseIndices] = None,
        to_values: Optional[TabularData2D] = None,
        weight_feature: Optional[str] = None,
    ) -> List[float]:
        """
        Computes pairwise distances between specified cases.

        Returns a list of computed distances between each respective pair of
        cases specified in either ``from_values`` or ``from_case_indices`` to
        ``to_values`` or ``to_case_indices``. If only one case is specified in any
        of the lists, all respective distances are computed to/from that one
        case.

        .. NOTE::
            - One of ``from_values`` or ``from_case_indices`` must be specified,
              not both.
            - One of ``to_values`` or ``to_case_indices`` must be specified,
              not both.

        Parameters
        ----------
        features : list of str, optional
            List of feature names to use when computing pairwise distances.
            If unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this ``action_feature``, otherwise uses targetless
            hyperparameters.
        from_case_indices : iterable of (str, int), optional
            An Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of ``to_values`` or ``to_case_indices``.
        from_values : DataFrame or 2-dimensional list of object, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of ``to_values`` or ``to_case_indices``.
        to_case_indices : iterable of (str, int), optional
            An Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of ``from_values`` or ``from_case_indices``.
        to_values : DataFrame or 2-dimensional list of object, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of ``from_values`` or ``from_case_indices``.
        use_case_weights : bool, default False
            If set to True, will scale influence weights by each case's
            ``weight_feature`` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        list of float
            A list of computed pairwise distances between each corresponding
            pair of cases in ``from_case_indices`` and ``to_case_indices``.
        """
        if isinstance(self.client, AbstractHowsoClient):
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
        else:
            raise ValueError("Client must have the 'get_pairwise_distances' method.")

    def get_distances(
        self,
        features: Optional[Iterable[str]] = None,
        *,
        use_case_weights: bool = False,
        action_feature: Optional[str] = None,
        case_indices: Optional[CaseIndices] = None,
        feature_values: Optional[DataFrame | List[object]] = None,
        weight_feature: Optional[str] = None
    ) -> dict:
        """
        Computes distances matrix for specified cases.

        Returns a dict with computed distances between all cases
        specified in ``case_indices`` or from all cases in local model as defined
        by ``feature_values``.

        Parameters
        ----------
        features : iterable of str, optional
            List of feature names to use when computing distances. If
            unspecified uses all features.
        action_feature : str, optional
            The action feature. If specified, uses targeted hyperparameters
            used to predict this ``action_feature``, otherwise uses targetless
            hyperparameters.
        case_indices : iterable of (str, int), optional
            List of tuples, of session id and index, where index is the
            original 0-based index of the case as it was trained into the
            session. If specified, returns distances for all of these
            cases. Ignored if ``feature_values`` is provided. If neither
            ``feature_values`` nor ``case_indices`` is specified, uses full dataset.
        feature_values : DataFrame or list of object
            If specified, returns distances of the local model relative to
            these values, ignores ``case_indices`` parameter. If provided a
            DataFrame, only the first row will be used.
        use_case_weights : bool, default False
            If set to True, will scale influence weights by each case's
            ``weight_feature`` weight.
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
        features_to_code_map: MutableMapping[str, str],
        *,
        aggregation_code: Optional[str] = None,
    ) -> dict:
        r"""
        Evaluates custom code on feature values of all cases in the trainee.

        Parameters
        ----------
        features_to_code_map : map of str -> str
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
        if isinstance(self.client, AbstractHowsoClient):
            return self.client.evaluate(
                self.id,
                features_to_code_map=features_to_code_map,
                aggregation_code=aggregation_code,
            )
        else:
            raise ValueError("Client must have the 'evaluate' method.")

    def _create(
        self, *,
        library_type: Optional[Library] = None,
        max_wait_time: Optional[int | float] = None,
        resources: Optional[TraineeResources | MutableMapping[str, Any]] = None,
        overwrite: bool = False,
    ):
        """
        Create the trainee at the API.

        Parameters
        ----------
        library_type : {"mt", "st"}, optional
            The library type of the Trainee.
        max_wait_time : int or float, optional
            The maximum time to wait for the trainee to be created.
        resources : TraineeResources or map of str -> any, optional
            The resources to provision for the trainee.
        overwrite : bool, default False
            If True, will overwrite an existing trainee with the same name.
        """
        if not self.id:
            trainee = BaseTrainee(**self.to_dict())
            new_trainee = None
            if isinstance(self.client, AbstractHowsoClient):
                new_trainee = self.client.create_trainee(
                    trainee=trainee,
                    library_type=library_type,
                    max_wait_time=max_wait_time,
                    overwrite_trainee=overwrite,
                    resources=resources
                )

            if new_trainee:
                self._update_attributes(new_trainee)
            else:
                raise ValueError("Trainee is unable to be created")

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
        trainee_dict : dict
            The Trainee parameters.

        Returns
        -------
        Trainee
            The trainee instance.
        """
        if not isinstance(trainee_dict, dict):
            raise ValueError("``trainee_dict`` parameter is not a dict")
        parameters = {"client": trainee_dict.get("client")}
        for key in cls.attribute_map.keys():
            if key in trainee_dict:
                if key == "project_id":
                    parameters["project"] = trainee_dict[key]
                else:
                    parameters[key] = trainee_dict[key]

        return cls(**parameters)  # type: ignore

    def __enter__(self) -> "Trainee":
        """Support context managers."""
        self.acquire_resources()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup trainee during exit of context manager."""
        try:
            if self.persistence != "never" and (self.name or self._was_saved):
                self.release_resources()
            else:
                self.delete()
        except HowsoApiError as error:
            if error.status != 404:
                raise

    def get_contribution_matrix(
        self,
        features: Optional[Iterable[str]] = None,
        robust: bool = True,
        targeted: bool = False,
        normalize: bool = False,
        normalize_method: NormalizeMethod | Callable | Iterable[
            NormalizeMethod | Callable
        ] = "relative",
        absolute: bool = False,
        fill_diagonal: bool = True,
        fill_diagonal_value: float | int = 1,
    ) -> DataFrame:
        """
        Gets the Feature Contribution matrix.

        Parameters
        ----------
        features : iterable of str, optional
            An iterable of feature names. If features are not provided, then the
            default trainee features will be used.
        robust : bool, default True
            Whether to use robust calcuations.
        targeted : bool, default False
            Whether to do a targeted re-analyze before each feature's contribution is calculated.
        normalize : bool, default False
            Whether to normalize the matrix row wise. Normalization method is set by the ``normalize_method``
            parameter.
        normalize_method : str or callable or iterable of str or callable, default "relative"
            The normalization method. The method may either one of the strings below that correspond to a
            default method or a custom callable.

            These methods may be passed in as an individual string or in a iterable where they will
            be processed sequentially.

            Default Methods:
            - 'relative': normalizes each row by dividing each value by the maximum absolute value in the row.
            - 'fractional': normalizes each row by dividing each value by the sum of absolute values in the row.
            - 'feature_count': normalizes each row by dividing by the feature count.

            Custom Callable:
            - If a custom Callable is provided, then it will be passed onto the DataFrame apply function:
                ``matrix.apply(Callable)``
        absolute : bool, default False
            Whether to transform the matrix values into the absolute values.
        fill_diagonal : bool, default False
            Whether to fill in the diagonals of the matrix. If set to true,
            the diagonal values will be filled in based on the ``fill_diagonal_value`` value.
        fill_diagonal_value : bool, default 1
            The value to fill in the diagonals with. ``fill_diagonal`` must be set to True in order
            for the diagonal values to be filled in. If `fill_diagonal is set to false, then this
            parameter will be ignored.

        Returns
        -------
        Dataframe
            The Feature Contribution matrix in a Dataframe.
        """
        feature_contribution_matrix = {}
        if not features:
            features = self.features
        for feature in features:
            if targeted:
                context_features = [context_feature for context_feature in features if context_feature != feature]
                self.analyze(action_features=[feature], context_features=context_features)
            # Suppresses expected warnings when trainee is targetless
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Results may be inaccurate because trainee has not been analyzed*",
                    category=HowsoWarning
                )
                if robust:
                    self.react_into_trainee(action_feature=feature, contributions_robust=True)
                else:
                    self.react_into_trainee(action_feature=feature, contributions=True)

            feature_contribution_matrix[feature] = self.get_prediction_stats(
                action_feature=feature,
                robust=robust,
                stats=['contribution']
            )

        matrix = concat(feature_contribution_matrix.values(), keys=feature_contribution_matrix.keys())
        matrix = matrix.droplevel(level=1)
        # Stores the preprocessed matrix, useful if the user wants a different form of processing
        # after calculation.
        self._calculated_matrices['contribution'] = deepcopy(matrix)
        matrix = matrix_processing(
            matrix,
            normalize=normalize,
            normalize_method=normalize_method,
            absolute=absolute,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )

        return matrix

    def get_mda_matrix(
        self,
        features: Optional[Iterable[str]] = None,
        robust: bool = True,
        targeted: bool = False,
        normalize: bool = False,
        normalize_method: NormalizeMethod | Callable | Iterable[
            NormalizeMethod | Callable
        ] = "relative",
        absolute: bool = False,
        fill_diagonal: bool = True,
        fill_diagonal_value: float | int = 1,
    ) -> DataFrame:
        """
        Gets the Mean Decrease in Accuracy (MDA) matrix.

        Parameters
        ----------
        features : iterable of str, optional
            An iterable of feature names. If features are not provided, then the default trainee
            features will be used.
        robust : bool, default True
            Whether to use robust calcuations.
        targeted : bool, default False
            Whether to do a targeted re-analyze before each feature's contribution is calculated.
        normalize : bool, default False
            Whether to normalize the matrix row wise. Normalization method is set by the ``normalize_method``
            parameter.
        normalize_method : str or callable or iterable of str or callable, default "relative"
            The normalization method. The method may either one of the strings below that correspond to a
            default method or a custom callable.

            These methods may be passed in as an individual string or in a iterable where they will
            be processed sequentially.

            Default Methods:
            - 'relative': normalizes each row by dividing each value by the maximum absolute value in the row.
            - 'fractional': normalizes each row by dividing each value by the sum of absolute values in the row.
            - 'feature_count': normalizes each row by dividing by the feature count.

            Custom Callable:
            - If a custom Callable is provided, then it will be passed onto the DataFrame apply function:
                ``matrix.apply(Callable)``
        absolute : bool, default False
            Whether to transform the matrix values into the absolute values.
        fill_diagonal : bool, default False
            Whether to fill in the diagonals of the matrix. If set to true,
            the diagonal values will be filled in based on the ``fill_diagonal_value`` value.
        fill_diagonal_value : bool, default 1
            The value to fill in the diagonals with. ``fill_diagonal`` must be set to True in order
            for the diagonal values to be filled in. If `fill_diagonal is set to false, then this
            parameter will be ignored.

        Returns
        -------
        Dataframe
            The MDA matrix in a Dataframe.
        """
        mda_matrix = {}
        if not features:
            features = self.features
        for feature in features:
            if targeted:
                context_features = [context_feature for context_feature in features if context_feature != feature]
                self.analyze(action_features=[feature], context_features=context_features)
            # Suppresses expected warnings when trainee is targetless
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Results may be inaccurate because trainee has not been analyzed*",
                    category=HowsoWarning
                )
                if robust:
                    self.react_into_trainee(action_feature=feature, mda_robust=True)
                else:
                    self.react_into_trainee(action_feature=feature, mda=True)

            mda_matrix[feature] = self.get_prediction_stats(
                action_feature=feature,
                robust=robust,
                stats=['mda']
            )

        matrix = concat(mda_matrix.values(), keys=mda_matrix.keys())
        matrix = matrix.droplevel(level=1)
        # Stores the preprocessed matrix, useful if the user wants a different form of processing
        # after calculation.
        self._calculated_matrices['mda'] = deepcopy(matrix)
        matrix = matrix_processing(
            matrix,
            normalize=normalize,
            normalize_method=normalize_method,
            absolute=absolute,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )

        return matrix


def delete_trainee(
    name_or_id: Optional[str] = None,
    file_path: Optional[PathLike] = None,
    client: Optional[AbstractHowsoClient] = None
):
    """
    Delete an existing Trainee.

    Loaded trainees exist in memory while also potentially existing on disk. This is a convenience function that
    allows the deletion of Trainees from both memory and disk.

    Parameters
    ----------
    name_or_id : str, optional
        The name or id of the trainee. Deletes the Trainees from memory and attempts to delete a Trainee saved under
        the same filename from the default save location if no ``file_path`` is provided.
    file_path : str or bytes or os.PathLike, optional
        The path of the file to load the Trainee from. Used for deleting trainees from disk.

        The file path must end with a filename, but file path can be either an absolute path, a
        relative path or just the file name.

        If ``name_or_id`` is not provided, in addition to deleting from disk, will attempt to
        delete a Trainee from memory assuming the Trainee has the same name as the filename.

        If ``file_path`` is a relative path the absolute path will be computed
        appending the ``file_path`` to the CWD.

        If ``file_path`` is an absolute path, this is the absolute path that
        will be used.

        If ``file_path`` is just a filename, then the absolute path will be computed
        appending the filename to the CWD.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.
    """
    client = client or get_client()
    if name_or_id is None and file_path is None:
        raise ValueError(
            'Either the ``name_or_id`` or the ``file_path`` must be provided.'
        )

    # Check if file exists
    if file_path:
        if not isinstance(client, LocalSaveableProtocol):
            raise HowsoError(
                "Deleting trainees from using a file path is only"
                "supported with a client that has disk access.")

        file_path = Path(file_path)
        file_path = file_path.expanduser().resolve()
        if not file_path.exists():
            raise ValueError(f"File '{file_path}' does not exist.")

        client.delete_trainee(trainee_id=str(name_or_id), file_path=file_path)
    else:
        client.delete_trainee(trainee_id=str(name_or_id))


def load_trainee(
    file_path: PathLike,
    client: Optional[AbstractHowsoClient] = None
) -> "Trainee":
    """
    Load an existing trainee from disk.

    Parameters
    ----------
    file_path : str or bytes or os.PathLike
        The path of the file to load the Trainee from. This path can contain
        an absolute path, a relative path or simply a file name. A ``.caml`` file name
        must be always be provided if file paths are provided.

        If ``file_path`` is a relative path the absolute path will be computed
        appending the ``file_path`` to the CWD.

        If ``file_path`` is an absolute path, this is the absolute path that
        will be used.

        If ``file_path`` is just a filename, then the absolute path will be computed
        appending the filename to the CWD.

    client : AbstractHowsoClient, optional
        The Howso client instance to use. Must have local disk access.

    Returns
    -------
    Trainee
        The trainee instance.
    """
    client = client or get_client()

    if not isinstance(client, LocalSaveableProtocol):
        raise HowsoError("To save, ``client`` must have local disk access.")

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

    if isinstance(client, LocalSaveableProtocol):
        trainee = client._get_trainee_from_core(trainee_id)
    else:
        raise ValueError("Loading a Trainee from disk requires a client with disk access.")
    if isinstance(client.trainee_cache, TraineeCache):
        client.trainee_cache.set(trainee)
    if trainee:
        trainee = Trainee.from_openapi(trainee, client=client)
    else:
        raise ValueError("Trainee not loaded correctly.")
    trainee._custom_save_path = file_path

    return trainee


def get_trainee(
    name_or_id: str,
    *,
    client: Optional[AbstractHowsoClient] = None
) -> "Trainee" | None:
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
    Trainee or None
        The trainee instance or None if a trainee with the specified name/id was not found.
    """
    client = client or get_client()
    trainee = client.get_trainee(str(name_or_id))
    if trainee:
        return Trainee.from_openapi(trainee, client=client)


def list_trainees(
    search_terms: Optional[str] = None,
    *,
    client: Optional[AbstractHowsoClient] = None,
    project: Optional[str | BaseProject] = None,
) -> List["TraineeIdentity"]:
    """
    Get listing of available trainees.

    This method only returns a simplified informational listing of available
    trainees, not full engine Trainee instances. To get a Trainee instance
    that can be used with the engine API call ``get_trainee``.

    Parameters
    ----------
    search_terms : str, optional
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

    # picks up base
    return client.get_trainees(**params)


def get_hierarchy(self) -> Dict:
    """
    Output the hierarchy for a trainee.

    Returns
    -------
    dict of {str: dict}
        Dictionary of the currently contained hierarchy as a nested dict
        with False for trainees that are stored independently.
    """
    return self.client.get_hierarchy(self.id)


def rename_subtrainee(
    self,
    new_name: str,
    *,
    child_id: Optional[str] = None,
    child_name_path: Optional[List[str]] = None
) -> None:
    """
    Renames a contained child trainee in the hierarchy.

    Parameters
    ----------
    new_name : str
        New name of child trainee
    child_id : str, optional
        Unique id of child trainee to rename. Ignored if child_name_path is
        specified.
    child_name_path : list of str, optional
        List of strings specifying the user-friendly path of the child
        subtrainee to rename.
    """
    self.client.rename_subtrainee(
        self.id,
        child_name_path=child_name_path,
        child_id=child_id,
        new_name=new_name
    )
