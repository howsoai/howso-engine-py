import logging
from pathlib import Path
import platform
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import uuid
import warnings

from amalgam.api import Amalgam
from howso.client.exceptions import HowsoError, HowsoWarning
from howso.utilities.internals import sanitize_for_json
import howso.utilities.json_wrapper as json
import six

_logger = logging.getLogger('howso.direct')

# Position under home directory of downloaded amalgam files
core_lib_dirname = ".howso/lib/core/"
amlg_lib_dirname = ".howso/lib/amlg/"


DEFAULT_CORE_PATH = Path(__file__).parent.parent.joinpath("howso-engine")


class HowsoCore:
    """
    Howso Core API.

    This class is used in conjunction with the Amalgam python interface to
    interact with the Howso Core and Amalgam binaries.

    Parameters
    ----------
    handle : str
        Handle for the Howso entity. If none is provided a random 6 digit
        alphanumeric handle will be assigned.
    library_path : str, optional
        Path to Amalgam library.
    gc_interval : int, default 100
        Number of Amalgam operations to perform before forcing garbage collection.
        Lower is better at memory management but compromises performance.
        Higher is better performance but may result in higher memory usage.
    howso_path : str, default `DEFAULT_CORE_PATH`
        Directory path to the Howso caml files.
    howso_fname : str, default "howso.caml"
        Name of the Howso caml file with extension.
    trace: bool, default False
        If true, sets debug flag for amlg operations. This will generate an
        execution trace useful in debugging with the standard name of
        howso_[random 6 byte hex]_execution.trace.
    sbf_datastore_enabled : bool, default True
        If true, sbf tree structures are enabled.
    max_num_threads : int, default 0
        If a multithreaded Amalgam binary is used, sets the maximum number of
        threads to the value specified. If 0, will use the number of visible
        logical cores.
    """

    PRIMITIVE_TYPES = (float, bool, bytes, six.text_type) + six.integer_types

    def __init__(  # noqa: C901
        self,
        library_path: Optional[str] = None,
        gc_interval: int = 100,
        howso_path: Path = DEFAULT_CORE_PATH,
        howso_fname: str = "howso.caml",
        trace: bool = False,
        sbf_datastore_enabled: bool = True,
        max_num_threads: int = 0,
        **kwargs
    ):
        if kwargs.get("amlg_debug", None) is not None:
            if trace is None:
                trace = kwargs["amlg_debug"]
            _logger.warning(
                'The "amlg_debug" parameter is deprecated use "trace" instead.')

        self.trace = bool(trace)

        self.trace_filename = f"howso_{self.random_handle()}_execution.trace"

        # The parameters to pass to the Amalgam object - compiled here, so that
        # they can be merged with config file params.
        amlg_params = {
            'library_path': library_path,
            'gc_interval': gc_interval,
            'sbf_datastore_enabled': sbf_datastore_enabled,
            'max_num_threads': max_num_threads,
            'trace': self.trace,
            'execution_trace_file': self.trace_filename,
        }

        if amalgam_opts := kwargs.get("amalgam", {}):
            # merge parameters from config.yml - favoring the configured params
            if amlg_params_intersection := amlg_params.keys() & amalgam_opts.keys():
                # Warn that there are conflicts
                _logger.warning(
                    "The following parameters from configuration file will "
                    "override the Amalgam parameters set in the code: " +
                    str(amlg_params_intersection)
                )
        amlg_params = {**amlg_params, **(amalgam_opts or {})}

        # Infer the os/arch from the running platform, unless set in config
        operating_system = amlg_params.setdefault(
            'os', platform.system().lower())
        if operating_system == 'windows':
            library_file_extension = "dll"
        elif operating_system == 'darwin':
            library_file_extension = "dylib"
        else:
            library_file_extension = "so"

        # Assemble the library file name - use multithreaded library by default
        library_postfix = amlg_params.get('library_postfix', '-mt')

        # Infer the architecture unless set, and normalize
        architecture = amlg_params.get('arch', platform.machine().lower())
        if architecture in ['x86_64', 'amd64']:
            architecture = 'amd64'
        elif architecture in ['aarch64_be', 'aarch64', 'armv8b', 'armv8l']:
            # see: https://stackoverflow.com/questions/45125516/possible-values-for-uname-m
            architecture = 'arm64'
        elif architecture == 'arm64_8a':
            # TODO 17132: 8a arm arch is a special case and not currently auto
            # selected by this routine. So if the user specifies it, use it as
            # is. Future work will auto select this based on env.
            pass

        # If download set, try and download the specified version using
        # howso-build-artifacts
        elif amlg_params.get('download'):
            # Download amalgam (unless already there) - and get the path
            amalgam_download_dir = self.download_amlg(amlg_params)
            amlg_params['library_path'] = str(Path(
                amalgam_download_dir, 'lib',
                f"amalgam{library_postfix}.{library_file_extension}"
            ))
            _logger.debug(
                'Using downloaded amalgam location: '
                f'{amlg_params["library_path"]}')

        # If version is set, but download not, use the default download location
        elif amlg_version := amlg_params.get('version'):
            versioned_amlg_location = Path(
                Path.home(), amlg_lib_dirname, operating_system,
                architecture, amlg_version, 'lib',
                f"amalgam{library_postfix}.{library_file_extension}"
            )
            if versioned_amlg_location.exists():
                amlg_params['library_path'] = str(versioned_amlg_location)
                _logger.debug(
                    'Using amalgam version located at: '
                    f'{amlg_params["library_path"]}')
            else:
                raise HowsoError(
                    f'No amalgam library found at {versioned_amlg_location}')

        # Using the defaults
        else:
            _logger.debug(
                'Using default amalgam location: '
                f'{amlg_params["library_path"]}')

        # Filter out invalid amlg_params, and instantiate.
        amlg_params = {
            k: v for k, v in amlg_params.items()
            if k in [
                'library_path', 'gc_interval', 'sbf_datastore_enabled',
                'max_num_threads', 'debug', 'trace', 'execution_trace_file',
                'execution_trace_dir', 'library_postfix', 'arch'
            ]
        }
        self.amlg = Amalgam(**amlg_params)

        core_params = kwargs.get('core') or {}

        # If download, then retrieve using howso-build-artifacts
        if core_params.get('download', False):
            self.howso_path = Path(
                self.download_core(core_params)).expanduser()
            self.default_save_path = Path(self.howso_path, 'trainee')

        # If version is set, but download not, use the default download location
        elif version := core_params.get('version'):
            # Set paths, ensuring tailing slash
            self.howso_path = Path(Path.home(), core_lib_dirname, version)
            self.default_save_path = Path(self.howso_path, "trainee")

        # .... otherwise use default locations
        else:
            # Set paths, ensuring tailing slash
            self.howso_path = Path(howso_path).expanduser()
            self.default_save_path = Path(self.howso_path, "trainee")

        # Allow for trainee save directory to be overridden
        if persisted_trainees_dir := core_params.get('persisted_trainees_dir'):
            self.default_save_path = Path(persisted_trainees_dir).expanduser()
            _logger.debug(
                'Trainee save directory has been overridden to '
                f'{self.default_save_path}')
        else:
            # If no specific location provided, use current working directory.
            self.default_save_path = Path.cwd()

        # make save dir if doesn't exist
        if not self.default_save_path.exists():
            self.default_save_path.mkdir(parents=True)

        self.howso_fname = howso_fname
        self.ext = howso_fname[howso_fname.rindex('.'):]

        self.howso_fully_qualified_path = Path(
            self.howso_path, self.howso_fname)
        if not self.howso_fully_qualified_path.exists():
            raise HowsoError(
                f'Howso core file {self.howso_fully_qualified_path} '
                'does not exist')
        _logger.debug(
            'Using howso-core location: '
            f'{self.howso_fully_qualified_path}')

    @staticmethod
    def random_handle() -> str:
        """
        Generate a random 6 byte hexadecimal handle.

        Returns
        -------
        str
            A random 6 byte hex.
        """
        try:
            # Use of secrets/uuid must be used instead of the "random" package
            # as they will not be affected by setting random.seed which could
            # cause duplicate handles to be generated.
            import secrets
            return secrets.token_hex(6)
        except (ImportError, NotImplementedError):
            # Fallback to uuid if operating system does not support secrets
            return uuid.uuid4().hex[-12:]

    def __str__(self) -> str:
        """Return a string representation of the HowsoCore object."""
        template = (
            "Howso Path:\t\t %s%s\n "
            "Save Path:\t\t %s\n")
        return template % (
            self.howso_path,
            self.howso_fname,
            self.default_save_path,
        )

    def get_trainee_version(
        self,
        trainee_id: str,
    ) -> str:
        """
        Return the version of the Trainee Template.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to get the version of.
        """
        return self._execute(trainee_id, "get_trainee_version", {})

    def create_trainee(self, trainee_id: str) -> Union[Dict, None]:
        """
        Create a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to create.

        Returns
        -------
        dict
            A dict containing the name of the trainee that was created.
        """
        status = self.amlg.load_entity(
            handle=trainee_id,
            amlg_path=str(self.howso_fully_qualified_path),
            persist=False,
            load_contained=True,
            escape_filename=False,
            escape_contained_filenames=False
        )
        self._execute(trainee_id, "initialize", {
            "trainee_id": trainee_id,
            "filepath": str(self.howso_path) + '/',
        })
        if not status.loaded:
            raise HowsoError("Error loading the Trainee.")
        return {"name": trainee_id}

    def get_loaded_trainees(self) -> List[str]:
        """
        Get loaded Trainees.

        Returns
        -------
        list of str
            A list of trainee identifiers that are currently loaded.
        """
        return self.get_entities()

    def get_entities(self) -> List[str]:
        """
        Get loaded entities.

        Returns
        -------
        list of str
            A list of entity identifiers that are currently loaded.
        """
        return self.amlg.get_entities()

    def load(
        self,
        trainee_id: str,
        filename: Optional[str] = None,
        filepath: Optional[str] = None,
    ) -> Union[Dict, None]:
        """
        Load a persisted Trainee from disk.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to load.
        filename : str, optional
            The filename to load.
        filepath : str, optional
            The path containing the filename to load.

        Returns
        -------
        dict
            A dict containing the name of the trainee that was created.
        """
        filename = trainee_id if filename is None else filename
        filepath = f"{self.default_save_path}/" if filepath is None else filepath

        status = self.amlg.load_entity(
            handle=trainee_id,
            amlg_path=str(Path(filepath, filename)) + self.ext,
            persist=False,
            load_contained=True,
            escape_filename=False,
            escape_contained_filenames=False,
        )
        if not status.loaded:
            raise HowsoError("Failed to load trainee.")
        return {"name": trainee_id}

    def persist(
        self,
        trainee_id: str,
        filename: Optional[str] = None,
        filepath: Optional[str] = None
    ) -> None:
        """
        Save a Trainee to disk.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to save.
        filename : str, optional
            The name of the file to save the Trainee to.
        filepath : str, optional
            The path of the file to save the Trainee to.
        """
        filename = trainee_id if filename is None else filename
        filepath = (
            f"{self.default_save_path}/" if filepath is None else filepath)

        self.amlg.store_entity(
            handle=trainee_id,
            amlg_path=str(Path(filepath, filename)) + self.ext
        )

    def delete(self, trainee_id: str) -> None:
        """
        Delete a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to delete.
        """
        return self.amlg.destroy_entity(trainee_id)

    def copy(self, trainee_id: str, target_trainee_id: str) -> Dict:
        """
        Copy the contents of one Trainee into another.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to copy from.
        target_trainee_id : str
            The identifier of the Trainee to copy into.

        Returns
        -------
        dict
            A dict containing the name of the trainee that was created by copy.
        """
        cloned_successfully = self.amlg.clone_entity(
            handle=trainee_id,
            clone_handle=target_trainee_id,
        )

        if not cloned_successfully:
            raise HowsoError("Cloning was unsuccessful.")
        return {'name': target_trainee_id}

    def copy_subtrainee(
        self,
        trainee_id: str,
        new_trainee_name: str,
        source_id: Optional[str] = None,
        source_name_path: Optional[List[str]] = None,
        target_id: Optional[str] = None,
        target_name_path: Optional[List[str]] = None,
    ) -> None:
        """
        Copy a subtrainee in trainee's hierarchy.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee whose hierarchy is to be modified.
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
        return self._execute(trainee_id, "copy_subtrainee", {
            "target_trainee": new_trainee_name,
            "source_id": source_id,
            "source_name_path": source_name_path,
            "target_id": target_id,
            "target_name_path": target_name_path
        })

    def delete_subtrainee(
        self,
        trainee_id: str,
        trainee_name: str
    ) -> None:
        """
        Delete a child subtrainee.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee whose hierarchy is to be modified.
        trainee_name: str
            The name of the subtrainee to be deleted.
        """
        return self._execute(trainee_id, "delete_subtrainee", {
            "trainee": trainee_name,
        })

    def load_subtrainee(
        self,
        trainee_id: str,
        *,
        filename: Optional[str] = None,
        filepath: Optional[str] = None,
        trainee_name_path: Optional[List[str]] = None,
    ) -> Union[Dict, None]:
        """
        Load a persisted Trainee from disk as a subtrainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to be modified.
        filename : str, optional
            The filename to load.
        filepath : str, optional
            The path containing the filename to load.
        trainee_name_path: list of str, optional
            list of strings specifying the user-friendly path of the child
            subtrainee to load.

        Returns
        -------
        dict
            A dict containing the name of the trainee that was created.
        """
        filename = trainee_id if filename is None else filename
        filepath = (
            f"{self.default_save_path}/" if filepath is None else filepath)

        return self._execute(trainee_id, "load_subtrainee", {
            "trainee": trainee_name_path,
            "filename": filename,
            "filepath": filepath
        })

    def save_subtrainee(
        self,
        trainee_id: str,
        *,
        filename: Optional[str] = None,
        filepath: Optional[str] = None,
        subtrainee_id: Optional[str] = None,
        trainee_name_path: Optional[List[str]] = None
    ) -> None:
        """
        Save a subtrainee to disk.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to be modified.
        filename : str, optional
            The name of the file to save the Trainee to.
        filepath : str, optional
            The path of the file to save the Trainee to.
        subtrainee_id: str, optional
            Unique id for subtrainee. Must be provided if subtrainee does not
            have one already specified.
        trainee_name_path: list of str, optional
            list of strings specifying the user-friendly path of the child
            subtrainee to save.
        """
        filename = trainee_id if filename is None else filename
        filepath = (
            f"{self.default_save_path}/" if filepath is None else filepath)

        return self._execute(trainee_id, "save_subtrainee", {
            "trainee": trainee_name_path,
            "trainee_id": subtrainee_id,
            "filename": filename,
            "filepath": filepath
        })

    def create_subtrainee(
            self,
            trainee_id: str,
            trainee_name: str,
            subtrainee_id: Optional[str] = None
    ) -> Union[Dict, None]:
        """
        Create a subtrainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to be modified.
        trainee_name: str
            Name of subtrainee to create.
        subtrainee_id: str, optional
            Unique id for subtrainee.

        Returns
        -------
        dict
            A dict containing the name of the subtrainee that was created.
        """
        return self._execute(trainee_id, "create_subtrainee", {
            "trainee": trainee_name,
            "trainee_id": subtrainee_id
        })

    def remove_series_store(self, trainee_id: str, series: Optional[str] = None
                            ) -> None:
        """
        Delete part or all of the series store from a Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to delete the series store from.
        series : str, optional
            The ID of the series to remove from the series store.
            If None, the entire series store will be deleted.
        """
        return self._execute(trainee_id, "remove_series_store", {
            "series": series,
        })

    def clean_data(
        self,
        trainee_id: str,
        context_features: Optional[Iterable[str]] = None,
        action_features: Optional[Iterable[str]] = None,
        remove_duplicates: Optional[bool] = None
    ) -> None:
        """
        Cleans up Trainee data.

        Removes unused sessions, cases or actions missing data, etc. If a
        trainee identifier is not specified, it will look to the entity's own
        label of the same name.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to clean.
        context_features : list of str, optional
            Only remove cases that don't have specified context features.
        action_features : list of str, optional
            Only remove cases that don't have specified action features.
        remove_duplicates : bool, default False
            If true, will remove duplicate cases (cases with identical values).
        """
        return self._execute(trainee_id, "clean_data", {
            "context_features": context_features,
            "action_features": action_features,
            "remove_duplicates": remove_duplicates,
        })

    def set_substitute_feature_values(
        self,
        trainee_id: str,
        substitution_value_map: Union[Dict, None]
    ) -> None:
        """
        Set substitution feature values used in case generation.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        substitution_value_map : dict or None
            A dictionary of feature name to value to substitution value. If the
            map is None, all substitutions will be disabled and cleared.
        """
        return self._execute(trainee_id, "set_substitute_feature_values",{
            "substitution_value_map": substitution_value_map,
        })

    def get_substitute_feature_values(self, trainee_id: str) -> Dict:
        """
        Get substitution feature values used in case generation.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        dict
            The dictionary of feature name to value to substitution value.
        """
        return self._execute(trainee_id, "get_substitute_feature_values", {})

    def set_session_metadata(
        self,
        trainee_id: str,
        session: str,
        metadata: Optional[Dict]
    ) -> None:
        """
        Set the Trainee session metadata.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        session : str
            The identifier of the Trainee session.
        metadata : dict
            The metadata to associate to the session.
        """
        return self._execute(trainee_id, "set_session_metadata", {
            "session": session,
            "metadata": metadata,
        })

    def get_session_metadata(self, trainee_id: str, session: str
                             ) -> Union[Dict, None]:
        """
        Get the Trainee session metadata.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        session : str
            The identifier of the Trainee session.

        Returns
        -------
        dict or None
            The metadata of the session. Or None if no metadata set.
        """
        return self._execute(trainee_id, "get_session_metadata", {
            "session": session,
        })

    def get_sessions(self, trainee_id: str, attributes: Optional[List[str]]
                     ) -> List[Dict]:
        """
        Get list of session names.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        attributes : list of str, optional
            List of attributes to return from the session. The session id is
            always included.

        Returns
        -------
        list of dict
            The list of Trainee sessions.
        """
        return self._execute(trainee_id, "get_sessions", {
            "attributes": attributes,
        })

    def remove_session(self, trainee_id: str, session: str) -> None:
        """
        Remove a Trainee session.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        session : str
            The identifier of the Trainee session.
        """
        return self._execute(trainee_id, "remove_session",{
            "session": session,
        })

    def remove_feature(
        self,
        trainee_id: str,
        feature: str,
        *,
        condition: Optional[Dict] = None,
        condition_session: Optional[str] = None,
        session: Optional[str] = None
    ) -> None:
        """
        Remove a feature.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        feature : str
            The feature name.
        condition : str, optional
            A condition map where features will only be removed when certain
            criteria is met.
        condition_session : str optional
            If specified, ignores the condition parameter and operates on cases
            for the specified session id.
        session : str, optional
            The identifier of the Trainee session to associate the feature
            removal with.
        """
        return self._execute(trainee_id, "remove_feature", {
            "feature": feature,
            "condition": condition,
            "session": session,
            "condition_session": condition_session,
        })

    def add_feature(
        self,
        trainee_id: str,
        feature: str,
        feature_value: Optional[Union[int, float, str]] = None,
        *,
        condition: Optional[Dict] = None,
        condition_session: Optional[str] = None,
        feature_attributes: Optional[Dict] = None,
        overwrite: bool = False,
        session: Optional[str] = None,
    ) -> None:
        """
        Add a feature.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        feature : str
            The feature name.
        feature_value : int or float or str, optional
            The feature value.
        condition : str, optional
            A condition map where features will only be removed when certain
            criteria is met.
        condition_session : str optional
            If specified, ignores the condition parameter and operates on cases
            for the specified session id.
        overwrite : bool, default False
            If True, the feature will be over-written if it exists.
        session : str, optional
            The identifier of the Trainee session to associate the feature
            addition with.
        """
        return self._execute(trainee_id, "add_feature", {
            "feature": feature,
            "feature_value": feature_value,
            "overwrite": overwrite,
            "condition": condition,
            "feature_attributes": feature_attributes,
            "session": session,
            "condition_session": condition_session,
        })

    def get_num_training_cases(self, trainee_id: str) -> Dict:
        """
        Return the number of trained cases in the model.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        dict
            A dictionary containing the key "count".
        """
        return self._execute(trainee_id, "get_num_training_cases", {})

    def get_auto_ablation_params(self, trainee_id: str):
        """
        Get trainee parameters for auto ablation set by :meth:`set_auto_ablation_params`.
        """
        return self._execute(
            trainee_id, "get_auto_ablation_params", {}
        )

    def set_auto_ablation_params(
        self,
        trainee_id: str,
        auto_ablation_enabled: bool = False,
        *,
        auto_ablation_weight_feature: str = ".case_weight",
        conviction_lower_threshold: Optional[float] = None,
        conviction_upper_threshold: Optional[float] = None,
        exact_prediction_features: Optional[List[str]] = None,
        influence_weight_entropy_threshold: float = 0.6,
        minimum_model_size: int = 1_000,
        relative_prediction_threshold_map: Optional[Dict[str, float]] = None,
        residual_prediction_features: Optional[List[str]] = None,
        tolerance_prediction_threshold_map: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs
    ):
        """
        Set trainee parameters for auto ablation.

        .. note::
            Auto-ablation is experimental and the API may change without deprecation.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set auto ablation parameters for.
        auto_ablation_enabled : bool, default False
            When True, the :meth:`train` method will ablate cases that meet the set criteria.
        auto_ablation_weight_feature : str, default ".case_weight"
            The weight feature that should be accumulated to when cases are ablated.
        minimum_model_size : int, default 1,000
            The threshold ofr the minimum number of cases at which the model should auto-ablate.
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
        return self._execute(
            trainee_id, "set_auto_ablation_params",
            {
                "auto_ablation_enabled": auto_ablation_enabled,
                "auto_ablation_weight_feature": auto_ablation_weight_feature,
                "minimum_model_size": minimum_model_size,
                "influence_weight_entropy_threshold": influence_weight_entropy_threshold,
                "exact_prediction_features": exact_prediction_features,
                "residual_prediction_features": residual_prediction_features,
                "tolerance_prediction_threshold_map": tolerance_prediction_threshold_map,
                "relative_prediction_threshold_map": relative_prediction_threshold_map,
                "conviction_lower_threshold": conviction_lower_threshold,
                "conviction_upper_threshold": conviction_upper_threshold,
            }
        )

    def auto_analyze_params(
        self,
        trainee_id: str,
        auto_analyze_enabled: bool = False,
        analyze_threshold: Optional[int] = None,
        auto_analyze_limit_size: Optional[int] = None,
        analyze_growth_factor: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Set trainee parameters for auto analysis.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        auto_analyze_enabled : bool, default False
            Enable auto analyze when training. Train will return a status
            indicating when to auto analyze.
        analyze_threshold : int, optional
            The threshold for the number of cases at which the model should be
            re-analyzed.
        auto_analyze_limit_size : int, optional
            The size of of the model at which to stop doing auto-analysis.
            Value of 0 means no limit.
        analyze_growth_factor : float, optional
            The factor by which to increase the analyze threshold every time
            the model grows to the current threshold size.
        kwargs : dict, optional
            Parameters specific for analyze() may be passed in via kwargs, and
            will be cached and used during future auto-analysis.
        """
        params = {
            "auto_analyze_enabled": auto_analyze_enabled,
            "analyze_threshold": analyze_threshold,
            "analyze_growth_factor": analyze_growth_factor,
            "auto_analyze_limit_size": auto_analyze_limit_size,
        }
        return self._execute(trainee_id, "set_auto_analyze_params", {**kwargs, **params})

    def auto_analyze(self, trainee_id: str) -> None:
        """
        Auto-analyze the Trainee model.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        """
        return self._execute(trainee_id, "auto_analyze", {})

    def compute_feature_weights(
        self,
        trainee_id: str,
        action_feature: Optional[str] = None,
        context_features: Optional[Iterable[str]] = None,
        robust: bool = False,
        weight_feature: Optional[str] = None,
        use_case_weights: bool = False
    ) -> Dict:
        """
        Compute feature weights for specified context and action features.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        action_feature : str, optional
            Action feature for which to set the specified feature weights for.
        context_features: iterable of str
            List of context feature names.
        robust : bool, default False.
            When true, the power set/permutations of features are used as
            contexts to calculate the residual for a given feature. When
            false, the full set of features is used to calculate the
            residual for a given feature.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        use_case_weights : bool, default False
            If set to True will scale influence weights by each
            case's weight_feature weight.

        Returns
        -------
        dict
            A dictionary of computed context features -> weights.
        """
        return self._execute(trainee_id, "compute_feature_weights", {
            "action_feature": action_feature,
            "context_features": context_features,
            "robust": robust,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })

    def clear_conviction_thresholds(self, trainee_id: str) -> None:
        """
        Set the conviction thresholds to null.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        """
        return self._execute(trainee_id, "clear_conviction_thresholds", {})

    def set_conviction_lower_threshold(self, trainee_id: str, threshold: float
                                       ) -> None:
        """
        Set the conviction lower threshold.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        threshold : float
            The threshold value.
        """
        return self._execute(trainee_id, "set_conviction_lower_threshold", {
            "conviction_lower_threshold": threshold,
        })

    def set_conviction_upper_threshold(self, trainee_id: str, threshold: float
                                       ) -> None:
        """
        Set the conviction upper threshold.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        threshold : float
            The threshold value.
        """
        return self._execute(trainee_id, "set_conviction_upper_threshold", {
            "conviction_upper_threshold": threshold,
        })

    def set_metadata(self, trainee_id: str, metadata: Union[Dict, None]
                     ) -> None:
        """
        Set trainee metadata.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        metadata : dict or None
            The metadata dictionary.
        """
        return self._execute(trainee_id, "set_metadata", {
            "metadata": metadata,
        })

    def get_metadata(self, trainee_id: str) -> Union[Dict, None]:
        """
        Get trainee metadata.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        dict or None
            The metadata dictionary.
        """
        return self._execute(trainee_id, "get_metadata", {})

    def retrieve_extreme_cases_for_feature(
        self,
        trainee_id: str,
        num: int,
        sort_feature: str,
        features: Optional[Iterable[str]] = None
    ) -> Dict:
        """
        Gets the extreme cases of a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        num : int
            The number of cases to get.
        sort_feature : str
            The feature name by which extreme cases are sorted by.
        features: iterable of str, optional
            An iterable of feature names to use when getting extreme cases.

        Returns
        -------
        dict
            A dictionary of keys 'cases' and 'features'.
        """
        return self._execute(trainee_id, "retrieve_extreme_cases_for_feature", {
            "features": features,
            "sort_feature": sort_feature,
            "num": num,
        })

    def train(
        self,
        trainee_id: str,
        input_cases: List[List[Any]],
        features: Optional[Iterable[str]] = None,
        *,
        accumulate_weight_feature: Optional[str] = None,
        derived_features: Optional[Iterable[str]] = None,
        input_is_substituted: bool = False,
        series: Optional[str] = None,
        session: Optional[str] = None,
        skip_auto_analyze: bool = False,
        train_weights_only: bool = False,
    ) -> Tuple[Dict, int, int]:
        """
        Train one or more cases into a trainee (model).

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        input_cases : list of list of object
            One or more cases to train into the model.
        features : iterable of str, optional
            An iterable of feature names corresponding to the input cases.
        accumulate_weight_feature : str, optional
            Name of feature into which to accumulate neighbors'
            influences as weight for ablated cases. If unspecified, will not
            accumulate weights.
        derived_features: iterable of str, optional
            List of feature names for which values should be derived
            in the specified order.
        input_is_substituted : bool, default False
            if True assumes provided nominal feature values have
            already been substituted.
        series : str, optional
            Name of the series to pull features and case values
            from internal series storage.
        session : str, optional
            The identifier of the Trainee session to associate the cases with.
        skip_auto_analyze : bool, default False
            When true, the Trainee will not auto-analyze when appropriate.
            Instead, the response object will contain an "analyze" status when
            the set auto-analyze parameters indicate that an analyze is needed.
        train_weights_only : bool, default False
            When true, and accumulate_weight_feature is provided,
            will accumulate all of the cases' neighbor weights instead of
            training the cases into the model.

        Returns
        -------
        dict
            A dictionary containing the trained details.
        int
            The request payload size.
        int
            The result payload size.
        """
        return self._execute_sized(trainee_id, "train", {
            "input_cases": input_cases,
            "accumulate_weight_feature": accumulate_weight_feature,
            "derived_features": derived_features,
            "features": features,
            "input_is_substituted": input_is_substituted,
            "series": series,
            "session": session,
            "skip_auto_analyze": skip_auto_analyze,
            "train_weights_only": train_weights_only,
        })

    def impute(
        self,
        trainee_id: str,
        *,
        batch_size: int = 1,
        features: Optional[Iterable[str]] = None,
        features_to_impute: Optional[Iterable[str]] = None,
        session: Optional[str] = None
    ) -> None:
        """
        Impute, or fill in the missing values, for the specified features.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        batch_size : int, default 1
            Larger batch size will increase accuracy and decrease speed.
            Batch size indicates how many rows to fill before recomputing
            conviction.
        features : iterable of str, optional
            An iterable of feature names to use for imputation. If not
            specified, all features will be used imputed.
        features_to_impute : iterable of str, optional
            An iterable of feature names to impute. If not specified, features
            will be used (see above).
        session : str, optional
            The identifier of the Trainee session to associate the edit with.
        """
        return self._execute(trainee_id, "impute", {
            "features": features,
            "features_to_impute": features_to_impute,
            "session": session,
            "batch_size": batch_size,
        })

    def clear_imputed_session(
        self,
        trainee_id: str,
        impute_session: str,
        *,
        session: Optional[str] = None
    ) -> None:
        """
        Clear values that were imputed during a specified session.

        Won't clear values that were manually set by user after the impute.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        impute_session : str,
            The impute session to clear.
        session : str, optional
            The identifier of the Trainee session to associate this edit with.
        """
        return self._execute(trainee_id, "clear_imputed_session", {
            "session": session,
            "impute_session": impute_session,
        })

    def get_cases(
        self,
        trainee_id: str,
        session: Optional[str] = None,
        *,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        indicate_imputed: bool = False,
        features: Optional[Iterable[str]] = None,
        condition: Optional[Dict] = None,
        num_cases: Optional[int] = None,
        precision: Optional[Literal["exact", "similar"]] = None
    ) -> Dict:
        """
        Retrieve cases from a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        session : str, optional
            The session identifier to retrieve cases for, in their trained
            order.
        case_indices : iterable of sequence of str, int, optional
            Iterable of Sequences, of session id and index, where index is the
            original 0-based index of the case as it was trained into the
            session. If specified, returns only these cases and ignores the
            session parameter.
        indicate_imputed : bool, default False
            If set, an additional value will be appended to the cases
            indicating if the case was imputed.
        features : iterable of str, optional
            A list of feature names to return values for in leu of all
            default features.
        condition : dict, optional
            The condition map to select the cases to retrieve that meet all the
            provided conditions.
        num_cases : int, default None
            The maximum amount of cases to retrieve. If not specified, the limit
            will be k cases if precision is "similar", or no limit if precision
            is "exact".
        precision : {"exact", "similar}, optional
            The precision to use when retrieving the cases via condition.
            If not provided, "exact" will be used.

        Returns
        -------
        dict
            A dictionary containing keys 'features' and 'cases'.
        """
        return self._execute(trainee_id, "get_cases", {
            "features": features,
            "session": session,
            "case_indices": case_indices,
            "indicate_imputed": indicate_imputed,
            "condition": condition,
            "num_cases": num_cases,
            "precision": precision,
        })

    def append_to_series_store(
        self,
        trainee_id: str,
        series: str,
        contexts: List[List[Any]],
        *,
        context_features: Optional[Iterable[str]] = None
    ) -> None:
        """
        Append the specified contexts to a series store.

        For use with train series.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to append to.
        series : str
            The name of the series store to append to.
        contexts : list of list of object
            The list of list of context values to append to the series.
        context_features : iterable of str, optional
            The list of feature names for contexts.
        """
        return self._execute(trainee_id, "append_to_series_store",  {
            "context_features": context_features,
            "context_values": contexts,
            "series": series,
        })

    def react(
        self,
        trainee_id: str,
        *,
        action_features: Optional[Iterable[str]] = None,
        action_values: Optional[List[List[object]]] = None,
        allow_nulls: bool = False,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        context_features: Optional[Iterable[str]] = None,
        context_values: Optional[List[List[object]]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        extra_features: Optional[Iterable[str]] = None,
        feature_bounds_map: Optional[Dict] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        input_is_substituted: bool = False,
        into_series_store: Optional[str] = None,
        leave_case_out: bool = False,
        new_case_threshold: Literal["max", "min", "most_similar"] = "min",
        ordered_by_specified_features: bool = False,
        post_process_features: Optional[Iterable[str]] = None,
        post_process_values: Optional[List[object]] = None,
        preserve_feature_values: Optional[Iterable[str]] = None,
        substitute_output: bool = True,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None
    ) -> Dict:
        """
        Single case react.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        context_values : list of list of object, optional
            A 2d list of context values to react to.
            If None for discriminative react, it is assumed that `session`
            and `session_id` keys are set in the `details`.
        action_features : iterable of str, optional
            An iterable of feature names to treat as action features during
            react.
        action_values : list of list of object, optional
            One or more action values to use for action features.
            If specified, will only return the specified explanation
            details for the given actions. (Discriminative reacts only)
        allow_nulls : bool, default False
            When true will allow return of null values if there
            are nulls in the local model for the action features, applicable
            only to discriminative reacts.
        context_features : iterable of str, optional
            An iterable of feature names to treat as context features during
            react.
        derived_context_features : iterable of str, optional
            An iterable of feature names whose values should be computed
            from the provided context in the specified order. Must be different
            than context_features.
        derived_action_features : iterable of str, optional
            An iterable of feature names whose values should be computed
            after generation from the generated case prior to output, in the
            specified order. Must be a subset of action_features.
        input_is_substituted : bool, default False
            if True assumes provided categorical (nominal or
            ordinal) feature values have already been substituted.
        substitute_output : bool, default True
            If False, will not substitute categorical feature
            values. Only applicable if a substitution value map has been set.
        details : dict, optional
            If details are specified, the response will
            contain the requested explanation data along with the reaction.
        desired_conviction : float
            If specified will execute a generative react. If not
            specified will executed a discriminative react. Conviction is the
            ratio of expected surprisal to generated surprisal for each
            feature generated, valid values are in the range of zero to
            infinity.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        use_case_weights : bool, default False
            If set to True will scale influence weights by each
            case's weight_feature weight.
        case_indices : Iterable of Sequence[Union[str, int]], defaults to None
            An Iterable of Sequences, of session id and index, where
            index is the original 0-based index of the case as it was trained
            into the session. If this case does not exist, discriminative react
            outputs null, generative react ignores it.
        post_process_features : iterable of str, optional
            List of feature names that will be made available during the
            execution of post_process feature attributes.
        post_process_values : list of object, optional
            A 2d list of values corresponding to post_process_features that
            will be made available during the execution of post_process feature
            attributes.
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
        into_series_store : str, optional
            The name of a series store. If specified, will store an internal
            record of all react contexts for this session and series to be used
            later with train series.
        use_regional_model_residuals : bool
            If false uses model feature residuals, if True
            recalculates regional model residuals.
        feature_bounds_map : dict of dict
            A mapping of feature names to the bounds for the
            feature values to be generated in.
        generate_new_cases : {"always", "attempt", "no"}, default "no"
            How to generate new cases.
        ordered_by_specified_features : bool, default False
            If True order of generated feature values will match
            the order of specified features.
        new_case_threshold : {"min", "max", "most_similar"}, optional
            Distance to determine the privacy cutoff. If None,
            will default to "min".
        exclude_novel_nominals_from_uniqueness_check : bool, default False
            If True, will exclude features which have a subtype defined in their feature
            attributes from the uniqueness check that happens when ``generate_new_cases``
            is True. Only applies to generative reacts.

        Returns
        -------
        dict
            The react result including audit details.
        """
        return self._execute(trainee_id, "react", {
            "context_features": context_features,
            "context_values": context_values,
            "action_features": action_features,
            "action_values": action_values,
            "details": details,
            "derived_action_features": derived_action_features,
            "derived_context_features": derived_context_features,
            "exclude_novel_nominals_from_uniqueness_check": exclude_novel_nominals_from_uniqueness_check,
            "extra_features": extra_features,
            "case_indices": case_indices,
            "allow_nulls": allow_nulls,
            "input_is_substituted": input_is_substituted,
            "substitute_output": substitute_output,
            "weight_feature": weight_feature,
            "leave_case_out": leave_case_out,
            "use_case_weights": use_case_weights,
            "use_regional_model_residuals": use_regional_model_residuals,
            "desired_conviction": desired_conviction,
            "feature_bounds_map": feature_bounds_map,
            "generate_new_cases": generate_new_cases,
            "ordered_by_specified_features": ordered_by_specified_features,
            "post_process_features": post_process_features,
            "post_process_values": post_process_values,
            "preserve_feature_values": preserve_feature_values,
            "new_case_threshold": new_case_threshold,
            "into_series_store": into_series_store,
        })

    def batch_react(
        self,
        trainee_id: str,
        *,
        action_features: Optional[Iterable[str]] = None,
        action_values: Optional[List[List[object]]] = None,
        allow_nulls: bool = False,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        context_features: Optional[Iterable[str]] = None,
        context_values: Optional[List[List[object]]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        extra_features: Optional[Iterable[str]] = None,
        feature_bounds_map: Optional[Dict] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        input_is_substituted: bool = False,
        into_series_store: Optional[str] = None,
        leave_case_out: bool = False,
        new_case_threshold: Literal["max", "min", "most_similar"] = "min",
        num_cases_to_generate: Optional[int] = None,
        ordered_by_specified_features: bool = False,
        post_process_features: Optional[Iterable[str]] = None,
        post_process_values: Optional[List[List[object]]] = None,
        preserve_feature_values: Optional[Iterable[str]] = None,
        substitute_output: bool = True,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None
    ) -> Tuple[Dict, int, int]:
        """
        Multiple case react.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        context_values : list of list of object, optional
            A 2d list of context values to react to.
            If None for discriminative react, it is assumed that `session`
            and `session_id` keys are set in the `details`.
        action_features : iterable of str, optional
            An iterable of feature names to treat as action features during
            react.
        action_values : list of list of object, optional
            One or more action values to use for action features.
            If specified, will only return the specified explanation
            details for the given actions. (Discriminative reacts only)
        allow_nulls : bool, default False
            When true will allow return of null values if there
            are nulls in the local model for the action features, applicable
            only to discriminative reacts.
        context_features : iterable of str, optional
            An iterable of feature names to treat as context features during
            react.
        derived_context_features : iterable of str, optional
            An iterable of feature names whose values should be computed
            from the provided context in the specified order. Must be different
            than context_features.
        derived_action_features : iterable of str, optional
            An iterable of feature names whose values should be computed
            after generation from the generated case prior to output, in the
            specified order. Must be a subset of action_features.
        exclude_novel_nominals_from_uniqueness_check : bool, default False
            If True, will exclude features which have a subtype defined in their feature
            attributes from the uniqueness check that happens when ``generate_new_cases``
            is True. Only applies to generative reacts.
        input_is_substituted : bool, default False
            if True assumes provided categorical (nominal or
            ordinal) feature values have already been substituted.
        substitute_output : bool, default True
            If False, will not substitute categorical feature
            values. Only applicable if a substitution value map has been set.
        details : dict, optional
            If details are specified, the response will
            contain the requested explanation data along with the reaction.
        desired_conviction : float
            If specified will execute a generative react. If not
            specified will executed a discriminative react. Conviction is the
            ratio of expected surprisal to generated surprisal for each
            feature generated, valid values are in the range of zero to
            infinity.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        use_case_weights : bool, default False
            If set to True will scale influence weights by each
            case's weight_feature weight.
        case_indices : Iterable of Sequence[Union[str, int]], defaults to None
            An Iterable of Sequences, of session id and index, where
            index is the original 0-based index of the case as it was trained
            into the session. If this case does not exist, discriminative react
            outputs null, generative react ignores it.
        post_process_features : iterable of str, optional
            List of feature names that will be made available during the
            execution of post_process feature attributes.
        post_process_values : list of list of object, optional
            A 2d list of values corresponding to post_process_features that
            will be made available during the execution of post_process feature
            attributes.
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
        into_series_store : str, optional
            The name of a series store. If specified, will store an internal
            record of all react contexts for this session and series to be used
            later with train series.
        use_regional_model_residuals : bool
            If false uses model feature residuals, if True
            recalculates regional model residuals.
        feature_bounds_map : dict of dict
            A mapping of feature names to the bounds for the
            feature values to be generated in.
        generate_new_cases : {"always", "attempt", "no"}, default "no"
            How to generate new cases.
        ordered_by_specified_features : bool, default False
            If True order of generated feature values will match
            the order of specified features.
        num_cases_to_generate : int, default 1
            The number of cases to generate.
        new_case_threshold : {"min", "max", "most_similar"}, optional
            Distance to determine the privacy cutoff. If None,
            will default to "min".

        Returns
        -------
        dict
            The react result including audit details.
        int
            The request payload size.
        int
            The result payload size.
        """
        return self._execute_sized(trainee_id, "batch_react", {
            "context_features": context_features,
            "context_values": context_values,
            "action_features": action_features,
            "action_values": action_values,
            "derived_context_features": derived_context_features,
            "derived_action_features": derived_action_features,
            "details": details,
            "exclude_novel_nominals_from_uniqueness_check": exclude_novel_nominals_from_uniqueness_check,
            "extra_features": extra_features,
            "case_indices": case_indices,
            "allow_nulls": allow_nulls,
            "input_is_substituted": input_is_substituted,
            "substitute_output": substitute_output,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
            "leave_case_out": leave_case_out,
            "num_cases_to_generate": num_cases_to_generate,
            "use_regional_model_residuals": use_regional_model_residuals,
            "desired_conviction": desired_conviction,
            "feature_bounds_map": feature_bounds_map,
            "generate_new_cases": generate_new_cases,
            "ordered_by_specified_features": ordered_by_specified_features,
            "post_process_features": post_process_features,
            "post_process_values": post_process_values,
            "preserve_feature_values": preserve_feature_values,
            "new_case_threshold": new_case_threshold,
            "into_series_store": into_series_store,
        })

    def batch_react_series(
        self,
        trainee_id: str,
        *,
        action_features: Optional[Iterable[str]] = None,
        action_values: Optional[List[List[object]]] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        context_values: Optional[List[List[object]]] = None,
        context_features: Optional[Iterable[str]] = None,
        continue_series: Optional[bool] = False,
        continue_series_features: Optional[Iterable[str]] = None,
        continue_series_values: Optional[Union[List[object], List[List[object]]]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        extra_features: Optional[Iterable[str]] = None,
        feature_bounds_map: Optional[Dict] = None,
        final_time_steps: Optional[Union[List[object], List[List[object]]]] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        init_time_steps: Optional[Union[List[object], List[List[object]]]] = None,
        initial_features: Optional[Iterable[str]] = None,
        initial_values: Optional[Union[List[object], List[List[object]]]] = None,
        input_is_substituted: bool = False,
        leave_case_out: bool = False,
        max_series_lengths: Optional[List[int]] = None,
        new_case_threshold: Literal["max", "min", "most_similar"] = "min",
        num_series_to_generate: int = 1,
        ordered_by_specified_features: bool = False,
        output_new_series_ids: bool = True,
        preserve_feature_values: Optional[Iterable[str]] = None,
        series_context_features: Optional[Iterable[str]] = None,
        series_context_values: Optional[Union[List[object], List[List[object]]]] = None,
        series_id_tracking: Literal["dynamic", "fixed", "no"] = "fixed",
        series_stop_maps: Optional[List[Dict[str, Dict]]] = None,
        substitute_output: bool = True,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None
    ) -> Tuple[Dict, int, int]:
        """
        React in a series until a series_stop_map condition is met.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        num_series_to_generate : int, optional
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
        series_context_features : iterable of str, optional
            List of context features corresponding to
            series_context_values, if specified must not overlap with any
            initial_features or context_features.
        series_context_values : list of list of list of object or list, optional
            3d-list of context values, one for each feature for each
            row for each series. If specified, max_series_lengths are ignored.
        output_new_series_ids : bool, default True
            If True, series ids are replaced with unique values on output.
            If False, will maintain or replace ids with existing trained values,
            but also allows output of series with duplicate existing ids.
        series_id_tracking : {"dynamic", "fixed", "no"}, default "fixed"
            Controls how closely generated series should follow existing series.
        context_values: list of list of object
            See parameter ``contexts`` in :meth:`react`.
        action_features: iterable of str
            See parameter ``action_features`` in :meth:`react`.
        action_values: list of list of object
            See parameter ``actions`` in :meth:`react`.
        context_features: iterable of str
            See parameter ``context_features`` in :meth:`react`.
        input_is_substituted : bool, default False
            See parameter ``input_is_substituted`` in :meth:`react`.
        substitute_output : bool
            See parameter ``substitute_output`` in :meth:`react`.
        details: dict, optional
            See parameter ``details`` in :meth:`react`.
        desired_conviction: float
            See parameter ``desired_conviction`` in :meth:`react`.
        exclude_novel_nominals_from_uniqueness_check : bool, default False
            See parameter ``exclude_novel_nominals_from_uniqueness_check`` in :meth:`react`.
        weight_feature : str
            See parameter ``weight_feature`` in :meth:`react`.
        use_case_weights : bool
            See parameter ``use_case_weights`` in :meth:`react`.
        case_indices: iterable of sequence of str, int
            See parameter ``case_indices`` in :meth:`react`.
        preserve_feature_values : iterable of str
            See parameter ``preserve_feature_values`` in :meth:`react`.
        new_case_threshold : str
            See parameter ``new_case_threshold`` in :meth:`react`.
        leave_case_out : bool
            See parameter ``leave_case_out`` in :meth:`react`.
        use_regional_model_residuals : bool
            See parameter ``use_regional_model_residuals`` in :meth:`react`.
        feature_bounds_map: dict of dict
            See parameter ``feature_bounds_map`` in :meth:`react`.
        generate_new_cases : {"always", "attempt", "no"}
            See parameter ``generate_new_cases`` in :meth:`react`.
        ordered_by_specified_features : bool
            See parameter ``ordered_by_specified_features`` in :meth:`react`.

        Returns
        -------
        dict
            A dictionary with keys `action_features` and `series`. Where
            `series` is a 2d list of values (rows of data per series), and
            `action_features` is the list of all action features
            (specified and derived).
        int
            The request payload size.
        int
            The result payload size.
        """
        return self._execute_sized(trainee_id, "batch_react_series", {
            "context_features": context_features,
            "context_values": context_values,
            "action_features": action_features,
            "action_values": action_values,
            "final_time_steps": final_time_steps,
            "init_time_steps": init_time_steps,
            "initial_features": initial_features,
            "initial_values": initial_values,
            "series_stop_maps": series_stop_maps,
            "max_series_lengths": max_series_lengths,
            "continue_series": continue_series,
            "continue_series_features": continue_series_features,
            "continue_series_values": continue_series_values,
            "derived_context_features": derived_context_features,
            "derived_action_features": derived_action_features,
            "series_context_features": series_context_features,
            "series_context_values": series_context_values,
            "series_id_tracking": series_id_tracking,
            "output_new_series_ids": output_new_series_ids,
            "details": details,
            "exclude_novel_nominals_from_uniqueness_check": exclude_novel_nominals_from_uniqueness_check,
            "extra_features": extra_features,
            "case_indices": case_indices,
            "input_is_substituted": input_is_substituted,
            "substitute_output": substitute_output,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
            "leave_case_out": leave_case_out,
            "num_series_to_generate": num_series_to_generate,
            "preserve_feature_values": preserve_feature_values,
            "new_case_threshold": new_case_threshold,
            "use_regional_model_residuals": use_regional_model_residuals,
            "desired_conviction": desired_conviction,
            "feature_bounds_map": feature_bounds_map,
            "generate_new_cases": generate_new_cases,
            "ordered_by_specified_features": ordered_by_specified_features,
        })

    def react_into_features(
        self,
        trainee_id: str,
        *,
        distance_contribution: bool = False,
        familiarity_conviction_addition: bool = False,
        familiarity_conviction_removal: bool = False,
        features: Optional[Iterable[str]] = None,
        influence_weight_entropy: Union[bool, str] = False,
        p_value_of_addition: bool = False,
        p_value_of_removal: bool = False,
        similarity_conviction: bool = False,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None,
    ) -> None:
        """
        Calculate and cache conviction and other statistics.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
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
        use_case_weights : bool, default False
            If set to True will scale influence weights by each
            case's weight_feature weight.
        """
        return self._execute(trainee_id, "react_into_features", {
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

    def batch_react_group(
        self,
        trainee_id: str,
        new_cases: List[List[List[object]]],
        *,
        features: Optional[Iterable[str]] = None,
        distance_contributions: bool = False,
        familiarity_conviction_addition: bool = True,
        familiarity_conviction_removal: bool = False,
        kl_divergence_addition: bool = False,
        kl_divergence_removal: bool = False,
        p_value_of_addition: bool = False,
        p_value_of_removal: bool = False,
        weight_feature: Optional[str] = None,
        use_case_weights: bool = False
    ) -> Dict:
        """
        Computes specified data for a **set** of cases.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        new_cases : list of list of list of object or list
            Specify a **set** using a list of cases to compute the conviction of
            groups of cases as shown in the following example.
        features : iterable of str, optional
            An iterable of feature names to consider while calculating
            convictions.
        distance_contributions : bool, default False
            Calculate and output distance contribution ratios in
            the output dict for each case.
        familiarity_conviction_addition : bool, default True
            Calculate and output familiarity conviction of adding the
            specified cases.
        familiarity_conviction_removal : bool, default False
            Calculate and output familiarity conviction of removing
            the specified cases.s
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
        use_case_weights : bool, default False
            If set to True will scale influence weights by each
            case's weight_feature weight.

        Returns
        -------
        dict
            The react response.
        """
        return self._execute(trainee_id, "batch_react_group", {
            "features": features,
            "new_cases": new_cases,
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

    def react_into_trainee(
        self,
        trainee_id: str,
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
        num_robust_influence_samples: Optional[int] = None,
        num_robust_residual_samples: Optional[int] = None,
        num_robust_influence_samples_per_case: Optional[int] = None,
        num_samples: Optional[int] = None,
        residuals: Optional[bool] = None,
        residuals_robust: Optional[bool] = None,
        sample_model_fraction: Optional[float] = None,
        sub_model_size: Optional[int] = None,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None
    ) -> None:
        """
        Compute and cache specified feature prediction stats.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        action_feature : str, optional
            Name of target feature for which to do computations. Default is
            whatever the model was analyzed for, e.g., action feature for MDA
            and contributions, or ".targetless" if analyzed for targetless.
            This parameter is required for MDA or contributions computations.
        context_features : iterable of str, optional
            List of features names to use as contexts for
            computations. Default is all trained non-unique features if
            unspecified.
        contributions: bool, optional
            For each context_feature, use the full set of all other
            context_features to compute the mean absolute delta between
            prediction of action_feature with and without the context_feature
            in the model. False removes cached values.
        contributions_robust: bool, optional
            For each context_feature, use the robust (power set/permutation)
            set of all other context_features to compute the mean absolute
            delta between prediction of action_feature with and without the
            context_feature in the model. False removes cached values.
        hyperparameter_param_path: iterable of str, optional.
            Full path for hyperparameters to use for computation. If specified
            for any residual computations, takes precedence over action_feature
            parameter.  Can be set to a 'paramPath' value from the results of
            'get_params()' for a specific set of hyperparameters.
        mda : bool, optional
            When True will compute Mean Decrease in Accuracy (MDA)
            for each context feature at predicting the action_feature. Drop
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
            will be updated to use num_robust_influence_samples in a future
            release.
        num_robust_influence_samples_per_case : int, optional
            Specifies the number of robust samples to use for each case for
            robust contribution computations.
            Defaults to 300 + 2 * (number of features).
        num_samples : int, optional
            Total sample size of model to use (using sampling with replacement)
            for all non-robust computation. Defaults to 1000.
            If specified overrides sample_model_fraction.```
        residuals : bool, optional
            For each context_feature, use the full set of all other
            context_features to predict the feature. When True computes and
            caches MAE (mean absolute error), R^2, RMSE (root mean squared
            error), and Spearman Coefficient for continuous features, and
            MAE, accuracy, precision and recall for nominal features.
            False removes cached values.
        residuals_robust : bool, optional
            For each context_feature, computes and caches the same stats as
            residuals but using the robust (power set/permutations) set of all
            other context_features to predict the feature.
            False removes cached values.
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
        self._execute(trainee_id, "react_into_trainee", {
            "context_features": context_features,
            "use_case_weights": use_case_weights,
            "weight_feature": weight_feature,
            "num_samples": num_samples,
            "residuals": residuals,
            "residuals_robust": residuals_robust,
            "contributions": contributions,
            "contributions_robust": contributions_robust,
            "mda": mda,
            "mda_permutation": mda_permutation,
            "mda_robust": mda_robust,
            "mda_robust_permutation": mda_robust_permutation,
            "num_robust_influence_samples": num_robust_influence_samples,
            "num_robust_residual_samples": num_robust_residual_samples,
            "num_robust_influence_samples_per_case":
                num_robust_influence_samples_per_case,
            "hyperparameter_param_path": hyperparameter_param_path,
            "sample_model_fraction": sample_model_fraction,
            "sub_model_size": sub_model_size,
            "action_feature": action_feature
        })

    def compute_conviction_of_features(
        self,
        trainee_id: str,
        *,
        features: Optional[Iterable[str]] = None,
        action_features: Optional[Iterable[str]] = None,
        familiarity_conviction_addition: bool = True,
        familiarity_conviction_removal: bool = False,
        weight_feature: Optional[str] = None,
        use_case_weights: bool = False
    ) -> Dict:
        """
        Get familiarity conviction for features in the model.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        features : iterable of str, optional
            An iterable of feature names to calculate convictions. At least 2
            features are required to get familiarity conviction. If not
            specified all features will be used.
        action_features : iterable of str, optional
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
            familiarity_conviction_removal.
        """
        return self._execute(trainee_id, "compute_conviction_of_features", {
            "features": features,
            "action_features": action_features,
            "familiarity_conviction_addition": familiarity_conviction_addition,
            "familiarity_conviction_removal": familiarity_conviction_removal,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })

    def get_session_indices(self, trainee_id: str, session: str) -> List[int]:
        """
        Get list of all session indices for a specified session.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        session : str
            The identifier of the session.
        Returns
        -------
        list of int
            A list of the session indices for the session.
        """
        return self._execute(trainee_id, "get_session_indices", {
            "session": session,
        })

    def get_session_training_indices(self, trainee_id: str, session: str
                                     ) -> List[int]:
        """
        Get list of all session training indices for a specified session.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        session : str
            The identifier of the session.

        Returns
        -------
        list of int
            A list of the session training indices for the session.
        """
        return self._execute(trainee_id, "get_session_training_indices", {
            "session": session,
        })

    def set_internal_parameters(self, trainee_id: str, params: Dict) -> None:
        """
        Sets specific model parameters in the Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        params : dict
            A dictionary containing the internal parameters.
        """
        return self._execute(trainee_id, "set_internal_parameters", {
            **params
        })

    def set_feature_attributes(
        self,
        trainee_id: str,
        feature_attributes: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Sets feature attributes for a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        feature_attributes : dict of str to dict
            A dictionary of feature name to dictionary of feature attributes.

        Returns
        -------
        dict
            The updated feature attributes.
        """
        return self._execute(trainee_id, "set_feature_attributes", {
            "features": feature_attributes,
        })

    def get_feature_attributes(self, trainee_id: str) -> Dict[str, Dict]:
        """
        Get Trainee feature attributes.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee

        Returns
        -------
        dict
            A dictionary of feature name to dictionary of feature attributes.
        """
        return self._execute(trainee_id, "get_feature_attributes", {})

    def export_trainee(
        self,
        trainee_id: str,
        path_to_trainee: Optional[Union[Path, str]] = None,
        decode_cases: bool = False,
        separate_files: bool = False
    ) -> None:
        """
        Export a saved Trainee's data to json files for migration.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        path_to_trainee : Path or str, optional
            The path to where the saved trainee file is located.
        decoded_cases : bool, default False.
            Whether to export decoded cases.
        separate_files : bool, default False
            Whether to load each case from its individual file.
        """
        if path_to_trainee is None:
            path_to_trainee = self.default_save_path

        return self._execute(trainee_id, "export_trainee", {
            "trainee_filepath": f"{path_to_trainee}/",
            "root_filepath": f"{self.howso_path}/",
            "decode_cases": decode_cases,
            "separate_files": separate_files,
        })

    def upgrade_trainee(
        self,
        trainee_id: str,
        path_to_trainee: Optional[Union[Path, str]] = None,
        separate_files: bool = False
    ) -> None:
        """
        Upgrade a saved Trainee to current version.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        path_to_trainee : Path or str, optional
            The path to where the saved Trainee file is located.
        separate_files : bool, default False
            Whether to load each case from its individual file.
        """
        if path_to_trainee is None:
            path_to_trainee = self.default_save_path

        return self._execute(trainee_id, "upgrade_trainee", {
            "trainee_filepath": f"{path_to_trainee}/",
            "root_filepath": f"{self.howso_path}/",
            "separate_files": separate_files,
        })

    def analyze(self, trainee_id: str, **kwargs) -> None:
        """
        Analyzes a trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        kwargs : dict
            Analysis arguments.
        """
        params = {**kwargs}
        return self._execute(trainee_id, "analyze", params)

    def get_feature_residuals(
        self,
        trainee_id: str,
        action_feature: Optional[str] = None,
        robust: Optional[bool] = None,
        robust_hyperparameters: Optional[bool] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict:
        """
        Get cached feature residuals.

        .. deprecated:: 1.0.0
            Use get_prediction_stats() instead.
        """
        return self._execute(trainee_id, "get_feature_residuals", {
            "robust": robust,
            "action_feature": action_feature,
            "robust_hyperparameters": robust_hyperparameters,
            "weight_feature": weight_feature,
        })

    def get_feature_mda(
        self,
        trainee_id: str,
        action_feature: str,
        permutation: Optional[bool] = None,
        robust: Optional[bool] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict:
        """
        Get cached feature Mean Decrease In Accuracy (MDA).

        .. deprecated:: 1.0.0
            Use get_prediction_stats() instead.
        """
        return self._execute(trainee_id, "get_feature_mda", {
            "robust": robust,
            "action_feature": action_feature,
            "permutation": permutation,
            "weight_feature": weight_feature,
        })

    def get_feature_contributions(
        self,
        trainee_id: str,
        action_feature: str,
        robust: Optional[bool] = None,
        directional: bool = False,
        weight_feature: Optional[str] = None,
    ) -> Dict:
        """
        Get cached feature contributions.

        .. deprecated:: 1.0.0
            Use get_prediction_stats() instead.
        """
        return self._execute(trainee_id, "get_feature_contributions", {
            "robust": robust,
            "action_feature": action_feature,
            "directional": directional,
            "weight_feature": weight_feature,
        })

    def get_prediction_stats(
        self, trainee_id, *,
        action_feature=None,
        condition=None,
        num_cases=None,
        num_robust_influence_samples_per_case=None,
        precision=None,
        robust=None,
        robust_hyperparameters=None,
        stats=None,
        weight_feature=None,
    ) -> Dict:
        """
        Get feature prediction stats.

        Parameters
        ----------
        trainee_id : str
            The id or name of the trainee.
        action_feature : str, optional
            When specified, will attempt to return stats that
            were computed for this specified action_feature.
            Note: ".targetless" is the action feature used during targetless
            analysis.
        condition : dict or None, optional
            A condition map to select which cases to compute prediction stats
            for.
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
        stats : iterable of str, optional
            List of stats to output. When unspecified, returns all.
        weight_feature : str, optional
            When specified, will attempt to return stats that
            were computed using this weight_feature.

        Returns
        -------
        dict of str to dict of str to float
            A map of feature to map of stat type to stat values.
        """
        return self._execute(trainee_id, "get_prediction_stats", {
            "robust": robust,
            "action_feature": action_feature,
            "condition": condition,
            "num_cases": num_cases,
            "num_robust_influence_samples_per_case": num_robust_influence_samples_per_case,
            "precision": precision,
            "robust_hyperparameters": robust_hyperparameters,
            "stats": stats,
            "weight_feature": weight_feature,
        })

    def get_marginal_stats(
        self,
        trainee_id: str,
        *,
        condition: Optional[Dict[str, Any]] = None,
        num_cases: Optional[int] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
        weight_feature: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get marginal stats for all features.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
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
        return self._execute(trainee_id, "get_marginal_stats", {
            "condition": condition,
            "num_cases": num_cases,
            "precision": precision,
            "weight_feature": weight_feature,
        })

    def set_random_seed(self, trainee_id: str, seed: Union[int, float, str]
                        ) -> None:
        """
        Sets the random seed for the Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        seed: int or float or str
            The random seed.
        """
        return self._execute(trainee_id, "set_random_seed", {
            "seed": seed,
        })

    def get_internal_parameters(
        self,
        trainee_id: str,
        *,
        action_feature: Optional[str] = None,
        context_features: Optional[Iterable[str]] = None,
        mode: Optional[Literal["robust", "full"]] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict:
        """
        Get the parameters used by the Trainee.

        If 'action_feature', 'context_features', 'mode', or 'weight_feature'
        are specified, then the best hyperparameters analyzed in the Trainee
        are the value of the 'hyperparameter_map' key, otherwise this value
        will be the dictionary containing all the hyperparameter sets in
        the Trainee.


        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
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
        return self._execute(trainee_id, "get_internal_parameters", {
            "action_feature": action_feature,
            "context_features": context_features,
            "mode": mode,
            "weight_feature": weight_feature,
        })

    def move_cases(
        self,
        trainee_id: str,
        num_cases: int = 1,
        *,
        case_indices: Optional[Iterable[Tuple[str, int]]] = None,
        condition: Optional[Dict] = None,
        condition_session: Optional[str] = None,
        distribute_weight_feature: Optional[str] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
        preserve_session_data: bool = False,
        session: Optional[str] = None,
        source_id: Optional[str] = None,
        source_name_path: Optional[List[str]] = None,
        target_name_path: Optional[List[str]] = None,
        target_id: Optional[str] = None
    ) -> Dict:
        """
        Moves cases from one trainee to another in the hierarchy.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee doing the moving.
        num_cases : int
            The number of cases to move; minimum 1 case must be moved.
            Ignored if case_indices is specified.
        case_indices : list of tuples
            A list of tuples containing session ID and session training index
            for each case to be removed.
        condition : dict, optional
            The condition map to select the cases to move that meet all the
            provided conditions. Ignored if case_indices is specified.
        condition_session : str, optional
            If specified, ignores the condition and operates on cases for
            the specified session id. Ignored if case_indices is specified.
        precision : {"exact", "similar"}, optional
            The precision to use when moving the cases. Options are 'exact'
            or 'similar'. If not specified, "exact" will be used.
            Ignored if case_indices is specified.
        preserve_session_data : bool, default False
            When True, will move cases without cleaning up session data.
        session : str, optional
            The identifier of the Trainee session to associate the move with.
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
        dict
            A dictionary with key 'count' for the number of moved cases.
        """
        result = self._execute(trainee_id, "move_cases", {
            "target_id": target_id,
            "case_indices": case_indices,
            "condition": condition,
            "condition_session": condition_session,
            "precision": precision,
            "num_cases": num_cases,
            "preserve_session_data": preserve_session_data,
            "session": session,
            "distribute_weight_feature": distribute_weight_feature,
            "source_id": source_id,
            "source_name_path": source_name_path,
            "target_name_path": target_name_path
        })
        if not result:
            return {'count': 0}
        return result

    def remove_cases(
        self,
        trainee_id: str,
        num_cases: int = 1,
        *,
        case_indices: Optional[Iterable[Tuple[str, int]]] = None,
        condition: Optional[Dict[str, Any]] = None,
        condition_session: Optional[str] = None,
        distribute_weight_feature: Optional[str] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
        preserve_session_data: bool = False,
        session: Optional[str] = None
    ) -> Dict:
        """
        Removes cases from a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        num_cases : int
            The number of cases to remove; minimum 1 case must be removed.
            Ignored if case_indices is specified.
        case_indices : list of tuples
            A list of tuples containing session ID and session training index
            for each case to be removed.
        condition : dict of str to object, optional
            The condition map to select the cases to remove that meet all the
            provided conditions. Ignored if case_indices is specified.
        condition_session : str, optional
            If specified, ignores the condition and operates on cases for
            the specified session id. Ignored if case_indices is specified.
        distribute_weight_feature : str, optional
            When specified, will distribute the removed cases' weights
            from this feature into their neighbors.
        precision : {"exact", "similar"}, optional
            The precision to use when moving the cases, defaults to "exact".
            Ignored if case_indices is specified.
        preserve_session_data : bool, default False
            When True, will remove cases without cleaning up session data.
        session : str, optional
            The identifier of the Trainee session to associate the removal with.

        Returns
        -------
        dict
            A dictionary with key 'count' for the number of removed cases.
        """
        result = self._execute(trainee_id, "remove_cases", {
            "case_indices": case_indices,
            "condition": condition,
            "condition_session": condition_session,
            "precision": precision,
            "num_cases": num_cases,
            "preserve_session_data": preserve_session_data,
            "session": session,
            "distribute_weight_feature": distribute_weight_feature,
        })
        if not result:
            return {'count': 0}
        return result

    def edit_cases(
        self,
        trainee_id: str,
        feature_values: Optional[Iterable[Any]] = None,
        *,
        case_indices: Optional[Iterable[Tuple[str, int]]] = None,
        condition: Optional[Dict[str, Any]] = None,
        condition_session: Optional[str] = None,
        features: Optional[Iterable[str]] = None,
        num_cases: Optional[int] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
        session: Optional[str] = None
    ) -> Dict:
        """
        Edit feature values for the specified cases.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        feature_values : list of object
            The feature values to edit the case(s) with. If specified as a list,
            the order corresponds with the order of the `features` parameter.
        case_indices : Iterable of Sequence[Union[str, int]], optional
            Iterable of Sequences containing the session id and index, where
            index is the original 0-based index of the case as it was trained
            into the session. This explicitly specifies the cases to edit. When
            specified, `condition` and `condition_session` are ignored.
        condition : dict, optional
            A condition map to select which cases to edit. Ignored when
            `case_indices` are specified.
        condition_session : str, optional
            If specified, ignores the condition and operates on all cases for
            the specified session.
        features : iterable of str, optional
            The names of the features to edit. Corresponds to feature_values.
        num_cases : int, default None
            The maximum amount of cases to edit. If not specified, the limit
            will be k cases if precision is "similar", or no limit if precision
            is "exact".
        precision : {"exact", "similar"}, optional
            The precision to use when moving the cases, defaults to "exact".
        session : str, optional
            The identifier of the Trainee session to associate the edit with.

        Returns
        -------
        dict
            A dictionary with key 'count' for the number of modified cases.
        """
        result = self._execute(trainee_id, "edit_cases", {
            "case_indices": case_indices,
            "condition": condition,
            "condition_session": condition_session,
            "features": features,
            "feature_values": feature_values,
            "precision": precision,
            "num_cases": num_cases,
            "session": session,
        })
        if not result:
            return {'count': 0}
        return result

    def pairwise_distances(
        self,
        trainee_id: str,
        features: Optional[Iterable[str]] = None,
        *,
        action_feature: Optional[str] = None,
        from_case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        from_values: Optional[List[List[Any]]] = None,
        to_case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        to_values: Optional[List[List[Any]]] = None,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None
    ) -> List:
        """
        Compute pairwise distances between specified cases.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
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
        from_values : list of list of object, optional
            A 2d-list of case values. If specified must be either length of
            1 or match length of `to_values` or `to_case_indices`.
        to_case_indices : Iterable of Sequence[Union[str, int]], optional
            An Iterable of Sequences, of session id and index, where index
            is the original 0-based index of the case as it was trained into
            the session. If specified must be either length of 1 or match
            length of `from_values` or `from_case_indices`.
        to_values : list of list of object, optional
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
        return self._execute(trainee_id, "pairwise_distances", {
            "features": features,
            "action_feature": action_feature,
            "from_case_indices": from_case_indices,
            "from_values": from_values,
            "to_case_indices": to_case_indices,
            "to_values": to_values,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })

    def distances(
        self,
        trainee_id: str,
        features: Optional[Iterable[str]] = None,
        *,
        action_feature: Optional[str] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        feature_values: Optional[Iterable[Any]] = None,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None,
        row_offset: int = 0,
        row_count: Optional[int] = None,
        column_offset: int = 0,
        column_count: Optional[int] = None
    ) -> Dict:
        """
        Compute distances matrix for specified cases.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
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
        feature_values : list of object, optional
            If specified, returns distances of the local model relative to
            these values, ignores `case_indices` parameter.
        use_case_weights : bool, default False
            If set to True, will scale influence weights by each case's
            `weight_feature` weight.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        dict
            A dictionary of keys, 'distances', 'row_case_indices' and
            'column_case_indices'.
        """
        return self._execute(trainee_id, "distances", {
            "features": features,
            "action_feature": action_feature,
            "case_indices": case_indices,
            "feature_values": feature_values,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
            "row_offset": row_offset,
            "row_count": row_count,
            "column_offset": column_offset,
            "column_count": column_count,
        })

    def evaluate(
        self,
        trainee_id: str,
        features_to_code_map: Dict[str, str],
        *,
        aggregation_code: Optional[str] = None
    ) -> Dict:
        """
        Evaluate custom code on feature values of all cases in the trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        features_to_code_map : dict of str to str
            A dictionary with feature name keys and custom Amalgam code string
            values.
        aggregation_code : str, optional
            A string of custom Amalgam code that can access the list of values
            derived form the custom code in features_to_code_map.

        Returns
        -------
        dict
            A dictionary with keys: 'evaluated' and 'aggregated'.
        """
        return self._execute(
            trainee_id, "evaluate",
            {
                "features_to_code_map": features_to_code_map,
                "aggregation_code": aggregation_code
            })

    def reset_parameter_defaults(self, trainee_id: str) -> None:
        """
        Reset Trainee hyperparameters and thresholds.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        """
        return self._execute(
            trainee_id, "reset_parameter_defaults", {})

    def get_hierarchy(self, trainee_id: str) -> Dict:
        """
        Output the hierarchy for a trainee.

        Returns
        -------
        dict of {str: dict}
            Dictionary of the currently contained hierarchy as a nested dict
            with False for trainees that are stored independently.
        """
        return self._execute(trainee_id, "get_hierarchy", {})

    def rename_subtrainee(
        self,
        trainee_id: str,
        new_name: str,
        *,
        child_id: Optional[str] = None,
        child_name_path: Optional[List[str]] = None
    ) -> None:
        """
        Renames a contained child trainee in the hierarchy.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee whose child to rename.
        new_name : str
            New name of child trainee
        child_id : str, optional
            Unique id of child trainee to rename. Ignored if child_name_path is specified
        child_name_path : list of str, optional
            List of strings specifying the user-friendly path of the child
            subtrainee to rename.
        """
        return self._execute(
            trainee_id, "rename_subtrainee",
            {
                "new_name": new_name,
                "child_name_path": child_name_path,
                "child_id": child_id
            })

    def execute_on_subtrainee(
        self,
        trainee_id: str,
        method: str,
        *,
        as_external: Optional[bool] = False,
        child_id: Optional[str] = None,
        child_name_path: Optional[List[str]] = None,
        payload: Optional[Dict] = None,
        load_external_trainee_id: Optional[str] = None
    ) -> object:
        """
        Executes any method in the engine API directly on any child trainee.

        Parameters
        ----------
        method : str, name of method to execute
        payload : dict, parameters specific to the method being called
        child_name_path : list of str, optional
            List of strings specifying the user-friendly path of the child
            subtrainee for execution of method.
        child_id : str, optional
            Unique id of child trainee to execute method. Ignored if
            child_name_path is specified.
        as_external : bool
            Applicable only to 'load' and 'save' methods and if specifying
            child_name_path or child_id.
            For 'save', stores the child out as an independent trainee and removes
            it as a contained entity.
            For 'load' updates hierarchy by adding the child as an independently
            stored trainee to the hierarchy without loading the trainee as a
            subtrainee.
        load_external_trainee_id : str, optional
            Trainee id of trainee being loaded, must be specified only
            when method is 'load' and as_external is true.
        trainee_id : str
            The id of the Trainee to execute methods on.

        Returns
        -------
        object
            Whatever output the executed method returns.
        """
        return self._execute(
            trainee_id, "execute_on_subtrainee",
            {
                "method": method,
                "as_external": as_external,
                "child_name_path": child_name_path,
                "child_id": child_id,
                "payload": payload,
                "load_external_trainee_id": load_external_trainee_id
            })

    @classmethod
    def _deserialize(cls, payload: Union[str, bytes]):
        """Deserialize core response."""
        try:
            deserialized_payload = json.loads(payload)
            if isinstance(deserialized_payload, dict):
                if deserialized_payload.get('status') != 'ok':
                    # If result is an error, raise it
                    errors = deserialized_payload.get('errors') or []
                    if errors:
                        # Raise first error
                        raise HowsoError(errors[0].get('detail'))
                    else:
                        # Unknown error occurred
                        raise HowsoError('An unknown error occurred while '
                                         'processing the core operation.')

                warning_list = deserialized_payload.get('warnings') or []
                for w in warning_list:
                    warnings.warn(w.get('detail'), category=HowsoWarning)

                return deserialized_payload.get('payload')
            return deserialized_payload
        except HowsoError:
            raise
        except Exception:  # noqa: Deliberately broad
            raise HowsoError('Failed to deserialize the core response.')

    def _execute(self, handle: str, label: str, payload: Any) -> Any:
        """
        Execute label in core.

        Parameters
        ----------
        handle : str
            The entity handle of the Trainee
        label : str
            The label to execute.
        payload : Any
            The payload to send to label.

        Returns
        -------
        Any
            The label's response.
        """
        payload = sanitize_for_json(payload)
        payload = self._remove_null_entries(payload)
        try:
            result = self.amlg.execute_entity_json(
                handle, label, json.dumps(payload))
        except ValueError as err:
            raise HowsoError(
                'Invalid payload - please check for infinity or NaN '
                f'values: {err}')

        if result is None or len(result) == 0:
            return None
        return self._deserialize(result)

    def _execute_sized(self, handle: str, label: str, payload: Any) -> Tuple[Any, int, int]:
        """
        Execute label in core and return payload sizes.

        Parameters
        ----------
        handle : str
            The entity handle of the Trainee
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
        payload = sanitize_for_json(payload)
        payload = self._remove_null_entries(payload)
        try:
            json_payload = json.dumps(payload)
            result = self.amlg.execute_entity_json(
                handle, label, json_payload)
        except ValueError as err:
            raise HowsoError(
                'Invalid payload - please check for infinity or NaN '
                f'values: {err}')

        if result is None or len(result) == 0:
            return None, len(json_payload), 0
        return self._deserialize(result), len(json_payload), len(result)

    @staticmethod
    def _remove_null_entries(payload) -> Dict:
        """Remove keys from dict whose value is None."""
        return dict((k, v) for k, v in payload.items() if v is not None)

    @classmethod
    def escape_filename(cls, s: str) -> str:
        """Escape filename."""
        escaped = ""
        i = 0
        for i in range(len(s)):
            if cls._is_char_safe(ord(s[i])):
                escaped += s[i]
            else:
                escaped += cls._escape_char(s[i])

        return escaped

    @classmethod
    def unescape_filename(cls, s: str) -> str:
        """Unescape filename."""
        unescaped = ""
        i = 0
        while i < len(s):
            if s[i] == '_' and (i + 2) < len(s):
                unescaped += cls._char_value_from_escape_hex(
                    s[i + 1], s[i + 2])
                i += 3
            else:
                unescaped += s[i]
                i += 1
        return unescaped

    @staticmethod
    def _is_char_safe(c):
        # UTF-8 chars below zero (U+0030) are unsafe
        if c < 0x30:
            return False
        # Chars between 0 and 9 are ok
        if c <= 0x39:
            return True
        # Chars above 9 (U+0039) and below A (U+0041) are unsafe
        if c < 0x41:
            return False
        # Chars between A and Z are ok
        if c <= 0x5A:
            return True
        # Chars between Z and a (exclusive) are unsafe
        if c < 0x61:
            return False
        # Chars between a and z are ok
        if c <= 0x7A:
            return True

        # Any other char is unsafe
        return False

    @classmethod
    def _escape_char(cls, c):
        """Escape character."""
        low = cls._decimal_to_hex(15 & ord(c))
        high = cls._decimal_to_hex(15 & (ord(c) >> 4))
        return '_' + high + low

    @staticmethod
    def _decimal_to_hex(c):
        """Decimal to hex."""
        if c >= 10:
            return chr(c - 10 + ord('a'))
        return chr(c + ord('0'))

    @classmethod
    def _char_value_from_escape_hex(cls, high, low):
        """Character code from hex."""
        chr_int_value = cls._hex_to_decimal(low) + (
            (cls._hex_to_decimal(high) << 4) & 240)
        return chr(chr_int_value)

    @staticmethod
    def _hex_to_decimal(c):
        """Convert hex to decimal."""
        if c >= '0':
            if c <= '9':
                return ord(c) - ord('0')
            if 'a' <= c <= 'f':
                return ord(c) - ord('a') + 10
            if 'A' <= c <= 'F':
                return ord(c) - ord('A') + 10

        # Invalid and possibly unsafe char is not a hex value, return 0 as
        # having no value
        return 0

    @classmethod
    def download_amlg(cls, config):
        """
        Download amalgam binaries.

        Requires the howso-build-artifacts dependency.

        Parameters
        ----------
        config : dict
            The amalgam configuration options.

        Returns
        -------
        Path
            The path to the downloaded amalgam directory. Or None if nothing
            was downloaded.
        """
        # Since direct client may be distributed without build downloads ..
        try:
            from howso.build.artifacts.repo import HowsoArtifactService  # noqa # type: ignore
        except ImportError as err:
            raise ImportError(
                "Amalgam Download functionality only available "
                "if howso-build-artifacts is installed"
            ) from err

        if config is None:
            raise ValueError("config may not be None")

        version = config.get('version', 'latest')
        api_key = config.get('download_apikey')
        repo = config.get('repo')

        service_config = {}
        if api_key:
            service_config['HOWSO_ARTIFACTORY_APIKEY'] = api_key
        if repo:
            repo_path = f'{repo}/amalgam/'
            service_config['HOWSO_AMALGAM_DOWNLOAD_PATH'] = repo_path

        service = HowsoArtifactService(service_config)
        download_dir = service.download_amalgam(
            version=version,
            operating_system=config.get('os'),
            architecture=config.get('arch')
        )

        _logger.info(f'Downloaded amalgam version: {version}')
        return download_dir

    @classmethod
    def download_core(cls, config):
        """
        Download core binaries.

        Requires the howso-build-artifacts dependency.

        Parameters
        ----------
        config : dict
            The core configuration options.

        Returns
        -------
        Path
            The path to the downloaded core directory. Or None if nothing
            was downloaded.
        """
        # Since direct client may be distributed without build downloads ..
        try:
            from howso.build.artifacts.repo import HowsoArtifactService  # noqa # type: ignore
        except ImportError as err:
            raise ImportError(
                "Amalgam Download functionality only available "
                "if howso-build-artifacts is installed"
            ) from err

        version = config.pop('version', 'latest')
        api_key = config.pop('download_apikey', None)
        repo = config.get('repo')

        service_config = {}
        if api_key:
            service_config['HOWSO_ARTIFACTORY_APIKEY'] = api_key
        if repo:
            repo_path = f'{repo}/howso-core/'
            service_config['HOWSO_CORE_DOWNLOAD_PATH'] = repo_path

        service = HowsoArtifactService(service_config)
        download_dir = service.download_core(version=version)

        _logger.info(f'Downloaded core version: {version}')
        return download_dir

    @staticmethod
    def default_library_ext() -> str:
        """Returns the default library extension based on runtime os."""
        if platform.system().lower() == 'windows':
            return ".dll"
        elif platform.system().lower() == 'darwin':
            return ".dylib"
        else:
            return ".so"
