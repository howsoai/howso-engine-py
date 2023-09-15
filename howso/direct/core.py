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
from pkg_resources import resource_filename
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
    trainee_template_path : str, default `DEFAULT_CORE_PATH`
        Directory path to the trainee_template caml files.
    howso_fname : str, default "howso.caml"
        Name of the Howso caml file with extension.
    trainee_template_fname : str, default "trainee_template.caml"
        Name of the trainee template file with extension.
    write_log : str, optional
        Absolute path to write log file.
    print_log : str, optional
        Absolute path to print log file.
    trace: bool, default False
        If true, sets debug flag for amlg operations. This will generate an
        execution trace useful in debugging with the standard name of
        [HANDLE]_execution.trace.
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
        handle: Optional[str] = None,
        library_path: Optional[str] = None,
        gc_interval: int = 100,
        howso_path: Path = DEFAULT_CORE_PATH,
        trainee_template_path: Path = DEFAULT_CORE_PATH,
        howso_fname: str = "howso.caml",
        trainee_template_fname: str = "trainee_template.caml",
        write_log: Optional[str] = None,
        print_log: Optional[str] = None,
        trace: bool = False,
        sbf_datastore_enabled: bool = True,
        max_num_threads: int = 0,
        **kwargs
    ):
        self.handle = handle if handle is not None else self.random_handle()
        if kwargs.get("amlg_debug", None) is not None:
            if trace is None:
                trace = kwargs["amlg_debug"]
            _logger.warning(
                'The "amlg_debug" parameter is deprecated use "trace" instead.')

        self.trace = bool(trace)

        if write_log is not None:
            self.write_log = Path(write_log).expanduser()
        else:
            self.write_log = ''

        if print_log is not None:
            self.print_log = Path(print_log).expanduser()
        else:
            self.print_log = ''

        # The parameters to pass to the Amalgam object - compiled here, so that
        # they can be merged with config file params.
        amlg_params = {
            'library_path': library_path,
            'gc_interval': gc_interval,
            'sbf_datastore_enabled': sbf_datastore_enabled,
            'max_num_threads': max_num_threads,
            'trace': self.trace,
            'execution_trace_file': self.handle + "_execution.trace",
        }

        try:
            # merge parameters from config.yml - favoring the configured params
            amlg_params_intersection = amlg_params.keys(
            ) & kwargs['amalgam'].keys()
            # Warn that there are conflicts
            if amlg_params_intersection:
                _logger.warning(
                    "The following parameters from configuration file will "
                    "override the Amalgam parameters set in the code: " +
                    str(amlg_params_intersection)
                )
            amlg_params = {**amlg_params, **kwargs['amalgam']}

        except KeyError:
            # No issue, if there is no amalgam key
            pass

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
        library_filename = f'amalgam{library_postfix}.{library_file_extension}'

        # Infer the architecture unless set, and normalize
        architecture = amlg_params.setdefault(
            'arch', platform.machine().lower())
        if architecture in ['x86_64', 'amd64']:
            architecture = 'amd64'
        elif architecture in ['aarch64_be', 'aarch64', 'armv8b', 'armv8l']:
            # see: https://stackoverflow.com/questions/45125516/possible-values-for-uname-m
            architecture = 'arm64'

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
        elif amlg_params.get('version'):
            versioned_amlg_location = Path(
                Path.home(), amlg_lib_dirname, operating_system,
                architecture, amlg_params.get('version'), 'lib',
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
                'execution_trace_dir', 'library_postfix'
            ]
        }
        self.amlg = Amalgam(**amlg_params)

        core_params = kwargs.get('core', {})

        # If download, then retrieve using howso-build-artifacts
        if core_params.get('download', False):
            self.howso_path = Path(
                self.download_core(core_params)).expanduser()
            self.trainee_template_path = self.howso_path
            self.default_save_path = Path(self.howso_path, 'trainee')

        # If version is set, but download not, use the default download location
        elif core_params.get('version'):
            # Set paths, ensuring tailing slash
            self.howso_path = Path(Path.home(), core_lib_dirname,
                                   core_params.get('version'))
            self.trainee_template_path = self.howso_path
            self.default_save_path = Path(self.howso_path, "trainee")

        # .... otherwise use default locations
        else:
            # Set paths, ensuring tailing slash
            self.howso_path = Path(howso_path).expanduser()
            self.trainee_template_path = Path(
                trainee_template_path).expanduser()
            self.default_save_path = Path(self.howso_path, "trainee")

        # Allow for trainee save directory to be overridden
        if core_params.get('persisted_trainees_dir'):
            self.default_save_path = Path(
                core_params.get("persisted_trainees_dir")).expanduser()
            _logger.debug(
                'Trainee save directory has been overridden to '
                f'{self.default_save_path}')
        else:
            # If no specific location provided, use current working directory.
            self.default_save_path = Path.cwd()

        # make save dir if doesn't exist
        if not self.default_save_path.exists():
            self.default_save_path.mkdir(parents=True)
        # make log dir(s) if they do not exist
        if self.write_log and not self.write_log.parent.exists():
            self.write_log.mkdir()
        if self.print_log and not self.print_log.parent.exists():
            self.print_log.mkdir()

        self.howso_fname = howso_fname
        self.trainee_template_fname = trainee_template_fname
        self.ext = trainee_template_fname[trainee_template_fname.rindex('.'):]

        self.howso_fully_qualified_path = Path(
            self.howso_path, self.howso_fname)
        if not self.howso_fully_qualified_path.exists():
            raise HowsoError(
                f'Howso core file {self.howso_fully_qualified_path} '
                'does not exist')
        _logger.debug(
            'Using howso-core location: '
            f'{self.howso_fully_qualified_path}')

        self.trainee_template_fully_qualified_path = Path(
            self.trainee_template_path, self.trainee_template_fname)
        if not self.trainee_template_fully_qualified_path.exists():
            raise HowsoError(
                'Howso core file '
                f'{self.trainee_template_fully_qualified_path} does not exist')
        _logger.debug('Using howso-core trainee template location: '
                      f'{self.trainee_template_fully_qualified_path}')

        if self.handle in self.get_entities():
            self.loaded = True
        else:
            self.loaded = self.amlg.load_entity(
                handle=self.handle,
                amlg_path=str(self.howso_fully_qualified_path),
                write_log=str(self.write_log),
                print_log=str(self.print_log)
            )

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
            "Trainee template:\t %s%s\n "
            "Howso Path:\t\t %s%s\n "
            "Save Path:\t\t %s\n "
            "Write Log:\t\t %s\n "
            "Print Log:\t\t %s\n "
            "Handle:\t\t\t %s\n %s")
        return template % (
            self.trainee_template_path,
            self.trainee_template_fname,
            self.howso_path,
            self.howso_fname,
            self.default_save_path,
            self.write_log,
            self.print_log,
            self.handle,
            str(self.amlg)
        )

    def version(self) -> str:
        """Return the version of the Howso Core."""
        if self.trainee_template_fname.split('.')[1] == 'amlg':
            version = "9.9.9"
        else:
            version = self._execute("version", {})
        return version

    def get_trainee_version(
        self,
        trainee_id: str,
        trace_version: Optional[str] = None
    ) -> str:
        """
        Return the version of the Trainee Template.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to get the version of.
        trace_version : str, optional
            A version comment to include in the trace file. Useful to capture
            client and amalgam versions.
        """
        return self._execute("get_trainee_version", {
            "trainee": trainee_id,
            "version": trace_version,
        })

    def create_trainee(self, trainee_id: str) -> Union[Dict, None]:
        """
        Create a Trainee using the Trainee Template.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to create.

        Returns
        -------
        dict
            A dict containing the name of the trainee that was created.
        """
        fname = self.trainee_template_fname.split('.')
        return self._execute("create_trainee", {
            "trainee": trainee_id,
            "filepath": f"{self.trainee_template_path}/",
            "trainee_template_filename": fname[0],
            "file_extension": fname[1],
        })

    def get_loaded_trainees(self) -> List[str]:
        """
        Get loaded Trainees.

        Returns
        -------
        list of str
            A list of trainee identifiers that are currently loaded.
        """
        return self._execute("get_loaded_trainees", {})

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
        ret = self._execute("load", {
            "trainee": trainee_id,
            "filename": filename,
            "filepath": filepath,
        })
        return ret

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
        return self._execute("save", {
            "trainee": trainee_id,
            "filename": filename,
            "filepath": filepath,
        })

    def delete(self, trainee_id: str) -> None:
        """
        Delete a Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to delete.
        """
        return self._execute("delete", {"trainee": trainee_id})

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
        return self._execute("copy", {
            "trainee": trainee_id,
            "target_trainee": target_trainee_id,
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
        return self._execute("remove_series_store", {
            "trainee": trainee_id,
            "series": series,
        })

    def clean_data(
        self,
        trainee_id: Optional[str],
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
        trainee_id : str, optional
            The identifier of the Trainee to clean.
        context_features : list of str, optional
            Only remove cases that don't have specified context features.
        action_features : list of str, optional
            Only remove cases that don't have specified action features.
        remove_duplicates : bool, default False
            If true, will remove duplicate cases (cases with identical values).
        """
        return self._execute("clean_data", {
            "trainee": trainee_id,
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
        return self._execute("set_substitute_feature_values", {
            "trainee": trainee_id,
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
        return self._execute("get_substitute_feature_values", {
            "trainee": trainee_id
        })

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
        return self._execute("set_session_metadata", {
            "trainee": trainee_id,
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
        return self._execute("get_session_metadata", {
            "trainee": trainee_id,
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
        return self._execute("get_sessions", {
            "trainee": trainee_id,
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
        return self._execute("remove_session", {
            "trainee": trainee_id,
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
        return self._execute("remove_feature", {
            "trainee": trainee_id,
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
        return self._execute("add_feature", {
            "trainee": trainee_id,
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
        return self._execute("get_num_training_cases", {"trainee": trainee_id})

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
            "trainee": trainee_id,
            "auto_analyze_enabled": auto_analyze_enabled,
            "analyze_threshold": analyze_threshold,
            "analyze_growth_factor": analyze_growth_factor,
            "auto_analyze_limit_size": auto_analyze_limit_size,
        }
        return self._execute("set_auto_analyze_params", {**kwargs, **params})

    def auto_analyze(self, trainee_id: str) -> None:
        """
        Auto-analyze the Trainee model.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        """
        return self._execute("auto_analyze", {"trainee": trainee_id})

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
        return self._execute("compute_feature_weights", {
            "trainee": trainee_id,
            "action_feature": action_feature,
            "context_features": context_features,
            "robust": robust,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })

    def set_feature_weights(
        self,
        trainee_id: str,
        feature_weights: Optional[Dict[str, float]] = None,
        action_feature: Optional[str] = None,
        use_feature_weights: bool = True
    ) -> None:
        """
        Set the weights for the features in the Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        action_feature : str, optional
            Action feature for which to set the specified feature weights for
        feature_weights : dict, optional
            A dictionary of feature names -> weight values.
            If not set, the feature weights are cleared in the model.
        use_feature_weights : bool, default True
            When set to true, forces the trainee to use the specified feature
            weights.
        """
        return self._execute("set_feature_weights", {
            "trainee": trainee_id,
            "action_feature": action_feature,
            "feature_weights_map": feature_weights,
            "use_feature_weights": use_feature_weights,
        })

    def set_feature_weights_matrix(
        self,
        trainee_id: str,
        feature_weights_matrix: Dict[str, Dict[str, float]],
        use_feature_weights: bool = True
    ) -> None:
        """
        Set the feature weights for all the features in the Trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        feature_weights_matrix : dict
            A dictionary of feature names to a dictionary of feature names to
            weight values.
        use_feature_weights : bool, default True
            When set to true, forces the trainee to use the specified feature
            weights
        """
        return self._execute("set_feature_weights_matrix", {
            "trainee": trainee_id,
            "feature_weights_matrix": feature_weights_matrix,
            "use_feature_weights": use_feature_weights,
        })

    def get_feature_weights_matrix(self, trainee_id: str
                                   ) -> Dict[str, Dict[str, float]]:
        """
        Get the full feature weights matrix.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.

        Returns
        -------
        dict
            A dictionary of action feature names to dictionary of feature names
            to feature weight.
        """
        return self._execute("get_feature_weights_matrix", {
            "trainee": trainee_id
        })

    def clear_conviction_thresholds(self, trainee_id: str) -> None:
        """
        Set the conviction thresholds to null.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        """
        return self._execute("clear_conviction_thresholds", {
            "trainee": trainee_id,
        })

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
        return self._execute("set_conviction_lower_threshold", {
            "trainee": trainee_id,
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
        return self._execute("set_conviction_upper_threshold", {
            "trainee": trainee_id,
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
        return self._execute("set_metadata", {
            "trainee": trainee_id,
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
        return self._execute("get_metadata", {"trainee": trainee_id})

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
        return self._execute("retrieve_extreme_cases_for_feature", {
            "trainee": trainee_id,
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
        ablatement_params: Optional[Dict[str, List[Any]]] = None,
        accumulate_weight_feature: Optional[str] = None,
        derived_features: Optional[Iterable[str]] = None,
        input_is_substituted: bool = False,
        series: Optional[str] = None,
        session: Optional[str] = None,
        train_weights_only: bool = False,
    ) -> Dict:
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
        ablatement_params : dict of str to list of object, optional
            Parameters describing how to ablate cases.
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
        train_weights_only : bool, default False
            When true, and accumulate_weight_feature is provided,
            will accumulate all of the cases' neighbor weights instead of
            training the cases into the model.

        Returns
        -------
        dict
            A dictionary containing the trained details.
        """
        return self._execute("train", {
            "trainee": trainee_id,
            "input_cases": input_cases,
            "features": features,
            "derived_features": derived_features,
            "session": session,
            "ablatement_params": ablatement_params,
            "series": series,
            "input_is_substituted": input_is_substituted,
            "accumulate_weight_feature": accumulate_weight_feature,
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
        return self._execute("impute", {
            "trainee": trainee_id,
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
        return self._execute("clear_imputed_session", {
            "trainee": trainee_id,
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
        return self._execute("get_cases", {
            "trainee": trainee_id,
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
        return self._execute("append_to_series_store", {
            "trainee": trainee_id,
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
        case_access_count_label: Optional[str] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        context_features: Optional[Iterable[str]] = None,
        context_values: Optional[List[List[object]]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        extra_audit_features: Optional[Iterable[str]] = None,
        feature_bounds_map: Optional[Dict] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        input_is_substituted: bool = False,
        into_series_store: Optional[str] = None,
        leave_case_out: bool = False,
        new_case_threshold: Literal["max", "min", "most_similar"] = "min",
        ordered_by_specified_features: bool = False,
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

        Returns
        -------
        dict
            The react result including audit details.
        """
        return self._execute("react", {
            "trainee": trainee_id,
            "context_features": context_features,
            "context_values": context_values,
            "action_features": action_features,
            "action_values": action_values,
            "details": details,
            "derived_action_features": derived_action_features,
            "derived_context_features": derived_context_features,
            "case_access_count_label": case_access_count_label,
            "extra_audit_features": extra_audit_features,
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
        case_access_count_label: Optional[str] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        context_features: Optional[Iterable[str]] = None,
        context_values: Optional[List[List[object]]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        extra_audit_features: Optional[Iterable[str]] = None,
        feature_bounds_map: Optional[Dict] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        input_is_substituted: bool = False,
        into_series_store: Optional[str] = None,
        leave_case_out: bool = False,
        new_case_threshold: Literal["max", "min", "most_similar"] = "min",
        num_cases_to_generate: Optional[int] = None,
        ordered_by_specified_features: bool = False,
        preserve_feature_values: Optional[Iterable[str]] = None,
        substitute_output: bool = True,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None
    ) -> Dict:
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
        """
        return self._execute("batch_react", {
            "trainee": trainee_id,
            "context_features": context_features,
            "context_values": context_values,
            "action_features": action_features,
            "action_values": action_values,
            "derived_context_features": derived_context_features,
            "derived_action_features": derived_action_features,
            "details": details,
            "case_access_count_label": case_access_count_label,
            "extra_audit_features": extra_audit_features,
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
        case_access_count_label: Optional[str] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        context_values: Optional[List[List[object]]] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        extra_audit_features: Optional[Iterable[str]] = None,
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
    ) -> Dict:
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
            list of features to condition just the first case in a
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
            3 * model_size, a 0 or less is no limit. Must provide
            either one for all series, or exactly one per series.
        derived_context_features : iterable of str, optional
            list of context features whose values should be computed
            from the entire series in the specified order. Must be
            different than context_features.
        derived_action_features : iterable of str, optional
            list of action features whose values should be computed
            from the resulting last row in series, in the specified order.
            Must be a subset of action_features.
        series_context_features : iterable of str, optional
            list of context features corresponding to
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
        """
        return self._execute("batch_react_series", {
            "trainee": trainee_id,
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
            "derived_context_features": derived_context_features,
            "derived_action_features": derived_action_features,
            "series_context_features": series_context_features,
            "series_context_values": series_context_values,
            "series_id_tracking": series_id_tracking,
            "output_new_series_ids": output_new_series_ids,
            "details": details,
            "case_access_count_label": case_access_count_label,
            "extra_audit_features": extra_audit_features,
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
        features: Optional[Iterable[str]] = None,
        familiarity_conviction_addition: bool = False,
        familiarity_conviction_removal: bool = False,
        p_value_of_addition: bool = False,
        p_value_of_removal: bool = False,
        similarity_conviction: bool = False,
        distance_contribution: bool = False,
        weight_feature: Optional[str] = None,
        use_case_weights: bool = False
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
        return self._execute("react_into_features", {
            "trainee": trainee_id,
            "features": features,
            "familiarity_conviction_addition": familiarity_conviction_addition,
            "familiarity_conviction_removal": familiarity_conviction_removal,
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
        *,
        new_cases: Optional[List[List[List[object]]]] = None,
        features: Optional[Iterable[str]] = None,
        trainees_to_compare: Optional[Iterable[str]] = None,
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
        new_cases : list of list of list of object or list, optional
            Specify a **set** using a list of cases to compute the conviction of
            groups of cases as shown in the following example.
        features : iterable of str, optional
            An iterable of feature names to consider while calculating
            convictions.
        trainees_to_compare : iterable of str, optional
            If specified ignores the 'new_cases' parameter and uses
            cases from this other specified trainee instead.
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
        return self._execute("batch_react_group", {
            "trainee": trainee_id,
            "features": features,
            "new_cases": new_cases,
            "trainees_to_compare": trainees_to_compare,
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
            whatever the model was analyzed for, i.e., action feature for MDA
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
        self._execute("react_into_trainee", {
            "trainee": trainee_id,
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
        return self._execute("compute_conviction_of_features", {
            "trainee": trainee_id,
            "features": features,
            "action_features": action_features,
            "familiarity_conviction_addition": familiarity_conviction_addition,
            "familiarity_conviction_removal": familiarity_conviction_removal,
            "weight_feature": weight_feature,
            "use_case_weights": use_case_weights,
        })

    def simplify_model(
        self,
        trainee_id: str,
        num_cases_to_remove: int,
        distribute_weight_feature: str
    ) -> None:
        """Perform data reduction."""
        return self._execute("simplify_model", {
            "trainee": trainee_id,
            "num_cases_to_remove": num_cases_to_remove,
            "distribute_weight_feature": distribute_weight_feature,
        })

    def forget_irrelevant_data(
        self,
        trainee_id: str,
        num_cases_to_remove: int,
        case_access_count_label: str,
        distribute_weight_feature: str
    ) -> None:
        """Perform data reduction."""
        return self._execute("forget_irrelevant_data", {
            "trainee": trainee_id,
            "num_cases_to_remove": num_cases_to_remove,
            "case_access_count_label": case_access_count_label,
            "distribute_weight_feature": distribute_weight_feature,
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
        return self._execute("get_session_indices", {
            "trainee": trainee_id,
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
        return self._execute("get_session_training_indices", {
            "trainee": trainee_id,
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
        return self._execute("set_internal_parameters", {
            "trainee": trainee_id,
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
        return self._execute("set_feature_attributes", {
            "trainee": trainee_id,
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
        return self._execute("get_feature_attributes", {"trainee": trainee_id})

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

        return self._execute("export_trainee", {
            "trainee": trainee_id,
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

        return self._execute("upgrade_trainee", {
            "trainee": trainee_id,
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
        params = {**kwargs, "trainee": trainee_id}
        return self._execute("analyze", params)

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
        return self._execute("get_feature_residuals", {
            "trainee": trainee_id,
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
        return self._execute("get_feature_mda", {
            "trainee": trainee_id,
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
        return self._execute("get_feature_contributions", {
            "trainee": trainee_id,
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
        return self._execute("get_prediction_stats", {
            "trainee": trainee_id,
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
        weight_feature: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get marginal stats for all features.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee.
        weight_feature : str, optional
            When specified, will attempt to return stats that were computed
            using this weight_feature.

        Returns
        -------
        dict of str to dict of str to float
            A map of feature names to map of stat type to stat values.
        """
        return self._execute("get_marginal_stats", {
            "trainee": trainee_id,
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
        return self._execute("set_random_seed", {
            "trainee": trainee_id,
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
        Get the parameters used by the Trainee. If 'action_feature',
        'context_features', 'mode', or 'weight_feature' are specified, then
        the best hyperparameters analyzed in the Trainee are the value of the
        'hyperparameter_map' key, otherwise this value will be the dictionary
        containing all the hyperparameter sets in the Trainee.


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
        return self._execute("get_internal_parameters", {
            "trainee": trainee_id,
            "action_feature": action_feature,
            "context_features": context_features,
            "mode": mode,
            "weight_feature": weight_feature,
        })

    def move_cases(
        self,
        trainee_id: str,
        target_trainee_id: str,
        num_cases: int = 1,
        *,
        case_indices: Optional[Iterable[Tuple[str, int]]] = None,
        condition: Optional[Dict] = None,
        condition_session: Optional[str] = None,
        distribute_weight_feature: Optional[str] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
        preserve_session_data: bool = False,
        session: Optional[str] = None
    ) -> Dict:
        """
        Moves cases from one trainee to another trainee.

        Parameters
        ----------
        trainee_id : str
            The identifier of the source Trainee.
        target_trainee_id : str
            The identifier of the target Trainee.
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

        Returns
        -------
        dict
            A dictionary with key 'count' for the number of moved cases.
        """
        result = self._execute("move_cases", {
            "trainee": trainee_id,
            "target_trainee": target_trainee_id,
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
        return self.move_cases(
            trainee_id,
            target_trainee_id=None,
            case_indices=case_indices,
            condition=condition,
            condition_session=condition_session,
            num_cases=num_cases,
            precision=precision,
            preserve_session_data=preserve_session_data,
            session=session,
            distribute_weight_feature=distribute_weight_feature,
        )

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
        result = self._execute("edit_cases", {
            "trainee": trainee_id,
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
        return self._execute("pairwise_distances", {
            "trainee": trainee_id,
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
        return self._execute("distances", {
            "trainee": trainee_id,
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
            "evaluate",
            {
                "trainee": trainee_id,
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
            "reset_parameter_defaults", {"trainee": trainee_id})

    @classmethod
    def _deserialize(cls, payload):
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

    def _get_label(self, label: str) -> Any:
        """Get label value from core."""
        result = self.amlg.get_json_from_label(self.handle, label)
        return result

    def _set_label(self, label: str, payload: Any) -> None:
        """Set label value in core."""
        payload = sanitize_for_json(payload)
        payload = self._remove_null_entries(payload)
        self.amlg.set_json_to_label(
            self.handle, label, json.dumps(payload))

    def _execute(self, label: str, payload: Any) -> Any:
        """
        Execute label in core.

        Parameters
        ----------
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
                self.handle, label, json.dumps(payload))
        except ValueError as err:
            raise HowsoError(
                'Invalid payload - please check for infinity or NaN '
                f'values: {err}')

        if result is None or len(result) == 0:
            return None
        return self._deserialize(result)

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
            from howso.build.artifacts.repo import HowsoArtifactService  # noqa
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
            from howso.build.artifacts.repo import HowsoArtifactService  # noqa
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
