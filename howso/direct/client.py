from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
import importlib.metadata
import inspect
import json
import logging
import operator
import os
from pathlib import Path
import platform
import typing as t
import uuid
import warnings

import certifi
from packaging.version import parse as parse_version
import urllib3
from urllib3.util import Retry, Timeout

from amalgam.api import Amalgam
from howso import utilities as util
from howso.client import get_configuration_path
from howso.client.base import AbstractHowsoClient
from howso.client.cache import TraineeCache
from howso.client.configuration import HowsoConfiguration
from howso.client.exceptions import HowsoError, HowsoWarning, UnsupportedArgumentWarning
from howso.client.schemas import (
    HowsoVersion,
    Project,
    Session,
    Trainee,
    TraineeRuntime,
    TraineeVersion,
)
from howso.client.typing import LibraryType, Persistence
from howso.utilities import internals

# Client version
CLIENT_VERSION = importlib.metadata.version('howso-engine')

# Configure howso base logger
logger = logging.getLogger('howso.direct')

_VERSION_CHECKED = False
DT_FORMAT_KEY = 'date_time_format'
HYPERPARAMETER_KEY = "hyperparameter_map"
VERSION_CHECK_HOST = "https://version-check.howso.com"
DEFAULT_ENGINE_PATH = Path(__file__).parent.parent.joinpath("howso-engine")

# Cache of trainee information shared across client instances
_trainee_cache = TraineeCache()


@contextmanager
def squelch_logs(log_level: int):
    """A context manager to temporarily disable logs."""
    _old_level = logging.root.manager.disable
    logging.disable(log_level)
    try:
        yield
    finally:
        logging.disable(_old_level)


class HowsoDirectClient(AbstractHowsoClient):
    """
    The direct Howso client.

    A client which provides access to the Howso Engine endpoints
    via a direct interface using dynamic libraries.

    Parameters
    ----------
    amalgam : Mapping, optional
        Keyword-argument overrides to the underlying Amalgam instance.
    config_path : str or Path or None, optional
        A configuration file in yaml format that specifies Howso engine
        settings.

        If not set, the client will also check in order of precedence:
            - HOWSO_CONFIG environment variable
            - The current directory for howso.yml, howso.yaml, config.yml
            - ~/.howso for howso.yml, howso.yaml, config.yml.
    debug : bool, default False
        Enable debug output.
    default_persist_path : str or Path, optional
        Override the default save location for created Trainees. If not specified the current working
        directory will be used.
    howso_path : str or Path, optional
        Directory path to the Howso caml files. Defaults to the built-in package location.
    howso_fname : str, default "howso.caml"
        Name of the Howso caml file with file extension.
    react_initial_batch_size: int, default 10
        The default number of cases to react to in the first batch
        for calls to :meth:`HowsoDirectClient.react`.
    trace : bool, default False
        When true, enables tracing of Amalgam operations. This will generate an
        execution trace file useful in debugging, the filename will use the
        standard name of `howso_[random 6 byte hex]_execution.trace`.
    train_initial_batch_size: int, default 100
        The default number of cases to train to in the first batch
        for calls to :meth:`HowsoDirectClient.train`.
    verbose : bool, default False
        Set verbose output.
    version_check : bool, default True
        Check if the latest version of Howso engine is installed.
    """

    #: The characters which are disallowed from being a part of a Trainee name or ID.
    BAD_TRAINEE_NAME_CHARS = {'..', '\\', '/', ':'}

    #: The supported values of precision for methods that accept it
    SUPPORTED_PRECISION_VALUES = ["exact", "similar"]
    INCORRECT_PRECISION_VALUE_WARNING = (
        "Supported values for 'precision' are \"exact\" and \"similar\". The "
        "operation will be completed as if the value of 'precision' is "
        "\"exact\"."
    )

    def __init__(
        self, *,
        amalgam: t.Optional[Mapping[str, t.Any]] = None,
        config_path: t.Optional[Path | str] = None,
        debug: bool = False,
        default_persist_path: t.Optional[Path | str] = None,
        howso_path: Path | str = DEFAULT_ENGINE_PATH,
        howso_fname: str = "howso.caml",
        react_initial_batch_size: int = 10,
        trace: bool = False,
        train_initial_batch_size: int = 100,
        verbose: bool = False,
        version_check: bool = True,
        **kwargs
    ):
        global _VERSION_CHECKED

        # Set the 'howso' logger level to debug
        if debug:
            # Don't alter if level already below debug
            if logger.level > logging.DEBUG or logger.level == 0:
                logger.setLevel(logging.DEBUG)

        with ThreadPoolExecutor(max_workers=1) as executor:
            if version_check and not _VERSION_CHECKED:
                _VERSION_CHECKED = True
                self.version_check_task = executor.submit(self.check_version)
                self.version_check_task.add_done_callback(self.report_version)

        super().__init__()

        # Show deprecation warnings to the user.
        warnings.filterwarnings("default", category=DeprecationWarning)

        # Load configuration
        config_path = get_configuration_path(config_path, verbose)
        self.configuration = HowsoConfiguration(config_path, verbose=verbose)
        self.debug = debug
        self._trace_enabled = bool(trace)
        self._trace_filename = f"howso_{internals.random_handle()}_execution.trace"
        self._howso_dir = Path(howso_path).expanduser()
        self._howso_filename = howso_fname
        self._howso_ext = Path(self._howso_filename).suffix or ".caml"
        self._react_generative_batch_threshold = 1
        self._react_discriminative_batch_threshold = 10
        self._react_initial_batch_size = react_initial_batch_size
        self._train_initial_batch_size = train_initial_batch_size

        if not self._howso_dir.is_dir():
            raise HowsoError(f"The provided 'howso_path' is not a directory: {self._howso_dir}")

        # Determine the default save directory
        if default_persist_path:
            self.default_persist_path = Path(default_persist_path).expanduser()
            logger.debug(f'The Trainee default save directory has been overridden to: {self.default_persist_path}')
        else:
            # If no specific location provided, use current working directory.
            self.default_persist_path = Path.cwd()

        if not self.default_persist_path.exists():
            # Make sure persist path exists
            self.default_persist_path.mkdir(parents=True)

        # Resolve path to engine caml
        self._howso_absolute_path = Path(self._howso_dir, self._howso_filename)
        if not self._howso_absolute_path.exists():
            raise HowsoError(f'Howso Engine file does not exist at: {self._howso_absolute_path}')
        logger.debug(f'Using Howso Engine file: {self._howso_absolute_path}')

        self.__init_amalgam(amalgam)
        self.begin_session()

    def __init_amalgam(self, options: t.Optional[Mapping[str, t.Any]] = None):
        """Initialize the Amalgam instance."""
        # The parameters to pass to the Amalgam instance
        amlg_params = {
            'library_path': None,
            'gc_interval': 100,
            'sbf_datastore_enabled': None,
            'max_num_threads': None,
            'trace': self._trace_enabled,
            'execution_trace_file': self._trace_filename,
        }
        if options:
            # Merge Amalgam override parameters - favoring the configured params
            if amlg_params_intersection := amlg_params.keys() & options.keys():
                # Warn that there are changes
                logger.warning(
                    "The following parameters from your configuration will "
                    f"override the default Amalgam parameters: {amlg_params_intersection}"
                )
            amlg_params.update(options)
        # Filter out invalid amlg_params, and instantiate
        allowed_amlg_params = inspect.signature(Amalgam).parameters.keys()
        if unknown_amlg_params := set(amlg_params) - set(allowed_amlg_params):
            warnings.warn(
                f"Unknown Amalgam() parameters were specified and ignored: {unknown_amlg_params}",
                UnsupportedArgumentWarning)
        amlg_params = {k: v for k, v in amlg_params.items() if k in allowed_amlg_params}
        self.amlg = Amalgam(**amlg_params)

    def check_version(self) -> str | None:
        """Check if there is a more recent version."""
        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                   ca_certs=certifi.where(),
                                   retries=Retry(total=1),
                                   timeout=Timeout(total=3),
                                   maxsize=10)
        url = f"{VERSION_CHECK_HOST}/v1/howso-engine?version={CLIENT_VERSION}"
        with squelch_logs(logging.WARNING + 1):
            response = http.request(method="GET", url=url)
        if HTTPStatus.OK <= response.status < HTTPStatus.MULTIPLE_CHOICES:
            payload = json.loads(response.data.decode('utf-8'))
            return payload.get('version')
        raise AssertionError("Not OK response.")

    def report_version(self, task: Future):
        """Report to end-user that there is a newer version available."""
        try:
            latest_version = task.result()
        except Exception:
            pass
        else:
            if latest_version and latest_version != CLIENT_VERSION:
                if parse_version(latest_version) > parse_version(CLIENT_VERSION):
                    logger.warning(
                        f"Version {latest_version} of Howso Engine™ is "
                        f"available. You are using version {CLIENT_VERSION}.")
                elif parse_version(latest_version) < parse_version(CLIENT_VERSION):
                    logger.debug(
                        f"Version {latest_version} of Howso Engine™ is "
                        f"available. You are using version {CLIENT_VERSION}. "
                        f"This is a pre-release version.")

    @property
    def react_initial_batch_size(self) -> int:
        """
        The default number of cases in the first react batch.

        Returns
        -------
        int
            The default number of cases to react to in the first batch.
        """
        return self._react_initial_batch_size

    @react_initial_batch_size.setter
    def react_initial_batch_size(self, initial_batch_size: int):
        """
        Set the default number of cases in the first react batch.

        Parameters
        ----------
        initial_batch_size : int
            The number of cases to react to in the first batch of
            :meth:`HowsoDirectClient.react`.
        """
        if isinstance(initial_batch_size, int):
            self._react_initial_batch_size = initial_batch_size
        else:
            raise ValueError("The initial batch size must be an integer.")

    @property
    def train_initial_batch_size(self) -> int:
        """
        The default number of cases in the first train batch.

        Returns
        -------
        int
            The default number of cases to train in the first batch.
        """
        return self._train_initial_batch_size

    @train_initial_batch_size.setter
    def train_initial_batch_size(self, initial_batch_size: int):
        """
        Set the default number of cases in the first train batch.

        Parameters
        ----------
        initial_batch_size : int
            The number of cases to train in the first batch of
            :meth:`HowsoDirectClient.train`.
        """
        if isinstance(initial_batch_size, int):
            self._train_initial_batch_size = initial_batch_size
        else:
            raise ValueError("The initial batch size must be an integer.")

    @property
    def active_session(self) -> Session:
        """
        Return the active session.

        Returns
        -------
        howso.client.schemas.Session
            The active session instance.
        """
        return deepcopy(self._active_session)

    @property
    def trainee_cache(self) -> TraineeCache:
        """
        Return the trainee cache.

        Returns
        -------
        howso.client.cache.TraineeCache
            The trainee cache.
        """
        return _trainee_cache

    @classmethod
    def _deserialize(cls, payload: str | bytes | None) -> t.Any:
        """Deserialize engine response."""
        if payload is None or len(payload) == 0:
            return None
        try:
            deserialized_payload = json.loads(payload)
            if isinstance(deserialized_payload, list):
                status = deserialized_payload[0]
                deserialized_payload = deserialized_payload[1]
                if status != 1:
                    # If result is an error, raise it
                    detail = deserialized_payload.get('detail') or []
                    if detail:
                        # Error detail can be either a string or a list of strings
                        if isinstance(detail, list):
                            raise HowsoError(detail[0])  # Raise first error
                        else:
                            raise HowsoError(detail)
                    else:
                        # Unknown error occurred
                        raise HowsoError('An unknown error occurred while '
                                         'processing the Howso Engine operation.')

                warning_list = deserialized_payload.get('warnings') or []
                for w in warning_list:
                    warnings.warn(w, category=HowsoWarning)

                return deserialized_payload.get('payload')
            return deserialized_payload
        except HowsoError:
            raise
        except Exception:  # noqa: Deliberately broad
            raise HowsoError('Failed to deserialize the Howso Engine response.')

    def _resolve_trainee(self, trainee_id: str, **kwargs) -> Trainee:
        """
        Resolve a Trainee and acquire its resources.

        Parameters
        ----------
        trainee_id : str
            The identifier of the Trainee to resolve.

        Returns
        -------
        str
            The normalized Trainee unique identifier.
        """
        if trainee_id not in self.trainee_cache:
            self.acquire_trainee_resources(trainee_id)
        return self.trainee_cache.get(trainee_id)

    def _auto_persist_trainee(self, trainee_id: str):
        """
        Automatically persists the Trainee if it has persistence set to True.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to persist.
        """
        try:
            trainee = self.trainee_cache.get(trainee_id)
            if trainee.persistence == 'always':
                self.amlg.store_entity(
                    handle=trainee_id,
                    amlg_path=self.resolve_trainee_filepath(trainee_id)
                )
        except KeyError:
            # Trainee not cached, ignore
            pass

    def _store_session(self, trainee_id: str, session: Session):
        """Store session details in a Trainee."""
        self.execute(trainee_id, "set_session_metadata", {
            "session": session.id,
            "metadata": session.to_dict(),
        })

    def _initialize_trainee(self, trainee_id: str):
        """Create a new Amalgam entity."""
        status = self.amlg.load_entity(
            handle=trainee_id,
            amlg_path=str(self._howso_absolute_path),
            persist=False,
            load_contained=True,
            escape_filename=False,
            escape_contained_filenames=False
        )
        if not status.loaded:
            raise HowsoError(
                f'Failed to initialize the Trainee "{trainee_id}": {status.message}')
        self.execute(trainee_id, "initialize", {
            "trainee_id": trainee_id,
            "filepath": str(self._howso_dir) + '/',
        })
        if self.is_tracing_enabled(trainee_id):
            # If tracing is enabled, log the trainee version
            self.execute(trainee_id, "get_trainee_version", {})

    def _get_trainee_from_engine(self, trainee_id: str) -> Trainee:
        """
        Retrieve the Howso Engine representation of a Trainee object.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to retrieve.

        Returns
        -------
        Trainee
            The requested Trainee.

        Raises
        ------
        HowsoError
            If no Trainee with the requested ID can be found.
        """
        metadata = self.execute(trainee_id, "get_metadata", {})
        if metadata is None:
            raise HowsoError(f"Trainee '{trainee_id}' not found.")

        persistence = metadata.get('persistence', 'allow')
        trainee_meta = metadata.get('metadata')
        trainee_name = metadata.get('name')

        return Trainee(
            name=trainee_name,
            id=trainee_id,
            persistence=persistence,
            metadata=trainee_meta,
        )

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
        return self.amlg.get_max_num_threads()

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
        if params.get('desired_conviction') is not None:
            if total_size > self._react_generative_batch_threshold:
                return True
        else:
            if total_size > self._react_discriminative_batch_threshold:
                return True
        return False

    def resolve_trainee_filepath(
        self,
        filename: str,
        *,
        filepath: t.Optional[str | Path] = None
    ) -> str:
        """
        Resolve the path to a persisted Trainee file.

        Parameters
        ----------
        filename : str
            The name of the Trainee file.
        filepath : str or Path, optional
            The directory of the file. If not provided, uses default persist path.

        Returns
        -------
        str
            The resolved path to the the Trainee file.
        """
        if not filename.endswith(self._howso_ext):
            filename += self._howso_ext
        if filepath is None:
            filepath = self.default_persist_path
        return str(Path(filepath, filename).expanduser())

    def execute(self, trainee_id: str, label: str, payload: t.Any, **kwargs) -> t.Any:
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
        payload = self.sanitize_for_json(payload, exclude_null=True)
        try:
            json_payload = json.dumps(payload)
            result = self.amlg.execute_entity_json(trainee_id, label, json_payload)
        except ValueError as err:
            raise HowsoError('Invalid payload - please check for infinity or NaN values') from err
        return self._deserialize(result)

    def execute_sized(self, trainee_id: str, label: str, payload: t.Any, **kwargs) -> tuple[t.Any, int, int]:
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
        payload = self.sanitize_for_json(payload, exclude_null=True)
        try:
            json_payload = json.dumps(payload)
            result = self.amlg.execute_entity_json(trainee_id, label, json_payload)
        except ValueError as err:
            raise HowsoError('Invalid payload - please check for infinity or NaN values') from err
        return self._deserialize(result), len(json_payload), len(result)

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
        return self._trace_enabled

    def get_version(self) -> HowsoVersion:
        """
        Return the Howso version.

        Returns
        -------
        HowsoVersion
           A version response that contains the version data for the current
           instance of Howso.
        """
        return HowsoVersion(client=CLIENT_VERSION)

    def check_name_valid_for_save(
        self,
        file_path: Path | str,
        clobber: bool = False,
    ) -> tuple[bool, str]:
        """
        Ensure that the given filename is a valid name for the host OS.

        Parameters
        ----------
        file_path : Path or str
            The full path of the desired Trainee.
        clobber : bool, default False
            If True, checks will pass if the file is writable even if it
            already exists.

        Returns
        -------
        bool
            Return True if the file has a valid filename, is a filepath (not a
            directory path), that the process (user) has sufficient permissions
            and, if `clobber` is False, also that the file does not already
            exist (optional check).
        str
            The reason. If the return is True, this will be 'OK'.

        """
        try:
            # Check for invalid chars in the whole path.
            if any((c for c in ["\0"] if c in str(file_path))):
                return False, 'Bad symbols'

            # Ensure file_path is a Path.
            if not isinstance(file_path, Path):
                file_path = Path(file_path)

            # Ensure that it resolves to an absolute path
            if not file_path.resolve(strict=False).is_absolute():
                return False, 'Not an absolute path'

            # Ensure that the parent directory exists and appears writable for
            # the /effective/ user on non-Windows.
            if platform.system().lower() != "windows":
                path = file_path.parent
                if not clobber and (
                    not path.exists() or
                    not os.access(path, os.W_OK, effective_ids=True,
                                  follow_symlinks=True)
                ):
                    return False, 'Cannot write to this path'

        except Exception as e:  # noqa: Deliberately broad
            return False, f'Exception {e} while checking file'
        else:
            return True, 'OK'

    def create_trainee(  # noqa: C901
        self,
        name: t.Optional[str] = None,
        features: t.Optional[Mapping[str, Mapping]] = None,
        *,
        id: t.Optional[str | uuid.UUID] = None,
        library_type: t.Optional[LibraryType] = None,
        max_wait_time: t.Optional[int | float] = None,
        metadata: t.Optional[MutableMapping[str, t.Any]] = None,
        overwrite_trainee: bool = False,
        persistence: Persistence = "allow",
        project: t.Optional[str | Project] = None,
        resources: t.Optional[Mapping[str, t.Any]] = None,
    ) -> Trainee:
        """
        Create a Trainee on the Howso service.

        A Trainee can be thought of as "model" in traditional ML sense.

        Parameters
        ----------
        name : str, optional
            A name to use for the Trainee.
        features : dict, optional
            The Trainee feature attributes.
        id : str or UUID, optional
            A custom unique identifier to use with the Trainee. If not specified
            the name will be used, or a new UUID if no name was provided.
        library_type : {"st", "mt"}, optional
            (Not implemented in this client)
        max_wait_time : int or float, optional
            (Not implemented in this client)
        metadata : dict, optional
            Arbitrary jsonifiable data to store along with the Trainee.
        overwrite_trainee : bool, default False
            If True, and if a trainee with the provided id already exists, the
            existing trainee will be deleted and a new trainee created.
        persistence : {"allow", "always", "never"}, default "allow"
            The requested persistence state of the Trainee.
        project : str or dict, optional
            (Not implemented in this client)
        resources : dict, optional
            (Not implemented in this client)

        Returns
        -------
        Trainee
            The `Trainee` object that was created.
        """
        if not id:
            # Default id to trainee name, or new uuid if no name
            id = name or uuid.uuid4()

        trainee_id = str(id)

        if features is None:
            features = {}

        # Check that the id is usable for saving later.
        if name:
            for sequence in self.BAD_TRAINEE_NAME_CHARS:
                if sequence in name:
                    success = False
                    reason = f'"{sequence}" is not permitted in trainee names'
                    break
            else:
                success, reason = True, 'OK'
            proposed_path: Path = self.default_persist_path.joinpath(name)
            if success:
                success, reason = self.check_name_valid_for_save(
                    proposed_path, clobber=overwrite_trainee)
            if not success:
                raise HowsoError(
                    f'Trainee file name "{proposed_path}" is not valid for '
                    f'saving (reason: {reason}).')

        # If overwriting the trainee, attempt to delete it first.
        if overwrite_trainee:
            try:
                util.dprint(
                    self.configuration.verbose,
                    f'Deleting existing Trainee "{trainee_id}" before creating.')
                self.amlg.destroy_entity(trainee_id)
            except Exception:  # noqa: Deliberately broad
                util.dprint(
                    self.configuration.verbose,
                    f'Unable to delete Trainee "{trainee_id}". Continuing.')
        elif trainee_id in self.trainee_cache:
            raise HowsoError(
                f'A Trainee already exists using the name "{trainee_id}".')

        if self.configuration.verbose:
            print('Creating new Trainee')
        # Initialize Amalgam entity
        self._initialize_trainee(trainee_id)

        # Store the metadata
        trainee_metadata = dict(
            name=name,
            persistence=persistence,
            metadata=metadata
        )
        self.execute(trainee_id, "set_metadata", {"metadata": trainee_metadata})

        # Set the feature attributes
        features = self.execute(trainee_id, "set_feature_attributes", {
            "feature_attributes": internals.preprocess_feature_attributes(features)
        })
        features = internals.postprocess_feature_attributes(features)

        # Cache and return the trainee
        new_trainee = Trainee(
            name=name,
            persistence=persistence,
            id=trainee_id,
            metadata=metadata
        )
        self.trainee_cache.set(new_trainee, feature_attributes=features)
        return new_trainee

    def update_trainee(self, trainee: Mapping | Trainee) -> Trainee:
        """
        Update an existing Trainee in the Howso service.

        Parameters
        ----------
        trainee : Mapping or Trainee
            A `Trainee` object defining the Trainee.

        Returns
        -------
        Trainee
            The `Trainee` object that was updated.
        """
        instance = Trainee.from_dict(trainee) if isinstance(trainee, Mapping) else trainee

        if not instance.id:
            raise ValueError("A Trainee id is required.")

        self._resolve_trainee(instance.id)
        if self.configuration.verbose:
            print(f'Updating Trainee with id: {instance.id}')

        metadata = {
            'name': instance.name,
            'metadata': instance.metadata,
            'persistence': instance.persistence,
        }
        self.execute(instance.id, "set_metadata", {"metadata": metadata})

        updated_trainee = deepcopy(instance)
        self.trainee_cache.set(updated_trainee)
        return updated_trainee

    def export_trainee(
        self,
        trainee_id: str,
        *,
        decode_cases: bool = False,
        filepath: t.Optional[Path | str] = None,
        path_to_trainee: t.Optional[Path | str] = None,
    ):
        """
        Export a saved Trainee's data to json files for migration.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        decode_cases : bool, default False.
            When True, decode (e.g., convert from epoch to datetime) any encoded
            feature values. When False, case feature values will be exported
            as is from the Trainee.
        filepath : Path or str, optional
            The directory to write the exported Trainee json files to.
        path_to_trainee : Path or str, optional
            Deprecated, use `filepath` instead.
        """
        if path_to_trainee is not None:
            warnings.warn(
                'The export trainee parameter `path_to_trainee` is deprecated and will be removed in '
                'a future release. Please use `filepath` instead.', DeprecationWarning)
            if filepath is None:
                filepath = path_to_trainee

        if filepath is None:
            filepath = self.default_persist_path
        filepath = Path(filepath).expanduser()

        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)
        elif not filepath.is_dir():
            raise ValueError(f'The export filepath "{filepath}" must be a directory.')

        if self.configuration.verbose:
            print(f'Exporting Trainee with id: {trainee_id}')

        self.execute(trainee_id, "export_trainee", {
            "trainee_filepath": f"{filepath}/",
            "trainee": str(trainee_id),
            "root_filepath": f"{self._howso_dir}/",
            "decode_cases": decode_cases,
        })

    def upgrade_trainee(
        self,
        trainee_id: str,
        *,
        filename: t.Optional[str] = None,
        filepath: t.Optional[Path | str] = None,
        path_to_trainee: t.Optional[Path | str] = None,
    ) -> Trainee:
        """
        Upgrade a saved Trainee to current version.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        filename : str, optional
            The base name of the exported Trainee json files. If not specified,
            uses the value of `trainee_id`. (e.g., [filename].meta.json)
        filepath : Path or str, optional
            The directory where the exported Trainee `.exp.json` and `.meta.json` files exist.
        path_to_trainee : Path or str, optional
            Deprecated, use `filepath` instead.

        Returns
        -------
        Trainee
            The Trainee that was upgraded.
        """
        if path_to_trainee is not None:
            warnings.warn(
                'The upgrade trainee parameter `path_to_trainee` is deprecated and will be removed in '
                'a future release. Please use `filepath` instead.', DeprecationWarning)
            if filepath is None:
                filepath = path_to_trainee

        if filepath is None:
            filepath = self.default_persist_path
        filepath = Path(filepath).expanduser()

        if not filepath.exists():
            raise ValueError(f'The upgrade filepath "{filepath}" does not exist.')

        if self.configuration.verbose:
            print(f'Upgrading Trainee with id: {trainee_id}')

        self._initialize_trainee(trainee_id)
        self.execute(trainee_id, "upgrade_trainee", {
            "trainee": filename or trainee_id,
            "trainee_json_filepath": f"{filepath}/",
            "root_filepath": f"{self._howso_dir}/",
        })
        trainee = self._get_trainee_from_engine(trainee_id)
        self.trainee_cache.set(trainee)
        self.resolve_feature_attributes(trainee_id)
        return trainee

    def get_trainee(self, trainee_id: str) -> Trainee:
        """
        Gets a trainee loaded in the Howso service.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.

        Returns
        -------
        Trainee
            A `Trainee` object representing the Trainee.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        if self.configuration.verbose:
            print(f'Getting Trainee with id: {trainee_id}')
        return self._get_trainee_from_engine(trainee_id)

    def get_trainee_runtime(self, trainee_id: str) -> TraineeRuntime:
        """
        Get information about the trainee.

        Including trainee version and configuration parameters.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.

        Returns
        -------
        TraineeRuntime
            The Trainee runtime details. Including Trainee version and
            configuration parameters.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        trainee_version = self.execute(trainee_id, "get_trainee_version", {})
        amlg_version = self.amlg.get_version_string().decode()
        library_type = 'mt'
        if self.amlg.library_postfix:
            library_type = self.amlg.library_postfix[1:]

        return TraineeRuntime(
            library_type=t.cast(LibraryType, library_type),
            tracing_enabled=self._trace_enabled,
            versions=TraineeVersion(trainee=trainee_version, amalgam=amlg_version)
        )

    def query_trainees(self, search_terms: t.Optional[str] = None) -> list[dict]:
        """
        Query accessible Trainees.

        Parameters
        ----------
        search_terms : str
            Keywords to filter Trainees by.

        Returns
        -------
        list of dict
            A list of the Trainee identities with schema {"name": TRAINEE_NAME, "id": TRAINEE_ID}
        """
        trainees = list()
        filter_terms = []
        if search_terms:
            filter_terms = search_terms.replace(',', ' ').split(' ')

        def is_match(name):
            # Check if name matches filter terms
            if filter_terms:
                return any((
                    str(term).lower() in name.lower()
                    for term in filter_terms
                ))
            return True

        # Collect in memory trainees
        for _, instance in self.trainee_cache.trainees():
            if is_match(instance.name):
                trainees.append(
                    {
                        "name": instance.name,
                        "id": instance.id
                    }
                )

        # Collect persisted trainees
        files = os.listdir(self.default_persist_path)
        for f in files:
            if not f.endswith(self._howso_ext):
                continue
            # remove the extension from the file name
            trainee_name = f[:f.rindex('.')]
            if (
                trainee_name not in self.trainee_cache and
                is_match(trainee_name)
            ):
                trainees.append(
                    {
                        "name": trainee_name,
                        "id": trainee_name
                    })

        return trainees

    def delete_trainee(
        self,
        trainee_id: t.Optional[str] = None,
        *,
        file_path: t.Optional[Path | str] = None
    ):
        """
        Delete a Trainee from the Howso service and filesystem.

        Includes all cases, model metadata, session data, persisted files, etc.

        Parameters
        ----------
        trainee_id : str, optional
            The ID of the Trainee. If full filepath with is provided, `trainee_id` will only be used
            to delete from memory.

        file_path : Path or str, optional
            The path of the file used to load the Trainee from. Used for deleting trainees from disk.

            The file path must end with a filename, but file path can be either an absolute path, a
            relative path or just the file name.

            If `trainee_id` is not provided, in addition to deleting from disk, will attempt to
            delete a Trainee from memory assuming the Trainee has the same name as the filename.

            If `file_path` is a relative path the absolute path will be computed
            appending the `file_path` to the CWD.

            If `file_path` is an absolute path, this is the absolute path that
            will be used.

            If `file_path` is just a filename, then the absolute path will be computed
            appending the filename to the CWD.
        """
        if file_path:
            if not isinstance(file_path, Path):
                file_path = Path(file_path)
            file_path = file_path.expanduser().resolve()

        if trainee_id:
            for sub in self.BAD_TRAINEE_NAME_CHARS:
                if sub in trainee_id:
                    raise ValueError(
                        f'"{sub}" is not permitted in trainee names for deletion.')
        elif file_path:
            trainee_id = file_path.stem
        else:
            raise ValueError("One of `trainee_id` or `file_path` must be provided.")

        if self.configuration.verbose:
            print(f'Deleting Trainee with id {trainee_id}')

        # Unload the trainee from engine
        self.amlg.destroy_entity(trainee_id)
        self.trainee_cache.discard(trainee_id)

        if file_path:
            # Either full filepath or filename
            if file_path.suffix:
                save_path = f"{file_path.parents[0]}/"
                trainee_id = file_path.stem
            # Just Directory
            else:
                raise ValueError("Filepath must end with a '.caml' filename.")
            if not file_path.is_absolute():
                file_path = self.default_persist_path.joinpath(file_path)

        else:
            save_path = self.default_persist_path

        trainee_path = Path(save_path, f'{trainee_id}{self._howso_ext}')

        # Delete Trainee
        if trainee_path.exists():
            trainee_path.unlink()

    def copy_trainee(
        self,
        trainee_id: str,
        new_trainee_name: t.Optional[str] = None,
        new_trainee_id: t.Optional[str] = None,
        *,
        library_type: t.Optional[LibraryType] = None,
        resources: t.Optional[dict] = None,
    ) -> Trainee:
        """
        Copies a trainee to a new trainee id in the Howso service.

        Parameters
        ----------
        trainee_id : str
            The trainee id of the trainee to be copied.
        new_trainee_name: str, optional
            The name of the new Trainee.
        new_trainee_id: str, optional
            The id of the new Trainee.

            If not provided, the id will be set to new_trainee_name
            (if provided), otherwise a new uuid4.
        library_type : str, optional
            (Not Implemented) The library type of the Trainee. If not specified,
            the new trainee will inherit the value from the original.
        resources : dict, optional
            (Not Implemented) Customize the resources provisioned for the
            Trainee instance. If not specified, the new trainee will inherit
            the value from the original.

        Returns
        -------
        Trainee
            The `Trainee` object that was created.

        Raises
        ------
        ValueError
            If the Trainee could not be copied.
        """
        original_trainee = self._resolve_trainee(trainee_id)
        trainee_id = original_trainee.id
        new_trainee_id = new_trainee_id or new_trainee_name or str(uuid.uuid4())

        if self.configuration.verbose:
            print(f'Copying Trainee {trainee_id} to {new_trainee_id}')

        is_cloned = self.amlg.clone_entity(
            handle=trainee_id,
            clone_handle=new_trainee_id,
        )
        if not is_cloned:
            raise HowsoError(
                f'Failed to copy the Trainee "{trainee_id}". '
                f"This may be due to incorrect filepaths to the Howso "
                f'binaries or camls, or a Trainee "{new_trainee_name}" already exists.')

        # Create the copy trainee
        new_trainee = deepcopy(original_trainee)
        new_trainee.name = new_trainee_name
        new_trainee._id = new_trainee_id  # type: ignore
        metadata = {
            'name': new_trainee.name,
            'metadata': new_trainee.metadata,
            'persistence': new_trainee.persistence,
        }
        self.execute(new_trainee_id, "set_metadata", {"metadata": metadata})
        # Add new trainee to cache
        feature_attributes = self.execute(new_trainee_id, "get_feature_attributes", {})
        feature_attributes = internals.postprocess_feature_attributes(feature_attributes)
        self.trainee_cache.set(new_trainee, feature_attributes=feature_attributes)

        return new_trainee

    def acquire_trainee_resources(
        self,
        trainee_id: str,
        *,
        max_wait_time: t.Optional[int | float] = None
    ):
        """
        Acquire resources for a trainee in the Howso service.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to acquire resources for.
        max_wait_time : int or float, optional
            (Not implemented) The number of seconds to wait to acquire trainee
            resources before aborting gracefully.

        Raises
        ------
        HowsoError
            If no Trainee with the requested ID can be found or loaded.
        """
        if trainee_id is None:
            raise HowsoError("A Trainee id is required.")
        if self.configuration.verbose:
            print(f'Acquiring resources for Trainee with id: {trainee_id}')

        if trainee_id in self.trainee_cache:
            # Trainee is already loaded
            return

        filepath = self.resolve_trainee_filepath(trainee_id)
        if not os.path.exists(filepath):
            raise HowsoError(
                f'Trainee not found. No Trainee file exists at: "{filepath}"', code="not_found")
        status = self.amlg.load_entity(
            handle=trainee_id,
            amlg_path=filepath,
            persist=False,
            load_contained=True,
            escape_filename=False,
            escape_contained_filenames=False,
        )
        if not status.loaded:
            raise HowsoError(
                f'Failed to acquire Trainee "{trainee_id}": {status.message}')

        # Cache the trainee details
        trainee = self._get_trainee_from_engine(trainee_id)
        feature_attributes = self.execute(trainee_id, "get_feature_attributes", {})
        feature_attributes = internals.postprocess_feature_attributes(feature_attributes)
        self.trainee_cache.set(trainee, feature_attributes=feature_attributes)

    def release_trainee_resources(self, trainee_id: str):
        """
        Release a trainee's resources from the Howso service.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to release resources for.

        Raises
        ------
        HowsoError
            If the requested Trainee has a persistence of "never".
        """
        if self.configuration.verbose:
            print(f'Releasing resources for Trainee with id: {trainee_id}')
        try:
            trainee = self.trainee_cache.get(trainee_id)

            if trainee.persistence in ['allow', 'always']:
                # Persist on unload
                self.amlg.store_entity(
                    handle=trainee_id,
                    amlg_path=self.resolve_trainee_filepath(trainee_id)
                )
            elif trainee.persistence == "never":
                raise HowsoError(
                    "Trainees set to never persist may not have their "
                    "resources released. Delete the Trainee instead.")
            self.trainee_cache.discard(trainee_id)
        except KeyError:
            # Trainee not cached, ignore
            pass
        self.amlg.destroy_entity(trainee_id)

    def persist_trainee(self, trainee_id: str):
        """
        Persists a Trainee in the Howso service storage.

        After persisting, the Trainee resources can be
        :func:`released <client.HowsoClient.release_trainee_resources>`.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to persist.

        Raises
        ------
        AssertionError
            If the requested Trainee's persistence is set to "never".
        """
        if self.configuration.verbose:
            print(f'Saving Trainee with id: {trainee_id}')

        if trainee_id in self.trainee_cache:
            trainee = self.trainee_cache.get(trainee_id)
            if trainee.persistence == 'never':
                raise AssertionError(
                    "Trainee is set to never persist. Update the trainee "
                    "persistence option to enable persistence.")
            # Enable auto persistence
            trainee.persistence = 'always'

        self.amlg.store_entity(
            handle=trainee_id,
            amlg_path=self.resolve_trainee_filepath(trainee_id)
        )

    def begin_session(self, name: str | None = "default", metadata: t.Optional[Mapping] = None) -> Session:
        """
        Begin a new session.

        Parameters
        ----------
        name : str or None, default "default"
            The name of the session.
        metadata : Mapping, optional
            Any key-value pair to store as custom metadata for the session.

        Returns
        -------
        howso.client.schemas.Session
            The new session instance.

        Raises
        ------
        TypeError
            If `name` is non-None and not a string or `metadata` is non-None
            and not a dictionary.
        """
        if not isinstance(name, str):
            raise TypeError("`name` must be a str")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("`metadata` must be a Mapping")

        if self.configuration.verbose:
            print('Starting new session')
        self._active_session = Session(
            id=str(uuid.uuid4()),
            name=name,
            metadata=metadata or dict(),
            created_date=datetime.now(timezone.utc),
            modified_date=datetime.now(timezone.utc),
        )
        return self._active_session

    def query_sessions(
        self,
        search_terms: t.Optional[str] = None,
        *,
        trainee: t.Optional[str | Trainee] = None,
        **kwargs
    ) -> list[Session]:
        """
        Return a list of all accessible sessions.

        .. NOTE::
            Returns sessions from across all loaded trainees. (The metadata will
            include the `trainee_id` from which the session was retrieved from)

        Parameters
        ----------
        search_terms : str, optional
            Space or comma delimited search terms to filter results by.
        trainee : str or Trainee, optional
            A Trainee to filter by.

        Returns
        -------
        list of Session
            The listing of session instances.
        """
        if self.configuration.verbose:
            print('Querying accessible sessions')
        filter_terms = []
        filtered_sessions = []
        if search_terms:
            filter_terms = search_terms.replace(',', ' ').split(' ')

        # Normalize trainee id filter
        if isinstance(trainee, Trainee):
            filter_trainee_id = str(trainee.id)
        else:
            filter_trainee_id = trainee

        for trainee_id in self.trainee_cache.ids():
            if filter_trainee_id and trainee_id != filter_trainee_id:
                continue
            sessions = self.execute(trainee_id, "get_sessions", {"attributes": list(Session.attribute_map)})
            if not sessions:
                continue

            for session in sessions:
                if filter_terms:
                    # Filter by search terms
                    for term in filter_terms:
                        if term.lower() in session.get('name', '').lower():
                            instance = Session.from_dict(session)
                            metadata = dict(instance.metadata) if instance.metadata else dict()
                            metadata['trainee_id'] = trainee_id
                            instance.metadata = metadata
                            filtered_sessions.append(instance)
                            break
                else:
                    instance = Session.from_dict(session)
                    metadata = dict(instance.metadata) if instance.metadata else dict()
                    metadata['trainee_id'] = trainee_id
                    instance.metadata = metadata
                    filtered_sessions.append(instance)
        return sorted(filtered_sessions,
                      key=operator.attrgetter('created_date'),
                      reverse=True)

    def get_session(self, session_id: str) -> Session:
        """
        Retrieve a session.

        .. NOTE::
            If multiple trainees are loaded, the session will be retrieved
            from the most recently loaded trainee that contains the requested
            session. (The metadata will include the `trainee_id` from which
            the session was retrieved from)

        Parameters
        ----------
        session_id : str
            The id of the session to retrieve.

        Returns
        -------
        Session
            The session instance.
        """
        if self.configuration.verbose:
            print(f'Getting session with id: {session_id}')

        if session_id == self.active_session.id:
            return self.active_session

        # Find session from most recently loaded trainee first
        loaded_trainees = list(self.trainee_cache.ids())
        loaded_trainees.reverse()

        session = None
        for trainee_id in loaded_trainees:
            try:
                session_data = self.execute(trainee_id, "get_session_metadata", {"session": session_id})
                if session_data is None:
                    continue  # Not found
            except HowsoError:
                # When session is not found, continue
                continue
            session = Session.from_dict(session_data)
            # Include trainee_id in the metadata
            metadata = dict(session.metadata) if session.metadata else dict()
            metadata['trainee_id'] = trainee_id
            session.metadata = metadata
            break
        if session is None:
            raise HowsoError("Session not found")
        return session

    def update_session(self, session_id: str, *, metadata: t.Optional[Mapping] = None) -> Session:
        """
        Update a session.

        .. NOTE::
            Updates the session across all loaded trainees.

        Parameters
        ----------
        session_id : str
            The id of the session to update metadata for.
        metadata : Mapping, optional
            Any key-value pair to store as custom metadata for the session.

        Returns
        -------
        howso.client.schemas.Session
            The updated session instance.

        Raises
        ------
        TypeError
            If `metadata` is non-None and not a dictionary.
        HowsoError
            If `session_id` is not found for the active session or any of
            the session(s) of a loaded Trainees.
        """
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("`metadata` must be a Mapping")
        if self.configuration.verbose:
            print(f'Updating session for session with id: {session_id}')

        updated_session = None
        modified_date = datetime.now(timezone.utc)
        # We remove the trainee_id since this may have been set by the
        # get_session(s) methods and is not needed to be stored in the model.
        if metadata is not None and 'trainee_id' in metadata:
            metadata = dict(metadata)
            metadata.pop('trainee_id', None)

        # Update session across all loaded trainees
        for trainee_id in self.trainee_cache.ids():
            try:
                session_data = self.execute(trainee_id, "get_session_metadata", {"session": session_id})
                if session_data is None:
                    continue  # Not found
            except HowsoError:
                # When session is not found, continue
                continue
            session_data['metadata'] = metadata
            session_data['modified_date'] = modified_date
            self.execute(trainee_id, "set_session_metadata", {
                "session": session_id,
                "metadata": session_data,
            })
            updated_session = Session.from_dict(session_data)

        if self.active_session.id == session_id:
            # Update active session
            self._active_session.metadata = metadata
            self._active_session._modified_date = modified_date  # type: ignore
            if updated_session is None:
                updated_session = self.active_session
        elif updated_session is None:
            # If session_id was not for the active session or any session
            # of loaded trainees, raise error
            raise HowsoError("Session not found")
        return updated_session
