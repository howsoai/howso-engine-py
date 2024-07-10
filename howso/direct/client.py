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
import multiprocessing
import operator
import os
from pathlib import Path
import platform
import typing as t
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)
import uuid
import warnings

import certifi
import numpy as np
from packaging.version import parse as parse_version
from pandas import DataFrame
import urllib3
from urllib3.util import Retry, Timeout

from amalgam.api import Amalgam
from howso import utilities as util
from howso.client import get_configuration_path
from howso.client.base import AbstractHowsoClient
from howso.client.cache import TraineeCache
from howso.client.configuration import HowsoConfiguration
from howso.client.exceptions import HowsoError, HowsoWarning, UnsupportedArgumentWarning
from howso.client.schemas import HowsoVersion, Project, Reaction, Session, Trainee
from howso.client.typing import Persistence
from howso.utilities import (
    build_react_series_df,
    internals,
    num_list_dimensions,
    ProgressTimer,
    replace_doublemax_with_infinity,
    serialize_cases,
    validate_case_indices,
    validate_list_shape
)

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

    A client which provides access to the Howso core endpoints
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
        config_path: Union[str, Path, None] = None,
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

    def check_version(self) -> Union[str, None]:
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
    def verbose(self) -> bool:
        """Get verbose flag."""
        return self.configuration.verbose

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

    def _resolve_trainee(self, trainee_id: str, **kwargs) -> str:
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

        Raises
        ------
        HowsoError
            If the requested Trainee is currently loaded by another core entity.
        """
        if trainee_id not in self.trainee_cache:
            self.acquire_trainee_resources(trainee_id)
        return trainee_id

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
                self.howso.persist(trainee_id)
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
                f'Failed to create the Trainee "{trainee_id}". '
                f"This may be due to incorrect filepaths to the Howso"
                f'binaries or camls, or the Trainee already exists.')
        self.execute(trainee_id, "initialize", {
            "trainee_id": trainee_id,
            "filepath": str(self._howso_dir) + '/',
        })

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
        file_path: Union[Path, str],
        clobber: bool = False,
    ) -> Tuple[bool, str]:
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
        name: Optional[str] = None,
        features: Optional[Mapping[str, Mapping]] = None,
        *,
        id: Optional[str | uuid.UUID] = None,
        library_type: Optional[Literal["st", "mt"]] = None,
        max_wait_time: Optional[int | float] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
        overwrite_trainee: bool = False,
        persistence: Persistence = "allow",
        project: Optional[str | Project] = None,
        resources: Optional[Mapping[str, Any]] = None,
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

        trainee_metadata = dict(
            name=name,
            persistence=persistence,
            metadata=metadata
        )
        new_trainee = Trainee(
            name=name,
            features=features,
            persistence=persistence,
            id=trainee_id,
            metadata=metadata
        )
        new_trainee = internals.preprocess_trainee(new_trainee)
        self.execute(trainee_id, "set_metadata", {"metadata": trainee_metadata})
        self.execute(trainee_id, "set_feature_attributes", {"feature_attributes": new_trainee.features})
        new_trainee.features = self.execute(trainee_id, "get_feature_attributes", {})
        new_trainee = internals.postprocess_trainee(new_trainee)

        if self.is_tracing_enabled(trainee_id):
            # If tracing is enabled, log the trainee version
            self.execute(trainee_id, "get_trainee_version", {})

        self.trainee_cache.set(new_trainee)
        return new_trainee

    def update_trainee(self, trainee: Union[Dict, Trainee]) -> Trainee:
        """
        Update an existing Trainee in the Howso service.

        Parameters
        ----------
        trainee : dict or Trainee
            A `Trainee` object defining the Trainee.

        Returns
        -------
        Trainee
            The `Trainee` object that was updated.
        """
        instance = Trainee.from_dict(trainee) if isinstance(trainee, dict) else trainee

        if not instance.id:
            raise ValueError("A trainee id is required.")

        self._auto_resolve_trainee(instance.id)
        if self.verbose:
            print(f'Updating trainee with id: {instance.id}')

        instance = internals.preprocess_trainee(instance)
        metadata = {
            'name': instance.name,
            'metadata': instance.metadata,
            'persistence': instance.persistence,
        }
        self.howso.set_metadata(instance.id, metadata)
        self.howso.set_feature_attributes(instance.id, instance.features)
        instance.features = self.howso.get_feature_attributes(instance.id)

        updated_trainee = internals.postprocess_trainee(instance)
        self.trainee_cache.set(updated_trainee)
        return updated_trainee

    def export_trainee(
        self,
        trainee_id: str,
        path_to_trainee: Optional[Union[Path, str]] = None,
        decode_cases: bool = False,
    ):
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
        """
        if self.verbose:
            print(f'Export trainee with id: {trainee_id}')

        self.howso.export_trainee(trainee_id, path_to_trainee, decode_cases)

    def upgrade_trainee(
        self,
        trainee_id: str,
        path_to_trainee: Optional[Union[Path, str]] = None,
        separate_files: bool = False
    ):
        """
        Upgrade a saved Trainee to current version.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        path_to_trainee : Path or str, optional
            The path to where the saved Trainee file is located.
        separate_files : bool, default False
            Whether to load each case from its individual file.
        """
        if self.verbose:
            print(f'Upgrade trainee with id: {trainee_id}')

        self.howso.upgrade_trainee(trainee_id, path_to_trainee, separate_files)

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
        if self.verbose:
            print(f'Getting trainee with id: {trainee_id}')
        self._auto_resolve_trainee(trainee_id)
        return self._get_trainee_from_core(trainee_id)

    def get_trainee_information(self, trainee_id: str) -> Dict:
        """
        Get information about the trainee.

        Including trainee version and configuration parameters.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.

        Returns
        -------
        Dict
            The Trainee information in the schema of:
            {
                "library_type": LIBRARY_TYPE,
                "version": {
                    "amalgam": AMALGAM_VERSION,
                    "trainee": TRAINEE_VERSION
                }
            }
        """
        self._auto_resolve_trainee(trainee_id)
        trainee_version = self.howso.get_trainee_version(trainee_id)
        amlg_version = self.howso.amlg.get_version_string().decode()
        library_type = 'st'
        if self.howso.amlg.library_postfix:
            library_type = self.howso.amlg.library_postfix[1:]

        version = {
            "amalgam": amlg_version,
            "trainee": trainee_version
        }

        return {
            "library_type": library_type,
            "version": version
        }

    def get_trainees(self, search_terms: Optional[str] = None) -> List[Dict]:
        """
        Return a list of all trainees.

        Parameters
        ----------
        search_terms : str
            Keywords to filter trainee list by.

        Returns
        -------
        list of Dict
            A list of the trainee identities with schema {"name": TRAINEE_NAME, "id": TRAINEE_ID}
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
        files = os.listdir(self.howso.default_save_path)
        for f in files:
            if not f.endswith(self.howso.ext):
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
        trainee_id: Optional[str] = None,
        *,
        file_path: Optional[Path | str] = None
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
        else:
            trainee_id = file_path.stem

        # Unload the trainee from core
        self.amlg.destroy_entity(trainee_id)
        self.trainee_cache.discard(trainee_id)

        if self.verbose:
            print(f'Deleting trainee with id {trainee_id}')

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
        new_trainee_name: Optional[str] = None,
        new_trainee_id: Optional[str] = None,
        *,
        library_type: Optional[Literal["st", "mt"]] = None,
        resources: Optional[Dict] = None,
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
        self._auto_resolve_trainee(trainee_id)
        original_trainee = self.trainee_cache.get(trainee_id)

        new_trainee_id = new_trainee_id or new_trainee_name or str(uuid.uuid4())
        output = self.howso.copy(trainee_id, new_trainee_id)

        if self.verbose:
            print(f'Copying trainee {trainee_id} to {new_trainee_id}')

        # copy in core succeeded
        if output and output.get('name') == new_trainee_id:
            # Create the copy trainee
            new_trainee = deepcopy(original_trainee)
            new_trainee.name = new_trainee_name
            new_trainee._id = new_trainee_id  # type: ignore
            metadata = {
                'name': new_trainee.name,
                'metadata': new_trainee.metadata,
                'persistence': new_trainee.persistence,
            }
            self.howso.set_metadata(new_trainee_id, metadata)
            self.trainee_cache.set(new_trainee)

            return new_trainee
        else:
            raise ValueError(
                f"Could not copy the trainee with name {trainee_id}. Possible "
                f"causes - howso couldn't find core binaries/camls or "
                f"{new_trainee_name} trainee already exists."
            )

    def copy_subtrainee(
        self,
        trainee_id: str,
        new_trainee_name: str,
        *,
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
        self._auto_resolve_trainee(trainee_id)

        self.howso.copy_subtrainee(
            trainee_id,
            new_trainee_name,
            source_id=source_id,
            source_name_path=source_name_path,
            target_id=target_id,
            target_name_path=target_name_path
        )

    def acquire_trainee_resources(
        self,
        trainee_id: str,
        *,
        max_wait_time: Optional[Union[int, float]] = None
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
            raise HowsoError("A trainee id is required.")
        if self.verbose:
            print(f'Acquiring resources for trainee with id: {trainee_id}')

        if trainee_id in self.trainee_cache:
            # Trainee is already loaded
            return

        ret = self.howso.load(trainee_id)

        if ret is None:
            raise HowsoError(f"Trainee '{trainee_id}' not found.")

        trainee = self._get_trainee_from_core(trainee_id)
        self.trainee_cache.set(trainee)

    def _get_trainee_from_core(self, trainee_id: str) -> Trainee:
        """
        Retrieve the core representation of a Trainee object.

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
        metadata = self.howso.get_metadata(trainee_id)
        if metadata is None:
            raise HowsoError(f"Trainee '{trainee_id}' not found.")

        persistence = metadata.get('persistence', 'allow')
        trainee_meta = metadata.get('metadata')
        trainee_name = metadata.get('name')

        features = self.howso.get_feature_attributes(trainee_id)
        loaded_trainee = Trainee(
            name=trainee_name,
            id=trainee_id,
            features=features,
            persistence=persistence,
            metadata=trainee_meta,
        )
        return internals.postprocess_trainee(loaded_trainee)

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
        if self.verbose:
            print(f'Releasing resources for trainee with id: {trainee_id}')
        try:
            cache_item = self.trainee_cache.get_item(trainee_id)
            trainee = cache_item['trainee']

            if trainee.persistence in ['allow', 'always']:
                # Persist on unload
                self.howso.persist(trainee_id)
            elif trainee.persistence == "never":
                raise HowsoError(
                    "Trainees set to never persist may not have their "
                    "resources released. Delete the Trainee instead.")
            self.trainee_cache.discard(trainee_id)
        except KeyError:
            # Trainee not cached, ignore
            pass
        self.howso.delete(trainee_id)

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
        if self.verbose:
            print(f'Saving trainee with id: {trainee_id}')

        if trainee_id in self.trainee_cache:
            trainee = self.trainee_cache.get(trainee_id)
            if trainee.persistence == 'never':
                raise AssertionError(
                    "Trainee is set to never persist. Update the trainee "
                    "persistence option to enable persistence.")
            # Enable auto persistence
            trainee.persistence = 'always'

        self.howso.persist(trainee_id)

    def remove_series_store(self, trainee_id: str, series: Optional[str] = None):
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
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print('Removing stored series from trainee with id: '
                  f'{trainee_id} and series with id: {series}')
        self.howso.remove_series_store(trainee_id, series)

    def get_trainee_sessions(self, trainee_id: str) -> List[Dict[str, str]]:
        """
        Get the sessions of a trainee.

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
        >>> print(cl.get_trainee_sessions(trainee.id))
        [{'id': '6c35e481-fb49-4178-a96f-fe4b5afe7af4', 'name': 'default'}]
        """
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Getting sessions from trainee with id: {trainee_id}')
        sessions = self.howso.get_sessions(trainee_id, attributes=['name', ])
        if isinstance(sessions, Iterable):
            return sessions
        else:
            return []

    def delete_trainee_session(self, trainee_id: str, session: str):
        """
        Deletes a session from a trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to delete the session from.
        session : str
            The id of the session to remove.
        """
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Deleting session {session} from trainee with id: '
                  f'{trainee_id}')
        self.howso.remove_session(trainee_id, session)
        self._auto_persist_trainee(trainee_id)

    def begin_session(
        self, name: str = "default", metadata: Optional[Dict] = None
    ) -> Session:
        """
        Begin a new session.

        Parameters
        ----------
        name : str, default "default"
            The name of the session.
        metadata : dict, optional
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
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("`metadata` must be a dict")

        if self.verbose:
            print('Starting new session')
        self._active_session = Session(
            id=str(uuid.uuid4()),
            name=name,
            metadata=metadata or dict(),
            created_date=datetime.now(timezone.utc),
            modified_date=datetime.now(timezone.utc),
        )
        return self._active_session

    def get_sessions(self, search_terms: Optional[str] = None) -> List[Session]:
        """
        Return a list of all accessible sessions.

        .. NOTE::
            Returns sessions from across all loaded trainees. (The metadata will
            include the `trainee_id` from which the session was retrieved from)

        Parameters
        ----------
        search_terms : str, optional
            Space or comma delimited search terms to filter results by.

        Returns
        -------
        list of Session
            The listing of session instances.
        """
        if self.verbose:
            print('Getting listing of sessions')
        filter_terms = []
        filtered_sessions = []
        if search_terms:
            filter_terms = search_terms.replace(',', ' ').split(' ')

        for trainee_id in self.trainee_cache.ids():
            sessions = self.howso.get_sessions(
                trainee_id, attributes=list(Session.attribute_map))
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
        if self.verbose:
            print(f'Getting session with id: {session_id}')

        if session_id == self.active_session.id:
            return self.active_session

        # Find session from most recently loaded trainee first
        loaded_trainees = list(self.trainee_cache.ids())
        loaded_trainees.reverse()

        session = None
        for trainee_id in loaded_trainees:
            try:
                session_data = self.howso.get_session_metadata(trainee_id, session_id)
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

    def update_session(self, session_id: str, *, metadata: Optional[Dict] = None) -> Session:
        """
        Update a session.

        .. NOTE::
            Updates the session across all loaded trainees.

        Parameters
        ----------
        session_id : str
            The id of the session to update metadata for.
        metadata : dict, optional
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
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("`metadata` must be a dict")
        if self.verbose:
            print(f'Updating session for session with id: {session_id}')

        updated_session = None
        modified_date = datetime.now(timezone.utc)
        # We remove the trainee_id since this may have been set by the
        # get_session(s) methods and is not needed to be stored in the model.
        if metadata is not None:
            metadata.pop('trainee_id', None)

        # Update session across all loaded trainees
        for trainee_id in self.trainee_cache.ids():
            try:
                session_data = self.howso.get_session_metadata(trainee_id, session_id)
                if session_data is None:
                    continue  # Not found
            except HowsoError:
                # When session is not found, continue
                continue
            session_data['metadata'] = metadata
            session_data['modified_date'] = modified_date
            self.howso.set_session_metadata(trainee_id, session_id, session_data)
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

    def react_series(  # noqa: C901
        self,
        trainee_id: str,
        *,
        action_features: Optional[Iterable[str]] = None,
        actions: Optional[Union[List[List[object]], DataFrame]] = None,
        batch_size: Optional[int] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        contexts: Optional[Union[List[List[object]], DataFrame]] = None,
        context_features: Optional[Iterable[str]] = None,
        continue_series: Optional[bool] = False,
        continue_series_features: Optional[Iterable[str]] = None,
        continue_series_values: Optional[Union[List[object], List[List[object]]]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        feature_bounds_map: Optional[Dict] = None,
        final_time_steps: Optional[Union[List[object], List[List[object]]]] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        init_time_steps: Optional[Union[List[object], List[List[object]]]] = None,
        initial_batch_size: Optional[int] = None,
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
        progress_callback: Optional[Callable] = None,
        series_context_features: Optional[Iterable[str]] = None,
        series_context_values: Optional[Union[List[object], List[List[object]]]] = None,
        series_id_tracking: Literal["dynamic", "fixed", "no"] = "fixed",
        series_index: Optional[str] = None,
        series_stop_maps: Optional[List[Dict[str, Dict]]] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None
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
            See parameter ``contexts`` in :meth:`HowsoDirectClient.react`.
        action_features: iterable of str
            See parameter ``action_features`` in :meth:`HowsoDirectClient.react`.
        actions: list of list of object or DataFrame
            See parameter ``actions`` in :meth:`HowsoDirectClient.react`.
        context_features: iterable of str
            See parameter ``context_features`` in :meth:`HowsoDirectClient.react`.
        input_is_substituted : bool, default False
            See parameter ``input_is_substituted`` in :meth:`HowsoDirectClient.react`.
        substitute_output : bool
            See parameter ``substitute_output`` in :meth:`HowsoDirectClient.react`.
        details: dict, optional
            See parameter ``details`` in :meth:`HowsoDirectClient.react`.
        desired_conviction: float
            See parameter ``desired_conviction`` in :meth:`HowsoDirectClient.react`.
        weight_feature : str
            See parameter ``weight_feature`` in :meth:`HowsoDirectClient.react`.
        use_case_weights : bool
            See parameter ``use_case_weights`` in :meth:`HowsoDirectClient.react`.
        case_indices: iterable of sequence of str, int
            See parameter ``case_indices`` in :meth:`HowsoDirectClient.react`.
        preserve_feature_values : iterable of str
            See parameter ``preserve_feature_values`` in :meth:`HowsoDirectClient.react`.
        new_case_threshold : str
            See parameter ``new_case_threshold`` in :meth:`HowsoDirectClient.react`.
        leave_case_out : bool
            See parameter ``leave_case_out`` in :meth:`HowsoDirectClient.react`.
        use_regional_model_residuals : bool
            See parameter ``use_regional_model_residuals`` in :meth:`HowsoDirectClient.react`.
        feature_bounds_map: dict of dict
            See parameter ``feature_bounds_map`` in :meth:`HowsoDirectClient.react`.
        generate_new_cases : {"always", "attempt", "no"}
            See parameter ``generate_new_cases`` in :meth:`HowsoDirectClient.react`.
        ordered_by_specified_features : bool
            See parameter ``ordered_by_specified_features`` in :meth:`HowsoDirectClient.react`.
        suppress_warning : bool
            See parameter ``suppress_warning`` in :meth:`HowsoDirectClient.react`.

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
        self._auto_resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features

        validate_list_shape(initial_features, 1, "initial_features", "str")
        validate_list_shape(initial_values, 2, "initial_values",
                            "list of object")
        validate_list_shape(initial_features, 1, "max_series_lengths", "num")
        validate_list_shape(series_stop_maps, 1, "series_stop_maps", "dict")

        validate_list_shape(series_context_features, 1, "series_context_features", "str")

        if continue_series_values and num_list_dimensions(continue_series_values) != 3:
            raise ValueError(
                "Improper shape of `continue_series_values` values passed. "
                "`continue_series_values` must be a 3d list of object.")
        if series_context_values and num_list_dimensions(series_context_values) != 3:
            raise ValueError(
                "Improper shape of `series_context_values` values passed. "
                "`series_context_values` must be a 3d list of object.")

        if continue_series_values is not None:
            continue_series = True

        action_features, actions, context_features, contexts = (
            self._preprocess_react_parameters(
                action_features=action_features,
                actions=actions,
                case_indices=case_indices,
                context_features=context_features,
                contexts=contexts,
                desired_conviction=desired_conviction,
                preserve_feature_values=preserve_feature_values,
                trainee_id=trainee_id,
                continue_series=continue_series,
            )
        )

        if action_features is not None and derived_action_features is not None:
            if not set(derived_action_features).issubset(set(action_features)):
                raise ValueError(
                    'Specified \'derived_action_features\' must be a subset of '
                    '\'action_features\'.')

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
                    serialize_cases(series, series_context_features,
                                    feature_attributes))

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
                    serialize_cases(series, continue_series_features,
                                    feature_attributes))

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
        initial_values = serialize_cases(initial_values, initial_features,
                                         feature_attributes)

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
                    '`case_indices`, `initial_values`, `max_series_lengths`'
                    ', or `series_stop_maps`, each must be of length 1 or the same '
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
                    contexts=contexts,
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
            if self.verbose:
                print(f'Batch series reacting on trainee with id: {trainee_id}')
            response = self._batch_react_series(
                trainee_id, react_params, total_size=total_size,
                batch_size=batch_size,
                initial_batch_size=initial_batch_size,
                progress_callback=progress_callback)
        else:
            if self.verbose:
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

        series_df = build_react_series_df(response, series_index=series_index)

        response = Reaction(series_df, response.get('details'))

        return response

    def _batch_react_series(  # noqa: C901
        self,
        trainee_id: str,
        react_params: dict,
        *,
        batch_size: Optional[int] = None,
        initial_batch_size: Optional[int] = None,
        total_size: int,
        progress_callback: Optional[Callable] = None
    ):
        """
        Make react series requests in batch.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
        react_params : dict
            The request object.
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

        actions = react_params.get('action_values')
        contexts = react_params.get('context_values')
        case_indices = react_params.get('case_indices')
        initial_values = react_params.get('initial_values')
        max_series_lengths = react_params.get('max_series_lengths')
        series_context_values = react_params.get('series_context_values')
        series_stop_maps = react_params.get('series_stop_maps')
        continue_values = react_params.get('continue_series_values')

        with ProgressTimer(total_size) as progress:
            batch_scaler = None
            gen_batch_size = None
            if not batch_size:
                if not initial_batch_size:
                    if self.howso.amlg.library_postfix[1:] == 'mt':
                        start_batch_size = max(multiprocessing.cpu_count(), 1)
                    else:
                        start_batch_size = 1
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
                    react_params['action_values'] = actions[
                        batch_start:batch_end]
                if contexts is not None and len(contexts) > 1:
                    react_params['context_values'] = contexts[
                        batch_start:batch_end]
                if case_indices is not None and len(case_indices) > 1:
                    react_params['case_indices'] = (
                        case_indices[batch_start:batch_end])
                if initial_values is not None and len(initial_values) > 1:
                    react_params['initial_values'] = (
                        initial_values[batch_start:batch_end])
                if (
                    max_series_lengths is not None and
                    len(max_series_lengths) > 1
                ):
                    react_params['max_series_lengths'] = (
                        max_series_lengths[batch_start:batch_end])
                if (
                    series_context_values is not None and
                    len(series_context_values) > 1
                ):
                    react_params['series_context_values'] = (
                        series_context_values[batch_start:batch_end])
                if series_stop_maps is not None and len(series_stop_maps) > 1:
                    react_params['series_stop_maps'] = (
                        series_stop_maps[batch_start:batch_end])
                if continue_values is not None and len(continue_values) > 1:
                    react_params['continue_series_values'] = (
                        continue_values[batch_start:batch_end])

                if react_params.get('desired_conviction') is not None:
                    react_params['num_series_to_generate'] = batch_size
                temp_result, in_size, out_size = self._react_series(
                    trainee_id, react_params)

                internals.accumulate_react_result(accumulated_result,
                                                  temp_result)
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

    def _react_series(self, trainee_id, react_params):
        """
        Make a single react series request.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        react_params : dict
            The request parameters.

        Returns
        -------
        dict
            The react series response.
        int
            The request payload size.
        int
            The response payload size.
        """
        batch_result, in_size, out_size = self.howso.batch_react_series(
            trainee_id, **react_params)

        if batch_result is None or batch_result.get('action_values') is None:
            raise ValueError('Invalid parameters passed to react_series.')

        ret = dict()
        batch_result = replace_doublemax_with_infinity(batch_result)

        # batch_result always has action_features and action_values
        ret['action_features'] = batch_result.pop('action_features') or []
        ret['action'] = batch_result.pop('action_values')

        # ensure all the details items are output as well
        for k, v in batch_result.items():
            ret[k] = [] if v is None else v

        return ret, in_size, out_size

    def append_to_series_store(
        self,
        trainee_id: str,
        series: str,
        contexts: Union[List[List[object]], DataFrame],
        *,
        context_features: Optional[Iterable[str]] = None
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
        context_features : iterable of str, optional
            The list of feature names for contexts.
        """
        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        validate_list_shape(contexts, 2, "contexts", "list of object",
                            allow_none=False)

        if context_features is None:
            context_features = internals.get_features_from_data(
                contexts,
                data_parameter='contexts',
                features_parameter='context_features'
            )

        if len(np.array(contexts).shape) == 1 and len(contexts) > 0:
            contexts = [contexts]

        # Preprocess contexts
        contexts = serialize_cases(
            contexts, context_features, cached_trainee.features)

        if self.verbose:
            print('Appending to series store for trainee with id: '
                  f'{trainee_id}, and series: {series}')

        self.howso.append_to_series_store(trainee_id,
                                          context_features=context_features,
                                          contexts=contexts,
                                          series=series)

    def react(  # noqa: C901
        self,
        trainee_id: str,
        *,
        action_features: Optional[Iterable[str]] = None,
        actions: Optional[Union[List[List[object]], DataFrame]] = None,
        allow_nulls: bool = False,
        batch_size: Optional[int] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        contexts: Optional[Union[List[List[object]], DataFrame]] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        exclude_novel_nominals_from_uniqueness_check: bool = False,
        feature_bounds_map: Optional[Dict] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        initial_batch_size: Optional[int] = None,
        input_is_substituted: bool = False,
        into_series_store: Optional[str] = None,
        leave_case_out: bool = False,
        new_case_threshold: Literal["max", "min", "most_similar"] = "min",
        num_cases_to_generate: int = 1,
        ordered_by_specified_features: bool = False,
        post_process_features: Optional[Iterable[str]] = None,
        post_process_values: Optional[Union[List[List[object]], DataFrame]] = None,
        preserve_feature_values: Optional[Iterable[str]] = None,
        progress_callback: Optional[Callable] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None,
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
            - global_case_feature_residual_convictions_robust : bool, optional
                If True, outputs this case's feature residual convictions for
                the global model. Computed as: global model feature residual
                divided by case feature residual. Uses robust calculations, which
                uses uniform sampling from the power set of features as the
                contexts for predictions.
            - global_case_feature_residual_convictions_full : bool, optional
                If True, outputs this case's feature residual convictions for
                the global model. Computed as: global model feature residual
                divided by case feature residual. Uses full calculations,
                which uses leave-one-out for cases for computations.
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
            - local_case_feature_residual_convictions_robust : bool, optional
                If True, outputs this case's feature residual convictions for
                the region around the prediction. Uses only the context
                features of the reacted case to determine that region.
                Computed as: region feature residual divided by case feature
                residual. Uses robust calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - local_case_feature_residual_convictions_full : bool, optional
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
        self._auto_resolve_trainee(trainee_id)
        trainee = self.trainee_cache.get(trainee_id)
        action_features, actions, context_features, contexts = (
            self._preprocess_react_parameters(
                action_features=action_features,
                actions=actions,
                case_indices=case_indices,
                context_features=context_features,
                contexts=contexts,
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
        post_process_values = serialize_cases(
            post_process_values, post_process_features,
            trainee.features)

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
                    'Specified \'derived_action_features\' must be a subset of '
                    '\'action_features\'.')

        if new_case_threshold not in [None, "min", "max", "most_similar"]:
            raise ValueError(
                f"The value '{new_case_threshold}' specified for the parameter "
                "`new_case_threshold` is not valid. It accepts one of the"
                " following values - ['min', 'max', 'most_similar',]"
            )

        if details is not None and 'robust_computation' in details:
            details['robust_influences'] = details['robust_computation']
            details['robust_residuals'] = details['robust_computation']
            del details['robust_computation']
            warnings.warn(
                'The detail "robust_computation" is deprecated and will be '
                'removed in a future release. Please use "robust_residuals" '
                'and/or "robust_influences" instead.', DeprecationWarning)

        if desired_conviction is None:
            if contexts is not None:
                for context in contexts:
                    if context is not None and \
                            (len(context) != len(context_features)):
                        raise ValueError(
                            "The number of provided context values in "
                            "`contexts` does not match the number of features "
                            "in `context_features`."
                        )
                total_size = len(contexts)
            else:
                total_size = len(case_indices)

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
                raise HowsoError("`num_cases_to_generate` must be an integer "
                                 "greater than 0.")
            total_size = num_cases_to_generate

            context_features, contexts = \
                self._preprocess_generate_parameters(
                    trainee_id,
                    action_features=action_features,
                    context_features=context_features,
                    contexts=contexts,
                    desired_conviction=desired_conviction,
                    num_cases_to_generate=num_cases_to_generate,
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
            if self.verbose:
                print(
                    'Batch reacting to context on trainee with id: '
                    f'{trainee_id}'
                )
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
            if self.verbose:
                print(f'Reacting to context on trainee with id: {trainee_id}')
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

        response = Reaction(response.get('action'), response.get('details'))

        return response

    def _batch_react(  # noqa: C901
        self,
        trainee_id: str,
        react_params: Dict,
        *,
        batch_size: Optional[int] = None,
        initial_batch_size: Optional[int] = None,
        total_size: int,
        progress_callback: Optional[Callable] = None
    ):
        """
        Make react requests in batch.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
        react_params : dict
            The request object.
        batch_size: int, optional
            Define the number of cases to react to at once. If left unspecified,
            the batch size will be determined automatically.
        initial_batch_size: int, optional
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

        actions = react_params.get('action_values')
        contexts = react_params.get('context_values')
        case_indices = react_params.get('case_indices')
        post_process_values = react_params.get('post_process_values')

        with ProgressTimer(total_size) as progress:
            gen_batch_size = None
            batch_scaler = None
            if not batch_size:
                # Scale the batch size automatically
                start_batch_size = initial_batch_size or self.react_initial_batch_size
                batch_scaler = self.batch_scaler_class(
                    start_batch_size, progress)
                gen_batch_size = batch_scaler.gen_batch_size()
                batch_size = next(gen_batch_size, None)

            while not progress.is_complete and batch_size is not None:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress, temp_result)
                batch_start = progress.current_tick
                batch_end = progress.current_tick + batch_size

                if actions is not None and len(actions) > 1:
                    react_params['action_values'] = actions[
                        batch_start:batch_end]
                if contexts is not None and len(contexts) > 1:
                    react_params['context_values'] = contexts[
                        batch_start:batch_end]
                if case_indices is not None and len(case_indices) > 1:
                    react_params['case_indices'] = (
                        case_indices[batch_start:batch_end])
                if (post_process_values is not None and
                        len(post_process_values) > 1):
                    react_params['post_process_values'] = (
                        post_process_values[batch_start:batch_end])

                if react_params.get('desired_conviction') is not None:
                    react_params['num_cases_to_generate'] = batch_size
                temp_result, in_size, out_size = self._react(
                    trainee_id, react_params)

                internals.accumulate_react_result(accumulated_result,
                                                  temp_result)
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

    def _react(self, trainee_id: str, react_params: Dict
               ) -> Tuple[Dict, int, int]:
        """
        Make a single react request.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
        react_params : dict
            The request parameters.

        Returns
        -------
        dict
            The react response.
        int
            The request payload size.
        int
            The response payload size.
        """
        ret, in_size, out_size = self.howso.batch_react(
            trainee_id, **react_params)

        # Action values and features should always be defined
        if not ret.get('action_values'):
            ret['action_values'] = []
        if not ret.get('action_features'):
            ret['action_features'] = []

        return ret, in_size, out_size

    def _should_react_batch(self, react_params: Dict, total_size: int) -> bool:
        """
        Determine if given react should be batched.

        Parameters
        ----------
        react_params : dict
            The react request parameters.
        total_size : int
            The size of the cases being reacted to.

        Returns
        -------
        bool
            Whether a react should be batched.
        """
        if react_params.get('desired_conviction') is not None:
            if total_size > self._react_generative_batch_threshold:
                return True
        else:
            if total_size > self._react_discriminative_batch_threshold:
                return True

        return False

    def _preprocess_react_parameters(
        self,
        *,
        action_features: Iterable[str],
        actions: List[object],
        case_indices: Iterable[Sequence[Union[str, int]]],
        context_features: Iterable[str],
        contexts: List[object],
        desired_conviction: float,
        preserve_feature_values: Iterable[str],
        trainee_id: str,
        continue_series: bool = False,
    ) -> Tuple[List[str], List[object], List[str], List[object]]:
        """
        Preprocess parameters for `react_` methods.

        Helper method to pre-process the parameters. Updates the passed-in
        parameters as necessary.

        Parameters
        ----------
        action_features: iterable of str
            See :meth:`HowsoDirectClient.react()`.
        actions: list of object
            See :meth:`HowsoDirectClient.react()`.
        context_features: iterable of str
            See :meth:`HowsoDirectClient.react()`.
        contexts: list of object
            See :meth:`HowsoDirectClient.react()`.
        desired_conviction : float
            See :meth:`HowsoDirectClient.react()`.
        case_indices: iterable of sequence of str, int
            See :meth:`HowsoDirectClient.react()`.
        preserve_feature_values: iterable of str
            See :meth:`HowsoDirectClient.react()`.
        trainee_id : str
            See :meth:`HowsoDirectClient.react()`.
        continue_series : bool
            See :meth:`HowsoDirectClient.react_series()`.

        Returns
        -------
        tuple
           Updated action_features, actions, context_features, contexts
        """
        # Validate case_indices if provided
        if case_indices is not None:
            validate_case_indices(case_indices)

        # Get cached trainee
        trainee = self.trainee_cache.get(trainee_id)

        # Preprocess contexts
        if contexts is not None and context_features is None:
            context_features = internals.get_features_from_data(
                contexts,
                data_parameter='contexts',
                features_parameter='context_features')
        contexts = serialize_cases(contexts, context_features, trainee.features)

        # Preprocess actions
        if actions is not None and action_features is None:
            validate_list_shape(actions, 2, "actions", "object")
            action_features = internals.get_features_from_data(
                actions,
                data_parameter='actions',
                features_parameter='action_features')
        actions = serialize_cases(actions, action_features, trainee.features)

        # validate discriminative-react only parameters
        if desired_conviction is None:
            validate_list_shape(contexts, 2, "contexts", "list of object")
            validate_list_shape(action_features, 1, "action_features", "str")
            validate_list_shape(context_features, 1, "context_features", "str")

            if self.verbose:
                print(f'Reacting to context on trainee with id: {trainee_id}')

            if contexts is None:
                # case_indices/preserve_feature_values are not necessary
                # when using continue_series, as initial_feature/values may be used
                if not continue_series and (
                    case_indices is None or preserve_feature_values is None
                ):
                    raise ValueError(
                        "If `contexts` are not specified, both `case_indices`"
                        " and `preserve_feature_values` must be specified."
                    )

        return action_features, actions, context_features, contexts

    def react_into_features(
        self,
        trainee_id: str,
        *,
        distance_contribution: Optional[Union[str, bool]] = False,
        familiarity_conviction_addition: Optional[Union[str, bool]] = False,
        familiarity_conviction_removal: Optional[Union[str, bool]] = False,
        features: Optional[Iterable[str]] = None,
        influence_weight_entropy: Union[bool, str] = False,
        p_value_of_addition: Optional[Union[str, bool]] = False,
        p_value_of_removal: Optional[Union[str, bool]] = False,
        similarity_conviction: Optional[Union[str, bool]] = False,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None,
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
        use_case_weights : bool, default False
            If set to True will scale influence weights by each
            case's weight_feature weight.
        """
        self._auto_resolve_trainee(trainee_id)
        validate_list_shape(features, 1, "features", "str")
        if self.verbose:
            print(f'Reacting into features on trainee with id: {trainee_id}')
        self.howso.react_into_features(
            trainee_id,
            features=features,
            familiarity_conviction_addition=familiarity_conviction_addition,
            familiarity_conviction_removal=familiarity_conviction_removal,
            influence_weight_entropy=influence_weight_entropy,
            p_value_of_addition=p_value_of_addition,
            p_value_of_removal=p_value_of_removal,
            similarity_conviction=similarity_conviction,
            distance_contribution=distance_contribution,
            weight_feature=weight_feature,
            use_case_weights=use_case_weights)
        self._auto_persist_trainee(trainee_id)

    def react_group(
        self,
        trainee_id: str,
        new_cases: Union[List[List[List[object]]], List[DataFrame]],
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

        features : iterable of str, optional
            An iterable of feature names to consider while calculating
            convictions.
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
        self._auto_resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features
        serialized_cases = None

        if num_list_dimensions(new_cases) != 3:
            raise ValueError(
                "Improper shape of `new_cases` values passed. "
                "`new_cases` must be a 3d list of object.")

        serialized_cases = []
        for group in new_cases:
            if features is None:
                features = internals.get_features_from_data(group)
            serialized_cases.append(serialize_cases(group, features,
                                                    feature_attributes))

        if self.verbose:
            print('Reacting to a set of cases on trainee with id: '
                  f'{trainee_id}')

        ret = self.howso.batch_react_group(
            trainee_id,
            features=features,
            new_cases=serialized_cases,
            familiarity_conviction_addition=familiarity_conviction_addition,
            familiarity_conviction_removal=familiarity_conviction_removal,
            kl_divergence_addition=kl_divergence_addition,
            kl_divergence_removal=kl_divergence_removal,
            p_value_of_addition=p_value_of_addition,
            p_value_of_removal=p_value_of_removal,
            distance_contributions=distance_contributions,
            weight_feature=weight_feature,
            use_case_weights=use_case_weights
        )

        return ret

    def react_aggregate(
        self,
        trainee_id: str,
        *,
        action_feature: Optional[str] = None,
        confusion_matrix_min_count: Optional[int] = None,
        context_features: Optional[Iterable[str]] = None,
        details: Optional[dict] = None,
        feature_influences_action_feature: Optional[str] = None,
        hyperparameter_param_path: Optional[Iterable[str]] = None,
        num_robust_influence_samples: Optional[int] = None,
        num_robust_residual_samples: Optional[int] = None,
        num_robust_influence_samples_per_case: Optional[int] = None,
        num_samples: Optional[int] = None,
        prediction_stats_action_feature: Optional[str] = None,
        residuals_hyperparameter_feature: Optional[str] = None,
        robust_hyperparameters: Optional[bool] = None,
        sample_model_fraction: Optional[float] = None,
        sub_model_size: Optional[int] = None,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None,
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
                features for computations. False removes cached values.
            - feature_residuals_full : bool, optional
                For each context_feature, use the full set of all other context_features to predict
                the feature. False removes cached values. When ``prediction_stats``
                in the ``details`` parameter is true, the Trainee will also calculate and cache the
                full feature residuals.
            - feature_residuals_robust : bool, optional
                For each context_feature, use the robust (power set/permutations) set of all other
                context_features to predict the feature. False removes cached values.
            - feature_contributions_full : bool, optional
                For each context_feature, use the full set of all other
                context_features to compute the mean absolute delta between
                prediction of action feature with and without the context features
                in the model. False removes cached values.
            - feature_contributions_robust : bool, optional
                For each context_feature, use the robust (power set/permutation)
                set of all other context_features to compute the mean absolute
                delta between prediction of the action feature with and without the
                context features in the model. False removes cached values.
            - feature_mda_full : bool, optional
                When True will compute Mean Decrease in Accuracy (MDA)
                for each context feature at predicting the action feature. Drop
                each feature and use the full set of remaining context features
                for each prediction. False removes cached values.
            - feature_mda_robust : bool, optional
                Compute Mean Decrease in Accuracy MDA by dropping each feature and using the
                robust (power set/permutations) set of remaining context features
                for each prediction. False removes cached values.
            - feature_feature_mda_permutation_full : bool, optional
                Compute MDA by scrambling each feature and using the
                full set of remaining context features for each prediction.
                False removes cached values.
            - feature_feature_mda_permutation_robust : bool, optional
                Compute MDA by scrambling each feature and using the
                robust (power set/permutations) set of remaining context features
                for each prediction. False removes cached values.
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
            - missing_value_accuracy_full : bool, optional
                The number of cases with missing values predicted to have missing values divided by the number
                of cases with missing values, applies to all features that contain missing values. Uses full
                calculations.
            - missing_value_accuracy_robust : bool, optional
                The number of cases with missing values predicted to have missing values divided by the number
                of cases with missing values, applies to all features that contain missing values. Uses robust
                calculations.
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
        use_case_weights : bool, default False
            If set to True will scale influence weights by each case's
            weight_feature weight.
        weight_feature : str, optional
            The name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.

        Returns
        -------
        dict of str to dict of str to float
            If specified, a map of feature to map of stat type to stat values is returned.
        """
        self._auto_resolve_trainee(trainee_id)

        validate_list_shape(context_features, 1, "context_features", "str")

        if isinstance(details, dict):
            if isinstance(details.get("action_condition_precision"), str):
                if details.get("action_condition_precision") not in self.SUPPORTED_PRECISION_VALUES:
                    warnings.warn(self.INCORRECT_PRECISION_VALUE_WARNING)

            if isinstance(details.get("context_condition_precision"), str):
                if details.get("context_condition_precision") not in self.SUPPORTED_PRECISION_VALUES:
                    warnings.warn(self.INCORRECT_PRECISION_VALUE_WARNING)

        if self.verbose:
            print(f'Reacting into trainee for trainee with id: {trainee_id}')

        stats = self.howso.react_aggregate(
            trainee_id,
            action_feature=action_feature,
            residuals_hyperparameter_feature=residuals_hyperparameter_feature,
            context_features=context_features,
            confusion_matrix_min_count=confusion_matrix_min_count,
            details=details,
            feature_influences_action_feature=feature_influences_action_feature,
            hyperparameter_param_path=hyperparameter_param_path,
            num_samples=num_samples,
            num_robust_influence_samples=num_robust_influence_samples,
            num_robust_residual_samples=num_robust_residual_samples,
            num_robust_influence_samples_per_case=num_robust_influence_samples_per_case,
            prediction_stats_action_feature=prediction_stats_action_feature,
            robust_hyperparameters=robust_hyperparameters,
            sample_model_fraction=sample_model_fraction,
            sub_model_size=sub_model_size,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )

        self._auto_persist_trainee(trainee_id)

        return stats

    def _preprocess_generate_parameters(  # noqa: C901
        self,
        trainee_id: str,
        *,
        action_features: Optional[Iterable[str]] = None,
        context_features: Optional[Iterable[str]] = None,
        contexts: Optional[List[object]] = None,
        desired_conviction: Optional[float] = None,
        num_cases_to_generate: Optional[int] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
    ) -> Tuple[List[str], List[List[object]]]:
        """
        Helper method to pre-process generative react parameters.

        Updates the passed-in parameters as necessary.

        Parameters
        ----------
        trainee_id : str
            See :meth:`HowsoDirectClient.react`.
        action_features: iterable of str, optionsl
            See :meth:`HowsoDirectClient.react`.
        context_features: iterable of str, optional
            See :meth:`HowsoDirectClient.react`.
        contexts: list of object, optional
            See :meth:`HowsoDirectClient.react`.
        desired_conviction: float, optional
            See :meth:`HowsoDirectClient.react`.
        num_cases_to_generate: int, optional
            See :meth:`HowsoDirectClient.react`.
        case_indices: Iterable of Sequence[Union[str, int]], optional
            See :meth:`HowsoDirectClient.react`.

        Returns
        -------
        tuple
            context_features, context
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
            validate_case_indices(case_indices)

        validate_list_shape(action_features, 1, "action_features", "str")

        if contexts is not None:
            if len(contexts) != 1 and len(contexts) != num_cases_to_generate:
                raise HowsoError(
                    "The number of case `contexts` provided does not match the "
                    "number of cases to generate."
                )

        if context_features and not contexts:
            raise HowsoError(
                "For generative reacts, when `context_features` are provided, "
                "`contexts` values must also be provided."
            )

        if context_features is not None or contexts is not None:
            if (
                context_features is None
                or contexts is None
                or not isinstance(contexts[0], Sized)
                or len(context_features) != len(contexts[0])
            ):
                raise HowsoError(
                    "The number of provided context values in `contexts` "
                    "does not match the number of features in "
                    "`context_features`."
                )

        if desired_conviction is not None and desired_conviction <= 0:
            raise HowsoError("Desired conviction must be greater than 0.")

        if self.verbose:
            print(f'Generating case from trainee with id: {trainee_id}')

        for i in range(num_cases_to_generate):
            context_values = None
            if contexts is not None:
                if len(contexts) == 1:
                    context_values = contexts[0]
                elif len(contexts) == num_cases_to_generate:
                    context_values = contexts[i]

            if context_features and (
                    not context_values or
                    not isinstance(context_values, Sized) or
                    len(context_values) != len(context_features)
            ):
                raise HowsoError(
                    f"The number of provided context values in `contexts[{i}]` "
                    "does not match the number of features in "
                    "`context_features`."
                )
        return context_features, contexts

    def get_trainee_session_indices(self, trainee_id: str, session: str
                                    ) -> List[int]:
        """
        Get list of all session indices for a specified session.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee get parameters from.
        session : str
            The id of the session to retrieve indices from.

        Returns
        -------
        list of int
            A list of the session indices for the session.
        """
        self._auto_resolve_trainee(trainee_id)
        return self.howso.get_session_indices(trainee_id, session)

    def get_trainee_session_training_indices(
        self,
        trainee_id: str,
        session: str
    ) -> List[int]:
        """
        Get list of all session training indices for a specified session.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee get parameters from.
        session : str
            The id of the session to retrieve indices from.

        Returns
        -------
        list of int
            A list of the session training indices for the session.
        """
        self._auto_resolve_trainee(trainee_id)
        result = self.howso.get_session_training_indices(trainee_id, session)
        if result is None:
            return []
        return result

    def get_hierarchy(self, trainee_id: str) -> Dict:
        """
        Output the hierarchy for a trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee get hierarchy from.

        Returns
        -------
        dict of {str: dict}
            Dictionary of the currently contained hierarchy as a nested dict
            with False for trainees that are stored independently.
        """
        self._auto_resolve_trainee(trainee_id)
        return self.howso.get_hierarchy(trainee_id)

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
        new_name : str,
            New name of child trainee
        child_id : str, optional
            Unique id of child trainee to rename. Ignored if child_name_path is specified
        child_name_path : list of str, optional
            List of strings specifying the user-friendly path of the child
            subtrainee to rename.
        """
        self._auto_resolve_trainee(trainee_id)
        return self.howso.rename_subtrainee(
            trainee_id,
            new_name=new_name,
            child_id=child_id,
            child_name_path=child_name_path
        )
