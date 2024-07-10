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
                f"This may be due to incorrect filepaths to the Howso "
                f"binaries or camls, or the Trainee already exists.")
        self.execute(trainee_id, "initialize", {
            "trainee_id": trainee_id,
            "filepath": str(self._howso_dir) + '/',
        })

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

        features = self.execute(trainee_id, "get_feature_attributes", {})
        loaded_trainee = Trainee(
            name=trainee_name,
            id=trainee_id,
            features=features,
            persistence=persistence,
            metadata=trainee_meta,
        )
        return internals.postprocess_trainee(loaded_trainee)

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
        trainee_id = self._resolve_trainee(trainee_id)
        if self.configuration.verbose:
            print(f'Getting Trainee with id: {trainee_id}')
        return self._get_trainee_from_engine(trainee_id)

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

        # Unload the trainee from engine
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
        trainee_id = self._resolve_trainee(trainee_id)
        original_trainee = self.trainee_cache.get(trainee_id)

        new_trainee_id = new_trainee_id or new_trainee_name or str(uuid.uuid4())
        output = self.howso.copy(trainee_id, new_trainee_id)

        if self.verbose:
            print(f'Copying Trainee {trainee_id} to {new_trainee_id}')

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
            raise HowsoError(
                f'Failed to copy the Trainee "{trainee_id}". '
                f"This may be due to incorrect filepaths to the Howso "
                f'binaries or camls, or a Trainee "{new_trainee_name}" already exists.')

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

        trainee = self._get_trainee_from_engine(trainee_id)
        self.trainee_cache.set(trainee)

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
