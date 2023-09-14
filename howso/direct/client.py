from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from http import HTTPStatus
import json
import logging
import multiprocessing
import operator
import os
from pathlib import Path
import platform
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
from howso import utilities as util
from howso.client import __version__ as local_version, AbstractHowsoClient
from howso.client.cache import TraineeCache
from howso.client.exceptions import HowsoError
import howso.openapi.models as client_models
from howso.openapi.models import (
    Session,
    Trainee,
)
from howso.utilities import (
    internals,
    num_list_dimensions,
    ProgressTimer,
    replace_doublemax_with_infinity,
    serialize_cases,
    validate_case_indices,
    validate_list_shape,
)
from howso.utilities.feature_attributes.base import (
    SingleTableFeatureAttributes,
    MultiTableFeatureAttributes,
)
import numpy as np
from packaging.version import parse as parse_version
from pandas import DataFrame
from typing_extensions import Never
import urllib3
from urllib3.util import Retry, Timeout

from ._utilities import model_from_dict
from .core import HowsoCore

# Configure howso base logger
logger = logging.getLogger('howso.direct')

_VERSION_CHECKED = False
DT_FORMAT_KEY = 'date_time_format'
HYPERPARAMETER_KEY = "hyperparameter_map"
VERSION_CHECK_HOST = "https://version-check.howso.com"

# Cache of trainee information shared across client instances
_trainee_cache = TraineeCache()

# Cache of core entities shared across client instances
_core_cache = dict()


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
    howso_core : howso.direct.HowsoCore, optional
        A specified howso core direct interface object.

        If None, an interface will be generated using the provided handle.
    debug : bool, default False
        Set debug output.
    handle : str, optional
        The howso core entity handle to use.

        If None, :attr:`HowsoDirectClient.DEFAULT_HANDLE` will be used.
    verbose : bool, default False
        Set verbose output.
    """

    #: The default Howso core entity handle.
    DEFAULT_HANDLE = "howso"

    #: The characters which are disallowed from being a part of a Trainee name or ID.
    BAD_TRAINEE_NAME_CHARS = {'..', '\\', '/', ':'}

    def __init__(
        self,
        howso_core: Optional[HowsoCore] = None,
        *,
        debug: bool = False,
        handle: Optional[str] = None,
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
        handle = str(handle or self.DEFAULT_HANDLE)

        # Show deprecation warnings to the user.
        warnings.filterwarnings("default", category=DeprecationWarning)

        self.verbose = verbose
        self.debug = debug

        if howso_core is None:
            if handle not in _core_cache:
                _core_cache[handle] = HowsoCore(
                    self.get_unique_handle(handle),
                    **kwargs
                )
            self.howso = _core_cache[handle]
        elif isinstance(howso_core, HowsoCore):
            self.howso = howso_core
        else:
            raise ValueError("The client parameter howso_core must be "
                             "an instance of HowsoCore")
        self.batch_scaler_class = internals.BatchScalingManager
        self._active_session = None
        self._react_generative_batch_threshold = 1
        self._react_discriminative_batch_threshold = 10
        self.begin_session()

    def check_version(self) -> Union[str, None]:
        """Check if there is a more recent version."""
        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                   ca_certs=certifi.where(),
                                   retries=Retry(total=1),
                                   timeout=Timeout(total=3),
                                   maxsize=10)
        url = f"{VERSION_CHECK_HOST}/v1/howso-engine?version={local_version}"
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
            if latest_version and latest_version != local_version:
                if parse_version(latest_version) > parse_version(local_version):
                    logger.warning(
                        f"Version {latest_version} of Howso Engine™ is "
                        f"available. You are using version {local_version}.")
                elif parse_version(latest_version) < parse_version(local_version):
                    logger.debug(
                        f"Version {latest_version} of Howso Engine™ is "
                        f"available. You are using version {local_version}. "
                        f"This is a pre-release version.")

    @property
    def active_session(self) -> Session:
        """
        Return the active session.

        Returns
        -------
        howso.openapi.models.Session
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

    @staticmethod
    def get_unique_handle(handle: str) -> str:
        """
        Append a unique 6 byte hex to the input handle.

        Parameters
        ----------
        handle : str
            String to which a unique 6 byte hex string will appended.

        Returns
        -------
        str
            A unique alphanumeric handle consisting of the input string and
            a unique 6 byte hex string.
        """
        return f"{handle}-{HowsoCore.random_handle()}"

    def get_entities(self) -> List[str]:
        """
        Return a list of loaded core entities.

        Returns
        -------
        iterable of str
            The list of loaded entity names.
        """
        return self.howso.get_entities()

    def get_version(self) -> client_models.ApiVersion:
        """
        Return the Howso version.

        Returns
        -------
        howso.openapi.models.ApiVersion
           A version response that contains the version data for the current
           instance of Howso.
        """
        from howso.client import __version__ as client_version
        from howso.openapi import __api_version__ as api_version

        return client_models.ApiVersion(api=api_version,
                                        client=client_version)

    def _output_version_in_trace(self, trainee: str):
        """
        Instruct Howso core to retrieve the version of the Trainee.

        If debugging is enabled, this version will appear in the trace file.

        Parameters
        ----------
        trainee : str
            The ID of the Trainee that should retrieve the Howso version.
        """
        from howso.client import __version__
        amlg_version = self.howso.amlg.get_version_string()
        self.howso.version()
        trace_version = f"client: {__version__}  amalgam: {amlg_version}"

        # don't need to return the output, make the call to core in order for
        # the stack version to show up in the trace file.
        self.howso.get_trainee_version(trainee, trace_version)

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
        trainee: Trainee,
        *,
        library_type: Optional[Literal["st", "mt"]] = None,
        max_wait_time: Optional[Union[int, float]] = None,
        overwrite_trainee: bool = False,
        resources: Optional[Union[client_models.TraineeResources, Dict]] = None,
    ) -> Trainee:
        """
        Create a Trainee on the Howso service.

        A Trainee can be thought of as "model" in traditional ML sense.

        Parameters
        ----------
        trainee : Trainee
            A `Trainee` object defining the Trainee.
        library_type : {"st", "mt"}, optional
            (Not implemented) The library type of the Trainee.
        max_wait_time : int or float, default 30
            (Not implemented) The number of seconds to wait for a trainee to
            be created before aborting gracefully.
        overwrite_trainee : bool, default False
            If True, and if a trainee with id `trainee.id`
            already exists, the given trainee will delete the old trainee and
            create the new trainee.
        resources : howso.openapi.models.TraineeResources or dict, optional
            (Not implemented) Customize the resources provisioned for the
            Trainee instance.

        Returns
        -------
        Trainee
            The `Trainee` object that was created.
        """
        if not trainee.id:
            # Default id to trainee name, or new uuid if no name
            trainee.id = trainee.name or str(uuid.uuid4())

        trainee_id = trainee.id

        # Check that the trainee.id is usable for saving later.
        if trainee.name:
            for sequence in self.BAD_TRAINEE_NAME_CHARS:
                if sequence in trainee.name:
                    success = False
                    reason = f'"{sequence}" is not permitted in trainee names'
                    break
            else:
                success, reason = True, 'OK'
            proposed_path: Path = self.howso.default_save_path.joinpath(trainee.name)
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
                util.dprint(self.verbose, f"Deleting existing {trainee_id} "
                                          "trainee before creating.")
                self.howso.delete(trainee_id)
            except Exception:  # noqa: Deliberately broad
                util.dprint(self.verbose, f"Failed to delete {trainee_id} "
                                          "trainee. Continuing.")
        elif trainee_id in self.trainee_cache:
            raise HowsoError(
                f'A trainee already exists using the name "{trainee_id}"')

        trainee = internals.preprocess_trainee(trainee)
        if self.verbose:
            print('Creating trainee')
        result = self.howso.create_trainee(trainee_id)
        if not result:
            raise ValueError(
                f"Could not create the trainee with name {trainee_id}. "
                f"Possible causes - Howso couldn't find core "
                f"binaries/camls or {trainee_id} trainee already exists.")

        metadata = {
            'name': trainee.name,
            'default_context_features': trainee.default_context_features,
            'default_action_features': trainee.default_action_features,
            'metadata': trainee.metadata,
            'persistence': trainee.persistence,
        }
        self.howso.set_metadata(trainee_id, metadata)
        self.howso.set_feature_attributes(trainee_id, trainee.features)
        trainee.features = self.howso.get_feature_attributes(trainee_id)

        self._output_version_in_trace(trainee_id)

        new_trainee = internals.postprocess_trainee(trainee)
        self.trainee_cache.set(new_trainee, entity_id=self.howso.handle)
        return new_trainee

    def update_trainee(self, trainee: Trainee):
        """
        Update an existing Trainee in the Howso service.

        Parameters
        ----------
        trainee : Trainee
            A `Trainee` object defining the Trainee.

        Returns
        -------
        Trainee
            The `Trainee` object that was updated.
        """
        if trainee.id:
            trainee_id = trainee.id
        else:
            trainee_id = trainee.id = trainee.name

        if not trainee_id:
            raise ValueError("A trainee id is required.")

        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Updating trainee with id: {trainee.id}')

        trainee = internals.preprocess_trainee(trainee)
        metadata = {
            'name': trainee.name,
            'default_context_features': trainee.default_context_features,
            'default_action_features': trainee.default_action_features,
            'metadata': trainee.metadata,
            'persistence': trainee.persistence,
        }
        self.howso.set_metadata(trainee_id, metadata)
        self.howso.set_feature_attributes(trainee_id, trainee.features)
        trainee.features = self.howso.get_feature_attributes(trainee_id)

        updated_trainee = internals.postprocess_trainee(trainee)
        self.trainee_cache.set(updated_trainee)
        return updated_trainee

    def export_trainee(
        self,
        trainee_id: str,
        path_to_trainee: Optional[Union[Path, str]] = None,
        decode_cases: bool = False,
        separate_files: bool = False
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
        separate_files : bool, default False
            Whether to load each case from its individual file.
        """
        if self.verbose:
            print(f'Export trainee with id: {trainee_id}')

        self.howso.export_trainee(trainee_id, path_to_trainee, decode_cases,
                                  separate_files)

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

    def get_trainee(self, trainee_id: str):
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

    def get_trainee_information(self, trainee_id: str
                                ) -> client_models.TraineeInformation:
        """
        Get information about the trainee.

        Including trainee version and configuration parameters.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.

        Returns
        -------
        howso.openapi.models.TraineeInformation
            The Trainee information.
        """
        self._auto_resolve_trainee(trainee_id)
        trainee_version = self.howso.get_trainee_version(trainee_id)
        core_version = self.howso.version()
        amlg_version = self.howso.amlg.get_version_string().decode()
        library_type = 'st'
        if self.howso.amlg.library_postfix:
            library_type = self.howso.amlg.library_postfix[1:]

        version = client_models.TraineeVersion(core=core_version,
                                               amalgam=amlg_version,
                                               trainee=trainee_version)

        return client_models.TraineeInformation(library_type=library_type,
                                                version=version)

    def get_trainee_metrics(self, trainee_id: str) -> Never:
        """
        This endpoint is not implemented for the direct Howso client.

        Raises
        ------
        NotImplementedError
            This endpoint is not implemented for the direct Howso client.
        """
        raise NotImplementedError("`get_trainee_metrics` not implemented")

    def get_trainees(self, search_terms: Optional[str] = None):
        """
        Return a list of all trainees.

        Parameters
        ----------
        search_terms : str
            Keywords to filter trainee list by.

        Returns
        -------
        list of howso.openapi.models.TraineeIdentity
            A list of the trainee identities.
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
                    client_models.TraineeIdentity(
                        name=instance.name, id=instance.id)
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
                    client_models.TraineeIdentity(name=trainee_name,
                                                  id=trainee_name))

        return trainees

    def delete_trainee(self, trainee_id: str):
        """
        Delete a Trainee.

        This deletes the Trainee, which includes all cases, model metadata,
        session data, persisted files, etc.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        """
        if not trainee_id:
            raise ValueError("`trainee_id` is required to delete a trainee.")

        # Ensure the trainee_id is valid before deleting.
        for sub in self.BAD_TRAINEE_NAME_CHARS:
            if sub in trainee_id:
                raise ValueError(
                    f'"{sub}" is not permitted in trainee names for deletion.')

        # Unload the trainee from core
        self.howso.delete(trainee_id)
        self.trainee_cache.discard(trainee_id)

        if self.verbose:
            print(f'Deleting trainee with id {trainee_id}')

        # Remove file from storage
        trainee_path = Path(self.howso.default_save_path, f'{trainee_id}{self.howso.ext}')
        trainee_ver_path = Path(self.howso.default_save_path, f'{trainee_id}Version.txt')

        if trainee_path.exists():
            trainee_path.unlink()

        # Do the same for the version file.
        if trainee_ver_path.exists():
            trainee_ver_path.unlink()

    def copy_trainee(
        self,
        trainee_id: str,
        new_trainee_name: Optional[str] = None,
        new_trainee_id: Optional[str] = None,
        *,
        library_type: Optional[Literal["st", "mt"]] = None,
        resources: Optional[Union[client_models.TraineeResources, Dict]] = None,
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
        resources : howso.openapi.models.TraineeResources or dict, optional
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
            new_trainee.id = new_trainee_id
            metadata = {
                'name': new_trainee.name,
                'default_context_features': new_trainee.default_context_features,
                'default_action_features': new_trainee.default_action_features,
                'metadata': new_trainee.metadata,
                'persistence': new_trainee.persistence,
            }
            self.howso.set_metadata(new_trainee_id, metadata)
            self.trainee_cache.set(new_trainee, entity_id=self.howso.handle)

            return new_trainee
        else:
            raise ValueError(
                f"Could not copy the trainee with name {trainee_id}. Possible "
                f"causes - howso couldn't find core binaries/camls or "
                f"{new_trainee_name} trainee already exists."
            )

    def load_trainee(self, trainee_id: str):
        """
        Load a Trainee that was persisted on the Howso service.

        .. deprecated:: 1.0.0
            Use :meth:`HowsoDirectClient.acquire_trainee_resources` instead.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee load.
        """
        warnings.warn(
            'The method `load_trainee()` is deprecated and will be removed in '
            'a future release. Please use `acquire_trainee_resources()` '
            'instead.', DeprecationWarning)
        self.acquire_trainee_resources(trainee_id)

    def unload_trainee(self, trainee_id: str):
        """
        Unload a Trainee from the Howso service.

        .. deprecated:: 1.0.0
            Use :meth:`HowsoDirectClient.release_trainee_resources` instead.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee unload.
        """
        warnings.warn(
            'The method `unload_trainee()` is deprecated and will be removed '
            'in a future release. Please use `release_trainee_resources()` '
            'instead.', DeprecationWarning)
        self.release_trainee_resources(trainee_id)

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
            If no Trainee with the requested ID can be found.
        """
        if trainee_id is None:
            raise HowsoError("A trainee id is required.")
        if self.verbose:
            print(f'Acquiring resources for trainee with id: {trainee_id}')

        if trainee_id in self.trainee_cache:
            # Trainee is already loaded
            cache_item = self.trainee_cache.get_item(trainee_id)
            if cache_item.get('entity_id') != self.howso.handle:
                raise HowsoError(
                    "Unable to acquire trainee resources for the trainee "
                    f"'{trainee_id}'. Trainee is already loaded in another "
                    "core entity. Use the HowsoClient instance with the "
                    f"entity handle '{self.howso.handle}' instead or release it "
                    "via the other client first."
                )
            return

        ret = self.howso.load(trainee_id)

        if ret is None:
            raise HowsoError(f"Trainee '{trainee_id}' not found.")

        trainee = self._get_trainee_from_core(trainee_id)
        self.trainee_cache.set(trainee, entity_id=self.howso.handle)

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

        action_features = metadata.get("default_action_features", list())
        context_features = metadata.get("default_context_features", list())
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
            default_action_features=action_features,
            default_context_features=context_features
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
            if cache_item.get('entity_id', self.howso.handle) == self.howso.handle:
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

    def _auto_resolve_trainee(self, trainee_id: str):
        """
        Resolve a Trainee and acquire its resources.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to persist.

        Raises
        ------
        HowsoError
            If the requested Trainee is currently loaded by another core entity.
        """
        if trainee_id not in self.trainee_cache:
            self.acquire_trainee_resources(trainee_id)
        else:
            entity_id = self.trainee_cache.get_item(trainee_id).get('entity_id')
            if entity_id != self.howso.handle:
                raise HowsoError(
                    f"Attempted to access the trainee '{trainee_id}' via a "
                    "client using a different core entity than the entity "
                    "where the trainee is currently loaded. Use the "
                    "HowsoClient instance with the core entity handle "
                    f"'{self.howso.handle}' instead to access this trainee or "
                    "release it via the other client first.")

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

    def set_substitute_feature_values(
        self, trainee_id: str, substitution_value_map: Dict[str, Dict]
    ):
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
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print('Setting substitute feature values for trainee with '
                  f'id: {trainee_id}')
        self.howso.set_substitute_feature_values(trainee_id, substitution_value_map)
        self._auto_persist_trainee(trainee_id)

    def get_substitute_feature_values(
        self, trainee_id: str, clear_on_get: bool = True
    ) -> Dict[str, Dict]:
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
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Getting substitute feature values from trainee with '
                  f'id: {trainee_id}')
        ret = self.howso.get_substitute_feature_values(trainee_id)
        if clear_on_get:
            self.set_substitute_feature_values(trainee_id, {})
        if ret is None:
            return dict()
        return ret

    def set_random_seed(self, trainee_id: str, seed: Union[int, float, str]):
        """
        Sets the random seed for the trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set the random seed for.
        seed: int or float or str
            The random seed.
            Ex: ``7998``, ``"bobtherandomseed"``
        """
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Setting random seed for trainee with id: {trainee_id}')
        self.howso.set_random_seed(trainee_id, seed)
        self._auto_persist_trainee(trainee_id)

    def train(  # noqa: C901
        self,
        trainee_id: str,
        cases: Union[List[List[object]], DataFrame],
        features: Optional[Iterable[str]] = None,
        *,
        ablatement_params: Optional[Dict[str, List[object]]] = None,
        accumulate_weight_feature: Optional[str] = None,
        batch_size: Optional[int] = None,
        derived_features: Optional[Iterable[str]] = None,
        input_is_substituted: bool = False,
        progress_callback: Optional[Callable] = None,
        series: Optional[str] = None,
        train_weights_only: bool = False,
        validate: bool = True,
    ):
        """
        Train one or more cases into a trainee (model).

        Parameters
        ----------
        trainee_id : str
            The ID of the target Trainee.
        cases : list of list of object or pandas.DataFrame
            One or more cases to train into the model.
        features : iterable of str, optional
            An iterable of feature names.
            This parameter should be provided in the following scenarios:

                a. When cases are not in the format of a DataFrame, or
                   the DataFrame does not define named columns.
                b. You want to train only a subset of columns defined in your
                   cases DataFrame.
                c. You want to re-order the columns that are trained.

        ablatement_params : dict of str to list of object, optional
            Where keys are a feature name and values are threshold_type where
            threshold_type is one of:

                - ['exact']: Don't train if prediction matches exactly
                - ['tolerance', MIN, MAX]: Don't train if ``prediction
                  >= (case value - MIN) & prediction <= (case value + MAX)``
                - ['relative', PERCENT]: Don't train if
                  ``abs(prediction - case value) / prediction <= PERCENT``
                - ['residual']: Don't train if
                  ``abs(prediction - case value) <= feature residual``

            >>> {'species': ['exact'], 'sepal_length': ['tolerance', 0.1, 0.25]}

        accumulate_weight_feature : str, optional
            Name of feature into which to accumulate neighbors'
            influences as weight for ablated cases. If unspecified, will not
            accumulate weights.
        batch_size: int, optional
            Define the number of cases to train at once. If left unspecified,
            the batch size will be determined automatically.
        derived_features: iterable of str, optional
            List of feature names for which values should be derived
            in the specified order. If this list is not provided, features with
            the 'auto_derive_on_train' feature attribute set to True will be
            auto-derived. If provided an empty list, no features are derived.
            Any derived_features that are already in the 'features' list will
            not be derived since their values are being explicitly provided.
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
        train_weights_only : bool, default False
            When true, and accumulate_weight_feature is provided,
            will accumulate all of the cases' neighbor weights instead of
            training the cases into the model.
        validate : bool, default True
            Whether to validate the data against the provided feature
            attributes. Issues warnings if there are any discrepancies between
            the data and the features dictionary.
        """
        self._auto_resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features

        # Make sure single table dicts are wrapped by SingleTableFeatureAttributes
        if isinstance(feature_attributes, Dict) and not isinstance(feature_attributes,
                                                                   MultiTableFeatureAttributes):
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
        for feature in unsupported_features:
            warnings.warn(f'Ignoring feature {feature} as it contains values that are too '
                          'large or small for your operating system. Please evaluate the '
                          'bounds for this feature.')
            cases.drop(feature, axis=1, inplace=True)

        validate_list_shape(features, 1, "features", "str")
        if self.verbose:
            print(f'Training session(s) on trainee with id: {trainee_id}')

        validate_list_shape(cases, 2, "cases", "list", allow_none=False)
        if features is None:
            features = internals.get_features_from_data(cases)
        cases = serialize_cases(cases, features, feature_attributes, warn=True)

        auto_analyze = False

        with ProgressTimer(len(cases)) as progress:
            gen_batch_size = None
            if series is not None:
                # If training series, always send full size
                batch_size = len(cases)
            if not batch_size:
                # Scale the batch size automatically
                batch_scaler = self.batch_scaler_class(100, progress)
                batch_scaler.size_limits = (1, 250_000)
                gen_batch_size = batch_scaler.gen_batch_size()
                batch_size = next(gen_batch_size, None)

            while not progress.is_complete and batch_size:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress)
                start = progress.current_tick
                end = progress.current_tick + batch_size
                response = self.howso.train(
                    trainee_id,
                    ablatement_params=ablatement_params,
                    accumulate_weight_feature=accumulate_weight_feature,
                    derived_features=derived_features,
                    features=features,
                    input_cases=cases[start:end],
                    input_is_substituted=input_is_substituted,
                    series=series,
                    session=self.active_session.id,
                    train_weights_only=train_weights_only
                )
                if response and response.get('status') == 'analyze':
                    auto_analyze = True
                if gen_batch_size is None:
                    progress.update(batch_size)
                else:
                    batch_size = next(gen_batch_size, None)

        # Final call to batch callback on completion
        if isinstance(progress_callback, Callable):
            progress_callback(progress)

        # Add session metadata to trainee
        self.howso.set_session_metadata(
            trainee_id,
            self.active_session.id,
            self.active_session
        )

        self._auto_persist_trainee(trainee_id)

        # kick off auto-analyze if the train response requests it
        if auto_analyze:
            self.auto_analyze(trainee_id)

    def impute(
        self,
        trainee_id: str,
        features: Optional[Iterable[str]] = None,
        features_to_impute: Optional[Iterable[str]] = None,
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
        features: iterable of str, optional
            An iterable of feature names to use for imputation.

            If not specified, all features will be used imputed.
        features_to_impute: iterable of str, optional
            An iterable of feature names to impute
            If not specified, features will be used (see above)
        batch_size: int, default 1
            Larger batch size will increase accuracy and decrease speed.
            Batch size indicates how many rows to fill before recomputing
            conviction.

            The default value (which is 1) should return the best accuracy but
            might be slower. Higher values should improve performance but may
            decrease accuracy of results.
        """
        self._auto_resolve_trainee(trainee_id)
        validate_list_shape(features, 1, "features", "str")
        validate_list_shape(features_to_impute, 1, "features_to_impute", "str")
        if self.verbose:
            print(f'Imputing trainee with id: {trainee_id}')
        self.howso.impute(
            trainee_id,
            session=self.active_session.id,
            features=features,
            features_to_impute=features_to_impute,
            batch_size=batch_size
        )
        self._auto_persist_trainee(trainee_id)

    def remove_cases(
        self,
        trainee_id: str,
        num_cases: int,
        *,
        case_indices: Optional[Iterable[Tuple[str, int]]] = None,
        condition: Optional[Dict[str, object]] = None,
        condition_session: Optional[str] = None,
        distribute_weight_feature: Optional[str] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
        preserve_session_data: bool = False
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
        case_indices : list of tuples
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
        preserve_session_data : bool, default False
            When True, will remove cases without cleaning up session data.

        Returns
        -------
        int
            The number of cases removed.

        Raises
        ------
        ValueError
            If `num_cases` is not at least 1.
        """
        self._auto_resolve_trainee(trainee_id)
        if num_cases < 1:
            raise ValueError('num_cases must be a value greater than 0')
        if self.verbose:
            print(f'Removing case(s) in trainee with id: {trainee_id}')

        # Convert session instance to id
        if (
            isinstance(condition, dict) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        result = self.howso.remove_cases(
            trainee_id,
            case_indices=case_indices,
            condition=condition,
            condition_session=condition_session,
            distribute_weight_feature=distribute_weight_feature,
            num_cases=num_cases,
            precision=precision,
            preserve_session_data=preserve_session_data,
            session=self.active_session.id
        )
        self._auto_persist_trainee(trainee_id)
        return result.get('count', 0)

    def edit_cases(
        self,
        trainee_id: str,
        feature_values: Union[List[object], DataFrame],
        *,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        condition: Optional[Dict[str, object]] = None,
        condition_session: Optional[str] = None,
        features: Optional[Iterable[str]] = None,
        num_cases: Optional[int] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
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
        case_indices : Iterable of Sequence[Union[str, int]], optional
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
        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        # Validate case_indices if provided
        if case_indices is not None:
            validate_case_indices(case_indices)

        # Serialize feature_values
        if feature_values is not None:
            if features is None:
                features = internals.get_features_from_data(
                    feature_values, data_parameter='feature_values')
            feature_values = serialize_cases(feature_values, features,
                                             cached_trainee.features)
            if feature_values:
                # Only a single case should be provided
                feature_values = feature_values[0]

        # Convert session instance to id
        if (
            isinstance(condition, dict) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        if self.verbose:
            print(f'Editing case(s) in trainee with id: {trainee_id}')
        result = self.howso.edit_cases(
            trainee_id,
            case_indices=case_indices,
            condition=condition,
            condition_session=condition_session,
            features=features,
            feature_values=feature_values,
            precision=precision,
            num_cases=num_cases,
            session=self.active_session.id
        )
        self._auto_persist_trainee(trainee_id)
        return result.get('count', 0)

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
        howso.openapi.models.Session
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
            created_date=datetime.utcnow(),
            modified_date=datetime.utcnow(),
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
        list of howso.openapi.models.Session
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
                            instance = model_from_dict(Session, session)
                            instance.metadata = instance.metadata or dict()
                            instance.metadata['trainee_id'] = trainee_id
                            filtered_sessions.append(instance)
                            break
                else:
                    instance = model_from_dict(Session, session)
                    instance.metadata = instance.metadata or dict()
                    instance.metadata['trainee_id'] = trainee_id
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
        howso.openapi.models.Session
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
                session_data = self.howso.get_session_metadata(
                    trainee_id, session_id)
            except HowsoError:
                # When session is not found, continue
                continue
            session = model_from_dict(Session, session_data)
            # Include trainee_id in the metadata
            session.metadata = session.metadata or dict()
            session.metadata['trainee_id'] = trainee_id
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
        howso.openapi.models.Session
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
        modified_date = datetime.utcnow()
        metadata = metadata or dict()
        # We remove the trainee_id since this may have been set by the
        # get_session(s) methods and is not needed to be stored in the model.
        metadata.pop('trainee_id', None)

        def _update_session(instance):
            instance.metadata = instance.metadata or dict()
            instance.metadata = metadata or dict()
            instance.modified_date = modified_date
            return instance

        # Update session across all loaded trainees
        for trainee_id in self.trainee_cache.ids():
            try:
                session_data = self.howso.get_session_metadata(
                    trainee_id, session_id)
            except HowsoError:
                # When session is not found, continue
                continue
            session = model_from_dict(Session, session_data)
            session = _update_session(session)
            self.howso.set_session_metadata(trainee_id, session_id, session)
            updated_session = session

        if self.active_session.id == session_id:
            # Update active session
            self._active_session = _update_session(self.active_session)
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
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        contexts: Optional[Union[List[List[object]], DataFrame]] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
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
        progress_callback: Optional[Callable] = None,
        series_context_features: Optional[Iterable[str]] = None,
        series_context_values: Optional[Union[List[object], List[List[object]]]] = None,
        series_id_tracking: Literal["dynamic", "fixed", "no"] = "fixed",
        series_stop_maps: Optional[List[Dict[str, Dict]]] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None
    ) -> Dict:
        """
        React in a series until a series_stop_map condition is met.

        Aggregates rows of data corresponding to the specified context, action,
        derived_context and derived_action features, utilizing previous rows to
        derive values as necessary. Outputs a dict of "action_features" and
        corresponding "series" where "series" is the completed 'matrix' for the
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

            .. TIP::
                Stop series when value exceeds max or is smaller than min::

                    {"feature_name":  {"min" : 1, "max": 2}}

                Stop series when feature value matches any of the values
                listed::

                    {"feature_name":  {"values": ["val1", "val2"]}}

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

            .. note::

                Both of these derived feature lists rely on the features'
                "derived_feature_code" attribute to compute the values. If
                "derived_feature_code" attribute references non-existing
                feature indices, the derived value will be null.

        series_context_features : iterable of str, optional
            list of context features corresponding to
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

        progress_callback : callable, optional
            A callback method that will be called before each
            batched call to react series and at the end of reacting. The method
            is given a ProgressTimer containing metrics on the progress and
            timing of the react series operation, and the batch result.
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
        dict
            A dictionary with keys `action_features` and `series`. Where
            `series` is a 2d list of values (rows of data per series), and
            `action_features` is the list of all action features
            (specified and derived).

            Example output for 2 short series with 3 features:

            .. code-block::

                {
                    'action_features': ['id','x','y'],
                    'series': [
                        [ ["A", 1, 2], ["A", 2, 2] ],
                        [ ["B", 4, 4], ["B", 6, 7], ["B", 8, 9] ]
                    ]
                }

        Raises
        ------
        ValueError
            If the number of provided context values does not match the length of
            context features.

            If `series_context_values` is not a 3d list of objects.

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

        validate_list_shape(initial_features, 1, "series_context_features", "str")
        if series_context_values and num_list_dimensions(series_context_values) != 3:
            raise ValueError(
                "Improper shape of `series_context_values` values passed. "
                "`series_context_values` must be a 3d list of object.")

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

        if desired_conviction is None:
            if contexts is not None:
                for context in contexts:
                    if context is not None and \
                            (len(context) != len(context_features)):
                        raise ValueError(
                            "Number of provided context values in `context` "
                            "does not match length of `context_features`."
                        )
                total_size = len(contexts)
            else:
                total_size = len(case_indices)

            react_params = {
                "action_features": action_features,
                "action_values": actions,
                "context_features": context_features,
                "context_values": contexts,
                "initial_features": initial_features,
                "initial_values": initial_values,
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
            total_size = num_series_to_generate
            if total_size > self._react_generative_batch_threshold:
                # Do not send details for generative reacts over threshold
                details = None

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

        if self._should_react_batch(react_params, total_size):
            if self.verbose:
                print(f'Batch series reacting on trainee with id: {trainee_id}')
            response = self._batch_react_series(
                trainee_id, react_params, total_size=total_size,
                progress_callback=progress_callback)
        else:
            if self.verbose:
                print(f'Series reacting on trainee with id: {trainee_id}')
            with ProgressTimer(total_size) as progress:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress, None)
                response = self._react_series(trainee_id, react_params)
                progress.update(total_size)

            if isinstance(progress_callback, Callable):
                progress_callback(progress, response)

        # If the number of series generated is less then requested, raise
        # warning, for generative reacts
        if desired_conviction is not None:
            len_action = len(response['series'])
            internals.insufficient_generation_check(
                num_series_to_generate, len_action,
                suppress_warning=suppress_warning
            )

        return response

    def _batch_react_series(  # noqa: C901
        self,
        trainee_id: str,
        react_params: dict,
        *,
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
        accumulated_result = {'series': []}

        actions = react_params.get('action_values')
        contexts = react_params.get('context_values')
        case_indices = react_params.get('case_indices')
        initial_values = react_params.get('initial_values')
        max_series_lengths = react_params.get('max_series_lengths')
        series_context_values = react_params.get('series_context_values')
        series_stop_maps = react_params.get('series_stop_maps')

        with ProgressTimer(total_size) as progress:
            if self.howso.amlg.library_postfix[1:] == 'mt':
                start_batch_size = max(multiprocessing.cpu_count(), 1)
            else:
                start_batch_size = 1
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

                if react_params.get('desired_conviction') is not None:
                    react_params['num_series_to_generate'] = batch_size
                temp_result = self._react_series(trainee_id, react_params)

                internals.accumulate_react_result(accumulated_result,
                                                  temp_result)
                batch_size = next(gen_batch_size, None)

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
        """
        batch_result = self.howso.batch_react_series(trainee_id, **react_params)

        if batch_result is None or batch_result.get('series') is None:
            raise ValueError('Invalid parameters passed to react_series.')

        ret = dict()

        ret['action_features'] = batch_result.pop('action_features') or []
        ret['series'] = replace_doublemax_with_infinity(
            batch_result.pop('series'))

        return ret

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
            default_context_features = cached_trainee.default_context_features
            context_features = internals.get_features_from_data(
                contexts,
                default_features=default_context_features,
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
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        contexts: Optional[Union[List[List[object]], DataFrame]] = None,
        context_features: Optional[Iterable[str]] = None,
        derived_action_features: Optional[Iterable[str]] = None,
        derived_context_features: Optional[Iterable[str]] = None,
        desired_conviction: Optional[float] = None,
        details: Optional[Dict] = None,
        feature_bounds_map: Optional[Dict] = None,
        generate_new_cases: Literal["always", "attempt", "no"] = "no",
        input_is_substituted: bool = False,
        into_series_store: Optional[str] = None,
        leave_case_out: bool = False,
        new_case_threshold: Literal["max", "min", "most_similar"] = "min",
        num_cases_to_generate: int = 1,
        ordered_by_specified_features: bool = False,
        preserve_feature_values: Optional[Iterable[str]] = None,
        progress_callback: Optional[Callable] = None,
        substitute_output: bool = True,
        suppress_warning: bool = False,
        use_case_weights: bool = False,
        use_regional_model_residuals: bool = True,
        weight_feature: Optional[str] = None,
    ) -> Dict:
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
            If details are specified, the response will
            contain the requested explanation data along with the reaction.
            Below are the valid keys and data types for the different audit
            details. Omitted keys, values set to None, or False values for
            Booleans will not be included in the audit data returned.

            - influential_cases: bool, optional
                If True outputs the most influential cases and their influence
                weights based on the surprisal of each case relative to the
                context being predicted among the cases. Uses only the context
                features of the reacted case.

            - influential_cases_familiarity_convictions:  bool, optional
                If True outputs familiarity conviction of addition for each of
                the influential cases.

            - influential_cases_raw_weights: bool, optional
                If True outputs the surprisal for each of the influential cases.

            - hypothetical_values: dict, optional
                A dictionary of feature name to feature value. If specified,
                shows how a prediction could change in a what-if scenario where
                the influential cases' context feature values are replaced with
                the specified values.  Iterates over all influential cases,
                predicting the action features each one using the updated
                hypothetical values. Outputs the predicted arithmetic over the
                influential cases for each action feature.

            - most_similar_cases: bool, optional
                If True outputs an automatically determined (when
                'num_most_similar_cases' is not specified) relevant number of
                similar cases, which will first include the influential cases.
                Uses only the context features of the reacted case.

            - num_most_similar_cases: int, optional
                Outputs this manually specified number of most similar cases,
                which will first include the influential cases.

                NOTE: The maximum number of cases that can be queried is
                `1000`.

            - num_most_similar_case_indices : int, optional
                Outputs the specified number of most similar case indices when
                'distance_ratio' is also set to True.

                NOTE: The maximum number of cases that can be queried is
                '1000'.

            - num_robust_influence_samples_per_case : int, optional
                Specifies the number of robust samples to use for each case.
                Applicable only for computing robust feature contributions or
                robust case feature contributions. Defaults to 2000. Higher
                values will take longer but provide more stable results.

            - boundary_cases: bool, optional
                If True outputs an automatically determined (when
                'num_boundary_cases' is not specified) relevant number of
                boundary cases. Uses both context and action features of the
                reacted case to determine the counterfactual boundary based on
                action features, which maximize the dissimilarity of action
                features while maximizing the similarity of context features.
                If action features aren't specified, uses familiarity conviction
                to determine the boundary instead.

            - num_boundary_cases: int, optional
                Outputs this manually specified number of boundary cases.

                NOTE: The maximum number of cases that can be queried is
                '1000'.

            - boundary_cases_familiarity_convictions: bool, optional
                If True outputs familiarity conviction of addition for each of
                the boundary cases.

            - distance_ratio: bool, optional
                If True outputs the ratio of distance (relative surprisal)
                between this reacted case and its nearest case to the minimum
                distance (relative surprisal) in between the closest two cases
                in the local area. All distances are computed using only the
                specified context features.

            - distance_contribution: bool, optional
                If True outputs the distance contribution (expected total
                surprisal contribution) for the reacted case. Uses both context
                and action feature values.

            - similarity_conviction: bool, optional
                If True outputs similarity conviction for the reacted case.
                Uses both context and action feature values as the case values
                for all computations. This is defined as expected (local)
                distance contribution divided by reacted case distance
                contribution.

            - prediction_residual_conviction: bool, optional
                If True outputs residual conviction for the reacted case's
                action features by computing the prediction residual for the
                action features in the local model area. Uses both context and
                action features to determine that area. This is defined as the
                expected (global) model residual divided by computed local
                residual.

            - outlying_feature_values: bool, optional
                If True outputs the reacted case's context feature values that
                are outside the min or max of the corresponding feature values
                of all the cases in the local model area. Uses only the context
                features of the reacted case to determine that area.

            - categorical_action_probabilities: bool, optional
                If True outputs probabilities for each class for the action.
                Applicable only to categorical action features.

            - observational_errors: bool, optional
                If True outputs observational errors for all features as
                defined in feature attributes.

            - robust_computation: bool, optional
                Default is False, uses leave-one-out for features (or cases,
                as needed) for all relevant computations. If True, uses
                uniform sampling from the power set of all combinations of
                features (or cases, as needed) instead.

            - feature_residuals: bool, optional
                If True outputs feature residuals for all (context and action)
                features locally around the prediction. Uses only the context
                features of the reacted case to determine that area. Relies on
                'robust_computation' parameter to determine whether to do
                standard or robust computation.

            - feature_mda: bool, optional
                If True outputs each context feature's mean decrease in
                accuracy of predicting the action feature given the context.
                Uses only the context features of the reacted case to determine
                that area. Relies on 'robust_computation' parameter to
                determine whether to do standard or robust computation.

            - feature_mda_ex_post: bool, optional
                If True outputs each context feature's mean decrease in
                accuracy of predicting the action feature as an explanation
                given that the specified prediction was already made as
                specified by the action value. Uses both context and action
                features of the reacted case to determine that area. Relies on
                'robust_computation' parameter to determine whether to do
                standard or robust computation.

            - feature_contributions: bool, optional
                If True outputs each context feature's differences between the
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

            - case_mda: bool, optional
                If True outputs each influential case's mean decrease in
                accuracy of predicting the action feature in the local model
                area, as if each individual case were included versus not
                included. Uses only the context features of the reacted case to
                determine that area. Relies on 'robust_computation' parameter
                to determine whether to do standard or robust computation.

            - case_contributions: bool, optional
                If True outputs each influential case's differences between the
                predicted action feature value and the predicted action feature
                value if each individual case were not included. Uses only the
                context features of the reacted case to determine that area.
                Relies on 'robust_computation' parameter to determine whether
                to do standard or robust computation.

            - case_feature_residuals: bool, optional
                If True outputs feature residuals for all (context and action)
                features for just the specified case. Uses leave-one-out for
                each feature, while using the others to predict the left out
                feature with their corresponding values from this case. Relies
                on 'robust_computation' parameter to determine whether to do
                standard or robust computation.

            - local_case_feature_residual_convictions: bool, optional
                If True outputs this case's feature residual convictions for
                the region around the prediction. Uses only the context
                features of the reacted case to determine that region.
                Computed as: region feature residual divided by case feature
                residual. Relies on 'robust_computation' parameter to determine
                whether to do standard or robust computation.

            - global_case_feature_residual_convictions: bool, optional
                If True outputs this case's feature residual convictions for
                the global model. Computed as: global model feature residual
                divided by case feature residual. Relies on
                'robust_computation' parameter to determine whether to do
                standard or robust computation.

            >>> details = {'num_most_similar_cases': 5,
            ...            'feature_residuals': True}

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

        Returns
        -------
        dict
            A dictionary with keys `action` and `explanation`. Where `action`
            is a list of dicts of action_features -> action_values, and
            `explanation` is a dict with the requested audit data.

            .. code-block::
                :caption: Example reaction for 2 contexts with 2 action features:

                {
                    'action': [{'size': 1, 'width': 1}, {'size': 2, 'width': 2}]
                    'explanation': {
                        'action_features': ['size', 'width'],
                        'distance_contribution': [3.45, 0.89],
                    }
                }

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

            react_params = {
                "action_values": actions,
                "context_features": context_features,
                "context_values": contexts,
                "action_features": action_features,
                "derived_context_features": derived_context_features,
                "derived_action_features": derived_action_features,
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
            if total_size > self._react_generative_batch_threshold:
                # Do not send details for generative reacts over threshold
                details = None

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

            react_params = {
                "num_cases_to_generate": num_cases_to_generate,
                "context_features": context_features,
                "context_values": contexts,
                "action_features": action_features,
                "derived_context_features": derived_context_features,
                "derived_action_features": derived_action_features,
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
            }

        if self._should_react_batch(react_params, total_size):
            # Run in batch
            if self.verbose:
                print('Batch reacting to context on trainee with id: '
                      f'{trainee_id}')
            response = self._batch_react(trainee_id, react_params,
                                         total_size=total_size,
                                         progress_callback=progress_callback)
        else:
            # Run as a single react request
            if self.verbose:
                print(f'Reacting to context on trainee with id: {trainee_id}')
            with ProgressTimer(total_size) as progress:
                if isinstance(progress_callback, Callable):
                    progress_callback(progress, None)
                response = self._react(trainee_id, react_params)
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

        return response

    def _batch_react(  # noqa: C901
        self,
        trainee_id: str,
        react_params: Dict,
        *,
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

        with ProgressTimer(total_size) as progress:
            batch_scaler = self.batch_scaler_class(10, progress)
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

                if react_params.get('desired_conviction') is not None:
                    react_params['num_cases_to_generate'] = batch_size
                temp_result = self._react(trainee_id, react_params)

                internals.accumulate_react_result(accumulated_result,
                                                  temp_result)
                batch_size = next(gen_batch_size, None)

        # Final call to callback on completion
        if isinstance(progress_callback, Callable):
            progress_callback(progress, temp_result)

        return accumulated_result

    def _react(self, trainee_id: str, react_params: Dict) -> Dict:
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
        """
        ret = self.howso.batch_react(trainee_id, **react_params)

        # Action values and features should always be defined
        if not ret.get('action_values'):
            ret['action_values'] = []
        if not ret.get('action_features'):
            ret['action_features'] = []

        return ret

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
        trainee_id: str
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

        Returns
        -------
        tuple
           Updated action_features, actions, context_features, contexts
        """
        # Validate case_indices if provided
        if case_indices is not None:
            validate_case_indices(case_indices)

        # Get default features
        self._auto_resolve_trainee(trainee_id)
        trainee = self.trainee_cache.get(trainee_id)

        # Preprocess contexts
        if contexts is not None and context_features is None:
            context_features = internals.get_features_from_data(
                contexts,
                default_features=trainee.default_context_features,
                data_parameter='contexts',
                features_parameter='context_features')
        contexts = serialize_cases(
            contexts, context_features or trainee.default_context_features,
            trainee.features)

        # Preprocess actions
        if actions is not None and action_features is None:
            validate_list_shape(actions, 2, "actions", "object")
            action_features = internals.get_features_from_data(
                actions,
                default_features=trainee.default_action_features,
                data_parameter='actions',
                features_parameter='action_features')
        actions = serialize_cases(
            actions, action_features or trainee.default_action_features,
            trainee.features)

        # validate discriminative-react only parameters
        if desired_conviction is None:
            validate_list_shape(contexts, 2, "contexts", "list of object")
            validate_list_shape(action_features, 1, "action_features", "str")
            validate_list_shape(context_features, 1, "context_features", "str")

            if self.verbose:
                print(f'Reacting to context on trainee with id: {trainee_id}')

            if action_features is None:
                action_features = trainee.default_action_features

            if context_features is None and preserve_feature_values is None:
                context_features = trainee.default_context_features

            if contexts is None:
                if case_indices is None or preserve_feature_values is None:
                    raise ValueError(
                        "If `contexts` are not specified, both `case_indices`"
                        " and `preserve_feature_values` must be specified."
                    )

        return action_features, actions, context_features, contexts

    def react_into_features(
        self,
        trainee_id: str,
        *,
        features: Optional[Iterable[str]] = None,
        familiarity_conviction_addition: Optional[Union[str, bool]] = False,
        familiarity_conviction_removal: Optional[Union[str, bool]] = False,
        p_value_of_addition: Optional[Union[str, bool]] = False,
        p_value_of_removal: Optional[Union[str, bool]] = False,
        similarity_conviction: Optional[Union[str, bool]] = False,
        distance_contribution: Optional[Union[str, bool]] = False,
        weight_feature: Optional[str] = None,
        use_case_weights: bool = False
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
            p_value_of_addition=p_value_of_addition,
            p_value_of_removal=p_value_of_removal,
            similarity_conviction=similarity_conviction,
            distance_contribution=distance_contribution,
            weight_feature=weight_feature,
            use_case_weights=use_case_weights)
        self._auto_persist_trainee(trainee_id)

    def get_cases(
        self,
        trainee_id: str,
        session: Optional[str] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        indicate_imputed: bool = False,
        features: Optional[Iterable[str]] = None,
        condition: Optional[Dict] = None,
        num_cases: Optional[int] = None,
        precision: Optional[Literal["exact", "similar"]] = None
    ) -> client_models.Cases:
        """
        Retrieve cases from a model given a trainee id.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee retrieve cases from.

        session : str, optional
            The session ID to retrieve cases for, in their trained order.

            NOTE: If a session is not provided, retrieves all feature values
                  for cases for all (unordered) sessions in the order they
                  were trained within each session.

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
        howso.openapi.models.Cases
            A cases object containing the feature names and cases.
        """
        # Validate case_indices if provided
        if case_indices is not None:
            validate_case_indices(case_indices)

        self._auto_resolve_trainee(trainee_id)
        validate_list_shape(features, 1, "features", "str")
        if session is None and case_indices is None:
            warnings.warn("Calling get_cases without session id does not "
                          "guarantee case order.")
        if self.verbose:
            print('Retrieving cases.')
        result = self.howso.get_cases(
            trainee_id,
            features=features,
            session=session,
            case_indices=case_indices,
            indicate_imputed=1 if indicate_imputed else 0,
            condition=condition,
            num_cases=num_cases,
            precision=precision
        )
        if result is None:
            result = dict()
        return client_models.Cases(features=result.get('features'),
                                   cases=result.get('cases'))

    def react_group(
        self,
        trainee_id: str,
        *,
        new_cases: Optional[Union[List[List[List[object]]], List[DataFrame]]] = None,
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
    ) -> client_models.ReactGroupResponse:
        """
        Computes specified data for a **set** of cases.

        Return the list of familiarity convictions (and optionally, distance
        contributions or p values) for each set.

        Parameters
        ----------
        trainee_id : str
            The trainee id.

        new_cases : list of list of list of object or list of DataFrame, optional
            Specify a **set** using a list of cases to compute the conviction of
            groups of cases as shown in the following example.

            >>> [ [[1, 2, 3], [4, 5, 6], [7, 8, 9]], # Group 1
            >>>   [[1, 2, 3]] ] # Group 2

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
        self._auto_resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features
        serialized_cases = None

        if new_cases is not None:
            # if trainees_to_compare is specified, ignore new_cases
            if trainees_to_compare is not None:
                new_cases = None
            elif num_list_dimensions(new_cases) != 3:
                raise ValueError(
                    "Improper shape of `new_cases` values passed. "
                    "`new_cases` must be a 3d list of object.")
        elif trainees_to_compare is None:
            raise ValueError(
                "Either `new_cases` or `trainees_to_compare` must be provided.")

        if trainees_to_compare is not None:
            # Ensure all trainees being compared are available
            for other_trainee_id in trainees_to_compare:
                self._auto_resolve_trainee(other_trainee_id)

        if new_cases is not None:
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
            trainees_to_compare=trainees_to_compare,
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

    def get_feature_conviction(
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
            familiarity_conviction_removal
        """
        self._auto_resolve_trainee(trainee_id)
        validate_list_shape(features, 1, "features", "str")
        validate_list_shape(action_features, 1, "action_features", "str")
        if self.verbose:
            print('Getting conviction of features for trainee with id: '
                  f'{trainee_id}')
        return self.howso.compute_conviction_of_features(
            trainee_id,
            features=features,
            action_features=action_features,
            familiarity_conviction_addition=familiarity_conviction_addition,
            familiarity_conviction_removal=familiarity_conviction_removal,
            weight_feature=weight_feature,
            use_case_weights=use_case_weights
        )

    def add_feature(
        self,
        trainee_id: str,
        feature: str,
        feature_value: Optional[Union[int, float, str]] = None,
        *,
        condition: Optional[Dict] = None,
        condition_session: Optional[str] = None,
        feature_attributes: Optional[Dict] = None,
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

        condition_session : str, optional
            If specified, ignores the condition and operates on cases for the
            specified session id.
        overwrite : bool, default False
            If True, the feature will be over-written if it exists.
        """
        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        if self.verbose:
            print(f'Adding feature {feature}.')

        if feature_attributes is not None:
            updated_attributes = internals.preprocess_feature_attributes(
                {feature: feature_attributes})
            feature_attributes = updated_attributes[feature]

        self.howso.add_feature(
            trainee_id, feature,
            feature_value=feature_value,
            condition=condition,
            condition_session=condition_session,
            feature_attributes=feature_attributes,
            overwrite=overwrite,
            session=self.active_session.id
        )
        self._auto_persist_trainee(trainee_id)

        # Update trainee in cache
        updated_feature_attributes = self.get_feature_attributes(trainee_id)
        cached_trainee.features = updated_feature_attributes

    def remove_feature(
        self,
        trainee_id: str,
        feature: str,
        *,
        condition: Optional[Dict] = None,
        condition_session: Optional[str] = None
    ):
        """
        Removes a feature from a trainee.

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
        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        if self.verbose:
            print(f'Removing feature "{feature}" for trainee with '
                  f'id: {trainee_id}')

        self.howso.remove_feature(
            trainee_id, feature,
            condition=condition,
            condition_session=condition_session,
            session=self.active_session.id
        )
        self._auto_persist_trainee(trainee_id)

        # Update trainee in cache
        updated_feature_attributes = self.get_feature_attributes(trainee_id)
        cached_trainee.features = updated_feature_attributes

    def get_feature_residuals(
        self,
        trainee_id: str,
        *,
        action_feature: Optional[str] = None,
        robust: Optional[bool] = None,
        robust_hyperparameters: Optional[bool] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get cached feature residuals.

        All keyword arguments are optional, when not specified will auto-select
        cached residuals for output, when specified will attempt to
        output the cached residuals best matching the requested parameters,
        if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`HowsoDirectClient.get_prediction_stats` instead.

        Parameters
        ----------
        trainee_id : str
            The id or name of the trainee.
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
        dict of str to float
            A map of feature names to residual values.
        """
        warnings.warn(
            'The method `get_feature_residuals()` is deprecated and will be'
            'removed in a future release. Please use `get_prediction_stats()` '
            'instead.', DeprecationWarning)
        self._auto_resolve_trainee(trainee_id)

        if self.verbose:
            print('Getting feature residuals for trainee with '
                  f'id: {trainee_id}')

        residuals = self.howso.get_feature_residuals(
            trainee_id,
            action_feature=action_feature,
            robust=robust,
            robust_hyperparameters=robust_hyperparameters,
            weight_feature=weight_feature)
        return residuals

    def get_feature_mda(
        self,
        trainee_id: str,
        action_feature: str,
        *,
        permutation: Optional[bool] = None,
        robust: Optional[bool] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get cached feature Mean Decrease In Accuracy (MDA).

        All keyword arguments are optional, when not specified will auto-select
        cached MDA for output, when specified will attempt to
        output the cached MDA best matching the requested parameters,
        if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`HowsoDirectClient.get_prediction_stats` instead.

        Parameters
        ----------
        trainee_id : str
            The id or name of the trainee.
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
        dict of str to float
            A map of feature names to MDA values.
        """
        warnings.warn(
            'The method `get_feature_mda()` is deprecated and will be'
            'removed in a future release. Please use `get_prediction_stats()` '
            'instead.', DeprecationWarning)
        self._auto_resolve_trainee(trainee_id)

        if self.verbose:
            print(f'Getting mean decrease in accuracy for trainee with id: '
                  f'{trainee_id}')

        mda = self.howso.get_feature_mda(
            trainee_id,
            action_feature=action_feature,
            permutation=permutation,
            robust=robust,
            weight_feature=weight_feature)
        return mda

    def get_feature_contributions(
        self,
        trainee_id: str,
        action_feature: str,
        *,
        robust: Optional[bool] = None,
        directional: bool = False,
        weight_feature: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get cached feature contributions.

        All keyword arguments are optional. When not specified, will
        auto-select cached contributions for output. When specified, will
        attempt to output the cached contributions best matching the requested
        parameters, if no cached match is found.

        .. deprecated:: 1.0.0
            Use :meth:`HowsoDirectClient.get_prediction_stats` instead.

        Parameters
        ----------
        trainee_id : str
            The id or name of the trainee.
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
        dict of str to float
            A map of feature names to contribution values.
        """
        warnings.warn(
            'The method `get_feature_contributions()` is deprecated and will be'
            'removed in a future release. Please use `get_prediction_stats()` '
            'instead.', DeprecationWarning)
        self._auto_resolve_trainee(trainee_id)

        if self.verbose:
            print(f'Getting feature contributions for trainee with id: '
                  f'{trainee_id}')

        contributions = self.howso.get_feature_contributions(
            trainee_id,
            action_feature=action_feature,
            robust=robust,
            directional=directional,
            weight_feature=weight_feature)
        return contributions

    def get_prediction_stats(
        self,
        trainee_id: str,
        *,
        action_feature: Optional[str] = None,
        condition: Optional[Dict[str, Any]] = None,
        num_cases: Optional[int] = None,
        num_robust_influence_samples_per_case=None,
        precision: Optional[str] = None,
        robust: Optional[bool] = None,
        robust_hyperparameters: Optional[bool] = None,
        stats: Optional[Iterable[str]] = None,
        weight_feature: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get feature prediction stats.

        Gets cached stats when condition is None.
        If condition is not None, then uses the condition to select cases and
        computes prediction stats for that set of cases.

        All keyword arguments are optional, when not specified will auto-select
        all cached stats for output, when specified will attempt to
        output the cached stats best matching the requested parameters,
        if no cached match is found.

        Parameters
        ----------
        trainee_id : str
            The id or name of the trainee.
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
        stats : iterable of str, optional
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
        dict of str to dict of str to float
            A map of feature to map of stat type to stat values.
        """
        self._auto_resolve_trainee(trainee_id)

        if self.verbose:
            print('Getting feature prediction stats for trainee with '
                  f'id: {trainee_id}')

        validate_list_shape(stats, 1, "stats", "str")
        valid_stats = {
            "accuracy", "contribution", "confusion_matrix", "mae", "mda",
            "mda_permutation", "precision", "r2", "recall", "rmse",
            "spearman_coeff",
        }

        if stats is not None and not set(stats).issubset(valid_stats):
            raise ValueError(
                'Invalid prediction stats. The following stats are supported: '
                f'{", ".join(valid_stats)}'
            )

        stats = self.howso.get_prediction_stats(
            trainee_id,
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
        return stats

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
            The ID of the Trainee to retrieve marginal stats for.
        weight_feature : str, optional
            When specified, will attempt to return stats that were computed
            using this weight_feature.

        Returns
        -------
        dict of str to dict of str to float
            A map of feature names to map of stat type to stat values.
        """
        self._auto_resolve_trainee(trainee_id)

        if self.verbose:
            print('Getting feature marginal stats for trainee with '
                  f'id: {trainee_id}')

        stats = self.howso.get_marginal_stats(
            trainee_id,
            weight_feature=weight_feature)
        return stats

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
    ):
        """
        Compute and cache specified feature prediction stats.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to react to.
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

        Returns
        -------
        None
        """
        self._auto_resolve_trainee(trainee_id)

        validate_list_shape(context_features, 1, "context_features", "str")

        # if any mda or contributions flags are specified,
        # must specify action_feature
        if [
            contributions,
            contributions_robust,
            mda,
            mda_robust,
            mda_permutation,
            mda_robust_permutation
        ].count(None) != 6:
            if action_feature is None:
                raise ValueError(
                    "Invalid value for `action_feature`, must not be `None`")

        if self.verbose:
            print(f'Reacting into trainee for trainee with id: {trainee_id}')
        self.howso.react_into_trainee(
            trainee_id,
            context_features=context_features,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
            num_samples=num_samples,
            residuals=residuals,
            residuals_robust=residuals_robust,
            contributions=contributions,
            contributions_robust=contributions_robust,
            mda=mda,
            mda_permutation=mda_permutation,
            mda_robust=mda_robust,
            mda_robust_permutation=mda_robust_permutation,
            num_robust_influence_samples=num_robust_influence_samples,
            num_robust_residual_samples=num_robust_residual_samples,
            num_robust_influence_samples_per_case=num_robust_influence_samples_per_case,
            hyperparameter_param_path=hyperparameter_param_path,
            sample_model_fraction=sample_model_fraction,
            sub_model_size=sub_model_size,
            action_feature=action_feature)

        self._auto_persist_trainee(trainee_id)

    def get_extreme_cases(
        self,
        trainee_id: str,
        num: int,
        sort_feature: str,
        features: Optional[Iterable[str]] = None
    ) -> client_models.Cases:
        """
        Gets the extreme cases of a trainee for the given feature(s).

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to retrieve extreme cases from.
        num : int
            The number of cases to get.
        sort_feature : str
            The feature name by which extreme cases are sorted by.
        features: iterable of str, optional
            An iterable of feature names to use when getting extreme cases.

        Returns
        -------
        howso.openapi.models.Cases
            A cases object containing the feature names and extreme cases.
        """
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Getting extreme cases for trainee with id: {trainee_id}')
        result = self.howso.retrieve_extreme_cases_for_feature(
            trainee_id,
            features=features,
            sort_feature=sort_feature,
            num=num)
        if result is None:
            result = dict()
        return client_models.Cases(features=result.get('features'),
                                   cases=result.get('cases'))

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

    def move_cases(
        self,
        trainee_id: str,
        target_trainee_id: str,
        num_cases: int,
        *,
        case_indices: Optional[Iterable[Tuple[str, int]]] = None,
        condition: Optional[Dict] = None,
        condition_session: Optional[str] = None,
        precision: Optional[Literal["exact", "similar"]] = None,
        preserve_session_data: bool = False
    ) -> int:
        """
        Moves training cases from one trainee to another trainee.

        Parameters
        ----------
        trainee_id : str
            The source trainee to move a cases from.
        target_trainee_id : str
            The target trainee to move the cases to.
        num_cases : int
            The number of cases to move; minimum 1 case must be moved.
            Ignored if case_indices is specified.
        case_indices : list of tuples
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

        Returns
        -------
        int
            The number of cases moved.
        """
        self._auto_resolve_trainee(trainee_id)
        self._auto_resolve_trainee(target_trainee_id)
        if num_cases < 1:
            raise ValueError('num_cases must be a value greater than 0')

        if self.verbose:
            print(f'Moving case from trainee with id: {trainee_id} to trainee '
                  f'with id: {target_trainee_id}')

        # Convert session instance to id
        if (
            isinstance(condition, dict) and
            isinstance(condition.get('.session'), Session)
        ):
            condition['.session'] = condition['.session'].id

        result = self.howso.move_cases(
            trainee_id, target_trainee_id,
            case_indices=case_indices,
            condition=condition,
            condition_session=condition_session,
            num_cases=num_cases,
            precision=precision,
            preserve_session_data=preserve_session_data,
            session=self.active_session.id
        )
        self._auto_persist_trainee(trainee_id)
        return result.get('count', 0)

    def get_params(
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
            that were analyzed using this weight feaure.

        Returns
        -------
        dict
            A dict including the either all of the Trainee's internal
            parameters or only the best hyperparameters selected using the
            passed parameters.
        """
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Getting model attributes from trainee with id: '
                  f'{trainee_id}')
        return self.howso.get_internal_parameters(
            trainee_id,
            action_feature=action_feature,
            context_features=context_features,
            mode=mode,
            weight_feature=weight_feature,
        )

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

    def set_params(self, trainee_id: str, params: Dict):
        """
        Sets specific hyperparameters in the trainee.

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
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f'Setting model attributes for trainee with id: {trainee_id}')

        deprecated_params = {
            'auto_optimize_enabled': 'auto_analyze_enabled',
            'optimize_threshold': 'analyze_threshold',
            'optimize_growth_factor': 'analyze_growth_factor',
            'auto_optimize_limit_size': 'auto_analyze_limit_size',
        }

        # replace any old params with new params and remove old param
        for old_param, new_param in deprecated_params.items():
            if old_param in params:
                params[new_param] = params[old_param]
                del params[old_param]
                warnings.warn(
                    f'The `{old_param}` parameter has been renamed to '
                    f'`{new_param}`, please use the new parameter '
                    'instead.', UserWarning)

        self.howso.set_internal_parameters(trainee_id, params)
        self._auto_persist_trainee(trainee_id)

    def get_num_training_cases(self, trainee_id: str) -> int:
        """
        Return the number of trained cases in the model.

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
        ret = self.howso.get_num_training_cases(trainee_id)
        if isinstance(ret, dict):
            return ret.get('count', 0)
        return 0

    def set_auto_analyze_params(  # noqa: C901
        self,
        trainee_id: str,
        auto_analyze_enabled: bool = False,
        analyze_threshold: Optional[int] = None,
        *,
        auto_analyze_limit_size: Optional[int] = None,
        analyze_growth_factor: Optional[float] = None,
        **kwargs
    ):
        """
        Set trainee parameters for auto analysis.

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
        self._auto_resolve_trainee(trainee_id)

        deprecated_params = {
            'auto_optimize_enabled': 'auto_analyze_enabled',
            'optimize_threshold': 'analyze_threshold',
            'optimize_growth_factor': 'analyze_growth_factor',
            'auto_optimize_limit_size': 'auto_analyze_limit_size',
        }
        analyze_deprecated_params = {
            'bypass_hyperparameter_optimization': 'bypass_hyperparameter_analysis',
            'num_optimization_samples': 'num_analysis_samples',
            'optimization_sub_model_size': 'analysis_sub_model_size',
            'optimize_level': 'analyze_level',
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
                    elif old_param == 'auto_optimize_limit_size':
                        auto_analyze_limit_size = kwargs[old_param]

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
        parameters = {}
        for k in dict(kwargs).keys():
            if k in client_models.SetAutoAnalyzeParamsRequest.attribute_map:
                v = kwargs.pop(k)
                if (
                    v is not None or
                    k in client_models.SetAutoAnalyzeParamsRequest.nullable_attributes
                ):
                    parameters[k] = v

        if kwargs:
            warn_params = ', '.join(kwargs)
            warnings.warn(
                f'The following auto analyze parameter(s) "{warn_params}" '
                'are not officially supported by analyze and may or may not '
                'have an effect.', UserWarning)

        self.howso.auto_analyze_params(
            trainee_id,
            auto_analyze_enabled,
            analyze_threshold,
            analyze_growth_factor,
            auto_analyze_limit_size,
            **parameters,
            **kwargs
        )
        self._auto_persist_trainee(trainee_id)

    def optimize(self, *args, **kwargs):
        """
        Optimizes a trainee.

        .. deprecated:: 6.0.0
            Use :meth:`HowsoDirectClient.analyze` instead.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
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

    def auto_optimize(self, trainee_id):
        """
        Auto-optimize the trainee model.

        Re-uses all parameters from the previous optimize or
        set_auto_optimize_params call. If optimize or set_auto_optimize_params
        has not been previously called, auto_optimize will default to a robust
        and versatile optimization.

        .. deprecated:: 6.0.0
            Use :meth:`HowsoDirectClient.auto_analyze` instead.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to auto-optimize.
        """
        warnings.warn(
            'The method `auto_optimize()` is deprecated and will be'
            'removed in a future release. Please use `auto_analyze()` '
            'instead.', DeprecationWarning)

        return self.auto_analyze(trainee_id)

    def set_auto_optimize_params(self, *args, **kwargs):
        """
        Set trainee parameters for auto optimization.

        .. deprecated:: 6.0.0
            Use :meth:`HowsoDirectClient.set_auto_analyze_params` instead.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to set auto optimization parameters for.
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

    def auto_analyze(self, trainee_id: str):
        """
        Auto-analyze the trainee model.

        Re-uses all parameters from the previous analyze or
        set_auto_analyze_params call. If analyze or set_auto_analyze_params
        has not been previously called, auto_analyze will default to a robust
        and versatile analysis.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to auto-analyze.
        """
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print(f"Auto-analyzing trainee with id: {trainee_id}")

        self.howso.auto_analyze(trainee_id)
        self._auto_persist_trainee(trainee_id)
        # when debugging output the auto-analyzed parameters into the
        # trace file.
        if self.howso.trace:
            self.howso.get_internal_parameters(trainee_id)

    def get_label(self, entity_id: str, label: str) -> object:
        """
        Get a label value from a Trainee.

        Parameters
        ----------
        entity_id : str
            The ID of the Trainee to get the label from.
        label : str
            The label name to get the value from.

        Returns
        -------
        object
            The value of the label requested.
        """
        if self.verbose:
            print(f'Gets a label from trainee with id: {entity_id}')
        return self.howso.amlg.get_json_from_label(entity_id, label)

    def set_label(self, entity_id: str, label: str, label_value: str):
        """
        Set a label value in the trainee.

        Parameters
        ----------
        entity_id : str
            The ID of the Trainee containing/to contain the label.
        label : str
            The name of the label.
        label_value : object
            The value to set to the label.
        """
        if self.verbose:
            print(f'Setting label for trainee with id: {entity_id}')
        return self.howso.amlg.set_json_to_label(
            entity_id, label, json.dumps(label_value))

    def execute_label(self, entity_id: str, label: str) -> object:
        """
        Execute a label in the trainee.

        Parameters
        ----------
        entity_id : str
            The ID of the Trainee that contains the label to be executed.
        label : str
            The name of the label to execute.

        Returns
        -------
            The raw response from the trainee.
        """
        if self.verbose:
            print(f'Executing label for trainee with id: {entity_id}')
        return self.howso.amlg.execute_entity_json(entity_id, label, "{}")

    def get_pairwise_distances(  # noqa: C901
        self,
        trainee_id: str,
        features: Optional[Iterable[str]] = None,
        *,
        action_feature: Optional[str] = None,
        from_case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        from_values: Optional[Union[List[List[object]], DataFrame]] = None,
        to_case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        to_values: Optional[Union[List[List[object]], DataFrame]] = None,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None
    ) -> List:
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
        self._auto_resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features

        validate_list_shape(from_values, 2, 'from_values',
                            'list of list of object')
        validate_list_shape(to_values, 2, 'to_values',
                            'list of list of object')

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
            validate_case_indices(from_case_indices)
        if to_case_indices:
            validate_case_indices(to_case_indices)

        # Serialize values if defined
        if from_values is not None:
            if features is None:
                features = internals.get_features_from_data(
                    from_values, data_parameter='from_values')
            from_values = serialize_cases(from_values, features,
                                          feature_attributes)
        if to_values is not None:
            if features is None:
                features = internals.get_features_from_data(
                    to_values, data_parameter='to_values')
            to_values = serialize_cases(to_values, features, feature_attributes)

        if self.verbose:
            print('Getting pairwise distances for trainee with id: '
                  f'{trainee_id}')

        result = self.howso.pairwise_distances(
            trainee_id,
            features=features,
            action_feature=action_feature,
            from_case_indices=from_case_indices,
            from_values=from_values,
            to_case_indices=to_case_indices,
            to_values=to_values,
            weight_feature=weight_feature,
            use_case_weights=use_case_weights)
        if result is None:
            return []
        return result

    def get_distances(  # noqa: C901
        self,
        trainee_id: str,
        features: Optional[Iterable[str]] = None,
        *,
        action_feature: Optional[str] = None,
        case_indices: Optional[Iterable[Sequence[Union[str, int]]]] = None,
        feature_values: Optional[Union[List[object], DataFrame]] = None,
        use_case_weights: bool = False,
        weight_feature: Optional[str] = None
    ) -> Dict:
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
        self._auto_resolve_trainee(trainee_id)
        feature_attributes = self.trainee_cache.get(trainee_id).features

        # Validate case_indices if provided
        if case_indices is not None:
            validate_case_indices(case_indices)

        if feature_values is not None:
            if (
                isinstance(feature_values, Iterable)
                and len(np.array(feature_values).shape) == 1
                and len(feature_values) > 0
            ):
                # Convert 1d list to 2d list for serialization
                feature_values = [feature_values]

            if features is None:
                features = internals.get_features_from_data(
                    feature_values, data_parameter='feature_values')
            feature_values = serialize_cases(feature_values, features,
                                             feature_attributes)
            if feature_values:
                # Only a single case should be provided
                feature_values = feature_values[0]
            # Ignored when feature_values specified
            case_indices = None

        if case_indices is not None and len(case_indices) < 2:
            raise ValueError("If providing `case_indices`, must provide at "
                             "least 2 cases for computation.")

        if self.verbose:
            print('Getting distances between cases for trainee with id: '
                  f'{trainee_id}')

        preallocate = True  # If matrix should be preallocated in memory
        page_size = 2000
        indices = []
        distances_matrix = []
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
            distances_matrix = np.zeros((num_cases, num_cases), dtype='float64')

        for row_offset in range(0, num_cases, page_size):
            for column_offset in range(0, num_cases, page_size):
                response = self.howso.distances(
                    trainee_id,
                    features=features,
                    action_feature=action_feature,
                    case_indices=case_indices,
                    feature_values=feature_values,
                    weight_feature=weight_feature,
                    use_case_weights=use_case_weights,
                    row_offset=row_offset,
                    row_count=page_size,
                    column_offset=column_offset,
                    column_count=page_size,
                )
                column_case_indices = response['column_case_indices']
                row_case_indices = response['row_case_indices']
                distances = response['distances']

                if preallocate:
                    # Fill in allocated matrix
                    try:
                        distances_matrix[
                            row_offset:row_offset + len(row_case_indices),
                            column_offset:column_offset + len(column_case_indices)
                        ] = distances
                    except ValueError as err:
                        # Unexpected shape when populating array
                        raise HowsoError(mismatch_msg) from err
                else:
                    if column_offset == 0:
                        # Append new rows
                        distances_matrix += distances
                    else:
                        # Extend existing columns
                        try:
                            for i, cols in enumerate(distances):
                                distances_matrix[row_offset + i].extend(cols)
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
        else:
            if distances_matrix:
                # Validate matrix shape
                if (
                    total_cols != total_rows or
                    not all(len(r) == total_cols for r in distances_matrix)
                ):
                    raise HowsoError(mismatch_msg)
            # If we didn't preallocate, matrix is a python list, convert it
            distances_matrix = np.array(distances_matrix, dtype='float64')

        return {
            'case_indices': indices,
            'distances': distances_matrix
        }

    def compute_feature_weights(
        self,
        trainee_id: str,
        action_feature: Optional[str] = None,
        context_features: Optional[Iterable[str]] = None,
        robust: bool = False,
        weight_feature: Optional[str] = None,
        use_case_weights: bool = False
    ) -> Dict[str, float]:
        """
        Compute and set feature weights for specified context and action features.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
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
            A dictionary of computed context features -> weights
        """
        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        if action_feature is None and cached_trainee.default_action_features:
            action_feature = cached_trainee.default_action_features[0]
        if context_features is None:
            context_features = cached_trainee.default_context_features

        weights = self.howso.compute_feature_weights(
            trainee_id, action_feature, context_features, robust,
            weight_feature, use_case_weights)
        self._auto_persist_trainee(trainee_id)
        return weights

    def set_feature_weights(
        self,
        trainee_id: str,
        feature_weights: Optional[Dict[str, float]] = None,
        action_feature: Optional[str] = None,
        use_feature_weights: bool = True
    ):
        """
        Set the weights for the features in the Trainee.

        If action_feature is not specified, it will set the passed in weights
        as targetless.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        action_feature : str, optional
            Action feature for which to set the specified feature weights for
        feature_weights : dict, optional
            A dictionary of feature names -> weight values.
            Ex {"a", 1.0, "b": 0.1, "c": 0.5, ... , "z": 1.0}
            If not set, the feature weights are cleared in the model
        use_feature_weights : bool, default True
            When set to true, forces the trainee to use the specified feature
            weights
        """
        self._auto_resolve_trainee(trainee_id)
        self.howso.set_feature_weights(
            trainee_id, feature_weights, action_feature, use_feature_weights)
        self._auto_persist_trainee(trainee_id)

    def set_feature_weights_matrix(
        self,
        trainee_id: str,
        feature_weights_matrix: Dict[str, Dict[str, float]],
        use_feature_weights: bool = True
    ):
        """
        Set the feature weights for all the features in the Trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        feature_weights_matrix : dict
            A dictionary of feature names to a dictionary of feature names to
            weight values.
            i.e. {"a" : {"a", 1.0, "b": 0.1, "c": 0.5, ... , "z": 1.0} }
        use_feature_weights : bool, default True
            When set to true, forces the trainee to use the specified feature
            weights.
        """
        self._auto_resolve_trainee(trainee_id)
        self.howso.set_feature_weights_matrix(
            trainee_id, feature_weights_matrix, use_feature_weights)
        self._auto_persist_trainee(trainee_id)

    def get_feature_weights_matrix(
        self,
        trainee_id: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the full feature weights matrix.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.

        Returns
        -------
        dict
            A dictionary of action feature names to dictionary of feature names
            to feature weight.
        """
        self._auto_resolve_trainee(trainee_id)
        return self.howso.get_feature_weights_matrix(trainee_id)

    def get_feature_attributes(self, trainee_id: str) -> Dict[str, Dict]:
        """
        Get stored feature attributes.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee

        Returns
        -------
        dict
            A dictionary of feature name to dictionary of feature attributes.
        """
        self._auto_resolve_trainee(trainee_id)
        if self.verbose:
            print('Getting feature attributes from trainee with '
                  f'id: {trainee_id}')
        feature_attributes = self.howso.get_feature_attributes(trainee_id)
        return internals.postprocess_feature_attributes(feature_attributes)

    def set_feature_attributes(
        self,
        trainee_id: str,
        feature_attributes: Dict[str, Dict],
    ):
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
        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        if not isinstance(feature_attributes, dict):
            raise ValueError("`feature_attributes` must be a dict")
        if self.verbose:
            print('Setting feature attributes for trainee with '
                  f'id: {trainee_id}')

        fixed_attribs = internals.preprocess_feature_attributes(
            feature_attributes)
        self.howso.set_feature_attributes(trainee_id, fixed_attribs)
        self._auto_persist_trainee(trainee_id)

        updated_feature_attributes = self.howso.get_feature_attributes(trainee_id)
        # Update trainee in cache
        cached_trainee.features = internals.postprocess_feature_attributes(
            updated_feature_attributes)

    def analyze(
        self,
        trainee_id: str,
        context_features: Optional[Iterable[str]] = None,
        action_features: Optional[Iterable[str]] = None,
        *,
        bypass_calculate_feature_residuals: bool = None,
        bypass_calculate_feature_weights: bool = None,
        bypass_hyperparameter_analysis: bool = None,
        dt_values: Optional[List[float]] = None,
        use_case_weights: bool = None,
        inverse_residuals_as_weights: bool = None,
        k_folds: Optional[int] = None,
        k_values: Optional[List[int]] = None,
        num_analysis_samples: Optional[int] = None,
        num_samples: Optional[int] = None,
        analysis_sub_model_size: Optional[int] = None,
        analyze_level: Optional[int] = None,
        p_values: Optional[List[float]] = None,
        targeted_model: Optional[Literal["omni_targeted", "single_targeted", "targetless"]] = None,
        use_deviations: bool = None,
        weight_feature: Optional[str] = None,
        **kwargs
    ):
        """
        Analyzes a trainee.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee.
        context_features : iterable of str, optional
            The context features to analyze for.
        action_features : iterable of str, optional
            The action features to analyze for.
        k_folds : int
            optional, (default 6) number of cross validation folds to do
        bypass_hyperparameter_analysis : bool
            optional, bypasses hyperparameter analysis
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
        dt_values : list of float
            optional list used in hyperparameter search
        analyze_level : int
            optional value, if specified, will analyze for the following
            flows:

                1. predictions/accuracy (hyperparameters)
                2. data synth (cache: global residuals)
                3. standard explanations
                4. full analysis
        targeted_model : {"omni_targeted", "single_targeted", "targetless"}
            optional, valid values as follows:

                "single_targeted" = analyze hyperparameters for the
                    specified action_features
                "omni_targeted" = analyze hyperparameters for each context
                    feature as an action feature, ignores action_features
                    parameter
                "targetless" = analyze hyperparameters for all context
                    features as possible action features, ignores
                    action_features parameter
        num_analysis_samples : int, optional
            If the dataset size to too large, analyze on
            (randomly sampled) subset of data. The
            `num_analysis_samples` specifies the number of
            observations to be considered for analysis.
        analysis_sub_model_size : int or Node, optional
            Number of samples to use for analysis. The rest
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
            Additional experimental analyze parameters.
        """
        self._auto_resolve_trainee(trainee_id)
        cached_trainee = self.trainee_cache.get(trainee_id)

        validate_list_shape(context_features, 1, "context_features", "str")
        validate_list_shape(action_features, 1, "action_features", "str")
        validate_list_shape(p_values, 1, "p_values", "int")
        validate_list_shape(k_values, 1, "k_values", "float")
        validate_list_shape(dt_values, 1, "dt_values", "float")

        if targeted_model not in ['single_targeted', 'omni_targeted', 'targetless', None]:
            raise ValueError(
                f'Invalid value "{targeted_model}" for targeted_model. '
                'Valid values include single_targeted, omni_targeted, '
                'and targetless.')

        if action_features is None:
            action_features = cached_trainee.default_action_features
        if context_features is None:
            context_features = cached_trainee.default_context_features

        deprecated_params = {
            'bypass_hyperparameter_optimization': 'bypass_hyperparameter_analysis',
            'num_optimization_samples': 'num_analysis_samples',
            'optimization_sub_model_size': 'analysis_sub_model_size',
            'optimize_level': 'analyze_level',
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
                    elif old_param == 'optimize_level':
                        analyze_level = kwargs[old_param]
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
            analyze_level=analyze_level,
            p_values=p_values,
            targeted_model=targeted_model,
            use_deviations=use_deviations,
            weight_feature=weight_feature,
        )

        # Filter out non nullable parameters
        analyze_params = {
            k: v for k, v in analyze_params.items()
            if v is not None or
            k in client_models.AnalyzeRequest.nullable_attributes
        }
        # Add experimental options
        analyze_params.update(kwargs)

        if kwargs:
            warn_params = ', '.join(kwargs)
            warnings.warn(
                f'The following analyze parameter(s) "{warn_params}" are '
                'not officially supported by analyze and may or may not '
                'have an effect.', UserWarning)

        if self.verbose:
            print(f'Analyzing trainee with id: {trainee_id}')
            print(f'Analyzing trainee with parameters: {analyze_params}')

        self.howso.analyze(trainee_id, **analyze_params)
        self._auto_persist_trainee(trainee_id)

    def evaluate(
        self,
        trainee_id: str,
        features_to_code_map: Dict[str, str],
        *,
        aggregation_code: Optional[str] = None
    ) -> Dict[str, Dict[str, object]]:
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
        self._auto_resolve_trainee(trainee_id)

        evaluate_params = dict(
            features_to_code_map=features_to_code_map,
            aggregation_code=aggregation_code,
        )

        return self.howso.evaluate(trainee_id, **evaluate_params)
