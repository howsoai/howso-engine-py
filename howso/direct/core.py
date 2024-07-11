from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

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
    sbf_datastore_enabled : bool, default None
        If true, sbf tree structures are enabled.
    max_num_threads : int, default None
        If a multithreaded Amalgam binary is used, sets the maximum number of
        threads to the value specified. If 0, will use the number of visible
        logical cores.
    """

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

    def export_trainee(
        self,
        trainee_id: str,
        path_to_trainee: Optional[Union[Path, str]] = None,
        decode_cases: bool = False,
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
        """
        if path_to_trainee is None:
            path_to_trainee = self.default_save_path

        return self._execute(trainee_id, "export_trainee", {
            "trainee_filepath": f"{path_to_trainee}/",
            "trainee": f"{trainee_id}",
            "root_filepath": f"{self.howso_path}/",
            "decode_cases": decode_cases,
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
