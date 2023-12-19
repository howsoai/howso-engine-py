from typing import List, Optional, TYPE_CHECKING

from howso.client.exceptions import HowsoError
from howso.client.protocols import ProjectClient
from howso.engine.client import get_client
from howso.openapi.models import Project as BaseProject

if TYPE_CHECKING:
    from datetime import datetime
    from howso.openapi.models import AccountIdentity

__all__ = [
    'delete_project',
    'get_project',
    'list_projects',
    'Project',
    'switch_project',
]


class Project(BaseProject):
    """
    A Howso Project.

    A Project is a container for a collection of Trainees. Allowing
    control over who may view and modify the Trainees based on their
    membership access to the project.

    Parameters
    ----------
    name : str
        The name of the project.
    client : ProjectClient, optional
        The Howso client instance to use. Must support the project API.
    """

    def __init__(
        self,
        name: str = None,
        *,
        id: Optional[str] = None,
        client: Optional[ProjectClient] = None
    ) -> None:
        """Implement the constructor."""
        self._created: bool = False
        self._updating: bool = False
        self.client = client or get_client()

        # Set the project properties
        self._id = id
        self._name = None
        self._is_private = True
        self._is_default = False
        self._created_by = None
        self._created_date = None
        self._modified_date = None
        self._permissions = None

        self.name = name

        # Create the project at the API
        self._create()

    @property
    def id(self) -> str:
        """
        The unique identifier of the project.

        Returns
        -------
        str
            The project ID.
        """
        return self._id

    @property
    def name(self) -> str:
        """
        The name of the project.

        Returns
        -------
        str
            The project name.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        Set the name of the project.

        Parameters
        ----------
        name : str
            The name of the project.

        Returns
        -------
        None
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if len(name) > 128:
            raise ValueError("Invalid value for `name`, length must be less "
                             "than or equal to `128`")
        self._name = name
        self._update()

    @property
    def is_private(self) -> bool:
        """
        Designates if the project is not publicly visible.

        Returns
        -------
        bool
            True, when the project not public.
        """
        return self._is_private

    @property
    def is_default(self) -> bool:
        """
        If this project is the current user's default project.

        Returns
        -------
        bool
            True, when the project is the user's default.
        """
        return self._is_default

    @property
    def created_by(self) -> Optional["AccountIdentity"]:
        """
        The user account that created this project.

        Returns
        -------
        AccountIdentity
            The user account information.
        """
        return self._created_by

    @property
    def created_date(self) -> Optional["datetime"]:
        """
        The timestamp of when the project was originally created.

        Returns
        -------
        datetime
            The creation timestamp.
        """
        return self._created_date

    @property
    def modified_date(self) -> Optional["datetime"]:
        """
        The timestamp of when the project was last modified.

        Returns
        -------
        datetime
            The modified timestamp.
        """
        return self._modified_date

    @property
    def permissions(self) -> Optional[List[str]]:
        """
        Permissions types the user has in this project.

        Returns
        -------
        list of str
            The list of permission types.
        """
        return self._permissions

    @property
    def client(self) -> ProjectClient:
        """
        The client instance used by the project.

        Returns
        -------
        ProjectClient
            The client instance.
        """
        return self._client

    @client.setter
    def client(self, client: ProjectClient) -> None:
        """
        Set the client instance used by the project.

        Parameters
        ----------
        client : ProjectClient
            The client instance. Must support the project API.

        Returns
        -------
        None
        """
        if not isinstance(client, ProjectClient):
            raise HowsoError("Projects are not supported by the active "
                             "Howso client.")
        self._client = client

    def delete(self) -> None:
        """
        Delete the project.

        Projects may only be deleted when they have no trainees in them.

        Returns
        -------
        None
        """
        if not self.id:
            return
        self.client.delete_project(self.id)
        self._created = False
        self._id = None

    def _update_attributes(self, project: BaseProject) -> None:
        """
        Update the protected attributes of the project.

        Parameters
        ----------
        project : BaseProject
            The base project instance.

        Returns
        -------
        None
        """
        for key in self.attribute_map.keys():
            # Update the protected attributes directly since the values
            # have already been validated by the "BaseProject" instance
            # and to prevent triggering an API update call
            setattr(self, f'_{key}', getattr(project, key))

    def _update(self) -> None:
        """
        Update the project at the API.

        Returns
        -------
        None
        """
        if (
            getattr(self, 'id', None)
            and getattr(self, '_created', False)
            and not getattr(self, '_updating', False)
        ):
            # Only update for projects that have been created
            self._updating = True
            updated_project = self.client.update_project(
                project_id=self.id,
                name=self.name
            )
            self._update_attributes(updated_project)
            self._updating = False

    def _create(self) -> None:
        """
        Create the project at the API.

        Returns
        -------
        None
        """
        if not self.id:
            project = self.client.create_project(name=self.name)
            self._update_attributes(project)
        self._created = True

    @classmethod
    def from_openapi(
        cls, project: BaseProject, *,
        client: Optional[ProjectClient] = None
    ) -> "Project":
        """
        Create Project from base class.

        Parameters
        ----------
        project : BaseProject
            The base project instance.
        client : ProjectClient, optional
            The Howso client instance to use.

        Returns
        -------
        Project
            The project instance.
        """
        project_dict = project.to_dict()
        project_dict['client'] = client
        return cls.from_dict(project_dict)

    @classmethod
    def from_dict(cls, project_dict: dict) -> "Project":
        """
        Create Project from dict.

        Parameters
        ----------
        project_dict : Dict
            The Project parameters.

        Returns
        -------
        Project
            The project instance.
        """
        if not isinstance(project_dict, dict):
            raise ValueError('`project_dict` parameter is not a dict')
        parameters = {
            'id': project_dict.get('id'),
            'name': project_dict.get('name'),
            'client': project_dict.get('client')
        }
        instance = cls(**parameters)
        for key in cls.attribute_map.keys():
            if key in project_dict:
                setattr(instance, f'_{key}', project_dict[key])
        return instance

    def __enter__(self) -> "Project":
        """Support context managers."""
        setattr(self, '_last_active_project', self.client.active_project)
        self.client.switch_project(self.id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Revert project during exit of context manager."""
        last_project = getattr(self, '_last_active_project', None)
        if last_project:
            self.client.switch_project(last_project.id)
            setattr(self, '_last_active_project', None)


def delete_project(
    project_id: str,
    *,
    client: Optional[ProjectClient] = None
) -> None:
    """
    Delete an existing project.

    Projects may only be deleted when they have no trainees in them.

    Parameters
    ----------
    project_id : str
        The id of the project.
    client : ProjectClient, optional
        The Howso client instance to use.

    Returns
    -------
    None
    """
    client = client or get_client()
    client.delete_project(str(project_id))


def get_project(
    project_id: str,
    *,
    client: Optional[ProjectClient] = None
) -> Project:
    """
    Get an existing project.

    Parameters
    ----------
    project_id : str
        The id of the project.
    client : ProjectClient, optional
        The Howso client instance to use.

    Returns
    -------
    Project
        The project instance.
    """
    client = client or get_client()
    if not isinstance(client, ProjectClient):
        raise HowsoError("Projects are not supported by the active "
                         "Howso client.")

    project = client.get_project(str(project_id))
    return Project.from_openapi(project, client=client)


def list_projects(
    search_terms: Optional[str] = None,
    *,
    client: Optional[ProjectClient] = None
) -> List[Project]:
    """
    Get listing of projects.

    Parameters
    ----------
    search_terms : str
        Terms to filter results by.
    client : ProjectClient, optional
        The Howso client instance to use.

    Returns
    -------
    list of Project
        The list of project instances.
    """
    client = client or get_client()
    if not isinstance(client, ProjectClient):
        raise HowsoError("Projects are not supported by the active "
                         "Howso client.")
    projects = client.get_projects(search_terms)
    return [Project.from_openapi(p, client=client) for p in projects]


def switch_project(
    project_id: str,
    *,
    client: Optional[ProjectClient] = None
) -> Project:
    """
    Set the active project.

    Parameters
    ----------
    project_id : str
        The id of the project.
    client : ProjectClient, optional
        The Howso client instance to use.

    Returns
    -------
    Project
        The newly active project instance.
    """
    client = client or get_client()
    if not isinstance(client, ProjectClient):
        raise HowsoError("Projects are not supported by the active "
                         "Howso client.")
    client.switch_project(str(project_id))
    return Project.from_openapi(client.active_project, client=client)
