from __future__ import annotations

from collections.abc import Mapping
from typing import Optional
from uuid import UUID
import warnings

from howso.client.exceptions import HowsoError
from howso.client.protocols import ProjectClient
from howso.client.schemas import BaseSchema, Project as BaseProject
from howso.engine.client import get_client


__all__ = [
    'delete_project',
    'get_project',
    'list_projects',
    'Project',
    'query_projects',
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
        The Howso client instance to use. Must support the Project API.
    """

    def __init__(
        self,
        name: str,
        *,
        id: Optional[str] = None,
        client: Optional[ProjectClient] = None
    ) -> None:
        """Implement the constructor."""
        self._created: bool = False
        self._updating: bool = False
        self.client = client or get_client()  # type: ignore

        # Initialize the project properties
        # The id will be initialized by _create
        super().__init__(id=id or '', name=name)

        # Create the project at the API
        self._create()

    @BaseProject.name.setter
    def name(self, name: str) -> None:
        """
        Set the name of the Project.

        Parameters
        ----------
        name : str
            The name of the Project.
        """
        if BaseProject.name.fset is None:
            raise AttributeError("Project.name has no setter")
        # Call super class setter
        BaseProject.name.fset(self, name)
        self._update()

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
            raise HowsoError("Projects are not supported by the active Howso client.")
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
        for key in self.attribute_map:
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
    def from_schema(
        cls,
        schema: BaseSchema,
        *,
        client: Optional[ProjectClient] = None
    ) -> "Project":
        """
        Create Project from base class.

        Parameters
        ----------
        schema : howso.client.schemas.Project
            The base Project object.
        client : ProjectClient, optional
            The Howso client instance to use.

        Returns
        -------
        Project
            The Project instance.
        """
        if isinstance(schema, cls) and client is None:
            return schema
        project_dict = schema.to_dict()
        project_dict['client'] = client
        return cls.from_dict(project_dict)

    @classmethod
    def from_dict(cls, schema: Mapping):
        """Returns a new Project using properties from dict."""
        if not isinstance(schema, Mapping):
            raise ValueError('`schema` parameter is not a Mapping')
        parameters: dict = {
            'id': schema.get('id'),
            'name': schema.get('name'),
            'client': schema.get('client'),
        }
        instance = cls(**parameters)
        for key in cls.attribute_map:
            if key in schema and key not in parameters:
                setattr(instance, f'_{key}', schema[key])
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
    project_id: str | UUID,
    *,
    client: Optional[ProjectClient] = None
) -> None:
    """
    Delete an existing project.

    Projects may only be deleted when they have no trainees in them.

    Parameters
    ----------
    project_id : str or UUID
        The id of the project.
    client : ProjectClient, optional
        The Howso client instance to use.

    Returns
    -------
    None
    """
    cl = client or get_client()
    if not isinstance(cl, ProjectClient):
        raise HowsoError("Projects are not supported by the active Howso client.")
    cl.delete_project(str(project_id))


def get_project(
    project_id: str | UUID,
    *,
    client: Optional[ProjectClient] = None
) -> Project:
    """
    Get an existing project.

    Parameters
    ----------
    project_id : str or UUID
        The id of the project.
    client : ProjectClient, optional
        The Howso client instance to use.

    Returns
    -------
    Project
        The project instance.
    """
    cl = client or get_client()
    if not isinstance(cl, ProjectClient):
        raise HowsoError("Projects are not supported by the active Howso client.")
    project = cl.get_project(str(project_id))
    return Project.from_schema(project, client=cl)


def list_projects(*args, **kwargs) -> list[Project]:
    """
    Query accessible Projects.

    DEPRECATED: use `get_projects` instead.
    """
    warnings.warn(
        "The method `list_projects` is deprecated. Use `query_projects` instead.", DeprecationWarning)
    return query_projects(*args, **kwargs)


def query_projects(
    search_terms: Optional[str] = None,
    *,
    client: Optional[ProjectClient] = None
) -> list[Project]:
    """
    Query accessible Projects.

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
    cl = client or get_client()
    if not isinstance(cl, ProjectClient):
        raise HowsoError("Projects are not supported by the active Howso client.")
    projects = cl.query_projects(search_terms)
    return [Project.from_schema(project, client=cl) for project in projects]


def switch_project(
    project_id: str | UUID,
    *,
    client: Optional[ProjectClient] = None
) -> Project:
    """
    Set the active project.

    Parameters
    ----------
    project_id : str or UUID
        The id of the project.
    client : ProjectClient, optional
        The Howso client instance to use.

    Returns
    -------
    Project
        The newly active project instance.
    """
    cl = client or get_client()
    if not isinstance(cl, ProjectClient):
        raise HowsoError("Projects are not supported by the active Howso client.")
    project = cl.switch_project(str(project_id))
    return Project.from_schema(project, client=cl)
