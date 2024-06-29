from collections.abc import Mapping
from copy import deepcopy
from typing import Optional, Union

from howso.client import AbstractHowsoClient
from howso.client.exceptions import HowsoError
from howso.client.protocols import ProjectClient
from howso.client.schemas import Project as BaseProject, Session as BaseSession
from howso.engine.client import get_client

__all__ = [
    'get_active_session',
    'get_session',
    'list_sessions',
    'Session',
]


class Session(BaseSession):
    """
    A Howso Session.

    Parameters
    ----------
    name : str, optional
        The name of the session.
    metadata : dict, optional
        Any key-value pair to store custom metadata for the session.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        id: Optional[str] = None,
        metadata: Optional[dict] = None,
        client: Optional[AbstractHowsoClient] = None,
    ) -> None:
        """Implement the constructor."""
        self._created: bool = False
        self._updating: bool = False
        self.client = client or get_client()

        # Set the session properties
        self._metadata = metadata
        self._name = name
        self._user = None
        self._created_date = None
        self._modified_date = None
        self._id = id

        # Create the session at the API
        self._create()

    @property
    def metadata(self) -> Union[dict, None]:
        """
        The Session metadata.

        .. WARNING::
            This returns a deep `copy` of the metadata. To update the
            metadata of the session, use the method :func:`set_metadata`.

        Returns
        -------
        dict
            The metadata of the Session.
        """
        return deepcopy(self._metadata)

    @property
    def client(self) -> AbstractHowsoClient:
        """
        The client instance used by the session.

        Returns
        -------
        AbstractHowsoClient
            The client instance.
        """
        return self._client

    @client.setter
    def client(self, client: AbstractHowsoClient) -> None:
        """
        Set the client instance used by the session.

        Parameters
        ----------
        client : AbstractHowsoClient
            The client instance. Must be a subclass of AbstractHowsoClient.

        Returns
        -------
        None
        """
        if not isinstance(client, AbstractHowsoClient):
            raise HowsoError("`client` must be a subclass of "
                             "AbstractHowsoClient")
        self._client = client

    def set_metadata(self, metadata: Optional[dict]) -> None:
        """
        Update the session metadata.

        Parameters
        ----------
        metadata : dict or None
            Any key-value pair to store as custom metadata for the session.
            Providing `None` will remove the current metadata.

        Returns
        -------
        None
        """
        self._metadata = metadata
        self._update()

    def _update_attributes(self, session: BaseSession) -> None:
        """
        Update the protected attributes of the session.

        Parameters
        ----------
        session : BaseSession
            The base session instance.

        Returns
        -------
        None
        """
        for key in self.attribute_map:
            # Update the protected attributes directly since the values
            # are provided from the client and to prevent triggering an
            # API update call.
            setattr(self, f'_{key}', getattr(session, key))

    def _update(self) -> None:
        """
        Update the session at the API.

        Returns
        -------
        None
        """
        if (
            getattr(self, 'id', None)
            and getattr(self, '_created', False)
            and not getattr(self, '_updating', False)
        ):
            # Only update for sessions that have been created
            self._updating = True
            updated_session = self.client.update_session(
                session_id=self.id,
                metadata=self.metadata
            )
            self._update_attributes(updated_session)
            self._updating = False

    def _create(self) -> None:
        """
        Create the session at the API.

        Returns
        -------
        None
        """
        if not self.id:
            session = self.client.begin_session(name=self.name,
                                                metadata=self.metadata)
            self._update_attributes(session)
        self._created = True

    @classmethod
    def from_dict(
        cls,
        schema: Mapping,
        *,
        client: Optional[AbstractHowsoClient] = None,
    ) -> "Session":
        """
        Create Session from dict.

        Parameters
        ----------
        schema : dict
            The session parameters.
        client : AbstractHowsoClient, optional
            The Howso client instance to use.

        Returns
        -------
        Session
            The session instance.
        """
        if not isinstance(schema, Mapping):
            raise ValueError('`schema` parameter is not a Mapping')
        parameters: dict = {k: schema[k] for k in cls.attribute_map if k in schema}
        parameters['client'] = client or schema.get('client')
        return cls(**parameters)


def get_active_session(
    *, client: Optional[AbstractHowsoClient] = None
) -> Session:
    """
    Get the active session.

    Parameters
    ----------
    client : AbstractHowsoClient, optional
        The Howso client instance to use.

    Returns
    -------
    Session
        The session instance.
    """
    client = client or get_client()
    if not client.active_session:
        raise ValueError("No session is active.")
    return Session.from_dict(client.active_session.to_dict(), client=client)


def get_session(
    session_id: str,
    *,
    client: Optional[AbstractHowsoClient] = None
) -> Session:
    """
    Get an existing Session.

    Parameters
    ----------
    session_id : str
        The id of the session.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.

    Returns
    -------
    Session
        The session instance.
    """
    client = client or get_client()
    session = client.get_session(str(session_id))
    return Session.from_dict(session.to_dict(), client=client)


def list_sessions(
    search_terms: Optional[str] = None,
    *,
    client: Optional[AbstractHowsoClient] = None,
    project: Optional[Union[str, BaseProject]] = None
) -> list[Session]:
    """
    Get listing of Sessions.

    Parameters
    ----------
    search_terms : str
        Terms to filter results by.
    client : AbstractHowsoClient, optional
        The Howso client instance to use.
    project : str or Project, optional
        The instance or id of a project to filter by. Ignored if client
        does not support projects.

    Returns
    -------
    list of Session
        The list of session instances.
    """
    client = client or get_client()

    params = {'search_terms': search_terms}

    # Only pass project_id for platform clients
    if project is not None and isinstance(client, ProjectClient):
        if isinstance(project, BaseProject):
            params["project_id"] = project.id
        else:
            params["project_id"] = project

    sessions = client.get_sessions(**params)
    return [Session.from_dict(s.to_dict(), client=client) for s in sessions]
