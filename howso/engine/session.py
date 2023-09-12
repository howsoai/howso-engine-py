from copy import deepcopy
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from howso.client import AbstractHowsoClient
from howso.client.exceptions import HowsoError
from howso.client.protocols import ProjectClient
from howso.engine.client import get_client
from howso.openapi.models import (
    Project as BaseProject,
    Session as BaseSession
)

if TYPE_CHECKING:
    from datetime import datetime
    from howso.openapi.models import AccountIdentity

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
    def id(self) -> str:
        """
        The unique identifier of the session.

        Returns
        -------
        str
            The session ID.
        """
        return self._id

    @property
    def name(self) -> str:
        """
        The name of the session.

        Returns
        -------
        str
            The session name.
        """
        return self._name

    @property
    def user(self) -> Optional["AccountIdentity"]:
        """
        The user account that the session belongs to.

        Returns
        -------
        AccountIdentity
            The user account information.
        """
        return self._user

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """
        The session metadata.

        .. WARNING::
            This returns a deep `copy` of the metadata. To update the
            metadata of the session, use the method :func:`set_metadata`.

        Returns
        -------
        dict
            The metadata of the session.
        """
        return deepcopy(self._metadata)

    @property
    def created_date(self) -> Optional["datetime"]:
        """
        The timestamp of when the session was originally created.

        Returns
        -------
        datetime
            The creation timestamp.
        """
        return self._created_date

    @property
    def modified_date(self) -> Optional["datetime"]:
        """
        The timestamp of when the session was last modified.

        Returns
        -------
        datetime
            The modified timestamp.
        """
        return self._modified_date

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

    def set_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
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
        for key in self.attribute_map.keys():
            # Update the protected attributes directly since the values
            # have already been validated by the "BaseSession" instance
            # and to prevent triggering an API update call
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
    def from_openapi(
        cls, session: BaseSession, *,
        client: Optional[AbstractHowsoClient] = None
    ) -> "Session":
        """
        Create Session from base class.

        Parameters
        ----------
        session : BaseSession
            The base session instance.
        client : AbstractHowsoClient, optional
            The Howso client instance to use.

        Returns
        -------
        Session
            The session instance.
        """
        session_dict = session.to_dict()
        session_dict['client'] = client
        return cls.from_dict(session_dict)

    @classmethod
    def from_dict(cls, session_dict: dict) -> "Session":
        """
        Create Session from dict.

        Parameters
        ----------
        session_dict : Dict
            The session parameters.

        Returns
        -------
        Session
            The session instance.
        """
        if not isinstance(session_dict, dict):
            raise ValueError('`session_dict` parameter is not a dict')
        parameters = {
            'id': session_dict.get('id'),
            'client': session_dict.get('client')
        }
        instance = cls(**parameters)
        for key in cls.attribute_map.keys():
            if key in session_dict:
                setattr(instance, f'_{key}', session_dict[key])
        return instance


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
    return client.active_session


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
    return Session.from_openapi(session, client=client)


def list_sessions(
    search_terms: Optional[str] = None,
    *,
    client: Optional[AbstractHowsoClient] = None,
    project: Optional[Union[str, BaseProject]] = None
) -> List[Session]:
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
    return [Session.from_openapi(s, client=client) for s in sessions]
