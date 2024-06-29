from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import typing as t
from uuid import UUID

from .base import BaseSchema

__all__ = [
    "Session",
    "SessionDict"
]


class SessionDict(t.TypedDict):
    """A dict representation of a Session object."""

    id: str
    name: str | None
    user: Mapping | None
    metadata: Mapping | None
    created_date: datetime | None
    modified_date: datetime | None


class Session(BaseSchema[SessionDict]):
    """
    Base representation of a Session object.

    Parameters
    ----------
    id : str or UUID, optional
        The unique identifier of the session.
    name : str, optional
        A name given to the session.
    user : Mapping, optional
        The details of the user who created the session.
    metadata : Mapping, optional
        Arbitrary user metadata to store with the session.
    created_date : str or datetime, optional
        The datetime of when the session was created. When specified as a string, the value should be an
        ISO 8601 timestamp.
    modified_date : str or datetime, optional
        The datetime of when the session details were last modified. When specified as a string, the value should
        be an ISO 8601 timestamp.
    """

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'user': 'user',
        'metadata': 'metadata',
        'created_date': 'created_date',
        'modified_date': 'modified_date',
    }

    def __init__(
        self,
        id: str | UUID,
        name: t.Optional[str] = None,
        *,
        metadata: t.Optional[Mapping] = None,
        user: t.Optional[Mapping] = None,
        created_date: t.Optional[str | datetime] = None,
        modified_date: t.Optional[str | datetime] = None,
    ):
        """Initialize the Session instance."""
        if id is None:
            raise ValueError("An `id` is required to create a Session object.")

        self.name = name
        self.metadata = metadata

        self._user = user
        self._id = str(id)

        if isinstance(created_date, str):
            self._created_date = datetime.fromisoformat(created_date)
        else:
            self._created_date = created_date

        if isinstance(modified_date, str):
            self._modified_date = datetime.fromisoformat(modified_date)
        else:
            self._modified_date = modified_date

    def _touch(self) -> None:
        """Update modified date."""
        self._modified_date = datetime.now(timezone.utc)

    @property
    def id(self) -> str:
        """
        The unique identifier of the Session.

        Returns
        -------
        str
            The Session ID.
        """
        return self._id

    @property
    def name(self) -> str | None:
        """
        The name of the Session.

        Returns
        -------
        str
            The Session name.
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        """
        Set the name of the Session.

        Parameters
        ----------
        name : str
            The name of the Session.
        """
        if name is not None and len(name) > 128:
            raise ValueError('Invalid value for `name`, length must be less than or equal to `128`')
        self._name = name
        self._touch()

    @property
    def user(self) -> Mapping | None:
        """
        The user account that the Session belongs to.

        Returns
        -------
        Mapping
            The user account information.
        """
        return self._user

    @property
    def metadata(self) -> Mapping | None:
        """
        The Session metadata.

        Returns
        -------
        Mapping
            The metadata of the Session.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Mapping | None) -> None:
        """
        Set the Session metadata.

        Parameters
        ----------
        metadata : Mapping, optional
            The new metadata of the Session.
        """
        self._metadata = metadata
        self._touch()

    @property
    def created_date(self) -> datetime | None:
        """
        The timestamp of when the Session was originally created.

        Returns
        -------
        datetime
            The creation timestamp.
        """
        return self._created_date

    @property
    def modified_date(self) -> datetime | None:
        """
        The timestamp of when the Session was last modified.

        Returns
        -------
        datetime
            The modified timestamp.
        """
        return self._modified_date
