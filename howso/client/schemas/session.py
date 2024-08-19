from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import typing as t
from uuid import UUID

from dateutil.parser import parse as dt_parse

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
    id : str or UUID
        The unique identifier of the Session.
    name : str, optional
        A name given to the Session.
    user : Mapping, optional
        The details of the user who created the Session.
    metadata : Mapping, optional
        Any key-value pair to store as custom metadata for the Session.
    created_date : str or datetime, optional
        The datetime of when the Session was created. When specified as a string, the value should be an
        ISO 8601 timestamp.
    modified_date : str or datetime, optional
        The datetime of when the Session details were last modified. When specified as a string, the value should
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
    nullable_attributes = {'name'}

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
        if name is not None and len(name) > 128:
            raise ValueError('Invalid value for `name`, length must be less than or equal to `128`.')

        self._id = str(id)
        self._name = name
        self._user = user
        self._metadata = metadata
        self._created_date = created_date
        self._modified_date = modified_date

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
            The new metadata for the Session.
        """
        self._metadata = metadata

    @property
    def created_date(self) -> datetime | None:
        """
        The timestamp of when the Session was originally created.

        Returns
        -------
        datetime
            The creation timestamp.
        """
        if isinstance(self._created_date, str):
            # Lazily resolve str datetimes
            self._created_date = dt_parse(self._created_date)
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
        if isinstance(self._modified_date, str):
            # Lazily resolve str datetimes
            self._modified_date = dt_parse(self._modified_date)
        return self._modified_date
