from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import typing as t
from uuid import UUID

from dateutil.parser import parse as dt_parse

from .base import BaseSchema

__all__ = [
    "Project",
    "ProjectDict",
]


class ProjectDict(t.TypedDict):
    """A dict representation of a Project object."""

    id: str
    name: str
    is_private: bool
    is_default: bool
    created_by: Mapping | None
    created_date: datetime | None
    modified_date: datetime | None
    permissions: list[str]


class Project(BaseSchema[ProjectDict]):
    """
    Base representation of a Project object.

    Parameters
    ----------
    id : str or UUID
        The unique identifier of the Project.
    name : str
        The name of the Project.
    created_by : dict, optional
        The user account that created this Project.
    created_date : str or datetime, optional
        The timestamp of when the Project was originally created.
    is_default : bool, optional
        If the Project is the current user's default Project.
    is_private : bool, optional
        Designates if the Project is not publicly visible.
    modified_date : str or datetime, optional
        The timestamp of when the Project was last modified.
    permissions : list of str, optional
        Permission types the user has in this Project.
    """

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'is_private': 'is_private',
        'is_default': 'is_default',
        'created_by': 'created_by',
        'created_date': 'created_date',
        'modified_date': 'modified_date',
        'permissions': 'permissions'
    }

    def __init__(
        self,
        id: str | UUID,
        name: str,
        *,
        created_by: t.Optional[Mapping] = None,
        created_date: t.Optional[str | datetime] = None,
        is_default: bool = False,
        is_private: bool = True,
        modified_date: t.Optional[str | datetime] = None,
        permissions: t.Optional[list[str]] = None,
    ):
        """Initialize the Project instance."""
        if id is None:
            raise ValueError("An `id` is required to create a Project object.")

        self._id = str(id)
        self._created_by = created_by
        self._is_default = is_default
        self._is_private = is_private
        self._permissions = permissions
        self._created_date = created_date
        self._modified_date = modified_date

        self.name = name

    @property
    def id(self) -> str:
        """
        The unique identifier of this Project.

        Returns
        -------
        str
            The Project ID.
        """
        return self._id

    @property
    def name(self) -> str:
        """
        The name of the Project.

        Returns
        -------
        str
            The Project name.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        Set the name of the Project.

        Parameters
        ----------
        name : str
            The name of the Project.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`.")
        if len(name) > 128:
            raise ValueError('Invalid value for `name`, length must be less than or equal to `128`.')
        self._name = name

    @property
    def is_private(self) -> bool:
        """
        Designates if the Project is not publicly visible.

        Returns
        -------
        bool
            True, when the Project not public.
        """
        return self._is_private

    @property
    def is_default(self) -> bool:
        """
        If this Project is the current user's default Project.

        Returns
        -------
        bool
            True, when the Project is the user's default.
        """
        return self._is_default

    @property
    def created_by(self) -> Mapping | None:
        """
        The user account that created this Project.

        Returns
        -------
        Mapping
            The user account information.
        """
        return self._created_by

    @property
    def created_date(self) -> datetime | None:
        """
        The timestamp of when the Project was originally created.

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
        The timestamp of when the Project was last modified.

        Returns
        -------
        datetime
            The modified timestamp.
        """
        if isinstance(self._modified_date, str):
            # Lazily resolve str datetimes
            self._modified_date = dt_parse(self._modified_date)
        return self._modified_date

    @property
    def permissions(self) -> list[str]:
        """
        Permission types the user has in this Project.

        Returns
        -------
        list of str
            The list of permission types.
        """
        if not self._permissions:
            return []
        else:
            return list(self._permissions)
