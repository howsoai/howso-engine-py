from __future__ import annotations

from collections.abc import Mapping
import typing as t
from uuid import UUID

from typing_extensions import NotRequired, ReadOnly, TypedDict

from .base import BaseSchema
from ..typing import LibraryType, Persistence

__all__ = [
    "Trainee",
    "TraineeDict",
]


class TraineeDict(TypedDict):
    """A dict representation of a Trainee object."""

    id: str
    name: str | None
    metadata: Mapping | None
    persistence: Persistence
    project_id: str | None
    features: NotRequired[Mapping[str, Mapping] | None]


class TraineeVersion(TypedDict, total=False):
    """Trainee version information."""

    trainee: str | None
    """The Amalgam version the Trainee's is at."""

    amalgam: str | None
    """The Amalgam library version."""


class ResourceLimit(TypedDict):
    """System resource (CPU/Memory) limit."""

    minimum: ReadOnly[NotRequired[int | None]]
    """Minimum allocated."""

    maximum: ReadOnly[NotRequired[int | None]]
    """Maximum allocated."""


class TraineeScalingResources(TypedDict, total=False):
    """Resources allocated to a Trainee."""

    cpu: ReadOnly[NotRequired[ResourceLimit | None]]
    """Trainee CPU limits in millicores."""

    memory: ReadOnly[NotRequired[ResourceLimit | None]]
    """Trainee memory limits in mebibytes."""


class TraineeScaling(TypedDict, total=False):
    """Trainee runtime scaling options."""

    resources: ReadOnly[NotRequired[TraineeScalingResources | None]]
    """The CPU and memory resources allocated to the Trainee."""


class TraineeRuntimeOptions(TypedDict, total=False):
    """Runtime options used when creating a Trainee."""

    library_type: ReadOnly[NotRequired[LibraryType | None]]
    """The Amalgam library type used by the Trainee."""

    scaling: ReadOnly[NotRequired[TraineeScaling | None]]
    """The runtime scaling options used by the Trainee."""


class TraineeRuntime(TypedDict):
    """Trainee runtime details."""

    library_type: ReadOnly[LibraryType]
    """The Amalgam library type used by the Trainee."""

    tracing_enabled: ReadOnly[bool]
    """If debug tracing is enabled for the Trainee."""

    versions: ReadOnly[TraineeVersion]
    """The Trainee runtime versions."""

    scaling: ReadOnly[NotRequired[TraineeScaling | None]]
    """The runtime scaling options used by the Trainee."""


class Trainee(BaseSchema[TraineeDict]):
    """
    Base representation of a Trainee object.

    Parameters
    ----------
    id : str or UUID
        The unique identifier of the Trainee.
    name : str or UUID, optional
        A name given to the Trainee.
    metadata : Mapping, optional
        Any key-value pair to store as custom metadata for the Trainee.
    persistence : {'allow', 'always', 'never'}, optional
        The requested persistence state of the Trainee.
    project_id : str or UUID, optional
        The id of the Project the Trainee is associated with.
    """

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'persistence': 'persistence',
        'project_id': 'project_id',
        'metadata': 'metadata',
    }
    nullable_attributes = {'name'}

    def __init__(
        self,
        id: str | UUID,
        name: t.Optional[str] = None,
        *,
        metadata: t.Optional[Mapping] = None,
        persistence: Persistence = 'allow',
        project_id: t.Optional[str | UUID] = None,
    ):
        """Initialize the Trainee instance."""
        if id is None:
            raise ValueError("An `id` is required to create a Trainee object.")

        self._id = str(id)
        self._project_id = str(project_id) if project_id else None
        self._metadata = metadata

        self.name = name
        self.persistence = persistence

    @property
    def id(self) -> str:
        """
        The unique identifier of the Trainee.

        Returns
        -------
        str
            The Trainee ID.
        """
        return self._id

    @property
    def name(self) -> str | None:
        """
        The name of the Trainee.

        Returns
        -------
        str or None
            The Trainee's name.
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        """
        Set the name of the Trainee.

        Parameters
        ----------
        name : str or None
            The new name.
        """
        if name is not None and len(name) > 128:
            raise ValueError('Invalid value for `name`, length must be less than or equal to `128`.')
        self._name = name

    @property
    def persistence(self) -> Persistence:
        """
        The persistence state of the Trainee.

        Returns
        -------
        str
            The Trainee's persistence value.
        """
        return self._persistence

    @persistence.setter
    def persistence(self, persistence: Persistence):
        """
        Set the persistence state of the Trainee.

        Parameters
        ----------
        persistence : {"allow", "always", "never"}
            The new persistence value.
        """
        allowed_values = {'allow', 'always', 'never'}
        if persistence not in allowed_values:
            raise ValueError(
                f'Invalid value for `persistence` ({persistence}), must be one of: {allowed_values}')
        self._persistence: Persistence = persistence

    @property
    def metadata(self) -> Mapping | None:
        """
        The Trainee metadata.

        Returns
        -------
        Mapping
            The metadata of the Trainee.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Mapping | None) -> None:
        """
        Set the Trainee metadata.

        Parameters
        ----------
        metadata : Mapping, optional
            The new metadata for the Trainee.
        """
        self._metadata = metadata

    @property
    def project_id(self) -> str | None:
        """
        The id of the Project this Trainee is associated with.

        Returns
        -------
        str or None
            The Project's unique identifier.
        """
        return self._project_id
