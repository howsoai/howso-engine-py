from __future__ import annotations

from collections.abc import Mapping
import typing as t
from uuid import UUID

from typing_extensions import TypeAlias

from .base import BaseSchema

__all__ = [
    "Trainee",
    "TraineeDict",
    "TraineePersistence",
]


TraineePersistence: TypeAlias = t.Literal['allow', 'always', 'never']


class TraineeDict(t.TypedDict):
    """A dict representation of a Trainee object."""

    id: str
    name: str | None
    features: Mapping[str, Mapping]
    metadata: Mapping | None
    persistence: TraineePersistence
    project_id: str | None


class Trainee(BaseSchema[TraineeDict]):
    """
    Base representation of a Trainee object.

    Parameters
    ----------
    id : str or UUID
        The unique identifier of the Trainee.
    name : str or UUID, optional
        A name given to the Trainee.
    features : Mapping, optional
        The feature attributes of the Trainee.
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
        'features': 'features',
        'persistence': 'persistence',
        'project_id': 'project_id',
        'metadata': 'metadata',
    }
    nullable_attributes = {'name'}

    def __init__(
        self,
        id: str | UUID,
        name: t.Optional[str] = None,
        features: t.Optional[Mapping[str, Mapping]] = None,
        *,
        metadata: t.Optional[Mapping] = None,
        persistence: TraineePersistence = 'allow',
        project_id: t.Optional[str | UUID] = None,
    ):
        """Initialize the Trainee instance."""
        if id is None:
            raise ValueError("An `id` is required to create a Trainee object.")

        self._id = str(id)
        self._project_id = str(project_id) if project_id else None
        self._metadata = metadata
        self._features = dict() if features is None else features

        self.name = name
        self.persistence = persistence

    @property
    def id(self) -> str:
        """Get the id of this Trainee."""
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
    def features(self) -> Mapping[str, Mapping]:
        """
        The feature attributes of the Trainee.

        Returns
        -------
        Mapping
            The feature attributes of the Trainee.
        """
        return self._features

    @features.setter
    def features(self, features: Mapping[str, Mapping]) -> None:
        """
        Set the feature attributes of the Trainee.

        Parameters
        ----------
        features : Mapping or None
            The new feature attributes.
        """
        self._features = features

    @property
    def persistence(self) -> TraineePersistence:
        """
        The persistence state of the Trainee.

        Returns
        -------
        str
            The Trainee's persistence value.
        """
        return self._persistence

    @persistence.setter
    def persistence(self, persistence: TraineePersistence):
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
        self._persistence: TraineePersistence = persistence

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
