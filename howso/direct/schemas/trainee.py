from __future__ import annotations

from collections.abc import Mapping
import typing as t
from uuid import UUID

from ...client.schemas.trainee import Trainee, TraineeDict
from ...client.typing import Persistence


class DirectTraineeDict(TraineeDict):
    """
    Direct-client-specific trainee state.

    .. versionadded:: 33.1

    """

    file_size: int


class DirectTrainee(Trainee):
    """
    Direct-client-specific internal representation of a trainee.

    .. versionadded:: 33.1

    """

    attribute_map = dict(Trainee.attribute_map, file_size='file_size')

    def __init__(
        self,
        id: str | UUID,
        name: t.Optional[str] = None,
        *,
        metadata: t.Optional[Mapping] = None,
        persistence: Persistence = 'allow',
        project_id: t.Optional[str | UUID] = None,
        file_size: t.Optional[int] = 0
    ):
        """Initialize the Trainee instance."""
        super().__init__(id, name, metadata=metadata, persistence=persistence, project_id=project_id)

        self._file_size = file_size or 0

    @property
    def file_size(self) -> int:
        """The last-known size of the trainee file on disk."""
        return self._file_size

    @file_size.setter
    def file_size(self, size: int | None):
        """
        Set the last-known size of the trainee file on disk.

        If `None`, set the size to 0, for compatibility with cases where the
        file size can't be determined or the file does not exist.

        """
        self._file_size = size or 0
