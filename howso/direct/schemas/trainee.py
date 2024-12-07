from __future__ import annotations

from collections.abc import Mapping
import typing as t
from uuid import UUID

from typing_extensions import NotRequired, ReadOnly

from ...client.schemas.trainee import Trainee, TraineeDict, TraineeRuntimeOptions
from ...client.typing import Persistence


class DirectTraineeDict(TraineeDict):
    """
    Direct-client-specific trainee state.

    .. versionadded:: 33.1

    """

    transactional: bool


class DirectTrainee(Trainee):
    """
    Direct-client-specific internal representation of a trainee.

    .. versionadded:: 33.1

    """

    attribute_map = dict(Trainee.attribute_map, transactional='transactional')

    def __init__(
        self,
        id: str | UUID,
        name: t.Optional[str] = None,
        *,
        metadata: t.Optional[Mapping] = None,
        persistence: Persistence = 'allow',
        project_id: t.Optional[str | UUID] = None,
        transactional: bool = False
    ):
        """Initialize the Trainee instance."""
        super().__init__(id, name, metadata=metadata, persistence=persistence, project_id=project_id)
        self._transactional = transactional

    @property
    def transactional(self) -> bool:
        """
        Whether this trainee is in transactional mode.

        Returns
        -------
        bool
            true if this trainee is running in transactional mode

        """
        return self._transactional


class TraineeDirectRuntimeOptions(TraineeRuntimeOptions):
    """
    Runtime options specific to the direct client.

    .. versionadded:: 33.1

    """

    transactional: ReadOnly[NotRequired[bool | None]]
    """Use transactional mode when `persistence='always'."""
