from __future__ import annotations

from pathlib import Path
from typing import (
    Optional,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from amalgam.api import Amalgam
    from howso.client.schemas import Project, Trainee
    from howso.client.typing import PathLike

__all__ = [
    "LocalSaveableProtocol",
    "ProjectClient",
]


@runtime_checkable
class LocalSaveableProtocol(Protocol):
    """Protocol to define a Howso client that has direct disk read/write access."""

    default_persist_path: Path

    @property
    def amlg(self) -> Amalgam:
        """Amalgam API."""
        ...

    def _get_trainee_from_engine(self, trainee_id: str) -> Trainee:
        """Retrieve the engine representation of a Trainee object."""
        ...

    def resolve_trainee_filepath(self, filename: str, *, filepath: Optional[PathLike] = None) -> str:
        """Resolve the path to a persisted Trainee file."""
        ...


@runtime_checkable
class ProjectClient(Protocol):
    """Protocol to define a Howso client that supports projects."""

    active_project: Optional[Project]

    def switch_project(self, project_id: str) -> Project:
        """Switch active project."""
        ...

    def create_project(self, name: str) -> Project:
        """Create new project."""
        ...

    def update_project(self, project_id: str, *, name: Optional[str] = None) -> Project:
        """Update existing project."""
        ...

    def delete_project(self, project_id: str) -> None:
        """Delete a project."""
        ...

    def get_project(self, project_id: str) -> Project:
        """Get existing project."""
        ...

    def query_projects(self, search_terms: Optional[str]) -> list[Project]:
        """Query available projects."""
        ...
