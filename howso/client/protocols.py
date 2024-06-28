from typing import Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from howso.direct.core import HowsoCore

__all__ = [
    "LocalSaveableProtocol",
    "ProjectClient",
]


@runtime_checkable
class LocalSaveableProtocol(Protocol):
    """Protocol to define a Howso client that has direct disk read/write access."""

    @property
    def howso(self) -> "HowsoCore":
        """Howso Core API."""
        ...

    def _get_trainee_from_core(self, trainee_id: str) -> dict:
        """Retrieve the core representation of a Trainee object."""
        ...


@runtime_checkable
class ProjectClient(Protocol):
    """Protocol to define a Howso client that supports projects."""

    active_project: Optional[dict]

    def switch_project(self, project_id: str) -> dict:
        """Switch active project."""
        ...

    def create_project(self, name: str) -> dict:
        """Create new project."""
        ...

    def update_project(self, project_id: str) -> dict:
        """Update existing project."""
        ...

    def delete_project(self, project_id: str) -> None:
        """Delete a project."""
        ...

    def get_project(self, project_id: str) -> dict:
        """Get existing project."""
        ...

    def get_projects(self, search_terms: Optional[str]) -> list[dict]:
        """Search and list projects."""
        ...
