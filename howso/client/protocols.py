from typing import (
    List,
    Protocol,
    Optional,
    runtime_checkable,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from howso.direct.core import HowsoCore
    from howso.openapi.models import Trainee
    from howso.openapi.models import Project

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

    def _get_trainee_from_core(self, trainee_id: str) -> "Trainee":
        """Retrieve the core representation of a Trainee object."""
        ...


@runtime_checkable
class ProjectClient(Protocol):
    """Protocol to define a Howso client that supports projects."""

    active_project: Optional["Project"]

    def switch_project(self, project_id: str) -> "Project":
        """Switch active project."""
        ...

    def create_project(self, name: str) -> "Project":
        """Create new project."""
        ...

    def update_project(self, project_id: str) -> "Project":
        """Update existing project."""
        ...

    def delete_project(self, project_id: str) -> None:
        """Delete a project."""
        ...

    def get_project(self, project_id: str) -> "Project":
        """Get existing project."""
        ...

    def get_projects(self, search_terms: Optional[str]) -> List["Project"]:
        """Search and list projects."""
        ...
