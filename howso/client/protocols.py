from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING
)

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

    def _get_trainee_from_core(self, trainee_id: str) -> "Dict":
        """Retrieve the core representation of a Trainee object."""
        ...


@runtime_checkable
class ProjectClient(Protocol):
    """Protocol to define a Howso client that supports projects."""

    active_project: Optional["Dict"]

    def switch_project(self, project_id: str) -> "Dict":
        """Switch active project."""
        ...

    def create_project(self, name: str) -> "Dict":
        """Create new project."""
        ...

    def update_project(self, project_id: str) -> "Dict":
        """Update existing project."""
        ...

    def delete_project(self, project_id: str) -> None:
        """Delete a project."""
        ...

    def get_project(self, project_id: str) -> "Dict":
        """Get existing project."""
        ...

    def get_projects(self, search_terms: Optional[str]) -> List["Dict"]:
        """Search and list projects."""
        ...
