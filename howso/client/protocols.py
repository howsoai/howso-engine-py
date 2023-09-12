import typing as t

if t.TYPE_CHECKING:
    from howso.openapi.models import Project

__all__ = [
    "ProjectClient",
]


@t.runtime_checkable
class ProjectClient(t.Protocol):
    """Protocol to define a Howso client that supports projects."""

    active_project: t.Optional["Project"]

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

    def get_projects(self, search_terms: t.Optional[str]) -> t.List["Project"]:
        """Search and list projects."""
        ...
