from typing import (
    Any,
    Dict,
    List,
    Protocol,
    Optional,
    runtime_checkable,
    TYPE_CHECKING
)

from howso.direct.core import HowsoCore
from howso.client.cache import TraineeCache

if TYPE_CHECKING:
    from howso.openapi.models import (
        Metrics,
        Trainee,
        TraineeInformation,
    )
    from howso.openapi.models import Project

__all__ = [
    "ProjectClient",
]


@runtime_checkable
class BaseClientProtocol(Protocol):
    @property
    def trainee_cache(self) -> "TraineeCache":
        ...

    def analyze(self, *args, **kwargs):
        ...

    def get_trainee_metrics(self, *args, **kwargs) -> "Metrics":
        ...

    def get_trainee_information(self, *args, **kwargs) -> "TraineeInformation":
        ...

    def remove_feature(self, *args, **kwargs):
        ...

    def edit_cases(self, *args, **kwargs) -> int:
        ...

    def copy_trainee(self, *args, **kwargs):
        ...

    def impute(self, *args, **kwargs):
        ...

    def remove_cases(self, *args, **kwargs) -> int:
        ...

    def delete_trainee_session(self, *args, **kwargs):
        ...

    def get_trainee_sessions(self, *args, **kwargs) -> List[Dict[str, str]]:
        ...

    def add_feature(self, *args, **kwargs):
        ...

    def get_num_training_cases(self, *args, **kwargs) -> int:
        ...

    def append_to_series_store(self, *args, **kwargs):
        ...

    def remove_series_store(self, *args, **kwargs):
        ...

    def set_substitute_feature_values(self, *args, **kwargs):
        ...

    def get_substitute_feature_values(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        ...

    def react_into_features(self, *args, **kwargs):
        ...

    def react_into_trainee(self, *args, **kwargs):
        ...

    def set_feature_attributes(self, *args, **kwargs):
        ...

    def set_params(self, *args, **kwargs):
        ...

    def get_params(self, *args, **kwargs) -> Dict[str, Any]:
        ...

    def get_pairwise_distances(self, *args, **kwargs) -> List[float]:
        ...

    def evaluate(self, *args, **kwargs) -> Dict:
        ...

    def delete_trainee(self, *args, **kwargs):
        ...

    def acquire_trainee_resources(self, *args, **kwargs):
        ...

    def release_trainee_resources(self, *args, **kwargs):
        ...

    def train(self, *args, **kwargs):
        ...

    def auto_analyze(self, *args, **kwargs):
        ...

    def set_auto_analyze_params(self, *args, **kwargs):
        ...

    def update_trainee(self, *args, **kwargs):
        ...

    def create_trainee(self, *args, **kwargs) -> "Trainee":
        ...


@runtime_checkable
class LocalSavableClient(Protocol):

    @property
    def howso(self) -> "HowsoCore":
        ...

    @property
    def trainee_cache(self, *args, **kwargs):
        ...

    def _get_trainee_from_core(self, *args, **kwargs):
        ...


@runtime_checkable
class PlatformCapableClient(Protocol):

    def __api_client(self):
        ...

    def acquire_trainee_resources(self, *args, **kwargs):
        ...

    def release_trainee_resources(self, *args, **kwargs):
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
