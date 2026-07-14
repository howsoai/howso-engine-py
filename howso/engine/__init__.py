"""The Python API for the Howso Engine Client."""

from howso.client import typing
from howso.engine.client import get_client, use_client
from howso.engine.project import (
    delete_project,
    get_project,
    list_projects,
    Project,
    query_projects,
    switch_project,
)
from howso.engine.session import (
    get_active_session,
    get_session,
    list_sessions,
    query_sessions,
    Session,
)
from howso.engine.trainee import (
    delete_trainee,
    get_trainee,
    list_trainees,
    load_trainee,
    query_trainees,
    Trainee,
)

__all__ = [
    "delete_project",
    "delete_trainee",
    "get_active_session",
    "get_client",
    "get_project",
    "get_session",
    "get_trainee",
    "list_projects",
    "list_sessions",
    "list_trainees",
    "load_trainee",
    "Project",
    "query_projects",
    "query_sessions",
    "query_trainees",
    "Session",
    "switch_project",
    "Trainee",
    "typing",
    "use_client",
]
