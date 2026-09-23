from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .registry import InstallationCheckRegistry


class Status(IntEnum):
    """Status Enum."""

    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    NOTICE = 3
    OK = 4


#: Classes or objects whose truthiness indicates an optional dependency is available.
Requirements: TypeAlias = Iterable[object]


class CheckFunction(Protocol):
    """The call signature that every registered check implements."""

    def __call__(self, *, registry: InstallationCheckRegistry) -> tuple[Status, str]:
        """Run the check and return its status and an optional message."""
        ...


@dataclass
class Check:
    """Store the specification of a single check."""

    name: str
    fn: CheckFunction
    client_required: str | None = None
    other_requirements: Requirements | None = None
