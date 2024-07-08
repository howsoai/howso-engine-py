from typing import Any, Literal

from typing_extensions import Sequence, TypeAlias, TypedDict


class Cases(TypedDict):
    """Representation of a table of cases."""

    cases: list[list[Any]]
    """Matrix of row and column values."""

    features: list[str]
    """The feature column names."""


CaseIndices: TypeAlias = Sequence[tuple[str, int]]
"""Type alias for ``case_indices`` parameters."""

Precision: TypeAlias = Literal["exact", "similar"]
"""Type alias for the valid values of ``precision`` parameters."""
