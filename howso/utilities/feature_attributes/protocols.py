from __future__ import annotations

from abc import abstractmethod
from collections.abc import Generator, Iterable
import typing as t

import pandas as pd


class TableNameProtocol(t.Protocol):
    """Protocol for a database table name object."""

    schema: str
    table: str


class SQLTableProtocol(t.Protocol):
    """Protocol for a SQL table object."""

    c: dict
    columns: dict
    name: str
    schema: str


class SessionProtocol(t.Protocol):
    """Protocol for a sqlalchemy Session object."""

    @abstractmethod
    def commit(self):
        """Flush pending changes and commit the current transaction."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close out the transactional resources and ORM objects used by this Session."""
        raise NotImplementedError

    @abstractmethod
    def rollback(self):
        """Rollback the current transaction in progress."""
        raise NotImplementedError


class AbstractDataProtocol(t.Protocol):
    """Protocol for an abstract data file object."""

    @property
    def foreign_keys(self) -> str | Iterable[str] | None:
        """Return the foreign key(s) of the table."""
        raise NotImplementedError

    @property
    @abstractmethod
    def headers(self) -> list[str]:
        """Return a list of the column names of the table."""

    @property
    def name(self) -> str:
        """Return a meaningful name for this data."""
        raise NotImplementedError

    @property
    def primary_keys(self) -> str | Iterable[str] | None:
        """Return the primary key(s) of the table."""
        raise NotImplementedError

    @abstractmethod
    def get_row_count(self):
        """Get the number of rows in a file."""
        raise NotImplementedError

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """Get the file as a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def get_group_map(self, column_name: t.Hashable) -> dict[t.Any, int]:
        """Get the group map."""
        raise NotImplementedError

    @abstractmethod
    def get_n_random_rows(self, samples: int, seed: t.Optional[int]) -> pd.DataFrame:
        """Get a specified number of random rows."""
        raise NotImplementedError

    @abstractmethod
    def write_chunk(self, chunk: pd.DataFrame, *,
                    if_exists: str = "append"
                    ) -> None:
        """Write a chunk."""
        raise NotImplementedError

    @abstractmethod
    def yield_chunk(self, chunk_size: int = 5000, *,
                    max_chunks: t.Optional[int] = None,
                    skip_chunks: t.Optional[int] = None,
                    ) -> Generator[pd.DataFrame, None, None]:
        """Provide a chunk generator."""
        raise NotImplementedError

    @abstractmethod
    def yield_grouped_chunk(self, column_name: t.Hashable,
                            groups: Iterable[Iterable[t.Any]]
                            ) -> Generator[pd.DataFrame, None, None]:
        """Provide a grouped chunk generator."""
        raise NotImplementedError

    @abstractmethod
    def map_keys(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Map keys to a chunk."""
        raise NotImplementedError

    @abstractmethod
    def get_unique_count(self, column_name: t.Hashable) -> int:
        """Get the number of unique values in the provided column."""

    @abstractmethod
    def is_unique(self, column_name: t.Hashable) -> bool:
        """Return whether the given column contains only unique values."""

    @abstractmethod
    def contains_nulls(self, column_name: t.Hashable) -> bool:
        """Return whether the given column contains any null values."""


@t.runtime_checkable
class IFACompatibleADCProtocol(t.Protocol):
    """
    Protocol for an abstract data file object with extended functionality.

    Includes functions that make it compatible with `infer_feature_attributes`.
    """

    @abstractmethod
    def get_decimal_places(self, column_name: t.Hashable) -> int:
        """Get the number of decimal places for values in the given column, if applicable."""

    @abstractmethod
    def get_random_value(self, column_name: t.Hashable, no_nulls: bool = False):
        """
        Return a random sample from the given DataFrame column.

        The return type is determined by the column type.

        if `no_nulls` is set, select a random value from the set of non-null
        values, if any. If there are no such non-nulls, this will return None.
        """

    @abstractmethod
    def get_min_max_values(self, column_name: t.Hashable) -> tuple[t.Any, t.Any]:
        """Get the smallest and largest values in the given column."""

    @abstractmethod
    def get_num_cases(self, column_name: t.Hashable) -> int:
        """Return the number of non-null cases in the given column."""

    @abstractmethod
    def get_mode(self, column_name: t.Hashable) -> list[tuple[t.Any, int]]:
        """
        Get the most common value in the given feature/column.

        If multiple values have the same mode all of them will be returned, as
        long as the count is a value greater than 1.
        """

    @abstractmethod
    def get_column_dtype(self, column_name: str) -> str:
        """Get the dtype of the given column."""

    @abstractmethod
    def get_first_non_null(self, column_name: str) -> str:
        """Get the first non-null value in the given column."""

    @abstractmethod
    def get_null_count(self, column_name) -> int:
        """Get the number of nulls in the given column."""


class RelationshipProtocol(t.Protocol):
    """Protocol for an object representing a relationship in a database."""

    source: TableNameProtocol
    source_columns: tuple[str]
    destination: TableNameProtocol
    destination_columns: tuple[str]


class ComponentProtocol(t.Protocol):
    """Protocol for an object representing an independent collection of DataFrame."""

    datastore: t.Any
    graph: t.Any


@t.runtime_checkable
class DatastoreProtocol(t.Protocol):
    """Protocol for a datastore object."""

    @abstractmethod
    def items(self) -> Generator[tuple[TableNameProtocol, AbstractDataProtocol], None, None]:
        """Get items in the datastore."""
        raise NotImplementedError

    @abstractmethod
    def is_degenerate_relationship(self, relationship: RelationshipProtocol,
                                   robust: bool = False) -> bool:
        """Get whether a relationship is degenerate."""
        raise NotImplementedError

    @abstractmethod
    def degenerate_relationships(self, *, robust: bool = False
                                 ) -> Generator[RelationshipProtocol, None, None]:
        """Get a generator of degenerate relationships in this datastore."""
        raise NotImplementedError

    @abstractmethod
    def components(self) -> Generator[ComponentProtocol, None, None]:
        """Get a generator of the components in this datastore."""
        raise NotImplementedError

    @abstractmethod
    def pre_synth_check(self, related_datastore: t.Any, **kwargs) -> bool:
        """Attempt a pre-synth check."""
        raise NotImplementedError

    @abstractmethod
    def reflect(self, source: t.Any, drop_existing: bool = False) -> None:
        """Do a reflection."""
        raise NotImplementedError

    @abstractmethod
    def has_feature_support(self, feature_key: str) -> bool:
        """Return whether the given feature is supported."""
        raise NotImplementedError

    @abstractmethod
    def get_row_count(self, table_name: TableNameProtocol) -> int | None:
        """Get the number of rows in the specified table."""
        raise NotImplementedError

    @abstractmethod
    def get_data(self, table_name) -> AbstractDataProtocol:
        """Get the data in a specified table."""
        raise NotImplementedError

    @abstractmethod
    def set_data(self, table_name, data: AbstractDataProtocol):
        """Set the data in a specified table."""
        raise NotImplementedError

    @abstractmethod
    def get_values(self,
                   table_name: TableNameProtocol,
                   primary_key_columns: list[str] | str,
                   primary_key_values: list[list[t.Any]] | list[t.Any],
                   column_name: str) -> list[t.Any]:
        """Get the column values in a specified table."""
        raise NotImplementedError

    @abstractmethod
    def replace_values(self,
                       table_name: TableNameProtocol,
                       primary_key_columns: list[str] | str,
                       primary_key_values: list[t.Any] | t.Any,
                       column_name: str,
                       replace_values: list[t.Any],
                       return_old: bool = False
                       ) -> list[t.Any] | None:
        """Replace the column values in a specified table."""
        raise NotImplementedError


@t.runtime_checkable
class RelationalDatastoreProtocol(DatastoreProtocol, t.Protocol):
    """Protocol for a relational datastore object."""

    graph: t.Any


@t.runtime_checkable
class SQLRelationalDatastoreProtocol(DatastoreProtocol, t.Protocol):
    """Protocol for a SQL relational datastore object."""

    engine: int
    graph: t.Any
