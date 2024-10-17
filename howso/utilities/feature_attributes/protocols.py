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

    @abstractmethod
    def get_row_count(self):
        """Get the number of rows in a file."""
        raise NotImplementedError

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """Get the file as a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def get_group_map(self, column_name: str) -> dict[t.Any, int]:
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
    def yield_grouped_chunk(self, column_name: str,
                            groups: Iterable[Iterable[t.Any]]
                            ) -> Generator[pd.DataFrame, None, None]:
        """Provide a grouped chunk generator."""
        raise NotImplementedError

    @abstractmethod
    def map_keys(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Map keys to a chunk."""
        raise NotImplementedError


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
