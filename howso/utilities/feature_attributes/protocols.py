from __future__ import annotations

from abc import abstractmethod
from collections.abc import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
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


class AbstractDataProtocol(t.Protocol):
    """Protocol for an abstract data class object."""

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
    def primary_keys(self) -> str | list[str] | None:
        """Return the primary key(s) of the table."""
        raise NotImplementedError

    @property
    def supports_non_nullable_columns(self) -> bool:
        """Return whether this data source supports columns with nullable constraints."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Perform any final clean-up."""
        raise NotImplementedError

    @abstractmethod
    def get_row_count(self) -> int | None:
        """Get the number of rows in a file."""
        raise NotImplementedError

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """Get the file as a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def get_group_map(self, column_name: Hashable | Sequence[Hashable], *,
                      seed: int | None = None
                      ) -> dict[Hashable, int]:
        """
        Get a map of each unique value of `column_name` to its group number.

        If a sequence of column names is provided, groups by the combination of
        all specified columns.
        """
        raise NotImplementedError

    @abstractmethod
    def get_n_random_rows(self, samples: int = 5000, seed: int | None = None) -> pd.DataFrame:
        """Get a specified number of random rows."""
        raise NotImplementedError

    @abstractmethod
    def write_chunk(self, chunk: pd.DataFrame, *,
                    if_exists: t.Literal["fail", "replace", "append"] = "append"
                    ) -> None:
        """Write a chunk."""
        raise NotImplementedError

    @abstractmethod
    def yield_variable_length_chunks(self, initial_chunk_size: int = 100, *,
                                     maintain_natural_order: bool = False,
                                     max_rows: int | None = None,
                                     skip_rows: int = 0,
                                     seed: int | None = None,
                                     ) -> Generator[pd.DataFrame, int, None]:
        """
        Provide a bidirectional generator of variable-length chunks.

        Call ``next(gen)`` or ``gen.send(None)`` for the first chunk; for each
        subsequent chunk, ``gen.send(n)`` returns a chunk of length ``n``.
        """
        raise NotImplementedError

    @abstractmethod
    def yield_chunk(self, chunk_size: int = 5000, *,
                    max_chunks: int | None = None,
                    skip_chunks: int | None = None,
                    maintain_natural_order: bool = False,
                    seed: int | None = None,
                    ) -> Iterator[pd.DataFrame]:
        """Provide a chunk generator."""
        raise NotImplementedError

    @abstractmethod
    def yield_grouped_chunk(self, column_name: Hashable | Sequence[Hashable] | None,
                            groups: Iterable[Iterable[t.Any]], *,
                            feature_attributes: Mapping[Hashable, t.Any] | None,
                            max_chunks: int | None = None,
                            skip_chunks: int = 0,
                            time_feature: str = "",
                            ) -> Iterator[pd.DataFrame]:
        """Provide a grouped chunk generator."""
        raise NotImplementedError

    def yield_grouped_chunk_with_lag_context(self, group_feature: Hashable | Sequence[Hashable],
                                             groups: Iterable[Iterable[t.Any]], *,
                                             id_feature_name: Hashable | Sequence[Hashable],
                                             time_feature: str,
                                             num_lags: int,
                                             feature_attributes: Mapping[Hashable, t.Any] | None = None,
                                             ) -> Iterator[pd.DataFrame]:
        """
        Provide a grouped chunk generator augmented with per-series lag context.

        Not all data sources support this; those that do not raise
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @abstractmethod
    def map_keys(self, chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[t.Any, dict[t.Any, list[t.Any]]]]:
        """Map keys to a chunk, returning the updated chunk and the primary key map."""
        raise NotImplementedError

    @abstractmethod
    def get_unique_values(self, column_name: Hashable | Sequence[Hashable]) -> Collection[t.Any]:
        """Return the set of unique values in the provided column(s)."""
        raise NotImplementedError

    @abstractmethod
    def get_unique_count(self, column_name: Hashable | Sequence[Hashable]) -> int:
        """Get the number of unique values in the provided column(s)."""

    @abstractmethod
    def is_unique(self, column_name: Hashable) -> bool:
        """Return whether the given column contains only unique values."""

    @abstractmethod
    def contains_nulls(self, column_name: Hashable) -> bool:
        """Return whether the given column contains any null values."""

    def is_nullable_column(self, column_name: Hashable) -> bool:
        """
        Return whether the given column allows nulls per the source schema.

        Unlike :meth:`contains_nulls`, which inspects the data itself, this
        reports the column's declared nullability. Data sources that do not
        constrain nullability report all columns nullable.
        """
        raise NotImplementedError


@t.runtime_checkable
class IFACompatibleADCProtocol(t.Protocol):
    """
    Protocol for an abstract data file object with extended functionality.

    Includes functions that make it compatible with `infer_feature_attributes`.
    Because this protocol is runtime-checkable, every member declared here is
    required of any object that will be recognized as an ADC. Only declare
    members that all IFA-compatible ADCs implement; members provided by a
    subset of backends (e.g. ``get_inspector``) belong on a narrower protocol
    such as :class:`IFACompatibleSQLADCProtocol`.
    """

    @property
    @abstractmethod
    def headers(self) -> list[str]:
        """Return a list of the column names of the data."""

    @property
    def primary_keys(self) -> str | list[str] | None:
        """Return the primary key(s) of the data."""
        raise NotImplementedError

    @property
    def foreign_keys(self) -> str | Iterable[str] | None:
        """Return the foreign key(s) of the data."""
        raise NotImplementedError

    @abstractmethod
    def get_row_count(self) -> int | None:
        """Get the number of rows in the data source."""

    @abstractmethod
    def get_n_random_rows(self, samples: int = 5000, seed: int | None = None) -> pd.DataFrame:
        """Get random samples from the given data frame as a data frame."""

    @abstractmethod
    def get_decimal_places(self, column_name: Hashable) -> int:
        """Get the number of decimal places for values in the given column, if applicable."""

    @abstractmethod
    def get_random_value(self, column_name: Hashable, no_nulls: bool = False,
                         count: int = 1) -> t.Any | list[t.Any]:
        """
        Return one or more random samples from the given column.

        The return type is determined by the column type.

        If `no_nulls` is set, select random values from the set of non-null
        values, if any. If there are no such non-nulls, this returns None when
        `count` is 1, or an empty list when `count` is greater than 1.

        When `count` is 1 (the default) a single value is returned; when it is
        greater than 1, a list of up to `count` values is returned.
        """

    @abstractmethod
    def get_min_max_values(self, column_name: Hashable, *, datetime_format: str | None = None) -> tuple[t.Any, t.Any]:
        """Get the smallest and largest values in the given column."""

    @abstractmethod
    def get_num_cases(self, column_name: Hashable) -> int:
        """Return the number of non-null cases in the given column."""

    @abstractmethod
    def get_mode(self, column_name: Hashable) -> list[tuple[t.Any, int]]:
        """
        Get the most common value in the given feature/column.

        If multiple values have the same mode all of them will be returned, as
        long as the count is a value greater than 1.
        """

    @abstractmethod
    def get_column_dtype(self, column_name: Hashable) -> str:
        """Get the dtype of the given column."""

    @abstractmethod
    def get_first_non_null(self, column_name: Hashable) -> t.Any | None:
        """Get the first non-null value in the given column."""

    @abstractmethod
    def get_null_count(self, column_name: Hashable) -> int:
        """Get the number of nulls in the given column."""

    @abstractmethod
    def get_unique_values(self, column_name: Hashable | Sequence[Hashable]) -> Collection[t.Any]:
        """Return the unique values in `column_name`."""

    @abstractmethod
    def get_unique_count(self, column_name: Hashable | Sequence[Hashable]) -> int:
        """Get the number of unique values in the provided column(s)."""

    @abstractmethod
    def get_value_count(self, column_name: Hashable, value: t.Any, *, max_rows_to_eval: int | None = None,
                        chunk_size: int = 5000) -> int:
        """
        Get the number of occurrences of `value` in `column_name`.

        If `max_rows_to_eval` is given, at most that many rows are evaluated
        and the result reflects only the rows inspected, not the full dataset.
        """

    @abstractmethod
    def get_group_map(self, column_name: Hashable | Sequence[Hashable], *,
                      seed: int | None = None) -> dict[Hashable, int]:
        """
        Get a map of each unique value of `column_name` to its group number.

        If a sequence of column names is provided, groups by the combination of
        all specified columns.
        """

    @abstractmethod
    def contains_nulls(self, column_name: Hashable) -> bool:
        """Return whether the given column contains any null values."""

    @abstractmethod
    def is_nullable_column(self, column_name: Hashable) -> bool:
        """
        Return whether the provided column allows nulls.

        Unlike :meth:`contains_nulls`, which inspects the data itself, this
        reports the column's declared nullability per the source schema. Data
        sources that do not constrain nullability report all columns nullable.
        """

    @abstractmethod
    def yield_chunk(self, chunk_size: int = 5000, *,
                    max_chunks: int | None = None,
                    skip_chunks: int | None = None,
                    maintain_natural_order: bool = False,
                    seed: int | None = None,
                    ) -> Iterator[pd.DataFrame]:
        """Yield `chunk_size` data frames from the data."""

    @abstractmethod
    def yield_grouped_chunk(self, column_name: Hashable | Sequence[Hashable] | None,
                            groups: Iterable[Iterable[t.Any]], *,
                            feature_attributes: Mapping[Hashable, t.Any] | None,
                            max_chunks: int | None = None,
                            skip_chunks: int = 0,
                            time_feature: str = "",
                            ) -> Iterator[pd.DataFrame]:
        """Yield a data frame per group from the given groups."""


@t.runtime_checkable
class IFACompatibleSQLADCProtocol(IFACompatibleADCProtocol, t.Protocol):
    """Protocol for a SQL-backed ADC compatible with `infer_feature_attributes`."""

    table_name: TableNameProtocol

    @abstractmethod
    def get_inspector(self) -> t.Any | None:
        """Get a ``sqlalchemy.Inspector`` for this table, if applicable."""


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
    def get_data(self, table_name: TableNameProtocol) -> AbstractDataProtocol:
        """Get the data in a specified table."""
        raise NotImplementedError

    @abstractmethod
    def set_data(self, table_name: TableNameProtocol, data: AbstractDataProtocol):
        """Set the data in a specified table."""
        raise NotImplementedError

    @abstractmethod
    def get_values(self,
                   table_name: TableNameProtocol,
                   primary_key_columns: list[str] | str,
                   primary_key_values: list[list[t.Any]] | list[t.Any],
                   column_name: Hashable) -> list[t.Any]:
        """Get the column values in a specified table."""
        raise NotImplementedError

    @abstractmethod
    def replace_values(
        self,
        table_name: TableNameProtocol,
        primary_key_columns: list[str] | str,
        primary_key_values: list[t.Any] | t.Any,
        column_name: Hashable,
        replace_values: list[t.Any],
        return_old: bool = False,
    ) -> list[t.Any] | None:
        """Replace the column values in a specified table."""
        raise NotImplementedError


@t.runtime_checkable
class SQLRelationalDatastoreProtocol(DatastoreProtocol, t.Protocol):
    """Protocol for a SQL relational datastore object."""

    engine: int
    graph: t.Any
