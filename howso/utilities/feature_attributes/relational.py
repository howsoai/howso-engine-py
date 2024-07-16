from contextlib import contextmanager
import datetime
from datetime import time, timedelta
import decimal
import logging
from math import ceil, isnan, log
import re
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Tuple
)
import warnings

import numpy as np
import pandas as pd

from .base import InferFeatureAttributesBase, MultiTableFeatureAttributes, SingleTableFeatureAttributes
from .protocols import (
    SessionProtocol,
    SQLRelationalDatastoreProtocol,
    SQLTableProtocol,
    TableNameProtocol,
)
from ..features import FeatureType
from ..utilities import (
    date_to_epoch,
    determine_iso_format,
    epoch_to_date,
    ISO_8601_DATE_FORMAT,
    ISO_8601_FORMAT,
    time_to_seconds,
)

logger = logging.getLogger(__name__)

# Import SQLAlchemy only if it's available. Not all users of this package will have or need
# this installed.
try:
    import sqlalchemy as db
    from sqlalchemy import func, inspect, MetaData
    from sqlalchemy.orm import sessionmaker
except ImportError:
    db = None

COLUMNS = 'COLUMNS'
COLUMN_TYPE_PATTERN = re.compile(
    r'(?P<type>\w+)\(?(?P<precision>\d+)?,?\s?(?P<scale>\d+)?\)?'
)
FOREIGN_KEYS = 'FOREIGN_KEYS'
IGNORED_FKEYS = 'IGNORED_FKEYS'
PRIMARY_KEYS = 'PRIMARY_KEYS'
REFLEXIVE_FKEYS = 'REFLEXIVE_FKEYS'
UNIQUE_COLUMNS = 'UNIQUE_COLUMNS'


class DatastoreColumnTypes:
    """
    Maps database column types to a data type.

    Note: Some dialects support different configurations of data types.
    """

    DEFAULTS = {
        # Mapping of numeric type to its storage size (in bytes)
        'exact_size_numbers': {
            # Integers
            'TINYINT': 1,
            'SMALLINTEGER': 2,
            'SMALLINT': 2,
            'MEDIUMINTEGER': 3,
            'MEDIUMINT': 3,
            'INTEGER': 4,
            'INT': 4,
            'SERIAL': 4,
            'BIGINTEGER': 8,
            'BIGINT': 8,
            'BIGSERIAL': 8,
            # Floats
            'BINARY_FLOAT': 4,
            'FLOAT': 4,
            'SMALLMONEY': 4,
            'REAL': 4,
            'BINARY_DOUBLE': 8,
            'DOUBLE': 8,
            'MONEY': 8,
            'DOUBLE_PRECISION': 8,
        },
        # Numeric types that have variable storage sizes
        'variable_size_numbers': ['DECIMAL', 'NUMERIC', ],
        # Data types that are always timezone aware
        'tz_aware_date_time_types': [],
        # Data type mappings
        'floating_point': [
            'BINARY_DOUBLE', 'BINARY_FLOAT', 'DECIMAL', 'DOUBLE',
            'DOUBLE_PRECISION', 'FLOAT', 'MONEY', 'NUMERIC', 'REAL',
            'SMALLMONEY',
        ],
        'integer': [
            'BIGINT', 'BIGINTEGER', 'INT', 'INTEGER', 'LONG', 'MEDIUMINT',
            'SMALLINT', 'SMALLINTEGER', 'TINYINT', 'YEAR', 'BIGSERIAL',
            'SERIAL',
        ],
        'boolean': ['BOOLEAN', ],
        'datetime': [
            'DATETIME', 'DATETIME2', 'SMALLDATETIME', 'TIMESTAMP',
        ],
        'date': ['DATE', ],
        'time': ['TIME', ],
        'timedelta': ['INTERVAL', ],
        'string': [
            'BFILE', 'BINARY', 'BIT', 'BLOB', 'BOOLEAN', 'BYTEA', 'CHAR',
            'CIDR', 'CLOB', 'CONCATENABLE', 'ENUM', 'IMAGE', 'INET', 'JSON',
            'JSONB', 'LARGEBINARY', 'LONGBLOB', 'LONGTEXT', 'MACADDR',
            'MEDIUMBLOB', 'MEDIUMTEXT', 'NCHAR', 'NCLOB', 'NTEXT', 'NVARCHAR',
            'OID', 'RAW', 'REGCLASS', 'ROWID', 'STRING', 'TEXT', 'TINYBLOB',
            'TINYTEXT', 'UNICODE', 'UNICODETEXT', 'UNIQUEIDENTIFIER', 'UUID',
            'VARBINARY', 'VARBINARY', 'VARCHAR', 'VARCHAR2', 'XML',
        ],
    }
    DIALECT_OVERRIDES = {
        'mssql': {
            'exact_size_numbers': {'FLOAT': 8, },
            'boolean': ['BIT', ],
            'tz_aware_date_time_types': ['DATETIMEOFFSET', ],
            'datetime': [
                'DATETIME', 'DATETIME2', 'SMALLDATETIME', 'DATETIMEOFFSET',
            ],
        }
    }

    def __init__(self, dialect) -> None:
        """Initialize this DatastoreColumnTypes class and set the dialect."""
        self._dialect = dialect

    def _get_data_types(self, key: str) -> List[str]:
        """
        Get data types by key.

        Parameters
        ----------
        key : str
            The type key.

        Returns
        -------
        list of str or dict
            The supported database column types.
        """
        try:
            return self.DIALECT_OVERRIDES[self._dialect.name][key]
        except (TypeError, KeyError):
            return self.DEFAULTS[key]

    @property
    def variable_size_numbers(self) -> List[str]:
        """Get variable size number column types."""
        return self._get_data_types('variable_size_numbers')

    @property
    def exact_size_numbers(self) -> Dict[str, int]:
        """Get exact size number column types."""
        try:
            overrides = self.DIALECT_OVERRIDES[
                self._dialect.name]['exact_size_numbers']
        except (TypeError, KeyError):
            overrides = {}
        return {
            **self.DEFAULTS['exact_size_numbers'],
            **overrides
        }

    @property
    def floating_point_types(self) -> List[str]:
        """Get floating point column types."""
        return self._get_data_types('floating_point')

    @property
    def integer_types(self) -> List[str]:
        """Get integer column types."""
        return self._get_data_types('integer')

    @property
    def boolean_types(self) -> List[str]:
        """Get boolean column types."""
        return self._get_data_types('boolean')

    @property
    def tz_aware_date_time_types(self) -> List[str]:
        """Get date time column types that are timezone aware."""
        return self._get_data_types('tz_aware_date_time_types')

    @property
    def datetime_types(self) -> List[str]:
        """Get date-time column types."""
        return self._get_data_types('datetime')

    @property
    def date_types(self) -> List[str]:
        """Get date column types."""
        return self._get_data_types('date')

    @property
    def time_types(self) -> List[str]:
        """Get time column types."""
        return self._get_data_types('time')

    @property
    def all_date_time_types(self) -> List[str]:
        """Get all date and time types."""
        return (self._get_data_types('datetime')
                + self._get_data_types('date')
                + self._get_data_types('time'))

    @property
    def timedelta_types(self) -> List[str]:
        """Return timedelta column types."""
        return self._get_data_types('timedelta')

    @property
    def string_types(self) -> List[str]:
        """Return string column types."""
        return self._get_data_types('string')


@contextmanager
def session_scope(session_class):
    """Provide a transactional scope around a series of operations."""
    session = session_class()
    try:
        yield session
        session.commit()
    except Exception:
        logger.exception('Database error')
        session.rollback()
        raise
    finally:
        session.close()


class InferFeatureAttributesSQLTable(InferFeatureAttributesBase):
    """Supports inferring feature attributes for SQL tables."""

    def __init__(self, data: SQLTableProtocol, table_name: TableNameProtocol,
                 column_types: DatastoreColumnTypes,
                 session_cls: SessionProtocol,
                 parent_datastore: SQLRelationalDatastoreProtocol):
        """
        Instantiate this InferFeatureAttributesSQLTable object.

        Parameters
        ----------
        data : SQLTableProtocol
            The SQL table containing the features whose attributes will be inferred.
        table_name : TableNameProtocol
            A TableName object with information about this SQL table.
        column_types : DatastoreColumnTypes
            The datastore column types.
        session_cls : SessionProtocol
            The SQLAlchemy Session object associated with this SQL table.
        parent_datastore : SQLRelationalDatastoreProtocol
            The datastore to which this SQL table belongs.
        """
        if not db:
            raise ImportError('Must have SQLAlchemy installed to instantiate '
                              'FeatureAttributesSQLTable. See synthesizer-data-services/'
                              'requirements.in for versioning.')
        self.data = data
        self.table_name = table_name
        self.column_types = column_types
        self.session_cls = session_cls
        self.datastore = parent_datastore
        # Keep track of features that contain unsupported data
        self.unsupported = []

    def __call__(self, **kwargs) -> SingleTableFeatureAttributes:
        """Process and return the feature attributes."""
        return SingleTableFeatureAttributes(self._process(**kwargs), kwargs,
                                            unsupported=self.unsupported)

    def _is_primary_key(self, feature_name: str) -> bool:
        """Return True if the given feature_name is a primary key."""
        # TODO nodes[self.data.name] actually needs to be nodes[TableName]
        return (
            feature_name in
            self.datastore.graph.nodes[self.table_name][PRIMARY_KEYS]
        )

    def _is_foreign_key(self, feature_name: str) -> bool:
        """Return True if the given feature_name is a foreign key."""
        # TODO see above todo
        for rel_obj in self.datastore.graph.nodes[self.table_name][FOREIGN_KEYS]:
            if feature_name in rel_obj.source_columns:
                return True

        return False

    def _get_first_non_null(self, feature_name: str) -> Optional[Any]:
        with session_scope(self.session_cls) as session:
            first_non_null = session.query(self.data.c[feature_name]).filter(
                self.data.c[feature_name].is_not(None)).first()
        return first_non_null

    def _get_random_value(self, feature_name: str, no_nulls: bool = False) -> Optional[Any]:
        """
        Return a random sample from the given table column.

        The return type is determined by the column type.

        if `no_nulls` is set, select a random value from the set of non-null
        values, if any. If there are no such non-nulls, this will return None.
        """
        with session_scope(self.session_cls) as session:
            # Get the total number of non-null rows in the given table
            if no_nulls:
                num_samples = session.query(self.data.c[feature_name]).filter(
                    self.data.c[feature_name].is_not(None)).count()
            else:
                num_samples = session.query(self.data.c[feature_name]).count()

            if num_samples == 0:
                random_value = None
            else:
                # Select a pseudo-random one by offset. This should be DB
                # agnostic and reasonably performant since we only want a
                # single row/value.
                idx = np.random.randint(num_samples)
                if no_nulls:
                    random_value = (
                        session.query(self.data.c[feature_name])
                               .filter(self.data.c[feature_name].is_not(None))
                               .order_by(feature_name)  # 19234: using order_by here is necessary for MSSQL,
                               .offset(idx)             # however, this should ultimately be re-written to
                               .first()                 # avoid the performance pentalty this imposes.
                    )
                else:
                    random_value = (
                        session.query(self.data.c[feature_name])
                               .order_by(feature_name)  # See above
                               .offset(idx)
                               .first()
                    )

        return random_value

    def _get_num_uniques(self, feature_name: str) -> int:
        """Return the count of unique values for the given table feature."""
        with session_scope(self.session_cls) as session:
            num_unique = (session.query(
                self.data.c[feature_name]).distinct().count())
        return num_unique

    def _has_unique_constraint(self, feature_name: str) -> bool:
        """Return True if the given feature_name has a unique constraint."""
        if self._is_primary_key(feature_name):
            # All PKs have a natural unique constraint
            return True

        # Inspect the information schema to see if the given non-pk column has
        # a unique constraint by itself.
        inspector = inspect(self.datastore.engine)
        try:
            uniques = inspector.get_unique_constraints(self.table_name.table,
                                                       schema=self.data.schema)
        except NotImplementedError:
            # MSSQL is not currently supported in sqlalchemy for get_unique_constraints;
            # however, get_indexes can also be used to identify UNIQUE constraints
            indexes = inspector.get_indexes(self.table_name.table,
                                            schema=self.data.schema)
            return any([c['column_names'] == [feature_name] and c['unique'] for c in indexes])

        return any([c['column_names'] == [feature_name] for c in uniques])

    def _get_unique_values(self, feature_name: str) -> List[Any]:
        """Get a list of all the unique values for a column."""
        with session_scope(self.session_cls) as session:
            distinct_values = (
                session.query(self.data.c[feature_name])
                       .distinct()
                       .all()
            )
        return distinct_values

    @classmethod
    def _value_to_number(cls, value: Any) -> Any:
        """Convert value to a number."""
        if pd.isna(value):
            return float('nan')
        elif isinstance(value, decimal.Decimal):
            return float(value)
        elif isinstance(value, timedelta):
            return value.total_seconds()
        elif isinstance(value, time):
            return time_to_seconds(value)
        else:
            return value

    def _get_min_max_values(self, feature_name: str) -> Tuple[Any, Any]:
        """
        Get the smallest and largest values for the given table column.

        The return type within the Tuple is determined by the column type.
        Smallness and largeness is determined by the SQLAlchemy functions
        `min()` and `max()`.
        """
        with session_scope(self.session_cls) as session:
            results = session.query(
                func.min(self.data.c[feature_name]).label('min_value'),
                func.max(self.data.c[feature_name]).label('max_value')
            ).one()

        return results.min_value, results.max_value

    def _get_mode(self, feature_name: str) -> List[Tuple[Any, int]]:
        """
        Get the most common value in the given feature/column.

        If multiple values have the same mode all of them will be returned, as
        long as the count is a value greater than 1.
        """
        with session_scope(self.session_cls) as session:
            subquery = session.query(
                func.count(self.data.c[feature_name]).label('count')
            ).group_by(self.data.c[feature_name]).subquery('all_counts')
            max_count = session.query(
                func.max(subquery.c.count).label('max_count')).scalar_subquery()
            result = (
                session.query(
                    self.data.c[feature_name],
                    func.count(self.data.c[feature_name]).label('num')
                ).group_by(
                    self.data.c[feature_name]
                ).having(
                    db.sql.and_(
                        func.count(self.data.c[feature_name]) > 1,
                        func.count(self.data.c[feature_name]) >= max_count
                    )
                )
            )

        if result is None:
            return []

        # value, count
        return [(r[0], r[1]) for r in result]

    def _get_num_features(self) -> int:
        return len(self._get_feature_names())

    def _get_num_cases(self) -> int:
        with session_scope(self.session_cls) as session:
            num_rows = session.query(self.data).count()
        return num_rows

    def _get_feature_names(self) -> List[str]:
        return [c.name for c in self.data.columns]

    def _get_feature_type(self, feature_name: str  # noqa: C901
                          ) -> Tuple[Optional[FeatureType], Optional[Dict]]:
        # Place here to avoid circular import
        from howso.client.exceptions import HowsoError
        for column in self.data.columns:
            if column.name == feature_name:
                column_type, column_options = self._parse_column_type(str(column.type))

                if column_type in self.column_types.floating_point_types:
                    typing_info = {}
                    precision = column_options.get('precision')
                    if precision and precision > 53:
                        # TODO #12919 - may need to be turned into a warning or
                        #  only raised if actual values exceed 64bit
                        raise HowsoError(
                            f'Unsupported column definition "{column.type}" '
                            f'found for column "{self.data.name}.{feature_name}", '
                            'Howso does not currently support numbers '
                            'larger than 64-bit.')

                    # Get size of float in bytes.
                    # NOTE: Some floating point numbers use variable storage
                    # sizes dependent on the dialect and the precision, we don't
                    # include the size for these
                    variable_types = self.column_types.variable_size_numbers
                    exact_types = self.column_types.exact_size_numbers
                    if column_type not in variable_types:
                        try:
                            typing_info['size'] = exact_types[column_type]
                        except (TypeError, KeyError):
                            # Not an exact size number
                            pass

                    try:
                        if column.type.python_type == decimal.Decimal:
                            typing_info['format'] = 'decimal'
                    except Exception:  # noqa: Deliberately broad
                        # Some type classes do not implement this attribute
                        pass

                    return FeatureType.NUMERIC, typing_info

                elif column_type in self.column_types.integer_types:
                    typing_info = {}

                    # Some dialects support unsigned integers, capture it
                    unsigned = getattr(column.type, 'unsigned', False)
                    if unsigned:
                        typing_info['unsigned'] = True

                    # Get size of integer in bytes
                    try:
                        typing_info['size'] = (
                            self.column_types.exact_size_numbers[column_type])
                    except (TypeError, KeyError):
                        # Not an exact size number
                        pass

                    return FeatureType.INTEGER, typing_info

                elif column_type in self.column_types.datetime_types:
                    return FeatureType.DATETIME, {}

                elif column_type in self.column_types.time_types:
                    tz_aware_types = self.column_types.tz_aware_date_time_types
                    if (
                        getattr(column.type, 'timezone', False) or
                        column_type in tz_aware_types
                    ):
                        # If timezone aware, treat as a datetime
                        return FeatureType.DATETIME, {}
                    return FeatureType.TIME, {}

                elif column_type in self.column_types.date_types:
                    return FeatureType.DATE, {}

                elif column_type in self.column_types.timedelta_types:
                    # All time deltas will be converted to seconds
                    return FeatureType.TIMEDELTA, {'unit': 'seconds'}

                elif column_type in self.column_types.boolean_types:
                    return FeatureType.BOOLEAN, {}

                elif column_type in self.column_types.string_types:
                    typing_info = {}
                    if column_options.get('length'):
                        typing_info['length'] = column_options.get('length')

                    return FeatureType.STRING, typing_info

                else:
                    return FeatureType.UNKNOWN, {}

        return None, None

    def _infer_floating_point_attributes(self, feature_name: str) -> Dict:
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
                'data_type': 'number',
            }

        attributes = {'type': 'continuous', 'data_type': 'number'}
        num_cases = self._get_num_cases()

        column = self.data.c[feature_name]
        with session_scope(self.session_cls) as session:
            num_nulls = session.query(column).filter(column.is_(None)).count()

        # Ensure we have at least one valid value before attempting to
        # introspect further.
        if num_nulls < num_cases:
            # Determine if nominal by checking if number of uniques <= 2
            if (
                    self._get_num_uniques(feature_name) <= 2 and
                    num_cases > 10
            ):
                attributes = {
                    'type': 'nominal',
                    'data_type': 'number',
                }

        column_type, _ = self._parse_column_type(str(column.type))

        # Capture scale as decimal_places (if applicable)
        if getattr(column.type, 'scale', None) is not None:
            attributes['decimal_places'] = column.type.scale

        # Capture precision as significant_digits (if applicable)
        # Note: Since core works in double precision, we can exclude the
        # significant digits for precision 53 since it won't be necessary and
        # including it will affect performance.
        precision = None
        if getattr(column.type, 'precision', None) is not None:
            precision = column.type.precision
        elif column_type in ('BINARY_FLOAT', 'FLOAT', 'REAL', ):
            precision = 24

        if precision and precision < 53:
            attributes['significant_digits'] = ceil(
                precision * (log(2) / log(10)))

        return attributes

    def _infer_datetime_attributes(self, feature_name: str) -> Dict:
        # Although rare, it is plausible that a datetime field could be a
        # primary- or foreign-key.
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
            }

        column = self.data.c[feature_name]
        column_type, _ = self._parse_column_type(str(column.type))
        if (
            getattr(column.type, 'timezone', False) or
            column_type in self.column_types.tz_aware_date_time_types
        ):
            dt_format = ISO_8601_FORMAT + '%z'
        else:
            dt_format = ISO_8601_FORMAT

        return {
            'type': 'continuous',
            'data_type': 'formatted_date_time',
            'date_time_format': dt_format,
        }

    def _infer_date_attributes(self, feature_name: str) -> Dict:
        # Although rare, it is plausible that a date field could be a
        # primary- or foreign-key.
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
            }

        return {
            'type': 'continuous',
            'data_type': 'formatted_date_time',
            'date_time_format': ISO_8601_DATE_FORMAT,
        }

    def _infer_time_attributes(self, feature_name: str) -> Dict:
        return {
            'type': 'continuous',
            'data_type': 'number',
        }

    def _infer_timedelta_attributes(self, feature_name: str) -> Dict:
        # Although rare, it is plausible that a timedelta field could be a
        # primary- or foreign-key.
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
            }

        return {
            'type': 'continuous',
            'data_type': 'number',
        }

    def _infer_boolean_attributes(self, feature_name: str) -> Dict:
        return {
            'type': 'nominal',
            'data_type': 'boolean',
        }

    def _infer_integer_attributes(self, feature_name: str) -> Dict:
        # Most primary keys will be integer types (but not all). These are
        # always treated as nominals.
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
                'data_type': 'number',
                'decimal_places': 0,
            }

        # Decide if categorical by checking number of uniques is fewer
        # than the square root of the total samples or if every value
        # has exactly the same length.
        num_uniques = self._get_num_uniques(feature_name)
        n_cases = self._get_num_cases()
        if num_uniques < pow(n_cases, 0.5):
            guess_nominals = True
        else:
            # Find the largest and smallest non-null values in column.
            try:
                col_min, col_max = (
                    self._get_min_max_values(feature_name))
                if col_min is None or col_max is None:
                    raise AssertionError('No data in the column?')
            except (AssertionError, TypeError):
                # Column is all None?
                guess_nominals = False
            else:
                # Guess nominals if ALL of:
                #   - `col_min` and `col_max` are both greater than zero
                #   - Their length is at least 5
                #   - They have the same length
                guess_nominals = (
                    col_min > 0 and col_max > 0 and
                    len(str(col_min)) >= 5 and
                    len(str(col_min)) == len(str(col_max))
                )

        if guess_nominals:
            attributes = {
                'type': 'nominal',
                'data_type': 'number',
                'decimal_places': 0,
            }
        else:
            attributes = {
                'type': 'continuous',
                'data_type': 'number',
                'decimal_places': 0,
            }

        return attributes

    def _infer_string_attributes(self, feature_name: str) -> Dict:
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
            }

        # Column has arbitrary string values, first check if they
        # are ISO8601 datetimes.
        if self._is_iso8601_datetime_column(feature_name):
            # if datetime, determine the iso8601 format it's using
            sample = self._get_first_non_null(feature_name)
            fmt = determine_iso_format(sample, feature_name)
            return {
                'type': 'continuous',
                'data_type': 'formatted_date_time',
                'date_time_format': fmt
            }
        else:
            return self._infer_unknown_attributes(feature_name)

    def _infer_unknown_attributes(self, feature_name: str) -> Dict:
        return {
            'type': 'nominal'
        }

    def _infer_feature_bounds(self,  # noqa: C901
                              feature_attributes: Mapping[str, Mapping],
                              feature_name: str,
                              tight_bounds: Optional[Iterable[str]] = None,
                              mode_bound_features: Optional[List[str]] = None,
                              ) -> Optional[Dict]:
        output = None
        allow_null = True
        original_type = feature_attributes[feature_name]['original_type']
        decimal_places = feature_attributes[feature_name].get('decimal_places')

        # Only integers by default do no allow nulls.
        if original_type.get('data_type') == FeatureType.INTEGER.value:
            allow_null = False

        if feature_attributes[feature_name].get('type') == 'continuous':
            # Grab the natural feature_type and raw_feature_type
            column = self.data.c[feature_name]
            column_type, _ = self._parse_column_type(str(column.type))
            format_dt = None

            if 'date_time_format' in feature_attributes[feature_name]:
                format_dt = (
                    feature_attributes[feature_name].get('date_time_format'))

                if column_type in self.column_types.all_date_time_types:
                    # The datetime values are stored in a datetime column. We
                    # can get bounds by asking the database.
                    min_date_obj, max_date_obj = (
                        self._get_min_max_values(feature_name))

                else:
                    # For some reason, the feature was declared as date/time
                    # information but it is not in a datetime database column
                    # (E.g. '01252022'). This is unfortunate because that means
                    # the column isn't sortable in a datetime-aware manner.

                    # This loop grabs all the distinct values, then converts
                    # them according to the `format_dt` to a proper datetime
                    # instance, then compares them to find min and max values.
                    min_date_obj = datetime.max
                    max_date_obj = datetime.min

                    try:
                        unique_values = self._get_unique_values(feature_name)
                        # The comma in this loop is necessary since
                        # unique_values is a list of sqlalchemy Row values
                        for dt_str, in unique_values:
                            # Parse using the `format_dt` into a datetime
                            if dt_str:  # skip any empty values
                                date_obj = datetime.strptime(dt_str, format_dt)
                                min_date_obj = min(min_date_obj, date_obj)
                                max_date_obj = max(max_date_obj, date_obj)
                        else:
                            warnings.warn(
                                f'Cannot guess the bounds for feature '
                                f'"{feature_name}" without samples.')
                            return None
                    except Exception:  # noqa: Intentionally broad
                        warnings.warn(
                            f'Feature {feature_name} does not match the '
                            f'provided date time format, unable to guess '
                            f'bounds.')
                        return None

                # Capture the timezone information, so it can be included
                # in the conversion back from epoch.
                min_date_tz = None
                max_date_tz = None
                if isinstance(min_date_obj, (datetime.datetime, datetime.time)):
                    min_date_tz = min_date_obj.tzinfo
                if isinstance(max_date_obj, (datetime.datetime, datetime.time)):
                    max_date_tz = max_date_obj.tzinfo

                # Convert the found date bounds to float seconds since Epoch
                min_value = date_to_epoch(min_date_obj, format_dt)
                max_value = date_to_epoch(max_date_obj, format_dt)

            else:
                min_value, max_value = (
                    self._get_min_max_values(feature_name))
                min_value = self._value_to_number(min_value)
                max_value = self._value_to_number(max_value)

            if (
                min_value is not None and max_value is not None and
                not isnan(min_value) and
                not isnan(max_value)
            ):
                actual_min_value = min_value
                actual_max_value = max_value
                if (
                    tight_bounds is None
                    or feature_name not in tight_bounds
                ):
                    min_value, max_value = (
                        self.infer_loose_feature_bounds(actual_min_value,
                                                        actual_max_value))

                    if (
                        mode_bound_features is None or
                        feature_name in mode_bound_features
                    ):
                        # If the mode for the feature is same as an original
                        # bound, set that appropriate bound to the mode value
                        # since in this case, it probably represents an
                        # application-specific min/max.
                        col_modes = self._get_mode(feature_name)

                        for mode_value, mode_count in col_modes:
                            if mode_count < 4:
                                # Only apply when the value has more than 3
                                # instances in the dataset
                                continue
                            if format_dt:
                                mode_f = date_to_epoch(mode_value, format_dt)
                            else:
                                mode_f = self._value_to_number(mode_value)
                            if actual_min_value == mode_f:
                                min_value = actual_min_value
                            if actual_max_value == mode_f:
                                max_value = actual_max_value
                # If this is a datetime feature, convert back from epoch time
                if format_dt is not None:
                    min_value = epoch_to_date(min_value, format_dt, min_date_tz)
                    max_value = epoch_to_date(max_value, format_dt, max_date_tz)
                output = {'min': min_value, 'max': max_value, 'allow_null': allow_null}
            else:
                # If no min/max were found from the data, use min/max size of
                # the data type.
                min_value, max_value = self._get_min_max_number_size_bounds(
                    feature_attributes, feature_name)
                if min_value is not None and max_value is not None:
                    output = {'min': min_value, 'max': max_value}

        else:
            output = {'allow_null': allow_null}

        if decimal_places is not None:
            if 'max' in output:
                output['max'] = round(output['max'], decimal_places)
            if 'min' in output:
                output['min'] = round(output['min'], decimal_places)

        return output

    def _parse_column_type(self, full_type_str: str) -> Tuple[str, dict]:
        """
        Determine column type from schema description of column.

        Given a full column type string, return a simplified type string and
        its sizes. Examples:

            'VARCHAR(255)' => 'VARCHAR', {'length': 255}
            'DECIMAL(5,2)' => 'DECIMAL', {'precision': 5, 'scale': 2}
            'FLOAT(5)'     => 'FLOAT', {'precision': 5}
            'INTEGER(16)'  => 'INTEGER', {'length': 16}

        Parameters
        ----------
        full_type_str : str
            A string describing an SQL table column type

        Returns
        -------
        tuple : str, dict
            The base type string,
            A dictionary of the relevant parts. The 'type', and depending on
            the type, additional keys 'precision', 'scale', or 'length'.
        """
        if '(' not in full_type_str:
            return full_type_str.upper(), {}

        matched = COLUMN_TYPE_PATTERN.match(full_type_str)
        type_str = matched.group('type').upper()

        if matched.group('precision') is None:
            return_dict = {}
        elif matched.group('scale') is None:
            # Handles 'VARCHAR(255)' or 'INTEGER(16)', etc.
            # length = int/string
            # precision = float
            numeric_types = self.column_types.floating_point_types
            key = 'precision' if type_str in numeric_types else 'length'
            return_dict = {key: int(matched.group('precision'))}
        else:
            # Handles 'DECIMAL(p,s)' or 'NUMERIC(p,s)', etc.
            precision = int(matched.group('precision'))
            scale = int(matched.group('scale'))
            return_dict = {'precision': precision, 'scale': scale}

        return type_str, return_dict


class InferFeatureAttributesSQLDatastore:
    """Supports inferring feature attributes for SQL datastores."""

    def __init__(self, datastore: SQLRelationalDatastoreProtocol,
                 table_names: Iterable[TableNameProtocol], **kwargs):
        """Instantiate this InferFeatureAttributesSQLDatastore object."""
        # Save kwargs for get_parameters
        self.params = kwargs

        # Set table information
        self.datastore = datastore
        self.session_cls = sessionmaker(bind=self.datastore.engine)
        self._dialect = self.datastore.engine.dialect
        self.column_types = DatastoreColumnTypes(self._dialect)
        self._metadata = MetaData()
        self.table_names = table_names

    def __call__(self, **kwargs) -> MultiTableFeatureAttributes:
        """Process feature attributes for all tables in the datastore, returned as one object."""
        # Infer feature attributes for each table in the datastore
        feature_attributes = {}

        for table_name in self.table_names:
            data = db.Table(
                table_name.table, self._metadata, schema=table_name.schema,
                autoload_with=self.datastore.engine
            )
            infer_table_attributes = InferFeatureAttributesSQLTable(data, table_name,
                                                                    self.column_types,
                                                                    self.session_cls,
                                                                    self.datastore)
            table_attributes = infer_table_attributes(**kwargs)
            feature_attributes[str(table_name)] = table_attributes

        return MultiTableFeatureAttributes(feature_attributes, kwargs)
