import decimal
from simple_ddl_parser import DDLParser
from typing import (
    Dict, Optional, Tuple
)

from .base import (
    MultiTableFeatureAttributes,
    SingleTableFeatureAttributes
)
from ..features import FeatureType
from .relational import DatastoreColumnTypes
from ..utilities import (
    date_to_epoch,
    determine_iso_format,
    epoch_to_date,
    ISO_8601_DATE_FORMAT,
    ISO_8601_FORMAT,
    time_to_seconds,
)


class InferFeatureAttributesSQLSchemaTable():
    """Supports inferring feature attributes for SQL tables based on schema."""

    def __init__(self,
                 table_schema: dict,
                 column_types: DatastoreColumnTypes):
        """
        Instantiate this InferFeatureAttributesSQLTable object.
        """

        self.table_schema = table_schema
        self.column_types = column_types

    def __call__(self, **kwargs) -> SingleTableFeatureAttributes:
        """Process and return the feature attributes."""
        return SingleTableFeatureAttributes(self._process(**kwargs), kwargs)

    def _process(self) -> dict:
        # from pprint import pprint as pretty
        # pretty(self.table_schema)
        feature_attributes = dict()
        for column in self.table_schema['columns']:
            # What type is this feature?
            feature_name = column.get('name')
            feature_type, typing_info = self._get_feature_type(column)
            typing_info = typing_info or dict()

            feature_attributes[feature_name] = {}

            # FLOATING POINT FEATURES
            if feature_type == FeatureType.NUMERIC:
                feature_attributes[feature_name] = (
                    self._infer_floating_point_attributes(feature_name))

            # IMPLICITLY DEFINED DATETIME FEATURES
            elif feature_type == FeatureType.DATETIME:
                feature_attributes[feature_name] = (
                    self._infer_datetime_attributes(feature_name))

            # DATE ONLY FEATURES
            elif feature_type == FeatureType.DATE:
                feature_attributes[feature_name] = (
                    self._infer_date_attributes(feature_name))

            # TIME ONLY FEATURES
            elif feature_type == FeatureType.TIME:
                feature_attributes[feature_name] = (
                    self._infer_time_attributes(feature_name))

            # TIMEDELTA FEATURES
            elif feature_type == FeatureType.TIMEDELTA:
                feature_attributes[feature_name] = (
                    self._infer_timedelta_attributes(feature_name))

            # INTEGER FEATURES
            elif feature_type == FeatureType.INTEGER:
                feature_attributes[feature_name] = (
                    self._infer_integer_attributes(feature_name))

            # BOOLEAN FEATURES
            elif feature_type == FeatureType.BOOLEAN:
                feature_attributes[feature_name] = (
                    self._infer_boolean_attributes(feature_name))

            # ALL OTHER FEATURES
            else:
                feature_attributes[feature_name] = (
                    self._infer_string_attributes(feature_name))
            
            # Add original type to feature
            if feature_type is not None:
                feature_attributes[feature_name]['original_type'] = {
                    'data_type': str(feature_type),
                    **typing_info
                }

        return feature_attributes

    def _is_primary_key(self, feature_name: str) -> bool:
        """Return True if the given feature_name is a primary key."""
        # TODO nodes[self.data.name] actually needs to be nodes[TableName]
        # return (
        #     feature_name in
        #     self.datastore.graph.nodes[self.table_name][PRIMARY_KEYS]
        # )
        return False

    def _is_foreign_key(self, feature_name: str) -> bool:
        """Return True if the given feature_name is a foreign key."""
        # TODO see above todo
        # for rel_obj in self.datastore.graph.nodes[self.table_name][FOREIGN_KEYS]:
        #     if feature_name in rel_obj.source_columns:
        #         return True

        return False

    def _get_feature_type(self, column: dict) -> Tuple[Optional[FeatureType], Optional[dict]]:   # noqa: C901
        # Place here to avoid circular import
        from howso.client.exceptions import HowsoError
        column_type = column.get('type').upper()
        feature_name = column.get('name')

        if column_type in self.column_types.floating_point_types:
            typing_info = {}
            precision = column.get('precision')
            if precision and precision > 53:
                # TODO #12919 - may need to be turned into a warning or
                #  only raised if actual values exceed 64bit
                raise HowsoError(
                    f'Unsupported column definition "{column_type}" '
                    f'found for column "{self.data.name}.{feature_name}", '
                    'Howso does not currently support numbers '
                    'larger than 64-bit.')

            # # Get size of float in bytes.
            # # NOTE: Some floating point numbers use variable storage
            # # sizes dependent on the dialect and the precision, we don't
            # # include the size for these
            # variable_types = self.column_types.variable_size_numbers
            # exact_types = self.column_types.exact_size_numbers
            # if column_type not in variable_types:
            #     try:
            #         typing_info['size'] = exact_types[column_type]
            #     except (TypeError, KeyError):
            #         # Not an exact size number
            #         pass

            return FeatureType.NUMERIC, typing_info

        elif column_type in self.column_types.integer_types:
            typing_info = {'size': 8}

            # # Some dialects support unsigned integers, capture it
            # unsigned = getattr(column.type, 'unsigned', False)
            # if unsigned:
            #     typing_info['unsigned'] = True

            # # Get size of integer in bytes
            # try:
            #     typing_info['size'] = (
            #         self.column_types.exact_size_numbers[column_type])
            # except (TypeError, KeyError):
            #     # Not an exact size number
            #     pass

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
            if column.get('length'):
                typing_info['length'] = column.get('length')

            return FeatureType.STRING, typing_info

        else:
            return FeatureType.UNKNOWN, {}

    def _infer_floating_point_attributes(self, feature_name: str) -> Dict:
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
                'data_type': 'number',
                'non_sensitive': True
            }

        attributes = {'type': 'continuous', 'data_type': 'number'}
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
                'non_sensitive': True
            }

        return {
            'type': 'continuous',
            'data_type': 'formatted_date_time',
            'date_time_format': ISO_8601_FORMAT,
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
                'non_sensitive': True
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
                'non_sensitive': True
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
                'non_sensitive': True
            }

        attributes = {
            'type': 'continuous',
            'data_type': 'number',
            'decimal_places': 0,
            "bounds": {
                "min": 0.0,
                "max": 1000000.0,
                "allow_null": False,
            }
        }

        return attributes

    def _infer_string_attributes(self, feature_name: str) -> Dict:
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
                'non_sensitive': True
            }

        return self._infer_unknown_attributes(feature_name)

    def _infer_unknown_attributes(self, feature_name: str) -> Dict:
        return {
            'type': 'nominal',
            'non_sensitive': True
        }


class InferFeatureAttributesSQLSchema:
    """Supports inferring feature attributes from SQL schemas."""

    def __init__(self, schema: str, **kwargs):
        """Instantiate this InferFeatureAttributesSchema object."""
        self.schema = schema

        self.column_types = DatastoreColumnTypes(None)

        # Save kwargs for get_parameters
        self.params = kwargs

    def __call__(self, **kwargs) -> MultiTableFeatureAttributes:
        """Process feature attributes for all tables in the datastore, returned as one object."""
        # Infer feature attributes for each table in the datastore
        feature_attr = {}

        parsed_schema = DDLParser(self.schema).run()
        for table_schema in parsed_schema:
            infer_table_attr = InferFeatureAttributesSQLSchemaTable(table_schema, self.column_types)
            feature_attr[table_schema['table_name']] = infer_table_attr(**kwargs)

        return MultiTableFeatureAttributes(feature_attr, kwargs)
