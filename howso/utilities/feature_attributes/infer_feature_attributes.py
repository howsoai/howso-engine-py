from typing import Iterable, Union

import pandas as pd

from .base import FeatureAttributesBase
from .pandas import InferFeatureAttributesDataFrame
from .protocols import SQLRelationalDatastoreProtocol, TableNameProtocol
from .relational import InferFeatureAttributesSQLDatastore
from .time_series import InferFeatureAttributesTimeSeries


def infer_feature_attributes(data: Union[pd.DataFrame, SQLRelationalDatastoreProtocol], *,
                             tables: Iterable[TableNameProtocol] = None,
                             time_feature_name: str = None, **kwargs) -> FeatureAttributesBase:
    """
    Return a dict-like feature attributes object with useful accessor methods.

    The returned object is a subclass of FeatureAttributesBase that is appropriate for the
    provided data type.

    Parameters
    ----------
    data : Any
        The data source to infer feature attributes from. Must be a supported data type.

    tables : Iterable of TableNameProtocol
        (Optional, required for datastores) An Iterable of table names to
        infer feature attributes for.

        If included, feature attributes will be generated in the form
        ``{table_name: {feature_attribute: value}}``.

    time_feature_name : str, default None
        (Optional, required for time series) The name of the time feature.

    features : dict or None, default None
        (Optional) A partially filled features dict. If partially filled
        attributes for a feature are passed in, those parameters will be
        retained as is and the rest of the attributes will be inferred.

        For example:
            >>> from pprint import pprint
            >>> df.head(2)
            ... sepal-length  sepal-width  petal-length  petal-width  target
            ... 0           6.7          3.0           5.2          2.3       2
            ... 1           6.0          2.2           5.0          1.5       2
            >>> # Partially filled features dict
            >>> partial_features = {
            ...     "sepal-length": {
            ...         "type": "continuous",
            ...         'bounds': {
            ...             'min': 2.72,
            ...             'max': 3,
            ...             'allow_null': True
            ...         },
            ...     },
            ...     "sepal-width": {
            ...         "type": "continuous"
            ...     }
            ... }
            >>> # Infer rest of the attributes
            >>> features = infer_feature_attributes(
            ...     df, features=partial_features
            ... )
            >>> # Inferred Feature dictionary
            >>> pprint(features)
            ... {
            ...     'sepal-length', {
            ...         'bounds': {
            ...             'allow_null': True, 'max': 3, 'min': 2.72
            ...         },
            ...         'type': 'continuous'
            ...     },
            ...     'sepal-width', {
            ...         'bounds': {
            ...             'allow_null': True, 'max': 7.38905609893065,
            ...             'min': 1.0
            ...         },
            ...         'type': 'continuous'
            ...     },
            ...     'petal-length', {
            ...         'bounds': {
            ...             'allow_null': True, 'max': 7.38905609893065,
            ...             'min': 1.0
            ...         },
            ...         'type': 'continuous'
            ...     },
            ...     'petal-width', {
            ...         'bounds': {
            ...             'allow_null': True, 'max': 2.718281828459045,
            ...             'min': 0.049787068367863944
            ...         },
            ...         'type': 'continuous'
            ...     },
            ...     'target', {
            ...         'bounds': {'allow_null': True},
            ...         'type': 'nominal'
            ...     }
            ... }

        Note that valid 'data_type' values for both nominal and continuous types are:
        'string', 'number', 'json', 'amalgam', and 'yaml'.
        The 'boolean' data_type is valid only when type is nominal.
        'string_mixable' is valid only when type is continuous (predicted values may result in
        interpolated strings containing a combination of characters from multiple original values).

    infer_bounds : bool, default True
        (Optional) If True, bounds will be inferred for the features if the
        feature column has at least one non NaN value

    datetime_feature_formats : dict, default None
        (Optional) Dict defining a custom (non-ISO8601) datetime format and
        an optional locale for features with datetimes.  By default datetime
        features are assumed to be in ISO8601 format.  Non-English datetimes
        must have locales specified.  If locale is omitted, the default
        system locale is used. The keys are the feature name, and the values
        are a tuple of date time format and locale string.

        Example::

            {
                "start_date": ("%Y-%m-%d %A %H.%M.%S", "es_ES"),
                "end_date": "%Y-%m-%d"
            }

    delta_boundaries : dict, default None
        (Optional) For time series, specify the delta boundaries in the form
        {"feature" : {"min|max" : {order : value}}}. Works with partial values
        by specifying only particular order of derivatives you would like to
        overwrite. Invalid orders will be ignored.

        Examples::

            {
                "stock_value": {
                    "min": {
                        '0' : 0.178,
                        '1': 3.4582e-3,
                        '2': None
                    }
                }
            }

    derived_orders : dict, default None
        (Optional) Dict of features to the number of orders of derivatives
        that should be derived instead of synthesized. For example, for a
        feature with a 3rd order of derivative, setting its derived_orders
        to 2 will synthesize the 3rd order derivative value, and then use
        that synthed value to derive the 2nd and 1st order.

    dropna : bool, default False
        (Optional) If True, all features will be populated with `'dropna':
        True` parameter. That would mean, rows containing NaNs will be
        automatically dropped when you train.

    lags : list or dict, default None
        (Optional) A list containing the specific indices of the desired lag
        features to derive for each feature (not including the series time
        feature). Specifying derived lag features for the feature specified by
        time_feature_name must be done using a dictionary. A dictionary can be
        used to specify a list of specific lag  indices for specific features.
        For example: {"feature1": [1, 3, 5]} would derive three different
        lag features for feature1. The resulting lag features hold values
        1, 3, and 5 timesteps behind the current timestep respectively.

        .. note ::
            Using the lags parameter will override the num_lags parameter per
            feature

        .. note ::
            A lag feature is a feature that provides a "lagging value" to a
            case by holding the value of a feature from a previous timestep.
            These lag features allow for cases to hold more temporal information.

    num_lags : int or dict, default None
        (Optional) An integer specifying the number of lag features to
        derive for each feature (not including the series time feature).
        Specifying derived lag features for the feature specified by
        time_feature_name must be done using a dictionary. A dictionary can be
        used to specify numbers of lags for specific features. Features that
        are not specified will default to 1 lag feature.

        .. note ::
            The num_lags parameter will be overridden by the lags parameter per
            feature.

    orders_of_derivatives : dict, default None
        (Optional) Dict of features and their corresponding order of
        derivatives for the specified type (delta/rate). If provided will
        generate the specified number of derivatives and boundary values. If
        set to 0, will not generate any delta/rate features. By default all
        continuous features have an order value of 1.

    rate_boundaries : dict, default None
        (Optional) For time series, specify the rate boundaries in the form
        {"feature" : {"min|max" : {order : value}}}. Works with partial values
        by specifying only particular order of derivatives you would like to
        overwrite. Invalid orders will be ignored.

        Examples::

            {
                "stock_value": {
                    "min": {
                        '0' : 0.178,
                        '1': 3.4582e-3,
                        '2': None
                    }
                }
            }

    tight_bounds: Iterable of str, default None
        (Optional) Set tight min and max bounds for the features
        specified in the Iterable.

    tight_time_bounds : bool, default False
        (optional) If True, will set tight bounds on time_feature.
        This will cause the bounds for the start and end times set
        to the same bounds as observed in the original data.

    time_series_type_default : str, default 'rate'
        (Optional) Type specifying how time series is generated.
        One of 'rate' or 'delta', default is 'rate'. If 'rate',
        it uses the difference of the current value from its
        previous value divided by the change in time since the
        previous value. When 'delta' is specified, just uses
        the difference of the current value from its previous value
        regardless of the elapsed time.

    time_series_types_override : dict, default None
        (Optional) Dict of features and their corresponding time series type,
        one of 'rate' or 'delta', used to override time_series_type_default
        for the specified features.

    mode_bound_features : list of str, default None
        (Optional) Explicit list of feature names to use mode bounds for
        when inferring loose bounds. If None, assumes all features. A mode
        bound is used instead of a loose bound when the mode for the
        feature is the same as an original bound, as it may represent an
        application-specific min/max.

    id_feature_name : str or list of str, default None
        (Optional) The name(s) of the ID feature(s).

    time_invariant_features : list of str, default None
        (Optional) Names of time-invariant features.

    attempt_infer_extended_nominals : bool, default False
        (Optional) If set to True, detections of extended nominals will be
        attempted. If the detection fails, the categorical variables will
        be set to `int-id` subtype.

        .. note ::
            Please refer to `kwargs` for other parameters related to
            extended nominals.

    nominal_substitution_config : dict of dicts, default None
        (Optional) Configuration of the nominal substitution engine
        and the nominal generators and detectors.

    include_extended_nominal_probabilities : bool, default False
        (Optional) If true, extended nominal probabilities will be appended
        as metadata into the feature object.

    datetime_feature_formats : dict, default None
        (optional) Dict defining a custom (non-ISO8601) datetime format and
        an optional locale for columns with datetimes.  By default datetime
        columns are assumed to be in ISO8601 format.  Non-English datetimes
        must have locales specified.  If locale is omitted, the default
        system locale is used. The keys are the column name, and the values
        are a tuple of date time format and locale string:

        Example::

            {
                "start_date" : ("%Y-%m-%d %A %H.%M.%S", "es_ES"),
                "end_date" : "%Y-%m-%d"
            }

    ordinal_feature_values : dict, default None
        (optional) Dict for ordinal string features defining an ordered
        list of string values for each feature, ordered low to high. If
        specified will set 'type' to be 'ordinal' for all features in
        this map.

        Example::

            {
                "grade" : [ "F", "D", "C", "B", "A" ],
                "size" : [ "small", "medium", "large", "huge" ]
            }

    dependent_features: dict, default None
        (Optional) Dict of features with their respective lists of features
        that either the feature depends on or are dependent on them. Should
        be used when there are multi-type value features that tightly
        depend on values based on other multi-type value features.

        Examples:
            If there's a feature name 'measurement' that contains
            measurements such as BMI, heart rate and weight, while the
            feature 'measurement_amount' contains the numerical values
            corresponding to the measurement, dependent features could be
            passed in as follows:

            .. code-block:: json

                {
                    "measurement": [ "measurement_amount" ]
                }

            Since dependence directionality is not important, this will
            also work:

            .. code-block:: json

                {
                    "measurement_amount": [ "measurement" ]
                }

    Returns
    -------
    FeatureAttributesBase
        A subclass of FeatureAttributesBase (Single/MultiTableFeatureAttributes)
        that extends `dict`, thus providing dict-like access to feature
        attributes and useful accessor methods.

    Examples
    --------
    .. code-block:: python

        # 'data' is a DataFrame
        >> attrs = infer_feature_attributes(data)
        # Can access feature attributes like a dict
        >> attrs
            {
                "feature_one": {
                    "type": "continuous",
                    "bounds": {"allow_null": True},
                },
                "feature_two": {
                    "type": "nominal",
                }
            }
        >> attrs["feature_one"]
            {
                "type": "continuous",
                "bounds": {"allow_null": True}
            }
        # Or can call methods to do other stuff
        >> attrs.get_parameters()
            {'dropna': True}

        # Now 'data' is an object that implements SQLRelationalDatastoreProtocol
        >> attrs = infer_feature_attributes(data, tables)
        >> attrs
            {
                "table_1": {
                    "feature_one": {
                        "type": "continuous",
                        "bounds": {"allow_null": True},
                    },
                    "feature_two": {
                        "type": "nominal",
                    }
                },
                "table_2" : {...},
            }
        >> attrs.to_json()
            '{"table_1" : {...}}'

    """
    # Check if time series attributes should be calculated
    if time_feature_name and isinstance(data, pd.DataFrame):
        infer = InferFeatureAttributesTimeSeries(data, time_feature_name)
    elif time_feature_name:
        raise ValueError("'time_feature_name' was included, but 'data' must be of type DataFrame "
                         "for time series feature attributes to be calculated.")
    # Else, check data type
    elif isinstance(data, pd.DataFrame):
        infer = InferFeatureAttributesDataFrame(data)
    elif isinstance(data, SQLRelationalDatastoreProtocol):
        if tables is None:
            raise TypeError("'tables' is a required parameter if 'data' implements "
                            "SQLRelationalDatastoreProtocol.")
        infer = InferFeatureAttributesSQLDatastore(data, tables)
    else:
        raise NotImplementedError('Data not recognized as a DataFrame or compatible datastore.')

    return infer(**kwargs)
