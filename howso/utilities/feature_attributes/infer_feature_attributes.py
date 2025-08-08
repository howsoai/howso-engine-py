from __future__ import annotations

from collections.abc import Iterable
import typing as t

import pandas as pd

from .abstract_data import InferFeatureAttributesAbstractData
from .base import FeatureAttributesBase
from .pandas import InferFeatureAttributesDataFrame
from .protocols import IFACompatibleADCProtocol, SQLRelationalDatastoreProtocol, TableNameProtocol
from .relational import InferFeatureAttributesSQLDatastore
from .time_series import IFATimeSeriesADC, IFATimeSeriesPandas


def infer_feature_attributes(data: pd.DataFrame | SQLRelationalDatastoreProtocol, *,
                             tables: t.Optional[Iterable[TableNameProtocol]] = None,
                             time_feature_name: t.Optional[str] = None,
                             **kwargs
                             ) -> FeatureAttributesBase:
    """
    Return a dict-like feature attributes object with useful accessor methods.

    The returned object is a subclass of FeatureAttributesBase that is appropriate for the
    provided data type.

    Parameters
    ----------
    data : Any
        The data source to infer feature attributes from. Must be a supported data type.

    attempt_infer_extended_nominals : bool, default False
        (Optional) If set to True, detections of extended nominals will be
        attempted. If the detection fails, the categorical variables will
        be set to ``int-id`` subtype.

        .. note ::
            Please refer to ``kwargs`` for other parameters related to
            extended nominals.

    datetime_feature_formats : dict, default None
        (Optional) Dict defining custom (non-ISO8601) datetime or time-only formats.
        By default, datetime features are assumed to be in ISO8601 format.  Non-English datetimes
        must have locales specified.  If locale is omitted, the default system locale is used.
        The keys are the feature name, and the values are a tuple of date/time format and locale
        string. Time-only feature formats are expected to adhere to the format codes used in
        strftime().

        Examples::

            {
                "start_date": ("%Y-%m-%d %A %H.%M.%S", "es_ES"),
                "end_date": "%Y-%m-%d",
                "start_time": "%H:%M:%S %p",
            }

    default_time_zone : str, default None
        (Optional) The fallback time zone for any datetime feature if one is not provided in
        ``datetime_feature_formats`` and it is not inferred from the data. If not specified
        anywhere, the Howso Engine will default to UTC.

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

    dependent_features : dict, optional
        Dict mapping a feature to a list of other feature(s) that it depends on or
        that are dependent on it. This restricts the cases that can be selected as
        neighbors (such as in :meth:`~howso.engine.Trainee.react`) to ones that
        satisfy the dependency, if possible. If this is not possible, either due to
        insufficient data which satisfy the dependency or because dependencies are
        probabilistic, the dependency may not be maintained. Be aware that dependencies
        introduce further constraints to data and so several dependencies or dependencies
        on already constrained datasets may restrict which operations are possible while
        maintaining the dependency. As a rule of thumb, sets of features that have
        dependency relationships should generally not include more than 1 continuous feature,
        unless the continuous features have a small number of values that are commonly used.

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

    derived_orders : dict, default None
        (Optional) Dict of features to the number of orders of derivatives
        that should be derived instead of synthesized. For example, for a
        feature with a 3rd order of derivative, setting its derived_orders
        to 2 will synthesize the 3rd order derivative value, and then use
        that synthed value to derive the 2nd and 1st order.

    id_feature_name : str or list of str, default None
        (Optional) The name(s) of the ID feature(s).

    include_extended_nominal_probabilities : bool, default False
        (Optional) If true, extended nominal probabilities will be appended
        as metadata into the feature object.

    include_sample: bool, default False
        If True, include a ``sample`` field containing a sample of the data
        from each feature in the output feature attributes dictionary.

    infer_bounds : bool, default True
        (Optional) If True, bounds will be inferred for the features if the
        feature column has at least one non NaN value

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

    max_workers: int, default None
        If unset or set to None (recommended), let the ProcessPoolExecutor
        choose the best maximum number of process pool workers to process
        columns in a multi-process fashion. In this case, if the product of the
        data's rows and columns > 25,000,000 or if the data is time series and the
        number of rows > 500,000 multiprocessing will be used.

        If defined with an integer > 0, manually set the number of max workers.
        Otherwise, the feature attributes will be calculated serially. Setting
        this parameter to zero (0) will disable multiprocessing.

    memory_warning_threshold : int, default 512
        (Optional) Maximum number of bytes that a feature's per-case average can compute to
        without raising a warning about memory usage (Pandas DataFrame only).

    mode_bound_features : list of str, default None
        (Optional) Explicit list of feature names to use mode bounds for
        when inferring loose bounds. If None, assumes all features. A mode
        bound is used instead of a loose bound when the mode for the
        feature is the same as an original bound, as it may represent an
        application-specific min/max.

    nominal_substitution_config : dict of dicts, default None
        (Optional) Configuration of the nominal substitution engine
        and the nominal generators and detectors.

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

    tables : Iterable of TableNameProtocol
        (Optional, required for datastores) An Iterable of table names to
        infer feature attributes for.

        If included, feature attributes will be generated in the form
        ``{table_name: {feature_attribute: value}}``.

    tight_bounds: Iterable of str, default None
        (Optional) Set tight min and max bounds for the features
        specified in the Iterable.

    time_feature_is_universal : bool, optional
        If True, the time feature will be treated as universal and future data
        is excluded while making predictions. If False, the time feature will
        not be treated as universal and only future data within the same series
        is excluded while making predictions. It is recommended to set this
        value to True if there is any possibility of global relevancy of time,
        which is the default behavior.

    time_feature_name : str, default None
        (Optional, required for time series) The name of the time feature.

    time_invariant_features : list of str, default None
        (Optional) Names of time-invariant features.

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
        one of 'rate' or 'delta', used to override ``time_series_type_default``
        for the specified features.

    types: dict, default None
        (Optional) Dict of features and their intended type (i.e., "nominal,"
        "ordinal," or "continuous"), or types mapped to MutableSequences of
        feature names. Any types provided here will override the types that would
        otherwise be inferred, and will direct ``infer_feature_attributes`` to
        compute the attributes accordingly.

        Example::

            {
                "feature_1": "nominal",
                "feature_2": "ordinal",
                "continuous": ["feature_3", "feature_4", "feature_5"]
            }

    Returns
    -------
    FeatureAttributesBase
        A subclass of ``FeatureAttributesBase`` (Single/MultiTableFeatureAttributes)
        that extends ``dict``, thus providing dict-like access to feature
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
            {'type': "continuous"}

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
    if time_feature_name:
        if isinstance(data, pd.DataFrame):
            infer = IFATimeSeriesPandas(data, time_feature_name)
        elif isinstance(data, IFACompatibleADCProtocol):
            infer = IFATimeSeriesADC(data, time_feature_name)
        else:
            raise NotImplementedError('`infer_feature_attributes` for time series only supported for DataFrames and '
                                      'AbstractData classes.')
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
    elif isinstance(data, IFACompatibleADCProtocol):
        infer = InferFeatureAttributesAbstractData(data)
    else:
        raise NotImplementedError('Data not recognized as a DataFrame, AbstractData class, or compatible datastore.')

    return infer(**kwargs)
