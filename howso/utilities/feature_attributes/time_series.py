from __future__ import annotations

from abc import abstractmethod
from collections.abc import Collection, Iterable
from concurrent.futures import (
    as_completed,
    Future,
    ProcessPoolExecutor,
)
import copy
import logging
from math import e
import multiprocessing as mp
import os
import typing as t
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype

from .abstract_data import InferFeatureAttributesAbstractData
from .base import SingleTableFeatureAttributes
from .pandas import InferFeatureAttributesDataFrame
from .protocols import IFACompatibleADCProtocol
from ..utilities import date_to_epoch, is_valid_datetime_format, yield_dataframe_as_chunks

logger = logging.getLogger(__name__)

SMALLEST_TIME_DELTA = 0.001


def _apply_date_to_epoch(df: pd.DataFrame, feature_name: str, dt_format: str):
    """Internal function to aid multiprocessing of series feature attributes."""
    return df[feature_name].apply(lambda x: date_to_epoch(x, dt_format))


class InferFeatureAttributesTimeSeries:
    """Infer feature attributes for time series data."""

    def __init__(self, data: pd.DataFrame | IFACompatibleADCProtocol, time_feature_name: str):
        """Instantiate this InferFeatureAttributesTimeSeries object."""
        self.data = data
        self.time_feature_name = time_feature_name
        # Keep track of features that contain unsupported data
        self.unsupported = []

    def _infer_delta_min_max_from_chunk(  # noqa: C901
        self,
        chunk: pd.DataFrame,
        features: dict,
        *,
        datetime_feature_formats: t.Optional[dict] = None,
        derived_orders: t.Optional[dict] = None,
        id_feature_name: t.Optional[str | Iterable[str]] = None,
        max_workers: t.Optional[int] = None,
        orders_of_derivatives: t.Optional[dict] = None,
    ) -> dict:
        """
        Infer rate and delta min/max for each continuous feature and update the features dict.

        This method will not introspect `self.data` but rather the provided DataFrame
        in order to maintain compatibility with chunk scaling.

        Parameters
        ----------
        chunk : pd.DataFrame
            The data to infer delta min and max from.

        features : dict, default None
            (Optional) A partially filled features dict. If partially filled
            attributes for a feature are passed in, those parameters will be
            retained as is and the delta_min and delta_max attributes will be
            inferred.

        datetime_feature_formats : dict, default None
            (Optional) Dict defining a custom (non-ISO8601) datetime format and
            an optional locale for features with datetimes.  By default
            datetime features are assumed to be in ISO8601 format.  Non-English
            datetimes must have locales specified.  If locale is omitted, the
            default system locale is used. The keys are the feature name, and
            the values are a tuple of date time format and locale string:

            Examples::

                {
                    "start_date" : ("%Y-%m-%d %A %H.%M.%S", "es_ES"),
                    "end_date" : "%Y-%m-%d"
                }

        id_feature_name : str or list of str, default None
            (Optional) The name(s) of the ID feature(s).

        orders_of_derivatives : dict, default None
            (Optional) Dict of features and their corresponding order of
            derivatives for the specified type (delta/rate). If provided will
            generate the specified number of derivatives and boundary values. If
            set to 0, will not generate any delta/rate features. By default all
            continuous features have an order value of 1.

        derived_orders : dict, default None
            (Optional) Dict of features to the number of orders of derivatives
            that should be derived instead of synthesized. For example, for a
            feature with a 3rd order of derivative, setting its derived_orders
            to 2 will synthesize the 3rd order derivative value, and then use
            that synthed value to derive the 2nd and 1st order.

        max_workers: int, optional
            If unset or set to None (recommended), let the ProcessPoolExecutor
            choose the best maximum number of process pool workers to process
            columns in a multi-process fashion. In this case, if the product of the
            data's rows and columns > 25,000,000 or the number of rows > 500,000
            multiprocessing will used.

            If defined with an integer > 0, manually set the number of max workers.
            Otherwise, the feature attributes will be calculated serially. Setting
            this parameter to zero (0) will disable multiprocessing.

        Returns
        -------
        features : dict
            Returns an updated feature attributes dictionary with inferred time series
            information added under an additional ``time_series`` attribute for each
            applicable feature.
        """
        # prevent circular import
        from howso.client.exceptions import DatetimeFormatWarning
        # Make a copy of the 'features' dict as it will be returned; don't modify original
        features = copy.deepcopy(features)
        # iterate over all features, ensuring that the time feature is the first
        # one to be processed so that its deltas are cached
        feature_names = set(features.keys())
        feature_names.remove(self.time_feature_name)
        feature_names = [self.time_feature_name] + list(feature_names)

        if orders_of_derivatives is None:
            orders_of_derivatives = dict()

        if derived_orders is None:
            derived_orders = dict()

        time_feature_deltas = None
        for f_name in feature_names:
            if features[f_name]['type'] == "continuous" and 'time_series' in features[f_name]:
                # Set delta_max for all continuous features to the observed maximum
                # difference between two values times e.
                dt_format = features[f_name].get('date_time_format')
                if dt_format is None and isinstance(datetime_feature_formats, dict):
                    dt_format = datetime_feature_formats.get(f_name)
                    if (
                        not isinstance(dt_format, str) and
                        isinstance(dt_format, Collection) and
                        len(dt_format) == 2
                    ):
                        dt_format, _ = dt_format  # (format, locale)

                # number of derivation orders, default to 1
                num_orders = orders_of_derivatives.get(f_name, 1)
                if num_orders > 1:
                    features[f_name]['time_series']['order'] = num_orders

                num_derived_orders = derived_orders.get(f_name, 0)
                if num_derived_orders >= num_orders:
                    old_derived_orders = num_derived_orders
                    num_derived_orders = num_orders - 1
                    warnings.warn(
                        f'Overwriting the `derived_orders` value of {old_derived_orders} '
                        f'for "{f_name}" with {num_derived_orders} because it must '
                        f'be smaller than the "orders" value of {num_orders}.',
                    )
                if num_derived_orders > 0:
                    features[f_name]['time_series']['derived_orders'] = num_derived_orders

                if dt_format is not None:
                    # copy just the id columns and the time feature
                    if isinstance(id_feature_name, str):
                        df_c = chunk.loc[:, [id_feature_name, f_name]]
                    elif isinstance(id_feature_name, list):
                        df_c = chunk.loc[:, id_feature_name + [f_name]]
                    else:
                        df_c = chunk.loc[:, [f_name]]

                    # convert time feature to epoch
                    if len(df_c) < 500_000 and max_workers is None:
                        max_workers = 0
                    if max_workers is None or max_workers >= 1:
                        if max_workers is None:
                            max_workers = os.cpu_count() or 1
                        mp_context = mp.get_context("spawn")
                        futures: dict[Future, str] = dict()

                        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
                            df_chunks_generator = yield_dataframe_as_chunks(df_c, max_workers)
                            for sub_chunk in df_chunks_generator:
                                future = pool.submit(
                                    _apply_date_to_epoch,
                                    df=sub_chunk,
                                    feature_name=f_name,
                                    dt_format=dt_format
                                )
                                futures[future] = f_name

                            temp_results = []
                            try:
                                for future in as_completed(futures):
                                    response = future.result()
                                    temp_results.append(response)
                            except ValueError:
                                # Cannot calculate deltas if date format is invalid, warn and continue
                                if f_name == self.time_feature_name:
                                    raise ValueError(
                                        f'The date time format "{dt_format}" does not match the data of the '
                                        f'time feature "{self.time_feature_name}".'
                                    )
                                warnings.warn(
                                    f'Feature "{f_name}" does not match the '
                                    f'provided date time format, unable to infer '
                                    f'time series delta min/max.',
                                    DatetimeFormatWarning
                                )
                                for future in futures:
                                    if not future.done():
                                        future.cancel()
                                continue

                        df_c[f_name] = pd.concat(temp_results)
                    else:
                        try:
                            df_c[f_name] = _apply_date_to_epoch(df_c, f_name, dt_format)
                        except ValueError:
                            # Cannot calculate deltas if date format is invalid, warn and continue
                            if f_name == self.time_feature_name:
                                raise ValueError(
                                    f'The date time format "{dt_format}" does not match the data of the '
                                    f'time feature "{self.time_feature_name}".'
                                )
                            warnings.warn(
                                f'Feature "{f_name}" does not match the '
                                f'provided date time format, unable to infer '
                                f'time series delta min/max.',
                                DatetimeFormatWarning
                            )
                            continue

                    # use Pandas' diff() to pull all the deltas for this feature
                    if isinstance(id_feature_name, list):
                        deltas = df_c.groupby(id_feature_name)[f_name].diff(1)
                    elif isinstance(id_feature_name, str):
                        deltas = df_c.groupby([id_feature_name])[f_name].diff(1)
                    else:
                        deltas = df_c[f_name].diff(1)

                else:
                    # Use pandas' diff() to pull all the deltas for this feature
                    if isinstance(id_feature_name, list):
                        deltas = chunk.groupby(id_feature_name)[f_name].diff(1)
                    elif isinstance(id_feature_name, str):
                        deltas = chunk.groupby([id_feature_name])[f_name].diff(1)
                    else:
                        deltas = chunk[f_name].diff(1)

                if f_name == self.time_feature_name:
                    time_feature_deltas = deltas

                # initial rates are same as deltas which will then be used as input
                # to compute actual first order rates
                rates = deltas

                for order in range(1, num_orders + 1):

                    # compute rate min and max for all rate features
                    if features[f_name]['time_series']['type'] == "rate":

                        if 'rate_max' not in features[f_name]['time_series']:
                            features[f_name]['time_series']['rate_max'] = []
                        if 'rate_min' not in features[f_name]['time_series']:
                            features[f_name]['time_series']['rate_min'] = []

                        # compute the deltas between previous rates as inputs
                        # for higher order rate computations
                        if order > 1:
                            rates = np.diff(np.array(rates))

                        # compute each 1st order rate as: delta x / delta time
                        # higher order rates as: delta previous rate / delta time
                        rates = [
                            dx / (dt if dt != 0 else SMALLEST_TIME_DELTA)
                            for dx, dt in zip(rates, time_feature_deltas)
                        ]

                        # remove NaNs
                        no_nan_rates = [x for x in rates if pd.isna(x) is False]
                        if len(no_nan_rates) == 0:
                            continue

                        # TODO: 15550: support user-specified min/max values
                        rate_max = max(no_nan_rates)
                        rate_max = rate_max * e if rate_max > 0 else rate_max / e
                        features[f_name]['time_series']['rate_max'].append(rate_max)

                        rate_min = min(no_nan_rates)
                        rate_min = rate_min / e if rate_min > 0 else rate_min * e
                        features[f_name]['time_series']['rate_min'].append(rate_min)
                    else:  # 'type' == "delta"

                        if 'delta_max' not in features[f_name]['time_series']:
                            features[f_name]['time_series']['delta_max'] = []
                        if 'delta_min' not in features[f_name]['time_series']:
                            features[f_name]['time_series']['delta_min'] = []

                        # compute new deltas between previous deltas as inputs
                        # for higher order delta computations
                        if order > 1:
                            deltas = deltas.diff(1)

                        no_nan_deltas: pd.Series = deltas.dropna()
                        if len(no_nan_deltas) == 0:
                            continue
                        delta_max = max(no_nan_deltas)
                        delta_max = delta_max * e if delta_max > 0 else delta_max / e
                        features[f_name]['time_series']['delta_max'].append(delta_max)

                        delta_min = min(no_nan_deltas)
                        # don't allow the time series time feature to go back in time
                        # TODO: 15550: support user-specified min/max values
                        if f_name == self.time_feature_name:
                            features[f_name]['time_series']['delta_min'].append(max(0, delta_min / e))
                        else:
                            delta_min = delta_min / e if delta_min > 0 else delta_min * e
                            features[f_name]['time_series']['delta_min'].append(delta_min)

        return features

    def _set_rate_delta_bounds(self, btype: str, bounds: dict, features: dict):
        """Set optimally-specified rate/delta bounds in the features dict."""
        for feature in bounds.keys():
            # Check for any problems
            if feature not in features.keys():
                raise ValueError(f"Unknown feature '{feature}' in {btype}_boundaries")
            elif features[feature]['time_series']['type'] != btype:
                warnings.warn(f"Ignoring {btype}_boundaries: feature type is not '{btype}'")
                continue
            # Set specified values
            if 'min' in bounds[feature] and f'{btype}_min' in features[feature]['time_series']:
                num_orders = len(features[feature]['time_series'][f'{btype}_min'])
                for order in bounds[feature]['min'].keys():
                    # Only adjust in-range values; ignore any others
                    if int(order) < num_orders:
                        features[feature]['time_series'][f'{btype}_min'][int(order)] = bounds[feature]['min'][order]
                    else:
                        warnings.warn(f"Ignoring {btype}_boundaries for order {order}: out of range")
            if 'max' in bounds[feature] and f'{btype}_max' in features[feature]['time_series']:
                num_orders = len(features[feature]['time_series'][f'{btype}_max'])
                for order in bounds[feature]['max'].keys():
                    if int(order) < num_orders:
                        features[feature]['time_series'][f'{btype}_max'][int(order)] = bounds[feature]['max'][order]
                    else:
                        warnings.warn(f"Ignoring {btype}_boundaries for order {order}: out of range")

    def _process(  # noqa: C901
        self,
        attempt_infer_extended_nominals: bool = False,
        datetime_feature_formats: t.Optional[dict] = None,
        default_time_zone: t.Optional[str] = None,
        delta_boundaries: t.Optional[dict] = None,
        dependent_features: t.Optional[dict] = None,
        derived_orders: t.Optional[dict] = None,
        id_feature_name: t.Optional[str | Iterable[str]] = None,
        include_extended_nominal_probabilities: t.Optional[bool] = False,
        include_sample: bool = False,
        infer_bounds: bool = True,
        lags: t.Optional[list | dict] = None,
        max_workers: t.Optional[int] = None,
        memory_warning_threshold: t.Optional[int] = 512,
        mode_bound_features: t.Optional[Iterable[str]] = None,
        nominal_substitution_config: t.Optional[dict[str, dict]] = None,
        num_lags: t.Optional[int | dict] = None,
        orders_of_derivatives: t.Optional[dict] = None,
        ordinal_feature_values: t.Optional[dict[str, list[str]]] = None,
        rate_boundaries: t.Optional[dict] = None,
        time_invariant_features: t.Optional[Iterable[str]] = None,
        tight_bounds: t.Optional[Iterable[str]] = None,
        time_feature_is_universal: t.Optional[bool] = None,
        time_series_type_default: t.Optional[str] = 'rate',
        time_series_types_override: t.Optional[dict] = None,
        types: t.Optional[dict[str, str] | dict[str, t.MutableSequence[str]]] = None,
    ) -> dict:
        """
        Infer time series attributes.

        Infers the feature types for each of the features in the dataframe.
        If the feature is of float, it is assumed that the feature is of type
        continuous. And any other type will be assumed to be nominal.

        Parameters
        ----------
        attempt_infer_extended_nominals : bool, default False
            (Optional) If set to True, detections of extended nominals will be
            attempted. If the detection fails, the categorical variables will
            be set to `int-id` subtype.

            .. note ::
                Please refer to `kwargs` for other parameters related to
                extended nominals.

        datetime_feature_formats : dict, default None
            (Optional) Dict defining a custom (non-ISO8601) datetime format and
            an optional locale for features with datetimes.  By default datetime
            features are assumed to be in ISO8601 format.  Non-English datetimes
            must have locales specified.  If locale is omitted, the default
            system locale is used. The keys are the feature name, and the values
            are a tuple of date time format and locale string:

            Examples::

                {
                    "start_date" : ("%Y-%m-%d %A %H.%M.%S", "es_ES"),
                    "end_date" : "%Y-%m-%d"
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

        id_feature_name : str or list of str default None
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
            1, 3, and 5 time steps behind the current time step respectively.

            .. note ::
                Using the lags parameter will override the num_lags parameter per
                feature

            .. note ::
                A lag feature is a feature that provides a "lagging value" to a
                case by holding the value of a feature from a previous time step.
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
            when inferring loose bounds. If None, assumes all features except the
            time feature. A mode bound is used instead of a loose bound when the
            mode for the feature is the same as an original bound, as it may
            represent an application-specific min/max.

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

        tight_bounds: Iterable of str, default None
            (Optional) Set tight min and max bounds for the features
            specified in the Iterable.

        time_invariant_features : list of str, default None
            (Optional) Names of time-invariant features.

        time_feature_is_universal : bool, optional
            If True, the time feature will be treated as universal and future data
            is excluded while making predictions. If False, the time feature will
            not be treated as universal and only future data within the same series
            is excluded while making predictions. It is recommended to set this
            value to True if there is any possibility of global relevancy of time,
            which is the default behavior.

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
            A subclass of FeatureAttributesBase that extends `dict`, thus providing
            dict-like access to feature attributes and useful helper methods.
        """
        if isinstance(self.data, IFACompatibleADCProtocol):
            infer = InferFeatureAttributesAbstractData(self.data)
        elif isinstance(self.data, pd.DataFrame):
            infer = InferFeatureAttributesDataFrame(self.data)
        else:
            raise ValueError('Cannot process data: unsupported type {type(self.data)} for time-series.')

        if mode_bound_features is None:
            feature_names = infer._get_feature_names()
            mode_bound_features = [
                f for f in feature_names if f != self.time_feature_name
            ]

        if isinstance(id_feature_name, str):
            id_feature_names = [id_feature_name]
        elif isinstance(id_feature_name, Iterable):
            id_feature_names = id_feature_name
        else:
            id_feature_names = []

        num_series = infer._get_unique_count(id_feature_names) if id_feature_names else 1

        features = infer(
            attempt_infer_extended_nominals=attempt_infer_extended_nominals,
            datetime_feature_formats=datetime_feature_formats,
            default_time_zone=default_time_zone,
            dependent_features=dependent_features,
            id_feature_name=id_feature_name,
            include_extended_nominal_probabilities=include_extended_nominal_probabilities,
            include_sample=include_sample,
            infer_bounds=infer_bounds,
            max_workers=max_workers,
            memory_warning_threshold=memory_warning_threshold,
            mode_bound_features=mode_bound_features,
            nominal_substitution_config=nominal_substitution_config,
            num_series=num_series,
            ordinal_feature_values=ordinal_feature_values,
            tight_bounds=set(tight_bounds) if tight_bounds else None,
            time_invariant_features=time_invariant_features,
            types=types,
        )

        # Add any features with unsupported data to this object's list
        self.unsupported.extend(features.unsupported)

        if isinstance(time_invariant_features, str):
            time_invariant_features = [time_invariant_features]
        elif isinstance(time_invariant_features, Iterable):
            time_invariant_features = list(time_invariant_features)
        else:
            time_invariant_features = []

        # ID features are time-invariant.
        for id_feature in id_feature_names:
            if id_feature not in time_invariant_features:
                time_invariant_features.append(id_feature)

        if self.time_feature_name in time_invariant_features:
            raise ValueError('time_feature_name cannot be in the '
                             'time_invariant_features list.')

        # Set all non time invariant features to be `time_series` features
        for f_name, _ in features.items():
            # Mark all features which are completely NaN as time-invariant.
            if self._is_null_column(f_name):
                time_invariant_features.append(f_name)

            if f_name not in time_invariant_features:
                if time_series_types_override and f_name in time_series_types_override:
                    features[f_name]['time_series'] = {
                        'type': time_series_types_override[f_name]
                    }
                else:
                    features[f_name]['time_series'] = {
                        'type': time_series_type_default,
                    }

        if num_lags is not None:
            if isinstance(num_lags, int):
                for f_name, _ in features.items():
                    if f_name != self.time_feature_name and 'time_series' in features[f_name]:
                        features[f_name]['time_series']['num_lags'] = int(num_lags)
            elif isinstance(num_lags, dict):
                for f_name, f_lags in num_lags.items():
                    if 'time_series' in features[f_name]:
                        features[f_name]['time_series']['num_lags'] = int(f_lags)
        if lags is not None:
            if isinstance(lags, list):
                for f_name, _ in features.items():
                    if f_name != self.time_feature_name and 'time_series' in features[f_name]:
                        if 'num_lags' in features[f_name]['time_series']:
                            del features[f_name]['time_series']['num_lags']
                        features[f_name]['time_series']['lags'] = lags
            elif isinstance(lags, dict):
                for f_name, f_lags in lags.items():
                    # If lag_list is specified, lags is not used
                    if 'num_lags' in features[f_name]['time_series']:
                        del features[f_name]['time_series']['num_lags']
                    if isinstance(f_lags, int):
                        f_lags = [f_lags]
                    elif not isinstance(f_lags, list):
                        raise TypeError(f'Unsupported type for {f_name} lags value (must be list)')
                    features[f_name]['time_series']['lags'] = f_lags

        if self.time_feature_name in features:
            features[self.time_feature_name]['time_series']['time_feature'] = True

            # Assign universal value if specified
            if time_feature_is_universal is not None:
                features[self.time_feature_name]['time_series']['universal'] = time_feature_is_universal
            # Force time_feature to be `continuous`
            features[self.time_feature_name]['type'] = "continuous"
            # Set time_series as 'delta' so that lag and delta are computed
            features[self.time_feature_name]['time_series']['type'] = "delta"
            # Time feature might have `sensitive` and `subtype` attribute
            # which is not applicable to time feature.
            features[self.time_feature_name].pop('subtype', None)

            time_feature_dtype = self._get_column_dtype(self.time_feature_name)

            # If a datetime format is defined, ensure values can be parsed with it
            if dt_format := features[self.time_feature_name].get("date_time_format"):
                test_value = infer._get_random_value(self.time_feature_name, no_nulls=True)
                if test_value is not None and not is_valid_datetime_format(test_value, dt_format):
                    raise ValueError(
                        f'The date time format "{dt_format}" does not match the data of the time feature '
                        f'"{self.time_feature_name}". Data sample: "{test_value}"')

            elif is_string_dtype(time_feature_dtype):
                # if the time feature has no datetime format and is stored as a string,
                # convert to a float for comparison since it's continuous
                self._cast_column(self.time_feature_name, float)

            # time feature cannot be null
            if 'bounds' in features[self.time_feature_name]:
                features[self.time_feature_name]['bounds']['allow_null'] = False
            else:
                features[self.time_feature_name]['bounds'] = {'allow_null': False}

        features = self._infer_delta_min_max(
            features=features,
            datetime_feature_formats=datetime_feature_formats,
            id_feature_name=id_feature_name,
            orders_of_derivatives=orders_of_derivatives,
            derived_orders=derived_orders,
            max_workers=max_workers
        )

        # Set any manually specified rate/delta boundaries
        if delta_boundaries is not None:
            self._set_rate_delta_bounds('delta', delta_boundaries, features)
        if rate_boundaries is not None:
            self._set_rate_delta_bounds('rate', rate_boundaries, features)

        return features

    def __call__(self, **kwargs) -> SingleTableFeatureAttributes:
        """Process and return feature attributes."""
        feature_attributes = self._process(**kwargs)
        # Put the time_feature_name back into the kwargs dictionary.
        kwargs["time_feature_name"] = self.time_feature_name
        return SingleTableFeatureAttributes(feature_attributes, params=kwargs, unsupported=self.unsupported)

    @abstractmethod
    def _is_null_column(self, feature_name: str) -> bool:
        """Determine whether the provided column is all null values."""
        raise NotImplementedError()

    @abstractmethod
    def _cast_column(self, feature_name: str, new_type: t.Any):
        """Convert the column of the provided ``feature_name`` to a different dtype."""
        raise NotImplementedError()

    @abstractmethod
    def _get_column_dtype(self, feature_name: str) -> np.dtype:
        """Get the dtype of the provided ``feature_name``."""
        raise NotImplementedError()


class IFATimeSeriesPandas(InferFeatureAttributesTimeSeries):
    """InferFeatureAttributesTimeSeries implementation for Pands DataFrames."""

    def _infer_delta_min_max(
        self,
        features: dict,
        datetime_feature_formats: t.Optional[dict] = None,
        id_feature_name: t.Optional[str | Iterable[str]] = None,
        orders_of_derivatives: t.Optional[dict] = None,
        derived_orders: t.Optional[dict] = None,
        max_workers: t.Optional[int] = None
    ):
        """Infer delta and rate min/max for each continuous feature and update the features dict."""
        return self._infer_delta_min_max_from_chunk(
            self.data,
            features,
            datetime_feature_formats=datetime_feature_formats,
            id_feature_name=id_feature_name,
            orders_of_derivatives=orders_of_derivatives,
            derived_orders=derived_orders,
            max_workers=max_workers,
        )

    def _cast_column(self, feature_name: str, new_type: t.Any):
        """Convert the column of the provided ``feature_name`` to a different dtype."""
        self.data[self.time_feature_name] = self.data[self.time_feature_name].astype(new_type)

    def _is_null_column(self, feature_name: str) -> bool:
        """Determine whether the provided column is all null values."""
        return self.data[feature_name].isnull().sum() == len(self.data[feature_name])

    def _get_column_dtype(self, feature_name: str) -> np.dtype:
        """Get the dtype of the provided ``feature_name``."""
        return self.data[self.time_feature_name].dtype


class IFATimeSeriesADC(InferFeatureAttributesTimeSeries):
    """InferFeatureAttributesTimeSeries implementation for AbstractData classes."""

    @staticmethod
    def _compute_time_series_min_max(
        chunk_features: dict,
        features: dict,
        f_name: str,
        key: str,
        op: t.Callable,
    ):
        """
        Return the min or max value between two competing time series feature attributes.

        Parameters
        ----------
        chunk_features : dict
            The prospective feature attributes.
        features : dict
            The existing feature attributes.
        f_name : str
            The feature to compare.
        key : str
            The key under the ``time_series`` attribute to compare.
        op : Callable
            The operation (min() or max()) to use.

        Returns
        -------
        The value returned by op(a, b), or the prospective feature attributes' value if none currently exist,
        or None if neither exist.
        """
        current_val = features[f_name].get('time_series', {}).get(key)
        possible_val = chunk_features[f_name].get('time_series', {}).get(key)
        if current_val and possible_val:
            return op(current_val, possible_val)
        elif possible_val and not current_val:
            return possible_val
        else:
            return None

    def _infer_delta_min_max(
        self,
        features: dict,
        datetime_feature_formats: t.Optional[dict] = None,
        id_feature_name: t.Optional[str | Iterable[str]] = None,
        orders_of_derivatives: t.Optional[dict] = None,
        derived_orders: t.Optional[dict] = None,
        max_workers: t.Optional[int] = None
    ):
        """Infer delta and rate min/max for each continuous feature and update the features dict."""
        feature_chunks = []
        for chunk in self.data.yield_chunk():
            feature_chunks.append(self._infer_delta_min_max_from_chunk(
                chunk,
                features,
                datetime_feature_formats=datetime_feature_formats,
                id_feature_name=id_feature_name,
                orders_of_derivatives=orders_of_derivatives,
                derived_orders=derived_orders,
                max_workers=max_workers,
            ))
        # Recompute min/max across all chunks
        for f_name in features.keys():
            for chunk in feature_chunks:
                rate_min = self._compute_time_series_min_max(chunk, features, f_name, 'rate_min', min)
                rate_max = self._compute_time_series_min_max(chunk, features, f_name, 'rate_max', min)
                delta_min = self._compute_time_series_min_max(chunk, features, f_name, 'delta_min', min)
                delta_max = self._compute_time_series_min_max(chunk, features, f_name, 'delta_max', min)
                if any([rate_min, rate_max, delta_min, delta_max]) and 'time_series' not in features[f_name]:
                    features[f_name]['time_series'] = {}
                if rate_min:
                    features[f_name]['time_series']['rate_min'] = rate_min
                if rate_max:
                    features[f_name]['time_series']['rate_max'] = rate_max
                if delta_min:
                    features[f_name]['time_series']['delta_min'] = delta_min
                if delta_min:
                    features[f_name]['time_series']['delta_max'] = delta_max
        return features

    def _cast_column(self, feature_name: str, new_type: t.Any):
        """Convert the column of the provided ``feature_name`` to a different dtype."""
        raise ValueError('Invalid time feature data type: must be numeric or datetime.')

    def _is_null_column(self, feature_name: str) -> bool:
        """Determine whether the provided column is all null values."""
        return self.data.get_num_cases == 0

    def _get_column_dtype(self, feature_name: str) -> np.dtype:
        """Get the dtype of the provided ``feature_name``."""
        dtype = self.data.get_column_dtype(feature_name)
        self.data
        if isinstance(dtype, str):
            try:
                dtype = np.dtype(dtype)
            except Exception:
                # Leave as-is
                pass
        return dtype
