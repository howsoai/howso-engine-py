from __future__ import annotations

from collections.abc import Collection, Iterable
from concurrent.futures import (
    as_completed,
    Future,
    ProcessPoolExecutor,
)
import logging
from math import e, isnan
import multiprocessing as mp
import os
import typing as t
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype

from .base import SingleTableFeatureAttributes
from .pandas import InferFeatureAttributesDataFrame
from .models import InferFeatureAttributesArgs
from ..utilities import date_to_epoch, is_valid_datetime_format, yield_dataframe_as_chunks

logger = logging.getLogger(__name__)

SMALLEST_TIME_DELTA = 0.001


def _apply_date_to_epoch(df: pd.DataFrame, feature_name: str, dt_format: str):
    """Internal function to aid multiprocessing of series feature attributes."""
    return df[feature_name].apply(lambda x: date_to_epoch(x, dt_format))


class InferFeatureAttributesTimeSeries:
    """Infer feature attributes for time series data."""

    def __init__(self, data: pd.DataFrame, time_feature_name: str):
        """Instantiate this InferFeatureAttributesTimeSeries object."""
        self.data = data
        self.time_feature_name = time_feature_name
        # Keep track of features that contain unsupported data
        self.unsupported = []

    def _infer_delta_min_and_max(  # noqa: C901
        self,
        features: dict,
        datetime_feature_formats: t.Optional[dict] = None,
        id_feature_name: t.Optional[str | Iterable[str]] = None,
        orders_of_derivatives: t.Optional[dict] = None,
        derived_orders: t.Optional[dict] = None,
        max_workers: t.Optional[int] = None
    ) -> dict:
        """
        Infer continuous feature delta_min, delta_max for each feature.

        Parameters
        ----------
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
            Returns dictionary of {`type`: "feature type"}} with column names
            in passed in df as key.
        """
        # prevent circular import
        from howso.client.exceptions import DatetimeFormatWarning
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
                        df_c = self.data.loc[:, [id_feature_name, f_name]]
                    elif isinstance(id_feature_name, list):
                        df_c = self.data.loc[:, id_feature_name + [f_name]]
                    else:
                        df_c = self.data.loc[:, [f_name]]

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
                            for chunk in df_chunks_generator:
                                future = pool.submit(
                                    _apply_date_to_epoch,
                                    df=chunk,
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
                        deltas = self.data.groupby(id_feature_name)[f_name].diff(1)
                    elif isinstance(id_feature_name, str):
                        deltas = self.data.groupby([id_feature_name])[f_name].diff(1)
                    else:
                        deltas = self.data[f_name].diff(1)

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
                        no_nan_rates = [x for x in rates if isnan(x) is False]
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
        args: t.Optional[InferFeatureAttributesArgs] = None,
        **kwargs
    ) -> dict:
        """
        Infer time series attributes.

        Infers the feature types for each of the features in the dataframe.
        If the feature is of float, it is assumed that the feature is of type
        continuous. And any other type will be assumed to be nominal.

        Parameters
        ----------
            args : InferFeatureAttributesArgs
                (Optional) Defines behaviors of infer_feature_attributes processing.

        Returns
        -------
        FeatureAttributesBase
            A subclass of FeatureAttributesBase that extends `dict`, thus providing
            dict-like access to feature attributes and useful helper methods.
        """
        _args = args or InferFeatureAttributesArgs()
        merged_args = _args.merge_kwargs(**kwargs)

        infer = InferFeatureAttributesDataFrame(self.data)

        if merged_args.mode_bound_features is None:
            feature_names = infer._get_feature_names()
            merged_args.mode_bound_features = [
                f for f in feature_names if f != self.time_feature_name
            ]

        features = infer(args=merged_args)

        # Add any features with unsupported data to this object's list
        self.unsupported.extend(features.unsupported)

        if isinstance(time_invariant_features, str):
            time_invariant_features = [time_invariant_features]
        elif isinstance(time_invariant_features, Iterable):
            time_invariant_features = list(time_invariant_features)
        else:
            time_invariant_features = []

        if isinstance(merged_args.id_feature_name, str):
            id_feature_names = [merged_args.id_feature_name]
        elif isinstance(merged_args.id_feature_name, Iterable):
            id_feature_names = merged_args.id_feature_name
        else:
            id_feature_names = []

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
            if self.data[f_name].isnull().sum() == len(self.data[f_name]):
                time_invariant_features.append(f_name)

            if f_name not in time_invariant_features:
                if merged_args.time_series_types_override and f_name in merged_args.time_series_types_override:
                    features[f_name]['time_series'] = {
                        'type': merged_args.time_series_types_override[f_name]
                    }
                else:
                    features[f_name]['time_series'] = {
                        'type': merged_args.time_series_type_default,
                    }

        if merged_args.num_lags is not None:
            if isinstance(merged_args.num_lags, int):
                for f_name, _ in features.items():
                    if f_name != self.time_feature_name and 'time_series' in features[f_name]:
                        features[f_name]['time_series']['num_lags'] = int(merged_args.num_lags)
            elif isinstance(merged_args.num_lags, dict):
                for f_name, f_lags in merged_args.num_lags.items():
                    if 'time_series' in features[f_name]:
                        features[f_name]['time_series']['num_lags'] = int(f_lags)
        if merged_args.lags is not None:
            if isinstance(merged_args.lags, list):
                for f_name, _ in features.items():
                    if f_name != self.time_feature_name and 'time_series' in features[f_name]:
                        if 'num_lags' in features[f_name]['time_series']:
                            del features[f_name]['time_series']['num_lags']
                        features[f_name]['time_series']['lags'] = merged_args.lags
            elif isinstance(merged_args.lags, dict):
                for f_name, f_lags in merged_args.lags.items():
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
            if merged_args.time_feature_is_universal is not None:
                features[self.time_feature_name]['time_series']['universal'] = merged_args.time_feature_is_universal
            # Force time_feature to be `continuous`
            features[self.time_feature_name]['type'] = "continuous"
            # Set time_series as 'delta' so that lag and delta are computed
            features[self.time_feature_name]['time_series']['type'] = "delta"
            # Time feature might have `sensitive` and `subtype` attribute
            # which is not applicable to time feature.
            features[self.time_feature_name].pop('subtype', None)

            if dt_format := features[self.time_feature_name].get("date_time_format"):
                # if a datetime format is defined, ensure values can be parsed with it
                test_value = infer._get_random_value(self.time_feature_name, no_nulls=True)
                if test_value is not None and not is_valid_datetime_format(test_value, dt_format):
                    raise ValueError(
                        f'The date time format "{dt_format}" does not match the data of the time feature '
                        f'"{self.time_feature_name}". Data sample: "{test_value}"')
            elif is_string_dtype(self.data[self.time_feature_name].dtype):
                # if the time feature has no datetime format and is stored as a string,
                # convert to an int for comparison since it's continuous
                self.data[self.time_feature_name] = self.data[self.time_feature_name].astype(float)

            # time feature cannot be null
            if 'bounds' in features[self.time_feature_name]:
                features[self.time_feature_name]['bounds']['allow_null'] = False
            else:
                features[self.time_feature_name]['bounds'] = {'allow_null': False}

        features = self._infer_delta_min_and_max(
            features=features,
            datetime_feature_formats=merged_args.datetime_feature_formats,
            id_feature_name=merged_args.id_feature_name,
            orders_of_derivatives=merged_args.orders_of_derivatives,
            derived_orders=merged_args.derived_orders,
            max_workers=merged_args.max_workers
        )

        # Set any manually specified rate/delta boundaries
        if merged_args.delta_boundaries is not None:
            self._set_rate_delta_bounds('delta', merged_args.delta_boundaries, features)
        if merged_args.rate_boundaries is not None:
            self._set_rate_delta_bounds('rate', merged_args.rate_boundaries, features)

        return features

    def __call__(self, **kwargs) -> SingleTableFeatureAttributes:
        """Process and return feature attributes."""
        feature_attributes = self._process(**kwargs)
        # Put the time_feature_name back into the kwargs dictionary.
        kwargs["time_feature_name"] = self.time_feature_name
        return SingleTableFeatureAttributes(feature_attributes, params=kwargs, unsupported=self.unsupported)
