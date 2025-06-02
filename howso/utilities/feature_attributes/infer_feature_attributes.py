from __future__ import annotations

from collections.abc import Iterable
import typing as t
import warnings

import pandas as pd

from .abstract_data import InferFeatureAttributesAbstractData
from .base import FeatureAttributesBase
from .models import InferFeatureAttributesArgs
from .pandas import InferFeatureAttributesDataFrame
from .protocols import IFACompatibleADCProtocol, SQLRelationalDatastoreProtocol, TableNameProtocol
from .relational import InferFeatureAttributesSQLDatastore
from .time_series import InferFeatureAttributesTimeSeries


def infer_feature_attributes(
    data: pd.DataFrame | SQLRelationalDatastoreProtocol, 
    *,
    args: t.Optional[InferFeatureAttributesArgs] = None,
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

    args : InferFeatureAttributesArgs
        (Optional) Defines behaviors of infer_feature_attributes processing.

    tables : Iterable of TableNameProtocol
        (Optional, required for datastores) An Iterable of table names to
        infer feature attributes for.

        If included, feature attributes will be generated in the form
        ``{table_name: {feature_attribute: value}}``.

    time_feature_name : str, default None
        (Optional, required for time series) The name of the time feature.

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
    # Ensure we have an instance of InferFeatureAttributesArgs, from args or on the fly.
    args = args or InferFeatureAttributesArgs()
    merged_args = args.merge_kwargs(**kwargs)
        
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
    elif isinstance(data, IFACompatibleADCProtocol):
        infer = InferFeatureAttributesAbstractData(data)
    else:
        raise NotImplementedError('Data not recognized as a DataFrame, AbstractData class, or compatible datastore.')

    return infer(args=merged_args)
