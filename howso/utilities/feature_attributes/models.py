from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, Iterable, List, Optional, Union
from typing_extensions import Self
import warnings

class InferFeatureAttributesArgs(BaseModel):
    """
    Behavioral controls for the infer_feature_attributes function.
    """

    attempt_infer_extended_nominals: Optional[bool] = Field(
        default=False,
        description=(
            "(Optional) If set to True, detections of extended nominals will be attempted. "
            "If the detection fails, the categorical variables will be set to ``int-id`` subtype.\n\n"
            ".. note ::\n"
            "    Please refer to ``InferFeatureAttributesArgs`` for other parameters related to extended nominals."
        )
    )
    datetime_feature_formats: Optional[Dict[str, Union[str, Iterable[str]]]] = Field(
        default=None,
        description=(
            "(Optional) Dict defining custom (non-ISO8601) datetime or time-only formats.\n"
            "By default, datetime features are assumed to be in ISO8601 format. Non-English datetimes\n"
            "must have locales specified. If locale is omitted, the default system locale is used.\n"
            "The keys are the feature name, and the values are a tuple of date/time format and locale\n"
            "string. Time-only feature formats are expected to adhere to the format codes used in\n"
            "strftime()."
        )
    )
    default_time_zone: Optional[str] = Field(
        default=None,
        description=(
            "(Optional) The fallback time zone for any datetime feature if one is not provided in\n"
            "``datetime_feature_formats`` and it is not inferred from the data. If not specified\n"
            "anywhere, the Howso Engine will default to UTC."
        )
    )
    delta_boundaries: Optional[Dict[str, Dict[str, Dict[str, Optional[float]]]]] = Field(
        default=None,
        description=(
            "(Optional) For time series, specify the delta boundaries in the form\n"
            "{\"feature\" : {\"min|max\" : {order : value}}}. Works with partial values\n"
            "by specifying only particular order of derivatives you would like to\n"
            "overwrite. Invalid orders will be ignored."
        )
    )
    dependent_features: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description=(
            "Dict mapping a feature to a list of other feature(s) that it depends on or\n"
            "that are dependent on it. This restricts the cases that can be selected as\n"
            "neighbors (such as in :meth:`~howso.engine.Trainee.react`) to ones that\n"
            "satisfy the dependency, if possible. If this is not possible, either due to\n"
            "insufficient data which satisfy the dependency or because dependencies are\n"
            "probabilistic, the dependency may not be maintained. Be aware that dependencies\n"
            "introduce further constraints to data and so several dependencies or dependencies\n"
            "on already constrained datasets may restrict which operations are possible while\n"
            "maintaining the dependency. As a rule of thumb, sets of features that have\n"
            "dependency relationships should generally not include more than 1 continuous feature,\n"
            "unless the continuous features have a small number of values that are commonly used."
        )
    )
    derived_orders: Optional[Dict[str, int]] = Field(
        default=None,
        description=(
            "(Optional) Dict of features to the number of orders of derivatives\n"
            "that should be derived instead of synthesized. For example, for a\n"
            "feature with a 3rd order of derivative, setting its derived_orders\n"
            "to 2 will synthesize the 3rd order derivative value, and then use\n"
            "that synthed value to derive the 2nd and 1st order."
        )
    )
    features: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "(Optional) Dict of features..." # TODO needs documentation. Used in Pandas `_shard` at least.
        )
    )
    id_feature_name: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="(Optional) The name(s) of the ID feature(s)."
    )
    include_extended_nominal_probabilities: Optional[bool] = Field(
        default=False,
        description=(
            "(Optional) If true, extended nominal probabilities will be appended\n"
            "as metadata into the feature object."
        )
    )
    include_sample: Optional[bool] = Field(
        default=False,
        description=(
            "If True, include a ``sample`` field containing a sample of the data\n"
            "from each feature in the output feature attributes dictionary."
        )
    )
    infer_bounds: Optional[bool] = Field(
        default=True,
        description=(
            "(Optional) If True, bounds will be inferred for the features if the\n"
            "feature column has at least one non NaN value"
        )
    )
    lags: Optional[Union[List[int], Dict[str, List[int]]]] = Field(
        default=None,
        description=(
            "(Optional) A list containing the specific indices of the desired lag\n"
            "features to derive for each feature (not including the series time\n"
            "feature). Specifying derived lag features for the feature specified by\n"
            "time_feature_name must be done using a dictionary. A dictionary can be\n"
            "used to specify a list of specific lag indices for specific features."
        )
    )
    max_workers: Optional[int] = Field(
        default=None,
        description=(
            "(Optional) If unset or set to None (recommended), let the ProcessPoolExecutor\n"
            "choose the best maximum number of process pool workers to process\n"
            "columns in a multi-process fashion. In this case, if the product of the\n"
            "data's rows and columns > 25,000,000 or if the data is time series and the\n"
            "number of rows > 500,000 multiprocessing will be used.\n\n"
            "If defined with an integer > 0, manually set the number of max workers.\n"
            "Otherwise, the feature attributes will be calculated serially. Setting\n"
            "this parameter to zero (0) will disable multiprocessing."
        )
    )
    mode_bound_features: Optional[List[str]] = Field(
        default=None,
        description=(
            "(Optional) Explicit list of feature names to use mode bounds for\n"
            "when inferring loose bounds. If None, assumes all features. A mode\n"
            "bound is used instead of a loose bound when the mode for the\n"
            "feature is the same as an original bound, as it may represent an\n"
            "application-specific min/max."
        )
    )
    nominal_substitution_config: Optional[Dict] = Field(
        default=None,
        description=(
            "(Optional) Configuration of the nominal substitution engine\n"
            "and the nominal generators and detectors."
        )
    )
    num_lags: Optional[Union[int, Dict[str, int]]] = Field(
        default=None,
        description=(
            "(Optional) An integer specifying the number of lag features to\n"
            "derive for each feature (not including the series time feature).\n"
            "Specifying derived lag features for the feature specified by\n"
            "time_feature_name must be done using a dictionary. A dictionary can be\n"
            "used to specify numbers of lags for specific features."
        )
    )
    orders_of_derivatives: Optional[Dict[str, int]] = Field(
        default=None,
        description=(
            "(Optional) Dict of features and their corresponding order of\n"
            "derivatives for the specified type (delta/rate). If provided will\n"
            "generate the specified number of derivatives and boundary values. If\n"
            "set to 0, will not generate any delta/rate features. By default all\n"
            "continuous features have an order value of 1."
        )
    )
    ordinal_feature_values: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description=(
            "(optional) Dict for ordinal string features defining an ordered\n"
            "list of string values for each feature, ordered low to high. If\n"
            "specified will set 'type' to be 'ordinal' for all features in\n"
            "this map."
        )
    )
    rate_boundaries: Optional[Dict[str, Dict[str, Dict[str, Optional[float]]]]] = Field(
        default=None,
        description=(
            "(Optional) For time series, specify the rate boundaries in the form\n"
            "{\"feature\" : {\"min|max\" : {order : value}}}. Works with partial values\n"
            "by specifying only particular order of derivatives you would like to\n"
            "overwrite. Invalid orders will be ignored."
        )
    )
    tight_bounds: Optional[List[str]] = Field(
        default=None,
        description=(
            "(Optional) Set tight min and max bounds for the features\n"
            "specified in the Iterable."
        )
    )
    time_feature_is_universal: Optional[bool] = Field(
        default=None,
        description=(
            "If True, the time feature will be treated as universal and future data\n"
            "is excluded while making predictions. If False, the time feature will\n"
            "not be treated as universal and only future data within the same series\n"
            "is excluded while making predictions. It is recommended to set this\n"
            "value to True if there is any possibility of global relevancy of time,\n"
            "which is the default behavior."
        )
    )
    time_invariant_features: Optional[List[str]] = Field(
        default=None,
        description="(Optional) Names of time-invariant features."
    )
    time_series_type_default: Optional[str] = Field(
        default='rate',
        description=(
            "(Optional) Type specifying how time series is generated.\n"
            "One of 'rate' or 'delta', default is 'rate'."
        )
    )
    time_series_types_override: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "(Optional) Dict of features and their corresponding time series type,\n"
            "one of 'rate' or 'delta', used to override ``time_series_type_default``\n"
            "for the specified features."
        )
    )
    types: Optional[Dict[str, Union[str, List[str]]]] = Field(
        default=None,
        description=(
            "(Optional) Dict of features and their intended type (i.e., \"nominal,\"\n"
            "\"ordinal,\" or \"continuous\"), or types mapped to MutableSequences of\n"
            "feature names. Any types provided here will override the types that would\n"
            "otherwise be inferred, and will direct ``infer_feature_attributes`` to\n"
            "compute the attributes accordingly."
        )
    )

    def merge_kwargs(self, **kwargs) -> Self:
        """Return a new copy of InferFeatureAttributesArgs handling deprecation and remapping from kwargs if needed."""
        if kwargs:
            warn_keys = list(kwargs.keys())
            warnings.warn(
                f"Usage of {warn_keys} directly is deprecated and will be removed in future versions. "
                "Please create an instance of InferFeatureAttributesArgs and provide it to the `args` parameter.",
                DeprecationWarning,
                stacklevel=2
            )
        return self.model_copy(update=kwargs)