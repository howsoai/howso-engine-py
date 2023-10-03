"""
Internal utilities.

Notice: These are internal utilities and are not intended to be
        referenced directly.
"""
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
import datetime
import decimal
from inspect import getfullargspec
import logging
import math
import random
import re
from typing import (
    Any, Dict, Generator, Iterable, List, Mapping, Optional, Tuple,
    TYPE_CHECKING, Union,
)
import unicodedata
import uuid
import warnings

from humanize import precisedelta
import numpy as np
import pandas as pd
from semantic_version import Version

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .monitors import ProgressTimer


def postprocess_trainee(trainee):
    """
    Post-process a trainee to update its data into the expected format.

    Should be used on trainee objects returned from the API.
    NOTE: Mutates the original trainee object.

    Parameters
    ----------
    trainee : Trainee
        The trainee instance.

    Returns
    -------
    Trainee
        The trainee instance.
    """
    trainee.features = postprocess_feature_attributes(trainee.features)
    return trainee


def preprocess_trainee(trainee):
    """
    Pre-process a trainee to update its data into the expected format.

    Should be used on trainee objects before sending to the API.
    Does not mutate the original trainee object.

    Parameters
    ----------
    trainee : Trainee
        The trainee instance.

    Returns
    -------
    Trainee
        Updated copy of the trainee instance.
    """
    trainee = deepcopy(trainee)
    trainee.features = preprocess_feature_attributes(trainee.features)
    return trainee


def deserialize_to_dataframe(
    data: Union[Iterable[Iterable[object]], Iterable[Dict[str, object]]],
    columns: Optional[Iterable[str]] = None,
    index: Optional[Iterable[Any]] = None
) -> pd.DataFrame:
    """
    Deserialize data into a DataFrame.

    Parameters
    ----------
    data : list of list of object or list of dict
        The data to deserialize.
    columns : list of str
        The column mapping. The order corresponds to how the data will
        be mapped to columns in the output. Ignored for list of dict where
        the dict key is the column name.
    index : list of Any
        The row index to use.

    Returns
    -------
    pandas.DataFrame
        The deserialized data as DataFrame.
    """
    if data is not None:
        if isinstance(data, dict) and all(x is None for x in data.values()):
            return pd.DataFrame([], columns=columns or data.keys())
        return pd.DataFrame(data, columns=columns, index=index)
    else:
        return pd.DataFrame([], columns=columns, index=index)


def get_features_from_data(
    data: Any, *,
    default_features: Optional[List[str]] = None,
    data_parameter: Optional[str] = 'cases',
    features_parameter: Optional[str] = 'features'
) -> List[str]:
    """
    Retrieve feature names from dataframe columns.

    Parameters
    ----------
    data : Any
        The data to inspect for feature names.
    default_features : list of str, optional
        Feature names to fallback to if unable to determine from DataFrame.
    data_parameter : str, optional
        The name of the data parameter to reference in the error message.
    features_parameter : str, optional
        The name of the parameter to require for features in the error message
        if features cannot be determined using the data.

    Returns
    -------
    list of str
        The feature names.

    Raises
    ------
    HowsoError
        When cases are not a DataFrame or when the DataFrame does not contain
        named columns.
    """
    # Import locally to prevent a circular import
    from howso.client.exceptions import HowsoError

    if isinstance(data, pd.DataFrame):
        if isinstance(data.columns, pd.RangeIndex):
            raise HowsoError(
                f"A `{features_parameter}` list is required when the "
                f"`{data_parameter}` DataFrame does not contain named "
                f"columns.")
        else:
            return data.columns.tolist()
    elif default_features is not None:
        return default_features
    else:
        raise HowsoError(
            f"A `{features_parameter}` list is required when "
            f"`{data_parameter}` are not provided as a DataFrame.")


def serialize_openapi_models(obj: Any, *, exclude_null: bool = False) -> Any:
    """
    Serialize OpenAPI client model instances.

    Parameters
    ----------
    obj : dict or list or object

    Returns
    -------
    list or dict or object
        The serialized model data.
    """
    if isinstance(obj, list):
        return [
            serialize_openapi_models(item, exclude_null=exclude_null)
            for item in obj
        ]
    if isinstance(obj, OrderedDict):
        # Use OrderedDict if input is an OrderedDict, for consistency
        result = OrderedDict()
        for k, v in obj.items():
            result[k] = serialize_openapi_models(v, exclude_null=exclude_null)
        return result
    if isinstance(obj, dict):
        return {
            k: serialize_openapi_models(v, exclude_null=exclude_null)
            for k, v in obj.items()
        }
    if hasattr(obj, 'to_dict'):
        args = getfullargspec(obj.to_dict).args
        if 'exclude_null' in args:
            return obj.to_dict(exclude_null=exclude_null)
        else:
            return obj.to_dict()
    return obj


def postprocess_feature_attributes(features):
    """
    Post-process feature attributes into the expected client format.

    Updates all date_time_format's to the original_format.python value,
    if it exists on the feature.

    Parameters
    ----------
    features : dict
        Dictionary of feature name to feature value.

    Returns
    -------
    dict or None
        The updated copy of features.
    """
    if features is None:
        return None

    # Serialize any OpenAPI models
    features = deepcopy(serialize_openapi_models(features))

    for feat in features.values():
        if feat is None:
            continue

        # Backwards compatibility for non-sensitive
        # TODO 15469 - Remove shim
        if 'non-sensitive' in feat:
            feat['non_sensitive'] = feat['non-sensitive']
            del feat['non-sensitive']

        # Replace any instances of 'date_time_format' with the original python
        # format if it is defined
        if 'date_time_format' in feat:
            try:
                if isinstance(feat['original_format']['python'], str):
                    # Backward compatibility shim for trainees created via
                    # API version <= 2.1.75, to translate string to dict format
                    feat['original_format']['python'] = {
                        'date_time_format': feat['original_format']['python']
                    }
                feat['date_time_format'] = (
                    feat['original_format']['python']['date_time_format'])
            except (TypeError, KeyError):
                pass

    return features


def preprocess_feature_attributes(features):
    """
    Pre-process feature attributes into the expected API format.

    Updates all date_time_format's for features by removing the Python-specific
    fractional '.%f' formatting used for high precision seconds. Keeps a copy
    of the original format in original_format.python.

    Parameters
    ----------
    features : dict
        Dictionary of feature name to feature value.

    Returns
    -------
    dict or None
        The updated copy of features.
    """
    if features is None:
        return None

    # Serialize any OpenAPI models
    features = deepcopy(serialize_openapi_models(features))

    regex = re.compile(r"%S.%f")
    for feat in features.values():
        if feat is None:
            continue

        # Shim to convert old "non-sensitive" properties to "non_sensitive"
        # TODO 15469 - Remove shim
        if 'non-sensitive' in feat:
            feat['non_sensitive'] = feat['non-sensitive']
            del feat['non-sensitive']
            warnings.warn(
                "The feature attribute 'non-sensitive' has been renamed to "
                "'non_sensitive', support for using the previous name will "
                "be removed in a future version.",
                DeprecationWarning
            )

        # Set decimal places to 0 when %S is in datetime format but %f is not.
        # This prevents core from returning microseconds
        try:
            if (
                '%S' in feat['date_time_format'] and
                '%f' not in feat['date_time_format'] and
                'decimal_places' not in feat
            ):
                feat['decimal_places'] = 0
        except (KeyError, TypeError, ValueError):
            pass

        # Replace any instances of %S.%f in 'date_time_format' with just %S
        # and store a copy of the original format in 'original_format.python'
        try:
            if regex.search(feat['date_time_format']):
                feat.setdefault('original_format', dict())
                feat['original_format'].setdefault('python', dict())
                feat['original_format']['python']['date_time_format'] = (
                    feat['date_time_format'])
                feat['date_time_format'] = regex.sub("%S",
                                                     feat['date_time_format'])
        except (KeyError, TypeError, ValueError):
            pass

    return features


def format_react_response(response, single_action=False):
    """
    Reformat the react response into a dict of action and explanation.

    Parameters
    ----------
    response : dict
        The raw react response object.
    single_action : bool, default False
        If response should be a single action value. (i.e. React into series)

    Returns
    -------
    dict
        A dict of two keys, action and explanation.
    """
    # Import locally to prevent a circular import
    from howso.utilities import replace_doublemax_with_infinity
    response = replace_doublemax_with_infinity(response)

    action_features = response['action_features'] or list()
    action_values = response['action_values']

    # Convert to format of a list of dicts of feature->value
    action = dict() if single_action else list()
    if (
        action_values is not None
        and len(action_values) > 0
        and len(action_features) > 0
    ):
        if single_action:
            action = dict(zip(action_features, action_values))
        else:
            action = [dict(zip(action_features, values)) for
                      values in action_values]

    # remove action_values from explanation to prevent output of dupe data
    del response['action_values']

    return {'action': action, 'explanation': response}


def accumulate_react_result(accumulated_result, result):
    """
    Accumulate the results from multiple reacts responses.

    Parameters
    ----------
    accumulated_result : dict
        Accumulated react responses. This object will be mutated in place.
    result : dict
        The react response object.

    Returns
    -------
    dict
         The updated accumulated_result dict.
    """
    for k, v in result.items():
        if accumulated_result.get(k) is None:
            if v is None:
                accumulated_result[k] = None
            else:
                accumulated_result[k] = []

        if k == 'action_features':
            # Only include action features once
            if not accumulated_result[k]:
                accumulated_result[k] = v
            continue

        if v is not None:
            accumulated_result[k].extend(v)

    return accumulated_result


def slugify(value, allow_unicode=False):
    """
    Slugify a value.

    Convert spaces or repeated dashes to single dashes. Remove characters that
    aren't alphanumerics, underscores, or hyphens. Convert to lowercase. Also
    strip leading and trailing whitespace, dashes, and underscores.

    Sourced from:
    https://github.com/django/django/blob/main/django/utils/text.py

    Parameters
    ----------
    value : Any
        The value to slugify.
    allow_unicode : bool, default False
        When False, converts to ASCII.

    Returns
    -------
    str
        The slugified version of the input value.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def generate_cache_file_name(identifier, ext="txt"):
    """
    Generate a unique cache file name.

    Parameters
    ----------
    identifier : str
        An identifier to include in the filename.
    ext : str
        The file extension.

    Returns
    -------
    str
        The filename.
    """
    nonce = f'{random.randint(0, 16 ** 8):08x}'
    return f".howso_recent_cache_{slugify(identifier)}_{nonce}.{ext}"


def insufficient_generation_check(
    requested_num_cases: int,
    gen_num_cases: int,
    suppress_warning: bool = False
) -> bool:
    """
    Warn user about not generating sufficient number of cases.

    Parameters
    ----------
    requested_num_cases : int
        Number of cases requested by the user.
    gen_num_cases : int
        Number of cases actually generated.
    suppress_warning : bool, defaults to False
        (Optional) If True, warnings will be suppressed.
        By default, warnings will be displayed.

    Returns
    -------
    bool
        Returns `True` if requested number of cases is not equal to
        number of cases generated. Otherwise, returns `False`.
    """
    if gen_num_cases < requested_num_cases:
        if not suppress_warning:
            warnings.warn(
                f"The number of cases generated is less than number of "
                f"cases requested ({gen_num_cases} < {requested_num_cases}"
                f"). This might happen when `generate_new_cases` "
                f"parameter is set to 'always', and the data is heavily "
                f"constrained.", RuntimeWarning
            )
        return True
    return False


def sanitize_for_json(obj: Any):  # noqa: C901
    """
    Sanitizes data for JSON serialization.

    If obj is None, return None.
    If obj is str, int, float, bool, bytes, return directly.
    If obj is NaN or NaT, return None.
    If obj is datetime, date or time, convert to string in iso8601 format.
    If obj is timedelta, return float total seconds.
    If obj is iterable, sanitize each element to a list.
    If obj is mappable, sanitize each value to a dict.
    If obj is OpenAPI model, sanitize each attribute value and return as dict.
    If obj is NumPy data type, return as related primitive type.
    If obj is any other type, raise ValueError.

    Parameters
    ----------
    obj : Any
        The data to serialize.

    Returns
    -------
    Any
        The sanitized data.

    Raises
    ------
    TypeError
        When obj is of an unsupported type.
    """
    if obj is None:
        return None

    # Check for numpy data types and cast them appropriately before
    # serialization.
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return [sanitize_for_json(sub_obj) for sub_obj in obj]
        elif isinstance(obj, np.datetime64):
            if np.isnat(obj):
                # Auto serialize NaT to None
                return None
            else:
                obj = obj.astype(datetime)
        elif issubclass(type(obj), (np.integer, int)):
            obj = int(obj)
        elif issubclass(type(obj), (np.floating, float)):
            obj = float(obj)
        elif issubclass(type(obj), str):
            obj = str(obj)
        elif isinstance(obj, bool):
            obj = bool(obj)
        else:
            # If it's not an int, float, array or bool, assume it is
            # meant to be passed as a string as there are no
            # byte/bytes data types supported by the core.
            logger.warning(
                f"Unknown numpy datatype {type(obj)} encountered during "
                f"serialization. Casting to a string.")
            obj = str(obj)

    # Serialize known types
    if isinstance(obj, (str, int, bool)):
        return obj
    elif isinstance(obj, bytes):
        logger.warning("Bytes data encountered during serialization. Casting "
                       "to a string.")
        return str(obj)
    elif isinstance(obj, float):
        if np.isnan(obj):
            # Auto serialize NaN to None
            return None
        return obj
    elif (
        isinstance(obj, Iterable) and
        not isinstance(obj, (str, bytes, Mapping))
    ):
        return [sanitize_for_json(sub_obj) for sub_obj in obj]
    elif isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        if pd.isnull(obj):
            return None
        # Format datetime objects
        return obj.isoformat()
    elif isinstance(obj, datetime.timedelta):
        # Convert time deltas to total seconds
        return obj.total_seconds()
    elif isinstance(obj, decimal.Decimal):
        if obj.is_nan():
            # Convert NaNs to None
            return None
        # FIXME: Core does not yet fully support stringified/decimal numbers,
        #  so we convert to a float here instead.
        return float(obj)
    elif isinstance(obj, uuid.UUID):
        return str(obj)

    # Serialize objects to dict
    if isinstance(obj, Mapping):
        obj_dict = obj
    elif hasattr(obj, 'openapi_types') and hasattr(obj, 'attribute_map'):
        # Convert openapi model to dict using the attribute mapping,
        # excluding values which are None
        obj_dict = {obj.attribute_map[attr]: getattr(obj, attr)
                    for attr, _ in obj.openapi_types.items()
                    if getattr(obj, attr) is not None}
    else:
        try:
            # In some cases a pandas NAType could be present, check for it as
            # a last resort before raising a TypeError
            if pd.isnull(obj):
                return None
        except Exception:  # noqa: Deliberately broad
            # Failed to check value is null, must not be null
            pass
        # Unhandled type, raise
        raise TypeError(f"Object of type {type(obj)} is not serializable")

    return {key: sanitize_for_json(val) for key, val in obj_dict.items()}


def readable_timedelta(delta: datetime.timedelta, *,
                       microsecond_places: int = 2,
                       precision: Union[bool, int] = True,
                       ) -> str:
    """
    Format timedelta to a readable string.

    Example output: 1 hour 12 minutes and 9 seconds

    Parameters
    ----------
    delta : datetime.timedelta
        The time delta to format.
    microsecond_places : int, default 2
        How many microsecond decimal places to include when formatting seconds.
    precision : bool or int, default True
        When True, all time places will be included. Otherwise, only the first
        time place greater than 0 will be included. If specified with an
        integer, includes up to that number of time places.

    Returns
    -------
    str
        The formatted time delta.
    """
    abs_delta = abs(delta)
    year, remainder_days = divmod(abs_delta.days, 365)
    month, day = divmod(remainder_days, 30)
    hour, remainder_seconds = divmod(abs_delta.seconds, 3600)
    minute, second = divmod(remainder_seconds, 60)
    microseconds = round(abs_delta.microseconds / 1_000_000, microsecond_places)
    periods = {'years': year, 'months': month, 'days': day, 'hours': hour,
               'minutes': minute, 'seconds': second}

    parts = []
    decimal_format = '%0.0f'
    for period in ['years', 'months', 'days', 'hours', 'minutes', 'seconds']:
        value = periods[period]
        if period == 'seconds' and microseconds > 0:
            value += microseconds
            decimal_format = f'%0.{microsecond_places}f'
        if value > 0:
            parts.append(period)
            if isinstance(precision, bool) and not precision:
                break
            elif len(parts) == precision:
                break

    if parts:
        minimum_unit = parts[-1]
    else:
        minimum_unit = 'seconds'
    return precisedelta(delta, minimum_unit=minimum_unit, format=decimal_format)


class BatchScalingManager:
    """
    Manages scaling batching operations.

    Parameters
    ----------
    starting_size : int
        The requested starting batch size.
    progress_monitor : ProgressTimer
        A progress timer instance to use for scaling.
    """

    # Threshold by which batch sizes will be increased/decreased until
    # request-response time falls between these two times
    time_threshold: Tuple[datetime.timedelta, datetime.timedelta] = (
        datetime.timedelta(seconds=60), datetime.timedelta(seconds=75))
    # The batch size min and max (respectively)
    size_limits: Tuple[int, Optional[int]] = (1, None)
    # The rate at which batches are scaled up and down (respectively)
    # See: https://en.wikipedia.org/wiki/Golden_ratio
    size_multiplier: Tuple[float, float] = (1.618, 0.809)

    def __init__(self, starting_size: int, progress_monitor: "ProgressTimer"):
        """Initialize a new BatchScalingManager instance."""
        self.starting_size = starting_size
        self.progress = progress_monitor

    def gen_batch_size(self) -> Generator[int, Optional[datetime.timedelta],
                                          None]:
        """
        Returns a generator to get the next batch size.

        When using "send" progress updating must be done manually and the
        last tick duration should be provided as the parameter to "send". When
        not using "send" progress updating will happen automatically.
        """
        if not self.progress.has_started:
            raise ValueError("Batching has not yet started")

        batch_size = self.starting_size
        while not self.progress.has_ended and not self.progress.is_complete:
            batch_size = self.clamp(batch_size, self.progress.current_tick,
                                    self.progress.total_ticks)
            tick_duration = yield batch_size
            if tick_duration is None:
                # If send is not used, automatically update progress
                tick_duration = self.progress.tick_duration
                self.progress.update(batch_size)
            batch_size = self.scale(batch_size, tick_duration)
        return None

    def scale(self, batch_size: int, batch_duration: datetime.timedelta) -> int:
        """
        Scale batch size based on duration of the batch.

        Parameters
        ----------
        batch_size : int
            The current batch size.
        batch_duration : datetime.timedelta
            The time the last batch took to complete.

        Returns
        -------
        int
            The new batch size.
        """
        if batch_duration <= self.time_threshold[0]:
            # If took less than threshold, increase batch size
            batch_size = math.ceil(batch_size * self.size_multiplier[0])
        elif batch_duration > self.time_threshold[1]:
            # If took longer than threshold, lower batch size
            if self.size_multiplier[1] < 1:
                batch_size = math.floor(batch_size * self.size_multiplier[1])
            else:
                batch_size = math.floor(batch_size / self.size_multiplier[1])
        return batch_size

    def clamp(self, batch_size: int, batch_offset: int, total: int) -> int:
        """
        Clamp batch size between min/max allowed value.

        Parameters
        ----------
        batch_size : int
            The current batch size.
        batch_offset : int
            The current batch offset.
        total : int
            The total number of the items being batched.

        Returns
        -------
        int
            The new batch size.
        """
        # Clamp batch size to the minimum requested batch size, but
        # ensure it does not exceed total number of items batched
        batch_size = min(max(batch_size, self.size_limits[0]),
                         total - batch_offset)
        if self.size_limits[1]:
            # Limit batch size to maximum value
            batch_size = min(batch_size, self.size_limits[1])
        return batch_size


def show_core_warnings(core_warnings):
    """Warns the user for each warning returned from the core."""
    # Import here to avoid circular import
    from ..client.exceptions import HowsoWarning

    if isinstance(core_warnings, Iterable):
        for w in core_warnings:
            if warning := w.get("detail"):
                warnings.warn(warning, category=HowsoWarning)


def to_pandas_datetime_format(f):
    """
    Normalize the pandas datetime format.

    Checks if format is an ISO8601 like format. If so and pandas version is
    2.0.0 or greater, return "ISO8601"

    Parameters
    ----------
    f : str
        The format string.

    Returns
    -------
    str
        The normalized format.
    """
    # Prevent circular import
    from .utilities import date_format_is_iso
    if date_format_is_iso(f):
        try:
            pd_ver = Version(pd.__version__)
            if pd_ver.major is not None and pd_ver.major >= 2:
                return "ISO8601"
        except Exception:  # noqa: Deliberately broad
            # Failed to check pandas version
            pass
    return f


class IgnoreWarnings:
    """
    Simple context manager to ignore Warnings.

    Parameters
    ----------
    warning_types : Warning or Iterable of Warnings
        The warning classes to ignore.
    """

    def __init__(
        self,
        warning_types: Union[Warning, Iterable[Warning]]
    ):
        """Initialize a new `catch_warnings` instance."""
        self._catch_warnings = warnings.catch_warnings()
        self._warning_types = warning_types

        if not isinstance(self._warning_types, Iterable):
            self._warning_types = [self._warning_types]
        for warning_type in self._warning_types:
            self._check_warning_class(warning_type)

    @staticmethod
    def _check_warning_class(warning_type):
        """Check correct warning type."""
        if not issubclass(warning_type, Warning):
            warnings.warn(
                f"{warning_type} is not a valid subclass of `Warning`. "
                "Warnings will not be ignored."
            )

    def __enter__(self):
        """Context entrance."""
        # Enters the  `catch_warnings` instance.
        self._catch_warnings.__enter__()
        for warning_type in self._warning_types:
            warnings.filterwarnings("ignore", category=warning_type)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context exit."""
        self._catch_warnings.__exit__(exc_type, exc_value, traceback)
        return False
