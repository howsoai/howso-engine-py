"""
Internal utilities.

Notice: These are internal utilities and are not intended to be
        referenced directly.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable, Collection, Generator, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
import datetime
import decimal
from inspect import getfullargspec
import json
import logging
from pathlib import Path
import random
import re
import typing as t
import unicodedata
import uuid
import warnings

from humanize import precisedelta
import numpy as np
import pandas as pd
from semantic_version import Version

from .monitors import ProgressTimer

logger = logging.getLogger(__name__)

T = t.TypeVar("T")


def deserialize_to_dataframe(
    data: Iterable[Iterable[t.Any]] | Iterable[Mapping[str, t.Any]] | None,
    columns: t.Optional[Iterable[str]] = None,
    index: t.Optional[Iterable[t.Any]] = None
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
    data: t.Any, *,
    data_parameter: t.Optional[str] = 'cases',
    features_parameter: t.Optional[str] = 'features'
) -> list[str]:
    """
    Retrieve feature names from dataframe columns.

    Parameters
    ----------
    data : Any
        The data to inspect for feature names.
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
    else:
        raise HowsoError(
            f"A `{features_parameter}` list is required when "
            f"`{data_parameter}` are not provided as a DataFrame.")


def serialize_models(obj: t.Any, *, exclude_null: bool = False) -> t.Any:
    """
    Serialize client model instances.

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
            serialize_models(item, exclude_null=exclude_null)
            for item in obj
        ]
    if isinstance(obj, OrderedDict):
        # Use OrderedDict if input is an OrderedDict, for consistency
        result = OrderedDict()
        for k, v in obj.items():
            result[k] = serialize_models(v, exclude_null=exclude_null)
        return result
    if isinstance(obj, dict):
        return {
            k: serialize_models(v, exclude_null=exclude_null)
            for k, v in obj.items()
        }
    if hasattr(obj, 'to_dict'):
        args = getfullargspec(obj.to_dict).args
        if 'exclude_null' in args:
            return obj.to_dict(exclude_null=exclude_null)
        else:
            return obj.to_dict()
    return obj


def postprocess_feature_attributes(features: Mapping | None) -> dict:
    """
    Post-process feature attributes into the expected client format.

    Updates all date_time_format's to the original_format.python value,
    if it exists on the feature.

    Parameters
    ----------
    features : dict or None
        Dictionary of feature name to feature value.

    Returns
    -------
    dict
        The updated copy of features.
    """
    if features is None:
        return {}

    # Serialize any OpenAPI models
    feature_attributes: dict = deepcopy(serialize_models(features))

    for feat in feature_attributes.values():
        if feat is None:
            continue

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

    return feature_attributes


def preprocess_feature_attributes(features: Mapping | None) -> dict | None:
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
    feature_attributes: dict = deepcopy(serialize_models(features))

    regex = re.compile(r"%S.%f")
    for key, feat in feature_attributes.items():
        if feat is None:
            continue

        if not isinstance(key, str):
            raise ValueError("Feature attribute keys must be strings.")
        elif not key.strip():
            raise ValueError("Feature attribute keys may not be blank.")

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

    return feature_attributes


def format_react_response(response: dict, single_action: bool = False):
    """
    Reformat the react response into a dict of action and details.

    Parameters
    ----------
    response : dict
        The raw react response object.
    single_action : bool, default False
        If response should be a single action value. (i.e. React into series)

    Returns
    -------
    dict
        A dict of two keys, action and details.
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

    # remove action_values from details to prevent output of dupe data
    del response['action_values']

    return {'action': action, 'details': response}


def accumulate_react_result(accumulated_result: dict, result: dict) -> dict:
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

        if k in {'action_features', 'context_features'}:
            # Only include action/context features once
            if not accumulated_result[k]:
                accumulated_result[k] = v
            continue

        if v is not None:
            accumulated_result[k].extend(v)

    return accumulated_result


def random_handle() -> str:
    """
    Generate a random 6 byte hexadecimal handle.

    Returns
    -------
    str
        A random 6 byte hex.
    """
    try:
        # Use of secrets/uuid must be used instead of the "random" package
        # as they will not be affected by setting random.seed which could
        # cause duplicate handles to be generated.
        import secrets
        return secrets.token_hex(6)
    except (ImportError, NotImplementedError):
        # Fallback to uuid if operating system does not support secrets
        return uuid.uuid4().hex[-12:]


def slugify(value: t.Any, allow_unicode: bool = False):
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


def generate_cache_file_name(identifier: str, ext: str = "txt"):
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


def sanitize_for_json(obj: t.Any):  # noqa: C901
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
        elif isinstance(obj, (np.bool, np.bool_, bool)):
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
                       precision: bool | int = True,
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


def get_packaged_engine_version() -> Version | None:
    """
    Get the packaged engine version.

    Returns
    -------
    Version or None
        The packaged engine version or None if not available.
    """
    file_path = Path(__file__).parent.parent.joinpath("howso-engine", "version.json")
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r") as f:
            detail = json.loads(f.read())
        return Version(detail["version"])
    except Exception:
        return None


class BaseBatchScalingManager(t.Protocol):
    """Interface definition for scaling batching operations."""

    @property
    def batch_size(self) -> int:
        """Get the current batch size."""
        ...

    @property
    def thread_count(self) -> int:
        """Get the current thread count."""
        ...

    @thread_count.setter
    def thread_count(self, thread_count: int) -> None:
        """Set the current thread count."""

    def update(self, batch_duration: datetime.timedelta, memory_sizes: tuple[int, int] | None) -> int:
        """
        Update the batch size in response to activity happening.

        Parameters
        ----------
        tick_duration : timedelta
            The amount of time it took to process the most recent batch.
        memory_sizes : tuple[int, int], optional
            The input and output data sizes, if known.

        Returns
        -------
        int
            The new batch size.
        """
        ...

class FixedBatchScalingManager(BaseBatchScalingManager):
    """
    A batch scaling manager that never changes the batch size.

    Parameters
    ----------
    batch_size : int
        The batch size.
    """

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """Get the current batch size."""
        return self._batch_size

    @property
    def thread_count(self) -> int:
        """Return a fixed thread count."""
        return 1

    @thread_count.setter
    def thread_count(self, thread_count: int) -> None:
        """Ignore changes to the thread count."""

    def update(self, batch_duration: datetime.timedelta, memory_sizes: tuple[int, int] | None) -> int:
        """Do nothing and return the fixed batch size."""
        return self._batch_size


class BatchScalingManager(BaseBatchScalingManager):
    """
    Manages scaling batching operations.

    Parameters
    ----------
    starting_size : int
        The requested starting batch size.
    thread_count : int
        The number of threads; reported batch size will always be a multiple
        of this.
    max_size : int, optional
        The largest allowable batch size.
    """

    # Threshold by which batch sizes will be increased/decreased until
    # request-response time falls between these two times
    time_threshold: tuple[datetime.timedelta, datetime.timedelta] = (
        datetime.timedelta(seconds=60),
        datetime.timedelta(seconds=75),
    )

    # Limit by memory usage of request or response size (respectively)
    # In bytes, zero means no limit.
    memory_limits: tuple[int, int] = (50_000_000, 50_000_000)  # 50MB

    # Prevent raising batch size when the size of the request or response
    # (respectively) is within this range of the limit.
    memory_limit_thresholds: tuple[float, float] = (0.1, 0.1)  # 10%

    # The rate at which batches are scaled up and down (respectively)
    # See: https://en.wikipedia.org/wiki/Golden_ratio
    size_multiplier: tuple[float, float] = (1.618, 0.5)

    def __init__(
        self,
        starting_size: int,
        thread_count: int = 1,
        max_size: int | None = None
    ) -> None:
        """Initialize a new BatchScalingManager instance."""
        # Internal to this class, `batch_size` is maintained as a floating point
        # value for scaling accuracy and to prevent potential "traps" where the
        # multiplier isn't enough to increase/decrease the amount to overcome the
        # rounding to the nearest multiple of the minimum batch size.
        self._batch_size = float(starting_size)

        self._thread_count = thread_count

        self.max_size = max_size
        """The largest allowable batch size."""

    @property
    def batch_size(self) -> int:
        """Get the current batch size."""
        return self.clamp(self.quantize(self._batch_size))

    @batch_size.setter
    def batch_size(self, batch_size: float) -> None:
        """Manually set the current batch size."""
        self._batch_size = batch_size

    @property
    def thread_count(self) -> int:
        """Get the current thread count."""
        return self._thread_count

    @thread_count.setter
    def thread_count(self, thread_count) -> None:
        """Set the current thread count."""
        # Approximate a new batch size based on
        # the new number of threads available. This allows scaling to
        # more quickly adapt to changes in the number of threads
        # available.
        self._batch_size = self.clamp(self._batch_size / self._thread_count * thread_count)
        self._thread_count = thread_count

    def update(self, batch_duration: datetime.timedelta, memory_sizes: tuple[int, int] | None) -> int:
        """
        Update the batch size in response to activity happening.

        Parameters
        ----------
        batch_duration : timedelta
            The amount of time it took to process the most recent batch.
        memory_sizes : tuple[int, int], optional
            The input and output data sizes, if known.

        Returns
        -------
        int
            The new batch size.
        """
        self._batch_size = self.scale(self._batch_size, batch_duration, memory_sizes)
        return self.batch_size

    def scale(
        self,
        batch_size: float,
        batch_duration: datetime.timedelta,
        memory_sizes: t.Optional[tuple[int, int]],
    ) -> float:
        """
        Scale batch size based on duration or memory size of the batch.

        Parameters
        ----------
        batch_size : float
            The current batch size.
        batch_duration : datetime.timedelta or None
            The time the last batch took to complete.
        memory_sizes : tuple of (int, int) or None
            The request and response payload sizes. (respectively)

        Returns
        -------
        int
            The new batch size.
        """
        adjust = None  # -1 = lower, 0/None = keep, 1 = raise

        # Adjust based on memory sizes
        # We use the threshold to prevent raising batch size when memory usage
        # is in range of the limit.
        max_in_mem, max_out_mem = self.memory_limits
        in_mem, out_mem = memory_sizes or (0, 0)
        mem_in_threshold, mem_out_threshold = self.memory_limit_thresholds
        # Adjust based on request size
        if max_in_mem and in_mem:
            if in_mem > max_in_mem:
                adjust = -1
            elif (
                adjust is None
                and max_in_mem - (max_in_mem * mem_in_threshold) <= in_mem
            ):
                adjust = 0

        # Adjust based on response size
        if max_out_mem and out_mem:
            if out_mem > max_out_mem:
                adjust = -1
            elif (
                adjust is None
                and max_out_mem - (max_out_mem * mem_out_threshold) <= out_mem
            ):
                adjust = 0

        # Adjust based on duration
        if adjust is None and batch_duration <= self.time_threshold[0]:
            # If took less than threshold, increase batch size
            # Only raise when an adjustment has not already been set
            adjust = 1
        elif batch_duration > self.time_threshold[1]:
            # If took longer than threshold, lower batch size
            adjust = -1

        if adjust == 1:
            # Raise batch size
            batch_size = batch_size * self.size_multiplier[0]
        elif adjust == -1:
            # Lower batch size
            if self.size_multiplier[1] < 1:
                batch_size = batch_size * self.size_multiplier[1]
            else:
                batch_size = batch_size / self.size_multiplier[1]

        return self.clamp(batch_size)

    @t.overload
    def clamp(self, batch_size: int) -> int: ...
    @t.overload
    def clamp(self, batch_size: float) -> float: ...
    def clamp(self, batch_size: int | float) -> int | float:
        """
        Clamp batch size between min/max allowed value.

        Parameters
        ----------
        batch_size : int | float
            The current batch size.

        Returns
        -------
        int | float
            The new batch size.
        """
        batch_size = max(batch_size, self.thread_count)
        if self.max_size:
            batch_size = min(batch_size, self.max_size)
        return batch_size

    def quantize(self, batch_size: int | float) -> int:
        """
        Make the batch size be a multiple of the thread count.

        Round to the nearest whole batch, but always emit at least one batch.
        """
        batches = max(round(batch_size / self.thread_count), 1)
        return batches * self.thread_count



class ReactInBatches:
    """
    Run some react-type operation in batches.

    This performs all of the machinery of setting up a progress reporter,
    setting and scaling the batch size, and potentially running requests
    in parallel if this client supports that.

    This is intended to be used with the client ``react``, ``react_series``,
    or ``react_series_stationary`` methods, which all have a similar call
    pattern.

    `run` is intended to be the main entry point to this class, and typical
    callers should not directly need the other methods or fields in this
    class.

    Parameters
    ----------
    trainee_id : str
        The ID of the Trainee to react to.
    params : dict[str, t.Any]
        The engine react parameters.
    progress : ProgressTimer
        Progress tracker.
    batch_scaler : BaseBatchScalingManager
        Automatically update the batch size.
    get_thread_count : Callable[[str], int]
        Callback to get the current thread count for a trainee ID.
    get_concurrency : Callable[[str], int | None]
        Callback to get the number of operations it is reasonable to run
        concurrently for a trainee ID.  Returns None if it is never
        reasonable to run operations concurrently.
    params_for_batch : Callable[[dict, int, int], dict]
        A function that takes the initial parameter set and batch start
        and end position, and returns the parameter set for this batch.
        The function must return a copy of the parameters and not mutate
        the parameters in place.
    react_function : Callable[[str, dict], tuple[dict, int, int]]
        The actual "react" function, taking the trainee ID and the
        per-batch parameter set as parameters.
    progress_callback : Callable[[ProgressTimer, dict], None], optional
        A method to be called during batching to retrieve the progress
        metrics.

    """
    def __init__(
            self,
            *,
            trainee_id: str,
            params: dict[str, t.Any],
            progress: ProgressTimer,
            batch_scaler: BaseBatchScalingManager,
            get_thread_count: Callable[[str], int],
            get_concurrency: Callable[[str], int | None],
            params_for_batch: Callable[[dict[str, t.Any], int, int], dict[str, t.Any]],
            react_function: Callable[[str, dict[str, t.Any]], tuple[dict[str, t.Any], int, int]],
            progress_callback: Callable[[ProgressTimer, dict[str, t.Any] | None], None] | None = None,
    ) -> None:
        self.result = {'action_values': []}
        """The final result of the computation."""

        self._trainee_id = trainee_id
        """The caller-supplied trainee ID."""

        self._params = params
        """The caller-supplied set of call parameters."""

        self._futures: deque[tuple[int, Future[tuple[dict[str, t.Any], int, int]]]] = deque()
        """
        A double-ended queue of triples of batch size, start time, and futures.

        These are in order the future was started.  The futures eventually
        produce the results of the react function.
        """

        self._running: set[Future[tuple[dict[str, t.Any], int, int]]] = set()
        """A set of incomplete futures."""

        self._progress = progress
        """The progress timer monitoring this execution."""

        self._batch_scaler = batch_scaler
        """Manager to dynamically scale the batch size."""

        self._batch_scaling_future: tuple[datetime.datetime, Future[tuple[dict[t.Any, t.Any], int, int]]] | None = None
        """A specific future that will update the batch scaler when complete, with its start time."""

        self._get_thread_count = get_thread_count
        """Dynamically produce the current trainee thread count."""

        self._get_concurrency = get_concurrency
        """Dynamically produce the number of concurrent requests that can be executed."""

        self._params_for_batch = params_for_batch
        """Produce the parameters for a specific batch."""

        self._react_function = react_function
        """The underlying react-type function to call."""

        self._progress_callback = progress_callback
        """A callback invoked after each batch completes with incremental results."""

    def _send_progress(self, results: dict[str, t.Any] | None) -> None:
        """Invoke the progress callback if needed."""
        if self._progress_callback:
            self._progress_callback(self._progress, results)

    def _update_batch_size(self, batch_duration: datetime.timedelta, in_size: int, out_size: int) -> None:
        """Update the batch scaler size if needed."""
        # Ensure the minimum batch size continues to match the number of
        # threads, even over scaling events.
        self._batch_scaler.thread_count = max(self._get_thread_count(self._trainee_id), 1)
        self._batch_scaler.update(batch_duration, (in_size, out_size))

    @classmethod
    def run(
        cls,
        *,
        trainee_id: str,
        params: dict[str, t.Any],
        total_size: int,
        batch_size: int | None,
        initial_batch_size: int | None,
        get_thread_count: Callable[[str], int],
        get_concurrency: Callable[[str], int | None],
        params_for_batch: Callable[[dict[str, t.Any], int, int], dict[str, t.Any]],
        react_function: Callable[[str, dict[str, t.Any]], tuple[dict[str, t.Any], int, int]],
        progress_callback: Callable[[ProgressTimer, dict[str, t.Any] | None], None] | None = None,
    ) -> dict[str, t.Any]:
        """Run a react-type operation in batches."""
        with ProgressTimer(total_size) as progress:
            # Come up with a batch size, if we weren't provided with one.
            batch_scaler: BaseBatchScalingManager
            if batch_size:
                batch_scaler = FixedBatchScalingManager(batch_size)
            else:
                if not initial_batch_size:
                    start_batch_size = max(get_thread_count(trainee_id), 1)
                else:
                    start_batch_size = initial_batch_size
                batch_scaler = BatchScalingManager(start_batch_size, thread_count=max(get_thread_count(trainee_id), 1))

            # Create the scaler object.
            react_in_batches = cls(
                trainee_id=trainee_id,
                params=params,
                progress=progress,
                batch_scaler=batch_scaler,
                get_thread_count=get_thread_count,
                get_concurrency=get_concurrency,
                params_for_batch=params_for_batch,
                react_function=react_function,
                progress_callback=progress_callback,
            )

            if get_concurrency(trainee_id) is None:
                react_in_batches.serial()
            else:
                react_in_batches.parallel()
            return react_in_batches.result

    def serial(self) -> None:
        """Run the operation running one batch at a time."""
        batch_start = 0
        self._send_progress(None)
        while batch_start < self._progress.total_ticks:
            batch_end = min(batch_start + self._batch_scaler.batch_size, self._progress.total_ticks)
            batch_params = self._params_for_batch(self._params, batch_start, batch_end)
            start = datetime.datetime.now(datetime.timezone.utc)
            temp_result, in_size, out_size = self._react_function(self._trainee_id, batch_params)
            end = datetime.datetime.now(datetime.timezone.utc)
            self._progress.update(batch_end - batch_start)
            self._send_progress(temp_result)
            accumulate_react_result(self.result, temp_result)
            self._update_batch_size(end - start, in_size, out_size)
            batch_start = batch_end

    def _consume_future(self) -> None:
        """
        Consume the oldest future in the queue.

        Does nothing if the queue is empty.  Blocks if the future is not already complete.
        """
        if len(self._futures) == 0:
            return
        (batch_size, future) = self._futures.popleft()
        logger.debug("committing batch of size %d", batch_size)
        # (In this next line, future.result() blocks if the future's not already done.)
        temp_result, _in_size, _out_size = future.result()
        self._send_progress(temp_result)
        accumulate_react_result(self.result, temp_result)

    def _consume_ready_futures(self) -> None:
        """Consume any completed futures as the oldest end of the queue."""
        while len(self._futures) > 0 and self._futures[0][1].done():
            self._consume_future()

    def _wait_for_future(self) -> None:
        """Wait for (at least) one future to finish and process its results."""
        done, _not_done = wait(self._running, return_when=FIRST_COMPLETED)
        logger.debug("finished %d batches", len(done))
        for (batch_size, future) in self._futures:
            # Update the progress monitor if this future just finished
            if future in done:
                self._running.remove(future)
                self._progress.update(batch_size)
                logger.debug("finished batch of size %d, total: %d/%d", batch_size, self._progress.current_tick, self._progress.total_ticks)
                # Update the batch scaler if needed
                if self._batch_scaling_future is not None and self._batch_scaling_future[1] is future:
                    start_time = self._batch_scaling_future[0]
                    _temp_result, in_size, out_size = future.result()
                    end_time = datetime.datetime.now(datetime.timezone.utc)
                    old_batch_size = self._batch_scaler.batch_size
                    batch_duration = end_time - start_time
                    self._update_batch_size(batch_duration, in_size, out_size)
                    logger.debug("updating batch size %d -> %d (%s)", old_batch_size, self._batch_scaler.batch_size, batch_duration)
                    # The next batch we submit will get to update the batch size
                    self._batch_scaling_future = None
        # Pop anything we can off the queue
        self._consume_ready_futures()

    def parallel(self) -> None:
        """Run the operation using parallel threads."""
        logger.debug("starting parallel batch react")
        batch_start = 0
        self._send_progress(None)
        executor = ThreadPoolExecutor()
        try:
            while batch_start < self._progress.total_ticks:
                max_running = self._get_concurrency(self._trainee_id) or 1
                if len(self._running) < max_running:
                    # Submit a new batch of cases
                    batch_end = min(batch_start + self._batch_scaler.batch_size, self._progress.total_ticks)
                    batch_params = self._params_for_batch(self._params, batch_start, batch_end)
                    future = executor.submit(self._react_function, self._trainee_id, batch_params)
                    self._futures.append((batch_end - batch_start, future))
                    self._running.add(future)
                    logger.debug("starting batch of size %d (%d/%d)", (batch_end - batch_start), len(self._futures), max_running)
                    if self._batch_scaling_future is None:
                        self._batch_scaling_future = (datetime.datetime.now(datetime.timezone.utc), future)
                    batch_start = batch_end
                else:
                    self._wait_for_future()
            # Now we've submitted all of the data; wait for any outstanding futures to complete.
            while len(self._futures) > 0:
                self._wait_for_future()
        finally:
            # If anything is left running, cancel those futures.  On normal
            # completion we'll have waited for everything we know about.
            executor.shutdown(wait=False, cancel_futures=True)
        logger.debug("finished parallel batch react")

class ParamsForBatch:
    """
    Helper callable class to get slices out of a react parameter set.

    Parameters
    ----------
    slice_keys : Collection[str]
        Names of keys in the parameters map that are lists, and need to be
        sliced with the provided batch range.
    num_to_generate_param : str, optional
        If ``desired_conviction`` is set, then set this parameter to the batch size.
    """
    def __init__(self, slice_keys: Collection[str], num_to_generate_param: str | None = None):
        self._slice_keys = slice_keys
        self._num_to_generate_param = num_to_generate_param

    def __call__(self, params: dict[str, t.Any], batch_start: int, batch_end: int) -> dict[str, t.Any]:
        result = {}
        for k, v in params.items():
            if k in self._slice_keys and v is not None and len(v) > 1:
                result[k] = v[batch_start:batch_end]
            else:
                result[k] = v
        if self._num_to_generate_param and params.get("desired_conviction") is not None:
            result[self._num_to_generate_param] = batch_end - batch_start
        return result


def batch_lists(lists: list[T], initial_batch_size: int) -> Generator[list[T], int]:
    """
    Break up a potentially long list of items into variable-size batches.

    This is a bidirectional generator.  Calling `next()` on its result yields
    a list of `initial_batch_size` items.  From this point, call `gen.send()`
    with a desired number of items, and it will return a list of that many
    items.  The generator raises `StopIteration` when the list is exhausted.

    """
    offset = 0
    batch_size = initial_batch_size
    while offset < len(lists):
        batch = lists[offset:offset+batch_size]
        next_batch_size = yield batch
        offset += batch_size
        batch_size = next_batch_size


def show_core_warnings(core_warnings: Iterable[str | dict]):
    """Warns the user for each warning returned from the core."""
    # Import here to avoid circular import
    from ..client.exceptions import HowsoWarning

    if isinstance(core_warnings, Iterable):
        for w in core_warnings:
            msg = w.get("detail") if isinstance(w, dict) else w
            if isinstance(msg, str):
                warnings.warn(msg, category=HowsoWarning)


def to_pandas_datetime_format(f: str):
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


def fix_feature_value_keys(
    input_dict: dict[str, t.Any],
    feature_attributes: Mapping[str, Mapping],
    feature_name: str
) -> dict[str | float | int, t.Any]:
    """
    Cleans up misformatted keys for a dict with feature values as keys.

    Non-string dictionary keys are converted to strings
    within the JSON-ification process in Amalgam.

    Parameters
    ----------
    input_dict : dict[str, Any]
        The mapping with feature values as keys that may need fixing.
    feature_attributes : Mapping[str, Mapping]
        The feature attributes of the data.
    feature_name : str
        The name of the feature whose feature values make the keys of the dict.

    Returns
    -------
    dict[str | float | int, Any]
        The updated dict with cleaned up feature values as keys.
    """
    output_dict = {}
    for k, v in input_dict.items():
        if k == "(null)":
            output_dict["null"] = v
        else:
            if feature_attributes[feature_name].get('data_type') == 'number':
                if feature_attributes[feature_name].get('original_type', {}).get('data_type') == 'integer':
                    output_dict[int(k)] = v
                else:
                    output_dict[float(k)] = v
            else:
                output_dict[str(k)] = v
    return output_dict

def update_caps_maps(
    caps_maps: list[dict[str, dict[str, float]]],
    feature_attributes: Mapping[str, Mapping]
) -> list[dict[str, dict[str | int | float, float]]]:
    """
    Cleans up misformatted keys from non-string nominal feature's CAP maps.

    Non-string dictionary keys are converted to strings
    within the JSON-ification process in Amalgam.

    Parameters
    ----------
    caps_maps : list[dict[str, dict[str, float]]]
        The list of CAP maps.
    feature_attributes : Mapping[str, Mapping]
        The feature attributes of the data.

    Returns
    -------
    list[dict[str, dict[str | int | float, float]]]
        The updated list of CAP maps with cleaned up feature values as keys.
    """
    updated_caps_maps = []
    for caps_map in caps_maps:
        updated_caps_map = {}

        for feature in caps_map:
            updated_caps_map[feature] = fix_feature_value_keys(
                caps_map[feature],
                feature_attributes,
                feature
            )
        updated_caps_maps.append(updated_caps_map)

    return updated_caps_maps

def update_confusion_matrix(
    confusion_matrix: dict[str, dict[str, float | dict[str, t.Any]]],
    feature_attributes: Mapping[str, Mapping]
) -> dict[str, t.Any]:
    """
    Cleans up misformatted keys from non-string nominal feature's confusion matrices.

    Non-string dictionary keys are converted to strings
    within the JSON-ification process in Amalgam.

    Parameters
    ----------
    confusion_matrix : dict[str, dict[str, float | dict[str, t.Any]]]
        The mapping that defines the confusion matrix.
    feature_attributes : Mapping[str, Mapping]
        The feature attributes of the data.

    Returns
    -------
    dict[str, Any]
        The updated map of confusion matrices for each feature that was given.
    """
    updated_confusion_matrix_map = {}
    for feature, feature_cm_map in confusion_matrix.items():
        updated_feature_cm_map = feature_cm_map.copy()
        updated_feature_cm_map['other_counts'] = fix_feature_value_keys(
            feature_cm_map['other_counts'],
            feature_attributes,
            feature
        )

        # The 'matrix' value is a double nested dict of feature values to feature values to counts.
        # So the inner keys must be fixed in addition to the outer keys.
        updated_matrix = {
            k: fix_feature_value_keys(v, feature_attributes, feature)
            for k, v in feature_cm_map['matrix'].items()
        }
        updated_matrix = fix_feature_value_keys(updated_matrix, feature_attributes, feature)
        updated_feature_cm_map['matrix'] = updated_matrix

        updated_confusion_matrix_map[feature] = updated_feature_cm_map

    return updated_confusion_matrix_map


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
        warning_types: type[Warning] | Iterable[type[Warning]]
    ):
        """Initialize a new `catch_warnings` instance."""
        self._catch_warnings = warnings.catch_warnings()
        if not isinstance(warning_types, Iterable):
            self._warning_types = [warning_types]
        else:
            self._warning_types = warning_types
        for warning_type in self._warning_types:
            self._check_warning_class(warning_type)

    @staticmethod
    def _check_warning_class(warning_type: type[Warning]):
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
