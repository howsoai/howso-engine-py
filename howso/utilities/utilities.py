import datetime as dt
import inspect
import locale as python_locale
from math import isnan
import re
import sys
import threading
from typing import (
    Collection, Dict, Iterable, List, Mapping, Optional, Tuple, Union)
import uuid
import warnings

from dateutil.parser import isoparse
from dateutil.parser import parse as dt_parse
from dateutil.tz import tzoffset
from howso.openapi.models import (
    FeatureAttributes,
    Trainee
)
import numpy as np
import pandas as pd

from .internals import serialize_openapi_models


_BASE_FEATURE_TYPES = ["nominal", "continuous", "ordinal"]
# Custom type for case_indices parameter
CaseIndices = Iterable[Union[List[Union[str, int]], Tuple[Union[str, int]]]]
DATETIME_TIMEZONE_PATTERN = re.compile(r"(?<!%)(?:%%)*(%z)", re.IGNORECASE)
DATETIME_UTC_Z_PATTERN = re.compile(r"\dZ$")
EPOCH = dt.datetime.utcfromtimestamp(0)
ISO_8601_DATE_FORMAT = "%Y-%m-%d"
ISO_8601_FORMAT = "%Y-%m-%dT%H:%M:%S"
ISO_8601_FORMAT_FRACTIONAL = "%Y-%m-%dT%H:%M:%S.%f"
# The number of Sequences to check in a non-thorough validation of case_indices
NON_THOROUGH_NUM = 100
# Match unescaped timezone character in datetime format strings
SMALLEST_TIME_DELTA = 0.001


def trainee_from_df(df, features: Optional[Mapping[str, Mapping]] = None,
                    action_features: Optional[Iterable[str]] = None,
                    name: Optional[str] = None,
                    persistence: str = 'allow',
                    trainee_metadata: Optional[Mapping] = None,
                    ) -> Trainee:
    """
    Create a Trainee from a dataframe.

    Assumes floats are continuous and all other values are nominal.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas Dataframe with column names corresponding to feature
        names.Features that are considered to be continuous should have a dtype
        of float.

    features : Optional[Mapping[str, Mapping]]
        (Optional) A dictionary of feature names to a dictionary of parameters.

    action_features : List of String, Default None
        (Optional) List of action features. Anything that's not in this list
        will be treated as a context feature. For example, if no action feature
        is specified the trainee won't have a target.

    name : str or None, defaults to None
        (Optional) The name of the trainee.

    persistence : str: default "allow"
        The persistence setting to use for the trainee. Valid values:
        "always", "allow", "never".

    trainee_metadata : Mapping, optional
        (Optional) mapping of key/value pairs of metadata for trainee.

    Returns
    -------
    howso.openapi.models.Trainee
        A trainee object
    """
    # Place this here to avoid circular imports
    from howso.utilities.feature_attributes import infer_feature_attributes
    action_features = [] if action_features is None else list(action_features)

    if features is None:
        features = infer_feature_attributes(df)

    context_features = []
    for fname, _ in features.items():
        if fname not in action_features:
            context_features.append(fname)

    return Trainee(name, features=features,
                   default_context_features=context_features,
                   default_action_features=action_features,
                   persistence=persistence,
                   metadata=trainee_metadata)


def date_to_epoch(
    date_obj: Union[dt.date, dt.datetime, dt.time, str],
    time_format: str
) -> Optional[Union[str, float]]:
    """
    Convert date into epoch (i.e seconds counted from Jan 1st 1970).

    .. note::
        If `date_str` is None or nan, it will be returned as is.

    Parameters
    ----------
    date_obj : str or datetime.date or datetime.time or datetime.datetime
        Time object.
    time_format : str
        Specify format of the time.
        Ex: ``%a %b %d %H:%M:%S %Y``

    Returns
    -------
    Union[str, float]
        The epoch date as a floating point value or 'np.nan', et al.
    """
    # pd.isnull covers the cases - None, `np.nan` and `pd.na`
    if pd.isnull(date_obj):
        return date_obj

    # if timestamp is passed in, convert it to string in the correct
    # format first
    if isinstance(date_obj, (dt.date, dt.datetime)):
        date_str = date_obj.strftime(time_format)
    elif isinstance(date_obj, dt.time):
        return time_to_seconds(date_obj)
    else:
        date_str = str(date_obj)

    # if there is time zone info in the format, use dt_parse because
    # datetime.strptime doesn't handle time zones
    if DATETIME_TIMEZONE_PATTERN.findall(time_format):
        datetime_object = dt_parse(date_str)
        time_zero = dt.datetime(1970, 1, 1, tzinfo=datetime_object.tzinfo)
    else:
        datetime_object = dt.datetime.strptime(date_str, time_format)
        time_zero = dt.datetime(1970, 1, 1)

    return (datetime_object - time_zero).total_seconds()


def epoch_to_date(epoch: Union[str, float], time_format: str,
                  tzinfo: Optional[dt.tzinfo] = None) -> str:
    """
    Convert epoch to date if epoch is not `None` or `nan` else, return as it is.

    Parameters
    ----------
    epoch : Union[str, float]
        The epoch date as a floating point value (or str if np.nan, et al)
    time_format : str
        Specify format of the time.
        Ex: ``%a %b %d %H:%M:%S %Y``
    tzinfo : datetime.tzinfo, optional
        Time zone information to include in datetime.

    Returns
    -------
    str
        A date string in the format similar to "Wed May 21 00:00:00 2008"
    """
    # pd.isnull covers the cases - None, `np.nan` and `pd.na`
    if pd.isnull(epoch):
        return epoch

    dt_value = (EPOCH + dt.timedelta(seconds=epoch))
    if tzinfo is not None:
        dt_value = dt_value.replace(tzinfo=tzinfo)
    return dt_value.strftime(time_format)


def time_to_seconds(time: Optional[dt.time]) -> Optional[float]:
    """
    Convert a time object to seconds since midnight.

    Parameters
    ----------
    time : datetime.time
        The time to convert.

    Returns
    -------
    float
        Seconds since midnight.
    """
    if not isinstance(time, dt.time):
        return None
    date = dt.datetime.combine(dt.date.min, time)
    delta = date - dt.datetime(1, 1, 1, tzinfo=time.tzinfo)
    return delta.total_seconds()


def seconds_to_time(seconds: Union[int, float, None], *,
                    tzinfo: Optional[dt.tzinfo] = None) -> Optional[dt.time]:
    """
    Convert seconds to a time object.

    Parameters
    ----------
    seconds : int or float
        The seconds to convert to time.
    tzinfo : datetime.tzinfo, optional
        Time zone to use for resulting time object.

    Returns
    -------
    datetime.time
        The time object.
    """
    if pd.isnull(seconds):
        return None
    time_value = (dt.datetime.min + dt.timedelta(seconds=seconds)).time()
    if tzinfo:
        return time_value.replace(tzinfo=tzinfo)
    return time_value


def replace_none_with_nan(dat: Mapping) -> List[Dict]:
    """
    Replace None values with NaN values.

    For use when retrieving data from Howso via the scikit module to
    conform to sklearn convention on missing values.

    Parameters
    ----------
    dat : list of dict of key-values

    Returns
    -------
    list[dict]
    """
    return [
        {
            key: float('nan') if value is None else value
            for key, value in action.items()
        } for action in dat
    ]


def replace_nan_with_none(dat):
    """
    Replace None values with NaN values.

    For use when feeding data to Howso from the scikit module to account
    for the different ways howso and sklearn represent missing values.

    Parameters
    ----------
    dat : list of list of object
        A 2d list of values.

    Returns
    -------
    list[list[object]]
    """
    return [[None if isinstance(value, (int, float)) and isnan(value) else
             value for value in case] for case in dat]


def reshape_data(x, y):
    """
    Reshapes X as a matrix and y as a vector.

    Parameters
    ----------
    x : np.ndarray
        Feature values ndarray.
    y : np.ndarray
        target values ndarray.

    Returns
    -------
    np.ndarray, np.ndarray
        X, y
    """
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)
    if len(y.shape) > 1:
        y = y.reshape(-1)
    return x, y


def align_data(x, y=None):
    """
    Check and fix type problems with the data and reshape it.

    x is a Matrix and y is a vector.

    Parameters
    ----------
    x : numpy.ndarray
        Feature values ndarray.
    y : numpy.ndarray, default None
        Target values ndarray.

    Returns
    -------
    numpy.ndarray, numpy.ndarray or numpy.ndarray
    """
    if x.dtype == object:
        x = x.astype(float)
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)
    if y is not None:
        if y.dtype == object:
            y = y.astype(float)
        return x, y
    return x


def replace_doublemax_with_infinity(dat):
    """
    Replace values of Double.MAX_VALUE (1.79769313486232E+308) with Infinity.

    For use when retrieving data from Howso.

    Parameters
    ----------
    dat : A dict, list, number, or string

    Returns
    -------
    A dict, list, number, or string - same as passed in for translation
    """
    if isinstance(dat, dict):
        dat = {k: replace_doublemax_with_infinity(v) for (k, v) in dat.items()}
    elif isinstance(dat, list):
        dat = [replace_doublemax_with_infinity(item) for item in dat]
    elif dat == sys.float_info.max:
        dat = float('inf')

    return dat


def dprint(debug, *argc, **kwargs):
    """
    Print based on debug levels.

    Parameters
    ----------
    debug : bool or int
        If true, user_debug level would be 1.
        Possible levels: 1, 2, 3 (print all)
    kwargs:
        default_priority : int, default 1
            The message is printed only if the `debug` >= `default_priority`.

    Examples
    --------
    >>> dprint(True, "hello", "howso", priority=1)
    `hello howso`
    """
    if debug:
        user_priority = debug
        if not isinstance(debug, int):
            user_priority = 1

        priority = kwargs.get("default_priority", 1)
        if user_priority >= priority:
            for item in argc:
                print(item, end=" ")
            if "end" in kwargs:
                print("", end=kwargs["end"])
            else:
                print()


def determine_iso_format(str_date: str, fname: str) -> str:  # noqa: C901
    """
    Determine which specific ISO8601 format the passed in date is in.

    Specifically if it's just a date, if it's zoned, and if zoned, whether it's
    a zone or an offset.

    Parameters
    ----------
    str_date: str
        The Date time passed in as a string.
    fname: str
        Name of feature to guess bounds for.

    Returns
    -------
    str
        The ISO_8601 format string that most matches the passed in date.
    """
    # parse with the standard parser first to support single digit month/day
    dt_object = dt_parse(str_date)
    if dt_object.tzinfo is None:
        # warn user if this date format is a subset of the ISO_8601
        warn_user = True

        try:
            # do a stricter check for iso format
            isoparse(str_date)
        except Exception:  # noqa: Intentionally broad
            # user was already warned, don't warn them again
            warn_user = False

        if len(str_date) <= 10:
            try:
                dt.datetime.strptime(str_date, ISO_8601_DATE_FORMAT)
                return ISO_8601_DATE_FORMAT
            except ValueError:
                if warn_user:
                    warnings.warn(f"Feature {fname} is a datetime but may not "
                                  f"work properly if user doesn't specify "
                                  f"the correct format.")
                return ISO_8601_FORMAT

        try:
            dt.datetime.strptime(str_date, ISO_8601_FORMAT_FRACTIONAL)
            return ISO_8601_FORMAT_FRACTIONAL
        except ValueError:
            pass

        try:
            dt.datetime.strptime(str_date, ISO_8601_FORMAT)
            return ISO_8601_FORMAT
        except ValueError:
            if warn_user:
                warnings.warn(f"Feature {fname} is a datetime but may not "
                              f"work properly if user doesn't specify "
                              f"the correct format.")
            return ISO_8601_FORMAT

    # detect iso formats ending in Z, signifying UTC
    if (
        dt_object.utcoffset() == dt.timedelta(0) and
        DATETIME_UTC_Z_PATTERN.findall(str_date)
    ):
        try:
            dt.datetime.strptime(str_date, ISO_8601_FORMAT_FRACTIONAL + "Z")
            return ISO_8601_FORMAT_FRACTIONAL + "Z"
        except ValueError:
            return ISO_8601_FORMAT + "Z"

    # date has time zone info, determine whether it's an offset
    if isinstance(dt_object.tzinfo, tzoffset):
        return "%Y-%m-%dT%H:%M:%S%z"

    # offsets of +0000 or -0000 won't parse as a TZ offset, but
    # the format still matches using %z, thus check if last char is '0'
    if str_date[-1] == '0':
        return "%Y-%m-%dT%H:%M:%S%z"

    return "%Y-%m-%dT%H:%M:%S%Z"


def validate_list_shape(values: Union[Collection, None], dimensions: int,
                        variable_name: str, var_types: str,
                        allow_none: bool = True
                        ) -> None:
    """
    Validate the shape of a list.

    Raise a ValueError if it does not match expected number of dimensions.

    Parameters
    ----------
    values : Collection or None
        A single or multidimensional list.
    dimensions : int
        The number of dimensions the list should be.
    variable_name : str
        The variable name for output.
    var_types : str
        The expected type of the data.
    allow_none : bool, default True
        If None should be allowed.
    """
    if values is None:
        if not allow_none:
            raise ValueError(
                f"Invalid value for `{variable_name}`, must not be `None`")
        return
    if len(np.array(values).shape) != dimensions:
        raise ValueError(
            f"Improper shape of `{variable_name}` values passed. "
            f"`{variable_name}` must be a {dimensions}d list of {var_types}.")


def validate_case_indices(case_indices: CaseIndices, thorough=False) -> None:
    """
    Validate the case_indices parameter to the react() method of a Howso client.

    Raises a ValueError if case_indices has sequences that do not contain the expected
    data types of (str, int).

    Parameters
    ----------
    case_indices : Iterable of Sequence[str, int]
        The case_indices argument to validate.
    thorough : bool, default False
        Whether to verify the data types in all sequences or only some (for performance)
    """
    try:
        amount_to_verify = case_indices[:len(case_indices) if thorough else NON_THOROUGH_NUM]
        if (
            not all((
                len(sequence) == 2 and
                not isinstance(sequence, str) and
                isinstance(sequence[0], str) and
                isinstance(sequence[1], (np.integer, int))
            ) for sequence in amount_to_verify)
        ):
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('Argument case_indices must be type Iterable of (non-string) Sequence[str, int].')


def num_list_dimensions(lst):
    """
    Return number of dimensions for a list.

    Assumption is that the input nested lists are also lists,
    or a list of dataframes.

    Parameters
    ----------
    lst : list
        The nested list of objects.

    Returns
    -------
    int
        The number of dimensions in the passed in list.
    """
    the_list = lst
    d = 0
    while True:
        if not isinstance(the_list, list):
            if isinstance(the_list, pd.DataFrame):
                # add the number of dimensions in the dataframe
                d += the_list.ndim
            break
        try:
            the_list = the_list[0]
            d += 1
        except (IndexError, TypeError):
            break
    return d


def validate_features(features: Mapping[str, Union[FeatureAttributes, Mapping]],
                      extended_feature_types: Optional[Iterable[str]] = None
                      ) -> None:
    """
    Validate the feature types in `features`.

    Parameters
    ----------
    features : dict
        The dict of feature name to feature attributes.

        The valid feature names are:

        a. "nominal"
        b. "continuous"
        c. "ordinal"
        d. along with passed in `extended_feature_types`
    extended_feature_types : list of str, optional
        (Optional) If a list is passed in, the feature types specified in the
        list will be considered as valid features.
    """
    valid_feature_types = _BASE_FEATURE_TYPES

    if extended_feature_types is not None:
        # Append extended types to valid feature types
        valid_feature_types += list(extended_feature_types)

    for f_name, f_desc in features.items():
        if isinstance(f_desc, FeatureAttributes):
            f_type = f_desc.type
        else:
            f_type = f_desc.get("type")
        if f_type not in valid_feature_types:
            raise ValueError(f"The feature name '{f_name}' has invalid "
                             f"feature type - '{f_type}'")


def validate_datetime_iso8061(datetime_value, feature):
    """
    Check that the passed in datetime value adheres to the ISO 8601 format.

    Warn the user if it doesn't check out.

    Parameters
    ----------
    datetime_value : str
        The date value as a string
    feature : str
        Name of feature
    """
    try:
        # general iso8601 checker, allows various 8601 formats
        isoparse(datetime_value)
    except Exception:  # noqa: Intentionally broad
        warnings.warn(
            f"Feature {feature} detected as having datetime values, but are "
            f"not in an ISO 8601 format, such as '{ISO_8601_FORMAT}', for "
            f"example: '2020-10-02T12:43:39'")


def serialize_datetimes(cases: List[List], columns: Iterable[str],  # noqa: C901
                        features: Dict, *, warn: bool = False) -> None:
    """
    Serialize datetimes in the given list of cases, in-place.

    Iterate over the passed in case values and serializes any datetime
    values according to the specified datetime format in feature attributes.

    Parameters
    ----------
    cases : list of list
        A 2d list of case values corresponding to the features of the cases.
    columns : list of str
        A list of feature names.
    features : dict
        Dictionary of feature attributes.
    warn : bool, default: False
        If set to true, will warn user when specified datetime format
        doesn't match the datetime strings.
    """
    # Import here to avoid circular import dependencies
    from .features import FeatureType
    features = serialize_openapi_models(features)
    if isinstance(columns, Iterable):
        columns = list(columns)

    # Populate the list of indices for the specified features for all
    # features that contain datetime values.
    dt_indices = list()
    for i, feature_name in enumerate(columns):
        if feature_name not in features:
            # Feature has no feature attributes, skip
            continue
        feature = features[feature_name]
        date_time_format = feature.get("date_time_format")
        try:
            original_data_type = feature['original_type']['data_type']
        except (TypeError, KeyError):
            original_data_type = None

        if (
            date_time_format is not None or
            original_data_type == FeatureType.TIME.value
        ):
            # If datetime format defined or time only without format we need to
            # serialize the value
            dt_indices.append(i)

    if not dt_indices:
        # If no datetime features found, nothing to serialize
        return

    warned_features = set()
    for case in cases:
        if case is None:
            continue

        for i in dt_indices:
            feature_name = columns[i]
            feature = features[feature_name]
            dt_value = case[i]
            dt_format = feature.get("date_time_format")
            locale = feature.get("locale")

            # NaN may be passed from a dataframe if a value is null.
            if repr(dt_value).lower() in ['nan', 'inf', 'nat']:
                dt_value = None
            # if the value is a datetime object, just serialize it
            elif isinstance(dt_value, (dt.date, dt.datetime)):
                if pd.isnull(dt_value):
                    dt_value = None
                else:
                    dt_value = dt_value.strftime(dt_format)
            elif isinstance(dt_value, dt.time):
                if pd.isnull(dt_value):
                    dt_value = None
                elif not dt_format:
                    dt_value = time_to_seconds(dt_value)
                else:
                    dt_value = dt.datetime.combine(dt.date(1970, 1, 1),
                                                   dt_value)
                    dt_value = dt_value.strftime(dt_format)
            # deal with string dates using default locale
            elif isinstance(dt_value, str):
                if locale is None:
                    bad_format = False
                    try:
                        # convert to datetime object, then serialize
                        dt_value = dt.datetime.strptime(
                            dt_value, dt_format
                        ).strftime(dt_format)
                    except ValueError:
                        bad_format = True
                        if warn and feature_name not in warned_features:
                            warnings.warn(
                                f"{feature_name} has values with incorrect "
                                f"datetime format, should be {dt_format}. "
                                f"This feature may not work properly."
                            )
                            warned_features.add(feature_name)

                    # if the format was ISO_8601 but the date string
                    # didn't match it, attempt to parse it
                    if bad_format and dt_format in ISO_8601_FORMAT:
                        try:
                            dt_object = dt_parse(dt_value)
                            dt_value = dt.datetime.strftime(dt_object,
                                                            dt_format)
                        except Exception:  # noqa: Intentionally broad
                            # do nothing because we already know that this is
                            # a bad format value and already warned the user
                            pass
                else:
                    # Given a locale string, use it instead of the system
                    # locale.
                    # NOTE: The provided locale must be installed on the
                    # system.
                    try:
                        with LocaleOverride(language_code=locale,
                                            category=python_locale.LC_TIME):
                            dt_value = (
                                dt.datetime.strptime(dt_value, dt_format)
                                           .strftime(dt_format))
                    except python_locale.Error:
                        warnings.warn(
                            f"The locale provided: '{locale}' does not appear "
                            f"to be available on this system. The provided "
                            f"value is left unchanged. This feature may not "
                            f"work properly until it is installed.")
                    except (TypeError, ValueError):
                        warnings.warn(
                            f"{feature_name} has values with incorrect "
                            f"datetime format for the given locale, should be "
                            f"{dt_format}. The provided value is left "
                            f"unchanged. This feature may not work properly.")
            else:
                # At this point, no valid date type has been processed. This
                # could mean the value is None which is an acceptable input.
                # Otherwise it meant that the value is malformed or of the
                # wrong type (ex int or other non datetime object) and a
                # warning should be issued and the value replaced with None.
                if dt_value is not None:
                    warnings.warn(
                        f"{feature_name} has a malformed value {dt_value} "
                        f"that cannot be parsed. Expected datetime formatted "
                        f"string or object. Replacing with None.")
                dt_value = None

            # store the serialized datetime value
            case[i] = dt_value


def is_valid_uuid(value, version=4):
    """
    Check if a given string is a valid uuid.

    Parameters
    ----------
    value : str or UUID
        The value to test
    version : int, optional
        The uuid version (Default: 4)

    Returns
    -------
    bool
        True if `value` is a valid uuid string
    """
    try:
        uuid_obj = uuid.UUID(str(value), version=version)
    except (TypeError, ValueError):
        return False
    return str(uuid_obj) == value


class LocaleOverride:
    """
    Implements a thread-safe context manager for switching locales temporarily.

    Background
    ----------
    Python's locale.setlocale() is not thread safe. In order to work with
    alternate locales temporarily, this ContextDecorator will use a thread
    lock on __enter__ and release said lock on __exit__.

    Important Notes
    ---------------
    All other threads will be blocked within the scope of the context. It is
    important to avoid time-consuming execution inside.

    Example Usage
    -------------
    >>> # Parse date string from French and format it in English.
    >>>
    >>> # System locale is 'en-us' (in this example)
    >>> from datetime import datetime
    >>> dt_format = '<some format>'
    >>> dt_obj = datetime()
    >>> with locale_override('fr-fr', category=locale.LC_DATE):
    >>>     # We're in French date-formatting zone here...
    >>>     date_obj = datetime.strptime(dt_value, dt_format)
    >>>
    >>> # Back in the 'en-us' locale again.
    >>> dt_value = dt_obj.strftime(dt_format)

    Parameters
    ----------
    language_code : str
        A language code /usually/ given as either:
            - 2 lower case letters for the base language Ex: `fr` for French.
            - 5 characters such as `fr_CA` where the first 2 designate the
              base language (French in this example) followed by an `_`
              followed by 2 upper case characters designating the country-
              specific dialect (Canada, in this example). This example
              designates the French-Canadian locale.
            - Any of the above, plus an optional encoding following a '.' Ex:
              `fr_FR.UTF-8`
    encoding : str
        An encoding such as 'UTF-8' or 'ISO8859-1', etc. If not provided and
        there is no embedded encoding within the language_code parameter,
        'UTF-8' is used. If an encoding is embedded in the `language_code`
        parameter and an explicit encoding provided here, the embedded encoding
        is dropped and ignored.
    category : int
        This is one of the constants set within the locale object.
        See: https://docs.python.org/3.9/library/locale.html for details.
        `locale.LC_ALL` is used if nothing provided.
    """

    def __init__(self, language_code, encoding=None,
                 category=python_locale.LC_ALL):
        """Construct the context manager."""
        if '.' in language_code:
            language_code, embedded_encoding = language_code.split('.', 1)
        else:
            embedded_encoding = 'UTF-8'

        if encoding is None:
            encoding = embedded_encoding

        self.new_locale = (language_code, encoding)
        self.category = category
        self.lock = threading.Lock()

    def setup(self):
        """
        Set a thread lock and the locale as desired.

        Use this method directly to setup a locale context when not using this
        class as a context manager.
        """
        self.lock.acquire()
        self.old_locale = python_locale.getlocale(self.category)
        python_locale.setlocale(self.category, self.new_locale)

    def restore(self):
        """
        Restore the original locale and release the thread lock.

        Use this method directly to restore the current context when not using
        this class as a context manager.
        """
        python_locale.setlocale(self.category, self.old_locale)
        if self.lock and self.lock.locked():
            self.lock.release()

    def __enter__(self):
        """Set a thread lock and the locale as desired."""
        self.setup()

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the original locale and release the thread lock."""
        self.restore()


class StopExecution(Exception):
    """Raise a StopExecution as this is a cleaner `exit()` for Notebooks."""

    def _render_traceback_(self):
        pass


class UserFriendlyExit:
    """
    Return a callable that, when called, simply prints `msg` and cleanly exits.

    Parameters
    ----------
    verbose : bool
        If True, emit more information
    """

    def __init__(self, verbose=False):
        """Construct a UserFriendlyExit instance."""
        self.verbose = verbose

    def __call__(self, msg="An unexpected exit occurred.", exception=None):
        """
        Exit, but print the exception first.

        Parameters
        ----------
        msg : str
            The **user-friendly** message to display before exiting.
        exception : Exception
            An exception to produce the message from.

        Returns
        -------
        None
        """
        print("Howso Client Error: " + msg)
        if self.verbose and exception is not None:
            print("More information:\n")
            print(str(exception))

        raise StopExecution


def get_kwargs(kwargs, descriptors, warn_on_extra=False):  # noqa: C901
    """
    Decompose kwargs into a tuple of return values.

    Each tuple corresponds to a descriptor in 'descriptors'. Optionally issue a
    warning on any items in `kwargs` that are not "consumed" by the descriptors.

    Parameters
    ----------
    kwargs : dict
        Mapping of keys and values (kwargs)

    descriptors :
        An iterable of descriptors for how to handle each item in kwargs. Each
        descriptor can be a mapping, another iterable, or a single string.

        If a mapping, it must at least include the key: 'key' but can also
        optionally include the keys: 'default' and 'test'.

        If a non-mapping iterable, the values will be interpreted as 'key'
        'default', 'test, in that order. Only the first is absolutely required
        the remaining will be evaluated to `None` if not provided.

        If a string provided, it is used as the 'key'. 'default' and 'test are
        set to `None`.

        If a 'key' is not found in the kwargs, then the 'default' value is
        returned.

        If a descriptor contains a 'test', it should be a callable that returns
        a boolean. If False, the 'default' value is returned.

        If the 'default' provided is an instance of an Exception, then, the
        exception is raised when the 'key' is not present, or the 'test' fails.

    warn_on_extra : bool
        If `True`, will issue warnings about any keys provided in kwargs that
        were not consumed by the descriptors. Default is `False`

    Returns
    -------
        A tuple of the found values in the same order as the
        provided descriptor.

    Raises
    ------
        May raise any exception given as a 'default' in the
        `descriptors` parameter.

    Usage
    -----
    An example of usage showing various ways to use descriptors:

    >>> def my_method(self, required, **kwargs):
    >>>     apple, banana, cherry, durian, elderberry = get_kwargs(kwargs, (
    >>>         # A simple string is interpreted as the 'key' with 'default of
    >>>         # `None` and no test. Very common use-case made simple.
    >>>         'apple',
    >>>
    >>>         # Another common use-case. Set value to 5 if not in kwargs.
    >>>         # This also shows using an tuple for the descriptor.
    >>>         ('banana', 5),
    >>>
    >>>         # Verbose input including a test using dict
    >>>         {'key': 'cherry', 'default': 5, 'test': lambda x: x > 0},
    >>>
    >>>         # The test, `is_durian`, is defined elsewhere
    >>>         ('durian', None, is_durian),
    >>>
    >>>         # Full example using iterable descriptor rather than mapping.
    >>>         ('elderberry', ValueError('"elderberry" must be > 5.'),
    >>>             lambda x: x > 5),
    >>>     ))

    """
    returns = []
    for descriptor in descriptors:
        if isinstance(descriptor, Mapping):
            key = descriptor['key']  # 'key' is required
            default = descriptor.get('default', None)
            test = descriptor.get('test', None)
        elif isinstance(descriptor, str):
            # A naked key name was provided.
            key, default, test = descriptor, None, None
        elif isinstance(descriptor, Collection):
            descriptor = list(descriptor)
            descriptor.extend([None, None])
            key, default, test = descriptor[0:3]
        else:
            raise ValueError('Each item of `descriptors` should be either a '
                             'dict, an ordered Iterable or a string.')

        try:
            value = kwargs.pop(key)
        except KeyError:
            # If the value doesn't exist, set the value to the default, unless
            # the default given is an instance of an exception, in that case,
            # raise it.
            if isinstance(default, Exception):
                raise default
            else:
                value = default

        if callable(test):
            if test(value):
                returns.append(value)
            else:
                if isinstance(default, Exception):
                    raise default
                else:
                    returns.append(default)
        else:
            returns.append(value)

    if warn_on_extra:
        # Attempt to provide the context of the caller...
        try:
            caller = inspect.stack()[1][3]
        except Exception:  # noqa: Intentionally broad
            caller = '[unknown]'
        num_extra_kwargs = len(kwargs)
        if num_extra_kwargs == 1:
            unexpected_param = list(kwargs.keys())[0]
            warnings.warn(f'The function/method "{caller}" received an '
                          f'unexpected parameter "{unexpected_param}". '
                          f'This will be ignored.')
        elif num_extra_kwargs > 1:
            params = ', '.join(list(kwargs.keys()))
            warnings.warn(f'The function/method "{caller}" received '
                          f'unexpected parameters: [{params}]. '
                          f'These will be ignored.')

    return tuple(returns)


def check_feature_names(features: Mapping,
                        expected_feature_names: Collection,
                        raise_error: bool = False) -> bool:
    """
    Check if features in `features` dict matches `expected_feature_names`.

    Parameters
    ----------
    features : Mapping
        A feature dictionary that maps feature names to its attributes.
    expected_feature_names : Collection
        A list (or a set) of expected column names in the given `features`
        dictionary.
    raise_error : bool, defaults to False
        Raise a value error in case the feature names doesn't match between
        `features` and `expected_feature_names`.

    Returns
    -------
    bool
        Returns `True` if the feature names in `features` matches the
        expected feature names passed via `expected_feature_names`. Otherwise,
        returns `False`.

    Raises
    ------
        If `raise_error` is `True`, raises `ValueError` to indicate that
        the feature names in `features` dict doesn't match the feature names
        `expected_feature_names`.
    """
    feature_names = set(features.keys())

    if isinstance(expected_feature_names, list):
        expected_feature_names = set(expected_feature_names)

    ret_value = True
    if feature_names != expected_feature_names:
        ret_value = False
        if raise_error:
            raise ValueError(
                f"The feature names in `features` differs from "
                f"`expected_feature_names`\n"
                f"Trained features names: {set(feature_names)} \n"
                f"Input column names: {expected_feature_names} \n"
                f"Please make sure that the feature names matches the columns "
                f"of the input DataFrame."
            )

    return ret_value


def build_react_series_df(react_series_response, series_index=None):
    """
    Build a DataFrame from the response from react_series.

    If series_index is set, use that
    as a name for an additional feature that will be the series index.

    Parameters
    ----------
    react_series_response : Dictionary
        The response dictionary from a call to react_series.
    series_index : String
        The name of the series index feature, which will index each series in
        the form 'series_<idx>', e.g., series_1, series_1, ..., series_n.
        If None, does not include the series index feature in the returned
        DataFrame.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame defined by the action features and series data in
        the react_series response. Optionally includes a series index feature.
    """
    # Columns are defined by action_features
    columns = react_series_response['action_features']
    # Series data from the response
    series = react_series_response['series']

    if series_index:
        # If series_index is specified, include it as a feature
        columns.insert(0, series_index)
        # >> 'row.insert(...) or row':
        # Insert the series index into the row and append that row to
        # the 'data' list immediately. 'insert()' returns None, so 'None or row' will resolve
        # to 'row' which will, at that point, have the series index inserted.
        #
        # >> 'row for idx, sublist in enumerate(series) for row in sublist':
        # Each item (aka 'sublist') in series is a list of data rows; evaluate every row in
        # sublist for every sublist in series.
        data = [row.insert(0, f'series_{idx + 1}') or row
                for idx, sublist in enumerate(series)
                for row in sublist]
    else:
        # Else use just the data returned by react_series
        data = [row for sublist in series for row in sublist]

    return pd.DataFrame(data, columns=columns)


def date_format_is_iso(f):
    """
    Check if datetime format is ISO8601.

    Does format match the iso8601 set that can be handled by the C parser?
    Generally of form YYYY-MM-DDTHH:MM:SS - date separator can be different
    but must be consistent.  Leading 0s in dates and times are optional.

    Sourced from Pandas:
    https://github.com/pandas-dev/pandas/blob/v1.5.3/pandas/_libs/tslibs/parsing.pyx
    """
    iso_template = '%Y{date_sep}%m{date_sep}%d{time_sep}%H:%M:%S{micro_or_tz}'.format
    excluded_formats = ['%Y%m%d', '%Y%m', '%Y']

    for date_sep in [' ', '/', '\\', '-', '.', '']:
        for time_sep in [' ', 'T']:
            for micro_or_tz in ['', 'Z', '%z', '%Z', '.%f', '.%f%z', '.%f%Z', '.%fZ']:
                if (iso_template(date_sep=date_sep,
                                 time_sep=time_sep,
                                 micro_or_tz=micro_or_tz,
                                 ).startswith(f) and f not in excluded_formats):
                    return True
    return False


def deep_update(base, updates):
    """
    Update dict `base` with updates from dict `updates` in a "deep" fashion.

    NOTE: This is a recursive function. Care should be taken to ensure that
    neither of the input dictionaries are self-referencing.

    Parameters
    ----------
    base : dict
        A dictionary
    updates : dict
        A dictionary of updates

    Returns
    -------
    dict : The updated dictionary.
    """
    if all((isinstance(d, dict) for d in (base, updates))):
        for k, v in updates.items():
            base[k] = deep_update(base.get(k), v)
        return base
    return updates
