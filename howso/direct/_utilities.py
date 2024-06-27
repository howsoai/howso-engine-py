import datetime
from importlib import metadata
from pathlib import Path
import sysconfig
from typing import Union


def session_convert_datetime(obj):
    """
    Converts datetime attributes stored as strings to datetime objects.

    Parameters
    ----------
    obj : dict or None
        The dict containing the class attributes.

    Returns
    -------
    Dict
        The session object.
    """
    date_attributes = ['created_date', 'modified_date']

    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ValueError('`obj` parameter is not a dict')
    # Only use known attributes for class instantiation
    for key in obj.keys():
        if key in date_attributes and isinstance(obj[key], str):
            obj[key] = datetime.fromisoformat(obj[key])
    return obj


def get_file_in_distribution(file_path) -> Union[Path, None]:
    """
    Locate the LICENSE.txt file in the distribution of this package.

    Parameters
    ----------
    file_path : str
        The name/path of the desired file relative to the package distribution.

    Returns
    -------
    Path or None
        The path to the requested file or None, if not found.
    """
    purelib_path = sysconfig.get_path('purelib')
    dist = metadata.distribution('howso-engine')
    for fp in dist.files:
        if fp.name == file_path:
            return Path(purelib_path, fp)
