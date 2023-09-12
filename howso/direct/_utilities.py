from datetime import datetime
from importlib import metadata
from pathlib import Path
import sysconfig
from typing import Union


def model_from_dict(klass, obj):
    """
    Create OpenAPI model instance from dict.

    Parameters
    ----------
    klass : Type
        The class to instantiate.
    obj : dict or None
        The dict containing the class attributes.

    Returns
    -------
    Any
        The class instance.
    """
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ValueError('`obj` parameter is not a dict')
    if not hasattr(klass, 'attribute_map'):
        raise ValueError("`klass` is not an OpenAPI model")
    # Only use known attributes for class instantiation
    parameters = dict()
    for key in obj.keys():
        if key in klass.attribute_map:
            dtype = klass.openapi_types[key]
            if dtype == 'datetime':
                parameters[key] = datetime.fromisoformat(obj[key])
            else:
                parameters[key] = obj[key]
    return klass(**parameters)


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
