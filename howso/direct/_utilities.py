from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sysconfig


def get_file_in_distribution(file_path: str) -> Path | None:
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
    if dist.files is None:
        raise AssertionError("The package howso-engine is not installed correctly, please reinstall.")
    for fp in dist.files:
        if fp.name == file_path:
            return Path(purelib_path, fp)
