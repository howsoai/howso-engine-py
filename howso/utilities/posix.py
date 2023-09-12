from ctypes import (
    byref, c_int32, c_int64, c_uint, cast,
    CDLL, create_string_buffer, POINTER,
)
from ctypes.util import find_library
import sys
from typing import Union


_LIBC = None


class PlatformError(Exception):
    """Wrong platform."""


class CLibError(Exception):
    """Problem instantiating the C Library."""


def sysctl_by_name(name: Union[str, bytes], output_type: str = 'raw',
                   encoding: str = "UTF-8") -> Union[bytes, int, str]:
    """
    Call `sysclt` with the provided key and return the result.

    Maintains a single connection to the C library per process.

    Parameters
    ----------
    name : str or bytes
        The key to use with `sysctl`. If given as `str`, will be decoded using
        the given `encoding`.
    output_type : str, default 'raw'
        Adapt the response into the given type: `int`, `str` and `raw` are
        currently supported.
    encoding : str, default "UTF-8"
        The locale to encode/decode strings from/to.

    Returns
    -------
    bytes, int or str
        The result of the `sysctl` call, possibly modified by the
        given `output_type`.

    Raises
    ------
    PlatformError
        If an attempt to call this method on a non-Posix system
    CLibError
        If unable to instantiate the C library.
    """
    global _LIBC

    if sys.platform == "win32":
        raise PlatformError('Unable to use `sysctl` on Win32 platforms.')

    if not _LIBC:
        try:
            _LIBC = CDLL(find_library("c"))
        except Exception as e:  # Deliberately broad
            raise CLibError('Unable to instantiate the C library.') from e

    size = c_uint(0)

    if isinstance(name, str):
        name = bytes(name, encoding=encoding)

    # Call to get buffer size...
    _LIBC.sysctlbyname(name, None, byref(size), None, 0)
    # Create the buffer
    buf = create_string_buffer(size.value)
    # Call with sized buffer
    _LIBC.sysctlbyname(name, buf, byref(size), None, 0)

    if output_type == 'int':
        if size.value == 4:
            return cast(buf, POINTER(c_int32)).contents.value
        if size.value == 8:
            return cast(buf, POINTER(c_int64)).contents.value
    elif output_type == 'str':
        return buf.value.decode(encoding=encoding)
    else:  # return_type == 'raw'
        return buf.raw
