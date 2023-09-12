import locale
import sys


def get_default_locale(envvars=('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE')):
    """
    Implement a copy of the Python method `locale.getdefaultlocale()`.

    The Python method has been deprecated in Python 3.11 in favor of
    `locale.getlocales()`. Unforunately, on Windows, `locale.getlocales()`
    which appears have issues with some variants of Windows.

    By redefining it here, we can avoid deprecation warnings and have a
    suitable abstraction point for a more permanent, future solution.

    -- original docstring follows --

    Tries to determine the default locale settings and returns them as tuple
    (language code, encoding).

    According to POSIX, a program which has not called setlocale(LC_ALL, "")
    runs using the portable 'C' locale. Calling setlocale(LC_ALL, "") lets it
    use the default locale as defined by the LANG variable. Since we don't want
    to interfere with the current locale setting we thus emulate the behavior
    in the way described above.

    To maintain compatibility with other platforms, not only the LANG variable
    is tested, but a list of variables given as envvars parameter. The first
    found to be defined will be used. envvars defaults to the search path used
    in GNU gettext; it must always contain the variable name 'LANG'.

    Except for the code 'C', the language code corresponds to RFC 1766. code
    and encoding can be None in case the values cannot be determined.
    """
    try:
        # check if it's supported by the _locale module
        import _locale  # noqa
        code, encoding = _locale._getdefaultlocale()
    except (ImportError, AttributeError):
        pass
    else:
        # make sure the code/encoding values are valid
        if sys.platform == "win32" and code and code[:2] == "0x":
            # map windows language identifier to language name
            code = locale.windows_locale.get(int(code, 0))
        # ...add other platform-specific processing here, if
        # necessary...
        return code, encoding

    # fall back on POSIX behaviour
    import os
    lookup = os.environ.get
    for variable in envvars:
        localename = lookup(variable, None)
        if localename:
            if variable == 'LANGUAGE':
                localename = localename.split(':')[0]
            break
    else:
        localename = 'C'
    return locale._parse_localename(localename)
