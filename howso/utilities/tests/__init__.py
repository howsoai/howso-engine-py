import subprocess
import warnings


def has_locales(locales):
    """
    This is for the `pytest.mark.skipif` decorator. Skip the test of the
    required locales are not installed in this system.

    Background
    ----------
    It seems there's no reliable way to get a list of available locales
    directly in Python. `locale.locale_alias` returns /only/ aliases to
    existing locales, but not the original locales themselves.

    Parameters
    ----------
    locales : iterable[str]
        Iterable of language_codes E.g.: ['en_US.utf8', 'fr_CA.utf8', ... ]

    Returns
    -------
    bool :
        True if the given locales are available.
    """
    try:
        out = subprocess.run(['locale', '-a'], stdout=subprocess.PIPE).stdout
        results = ''
        for encoding in ['utf8', 'ISO8859-1', 'Windows-1252']:
            try:
                results = out.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise Exception('Could not decode subprocess output')
        locale_list = results.rstrip('\n').splitlines()
        return all(locale_item in locale_list for locale_item in locales)
    except Exception:  # noqa: Deliberately broad
        warnings.warn('Failed to check system locale')
        return False
