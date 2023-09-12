import os


try:
    import howso.nominal_substitution as _  # noqa
except ImportError:
    NOMINAL_SUBSTITUTION_AVAILABLE = False
else:
    NOMINAL_SUBSTITUTION_AVAILABLE = True
