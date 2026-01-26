from collections import MutableSequence
from enum import Enum
import warnings


class FeatureType(Enum):
    """Feature type enum."""

    UNKNOWN = "object"
    STRING = "string"
    TOKENIZABLE_STRING = "tokenizable_string"
    NUMERIC = "numeric"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    TIMEDELTA = "timedelta"
    CONTAINER = "container"

    def __str__(self):
        """Return a string representation."""
        return str(self.value)


class IFAWarning:

    def __init__(self):
        pass


class IFAWarningsCollector(MutableSequence):

    def __init__(self):
        pass

    def emit_all(self):
        pass
