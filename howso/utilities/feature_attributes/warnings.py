from abc import ABC
from enum import Enum
import warnings


class IFAWarningEmitterType(Enum):
    """IFAWarningEmitter enum."""

    NEAR_UNIQUE_DEPENDENT_FEATURES = "near_unique_dependent_features"
    MISSING_TZ_FEATURES = "missing_tz_features"
    UNKNOWN_DATETIME_FORMAT = "unknown_datetime_format"
    UTC_OFFSET = "utc_offset"


class IFAWarningEmitter(ABC):
    """Base class for a warning emitter common to IFA that must list applicable features discovered across shards."""

    def __init__(self, features: set = None):
        self._features = features or set()

    @property
    def features(self) -> set:
        """The features relevant to the warning."""
        return self._features

    @features.setter
    def features(self, features: set):
        self._features = features

    @property
    def features_list(self) -> str:
        """The list of features, formatted in a hyphenated list."""
        msg = ""
        for feature_name in self.features:
            msg += f"\n\t- {feature_name}"
        return msg

    def emit(self):
        """Emit the warning."""


class NearUniqueDependentFeaturesWarningEmitter(IFAWarningEmitter):
    """Emitter for a warning about dependent features having too many unique values."""

    def emit(self):
        """Emit the warning."""
        warnings.warn("The following provided `dependent_features` have a large share of values that are unique: "
                      f"{self.features_list}"
                      "Dependent features with many unique values can severely impact performance.", UserWarning)


class MissingTZFeaturesWarningEmitter(IFAWarningEmitter):
    """Emitter for a warning about features not including time zones."""

    def emit(self):
        """Emit the warning."""
        warnings.warn("The provided or inferred `date_time_formats` for the following "
                      f"features do not include a time zone and will default to UTC: {self.features_list}"
                      "\nTo change the default time zone, please specify the `default_time_zone` "
                      "argument to `infer_feature_attributes`.", UserWarning)


class UnknownDatetimeFormatWarningEmitter(IFAWarningEmitter):
    """Emitter for a warning about indeterminate date time formats."""

    def emit(self):
        """Emit the warning."""
        warnings.warn("The following features were detected as possible datetimes, but we cannot assume "
                      "their formats. Please provide them using `datetime_feature_formats` if desired. "
                      f"Otherwise, these features will be treated as nominal strings: {self.features_list}",
                      UserWarning)


class UTCOffsetFeaturesWarningEmitter(IFAWarningEmitter):
    """Emitter for a warning about the inclusion of UTC offsets."""

    def emit(self):
        """Emit the warning."""
        warnings.warn(f"The following features are using UTC offsets (%z) for their time zones: {self.features_list}"
                      "\nThis could lead to unexpected results due to daylight savings time. We recommend "
                      "using explicit time zone strings, e.g., \"GMT\", which are represented by the \"%Z\" "
                      "identifier.", UserWarning)


class IFAWarningCollector:
    """A collector for IFAWarningEmitters that can triage new feature entries."""

    def __init__(self, emitters: dict[str, IFAWarningEmitter] = None):
        self._emitters = emitters or {}

    def triage(self, emitter_type: IFAWarningEmitterType, feature_name: str):
        """
        Sort the provided feature into the correct emitter bucket.

        Parameters
        ----------
        emitter_type : IFAWarningEmitterType
            The type of Warning Emitter this feature should be sorted to.
        feature_name : str
            The name of the feature applicable to the warning.
        """
        if emitter_type == IFAWarningEmitterType.NEAR_UNIQUE_DEPENDENT_FEATURES:
            key = IFAWarningEmitterType.NEAR_UNIQUE_DEPENDENT_FEATURES.value
            emitter = NearUniqueDependentFeaturesWarningEmitter
        elif emitter_type == IFAWarningEmitterType.MISSING_TZ_FEATURES:
            key = IFAWarningEmitterType.MISSING_TZ_FEATURES.value
            emitter = MissingTZFeaturesWarningEmitter
        elif emitter_type == IFAWarningEmitterType.UNKNOWN_DATETIME_FORMAT:
            key = IFAWarningEmitterType.UNKNOWN_DATETIME_FORMAT.value
            emitter = UnknownDatetimeFormatWarningEmitter
        elif emitter_type == IFAWarningEmitterType.UTC_OFFSET:
            key = IFAWarningEmitterType.UTC_OFFSET.value
            emitter = UTCOffsetFeaturesWarningEmitter
        else:
            raise ValueError("Unknown `emitter_type` provided.")

        if key not in self._emitters.keys():
            self._emitters[key] = emitter(features={feature_name})
        else:
            self._emitters[key].features.add(feature_name)

    def emit_all(self):
        """Emit all warnings collected."""
        for emitter in self._emitters.values():
            emitter.emit()
