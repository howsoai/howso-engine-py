from abc import ABC
from enum import Enum
import warnings


class IFAWarningEmitterType(Enum):
    """IFAWarningEmitter enum."""

    NEAR_UNIQUE_DEPENDENT_FEATURES = "near_unique_dependent_features"
    MISSING_TZ_FEATURES = "missing_tz_features"
    UNKNOWN_DATETIME_FORMAT = "unknown_datetime_format"
    UTC_OFFSET = "utc_offset"
    VALUE_COUNTS_PROCESSING = "value_counts_processing"
    EXCESSIVE_FLOAT_PRECISION = "excessive_float_precision"
    POSSIBLE_EXCESSIVE_FLOAT_PRECISION = "possible_excessive_float_precision"
    SIMPLE = "simple"


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
        return msg + "\n"

    def emit(self):
        """Emit the warning."""


class NearUniqueDependentFeaturesWarningEmitter(IFAWarningEmitter):
    """Emitter for a warning about dependent features having too many unique values."""

    def emit(self):
        """Emit the warning."""
        warnings.warn("The following provided `dependent_features` have a large share of values that are unique: "
                      f"{self.features_list}"
                      "Dependent features with many unique values can severely impact the quality of results.",
                      UserWarning)


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


class ValueCountsProcessing(IFAWarningEmitter):
    """Emitter for a warning about the inclusion of UTC offsets."""

    def emit(self):
        """Emit the warning."""
        warnings.warn("Could not process some value counts for the following features, likely due to the presence of "
                      f"unhashable values: {self.features_list}\nThis may affect the accuracy and completeness of "
                      "suggested or computed `preserve_rare_values` configurations`.", UserWarning)


class FloatPrecisionWarningEmitter(IFAWarningEmitter):
    """Base emitter for warnings about float features that exceed the precision the engine supports."""

    #: How the warning relates the features to the precision limit.
    _certainty: str

    def emit(self):
        """Emit the warning."""
        warnings.warn(f"The following features {self._certainty} floating point values that exceed the "
                      f"maximum supported precision of 64 bits: {self.features_list}"
                      "\nThese features are trained without a `decimal_places` attribute.", UserWarning)


class ExcessiveFloatPrecisionWarningEmitter(FloatPrecisionWarningEmitter):
    """Emitter for a warning about float features whose dtype is wider than 64 bits."""

    _certainty = "contain"


class PossibleExcessiveFloatPrecisionWarningEmitter(FloatPrecisionWarningEmitter):
    """Emitter for a warning about float features whose dtype does not report its size."""

    _certainty = "may contain"


class SimpleWarningEmitter(IFAWarningEmitter):
    """Emitter for simple warnings that are saved via the `features_list`."""

    def emit(self):
        """Emit the warning."""
        for msg in self.features:
            warnings.warn(msg, UserWarning)


class IFAWarningCollector:
    """A collector for IFAWarningEmitters that can triage new feature entries."""

    #: The emitter that serves each type of warning.
    _EMITTERS: dict[IFAWarningEmitterType, type[IFAWarningEmitter]] = {
        IFAWarningEmitterType.NEAR_UNIQUE_DEPENDENT_FEATURES: NearUniqueDependentFeaturesWarningEmitter,
        IFAWarningEmitterType.MISSING_TZ_FEATURES: MissingTZFeaturesWarningEmitter,
        IFAWarningEmitterType.UNKNOWN_DATETIME_FORMAT: UnknownDatetimeFormatWarningEmitter,
        IFAWarningEmitterType.UTC_OFFSET: UTCOffsetFeaturesWarningEmitter,
        IFAWarningEmitterType.VALUE_COUNTS_PROCESSING: ValueCountsProcessing,
        IFAWarningEmitterType.EXCESSIVE_FLOAT_PRECISION: ExcessiveFloatPrecisionWarningEmitter,
        IFAWarningEmitterType.POSSIBLE_EXCESSIVE_FLOAT_PRECISION: PossibleExcessiveFloatPrecisionWarningEmitter,
        IFAWarningEmitterType.SIMPLE: SimpleWarningEmitter,
    }

    def __init__(self, emitters: dict[str, IFAWarningEmitter] | None = None) -> None:
        self._emitters = emitters or {}

    def triage(self, emitter_type: IFAWarningEmitterType, feature_name: str) -> None:
        """
        Sort the provided feature into the correct emitter bucket.

        Parameters
        ----------
        emitter_type : IFAWarningEmitterType
            The type of Warning Emitter this feature should be sorted to.
        feature_name : str
            The name of the feature applicable to the warning.

        Raises
        ------
        ValueError
            If `emitter_type` is not a known type of warning.
        """
        try:
            emitter = self._EMITTERS[emitter_type]
        except KeyError:
            raise ValueError("Unknown `emitter_type` provided.") from None

        key = emitter_type.value
        if key not in self._emitters:
            self._emitters[key] = emitter(features={feature_name})
        else:
            self._emitters[key].features.add(feature_name)

    def emit_all(self) -> None:
        """Emit all warnings collected."""
        for emitter in self._emitters.values():
            emitter.emit()

    def merge(self, other: object) -> None:
        """Merge another IFAWarningCollector object with this one."""
        for key, emitter in other._emitters.items():
            if key not in self._emitters:
                self._emitters[key] = emitter
            else:
                for f in emitter.features:
                    self._emitters[key].features.add(f)
