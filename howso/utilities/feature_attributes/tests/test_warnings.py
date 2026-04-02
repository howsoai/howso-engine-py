import pytest

from howso.utilities.feature_attributes.warnings import IFAWarningCollector, IFAWarningEmitterType


def test_warnings_emitters():
    """Test the warnings collector and emitters."""
    # Add a warning of each type
    collector = IFAWarningCollector()
    collector.triage(IFAWarningEmitterType.NEAR_UNIQUE_DEPENDENT_FEATURES, "a")
    collector.triage(IFAWarningEmitterType.MISSING_TZ_FEATURES, "b")
    collector.triage(IFAWarningEmitterType.UNKNOWN_DATETIME_FORMAT, "c")
    collector.triage(IFAWarningEmitterType.UTC_OFFSET, "d")

    with pytest.warns(UserWarning, match=r"- [a-d]") as record:
        collector.emit_all()
        assert len(record) == 4
