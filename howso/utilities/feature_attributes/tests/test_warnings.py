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
    collector.triage(IFAWarningEmitterType.EXCESSIVE_FLOAT_PRECISION, "e")
    collector.triage(IFAWarningEmitterType.POSSIBLE_EXCESSIVE_FLOAT_PRECISION, "f")

    with pytest.warns(UserWarning, match=r"- [a-f]") as record:
        collector.emit_all()
        assert len(record) == 6


# `SIMPLE` collects whole messages rather than feature names, so it emits one warning per message.
@pytest.mark.parametrize("emitter_type", [t for t in IFAWarningEmitterType if t != IFAWarningEmitterType.SIMPLE])
def test_warnings_emitters_list_all_features(emitter_type):
    """Test that features sharing an emitter are listed in a single warning."""
    collector = IFAWarningCollector()
    for feature in ("a", "b", "c"):
        collector.triage(emitter_type, feature)

    with pytest.warns(UserWarning) as record:
        collector.emit_all()

    assert len(record) == 1
    message = str(record[0].message)
    assert all(feature in message for feature in ("a", "b", "c"))


def test_warnings_emitters_unknown_type():
    """Test that an unknown emitter type is rejected."""
    with pytest.raises(ValueError, match="Unknown `emitter_type` provided."):
        IFAWarningCollector().triage("not_an_emitter_type", "a")
