import warnings
import pytest

from howso.utilities.feature_attributes.models import InferFeatureAttributesArgs


def test_merge_kwargs_updates_fields():
    """
    Ensure fields are passed through and updated in the returned model.
    """
    initial_args = InferFeatureAttributesArgs(attempt_infer_extended_nominals=False)

    update_kwargs = {
        "attempt_infer_extended_nominals": True,
        "default_time_zone": "America/New_York"
    }

    # Use pytest.warns to capture the DeprecationWarning
    with pytest.warns(DeprecationWarning) as record:
        merged_args = initial_args.merge_kwargs(**update_kwargs)

    # Check that a warning was issued
    assert len(record) == 1
    warning_message = record[0].message.args[0]
    expected_keys = sorted(list(update_kwargs.keys()))
    assert f"Usage of {expected_keys} directly is deprecated" in warning_message

    # Check that the new model has the updated values
    assert merged_args.attempt_infer_extended_nominals is True
    assert merged_args.default_time_zone == "America/New_York"

    # Check that other fields retain their original values or defaults
    assert merged_args.include_sample is False # Default value


def test_merge_kwargs_no_kwargs_no_warning():
    """
    Ensure no warning is issued if no kwargs are provided.
    """
    initial_args = InferFeatureAttributesArgs(attempt_infer_extended_nominals=False)

    with warnings.catch_warnings(record=True) as record:
        merged_args = initial_args.merge_kwargs(**{})

    assert not record # No warnings should be captured

    # Ensure it returns a copy, even if no changes
    assert merged_args is not initial_args
    assert merged_args.model_dump() == initial_args.model_dump()
