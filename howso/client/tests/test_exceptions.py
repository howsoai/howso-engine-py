import pytest

from ..exceptions import HowsoApiError, HowsoApiValidationError, HowsoValidationError


@pytest.mark.parametrize("collection, size", [
    ({}, 0),
    (None, 0),
    (False, 0),
    ("unexpected", 0),
    ({"test": []}, 0),
    ({"test": [{"foo": "bar"}]}, 1),
    ({"test": [{"message": "abc123"}, {"message": "required"}]}, 2),
    ({"test": [{"message": "abc123"}], "abc": [{"message": "123"}]}, 2)
])
def test_validation_error(collection, size):
    """Test validation error initialization."""
    error = HowsoValidationError("test", errors=collection)
    assert error.message == "test"
    assert error.errors == collection
    assert len(list(error.iter_errors())) == size


def test_validation_iter_errors():
    """Test validation iter_errors method."""
    collection = {
        "abc": [{"message": "abc"}],
        "body": {
            "payload": {
                "test": [{"foo": "bar"}],
                "context": [
                    {"message": "context", "code": "CODE"},
                    {"message": "required", "field": "bla"},
                ]
            }
        }
    }
    error = HowsoApiValidationError("Request validation failed", code="abc123", url="test_url", errors=collection)
    assert isinstance(error, HowsoValidationError)
    assert isinstance(error, HowsoApiError)
    assert error.status == 400
    assert error.code == "abc123"
    assert error.message == "Request validation failed"
    assert error.url == "test_url"
    all_errors = list(error.iter_errors())
    assert len(all_errors) == 4

    assert dict(message="abc", field=["abc"]) in all_errors
    assert dict(message="An unknown error occurred.", field=["body", "payload", "test"]) in all_errors
    assert dict(message="context", code="CODE", field=["body", "payload", "context"]) in all_errors
    assert dict(message="required", field=["body", "payload", "context"]) in all_errors
