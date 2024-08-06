from __future__ import annotations

import typing as t

import pytest

from howso.client.schemas.base import BaseSchema


class MockSchema(BaseSchema):
    """Schema for testing."""

    attribute_map = {
        'tester': 'tester',
        'abc': 'abc',
        'target': 'serialized-name'
    }
    nullable_attributes = {'abc'}

    def __init__(self, tester: str, *, abc: t.Optional[int] = None, target: t.Optional[str] = None) -> None:
        self.tester = tester
        self.abc = abc
        self.target = target


@pytest.mark.parametrize('a, b, equals', [
    ({'tester': 'a'}, {'tester': 'b'}, False),
    ({'tester': '1'}, {'tester': '1'}, True),
    ({'tester': '1'}, {'tester': '1', 'abc': 123}, False),
    ({'tester': '1', 'abc': 123}, {'tester': '1', 'abc': 123}, True),
])
def test_schema_equality(a, b, equals):
    """Test schema equality."""
    schema_a = MockSchema.from_dict(a)
    schema_b = MockSchema.from_dict(b)
    # Schemas should equal themselves
    assert schema_a == schema_a
    assert schema_b == schema_b
    # Original dict is not a schema instance
    assert schema_a != a
    assert schema_b != b
    # Compare a and b
    if equals:
        assert schema_a == schema_a
    else:
        assert schema_a != schema_b


def test_schema_from_dict():
    """Test schema from dict."""
    schema = MockSchema.from_dict({'tester': 'test', 'abc': 123, 'dne': 123})
    assert schema.tester == 'test'
    assert schema.abc == 123
    assert schema.target is None
    assert not hasattr(schema, 'dne')
    with pytest.raises(ValueError):
        MockSchema.from_dict("invalid")  # type: ignore


@pytest.mark.parametrize('schema, expected, exclude_null, serialize', [
    (
        MockSchema('mock'),
        {'tester': 'mock', 'abc': None},
        False, False
    ),
    (
        MockSchema('mock', abc=123, target='xyz'),
        {'tester': 'mock', 'abc': 123, 'target': 'xyz'},
        False, False
    ),
    (
        MockSchema('mock'),
        {'tester': 'mock'},
        True, False
    ),
    (
        MockSchema('mock', target='xyz'),
        {'tester': 'mock', 'abc': None, 'serialized-name': 'xyz'},
        False, True
    ),
    (
        MockSchema('mock', target='xyz'),
        {'tester': 'mock', 'serialized-name': 'xyz'},
        True, True
    ),
])
def test_schema_to_dict(schema, expected, exclude_null, serialize):
    """Test schema to dict."""
    value = schema.to_dict(exclude_null=exclude_null, serialize=serialize)
    assert value == expected
