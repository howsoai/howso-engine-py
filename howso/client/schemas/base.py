from abc import ABC
from collections.abc import Mapping
import typing as t

from typing_extensions import TypeVar

DT = TypeVar("DT", bound=Mapping[str, t.Any])


class BaseSchema(ABC, t.Generic[DT]):
    """Base class for engine type schemas."""

    attribute_map = {}
    nullable_attributes = []

    def to_dict(self, *, exclude_null: bool = False) -> DT:
        """Returns the schema's properties as a dict."""
        result = {}

        for attr in self.attribute_map:
            value = getattr(self, attr)
            attr = self.attribute_map.get(attr, attr)
            if value is None and (exclude_null or attr not in self.nullable_attributes):
                continue
            else:
                result[attr] = value
        return t.cast(DT, result)

    @classmethod
    def from_dict(cls, schema: Mapping):
        """Returns a new schema using properties from dict."""
        parameters = {k: schema[k] for k in cls.attribute_map if k in schema}
        return cls(**parameters)

    def __str__(self) -> str:
        """Returns the string representation of the schema."""
        return str(self.to_dict())

    def __eq__(self, other: t.Any) -> bool:
        """Returns true if both objects are equal."""
        if not isinstance(other, BaseSchema):
            return False
        return self.to_dict() == other.to_dict()

    def __ne__(self, other: t.Any) -> bool:
        """Returns true if both objects are not equal."""
        if not isinstance(other, BaseSchema):
            return True
        return self.to_dict() != other.to_dict()
