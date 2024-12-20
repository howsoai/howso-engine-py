from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
import typing as t
from uuid import uuid4

from typing_extensions import NotRequired, TypeAlias, TypedDict

from amalgam.api import Amalgam
from .exceptions import HowsoError

DEFAULT_ENGINE_PATH = Path(__file__).parent.parent.joinpath("howso-engine")

SchemaTypeOption: TypeAlias = t.Literal["any", "assoc", "boolean", "list", "number", "string", "null"]
"""Enum of valid Engine types."""

SchemaType: TypeAlias = t.Union[SchemaTypeOption, list[SchemaTypeOption]]
"""Valid schema type options."""

TypeDefinition: TypeAlias = t.Union[SchemaType, "Ref", "Schema", "AnyOf"]
"""Any valid type definition. Type(s), Reference, Schema, or Schema composition."""


class Ref(TypedDict):
    """Reference to another schema."""

    ref: str
    default: NotRequired[t.Any]
    description: NotRequired[str | None]
    required: NotRequired[bool]


class AnyOf(TypedDict):
    """An "OR" schema composition."""

    any_of: TypeDefinition
    default: NotRequired[t.Any]
    description: NotRequired[str | None]
    required: NotRequired[bool]


class Schema(TypedDict):
    """A definition of a schema in Engine."""

    type: SchemaType
    default: NotRequired[t.Any]
    description: NotRequired[str]
    enum: NotRequired[list[int | float | str]]
    required: NotRequired[bool]
    # Numbers
    exclusive_max: NotRequired[int | float]
    exclusive_min: NotRequired[int | float]
    max: NotRequired[int | float]
    min: NotRequired[int | float]
    # Lists
    max_size: NotRequired[int]
    min_size: NotRequired[int]
    values: NotRequired[TypeDefinition]
    # Mappings
    additional_indices: NotRequired[TypeDefinition | bool]
    dynamic_indices: NotRequired[TypeDefinition]
    indices: NotRequired[Mapping[str, TypeDefinition]]
    max_indices: NotRequired[int]
    min_indices: NotRequired[int]


class LabelDefinition(TypedDict):
    """A definition to an Engine label."""

    parameters: Mapping[str, TypeDefinition] | None
    returns: TypeDefinition | None
    description: NotRequired[str | None]
    # Annotations
    attribute: NotRequired[bool]
    idempotent: NotRequired[bool]
    long_running: NotRequired[bool]
    payload: NotRequired[bool]
    read_only: NotRequired[bool]
    statistically_idempotent: NotRequired[bool]
    use_active_session: NotRequired[bool]


class EngineApi(TypedDict):
    """The Howso Engine Api documentation object."""

    description: str
    """Description of the API."""

    labels: Mapping[str, LabelDefinition]
    """Map of Engine label name to label definition."""

    schemas: Mapping[str, Schema | Ref | AnyOf]
    """Mapping of schema name to schema definition."""


@lru_cache(16)
def get_api(engine_path: t.Optional[Path | str] = None) -> EngineApi:
    """
    Get api documentation from the Howso Engine.

    Parameters
    ----------
    engine_path : Path, optional
        The path to the Howso Engine caml. Defaults to the built-in engine.

    Returns
    -------
    EngineApi
        The engine api documentation details.

    Raises
    ------
    HowsoError
        If the API cannot be retrieved.
    """
    entity_id = str(uuid4())
    if not engine_path:
        engine_path = DEFAULT_ENGINE_PATH.joinpath("howso.caml")
    if not Path(engine_path).exists():
        raise HowsoError(f"The Howso Engine file path does not exist: {engine_path}.")

    amlg = Amalgam()
    try:
        status = amlg.load_entity(entity_id, str(engine_path))
        if status.loaded:
            initialized = amlg.execute_entity_json(entity_id, "initialize", json.dumps({"trainee_id": entity_id}))
            if not initialized:
                raise ValueError("Not initialized")
            data = amlg.execute_entity_json(entity_id, "get_api", "")
            result = json.loads(data)
            if isinstance(result, list):
                if result[0] == 1 and isinstance(result[1], dict):
                    return EngineApi(result[1]["payload"])
        raise ValueError("Invalid response")
    except Exception as e:
        raise HowsoError('Failed to retrieve the Howso Engine API schema.') from e
    finally:
        amlg.destroy_entity(entity_id)
        del amlg
