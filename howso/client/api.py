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

SchemaTypeOption: TypeAlias = t.Literal["any", "assoc", "boolean", "list", "number", "string"]
SchemaType: TypeAlias = SchemaTypeOption | list[SchemaTypeOption]


class RefSchema(TypedDict):
    """Reference to another schema."""

    ref: str
    description: NotRequired[str]
    optional: NotRequired[bool]
    default: NotRequired[t.Any]


class Schema(TypedDict):
    """A definition of a parameter or return value in Engine."""

    type: SchemaType
    description: NotRequired[str]
    optional: NotRequired[bool]
    default: NotRequired[t.Any]
    enum: NotRequired[list[int | float | str]]
    min: NotRequired[int | float]
    max: NotRequired[int | float]
    min_length: NotRequired[int]
    max_length: NotRequired[int]
    values: NotRequired[SchemaType | Schema | RefSchema]
    indices: NotRequired[SchemaType | Mapping[str, SchemaType | Schema | RefSchema]]
    additional_indices: NotRequired[SchemaType | Mapping[str, SchemaType | Schema | RefSchema]]


class LabelDefinition(TypedDict):
    """Engine label definition."""

    parameters: Mapping[str, Schema | RefSchema] | None
    returns: NotRequired[Schema | RefSchema | None]
    description: NotRequired[str | None]
    long_running: NotRequired[bool]
    idempotent: NotRequired[bool]
    statistically_idempotent: NotRequired[bool]
    read_only: NotRequired[bool]


class EngineApi(TypedDict):
    """The Howso Engine Api documentation object."""

    labels: Mapping[str, LabelDefinition]
    """Engine labels."""

    schemas: Mapping[str, Schema | RefSchema]
    """Mapping of shared schemas."""

    description: str
    """Description of the API."""


@lru_cache(16)
def get_api(engine_path: t.Optional[Path] = None) -> EngineApi:
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
    if not engine_path.exists():
        raise HowsoError(f"The Howso Engine file path does not exist: {engine_path}.")

    amlg = Amalgam()
    try:
        status = amlg.load_entity(entity_id, str(engine_path))
        if status.loaded:
            data = amlg.execute_entity_json(entity_id, "get_api", "")
            result = json.loads(data)
            if isinstance(result, list):
                if result[0] == 1 and isinstance(result[1], dict):
                    return EngineApi(result[1]["payload"])
        raise ValueError("Invalid response")
    except Exception:
        raise HowsoError('Failed to retrieve the Howso Engine API schema.')
    finally:
        amlg.destroy_entity(entity_id)
        del amlg


def get_api_label(label: str) -> LabelDefinition | None:
    """
    Get the API definition for a given label.

    Parameters
    ----------
    label : str
        The label to retrieve.

    Returns
    -------
    LabelDefinition or None
        The definition of the label, or None if the label is not defined.
    """
    api = get_api()
    return api["labels"].get(label)


def get_api_schema(name: str) -> Schema | RefSchema | None:
    """
    Get a schema definition by name.

    Parameters
    ----------
    name : str
        The name of the schema to retrieve.

    Returns
    -------
    Schema or RefSchema or None
        The schema definition, or None if not found.
    """
    api = get_api()
    return api["schemas"].get(name)
