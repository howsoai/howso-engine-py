from typing_extensions import TypedDict


class HowsoVersion(TypedDict, total=False):
    """Howso version numbers."""

    client: str
    """The howso-engine python client version."""
