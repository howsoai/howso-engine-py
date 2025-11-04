from collections.abc import Iterable, Generator
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
import pytest

from howso.client.schemas import Trainee
from howso.direct import HowsoDirectClient
from howso.utilities.testing import get_configurationless_test_client


@contextmanager
def simple_trainee(
    client: HowsoDirectClient,
    values: Iterable[int],
    *,
    id: str | None = None,
    start_case_index: int = 0
) -> Generator[Trainee, None, None]:
    """Create a simple Trainee."""
    t = client.create_trainee(
        id=id or uuid4(),
        name="simple-trainee",
        metadata={"simple": True},
        features={"x": {"type": "continuous"}},
    )
    try:
        # TODO - start_case_index
        client.train(t.id, [[v] for v in values], features=["x"])
        client.analyze(t.id)
        yield t
    finally:
        client.delete_trainee(t.id)


@pytest.fixture
def client(tmp_path: Path):
    """Direct client instance using latest binaries."""
    return get_configurationless_test_client(
        client_class=HowsoDirectClient,
        verbose=True,
        trace=True,
        default_persist_path=tmp_path,
    )


@pytest.mark.parametrize(["file_type"], [("amlg",), ("caml",)])
def test_create_trainee_from_bytes(client: HowsoDirectClient, file_type: Literal["amlg", "caml"]) -> None:
    """Test creation of a trainee from bytes."""
    dne_content = client.trainee_to_bytes("dne", file_type=file_type)
    assert dne_content is None

    with simple_trainee(client, range(1, 5)) as trainee:
        # Convert trainee to bytes
        content = client.trainee_to_bytes(trainee.id, file_type=file_type)
        assert content is not None
        if file_type == "amlg":
            assert content.startswith(b"(declare")
        elif file_type == "caml":
            assert content.startswith(b"caml")

        # Re-create it from bytes
        new_trainee = client.create_trainee_from_bytes("test-from-bytes", content)
        assert new_trainee is not None
        try:
            assert new_trainee.id == "test-from-bytes"
            assert new_trainee.name == "simple-trainee"
            assert new_trainee.metadata == {"simple": True}
            assert new_trainee.persistence == "allow"
        finally:
            client.delete_trainee(new_trainee.id)

        # Re-create it as a sub-trainee from bytes
        child = client.create_trainee_from_bytes(trainee.id, content, path=["child"], child_id="child-from-bytes")
        assert child.id == "child-from-bytes"
        assert child.name == "simple-trainee"
        assert child.metadata == {"simple": True}
        assert child.persistence == "allow"
        hierarchy = client.get_hierarchy(trainee.id)
        children = hierarchy.get("children")
        assert isinstance(children, list)
        assert len(children) == 1
        assert children[0]["id"] == "child-from-bytes"
        assert children[0]["name"] == "simple-trainee"


@pytest.mark.parametrize(["child_ids", "expected_cases"], [
    (None, 15),
    (["child1"], 10),
    (["child1", "child2"], 15),
])
def test_combine_trainee_with_subtrainees(
    client: HowsoDirectClient,
    expected_cases: int,
    child_ids: list[str] | None,
) -> None:
    """Test combining sub-trainees into parent."""
    with ExitStack() as stack:
        parent = stack.enter_context(simple_trainee(client, range(0, 5)))
        child1 = stack.enter_context(simple_trainee(client, range(5, 10), id="child1", start_case_index=5))
        child2 = stack.enter_context(simple_trainee(client, range(10, 15), id="child2", start_case_index=10))

        for child in [child1, child2]:
            # Currently the only way to create the hierarchy is to do so from bytes, recreate trainees as children
            content = client.trainee_to_bytes(child.id)
            assert content is not None
            client.create_trainee_from_bytes(parent.id, content, path=[child.id], child_id=child.id)

        # Validate they are now children
        hierarchy = client.get_hierarchy(parent.id)
        assert len(hierarchy.get("children", [])) == 2

        result = client.combine_trainee_with_subtrainees(parent.id, child_ids)
        assert result["status"] == "analyzed"

        # Child trainees should now be deleted
        hierarchy = client.get_hierarchy(parent.id)
        assert hierarchy.get("children", []) == []

        # Validate the cases are now in the parent
        cases = client.get_cases(parent.id, features=[".session_training_index", "x"])
        assert len(cases) == expected_cases
        assert cases["cases"][0] == [0, 0]
        assert cases["cases"][6] == [6, 6]