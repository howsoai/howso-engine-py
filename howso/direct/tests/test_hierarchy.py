from collections.abc import Iterable, Generator
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest

from howso.client.schemas import Trainee
from howso.direct import HowsoDirectClient
from howso.utilities import is_valid_uuid
from howso.utilities.testing import get_configurationless_test_client


@contextmanager
def simple_trainee(
    client: HowsoDirectClient,
    values: Iterable[int],
    *,
    id: str | None = None,
) -> Generator[Trainee, None, None]:
    """Create a simple Trainee."""
    t = client.create_trainee(
        id=id or uuid4(),
        name="simple-trainee",
        metadata={"simple": True},
        features={"x": {"type": "continuous"}},
    )
    try:
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
        new_trainee = client.create_trainee_from_bytes("test-from-bytes", content, file_type=file_type)
        assert new_trainee is not None
        try:
            assert new_trainee.id == "test-from-bytes"
            assert new_trainee.name == "simple-trainee"
            assert new_trainee.metadata == {"simple": True}
            assert new_trainee.persistence == "allow"
            # internal id should match handle
            assert client.execute(new_trainee.id, "get_trainee_id", {}) == new_trainee.id
        finally:
            client.delete_trainee(new_trainee.id)


@pytest.mark.parametrize(["file_type"], [("amlg",), ("caml",)])
def test_create_sub_trainees_from_bytes(client: HowsoDirectClient, file_type: Literal["amlg", "caml"]) -> None:
    """Test creation of a trainee from bytes."""
    with simple_trainee(client, range(1, 5)) as trainee:
        # Convert trainee to bytes
        content = client.trainee_to_bytes(trainee.id, file_type=file_type)
        assert content is not None
        if file_type == "amlg":
            assert content.startswith(b"(declare")
        elif file_type == "caml":
            assert content.startswith(b"caml")

        # Re-create it as a sub-trainee from bytes
        child = client.create_trainee_from_bytes(
            trainee.id,
            content,
            path=["child"],
            child_id="child-from-bytes",
            file_type=file_type,
        )
        assert child.id == "child-from-bytes"
        assert child.name == "simple-trainee"
        assert child.metadata == {"simple": True}
        assert child.persistence == "allow"
        assert client.execute(trainee.id, "get_trainee_id", {}, path=["child"]) == child.id

        # Re-create it as a sub-sub-trainee from bytes
        grandchild = client.create_trainee_from_bytes(
            trainee.id,
            content,
            path=["child", "grand-child"],
            child_id="grand-child-from-bytes",
            file_type=file_type,
        )
        assert grandchild.id == "grand-child-from-bytes"
        assert grandchild.name == "simple-trainee"
        assert grandchild.metadata == {"simple": True}
        assert grandchild.persistence == "allow"
        assert client.execute(trainee.id, "get_trainee_id", {}, path=["child", "grand-child"]) == grandchild.id

        # Check the hierarchy schema
        hierarchy = client.get_hierarchy(trainee.id)
        children = hierarchy.get("children")
        assert isinstance(children, list)
        assert len(children) == 1
        assert children[0]["id"] == "child-from-bytes"
        assert children[0]["name"] == "simple-trainee"
        grand_children = children[0].get("children")
        assert isinstance(grand_children, list)
        assert len(grand_children) == 1
        assert grand_children[0]["id"] == "grand-child-from-bytes"
        assert grand_children[0]["name"] == "simple-trainee"


def test_create_sub_trainee_from_bytes_auto_path(client: HowsoDirectClient) -> None:
    """Test creating sub-trainees from bytes with auto generated path and id."""
    with ExitStack() as stack:
        parent = stack.enter_context(simple_trainee(client, range(0, 5)))
        trainee = stack.enter_context(simple_trainee(client, range(5, 10), id="child1"))
        trainee_content = client.trainee_to_bytes(trainee.id)
        assert trainee_content is not None

        # Re-create trainee as a child
        child = client.create_trainee_from_bytes(
            parent.id,
            content=trainee_content,
            path=[],  # Auto generate the path
        )
        assert is_valid_uuid(child.id), child.id  # Should be auto-generated uuid
        assert child.id != trainee.id
        assert child.name == "simple-trainee"
        assert child.metadata == {"simple": True}
        assert child.persistence == "allow"

        # Check hierarchy schema
        hierarchy = client.get_hierarchy(parent.id)
        assert hierarchy["id"] == parent.id
        children = hierarchy.get("children")
        assert isinstance(children, list)
        assert len(children) == 1
        assert children[0]["id"] == child.id
        assert len(children[0]["path"]) == 1


@pytest.mark.parametrize(["child_ids", "expected_cases"], [
    (None, 250),
    (["child1"], 200),
    (["child1", "child2"], 250),
])
def test_combine_trainee_with_subtrainees(
    client: HowsoDirectClient,
    expected_cases: int,
    child_ids: list[str] | None,
) -> None:
    """Test combining sub-trainees into parent."""
    with ExitStack() as stack:
        parent_session = client.begin_session("parent")
        parent = stack.enter_context(simple_trainee(client, range(0, 100)))
        child1_session = client.begin_session("child1")
        child1 = stack.enter_context(simple_trainee(client, range(100, 200), id="child1"))
        child2_session = client.begin_session("child2")
        child2 = stack.enter_context(simple_trainee(client, range(200, 250), id="child2"))

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
        expected_child_count = 0 if child_ids is None else 2 - len(child_ids)
        assert len(hierarchy.get("children", [])) == expected_child_count

        # Validate the cases are now in the parent
        cases = client.get_cases(parent.id, features=[".session", "x"])["cases"]
        assert len(cases) == expected_cases
        parent_cases = [case[1:] for case in cases if case[0] == parent_session.id]
        child1_cases = [case[1:] for case in cases if case[0] == child1_session.id]
        child2_cases = [case[1:] for case in cases if case[0] == child2_session.id]

        assert parent_cases == [[i] for i in range(0, 100)]
        if child_ids is None or "child1" in child_ids:
            assert child1_cases == [[i] for i in range(100, 200)]
        else:
            assert child1_cases == []
        if child_ids is None or "child2" in child_ids:
            assert child2_cases == [[i] for i in range(200, 250)]
        else:
            assert child2_cases == []
