from pathlib import Path
import re
from types import SimpleNamespace
import typing as t

from pandas.testing import assert_frame_equal
import pytest

from howso.client.exceptions import HowsoError
from howso.direct.client import HowsoDirectClient
from howso.engine import (
    delete_trainee,
    load_trainee,
    Trainee,
)
from howso.utilities import matrix_processing


class TestEngine:
    """Test the Howso Engine module."""

    @pytest.fixture(autouse=True)
    def trainee(self, data, features):
        """Return a managed trainee that will delete itself upon completion."""
        t = Trainee(features=features)
        t.train(data)

        try:
            yield t
        finally:
            t.delete()

    @pytest.mark.parametrize(
        "from_values,to_values,expected",
        [
            ([[0]], [[0]], 0),
            ([[0]], [[1]], 1),
            ([[0, 0, 0]], [[0, 0, 0]], 0),
            ([[0, 0]], [[1, 0]], 1),
        ],
    )
    def test_pairwise_distances(self, trainee, from_values, to_values, expected):
        """Test get_pairwise_distances returns values from simple vectors."""
        features_list = [str(i) for i in range(len(from_values[0]))]
        result = trainee.get_pairwise_distances(
            features=features_list, from_values=from_values, to_values=to_values
        )
        assert result[0] == expected

    @pytest.mark.parametrize(
        "case_indices,expected",
        [
            ([19, 122], [[0, 0], [0, 0]]),
            ([11, 41, 102], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ],
    )
    def test_distances_same(self, trainee, features, case_indices, expected):
        """
        Test that get_distances returns values as expected.

        Note that, in the iris dataset, rows 19 and 122 are identical
        and rows 11, 41, and 102 are identical and as such their distances
        should be zero.
        """
        sessions = trainee.get_sessions()
        session = sessions[0]
        session_case_indices = []
        for case_index in case_indices:
            session_case_indices.append((session['id'], case_index))

        result = trainee.get_distances(case_indices=session_case_indices)
        result = result['distances'].values.tolist()

        assert result == expected

    @pytest.mark.parametrize(
        "case_indices,unexpected",
        [
            ([0, 1], [[0, 0], [0, 0]]),
            ([2, 3, 4], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ],
    )
    def test_distances_different(self, trainee, features, case_indices, unexpected):
        """
        Test that get_distances returns values as expected.

        The indices 0, 1 and 2, 3, and 4 are not the same, so the distance
        should be nonzero.
        """
        sessions = trainee.get_sessions()
        session = sessions[0]
        session_case_indices = []
        for case_index in case_indices:
            session_case_indices.append((session['id'], case_index))

        result = trainee.get_distances(case_indices=session_case_indices)
        result = result['distances'].values.tolist()

        assert result != unexpected

    def test_get_cases(self, trainee):
        """
        Test that get_cases works as expected.

        Test that get_cases works with and without a session ID to
        get the cases in the order they were trained. This functionality
        only works in a single-user environment and assumes a single session.
        """
        c1 = trainee.get_cases()

        sessions = trainee.get_sessions()
        session = sessions[0]
        c2 = trainee.get_cases(session=session['id'])

        assert c1.equals(c2)

    def test_predict(self, trainee):
        """Test that predict returns the same results as react."""
        action_features = ['target']
        context_features = [k for k in trainee.features.keys() if k not in action_features]

        test_data = [[5.5, 3.6, 1.6, 0.2], [5.2, 3.2, 1.2, 0.2]]

        prediction = trainee.predict(test_data, action_features=action_features, context_features=context_features)
        react = trainee.react(test_data, action_features=action_features, context_features=context_features)

        assert_frame_equal(prediction, react['action'])

    @pytest.mark.parametrize(
        "file_path_type",
        [
            ('directory_only'),
            ('full_path'),
            ('name_only')
        ]
    )
    def test_save_load_good(self, trainee, file_path_type):
        """Test valid disk save and load methods."""
        trainee_name = 'save_load_trainee'
        save_example_trainee = trainee.copy(name=trainee_name)

        cwd = Path.cwd()
        current_directory = f"{cwd}/"

        if file_path_type == 'directory_only':
            file_path = current_directory
        elif file_path_type == 'full_path':
            file_path = current_directory + 'save_load_trainee.caml'
        elif file_path_type == 'name_only':
            file_path = 'save_load_trainee.caml'
        else:
            raise ValueError(f"unexpected file_path_type {file_path_type}")

        # Save Method
        save_example_trainee.save(file_path=file_path)

        # Load
        if file_path_type == 'directory_only':
            file_path = file_path + 'save_load_trainee.caml'
        load_example_trainee = load_trainee(file_path=file_path)

        load_training_cases = load_example_trainee.get_num_training_cases()

        assert trainee.features is not None
        assert trainee.features == save_example_trainee.features
        assert trainee.features == load_example_trainee.features

        # Delete trainees
        save_example_trainee.delete()
        load_example_trainee.delete()

        assert load_training_cases == 150

    def test_save_load_warning(self, trainee):
        """
        Test that the save and load methods raise warnings when expected.

        The save and load methods that should raise a UserWarning but continue
        to work.
        """
        trainee_name = 'save_load_trainee'
        save_example_trainee = trainee.copy(name=trainee_name)

        cwd = Path.cwd()
        file_path = f"{cwd}/save_load_trainee.json"

        # Save Method
        with pytest.warns(
            UserWarning,
            match=('Filepath with a non `.caml` extension was provided.')
        ):
            save_example_trainee.save(file_path=file_path)

        # Set to correct path
        file_path = f"{cwd}/save_load_trainee.caml"

        load_example_trainee = load_trainee(file_path=file_path)
        load_training_cases = load_example_trainee.get_num_training_cases()

        # Delete
        save_example_trainee.delete()
        load_example_trainee.delete()

        assert load_training_cases == 150

    def test_save_load_bad_load(self):
        """Test bad disk load methods."""
        cwd = Path.cwd()
        current_directory = f"{cwd}/"
        file_path = current_directory

        # Load
        with pytest.raises(
            HowsoError,
            match='A `.caml` file must be provided.'
        ):
            load_trainee(file_path=file_path)

    @pytest.mark.parametrize("status_msg, expected_msg", [
        ("This is a test", "This is a test"),
        ("", "An unknown error occurred"),
    ])
    def test_load_status_message(self, mocker, monkeypatch, status_msg, expected_msg):
        """Test load_trainee raises status message from Amalgam."""
        file_path = f"{Path.cwd()}/test_load.caml"

        monkeypatch.setattr(Path, "exists", lambda *args: True)
        mocker.patch(
            "amalgam.api.Amalgam.load_entity",
            return_value=SimpleNamespace(loaded=False, message=status_msg, version="")
        )

        with pytest.raises(
            HowsoError,
            match=f'Failed to load Trainee file "{re.escape(file_path)}": {expected_msg}'
        ):
            load_trainee(file_path=file_path)

    def test_always_persist_load(self, tmp_path: Path, data, features):
        """Test that an auto-persist trainee can be reloaded."""
        trainee = Trainee(features=features, persistence="always")
        try:
            trainee.train(data)
            file_path = Path(t.cast(HowsoDirectClient, trainee.client).resolve_trainee_filepath(trainee.id))
            save_path = tmp_path / "save.caml"
            save_path.write_bytes(file_path.read_bytes())
        finally:
            trainee.delete()

        load_example_trainee = load_trainee(file_path=save_path, persistence="always")
        try:
            assert load_example_trainee.get_num_training_cases() == 150
        finally:
            load_example_trainee.delete()

    def test_delete_method_standalone_good(self, trainee, tmp_path: Path):
        """Test the standalone trainee deletion method for both strings and Path."""
        # Path and string file path
        Path_file_path = tmp_path / 'Path_save_load_trainee.caml'
        string_file_path = str(tmp_path / 'string_save_load_trainee.caml')

        # Save two trainees to test deletion
        trainee.save(file_path=Path_file_path)
        trainee.save(file_path=string_file_path)

        # Delete both trainee's using Path and string file paths
        delete_trainee(file_path=Path_file_path)
        delete_trainee(file_path=string_file_path)

        # Checks to make sure directory is empty
        assert not any(Path_file_path.parents[0].iterdir())

    def test_delete_method_trainee_good_save(self, trainee, tmp_path: Path):
        """Test the Trainee deletion function method for saved trainee, should delete from last saved location."""
        trainee_name = 'delete_trainee'
        delete_example_trainee = trainee.copy(name=trainee_name)

        # Path and string file path
        file_path = tmp_path / f'Path_{trainee_name}.caml'

        # Save trainee to test deletion
        delete_example_trainee.save(file_path=file_path)

        # Make sure delete works on saved trainee
        delete_example_trainee.delete()

        # Checks to make sure directory is empty
        assert not any(file_path.parents[0].iterdir())

    def test_delete_method_trainee_load_good(self, trainee, tmp_path: Path):
        """Test the Trainee deletion function method for loaded trainee, should delete from loaded location."""
        trainee_name = 'delete_trainee'
        delete_example_trainee = trainee.copy(name=trainee_name)

        # Path and string file path
        file_path = tmp_path / f'Path_{trainee_name}.caml'

        delete_example_trainee.save(file_path=file_path)

        # Make sure delete works on loaded trainee
        delete_trainee = load_trainee(file_path)
        delete_trainee.delete()

        # Checks to make sure directory is empty
        assert not any(file_path.parents[0].iterdir())

        # remove from memory
        delete_example_trainee.delete()

    def test_delete_method_standalone_bad(self):
        """Test attempting to delete non-existant trainee."""
        directory_path = Path('test_directory')
        file_path = directory_path.joinpath('Path_non_existant.caml')

        # Delete
        with pytest.raises(
            ValueError,
            match='does not exist.'
        ):
            delete_trainee(file_path=file_path)

    def test_get_contribution_matrix(self, trainee):
        """Test `get_contribution_matrix`."""
        matrix = trainee.get_contribution_matrix(
            normalize=True,
            fill_diagonal=True
        )
        assert len(matrix) == 5
        assert len(matrix.columns) == 5

        # The raw matrix is saved in the trainee. This section
        # tests to make sure the matrix processing parameters are
        # passed through correctly.
        saved_matrix = trainee.calculated_matrices
        assert len(saved_matrix['contribution']) == 5
        assert len(saved_matrix['contribution'].columns) == 5

        saved_matrix = matrix_processing(
            saved_matrix['contribution'],
            normalize=True,
            fill_diagonal=True
        )

        assert_frame_equal(matrix, saved_matrix)

    def test_get_mda_matrix(self, trainee):
        """Test `get_mda_matrix`."""
        matrix = trainee.get_mda_matrix(
            absolute=True,
            fill_diagonal=True
        )
        assert len(matrix) == 5
        assert len(matrix.columns) == 5

        # The raw matrix is saved in the trainee. This section
        # tests to make sure the matrix processing parameters are
        # passed through correctly.
        saved_matrix = trainee.calculated_matrices
        assert len(saved_matrix['mda']) == 5
        assert len(saved_matrix['mda'].columns) == 5

        saved_matrix = matrix_processing(
            saved_matrix['mda'],
            absolute=True,
            fill_diagonal=True
        )

        assert_frame_equal(matrix, saved_matrix)

    @pytest.mark.filterwarnings("ignore:Calling get_cases*")
    def test_reduce_data(self, trainee):
        """Test `reduce_data`."""
        pre_reduction_cases = trainee.get_cases()

        trainee.set_auto_ablation_params(min_num_cases=50)
        trainee.reduce_data(influence_weight_entropy_threshold=0.5)

        post_reduction_cases = trainee.get_cases(features=[".case_weight"])

        assert len(pre_reduction_cases) == 150
        assert len(post_reduction_cases) == 50

        assert any(post_reduction_cases[".case_weight"] != 0)
