from pathlib import Path

from howso.client.exceptions import HowsoError
from howso.engine import load_trainee, Trainee
from pandas.testing import assert_frame_equal
import pytest


class TestEngine:
    """Test the Howso Engine module."""

    @pytest.fixture(autouse=True)
    def trainee(self, data, features):
        """Return a managed trainee that will delete itself upon completion."""
        t = Trainee(features=features)
        t.train(data)

        try:
            yield t
        except Exception:
            raise
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

        # Save Method
        save_example_trainee.save(file_path=file_path)

        # Load
        if file_path_type == 'directory_only':
            file_path = file_path + 'save_load_trainee.caml'
        load_example_trainee = load_trainee(file_path=file_path)

        load_training_cases = load_example_trainee.get_num_training_cases()

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
