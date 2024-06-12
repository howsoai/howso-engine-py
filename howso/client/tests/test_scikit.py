import pytest
try:
    import sklearn  # noqa
except ImportError:
    pytest.skip(allow_module_level=True)

import gc
import pickle
import uuid

from howso import engine
from howso.client.exceptions import HowsoNotUniqueError
from howso.client.pandas import HowsoPandasClient
from howso.direct import HowsoDirectClient
from howso.scikit import (
    CLASSIFICATION,
    HowsoClassifier,
    HowsoEstimator,
    HowsoRegressor,
)
from howso.utilities.testing import get_configurationless_test_client, get_test_options
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_validate

TEST_OPTIONS = get_test_options()


@pytest.fixture(scope='function')
def classifier():
    """Creates a pre-populated Howso estimator."""
    # Let's learn how to classify the XOR operation.
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
                  [0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    howso = HowsoClassifier(
        client=get_configurationless_test_client(client_class=HowsoPandasClient))
    howso.fit(X, y)
    return howso


class TestHowso:
    """A test class for testing the estimators."""

    @classmethod
    def setup_class(cls):
        """Setup a test client to use for each test method."""
        cls.client = get_configurationless_test_client(client_class=HowsoPandasClient)

    def test_regressor(self):
        """Tests the HowsoRegressor from the external client."""
        X = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        howso = HowsoRegressor(client=self.client)
        howso.fit(X, y)
        print(howso.score(X, y))
        print(howso.predict(np.array([[4], [5], [6]])))
        # Ensure that the trainee is unnamed, so it will be deleted.
        howso.trainee_name = None

    def test_classifier(self):
        """Tests the HowsoClassifier from the external client."""
        # Use two instances of data indicating the xor operation to make sure
        # it is able to learn it.
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
                      [0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        howso = HowsoClassifier(client=self.client)
        howso.fit(X, y)
        assert howso.score(X, y) == 1.0
        assert howso.predict(np.array([[1, 1]])).sum() == 0
        # Ensure that the trainee is unnamed, so it will be deleted.
        howso.trainee_name = None

    def test_trainee_name_getter_setter(self, classifier):
        """Test that the trainee_name setter works as expected."""
        assert classifier.trainee_name is None
        new_name = str(uuid.uuid4())
        # Invoke and test setter
        classifier.trainee_name = new_name
        # Invoke and test getter
        assert classifier.trainee_name == new_name
        # Ensure that the trainee is unnamed, so it will be deleted.
        classifier.trainee_name = None

    @pytest.mark.parametrize('trainee_is_named', [True, False])
    def test_destructor(self, mocker, trainee_is_named):
        """
        Ensure that the destructor properly unloads or deletes the trainee.

        When the trainee is named, it should unload and NOT delete and
        when the trainee is unnamed, it should delete and NOT unload.

        Since this is going to explicitly destroy the classifier, it should
        create its own.
        """
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 1, 1, 0])
        howso = HowsoClassifier(client=self.client)
        howso.fit(X, y)

        if trainee_is_named:
            howso.trainee_name = str(uuid.uuid4())
            _trainee_id = howso.trainee["id"]
        else:
            howso.trainee_name = None
            _trainee_id = None

        # Let's spy on the delete_trainee and release_trainee_resources methods
        delete_spy = mocker.spy(engine.Trainee, 'delete')
        unload_spy = mocker.spy(engine.Trainee, 'release_resources')

        # Capture their initial state (should be zero though)
        delete_count = delete_spy.call_count
        unload_count = unload_spy.call_count

        # Delete the estimator and ensure that garbage collection has occurred.
        del howso
        gc.collect()

        if trainee_is_named:
            assert delete_spy.call_count >= delete_count
            assert unload_spy.call_count > unload_count
        else:
            assert delete_spy.call_count > delete_count
            assert unload_spy.call_count >= unload_count

        # If the trainee was named, it would have been simply unloaded. Now
        # that testing has completed, ensure it is manually removed. For this
        # we'll just use `self.client``.
        if trainee_is_named and _trainee_id:
            self.client.delete_trainee(_trainee_id)

    def test_save(self, classifier):
        """
        Test that save works as intended.

        In particular, we should ensure that calling save() sets the trainee
        name, if necessary.
        """
        assert classifier.trainee_name is None
        # If this raises, obviously the test fails.
        classifier.save()
        assert classifier.trainee_name is not None

        # Testing has completed, ensure the name is reset so the estimator will
        # delete it as it is destructed.
        classifier.trainee_name = None

    def test_uniqueness_check(self, classifier):
        """Test Setting `trainee_name` to a used name fails as expected."""
        # Create a (degenerate) trainee with a known name outside of the
        # estimator (but using the same client for convenience).
        known_name = f'known-name-{uuid.uuid4()}'
        rogue_trainee = engine.Trainee(
            metadata={'fake-trainee': True},
            features={
                'a': {'type': 'nominal'},
                'b': {'type': 'nominal'},
                'c': {'type': 'nominal'},
                'd': {'type': 'nominal'}
            }
        )
        rogue_trainee["name"] = known_name
        # NOTE: This just uses the client embedded in the classifier here.
        #       This does not create a trainee for the classifier. And because
        #       of this, this test needs to explicitly delete this trainee
        #       when it is done with it.

        # Attempt to set name to existing name
        if not isinstance(classifier.client, HowsoDirectClient):
            with pytest.raises(HowsoNotUniqueError) as exc_info:
                rogue_trainee["name"] = known_name
            assert "Please use a unique name" in str(exc_info.value)
        else:
            # Ensure this doesn't raise
            rogue_trainee["name"] = known_name

        # Explicitly delete the rogue_trainee
        rogue_trainee.delete()
        # Reset to none so it won't be saved.
        rogue_trainee["name"] = None

    @pytest.mark.skipif('WIP' not in TEST_OPTIONS, reason='Local devs only')
    def test_pickle(self):
        """Test the pickling function with HowsoEstimator scikit Estimator."""
        # Test estimator; ensure it functions and has access to the client.
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
                      [0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        howso = HowsoEstimator(client=self.client, method=CLASSIFICATION)
        howso.fit(X, y)
        assert howso.score(X, y) == 1.0
        assert howso.predict(np.array([[1, 1]])).sum() == 0
        assert type(howso.client) is HowsoDirectClient

        # Pickle the Estimator; this should call howso.client.save which should
        # give the trainee a name save the trainee in the cloud.
        pickle_string = pickle.dumps(howso)
        print(f'Trainee name: {howso.trainee_name}')

        # Explicitly delete the estimator, which in turn will delete the
        # trainee and clear the variable that held the Estimator.
        howso.client.release_trainee_resources(howso.trainee_id)
        del howso

        # Load the estimator from the pickle to a new variable and conduct
        # the same assertion tests.
        howso2 = pickle.loads(pickle_string)
        assert howso2.score(X, y) == 1.0
        assert howso2.predict(np.array([[1, 1]])).sum() == 0
        assert type(howso2.client) is HowsoDirectClient

        # Delete the saved trainee to save resources.
        howso2.delete()

    def test_regressor_cv(self):
        """Test that HowsoRegressor works using cross-validation."""
        X = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        howso = HowsoRegressor(client=self.client)
        results = cross_validate(howso, X, y, cv=3)
        print(results["test_score"])

    def test_classifier_cv(self):
        """Test the HowsoClassifier works using cross-validation."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 1, 1, 0])
        howso = HowsoClassifier(client=self.client)
        results = cross_validate(howso, X, y, cv=3)
        print(results["test_score"])

    def test_clone(self):
        """Tests the ability of HowsoClassifier to be cloned by sklearn."""
        howso = HowsoClassifier(client=self.client)
        new_howso = clone(howso)
        assert howso.get_params() == new_howso.get_params()
