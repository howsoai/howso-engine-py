from collections.abc import Mapping
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from pprint import pprint
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

import howso
from howso.client import HowsoClient
from howso.client.client import _check_isfile  # type: ignore reportPrivateUsage
from howso.client.client import (
    get_configuration_path,
    get_howso_client_class,
    LEGACY_CONFIG_FILENAMES,
)
from howso.client.exceptions import (
    HowsoApiError,
    HowsoConfigurationError,
    HowsoError,
)
from howso.client.protocols import ProjectClient
from howso.client.schemas.reaction import Reaction
from howso.direct import HowsoDirectClient
from howso.utilities.constants import (  # type: ignore reportPrivateUsage
    _RENAMED_DETAIL_KEYS,
    _RENAMED_DETAIL_KEYS_EXTRA,
)
from howso.utilities.testing import (
    get_configurationless_test_client,
    get_test_options,
)

TEST_OPTIONS = get_test_options()

iris_file_path = (
    Path(howso.client.__file__).parent.parent
).joinpath("utilities/tests/data/iris.csv")
np.random.default_rng(2018)

module_client = get_configurationless_test_client(client_class=HowsoClient, verbose=True)


def test_xdg_configuration_path(mocker):
    """Tests for expected behavior when XDG_CONFIG_HOME is set."""
    fake_xdg_config_path_unix = '/tmp/xdg/path'
    fake_xdg_config_path_win = 'C:\\Users\\runneradmin\\xdg\\path'
    if platform.system().lower() == 'windows':
        fake_xdg_config_path = fake_xdg_config_path_win
    else:
        fake_xdg_config_path = fake_xdg_config_path_unix
    fake_xdg_path = 'howso'
    target_path = Path(fake_xdg_config_path, fake_xdg_path, 'howso.yaml')
    names_to_remove = ('XDG_CONFIG_HOME', 'HOWSO_CONFIG')
    modified_environ = {
        k: v for k, v in os.environ.items()
        if k not in names_to_remove
    }
    modified_environ["XDG_CONFIG_HOME"] = fake_xdg_config_path

    # Only return true if our mock xdg path is set
    def _is_file(filepath):
        if filepath == target_path:
            return True
        return False

    mocker.patch.object(Path, 'is_file', autospec=True,
                        side_effect=_is_file)
    mocker.patch.dict(os.environ, modified_environ, clear=True)
    path = get_configuration_path()
    assert path == target_path


def test_xdg_not_abs_error(mocker):
    """Tests for expected error when XDG_CONFIG_HOME is not an absolute path."""
    fake_xdg_config_path = 'tmp/xdg/path'
    fake_xdg_path = 'howso'
    target_path = Path(fake_xdg_config_path, fake_xdg_path, 'howso.yaml')
    names_to_remove = ('XDG_CONFIG_HOME', 'HOWSO_CONFIG')
    modified_environ = {
        k: v for k, v in os.environ.items()
        if k not in names_to_remove
    }
    modified_environ["XDG_CONFIG_HOME"] = fake_xdg_config_path

    # Only return true if our mock xdg path is set
    def _is_file(filepath):
        if filepath == target_path:
            return True
        return False

    mocker.patch.object(Path, 'is_file', autospec=True,
                        side_effect=_is_file)
    mocker.patch.dict(os.environ, modified_environ, clear=True)
    with pytest.raises(HowsoConfigurationError) as exc:
        get_configuration_path()
    assert 'The path set in the XDG_CONFIG_HOME environment' in str(exc.value)


@pytest.mark.parametrize('existing_files, result', (
    (('config.yml', 'config.yaml'), 'config.yml'),
    (('config.yaml', 'config.yml'), 'config.yaml'),
    (('apple', 'banana', 'cherry'), None),
))
def test_check_isfile(mocker, existing_files, result):
    """Test that `_check_isfile` works as expected."""

    def _is_file(filepath):
        if str(filepath) in LEGACY_CONFIG_FILENAMES:
            return True
        return False

    mocker.patch.object(Path, 'is_file', autospec=True,
                        side_effect=_is_file)

    assert str(_check_isfile(existing_files)) == str(result)


def test_get_howso_client_class(mocker):
    """Test that the HowsoDirectClient is returned if no config is provided."""
    mocker.patch('howso.client.client.get_configuration_path',
                 return_value=None)
    klass, _ = get_howso_client_class()
    assert klass == HowsoDirectClient


class TraineeBuilder:
    """Define a TraineeBuilder."""

    def __init__(self):
        """Initialize the TraineeBuilder."""
        self.trainees = []
        self.client = module_client
        super().__init__()

    def __del__(self):
        """Delete all created trainees upon destruction of the builder."""
        self.delete_all()

    def create(self, **kwargs):
        """Create a new trainee."""
        new_trainee = self.client.create_trainee(**kwargs)
        self.trainees.append(new_trainee)
        return new_trainee

    def copy(self, trainee_id, new_trainee_name=None, project=None):
        """Copy an existing trainee."""
        if isinstance(self.client, ProjectClient):
            new_trainee = self.client.copy_trainee(trainee_id, new_trainee_name,
                                                   project)
        else:
            new_trainee = self.client.copy_trainee(trainee_id, new_trainee_name)
        self.trainees.append(new_trainee)
        return new_trainee

    def delete(self, trainee):
        """Delete a single trainee created by the builder."""
        if trainee in self.trainees:
            try:
                self.client.delete_trainee(trainee.id)
                self.trainees.remove(trainee)
            except HowsoApiError as err:
                if err.status == 404:
                    self.trainees.remove(trainee)
                else:
                    raise

    def delete_all(self):
        """Delete all trainees created by the builder."""
        for trainee in reversed(self.trainees):
            self.delete(trainee)

    def unload_all(self):
        """Unload all trainees created by the builder."""
        for trainee in self.trainees:
            self.client.release_trainee_resources(trainee.id)


@pytest.fixture(name='trainee_builder', scope='module')
def trainee_builder():
    """Define a fixture."""
    return TraineeBuilder()


@pytest.fixture(scope="function")
def howso_client():
    """Return a non-shared client."""
    return get_configurationless_test_client(client_class=HowsoClient, verbose=True)


class TestDatetimeSerialization:
    """Define a DateTimeSerialization test class."""

    @classmethod
    def setup_class(cls):
        """Setup a test client to use for the whole class."""
        cls.client = module_client

    @pytest.fixture(autouse=True)
    def trainee(self, trainee_builder):
        """Define a trainee fixture."""
        features = {'nom': {'type': 'nominal'},
                    'datetime': {'type': 'continuous',
                                 'date_time_format': '%Y-%m-%dT%H:%M:%S.%f'}
                    }
        trainee = trainee_builder.create(features=features, overwrite_trainee=True)
        try:
            yield trainee
        finally:
            trainee_builder.delete(trainee)

    def test_train(self, trainee):
        """Test that train works as expected."""
        df = pd.DataFrame(data=np.asarray([
            ['a', 'b', 'c', 'd'],
            ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.123', '2020-12-12T12:12:12.123',
             '2020-10-11T11:11:11.123']
        ]).transpose(), columns=['nom', 'datetime'])
        self.client.train(trainee.id, cases=df.values.tolist(),
                          features=df.columns.tolist())
        case_response = self.client.get_cases(
            trainee.id, session=self.client.active_session.id,
            features=['nom', 'datetime'])
        for case in case_response["cases"]:
            print(case)
        assert len(case_response["cases"]) == 4
        # datetime is fixed: zero-padded and has fractional seconds
        assert case_response["cases"][0][1].rstrip("0") == '2020-09-12T09:09:09.123'

    def test_train_warn_fix(self, trainee_builder):
        """Test that train warns when it should."""
        features = {'nom': {'type': 'nominal'},
                    'datetime': {'type': 'continuous',
                                 'date_time_format': '%Y-%m-%dT%H:%M:%S'}
                    }
        trainee = trainee_builder.create(features=features, overwrite_trainee=True)
        df = pd.DataFrame(data=np.asarray([
            ['a', 'b', 'c', 'd'],
            # missing seconds in the provided values, don't match format
            ['2020-9-12T9:09', '2020-10-12T10:10', '2020-12-12T12:12',
             '2020-10-11T11:11']
        ]).transpose(), columns=['nom', 'datetime'])
        with pytest.warns(UserWarning) as warning_list:
            self.client.train(trainee.id, cases=df.values.tolist(),
                              features=df.columns.tolist())
            # There might be other UserWarnings but assert that at least one
            # of them has the correct message.
            assert any([str(warning.message) == (
                '"datetime" has values with an incorrect datetime format, should '
                'be "%Y-%m-%dT%H:%M:%S". This feature may not work properly.')
                for warning in warning_list])

        case_response = self.client.get_cases(
            trainee.id, session=self.client.active_session.id,
            features=['nom', 'datetime'])
        for case in case_response["cases"]:
            print(case)
        assert len(case_response["cases"]) == 4
        # datetime is fixed: zero-padded and has seconds to adhere to format
        assert case_response["cases"][0][1] == '2020-09-12T09:09:00'

    def test_train_warn_bad_feature(self, trainee_builder):
        """Test that train warns on bad features."""
        features = {'nom': {'type': 'nominal'},
                    'datetime': {'type': 'continuous',
                                 'date_time_format': '%H %Y'}
                    }
        trainee = trainee_builder.create(features=features, overwrite_trainee=True)
        df = pd.DataFrame(data=np.asarray([
            ['a', 'b', 'c', 'd'],
            # missing seconds in the provided values, don't match format
            ['2020-10-11T11:11', '2020-10-11T11:11', '2020-10-11T11:11',
             '2020-10-11T11:11']
        ]).transpose(), columns=['nom', 'datetime'])
        with pytest.warns(UserWarning) as warning_list:
            self.client.train(trainee.id, df.values.tolist(), df.columns.tolist())
            assert any([str(warning.message) == (
                '"datetime" has values with an incorrect datetime format, should '
                'be "%H %Y". This feature may not work properly.')
                for warning in warning_list])

        case_response = self.client.get_cases(
            trainee.id, session=self.client.active_session.id)
        for case in case_response["cases"]:
            print(case)
        assert len(case_response["cases"]) == 4
        # datetime is not fixed and the value returned doesn't match originals
        assert case_response["cases"][0][1] != '2020-10-11T11:11'

    def test_react(self, trainee):
        """Test that react works as expected."""
        df = pd.DataFrame(data=np.asarray([
            ['a', 'b', 'c', 'd'],
            ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.333',
             '2020-12-12T12:12:12.444', '2020-10-11T11:11:11.222']
        ]).transpose(), columns=['nom', 'datetime'])
        self.client.train(trainee.id, cases=df.values.tolist(),
                          features=df.columns.tolist())
        response = self.client.react(trainee.id,
                                     contexts=[["2020-10-12T10:10:10.333"]],
                                     context_features=["datetime"],
                                     action_features=["nom"])
        assert isinstance(response, Reaction)
        assert response['action']['nom'].iloc[0] == "b"

        response = self.client.react(trainee.id, contexts=[["b"]],
                                     context_features=["nom"],
                                     action_features=["datetime"])
        assert "2020-10-12T10:10:10.333000" in response['action']['datetime'].iloc[0]

    def test_react_series(self, trainee):
        """Test that react series works as expected."""
        df = pd.DataFrame(data=np.asarray([
            ['a', 'b', 'c', 'd'],
            ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.333',
             '2020-12-12T12:12:12.444', '2020-10-11T11:11:11.222']
        ]).transpose(), columns=['nom', 'datetime'])
        self.client.train(trainee.id, cases=df.values.tolist(),
                          features=df.columns.tolist())
        response = self.client.react_series(trainee.id,
                                            series_context_values=[[["2020-10-12T10:10:10.333"]]],
                                            series_context_features=["datetime"],
                                            action_features=["nom"])
        assert isinstance(response, Reaction)
        assert response['action']['nom'].iloc[0] == "b"


class TestClient:
    """Define a Client test class."""

    @classmethod
    def setup_class(cls):
        """Setup a test client to use for the whole class."""
        cls.client = module_client

    @pytest.fixture(autouse=True)
    def trainee(self, trainee_builder):
        """Define a trainee fixture."""
        feats = {
            "penguin": {"type": "nominal"},
            "play": {"type": "nominal"},
        }
        trainee_name = uuid.uuid4().hex
        trainee = trainee_builder.create(
            name=trainee_name,
            features=feats,
            metadata={'ttl': 600000}
        )
        try:
            yield trainee
        except Exception:
            raise
        finally:
            trainee_builder.delete(trainee)

    def _train(self, trainee):
        """
        Trains default cases.

        Parameters
        ----------
        trainee
        """
        # Add more cases
        cases = [['5', '6'],
                 ['7', '8'],
                 ['9', '10'],
                 ['11', '12'],
                 ['13', '14'],
                 ['15', '16'],
                 ['17', '18'],
                 ['19', '20'],
                 ['21', '22'],
                 ['23', '24'],
                 ['25', '26']]
        self.client.train(trainee.id, cases, features=['penguin', 'play'])

    def test_constructor(self):
        """Test that the client instantiates without issue."""
        assert self.client

    def test_get(self, trainee):
        """
        Test the /get endpoint.

        Parameters
        ----------
        trainee
        """
        returned_trainee = self.client.get_trainee(trainee.id)
        assert returned_trainee == trainee

    def test_update(self, trainee):
        """
        Test the PATCH / endpoint to update a trainee.

        Parameters
        ----------
        trainee
        """
        feats = {
            "dog": {"type": "nominal"},
            "cat": {"type": "continuous"}
        }
        updated_trainee = trainee.to_dict() | dict(
            features=feats,
            metadata={'date': 'now'}
        )
        updated_trainee = self.client.update_trainee(updated_trainee)
        trainee2 = self.client.get_trainee(trainee.id)
        assert trainee2 == updated_trainee
        self.client.update_trainee(trainee)
        trainee3 = self.client.get_trainee(trainee.id)
        assert trainee3 == trainee

    def test_train_and_react(self, trainee):
        """
        Test the /train and /react endpoint.

        Parameters
        ----------
        trainee
        """
        cases = [['1', '2'], ['3', '4']]
        self.client.train(trainee.id, cases, features=['penguin', 'play'])
        react_response = self.client.react(
            trainee.id,
            contexts=[['1']],
            context_features=['penguin'],
            action_features=['play'])
        assert isinstance(react_response, Reaction)
        assert react_response['action']['play'].iloc[0] == '2'
        case_response = self.client.get_cases(
            trainee.id, session=self.client.active_session.id)
        for case in case_response["cases"]:
            print(case)
        assert len(case_response["cases"]) == 2

    def test_copy(self, trainee, trainee_builder):
        """
        Test the /copy/{trainee_id} endpoint.

        Parameters
        ----------
        trainee
        """
        new_name = trainee.name + "_copy"
        new_trainee = trainee_builder.copy(trainee.id, new_name)
        trainee_bob = self.client.get_trainee(new_trainee.id)
        assert trainee.name != new_name
        assert new_trainee.name == new_name
        assert trainee_bob.name == new_name
        orig_features = self.client.get_feature_attributes(trainee.id)
        copy_features = self.client.get_feature_attributes(new_trainee.id)
        assert orig_features == copy_features

    def test_trainee_conviction(self, trainee, trainee_builder):
        """
        Test the /copy/{trainee_id} endpoint.

        Parameters
        ----------
        trainee
        """
        cases = [['5', '6'],
                 ['7', '8'],
                 ['9', '10'],
                 ['11', '12'],
                 ['13', '14'],
                 ['15', '16'],
                 ['17', '18'],
                 ['19', '20'],
                 ['21', '22'],
                 ['23', '24'],
                 ['25', '26']]
        self.client.train(trainee.id, cases, features=['penguin', 'play'])
        cases2 = [['5', '6'],
                  ['7', '8'],
                  ['9', '10'],
                  ['11', '12'],
                  ['13', '14'],
                  ['15', '16'],
                  ['17', '18'],
                  ['19', '20'],
                  ['21', '22'],
                  ['23', '24'],
                  ['25', '26'],
                  ['27', '28']]
        conviction = self.client.react_group(trainee.id, new_cases=[cases2], features=['penguin', 'play'])
        assert conviction is not None

    def test_impute(self, trainee):
        """
        Test the /impute endpoint.

        Parameters
        ----------
        trainee
        """
        self._train(trainee)
        features = ['penguin', 'play']
        try:
            session = self.client.begin_session('impute')
            self.client.train(trainee.id, [['15', None]], features=features)
            self.client.impute(trainee.id, features=features, batch_size=2)
            imputed_session = self.client.get_cases(trainee.id, session.id,
                                                    indicate_imputed=True)
            assert imputed_session['cases'][0][1] is not None
            assert '.imputed' in imputed_session['features']
        finally:
            self.client.begin_session()  # Reset session

    def test_conviction_store(self, trainee):
        """
        Test the /conviction/store endpoint.

        Parameters
        ----------
        trainee
        """
        self._train(trainee)
        self.client.react_into_features(trainee.id, familiarity_conviction_addition=True)
        cases = self.client.get_cases(trainee.id,
                                      features=['play', 'penguin', 'familiarity_conviction_addition'],
                                      session=self.client.active_session.id)
        pprint(cases)
        assert 'familiarity_conviction_addition' in cases['features']

    def test_save(self, trainee):
        """
        Test the /save endpoint.

        Parameters
        ----------
        trainee
        """
        self.client.persist_trainee(trainee.id)

    def test_a_la_cart_data(self, trainee):
        """
        Test a-la-cart data.

        Systematically test a la cart options to ensure only the specified
        options are returned in the details data.

        Parameters
        ----------
        trainee : Trainee
            A trainee object to use.
        """
        # Add more cases
        cases = [['5', '6'],
                 ['7', '8'],
                 ['9', '10'],
                 ['11', '12'],
                 ['13', '14'],
                 ['15', '16'],
                 ['17', '18'],
                 ['19', '20'],
                 ['21', '22'],
                 ['23', '24'],
                 ['25', '26']]
        self.client.train(trainee.id, cases, features=['penguin', 'play'])

        details_sets = [
            (
                {'categorical_action_probabilities': True, },
                ['categorical_action_probabilities', ]
            ),
            (
                {'distance_contribution': True, },
                ['distance_contribution', ]
            ),
            (
                {'influential_cases': True, 'influential_cases_familiarity_convictions': True, },
                ['influential_cases', ]
            ),
            (
                {'num_boundary_cases': 1, 'boundary_cases_familiarity_convictions': True, },
                ['boundary_cases', ]
            ),
            (
                {'outlying_feature_values': True, },
                ['outlying_feature_values', ]
            ),
            (
                {'similarity_conviction': True, },
                ['similarity_conviction', ]
            ),
            (
                {'feature_robust_residuals': True, },
                ['feature_robust_residuals', ]
            ),
        ]
        for audit_detail_set, keys_to_expect in details_sets:
            response = self.client.react(trainee.id,
                                         contexts=[['1']],
                                         context_features=['penguin'],
                                         action_features=['play'],
                                         details=audit_detail_set)
            details = response['details']
            assert (all(details[key] is not None for key in keys_to_expect))

    @pytest.mark.parametrize('old_key,new_key', _RENAMED_DETAIL_KEYS.items())
    def test_deprecated_detail_keys_react(self, trainee, old_key, new_key):
        """Ensure using any of the deprecated keys raises a warning, but continues to work."""
        # These keys shouldn't be tested like this:
        if new_key in [
            "feature_full_directional_prediction_contributions",
            "feature_robust_directional_prediction_contributions",
            "feature_full_accuracy_contributions_permutation",
            "feature_robust_accuracy_contributions_permutation",
        ]:
            return

        with pytest.warns(DeprecationWarning) as record:
            self.client.train(
                trainee.id, [[1, 2], [1, 2], [1, 2]],
                features=['penguin', 'play']
            )
            reaction = self.client.react(
                trainee.id,
                contexts=[['1']],
                context_features=['penguin'],
                action_features=['play'],
                details={old_key: True}
            )

        # Check that the correct warning was raised.
        assert len(record)
        # There may be multiple warnings. Ensure at least one of them contains
        # the deprecation message.
        assert any([
            f"'{old_key}' is deprecated" in str(r.message)
            for r in record
        ])

        # We DO want the old_key to be present during the deprecation period.
        assert old_key in reaction.get('details', {}).keys()

        # We do NOT want the new_key present during the deprecation period.
        assert new_key not in reaction.get('details', {}).keys()

        # Some keys request multiple keys to be returned, these too should be
        # converted to the old names if the old name was originally used.
        if old_key in _RENAMED_DETAIL_KEYS_EXTRA.keys():
            for old_extra_key, new_extra_key in _RENAMED_DETAIL_KEYS_EXTRA[old_key]["additional_keys"].items():
                assert new_extra_key not in reaction.get('details', {}).keys()
                assert old_extra_key in reaction.get('details', {}).keys()

    @pytest.mark.parametrize('old_key,new_key', _RENAMED_DETAIL_KEYS.items())
    def test_deprecated_detail_keys_react_aggregate(self, trainee, old_key, new_key):
        """Ensure using any of the deprecated keys raises a warning, but continues to work."""
        # These keys shouldn't be tested like this:
        if new_key in {
            "case_full_prediction_contributions",
            "case_robust_prediction_contributions",
            "feature_full_prediction_contributions_for_case",
            "feature_robust_prediction_contributions_for_case",
            "feature_full_residual_convictions_for_case",
            "feature_full_residuals_for_case",
            "feature_robust_residuals_for_case",
            "case_full_accuracy_contributions",
            "case_robust_accuracy_contributions",
            "feature_full_directional_prediction_contributions",
            "feature_robust_directional_prediction_contributions",
            "feature_full_accuracy_contributions_ex_post",
            "feature_robust_accuracy_contributions_ex_post",
        }:
            return

        with pytest.warns(DeprecationWarning) as record:
            self.client.train(
                trainee.id, [[1, 2], [1, 2], [1, 2]],
                features=['penguin', 'play']
            )
            response = self.client.react_aggregate(
                trainee.id,
                prediction_stats_action_feature='penguin',
                num_samples=1,
                details={old_key: True}
            )

        # Check that the correct warning was raised.
        assert len(record)
        # There may be multiple warnings. Ensure at least one of them contains
        # the deprecation message.
        assert any([
            f"'{old_key}' is deprecated" in str(r.message)
            for r in record
        ])

        # No point in testing further if we didn't get back a Mapping instance.
        assert isinstance(response, Mapping), "react_aggregate did not return a Mapping."

        # We DO want the old_key to be present during the deprecation period.
        assert old_key in response.keys()

        # We do NOT want the new_key present during the deprecation period.
        assert new_key not in response.keys()

        # Some keys request multiple keys to be returned, these too should be
        # converted to the old names if the old name was originally used.
        if old_key in _RENAMED_DETAIL_KEYS_EXTRA.keys():
            for old_extra_key, new_extra_key in _RENAMED_DETAIL_KEYS_EXTRA[old_key]["additional_keys"].items():
                assert new_extra_key not in response.keys()
                assert old_extra_key in response.keys()

    def test_get_version(self):
        """Test get_version()."""
        version = self.client.get_version()
        assert version['client'] == importlib.metadata.version('howso-engine')

    def test_doublemax_to_infinity_translation(self):
        """Test the translation from Double.MAX_VALUE to Infinity."""
        from howso.utilities import replace_doublemax_with_infinity

        dat = {
            'familiarity_conviction': [sys.float_info.max],
            'influential_cases': [{
                'some_feature': {
                    'familiarity_conviction': sys.float_info.max
                }
            }]
        }
        dat = replace_doublemax_with_infinity(dat)
        assert (np.isinf(dat['familiarity_conviction'][0]))
        assert (np.isinf(dat['influential_cases'][0]['some_feature']['familiarity_conviction']))

    def test_number_overflow(self, trainee):
        """Test an exception is raised for a number that is too large."""
        # Should not raise
        self.client.train(trainee.id, [[1.8e307]], features=['penguin'])

        # Training with a number that is > 64bit should raise
        with pytest.raises(HowsoError):
            self.client.train(trainee.id, [[1.8e309, 2]],
                              features=['penguin', 'play'])


class TestBaseClient:
    """Define a BaseClient test class."""

    @classmethod
    def setup_class(cls):
        """Setup a test client to use for the whole class."""
        cls.client = module_client

    @pytest.fixture(autouse=True)
    def trainee(self, trainee_builder):
        """Define a trainee fixture."""
        df = pd.read_csv(iris_file_path)
        header = list(df.columns)
        features = {header[0]: {'type': 'continuous'},
                    header[1]: {'type': 'continuous'},
                    header[2]: {'type': 'continuous'},
                    header[3]: {'type': 'continuous'},
                    header[4]: {'type': 'nominal'}
                    }
        trainee = trainee_builder.create(features=features, overwrite_trainee=True)
        try:
            yield trainee
        except Exception:
            raise
        finally:
            trainee_builder.delete(trainee)

    def test_configuration_exception(self):
        """Test the Configuration Exception Class Constructor."""
        message = "test message"
        configuration_exception = HowsoConfigurationError(message=message)
        assert configuration_exception.message == message

    def test_impute_verbose(self, trainee, capsys):
        """Test the verbose output expected during the execution of impute."""
        self.client.impute(trainee.id)
        out, _ = capsys.readouterr()
        assert f'Imputing Trainee with id: {trainee.id}' in out

    def test_remove_cases_verbose(self, trainee, capsys):
        """
        Test that remove_cases has verbose output when enabled.

        Test for verbose output expected when remove_cases is called with.
        """
        condition = {"class": None}
        self.client.remove_cases(trainee.id, 1, condition=condition)
        out, _ = capsys.readouterr()
        assert f"Removing case(s) from Trainee with id: {trainee.id}" in out

    def test_update_trainee_verbose(self, trainee_builder, capsys):
        """
        Test that update_trainee has verbose output when enabled.

        Test for verbose output expected when update_trainee is called.
        """
        df = pd.read_csv(iris_file_path)
        header = list(df.columns)
        features = {header[0]: {'type': 'continuous'},
                    header[1]: {'type': 'continuous'},
                    header[2]: {'type': 'continuous'},
                    header[3]: {'type': 'continuous'},
                    header[4]: {'type': 'nominal'}
                    }
        trainee = trainee_builder.create(features=features, overwrite_trainee=True)
        trainee.name = 'test-update-verbose'
        updated_trainee = self.client.update_trainee(trainee)
        assert trainee.name == updated_trainee.name
        assert updated_trainee.name == 'test-update-verbose'
        out, _ = capsys.readouterr()
        assert f'Updating Trainee with id: {trainee.id}' in out

    def test_get_trainee_verbose(self, trainee, capsys):
        """
        Test that get_trainee has verbose output when enabled.

        Test for verbose output expected when get_trainee is called.
        """
        self.client.get_trainee(trainee.id)
        out, _ = capsys.readouterr()
        assert f'Getting Trainee with id: {trainee.id}' in out

    def test_set_feature_attributes(self, trainee, capsys):
        """Test that set_feature_attributes works as expected."""
        attributes = {
            "length": {"type": "continuous", "decimal_places": 1},
            "width": {"type": "continuous", "significant_digits": 4},
            "degrees": {"type": "continuous", "cycle_length": 360},
            "class": {"type": "nominal"}
        }
        self.client.set_feature_attributes(trainee.id,
                                           feature_attributes=attributes)
        out, _ = capsys.readouterr()
        assert (f'Setting feature attributes for Trainee '
                f'with id: {trainee.id}') in out

    def test_get_feature_attributes(self, trainee, capsys):
        """Test get_feature attributes returns the expected output."""
        attributes = {
            "length": {"type": "continuous", "decimal_places": 1},
            "width": {"type": "continuous", "significant_digits": 4},
            "degrees": {"type": "continuous", "cycle_length": 360},
            "class": {"type": "nominal"}
        }

        self.client.set_feature_attributes(trainee.id,
                                           feature_attributes=attributes)
        output = self.client.get_feature_attributes(trainee.id)
        out, _ = capsys.readouterr()
        assert (f'Getting feature attributes from Trainee '
                f'with id: {trainee.id}') in out
        assert output == attributes

    def test_get_sessions_verbose(self, trainee, capsys):
        """
        Test that get_sessions has verbose output when enabled.

        Test for verbose output expected when get_sessions is called.
        """
        self.client.get_sessions(trainee.id)
        out, _ = capsys.readouterr()
        assert f'Getting sessions from Trainee with id: {trainee.id}' in out

    def test_analyze_verbose(self, trainee, capsys):
        """Test for verbose output expected when analyze is called."""
        context_features = ['class']
        self.client.train(trainee.id, [['iris-setosa']], context_features)
        self.client.analyze(trainee.id, context_features)
        out, _ = capsys.readouterr()
        assert f'Analyzing Trainee with id: {trainee.id}' in out
        assert 'Analyzing Trainee with parameters: ' in out

    def test_acquire_trainee_resources_verbose(self, trainee, capsys):
        """
        Test that acquire_trainee_resources is verbose when enabled.

        Test for the verbose output expected when acquiring trainee resources.
        """
        self.client.persist_trainee(trainee.id)
        self.client.acquire_trainee_resources(trainee.id)
        out, _ = capsys.readouterr()
        assert f'Acquiring resources for Trainee with id: {trainee.id}' in out

    def test_save_trainee_verbose(self, trainee, capsys):
        """
        Test that save_trainee is verbose when enabled.

        Test for the verbose output expected when persist_trainee is called.
        """
        self.client.persist_trainee(trainee.id)
        out, _ = capsys.readouterr()
        assert f'Saving Trainee with id: {trainee.id}' in out

    def test_release_trainee_resources_verbose(self, trainee, capsys):
        """
        Test that release_trainee_resources is verbose when enabled.

        Test for the verbose output expected when releasing trainee resources.
        """
        self.client.release_trainee_resources(trainee.id)
        out, _ = capsys.readouterr()
        assert f'Releasing resources for Trainee with id: {trainee.id}' in out

    def copy_trainee_verbose(self, trainee, trainee_builder, capsys):
        """
        Test that copy_trainee is verbose when enabled.

        Test for the verbose output expected when copy_trainee is called.
        """
        trainee_builder.copy(trainee.id, f'new_{trainee.id}')
        out, _ = capsys.readouterr()
        assert f'Copying Trainee {trainee.id} to new_{trainee.id}' in out

    def test_delete_trainee_verbose(self, trainee, capsys):
        """
        Test that delete_trainee is verbose when enabled.

        Test for the verbose output expected when delete_trainee is called.
        """
        self.client.delete_trainee(trainee.id)
        out, _ = capsys.readouterr()
        assert 'Deleting Trainee' in out

    def test_remove_cases_exception(self, trainee):
        """
        Test that remove_cases raises when expected.

        Test for the expected exception when num_cases=0 is passed into
        remove cases. (num_cases must be a value greater than 0).
        """
        condition = {"feature_name": None}
        with pytest.raises(ValueError) as exc:
            self.client.remove_cases(trainee.id, 0, condition=condition)
        assert str(exc.value) == '`num_cases` must be a value greater than 0 if specified.'

    def test_react_exception(self, trainee):
        """
        Test that react raises when expected.

        Test for the expected Exception when contexts and case_indices are not
        specified when calling react.
        """
        with pytest.raises((ValueError, HowsoApiError)) as exc:
            self.client.react(trainee.id, desired_conviction=None, contexts=None)
        msg = ("If `contexts` are not specified, both `case_indices` and "
               "`preserve_feature_values` must be specified.")
        assert msg in str(exc.value)

    def test_get_num_training_cases(self, trainee):
        """Test the that output is the expected type: int."""
        number_cases = self.client.get_num_training_cases(trainee.id)
        assert isinstance(number_cases, int)

    def test_react_into_features_updated_feature_attributes(self, trainee):
        """Test that react_into_features updates the feature attributes."""
        self.client.react_into_features(trainee.id, familiarity_conviction_addition=True)
        trainee_cache = self.client.trainee_cache.get_item(trainee.id)

        assert 'familiarity_conviction_addition' in trainee_cache.get('feature_attributes', {}).keys()

    def test_react_into_features_verbose(self, trainee, capsys):
        """
        Test that react_into_features is verbose when enabled.

        Test the verbose output expected when react_into_features is called.
        """
        self.client.react_into_features(trainee.id, familiarity_conviction_addition=True)
        out, _ = capsys.readouterr()
        assert ('Reacting into features on Trainee with id') in out

    def test_get_feature_conviction_verbose(self, trainee, capsys):
        """
        Test that get_feature_conviction is verbose when enabled.

        Test for the verbose output expected when get_feature_conviction
        is called.
        """
        self.client.get_feature_conviction(trainee.id)
        out, _ = capsys.readouterr()
        assert 'Getting conviction of features for Trainee with id' in out

    def test_get_params_verbose(self, trainee, capsys):
        """
        Test that get_params is verbose when enabled.

        Test for the verbose output expected when get_params is called.
        """
        self.client.get_params(trainee.id)
        out, _ = capsys.readouterr()
        assert (f'Getting model attributes from Trainee with '
                f'id: {trainee.id}') in out

    @pytest.mark.parametrize('params', (
        {"hyperparameter_map": {
            "targetless": {
                "f1.f2.f3.": {
                    ".none": {"dt": -1, "p": .1, "k": 8}
                }
            }
        }},
    ))
    def test_set_params_verbose(self, trainee, capsys, params):
        """Test for the verbose output expected when set_params is called."""
        self.client.set_params(trainee.id, params)
        out, _ = capsys.readouterr()
        assert (f'Setting model attributes for Trainee with '
                f'id: {trainee.id}') in out

    def test_set_and_get_params(self, trainee, trainee_builder):
        """Test for set_params and get_params functionality."""
        param_map = {"hyperparameter_map": {
            "targeted": {
                "petal_length": {
                    "sepal_length.sepal_width.": {
                        ".none": {
                            "dt": -1, "p": .1, "k": 2
                        }
                    }
                }
            }
        }}
        new_cases = [[1, 2, 3, 4, 5],
                     [4, 5, 6, 7, 8],
                     [7, 8, 9, 10, 11],
                     [1, 2, 3, 4, 5]]
        features = ['sepal_length', 'sepal_width', 'petal_length',
                    'petal_width', 'class']

        self.client.train(trainee.id, new_cases, features=features)
        self.client.set_params(trainee.id, param_map)

        # get a prediction with the set parameters
        first_pred = self.client.react(
            trainee.id,
            contexts=[[2, 2]],
            context_features=['sepal_length', 'sepal_width'],
            action_features=['petal_length'],
        )['action']['petal_length'].iloc[0]

        # create another trainee
        other_trainee = trainee_builder.create(
            features={"sepal_length": {'type': 'continuous'},
                      "sepal_width": {'type': 'continuous'},
                      "petal_length": {'type': 'continuous'},
                      "petal_width": {'type': 'continuous'},
                      "class": {'type': 'nominal'}},
            overwrite_trainee=True
        )
        other_trainee = self.client.update_trainee(other_trainee)
        self.client.train(other_trainee.id, new_cases, features=features)

        # make a prediction on the same case, prediction should be different
        second_pred = self.client.react(
            other_trainee.id,
            contexts=[[2, 2]],
            context_features=['sepal_length', 'sepal_width'],
            action_features=['petal_length'],
        )['action']['petal_length'].iloc[0]
        assert first_pred != second_pred

        # align parameters, make another prediction that should be the same
        self.client.set_params(other_trainee.id, param_map)
        third_pred = self.client.react(
            other_trainee.id,
            contexts=[[2, 2]],
            context_features=['sepal_length', 'sepal_width'],
            action_features=['petal_length'],
        )['action']['petal_length'].iloc[0]
        assert first_pred == third_pred

        # verify that both trainees have the same hyperparameter_map now
        first_params = self.client.get_params(trainee.id)
        second_params = self.client.get_params(other_trainee.id)
        assert first_params['hyperparameter_map'] == second_params['hyperparameter_map']

    def test_get_specific_hyperparameters(self, trainee):
        """Test to verify parameters of get_params are functional."""
        param_map = {"hyperparameter_map": {
            "targeted": {
                "petal_length": {
                    "sepal_length.sepal_width.": {
                        ".none": {
                            "dt": -1, "p": .1, "k": 2
                        }
                    }
                }
            },
            "targetless": {
                "sepal_length.sepal_width.": {
                    ".none": {
                        "dt": -1, "p": .5, "k": 3
                    }
                }
            }
        }}

        self.client.set_params(trainee.id, param_map)

        params = self.client.get_params(trainee.id, action_feature='')
        assert params['hyperparameter_map'] == {"dt": -1, "p": .5, "k": 3}

        params = self.client.get_params(trainee.id, action_feature='petal_length')
        assert params['hyperparameter_map'] == {"dt": -1, "p": .1, "k": 2}

    def test_get_configuration_path_exceptions(self):
        """
        Test that get_configuration_path raises when expected.

        Test for multiple expected when there are issues with the path passed
        in or the configuration yaml file.
        """
        with pytest.raises(HowsoConfigurationError) as exc:
            get_configuration_path(config_path='Fake/Path')
        assert 'Specified configuration file was not found' in str(exc.value)

    def test_base_client_initializer_exception(self):
        """
        Test that the base client initializer raises when expected.

        Tests for the expected exception when an instance of the
        HowsoClient is created with an invalid path
        """
        with pytest.raises(HowsoConfigurationError) as exc:
            HowsoClient(config_path="Fake/Path", verbose=True)
        assert "Specified configuration file was not found" in str(exc)

    def test_set_get_substitute_feature_values(self, trainee, capsys):
        """
        Test that set_substitute_feature_values works as expected.

        Test the functionality of both set_substitute_feature_values and
        get_feature_values as well as checks for expected verbose output.
        """
        df = pd.read_csv(iris_file_path)
        header = list(df.columns)
        substitution_value_map = {header[0]: {'type': 'continuous'},
                                  header[1]: {'type': 'continuous'},
                                  header[2]: {'type': 'continuous'},
                                  header[3]: {'type': 'continuous'},
                                  header[4]: {'type': 'nominal'}
                                  }
        self.client.set_substitute_feature_values(
            trainee.id, substitution_value_map)
        out, _ = capsys.readouterr()
        assert (f'Setting substitute feature values for '
                f'Trainee with id: {trainee.id}') in out
        ret = self.client.get_substitute_feature_values(trainee.id)
        out, _ = capsys.readouterr()
        assert (f'Getting substitute feature values from Trainee with '
                f'id: {trainee.id}') in out
        assert ret == substitution_value_map
        ret = self.client.get_substitute_feature_values(trainee.id)
        assert ret == {}

    def test_set_get_feature_attributes(self, trainee, capsys):
        """
        Test that set_- and get_feature_attributes work as expected.

        Test the the functionality of both set_substitute_feature_attributes
        and get_feature_attributes as well as checks for expected verbose
        output.
        """
        feats = {
            "length": {"type": "continuous", "decimal_places": 1},
            "width": {"type": "continuous", "significant_digits": 4},
            "degrees": {"type": "continuous", "cycle_length": 360},
            "class": {"type": "nominal"}
        }

        self.client.set_feature_attributes(trainee.id, feature_attributes=feats)
        out, _ = capsys.readouterr()
        assert (f'Setting feature attributes for Trainee '
                f'with id: {trainee.id}') in out

        ret = self.client.get_feature_attributes(trainee.id)
        out, _ = capsys.readouterr()
        assert (f'Getting feature attributes from Trainee with '
                f'id: {trainee.id}') in out
        assert ret == feats

    def test_react_exceptions(self, trainee):
        """
        Test that react raises when expected.

        Test for expected exceptions when react is called with certain values
        passed in.
        """
        df = pd.read_csv(iris_file_path)
        features = list(df.columns)
        num_cases_to_generate = 10
        with pytest.raises(HowsoError) as exc:
            self.client.react(trainee.id,
                              desired_conviction=1.0,
                              preserve_feature_values=features,
                              case_indices=[('test', 1), ('test', 2)],
                              num_cases_to_generate=num_cases_to_generate)
        assert (
            "The number of `case_indices` provided does not match "
            "the number of cases to generate."
        ) in str(exc.value)
        num_cases_to_generate = 1
        with pytest.raises(HowsoError) as exc:
            self.client.react(trainee.id,
                              desired_conviction=1.0,
                              preserve_feature_values=features,
                              num_cases_to_generate=num_cases_to_generate,
                              contexts=[['test']])
            assert ("The number of provided context values in "
                    "`contexts` does not match the number of features in "
                    "`context_features`.") in str(exc.value)

    def test_react_group(self, trainee, capsys):
        """
        Test that react_group works as expected.

        Tests for expected verbose output and expected return type (list) when
        get_case_familiarity conviction is called.
        """
        new_cases = [[[1, 2, 3, 4, 5],
                      [4, 5, 6, 7, 8],
                      [7, 8, 9, 10, 11]],
                     [[1, 2, 3, 4, 5]]]
        features = ['sepal_length', 'sepal_width', 'petal_length',
                    'petal_width', 'class']
        ret = self.client.react_group(
            trainee.id, new_cases=new_cases, features=features)
        out, _ = capsys.readouterr()
        assert isinstance(ret, dict)

        df = pd.DataFrame([[1, 2, 4, 4, 4]], columns=features)
        new_cases = [df]
        ret = self.client.react_group(
            trainee.id, new_cases=new_cases, features=features)
        out, _ = capsys.readouterr()
        assert isinstance(ret, dict)

        # 2d list of cases, which is invalid
        new_cases = [[1, 2, 3, 4, 5],
                     [4, 5, 6, 7, 8],
                     [7, 8, 9, 10, 11]]
        with pytest.raises(ValueError) as exc:
            ret = self.client.react_group(
                trainee.id, new_cases=new_cases, features=features)
        assert (
            "Improper shape of `new_cases` values passed. "
            "`new_cases` must be a 3d list of object."
        ) in str(exc.value)

    def test_marginal_stats(self, trainee):
        """Test for get_marginal_stats and its parameters."""
        new_cases = [[1, 2, 3, 4, 0],
                     [4, 6, 6, 7, 1],
                     [6, 8, 9, 9, 1],
                     [1, 2, 3, 4, 2]]
        features = ['sepal_length', 'sepal_width', 'petal_length',
                    'petal_width', 'class']
        self.client.train(trainee.id, new_cases, features=features)

        marginal_stats = self.client.get_marginal_stats(trainee.id)
        assert marginal_stats['sepal_length']['mean'] == 3.0
        assert marginal_stats['sepal_width']['mean'] == 4.5
        assert marginal_stats['sepal_width']['count'] == 4

        conditional_marginal_stats = self.client.get_marginal_stats(
            trainee.id,
            condition={"class": '1'},
        )
        assert conditional_marginal_stats['class']['count'] == 2
        assert conditional_marginal_stats['petal_width']['mean'] == 8

    def test_remove_feature_verbose(self, trainee, capsys):
        """Test for expected verbose output when remove_feature is called."""
        feature = 'test'
        self.client.remove_feature(trainee.id, feature=feature)
        out, _ = capsys.readouterr()
        assert (f'Removing feature "{feature}" from Trainee with id: '
                f'{trainee.id}') in out

    @pytest.mark.parametrize(
        'json_map, expected_output',
        [
            (
                json.loads('{"message":"java.lang.IllegalStateException: '
                           'Expected a", "code":0}'),
                'Howso Error: HowsoError /0'
            ),
            (
                json.loads('{"message":"java.lang.IllegalStateException: '
                           'Expected a "}'),
                'Howso Error:  HowsoError /-1'
            ),
            (
                json.loads('{"test":"general"}'),
                'General Exception HowsoError'
            )
        ]
    )
    def test_raise_howso_error(self, capsys, json_map, expected_output):
        """Test for various expected error outputs with various json_maps."""
        with pytest.raises(Exception):
            self.client._raise_howso_error(json_map)
            out, _ = capsys.readouterr()
            assert expected_output in out

    def test_get_configuration_path(self, mocker):
        """
        Test that get_configuration_path works as expected.

        Tests for expected output when the mocked functions return None and
        True respectively.
        """
        mocker.patch('os.environ.get', return_value=None)
        mocker.patch.object(Path, 'is_file', return_value=True)
        path = get_configuration_path()
        assert str(path) == 'howso.yml'
