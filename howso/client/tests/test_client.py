from collections.abc import Mapping
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from pprint import pprint
import sys
import uuid
import warnings

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
from howso.client.schemas import Reaction, GroupReaction
from howso.direct import HowsoDirectClient
from howso.engine import Trainee
from howso.utilities import infer_feature_attributes
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
        conviction = self.client.react_group(
            trainee.id,
            features=['penguin', 'play'],
            new_cases=[cases2],
            familiarity_conviction_addition=True
        )
        assert conviction['metrics'].shape == (1, 1)
        assert conviction['metrics']['familiarity_conviction_addition'][0] > 0

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
        self.client.react_into_features(trainee.id, familiarity_conviction_addition=True, overwrite=True)
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

    # TODO: uncomment once 24545 is completed and json library does not replace 1.8e309 with a bareword
    # def test_number_overflow(self, trainee):
    #     """Test an exception is raised for a number that is too large."""
    #     # Should not raise
    #     self.client.train(trainee.id, [[1.8e307]], features=['penguin'])

    #     # Training with a number that is > 64bit should raise
    #     with pytest.raises(HowsoError):
    #         self.client.train(trainee.id, [[1.8e309, 2]],
    #                           features=['penguin', 'play'])


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
        self.client.react_into_features(trainee.id, familiarity_conviction_addition=True, overwrite=True)
        trainee_cache = self.client.trainee_cache.get_item(trainee.id)

        assert 'familiarity_conviction_addition' in trainee_cache.get('feature_attributes', {}).keys()

    def test_react_into_features_verbose(self, trainee, capsys):
        """
        Test that react_into_features is verbose when enabled.

        Test the verbose output expected when react_into_features is called.
        """
        self.client.react_into_features(trainee.id, familiarity_conviction_addition=True, overwrite=True)
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
            trainee.id, new_cases=new_cases, features=features,
            familiarity_conviction_addition=True)
        out, _ = capsys.readouterr()
        assert isinstance(ret, GroupReaction)

        df = pd.DataFrame([[1, 2, 4, 4, 4]], columns=features)
        new_cases = [df]
        ret = self.client.react_group(
            trainee.id, new_cases=new_cases, features=features,
            familiarity_conviction_addition=True)
        out, _ = capsys.readouterr()
        assert isinstance(ret, GroupReaction)

        # 2d list of cases, which is invalid
        new_cases = [[1, 2, 3, 4, 5],
                     [4, 5, 6, 7, 8],
                     [7, 8, 9, 10, 11]]
        with pytest.raises(ValueError) as exc:
            ret = self.client.react_group(
                trainee.id, new_cases=new_cases, features=features,
                familiarity_conviction_addition=True)
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

    @pytest.mark.parametrize(
        'details',
        [
            (
                {'categorical_action_probabilities': True, },
                'categorical_action_probabilities',
            ),
            (
                {'distance_contribution': True, },
                'distance_contribution',
            ),
            (
                {'influential_cases': True, 'influential_cases_familiarity_convictions': True, },
                'influential_cases',
            ),
            (
                {'num_boundary_cases': 1, 'boundary_cases_familiarity_convictions': True},
                'boundary_cases',
            ),
            (
                {'outlying_feature_values': True, },
                'outlying_feature_values',
            ),
            (
                {'similarity_conviction': True, },
                'similarity_conviction',
            ),
            (
                {'feature_robust_residuals': True, },
                'feature_robust_residuals',
            ),
            (
                {'generate_attempts': True, },
                'generate_attempts',
            ),
            (
                {'prediction_stats': True, },
                'prediction_stats',
            ),
            (
                {'observational_errors': True, },
                'observational_errors'
            ),
            (
                {'relevant_values': True, },
                'relevant_values',
            ),
            (
                {'feature_full_accuracy_contributions': True, },
                'feature_full_accuracy_contributions',
            ),
            (
                {'feature_full_residuals': True, },
                'feature_full_residuals',
            ),
            (
                {'hypothetical_values': {'sepal_width': [1, 1.4]}, },
                'hypothetical_values',
            ),
            (
                {'feature_full_residuals_for_case': True},
                'feature_full_residuals_for_case',
            ),
        ]
    )
    def test_react_format(self, details, trainee):
        """Test that various Reaction details are correctly formatted."""
        df = pd.read_csv(iris_file_path)
        with warnings.catch_warnings():
            detail_param, detail_name = details
            self.client.train(trainee.id, df)
            response = self.client.react(trainee.id,
                                         contexts=[[2, 2], [1, 2]],
                                         context_features=['sepal_width', 'sepal_length'],
                                         action_features=['petal_width'],
                                         generate_new_cases='attempt',
                                         num_cases_to_generate=2,
                                         desired_conviction=1,
                                         details=detail_param)
            details_resp = response["details"]

            if detail_name in ["influential_cases", "boundary_cases", "prediction_stats"]:
                assert isinstance(details_resp[detail_name], list)
                assert all(isinstance(item, pd.DataFrame) for item in details_resp[detail_name])
            elif detail_name in ["generate_attempts", "similarity_conviction", "distance_contribution"]:
                assert isinstance(details_resp[detail_name], list)
            elif detail_name in ["categorical_action_probabilities"]:
                assert isinstance(details_resp[detail_name], list)
                assert all(isinstance(item, dict) for item in details_resp[detail_name])
            elif detail_name in ["relevant_values"]:
                assert all(isinstance(v, pd.Series) for item in details_resp[detail_name] for v in item.values())
            elif detail_name in ["outlying_feature_values"]:
                assert all(isinstance(v, dict) for item in details_resp[detail_name] for v in item.values())
            else:  # All other details expected to be a DataFrame
                assert isinstance(details_resp[detail_name], pd.DataFrame)

    @pytest.mark.parametrize(
        'details',
        [
            (
                {'categorical_action_probabilities': True, },
                'categorical_action_probabilities',
            ),
            (
                {'distance_contribution': True, },
                'distance_contribution',
            ),
            (
                {'influential_cases': True, 'influential_cases_familiarity_convictions': True, },
                'influential_cases',
            ),
            (
                {'num_boundary_cases': 1, 'boundary_cases_familiarity_convictions': True},
                'boundary_cases',
            ),
            (
                {'outlying_feature_values': True, },
                'outlying_feature_values',
            ),
        ])
    def test_accumulate_reaction(self, details, trainee):
        """Test the accumulation of two Reaction objects."""
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.client.train(trainee.id, cases, features=['penguin', 'play'])

            detail_param, detail_name = details

            reaction_0 = self.client.react(trainee.id,
                                           contexts=[['5']],
                                           context_features=['penguin'],
                                           action_features=['play'],
                                           generate_new_cases="attempt",
                                           desired_conviction=1,
                                           details=detail_param)

            reaction_0_accum = deepcopy(reaction_0)

            reaction_1 = self.client.react(trainee.id,
                                           contexts=[['17']],
                                           context_features=['penguin'],
                                           action_features=['play'],
                                           generate_new_cases="attempt",
                                           desired_conviction=1,
                                           details=detail_param)

            reaction_0_accum.accumulate(reaction_1)

            if isinstance(reaction_0["details"][detail_name], list):
                combined_detail = reaction_0["details"][detail_name] + reaction_1["details"][detail_name]
                assert isinstance(reaction_0_accum["details"][detail_name], list)
                for idx, v in enumerate(reaction_0_accum["details"][detail_name]):
                    if isinstance(v, pd.DataFrame):
                        assert v.equals(combined_detail[idx])
                    else:
                        assert v == combined_detail[idx]
            else:
                combined_detail = pd.concat([reaction_0["details"][detail_name], reaction_1["details"][detail_name]])
                assert isinstance(reaction_0_accum["details"][detail_name], pd.DataFrame)
                assert reaction_0_accum["details"][detail_name].equals(combined_detail)

    def test_tokenizable_strings_reaction(self):
        """Test that tokenizable strings can be processed and are correctly serialized and deserialized."""
        data = {
            "product": [
                "turbo-encabulator",
                "banana-phone",
                "boneless-pizza",
            ],
            "rating": [
                5,
                3,
                1,
            ],
            "review": [
                "Not only provides inverse reactive current for use in unilateral phase detractors, but is also "
                "capable of automatically synchronizing cardinal gram-meters.",
                "it's ok. works well enough. the connection isn't very clear but what else can you expect from a "
                "banana.",
                "they forgot to take the bones out!!!!11",
            ],
        }
        df = pd.DataFrame(data)
        feature_attributes = infer_feature_attributes(df, types={"review": "continuous"})
        assert feature_attributes["review"]["original_type"]["data_type"] == "tokenizable_string"
        assert feature_attributes["review"]["data_type"] == "json"
        assert feature_attributes["review"]["type"] == "continuous"
        client = HowsoClient()
        t = Trainee()
        client.set_feature_attributes(t.id, feature_attributes)
        client.train(t.id, df)
        reaction = client.react(
            t.id,
            contexts=[[5, 'turbo-encabulator']],
            context_features=['rating', 'product'],
            action_features=['review'],
            generate_new_cases='attempt',
            details={"influential_cases": True},
            desired_conviction=5,
        )
        assert reaction["action"].iloc[0]["review"] == df.iloc[0]["review"]
        assert reaction["details"]["influential_cases"][0].iloc[0]["review"] == df.iloc[0]["review"]

    def test_json_feature_types(self):
        """Test that JSON features stored as Python data structures have their primitive types maintained."""
        tests = [
            ({"a": "str", "b": 1, "c": 2.7, "d": True, "e": {"a1": "str", "b1": {"c1": [1, 2, 3]}}},
            {"a": "string", "b": "integer", "c": "numeric", "d": "boolean", "e": {"a1": "string", "b1": {"c1": "integer"}}}),
            ({"a": "str", "b": 9, "c": 3.3, "d": False, "e": {"a1": "str2", "b1": {"c1": [1, 2, 3, 4, 5, 6, 7]}}},
            {"a": "string", "b": "integer", "c": "numeric", "d": "boolean", "e": {"a1": "string", "b1": {"c1": "integer"}}}),
            ({"a": 3, "b": 1.5, "c": 2.7, "d": True, "e": {"a1": 5, "b1": {"c1": [1, 2, 3]}}},
            {"a": "integer", "b": "numeric", "c": "numeric", "d": "boolean", "e": {"a1": "integer", "b1": {"c1": "integer"}}}),
            ({"a": 3, "b": 1, "c": 2.7, "d": True, "e": {"a1": "str", "b1": {"c1": [1, True, "foo"]}}},
            {"a": "integer", "b": "integer", "c": "numeric", "d": "boolean", "e": {"a1": "string", "b1": {"c1": "object"}}}),
        ]
        data_uniform_types = pd.DataFrame({"foo": [tests[0][0], tests[1][0]], "bar": ["a", "b"]})
        data_uniform_except_list = pd.DataFrame({"foo": [tests[2][0], tests[3][0]], "bar": ["a", "b"]})
        data_non_uniform = pd.DataFrame({"foo": [tests[0][0], tests[1][0], tests[2][0]], "bar": ["a", "b", "c"]})

        # Types should be preserved with no warnings (dict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feature_attributes = infer_feature_attributes(data_uniform_types)
            assert feature_attributes["foo"]["data_type"] == "json"
            assert feature_attributes["foo"]["type"] == "continuous"
            client = HowsoClient()
            t = Trainee()
            client.set_feature_attributes(t.id, feature_attributes)
            client.train(t.id, data_uniform_types)
            reaction = client.react(
                t.id,
                contexts=[["a"]],
                context_features=['bar'],
                action_features=['foo'],
                details={"influential_cases": True},
                desired_conviction=5,
            )
            # Cannot compare
            assert reaction["action"].iloc[0]["foo"] == data_uniform_types.iloc[0]["foo"]
            assert reaction["details"]["influential_cases"][0].iloc[0]["foo"] == tests[0][0]

        # All types except for the nested list should be preserved and a warning issued
        with pytest.warns(match="contains a key 'c1' whose value is a list of mixed types"):
            feature_attributes = infer_feature_attributes(data_uniform_except_list)
            assert feature_attributes["foo"]["data_type"] == "json"
            assert feature_attributes["foo"]["type"] == "continuous"
            client = HowsoClient()
            t = Trainee()
            client.set_feature_attributes(t.id, feature_attributes)
            client.train(t.id, data_uniform_except_list)
            reaction = client.react(
                t.id,
                contexts=[["b"]],
                context_features=['bar'],
                action_features=['foo'],
                generate_new_cases='attempt',
                details={"influential_cases": True},
                desired_conviction=5,
            )
            expected_case = deepcopy(tests[3][0])
            # The list under this key has mixed types so it will come back as-is when deserialized
            expected_case["e"]["b1"]["c1"] = json.loads(json.dumps(expected_case["e"]["b1"]["c1"]))
            assert reaction["action"].iloc[0]["foo"] == expected_case
            assert reaction["details"]["influential_cases"][0].iloc[0]["foo"] == expected_case

        # Types cannot be preserved, warning issued
        with pytest.warns(match="inconsistent types and/or keys across cases."):
            feature_attributes = infer_feature_attributes(data_non_uniform)
            assert feature_attributes["foo"]["data_type"] == "json"
            assert feature_attributes["foo"]["type"] == "continuous"
            client = HowsoClient()
            t = Trainee()
            client.set_feature_attributes(t.id, feature_attributes)
            client.train(t.id, data_non_uniform)
            reaction = client.react(
                t.id,
                contexts=[["a"]],
                context_features=['bar'],
                action_features=['foo'],
                generate_new_cases='attempt',
                details={"influential_cases": True},
                desired_conviction=5,
            )
            # Cases of "foo" have mixed types so they will come back as-is when deserialized
            expected_case = json.loads(json.dumps(tests[0][0]))
            assert reaction["action"].iloc[0]["foo"] == expected_case
            assert reaction["details"]["influential_cases"][0].iloc[0]["foo"] == expected_case
