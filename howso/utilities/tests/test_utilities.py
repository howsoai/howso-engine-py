from collections.abc import Iterable
from datetime import datetime
import locale
import platform
import warnings

from dateutil import parser
import howso.utilities as utils
from howso.utilities import get_kwargs, get_matrix_diff, matrix_processing, LocaleOverride
from howso.utilities.reaction import Reaction
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from . import has_locales


@pytest.mark.skipif(
    not has_locales(['fr_FR.utf8']),
    reason="Test locale: 'fr_FR' is not available")
@pytest.mark.parametrize('language_code, encoding, category, result,', (
    # Tests simple parameters for a language_code and an encoding.
    ('fr_FR', 'utf-8', locale.LC_ALL, 'mar. 13 oct. 2020 17:02:27'),
    # Tests that utf-8 will be used by default
    ('fr_FR', None, locale.LC_TIME, 'mar. 13 oct. 2020 17:02:27'),
    # Tests that an explicit encoding (utf-8) overrides an embedded one.
    ('fr_FR.crazy_encoding_designation', 'utf-8', locale.LC_TIME,
        'mar. 13 oct. 2020 17:02:27'),
))
def test_locale_override(language_code, encoding, category, result):
    """Test that locale_override correctly switches context as desired."""
    dt = parser.parse('2020-10-13 17:02:27.243860T-0500')
    # Unfortunately just using `%c` is inconsistent across operating systems.
    format_str = '%a %d %b %Y %H:%M:%S'
    orig_locale = locale.getlocale(category=category)
    orig_dt_str = datetime.strftime(dt, format_str)

    # This will only be displayed in the output if the test fails and can
    # provide useful context.
    print([
        (k, v) for k, v in locale.locale_alias.items()
        if language_code.lower() in k
    ])

    # Ensure we were able to perform the switch for the duration of the body
    # of the context manager.
    with LocaleOverride(language_code=language_code, encoding=encoding,
                        category=category):
        date_str = datetime.strftime(dt, format_str)
    assert date_str == result

    # Now, ensure it's now back in the original locale
    assert locale.getlocale(category=category) == orig_locale
    # Windows default locale is 1252
    assert locale.getlocale(category=category)[1].lower() == '1252' if platform.system() == 'Windows' else 'utf-8'
    assert datetime.strftime(dt, format_str) == orig_dt_str


def test_get_kwargs_simple_cases():
    """Test that providing a simple iterable with strings works as expected."""
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7}
    assert get_kwargs(kwargs, ('apple', 'banana', 'cherry')) == (5, 6, 7)


def test_get_kwargs_simple_extra_with_no_warnings():
    """Test get_kwargs with no warnings."""
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7, 'durian': 8}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_kwargs(kwargs, ('apple', 'banana', 'cherry')) == (5, 6, 7)


def test_get_kwargs_simple_extra_with_singular_warnings():
    """Test get_kwargs with singular warnings."""
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7, 'durian': 8}
    with pytest.warns(UserWarning) as warn_record:
        assert get_kwargs(kwargs, ('apple', 'banana', 'cherry'),
                          warn_on_extra=True) == (5, 6, 7)
    assert 'This will be ignored.' in str(warn_record[0].message.args[0])


def test_get_kwargs_simple_extra_with_plural_warnings():
    """Test get_kwargs with plural warnings."""
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7, 'durian': 8,
              'elderberry': 9}
    with pytest.warns(UserWarning) as warn_record:
        assert get_kwargs(kwargs, ('apple', 'banana', 'cherry'),
                          warn_on_extra=True) == (5, 6, 7)
    assert ('received unexpected parameters: [durian, elderberry]. These will '
            'be ignored.' in warn_record[0].message.args[0])


def test_get_kwargs_dict_descriptors_missing_item():
    """Test get_kwargs with a missing item."""
    kwargs = {'apple': 5, 'banana': 6}
    assert get_kwargs(kwargs, (
        {'key': 'apple', 'default': 5},
        {'key': 'banana', 'default': 6},
        {'key': 'cherry', 'default': 7},
    )) == (5, 6, 7)


def test_get_kwargs_dict_descriptors_missing_item_with_exception():
    """Test get_kwargs with a missing item and exception."""
    kwargs = {'apple': 5, 'banana': 6}
    with pytest.raises(ValueError) as excinfo:
        get_kwargs(kwargs, (
            {'key': 'apple', 'default': 5},
            {'key': 'banana', 'default': 6},
            {'key': 'cherry', 'default': ValueError('Must provide "cherry".')},
        ))
    assert 'Must provide "cherry".' in str(excinfo.value)


def test_get_kwargs_dict_descriptors_with_tests():
    """Test get_kwargs descriptors."""
    kwargs = {'apple': [5, ], 'banana': 6, 'cherry': []}

    def is_non_empty_iterable(value):
        return isinstance(value, Iterable) and len(value) > 0

    # Check the non-exception raising entries first
    assert (
        get_kwargs(kwargs, (
            {'key': 'apple', 'default': [5, ], 'test': is_non_empty_iterable},
            {'key': 'banana', 'default': [6, ], 'test': is_non_empty_iterable},
            {'key': 'cherry', 'default': [7, ], 'test': is_non_empty_iterable},
        )) == ([5, ], [6, ], [7, ])
    )

    # Now check the failing test for an existing item:
    with pytest.raises(ValueError) as excinfo:
        get_kwargs(kwargs, ({
            'key': 'cherry',
            'default': ValueError('Must provide a non-empty list for "cherry".'),  # noqa
            'test': is_non_empty_iterable,
        }, ))
    assert 'Must provide a non-empty list for "cherry".' in str(excinfo.value)

    # Now check the failing test for a non-existent item:
    with pytest.raises(ValueError) as excinfo:
        get_kwargs(kwargs, ({
            'key': 'durian',
            'default': ValueError('Must provide a non-empty list for "durian".'),  # noqa
            'test': is_non_empty_iterable,
        }, ))
    assert 'Must provide a non-empty list for "durian".' in str(excinfo.value)


def test_check_feature_names():
    """Test the `check_feature_names` under different scenarios."""
    # 1. Pass in a valid list
    columns = ["a", "b", "c"]
    features = {"a": 1, "b": 2, "c": 3}

    assert utils.check_feature_names(
        features, columns, raise_error=True
    ) is True

    # 2. Pass in a valid set
    columns = {"a", "b", "c"}
    features = {"a": 1, "b": 2, "c": 3}

    assert utils.check_feature_names(
        features, columns, raise_error=True
    ) is True

    # 3. Pass in a invalid list with `raise_error`=False (default)
    columns = ["a", "b"]
    features = {"a": 1, "b": 2, "c": 3}

    assert utils.check_feature_names(features, columns) is not True

    # 4. Pass in a invalid set `raise_error`=False (default)
    columns = {"a", "b"}
    features = {"a": 1, "b": 2, "c": 3}

    assert utils.check_feature_names(features, columns) is not True

    # 5. Pass in a invalid list with `raise_error`=True (default)
    columns = ["a", "b"]
    features = {"a": 1, "b": 2, "c": 3}

    with pytest.raises(ValueError) as error_:
        utils.check_feature_names(features, columns, raise_error=True)
        assert (
            "The feature names in `features` differs from `column_names`" in
            str(error_)
        )

    # 6. Pass in a invalid set with `raise_error`=True (default)
    columns = {"a", "b"}
    features = {"a": 1, "b": 2, "c": 3}

    with pytest.raises(ValueError) as error_:
        utils.check_feature_names(features, columns, raise_error=True)
        assert (
            "The feature names in `features` differs from `column_names`" in
            str(error_)
        )


def test_validate_case_indices():
    """Tests that 'validate_case_indices' correctly detects invalid case_indices arguments."""
    msg = 'Argument case_indices must be type Iterable of (non-string) Sequence[str, int].'
    # Test a case_indices that is not an iterable
    with pytest.raises(ValueError) as exc:
        utils.validate_case_indices(12345)
    assert str(exc.value) == msg

    # Test case_indices that do not contain sequences
    with pytest.raises(ValueError) as exc:
        utils.validate_case_indices([1, 2, 3])
    assert str(exc.value) == msg

    with pytest.raises(ValueError) as exc:
        utils.validate_case_indices([("test", 0), dict(idx=0), ["test", 1]])
    assert str(exc.value) == msg

    # Test case_indices that contain sequences with invalid types
    with pytest.raises(ValueError) as exc:
        utils.validate_case_indices([(1, "test"), ("test", 1)])
    assert str(exc.value) == msg

    with pytest.raises(ValueError) as exc:
        utils.validate_case_indices([("test", 1), ("test", dict(idx=0)), ["test", 3]])
    assert str(exc.value) == msg

    with pytest.raises(ValueError) as exc:
        utils.validate_case_indices([("test", 1), ("test", 2), 'a1'])
    assert str(exc.value) == msg

    # Test thorough validation
    long_invalid = [("valid", 0) for _ in range(1, 101)] + [(101, "invalid")]
    with pytest.raises(ValueError) as exc:
        utils.validate_case_indices(long_invalid, thorough=True)
    assert str(exc.value) == msg

    # A non-thorough call should not catch the invalid sequence
    utils.validate_case_indices(long_invalid)

    # Test case_indices that are valid. No exception should be raised.
    utils.validate_case_indices([("test", 1)])
    utils.validate_case_indices([["test", 0], ("test", 1)])


def test_build_react_series_df():
    """
    Tests that build_react_series_df correctly builds the DataFrame.

    Also tests that it includes the series index feature when specified.
    """
    test_react_series_response = {
        'details': {'action_features': ['id', 'x', 'y']},
        'action': [
            [["A", 1, 2], ["A", 2, 2]],
            [["B", 4, 4], ["B", 6, 7], ["B", 8, 9]]
        ]
    }

    # Without the series index feature
    columns = test_react_series_response['details']['action_features']
    expected_data = [["A", 1, 2], ["A", 2, 2], ["B", 4, 4], ["B", 6, 7], ["B", 8, 9]]
    expected_df = pd.DataFrame(expected_data, columns=columns)
    df = utils.build_react_series_df(test_react_series_response)
    assert df.equals(expected_df)

    # With the series index feature
    columns = [".series"] + columns
    expected_data = [
        ["series_1", "A", 1, 2],
        ["series_1", "A", 2, 2],
        ["series_2", "B", 4, 4],
        ["series_2", "B", 6, 7],
        ["series_2", "B", 8, 9]
    ]
    expected_df = pd.DataFrame(expected_data, columns=columns)
    df = utils.build_react_series_df(test_react_series_response, series_index='.series')
    assert df.equals(expected_df)


@pytest.mark.parametrize('date_str, format_str', (
    ("2020-01-01", "%Y-%m-%d"),
    ("2020-01-01T20:10:10", "%Y-%m-%dT%H:%M:%S"),
    ("2020-01-01T20:10:10.123", "%Y-%m-%dT%H:%M:%S.%f"),
    ("2020-01-01T20:10:10+0000", "%Y-%m-%dT%H:%M:%S%z"),
    ("2020-01-01T20:10:10-0000", "%Y-%m-%dT%H:%M:%S%z"),
    ("2020-01-01T20:10:10+05:00", "%Y-%m-%dT%H:%M:%S%z"),
    ("2020-01-01T20:10:10UTC", "%Y-%m-%dT%H:%M:%S%Z"),
    ("2020-01-01T20:10:10Z", "%Y-%m-%dT%H:%M:%SZ"),
    ("2020-01-01T20:10:10.123Z", "%Y-%m-%dT%H:%M:%S.%fZ"),
))
def test_determine_iso_format(date_str, format_str):
    """Tests utils.determine_iso_format."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert format_str == utils.determine_iso_format(date_str, "_")


def test_cases_with_details_add_reaction():
    """Tests that `Reaction` `add_reaction` works with different data types."""
    df = pd.DataFrame(data=np.asarray([
        ['a', 'b', 'c', 'd'],
        ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.333',
            '2020-12-12T12:12:12.444', '2020-10-11T11:11:11.222']
    ]).transpose(), columns=['nom', 'datetime'])

    react_response = {
        'details': {'action_features': ['datetime']},
        'action': df
    }

    cwd = Reaction()
    cwd.add_reaction(react_response['action'], react_response['details'])
    cwd.add_reaction(react_response['action'].to_dict(), react_response['details'])
    # List of dicts
    cwd.add_reaction(react_response['action'].to_dict(orient='records'), react_response['details'])
    cwd.add_reaction(Reaction(react_response['action'], react_response['details']))

    assert cwd['action'].shape[0] == 16


def test_cases_with_details_instantiate():
    """Tests that `Reaction` can be instantiated with different data types."""
    df = pd.DataFrame(data=np.asarray([
        ['a', 'b', 'c', 'd'],
        ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.333',
            '2020-12-12T12:12:12.444', '2020-10-11T11:11:11.222']
    ]).transpose(), columns=['nom', 'datetime'])

    react_response = {
        'details': {'action_features': ['datetime']},
        'action': df
    }

    cwd = Reaction(react_response['action'], react_response['details'])
    assert cwd['action'].shape[0] == 4

    cwd = Reaction(react_response['action'].to_dict(), react_response['details'])
    assert cwd['action'].shape[0] == 4

    cwd = Reaction(react_response['action'].to_dict(orient='records'), react_response['details'])
    assert cwd['action'].shape[0] == 4


def test_reaction_reorganized_details_invalid():
    """Tests that `Reaction` `reorganized_details` property works."""
    df = pd.DataFrame(data=np.asarray([
        ['a', 'b', 'c', 'd'],
        ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.333',
            '2020-12-12T12:12:12.444', '2020-10-11T11:11:11.222']
    ]).transpose(), columns=['nom', 'datetime'])

    # 'action_features' is a special key and should also not raise any warnings.
    react_response = {
        'details': {
            'action_features': ['datetime'],
            'similarity_conviction': [5],
            'invalid': [4],
        },
        'action': df
    }

    cwd = Reaction(react_response['action'], react_response['details'])
    with pytest.warns(UserWarning, match="Unrecognized detail keys found: \\[invalid\\] and"):
        cwd.reorganized_details


def test_get_matrix_diff():
    """Tests that `get_matrix_diff` works properly."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    }, index=['a', 'b', 'c']).T

    differences_dict = get_matrix_diff(df)

    correct_dict = {
        ('a', 'c'): 4,
        ('a', 'b'): 2,
        ('b', 'c'): 2
    }

    assert differences_dict == correct_dict


@pytest.mark.parametrize(
    'normalize, ignore_diagonals_normalize, abval, fill_diagonal, fill_diagonal_value',
    (
        (False, True, False, False, 2),
        (True, False, False, False, 2),
        (True, True, False, False, 2),
        (False, True, True, False, 2),
        (False, True, False, True, 2),
        (True, False, True, False, 2),
    )
)
def test_matrix_processing(
    normalize,
    ignore_diagonals_normalize,
    abval,
    fill_diagonal,
    fill_diagonal_value
):
    """Tests that `matrix_processing` works properly."""
    df = pd.DataFrame({
        'a': [1.5, -3.0, 6.0],
        'b': [0.75, -1.5, 3.0],
        'c': [0.75, 1.5, 3.0],
    }, index=['a', 'b', 'c']).T

    # `matrix_processing` only sorts if all other parameters are False
    if not any([normalize, abval, fill_diagonal]):
        processed_matrix = matrix_processing(
            matrix=df,
            normalize=normalize,
            ignore_diagonals_normalize=ignore_diagonals_normalize,
            abval=abval,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )
        assert_frame_equal(processed_matrix, df)

    # Tests `normalize` parameter with `ignore_diagonals_normalize` set to False
    if normalize and not any([ignore_diagonals_normalize, abval, fill_diagonal]):
        processed_matrix = matrix_processing(
            matrix=df,
            normalize=normalize,
            ignore_diagonals_normalize=ignore_diagonals_normalize,
            abval=abval,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )
        correct_matrix = pd.DataFrame({
            'a': [0.25, -0.5, 1.0],
            'b': [0.25, -0.5, 1.0],
            'c': [0.25, 0.5, 1.0]
        }, index=['a', 'b', 'c']).T

        assert_frame_equal(processed_matrix, correct_matrix)

    # Tests `normalize` parameter with `ignore_diagonals_normalize` set to True
    if normalize and ignore_diagonals_normalize and not any(
        [abval, fill_diagonal]
    ):
        processed_matrix = matrix_processing(
            matrix=df,
            normalize=normalize,
            ignore_diagonals_normalize=ignore_diagonals_normalize,
            abval=abval,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )
        # The diagonals are preserved
        correct_matrix = pd.DataFrame({
            'a': [1.50, -0.5, 1.0],
            'b': [0.25, -1.5, 1.0],
            'c': [0.50, 1.0, 3.0]
        }, index=['a', 'b', 'c']).T

        assert_frame_equal(processed_matrix, correct_matrix)

    # Tests `abval` parameter
    if abval and not any([normalize, fill_diagonal]):
        processed_matrix = matrix_processing(
            matrix=df,
            normalize=normalize,
            ignore_diagonals_normalize=ignore_diagonals_normalize,
            abval=abval,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )
        correct_matrix = pd.DataFrame({
            'a': [1.5, 3.0, 6.0],
            'b': [0.75, 1.5, 3.0],
            'c': [0.75, 1.5, 3.0],
        }, index=['a', 'b', 'c']).T

        assert_frame_equal(processed_matrix, correct_matrix)

    # Tests `fill_diagonal` parameter
    if fill_diagonal and not any([abval, normalize]):
        processed_matrix = matrix_processing(
            matrix=df,
            normalize=normalize,
            ignore_diagonals_normalize=ignore_diagonals_normalize,
            abval=abval,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )
        correct_matrix = pd.DataFrame({
            'a': [fill_diagonal_value, -3.0, 6.0],
            'b': [0.75, fill_diagonal_value, 3.0],
            'c': [0.75, 1.5, fill_diagonal_value],
        }, index=['a', 'b', 'c']).T

        assert_frame_equal(processed_matrix, correct_matrix)

    # Tests `normalize` and `abval` parameter with `ignore_diagonals_normalize` set to False
    if all([abval, normalize]) and not fill_diagonal:
        processed_matrix = matrix_processing(
            matrix=df,
            normalize=normalize,
            ignore_diagonals_normalize=ignore_diagonals_normalize,
            abval=abval,
            fill_diagonal=fill_diagonal,
            fill_diagonal_value=fill_diagonal_value
        )
        correct_matrix = pd.DataFrame({
            'a': [0.25, 0.5, 1.0],
            'b': [0.25, 0.5, 1.0],
            'c': [0.25, 0.5, 1.0],
        }, index=['a', 'b', 'c']).T

        assert_frame_equal(processed_matrix, correct_matrix)
