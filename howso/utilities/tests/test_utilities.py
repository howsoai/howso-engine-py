from collections.abc import Iterable
from datetime import datetime
from dateutil import parser
import locale
import warnings


import howso.utilities as utils
from howso.utilities import get_kwargs, LocaleOverride
import pandas as pd
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
    """
    Test that locale_override correctly switches context as desired.
    """
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
    assert locale.getlocale(category=category)[1].lower() == 'utf-8'
    assert datetime.strftime(dt, format_str) == orig_dt_str


def test_get_kwargs_simple_cases():
    """
    Test that providing a simple iterable with strings works as expected.
    """
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7}
    assert get_kwargs(kwargs, ('apple', 'banana', 'cherry')) == (5, 6, 7)


def test_get_kwargs_simple_extra_with_no_warnings():
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7, 'durian': 8}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_kwargs(kwargs, ('apple', 'banana', 'cherry')) == (5, 6, 7)


def test_get_kwargs_simple_extra_with_singular_warnings():
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7, 'durian': 8}
    with pytest.warns(UserWarning) as warn_record:
        assert get_kwargs(kwargs, ('apple', 'banana', 'cherry'),
                          warn_on_extra=True) == (5, 6, 7)
    assert 'This will be ignored.' in str(warn_record[0].message.args[0])


def test_get_kwargs_simple_extra_with_plural_warnings():
    kwargs = {'apple': 5, 'banana': 6, 'cherry': 7, 'durian': 8,
              'elderberry': 9}
    with pytest.warns(UserWarning) as warn_record:
        assert get_kwargs(kwargs, ('apple', 'banana', 'cherry'),
                          warn_on_extra=True) == (5, 6, 7)
    assert ('received unexpected parameters: [durian, elderberry]. These will '
            'be ignored.' in warn_record[0].message.args[0])


def test_get_kwargs_dict_descriptors_missing_item():
    kwargs = {'apple': 5, 'banana': 6}
    assert get_kwargs(kwargs, (
        {'key': 'apple', 'default': 5},
        {'key': 'banana', 'default': 6},
        {'key': 'cherry', 'default': 7},
    )) == (5, 6, 7)


def test_get_kwargs_dict_descriptors_missing_item_with_exception():
    kwargs = {'apple': 5, 'banana': 6}
    with pytest.raises(ValueError) as excinfo:
        get_kwargs(kwargs, (
            {'key': 'apple', 'default': 5},
            {'key': 'banana', 'default': 6},
            {'key': 'cherry', 'default': ValueError('Must provide "cherry".')},
        ))
    assert 'Must provide "cherry".' in str(excinfo.value)


def test_get_kwargs_dict_descriptors_with_tests():
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
    """
    Test the `check_feature_names` under different scenarios.
    """
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
    """
    Tests that 'validate_case_indices' correctly detects invalid case_indices arguments.
    """
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
    Tests that build_react_series_df correctly builds the DataFrame and that it includes
    the series index feature when specified.
    """
    test_react_series_response = {
        'action_features': ['id', 'x', 'y'],
        'series': [
            [["A", 1, 2], ["A", 2, 2]],
            [["B", 4, 4], ["B", 6, 7], ["B", 8, 9]]
        ]
    }

    # Without the series index feature
    columns = test_react_series_response['action_features']
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert format_str == utils.determine_iso_format(date_str, "_")
