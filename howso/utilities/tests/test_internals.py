from collections.abc import Iterable
import warnings
from howso.utilities import internals
import pytest
from semantic_version import Version


@pytest.mark.parametrize(('features', 'result'), (
    (None, None),
    ({'test': {'type': 'ordinal'}}, {'test': {'type': 'ordinal'}}),
    ({'test': {'date_time_format': ''}}, {'test': {'date_time_format': ''}}),
    (
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S'}},
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S', 'decimal_places': 0}}
    ),
    (
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S-%f'}},
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S',
                  'original_format': {
                      'python': {'date_time_format': '%Y-%m-%dT%H:%M:%S-%f'}}
                  }}
    ),
    (
        {'test': {'date_time_format': '%H:%M:%S.%f'}},
        {'test': {'date_time_format': '%H:%M:%S',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S.%f'}
                  }}}
    ),
    (
        {'test': {'date_time_format': '%H:%M:%S,%fT%Y-%m-%d'}},
        {'test': {'date_time_format': '%H:%M:%ST%Y-%m-%d',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S,%fT%Y-%m-%d'}
                  }}}
    ),
    (
        {'test': {'date_time_format': '%H:%M:%S,%fT%Y-%m-%d'},
         'test2': {'date_time_format': '%H:%M:%S.%f'}},
        {'test': {'date_time_format': '%H:%M:%ST%Y-%m-%d',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S,%fT%Y-%m-%d'}
                  }},
         'test2': {'date_time_format': '%H:%M:%S',
                   'original_format': {
                       'python': {'date_time_format': '%H:%M:%S.%f'}
                   }}}
    ),
    (
        {'test': {'date_time_format': '%Y-%m-%d'},
         'test2': {'date_time_format': '%H:%M:%S.%f'},
         'test3': {'date_time_format': '%H:%M:%S'}},
        {'test': {'date_time_format': '%Y-%m-%d'},
         'test2': {'date_time_format': '%H:%M:%S',
                   'original_format': {
                       'python': {'date_time_format': '%H:%M:%S.%f'}
                   }},
         'test3': {'date_time_format': '%H:%M:%S', 'decimal_places': 0}}
    ),
))
def test_preprocess_feature_attributes(features, result):
    """Test preprocess_feature_attributes returns expected result."""
    output = internals.preprocess_feature_attributes(features)
    assert output == result


@pytest.mark.parametrize(('features', 'result'), (
    (None, None),
    ({'test': {'type': 'ordinal'}}, {'test': {'type': 'ordinal'}}),
    ({'test': {'date_time_format': ''}}, {'test': {'date_time_format': ''}}),
    (
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S'}},
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S'}}
    ),
    (
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S',
                  'original_format': {
                      'test': {'date_time_format': '%Y-%m-%d'}
                  }}},
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S',
                  'original_format': {
                      'test': {'date_time_format': '%Y-%m-%d'}
                  }}}
    ),
    (
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S',
                  'original_format': {
                      'python': {'date_time_format': '%Y-%m-%dT%H:%M:%S-%f'}
                  }}},
        {'test': {'date_time_format': '%Y-%m-%dT%H:%M:%S-%f',
                  'original_format': {
                      'python': {'date_time_format': '%Y-%m-%dT%H:%M:%S-%f'}
                  }}},
    ),
    (
        {'test': {'date_time_format': '%H:%M:%S',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S.%f', 'a': 'test'}
                  }}},
        {'test': {'date_time_format': '%H:%M:%S.%f',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S.%f', 'a': 'test'}
                  }}},
    ),
    (
        {'test': {'date_time_format': '%H:%M:%ST%Y-%m-%d',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S,%fT%Y-%m-%d'}
                  }}},
        {'test': {'date_time_format': '%H:%M:%S,%fT%Y-%m-%d',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S,%fT%Y-%m-%d'}
                  }}},
    ),
    (
        {'test': {'date_time_format': '%Y-%m-%d'},
         'test2': {'date_time_format': '%H:%M:%S',
                   'original_format': {
                       'python': {'date_time_format': '%H:%M:%S.%f'}
                   }},
         'test3': {'date_time_format': '%H:%M:%S'}},
        {'test': {'date_time_format': '%Y-%m-%d'},
         'test2': {'date_time_format': '%H:%M:%S.%f',
                   'original_format': {
                       'python': {'date_time_format': '%H:%M:%S.%f'}
                   }},
         'test3': {'date_time_format': '%H:%M:%S'}},
    ),
    # Test backwards compatibility shim
    (
        {'test': {'date_time_format': '%H:%M:%S',
                  'original_format': {'python': '%H:%M:%S.%f'}}},
        {'test': {'date_time_format': '%H:%M:%S.%f',
                  'original_format': {
                      'python': {'date_time_format': '%H:%M:%S.%f'}
                  }}},
    ),
))
def test_postprocess_feature_attributes(features, result):
    """Test postprocess_feature_attributes returns expected result."""
    output = internals.postprocess_feature_attributes(features)
    assert output == result


@pytest.mark.parametrize(
    'n_gen, n_requested, suppress_warning',
    ((10, 11, True), (10, 10, True), (10, 15, False))
)
def test_insufficient_case_generation_warnings(
    n_gen, n_requested, suppress_warning
):
    """
    Test to make sure `insufficient_generation_check` works.

    Parameters
    ----------
    requested_num_cases : int
        Number of cases requested by the user.
    gen_num_cases : int
        Number of cases actually generated.
    suppress_warning : bool, defaults to False
        (Optional) If True, warnings will be suppressed.
        By default, warnings will be displayed.
    """
    if n_gen < n_requested:
        # Got back less num cases, should warn the user if
        # suppress_warning is False
        if suppress_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                internals.insufficient_generation_check(
                    n_requested, n_gen, suppress_warning=suppress_warning
                )
        else:
            with pytest.warns(RuntimeWarning):
                internals.insufficient_generation_check(
                    n_requested, n_gen, suppress_warning=suppress_warning
                )
    else:
        # Got back correct num cases, shouldn't warn the user
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            internals.insufficient_generation_check(
                n_requested, n_gen, suppress_warning=suppress_warning
            )


@pytest.mark.parametrize('pandas_ver', ('2.0.0', '1.5.3'))
@pytest.mark.parametrize('format_str, is_iso', (
    ('%Y-%m-%d', True),
    ('%Y-%m-%d %H:%M:%S', True),
    ('%Y-%m-%dT%H:%M:%S', True),
    ('%Y-%m-%dT%H:%M:%SZ', True),
    ('%Y-%m-%dT%H:%M:%S%z', True),
    ('%Y-%m-%dT%H:%M:%S%Z', True),
    ('%Y-%m-%dT%H:%M:%S.%f', True),
    ('%Y-%m-%dT%H:%M:%S.%fZ', True),
    ('%Y-%m-%dT%H:%M:%S.%f%z', True),
    ('%Y-%m-%dT%H:%M:%S.%f%Z', True),
    ('%m-%d-%Y', False),
    ('%H:%M:%S', False),
    ('%Z', False),
    ('a%Y-%m-%d %H:%M:%S', False),
    ('%Y-%m-%dT%H:%M:%S.%fa', False),
))
def test_to_pandas_datetime_format(mocker, pandas_ver, format_str, is_iso):
    """Test pandas datetime format utility."""
    mocker.patch('pandas.__version__', pandas_ver)
    fmt = internals.to_pandas_datetime_format(format_str)
    ver = Version(pandas_ver)
    if ver.major is not None and ver.major >= 2:
        if is_iso:
            assert fmt == "ISO8601"
        else:
            assert fmt == format_str
    else:
        assert fmt == format_str


@pytest.mark.parametrize(
    'warning_type', (DeprecationWarning, FutureWarning, UserWarning)
)
def test_ignore_warnings_individual(warning_type):
    """Test that individual warnings are ignored."""

    def raise_future_warning(a, b):
        """Simple function that raises a Warning."""
        warnings.warn("Test Warning", warning_type)
        return a + b

    with warnings.catch_warnings(record=True) as warnings_list:
        with internals.IgnoreWarnings(warning_type):
            c = raise_future_warning(1, 2)

    assert len(warnings_list) == 0
    assert c == 3


def test_ignore_warnings_iterable(warning_type=[FutureWarning, UserWarning]):
    """Test that an iterable of warnings are ignored."""

    def raise_future_warning(a, b):
        """Simple function that raises a Warning."""
        for warning in warning_type:
            warnings.warn("Test Warning", warning)
        return a + b

    with warnings.catch_warnings(record=True) as warnings_list:
        with internals.IgnoreWarnings(warning_type):
            c = raise_future_warning(1, 2)

    assert len(warnings_list) == 0
    assert c == 3
