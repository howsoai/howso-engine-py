from concurrent.futures import ThreadPoolExecutor
import datetime
import threading
from typing import Any
import warnings

import pandas as pd
import pytest
from semantic_version import Version

from howso.utilities import internals
from howso.utilities.monitors import ProgressTimer


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
    (None, {}),
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


def test_fixed_batch_scaler() -> None:
    batch_scaler = internals.FixedBatchScalingManager(100)
    assert batch_scaler.batch_size == 100

    # Even with the scale-up and scale-down events from the normal batch
    # scaler tests, this still emits a fixed size.
    batch_scaler.update(datetime.timedelta(seconds=30), None)
    assert batch_scaler.batch_size == 100

    batch_scaler.update(datetime.timedelta(seconds=90), None)
    assert batch_scaler.batch_size == 100

    # Cannot manually set the size.
    with pytest.raises(AttributeError):
        batch_scaler.batch_size = 50  # pyright: ignore[reportAttributeAccessIssue]


def test_batch_scaler_time() -> None:
    batch_scaler = internals.BatchScalingManager(100, thread_count=4, max_size=250)
    assert batch_scaler.batch_size == 100

    # Send a batch shorter than 60 seconds and it should scale up.
    # Internal factor increases by sqrt(5)/2, rounded to a multiple of 4.
    batch_scaler.update(datetime.timedelta(seconds=30), None)
    assert batch_scaler.batch_size == 160

    # This will hit the maximum size, which rounds down to the thread count.
    batch_scaler.update(datetime.timedelta(seconds=30), None)
    assert batch_scaler.batch_size == 248

    # This will stay there
    batch_scaler.update(datetime.timedelta(seconds=30), None)
    assert batch_scaler.batch_size == 248

    # Send a batch longer than 75 seconds and it should scale down.
    # Internal factor decreases by 0.5.
    batch_scaler.update(datetime.timedelta(seconds=90), None)
    assert batch_scaler.batch_size == 124

    # Anything 60-75 seconds should be unchanged.
    batch_scaler.update(datetime.timedelta(seconds=70), None)
    assert batch_scaler.batch_size == 124


def test_batch_scaler_threads() -> None:
    batch_scaler = internals.BatchScalingManager(100, thread_count=4, max_size=250)
    batch_scaler.thread_count = 8
    assert batch_scaler.batch_size == 200

    # This hits the maximum size, and it winds up rounding up to the limit.
    batch_scaler.thread_count = 16
    assert batch_scaler.batch_size == 250

    # This will divide it in half; so it's 125; but rounds to the nearest
    # multiple of 8
    batch_scaler.thread_count = 8
    assert batch_scaler.batch_size == 128


def test_batch_scaler_modify() -> None:
    batch_scaler = internals.BatchScalingManager(100, thread_count=4, max_size=250)
    assert batch_scaler.batch_size == 100

    # We can set the size to whatever we want, but it rounds to the thread count.
    batch_scaler.batch_size = 83
    assert batch_scaler.batch_size == 84  # which is different (rounded)

    # Scaling follows the internal size, even if it rounds differently.
    batch_scaler.update(datetime.timedelta(seconds=90), None)
    assert batch_scaler.batch_size == 40  # which is not half of the previous value


@pytest.mark.parametrize(
    "date_time_values,time_feature_format,expected_valid,expected_coerced,expected_invalid",
    [
        # All values already match the format perfectly and are strings
        (
            ["2024-01-15 14:30:00", "2024-02-20 09:15:00"],
            "%Y-%m-%d %H:%M:%S",
            ["2024-01-15 14:30:00", "2024-02-20 09:15:00"],
            [],
            [],
        ),
        # datetime.date objects coerced to datetime format (time added as 00:00:00)
        (
            [datetime.date(2024, 1, 15), datetime.date(2024, 2, 20)],
            "%Y-%m-%d %H:%M:%S",
            ["2024-01-15 00:00:00", "2024-02-20 00:00:00"],
            [
                (datetime.date(2024, 1, 15), "2024-01-15 00:00:00"),
                (datetime.date(2024, 2, 20), "2024-02-20 00:00:00"),
            ],
            [],
        ),
        # datetime.date objects matching date-only format (no coercion)
        (
            [datetime.date(2024, 1, 15), datetime.date(2024, 2, 20)],
            "%Y-%m-%d",
            ["2024-01-15", "2024-02-20"],
            [],
            [],
        ),
        # String values that need format coercion
        (
            ["01/15/2024", "02/20/2024"],
            "%Y-%m-%d",
            ["2024-01-15", "2024-02-20"],
            [("01/15/2024", "2024-01-15"), ("02/20/2024", "2024-02-20")],
            [],
        ),
        # Mixed valid and invalid values
        (
            ["2024-01-15", "invalid-date", datetime.date(2024, 2, 20)],
            "%Y-%m-%d",
            ["2024-01-15", "2024-02-20"],
            [],
            ["invalid-date"],
        ),
        # Timezone-aware datetime rejected when format has no timezone directive
        (
            [datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=datetime.timezone.utc)],
            "%Y-%m-%d %H:%M:%S",
            [],
            [],
            [datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=datetime.timezone.utc)],
        ),
        # Timezone-naive datetime rejected when format has timezone directive
        (
            [datetime.datetime(2024, 1, 15, 14, 30, 0)],
            "%Y-%m-%d %H:%M:%S%z",
            [],
            [],
            [datetime.datetime(2024, 1, 15, 14, 30, 0)],
        ),
        # Timezone-aware datetime accepted when format has timezone directive
        (
            [datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=datetime.timezone.utc)],
            "%Y-%m-%d %H:%M:%S%z",
            ["2024-01-15 14:30:00+0000"],
            [],
            [],
        ),
        # datetime.datetime with time component coerced to different datetime format
        (
            [datetime.datetime(2024, 1, 15, 14, 30, 45)],
            "%m/%d/%Y %I:%M %p",
            ["01/15/2024 02:30 PM"],
            [(datetime.datetime(2024, 1, 15, 14, 30, 45), "01/15/2024 02:30 PM")],
            [],
        ),
        # datetime.datetime with time component coerced to date-only format
        (
            [datetime.datetime(2024, 1, 15, 14, 30, 45)],
            "%m-%d-%Y",
            ["01-15-2024"],
            [(datetime.datetime(2024, 1, 15, 14, 30, 45), "01-15-2024")],
            [],
        ),
        # Integer time feature values and no time feature format
        (
            [1, 2, 3],
            None,
            [1, 2, 3],
            [],
            [],
        ),
    ],
)
def test_coerce_date_time_formats(
    date_time_values, time_feature_format, expected_valid, expected_coerced, expected_invalid
):
    """Test the coerce_date_time_formats function with various input scenarios."""
    # Create minimal feature_attributes structure
    feature_attributes = {
        "time_feature": {
            "time_series": {"time_feature": True},
            "date_time_format": time_feature_format,
        }
    }

    valid, coerced, invalid, _ = internals.coerce_date_time_formats(date_time_values, feature_attributes)

    assert valid == expected_valid, f"Valid values mismatch: {valid} != {expected_valid}"
    assert coerced == expected_coerced, f"Coerced values mismatch: {coerced} != {expected_coerced}"
    assert invalid == expected_invalid, f"Invalid values mismatch: {invalid} != {expected_invalid}"


def test_coerce_date_time_formats_missing_time_feature():
    """Test that ValueError is raised when time feature is missing."""
    feature_attributes = {"some_feature": {"type": "continuous"}}

    with pytest.raises(ValueError, match="The provided feature attributes do not indicate a time feature"):
        internals.coerce_date_time_formats(["2024-01-15"], feature_attributes)


def test_coerce_date_time_formats_missing_date_time_format():
    """Test that ValueError is raised when date_time_format is missing."""
    feature_attributes = {
        "time_feature": {
            "time_series": {"time_feature": True},
            # Missing date_time_format
        }
    }

    # Should come back as invalid
    _, _, invalid, _ = internals.coerce_date_time_formats(["2024-01-15"], feature_attributes)
    assert len(invalid) == 1
    assert invalid[0] == "2024-01-15"


class _RecordingReact:
    """React function that records the parameters of every batch it receives."""

    def __init__(self) -> None:
        self.batch_params: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, _trainee_id: str, params: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        """Record ``params`` and echo the batch's context values as its action values."""
        with self._lock:
            self.batch_params.append(params)
        values = params["context_values"]
        if values is None:
            values = [[0]] * params["num_cases_to_generate"]
        return {"action_features": ["y"], "action_values": list(values)}, 0, 0


def _run_react_in_batches(
    params: dict[str, Any],
    *,
    total_size: int,
    concurrency: int | None,
    react: _RecordingReact,
    num_to_generate_param: str | None = None,
) -> dict[str, Any]:
    """Run ``ReactInBatches`` over ``params`` in fixed batches of 2."""
    return internals.ReactInBatches.run(
        trainee_id="trainee",
        params=params,
        total_size=total_size,
        batch_size=2,
        initial_batch_size=None,
        get_thread_count=lambda _trainee_id: 1,
        get_concurrency=lambda _trainee_id: concurrency,
        params_for_batch=internals.ParamsForBatch({"context_values"}, num_to_generate_param=num_to_generate_param),
        react_function=react,
    )


@pytest.mark.parametrize("concurrency", [1, 3])
def test_react_in_batches_parallel_omits_task_id(concurrency: int) -> None:
    """Batches submitted concurrently carry no task_id, and the caller's params keep theirs."""
    params: dict[str, Any] = {"context_values": [[i] for i in range(6)], "task_id": "shared", "details": None}
    react = _RecordingReact()

    result = _run_react_in_batches(params, total_size=6, concurrency=concurrency, react=react)

    assert len(react.batch_params) == 3
    assert all("task_id" not in batch for batch in react.batch_params)
    assert all("details" in batch for batch in react.batch_params)
    assert result["action_values"] == [[i] for i in range(6)]
    assert params["task_id"] == "shared"


def test_react_in_batches_parallel_generative_omits_task_id() -> None:
    """Generative batches get their own case count and no task_id."""
    params: dict[str, Any] = {
        "context_values": None,
        "desired_conviction": 5.0,
        "num_cases_to_generate": 5,
        "task_id": "shared",
    }
    react = _RecordingReact()

    result = _run_react_in_batches(
        params, total_size=5, concurrency=2, react=react, num_to_generate_param="num_cases_to_generate"
    )

    assert sorted(batch["num_cases_to_generate"] for batch in react.batch_params) == [1, 2, 2]
    assert all("task_id" not in batch for batch in react.batch_params)
    assert len(result["action_values"]) == 5
    assert params["task_id"] == "shared"
    assert params["num_cases_to_generate"] == 5


def test_react_in_batches_serial_keeps_task_id() -> None:
    """Batches run one at a time all carry the caller's task_id."""
    params: dict[str, Any] = {"context_values": [[i] for i in range(6)], "task_id": "shared"}
    react = _RecordingReact()

    result = _run_react_in_batches(params, total_size=6, concurrency=None, react=react)

    assert [batch["task_id"] for batch in react.batch_params] == ["shared", "shared", "shared"]
    assert result["action_values"] == [[i] for i in range(6)]


class _RecordingProgressTimer(ProgressTimer):
    """Progress timer that records the tick count of every update."""

    def __init__(self, total_ticks: int) -> None:
        super().__init__(total_ticks)
        self.updates: list[int] = []

    def update(self, ticks: int = 1) -> None:
        """Record ``ticks`` and advance the timer."""
        self.updates.append(ticks)
        super().update(ticks)


class _RecordingBatchScaler(internals.FixedBatchScalingManager):
    """Fixed-size batch scaler that counts its timing updates."""

    def __init__(self, batch_size: int) -> None:
        super().__init__(batch_size)
        self.update_count = 0

    def update(self, batch_duration: datetime.timedelta, memory_sizes: tuple[int, int] | None) -> int:
        """Count the update and return the fixed batch size."""
        self.update_count += 1
        return super().update(batch_duration, memory_sizes)


class _QueueDrainRace:
    """
    Gated batch react that finishes batch 1 while batch 0 is being consumed.

    Batch 1 blocks until the progress callback for batch 0's result releases
    it, and that callback returns only once batch 1's future is done.  So
    batch 1 finishes after ``wait()`` has reported only batch 0, and before
    ``ReactInBatches`` looks at the head of its queue again.  After batch 1
    finishes, ``get_concurrency`` reports ``drained_concurrency`` until
    ``unstick`` is set, and 2 otherwise.
    """

    def __init__(self, *, drained_concurrency: int) -> None:
        self.react_in_batches: internals.ReactInBatches | None = None
        self.unstick = threading.Event()
        self._drained_concurrency = drained_concurrency
        self._batch_1_gate = threading.Event()
        self._batch_1_done = threading.Event()

    def react(self, _trainee_id: str, params: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        """Echo the batch's context values, holding batch 1 until its gate opens."""
        values = params["context_values"]
        if values[0][0] == 1:
            self._batch_1_gate.wait()
        return {"action_features": ["y"], "action_values": list(values)}, 0, 0

    def progress_callback(self, _progress: ProgressTimer, results: dict[str, Any] | None) -> None:
        """On batch 0's result, release batch 1 and block until its future is done."""
        if results is None or results["action_values"] != [[0]]:
            return
        assert self.react_in_batches is not None
        self._batch_1_gate.set()
        # Batch 0 has already left the queue, so batch 1's future is at its head.
        self.react_in_batches._futures[0][1].result()
        self._batch_1_done.set()

    def get_concurrency(self, _trainee_id: str) -> int:
        """Report 2 concurrent requests, or ``drained_concurrency`` once batch 1 is done."""
        if self._batch_1_done.is_set() and not self.unstick.is_set():
            return self._drained_concurrency
        return 2


@pytest.mark.parametrize("drained_concurrency", [2, 1])
def test_react_in_batches_parallel_batch_finishing_during_drain(drained_concurrency: int) -> None:
    """A batch that finishes while earlier results are consumed is counted, freed, and timed once."""
    total = 3
    scenario = _QueueDrainRace(drained_concurrency=drained_concurrency)
    scaler = _RecordingBatchScaler(1)
    progress = _RecordingProgressTimer(total)
    with progress:
        react_in_batches = internals.ReactInBatches(
            trainee_id="trainee",
            params={"context_values": [[i] for i in range(total)]},
            progress=progress,
            batch_scaler=scaler,
            get_thread_count=lambda _trainee_id: 1,
            get_concurrency=scenario.get_concurrency,
            params_for_batch=internals.ParamsForBatch({"context_values"}),
            react_function=scenario.react,
            progress_callback=scenario.progress_callback,
        )
        scenario.react_in_batches = react_in_batches

        # parallel() runs on a worker thread so that a livelock fails this test
        # rather than hanging it.  result() re-raises anything parallel() raises.
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(react_in_batches.parallel)
            try:
                running.result(timeout=10)
            except TimeoutError:
                # Restore concurrency so a stuck submit loop can finish and the
                # worker exits before the executor shuts down.
                scenario.unstick.set()
                running.result(timeout=10)
                pytest.fail("ReactInBatches.parallel() did not finish")

    assert react_in_batches.result["action_values"] == [[0], [1], [2]]
    assert progress.updates == [1, 1, 1]
    assert react_in_batches._running == set()
    assert scaler.update_count == 2


def test_react_in_batches_parallel_batch_error_propagates() -> None:
    """An exception from one parallel batch propagates out of ``run``."""

    def react(_trainee_id: str, params: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        """Fail batch 1 and echo every other batch's context values."""
        values = params["context_values"]
        if values[0][0] == 1:
            raise RuntimeError("batch failed")
        return {"action_features": ["y"], "action_values": list(values)}, 0, 0

    with pytest.raises(RuntimeError, match="batch failed"):
        internals.ReactInBatches.run(
            trainee_id="trainee",
            params={"context_values": [[i] for i in range(4)]},
            total_size=4,
            batch_size=1,
            initial_batch_size=None,
            get_thread_count=lambda _trainee_id: 1,
            get_concurrency=lambda _trainee_id: 2,
            params_for_batch=internals.ParamsForBatch({"context_values"}),
            react_function=react,
        )
