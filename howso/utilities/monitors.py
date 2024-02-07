from datetime import datetime, timedelta
from typing import Union

__all__ = [
    "ProgressTimer",
    "Timer",
]


class Timer:
    """
    Simple context manager to capture run duration of the inner context.

    Usage::

        with Timer() as my_timer:
            # perform time-consuming task here...
        print(f"The task took {my_timer.duration}."

    Results in::

        "The task took 1:30:10.454419"
    """

    def __init__(self):
        """Initialize a new Timer instance."""
        self.start_time = None
        self.end_time = None

    def start(self) -> "Timer":
        """Start the timer."""
        self.reset()
        self.start_time = datetime.now()
        return self

    def end(self) -> None:
        """End the timer."""
        self.end_time = datetime.now()

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None

    @property
    def has_started(self) -> bool:
        """If the timer has started."""
        return self.start_time is not None

    @property
    def has_ended(self) -> bool:
        """If the timer has ended."""
        return self.end_time is not None

    @property
    def duration(self) -> Union[timedelta, None]:
        """
        The total computed duration of the timer.

        Returns
        -------
        timedelta or None
            The total duration of the timer. When the timer has not yet ended,
            the duration between now and when the timer started will be
            returned. If the timer has not yet started, returns None.
        """
        if self.start_time is None:
            return None
        if self.end_time is None:
            return datetime.now() - self.start_time
        return self.end_time - self.start_time

    @property
    def seconds(self) -> Union[float, None]:
        """The total seconds representing the duration of timer instance."""
        if duration := self.duration:
            return duration.total_seconds()
        return None

    def __enter__(self):
        """Context entrance."""
        return self.start()

    def __exit__(self, *args):
        """Context exit."""
        self.end()


class ProgressTimer(Timer):
    """
    Monitor progress of a task.

    Parameters
    ----------
    total_ticks : int, default 100
        The total number of ticks in the progress meter.
    start_tick : int, default 0
        The starting tick.
    """

    def __init__(self, total_ticks: int = 100, *, start_tick: int = 0):
        """Initialize a new ProgressTimer instance."""
        super().__init__()
        self.last_tick_time = None
        self.total_ticks = total_ticks
        self.current_tick = start_tick
        self.update_count = 0
        self._start_tick = start_tick

    @property
    def time_remaining(self) -> timedelta:
        """
        The estimated time remaining.

        Returns
        -------
        timedelta
            The time estimated to be remaining.

        Raises
        ------
        ValueError
            If timer not yet started.
        """
        if not self.has_started:
            raise ValueError("Progress timer not started")
        elapsed_time = datetime.now() - self.start_time
        return ((self.total_ticks * elapsed_time /
                 max(self.current_tick, 1)) - elapsed_time)

    @property
    def tick_duration(self) -> Union[timedelta, None]:
        """
        The duration since the last tick.

        Returns
        -------
        timedelta or None
            The duration since the last tick, or None if not yet started.
        """
        if not self.last_tick_time:
            return None
        return datetime.now() - self.last_tick_time

    @property
    def is_complete(self) -> bool:
        """If progress has reached completion."""
        return self.current_tick == self.total_ticks

    @property
    def progress(self) -> float:
        """The current progress percentage."""
        return self.current_tick / self.total_ticks

    def update(self, ticks: int = 1) -> None:
        """
        Update the progress by given ticks.

        Parameters
        ----------
        ticks : int, default 1
            The number of ticks to increment/decrement by.
        """
        self.current_tick += ticks
        if self.current_tick > self.total_ticks:
            self.current_tick = self.total_ticks
        elif self.current_tick < 0:
            self.current_tick = 0
        self.last_tick_time = datetime.now()
        self.update_count += 1

    def reset(self) -> None:
        """Reset the progress timer."""
        super().reset()
        self.last_tick_time = None
        self.update_count = 0
        self.current_tick = self._start_tick

    def start(self) -> "ProgressTimer":
        """Start the progress timer."""
        super().start()
        self.last_tick_time = self.start_time
        return self

    def __enter__(self) -> "ProgressTimer":
        """Context entrance."""
        return self.start()
