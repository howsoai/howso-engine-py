from __future__ import annotations

from contextlib import ContextDecorator, nullcontext
import sys
import threading
import time
from typing import Any


class ConsoleFeedback(ContextDecorator):
    """
    Processing feedback helper.

    Can be used as either a decorator or a context manager.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        text: str = "Processing",
        dormant_seconds: int = 60,
        frame_delay_seconds: int = 15,
    ) -> None:
        if dormant_seconds < 0:
            raise ValueError("`dormant_seconds` must be >= 0.")
        if frame_delay_seconds < 0:
            raise ValueError("`frame_delay_seconds` must be >= 0.")
        self.text = text
        self.dormant_seconds = dormant_seconds
        self.frame_delay_seconds = frame_delay_seconds
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._granularity: float = 0.1

    def _run_spinner(self) -> None:
        # Wait for the dormant period with fine-grained checks
        for _ in range(int(self.dormant_seconds / self._granularity)):
            if self._stop_event is None or self._stop_event.is_set():
                return
            time.sleep(self._granularity)

        # Animation frames for the ellipsis
        frames = [" ...", ". ..", ".. .", "... "]
        frame_index = 0

        while self._stop_event is not None and not self._stop_event.is_set():
            frame = frames[frame_index % len(frames)]
            sys.stdout.write(f"\r{self.text}{frame}")
            sys.stdout.flush()

            # Wait for the next frame delay with fine-grained checks
            for _ in range(int(self.frame_delay_seconds / self._granularity)):
                if self._stop_event is None or self._stop_event.is_set():
                    break
                time.sleep(self._granularity)

            frame_index += 1

        # Clear the line after completion
        sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
        sys.stdout.flush()

    def _start(self) -> None:
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_spinner, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        if self._stop_event is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> "ConsoleFeedback":
        """Enter a managed context."""
        self._start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Exit the managed context."""
        self._stop()
        return False