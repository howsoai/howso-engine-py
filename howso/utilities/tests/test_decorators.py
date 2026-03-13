from time import sleep

import pytest

from howso.utilities import ConsoleFeedback


def test_console_feedback_context_manager_outputs_after_dormant(capsys):
    """Ensure output appears once dormant period has elapsed."""
    with ConsoleFeedback("Working", dormant_seconds=0, frame_delay_seconds=0.01):
        sleep(0.03)
    assert "Working" in capsys.readouterr().out


def test_console_feedback_context_manager_no_output_before_dormant(capsys):
    """Ensure no output is written if work completes before dormant period."""
    with ConsoleFeedback("Working", dormant_seconds=0.5, frame_delay_seconds=0.01):
        sleep(0.01)
    assert capsys.readouterr().out == ""


def test_console_feedback_as_decorator(capsys):
    """Ensure decorator usage behaves like context manager usage."""

    @ConsoleFeedback("Decorating", dormant_seconds=0, frame_delay_seconds=0.01)
    def run_task() -> str:
        sleep(0.03)
        return "done"

    assert run_task() == "done"
    assert "Decorating" in capsys.readouterr().out


def test_console_feedback_does_not_swallow_exception(capsys):
    """Ensure errors propagate when used as a context manager."""
    with pytest.raises(RuntimeError, match="test"):
        with ConsoleFeedback("Failing", dormant_seconds=0, frame_delay_seconds=0.01):
            sleep(0.02)
            raise RuntimeError("test")
    assert "Failing" in capsys.readouterr().out


@pytest.mark.parametrize(
    "dormant_seconds, frame_delay_seconds",
    [(-1, 1), (0, -1)],
)
def test_console_feedback_validates_durations(dormant_seconds, frame_delay_seconds):
    """Ensure invalid durations are rejected early."""
    with pytest.raises(ValueError):
        ConsoleFeedback(
            dormant_seconds=dormant_seconds,
            frame_delay_seconds=frame_delay_seconds,
        )