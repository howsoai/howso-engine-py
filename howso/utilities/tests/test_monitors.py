from datetime import timedelta
from time import sleep

from howso.utilities import FrozenTimer, Timer


def test_timer():
    """Simple test to ensure that the timer works."""
    with Timer() as simple_timer:
        sleep(0.1)
    assert (simple_timer.duration or timedelta(0)) >= timedelta(seconds=0.1)


def test_message_timer(capsys):
    """Simple test that the timer with a message works as intended."""
    with Timer(message="Timing test"):
        sleep(0.1)
    assert "Timing test : 0:00:00.1" in capsys.readouterr().out


def test_frozen_timer():
    """Test that Timer.to_frozen_timer() works as expected."""
    with Timer(message="Build snowman") as msg_timer:
        sleep(0.1)
    frozen_timer = msg_timer.to_frozen_timer()
    assert isinstance(frozen_timer, FrozenTimer)
    assert (frozen_timer.duration or timedelta(0)) >= timedelta(seconds=0.1)
    assert frozen_timer.message == "Build snowman"
    assert frozen_timer.start_time == msg_timer.start_time
    assert frozen_timer.end_time == msg_timer.end_time
