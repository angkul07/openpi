"""Tests for `openpi.training.early_stop`.

The tracker is the only new logic in the early-stopping path -- `train.py` just calls
`should_stop` at its existing log points -- so this is where it gets pinned down.
"""

import pytest

from openpi.training import early_stop as _early_stop


def _tracker(**kwargs) -> _early_stop.EarlyStopTracker:
    spec = _early_stop.EarlyStop(patience_steps=300, min_rel_delta=0.01, **kwargs)
    return _early_stop.EarlyStopTracker(spec, start_step=0)


def _run(tracker, values, *, log_interval=100):
    """Feed one value per log point; return the step it stopped at, or None."""
    for i, value in enumerate(values):
        step = i * log_interval
        if tracker.should_stop({"loss": value}, step) is not None:
            return step
    return None


def test_a_steadily_improving_run_never_stops():
    # The realistic case under cosine decay: the loss keeps creeping down, so patience
    # never runs out and the run goes its full length.
    assert _run(_tracker(), [1.0 * 0.9**i for i in range(50)]) is None


def test_a_flat_run_stops_once_patience_elapses():
    # Best is set at step 0; patience is 300, so steps 100 and 200 are still inside it
    # and 300 is the first that can trip.
    assert _run(_tracker(), [1.0] * 10) == 300


def test_an_improvement_resets_the_patience_window():
    # Flat for 200 steps (inside patience), improves at 300, then flat again. The reset
    # is what moves the stop from 300 to 600; without it the second flat stretch would
    # be counted from step 0.
    values = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    assert _run(_tracker(), values) == 600


def test_an_improvement_smaller_than_min_rel_delta_does_not_reset_patience():
    # 0.1% better against a 1% threshold is noise, not progress.
    assert _run(_tracker(), [1.0, 0.999, 0.998, 0.997]) == 300


def test_min_steps_holds_the_stop_off_even_when_the_metric_is_flat():
    # Same flat run as above, but nothing may stop before step 1000.
    assert _run(_tracker(min_steps=1_000), [1.0] * 20) == 1_000


def test_the_first_value_always_counts_as_an_improvement():
    tracker = _tracker()
    assert tracker.should_stop({"loss": 7.0}, 0) is None
    assert tracker.best == (7.0, 0)


def test_a_missing_metric_raises_rather_than_disabling_the_stop():
    # A run that quietly ignores its own stopping rule is worse than one that fails at
    # the first log point.
    tracker = _early_stop.EarlyStopTracker(_early_stop.EarlyStop(metric="flow_loss"), start_step=0)
    with pytest.raises(KeyError, match="flow_loss"):
        tracker.should_stop({"loss": 1.0}, 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"metric": ""}, "non-empty info key"),
        ({"patience_steps": 0}, "patience_steps must be positive"),
        ({"patience_steps": -1}, "patience_steps must be positive"),
        ({"min_rel_delta": 1.0}, r"min_rel_delta must be in \[0, 1\)"),
        ({"min_rel_delta": -0.1}, r"min_rel_delta must be in \[0, 1\)"),
        ({"min_steps": -1}, "min_steps must be >= 0"),
    ],
)
def test_invalid_specs_raise_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _early_stop.EarlyStop(**kwargs)


def test_the_stop_reason_names_the_numbers_it_stopped_on():
    tracker = _tracker()
    tracker.should_stop({"loss": 0.5}, 0)
    message = tracker.should_stop({"loss": 0.5}, 300)
    assert message is not None
    assert "0.500000" in message
    assert "step 0" in message
    assert "300" in message
