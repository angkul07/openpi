"""Early stopping for the training loop.

openpi has none of this upstream -- `scripts/train.py` is a plain
`for step in range(num_train_steps)` -- so a run that flattens at 30k on a 70k schedule
burns the other 40k. This module is the whole mechanism: a spec on `TrainConfig` and a
tracker the loop consults at its existing log points. It touches nothing else. A config
that leaves `early_stop=None` runs exactly as openpi always has.

It lives here rather than in `config.py` or inline in `train.py` so that both halves can
be tested without pulling in jax.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class EarlyStop:
    """Stop training when a logged metric stops improving.

    openpi has none of this upstream -- `scripts/train.py` is a plain
    `for step in range(num_train_steps)` -- so a run that flattens at 30k on a 70k
    schedule burns the other 40k. This adds the check and nothing else: it does not
    touch the optimizer, the schedule or the data stream, and a config that leaves
    `early_stop=None` runs exactly as before.

    THE COSINE SCHEDULE MAKES THIS FIRE LESS OFTEN THAN YOU EXPECT, and that is worth
    understanding before setting `patience_steps`. `Schedule.lr_schedule()` pins
    `decay_steps` to `num_train_steps`, so the LR is still falling right up to the last
    step, and a falling LR drags the training loss down with it. A run under cosine
    decay therefore rarely plateaus in the way a constant-LR run does. Treat this as a
    guard against a genuine stall or divergence -- and as a way to stop paying for GPU
    once the curve is flat -- rather than as a convergence detector.

    AND WHATEVER IT STOPS ON IS AN UN-ANNEALED CHECKPOINT. Stopping a cosine at step k
    of N leaves the model at whatever LR step k happens to sit at, which is not the same
    model a k-step run would have produced (that one would have annealed to the floor).
    So an early stop is a decision to stop spending, not a decision about which
    checkpoint to ship. Keep `keep_period` set and pick the checkpoint by holdout score.

    THE METRIC IS TRAINING LOSS, WITH EVERYTHING THAT IMPLIES. It is read from the same
    `log_interval` average that already goes to stdout and wandb, so no extra host sync
    and no per-step stall. It is also not a validation metric -- openpi has no
    in-training validation -- so it says the optimiser has stopped moving, never that
    the model has stopped overfitting. Those come apart exactly where it matters.
    """

    # Which key of the per-step info dict to watch. Missing keys raise rather than
    # silently disabling the stop -- `flow_loss_chunk_last` is the interesting
    # alternative on pi0.5, since delta actions make the t=0 term nearly free and the
    # aggregate loss is dominated by it.
    metric: str = "loss"

    # Stop when this many steps pass with no improvement. Counted from the step of the
    # best value seen, not from the last log point.
    patience_steps: int = 3_000

    # How much better a value has to be to count as an improvement, as a FRACTION of the
    # best so far. Relative rather than absolute because loss scale is arbitrary --
    # per-arm norm stats mean the same model quality reads as a different number on a
    # different mixture, and an absolute epsilon tuned on one run is meaningless on the
    # next. 1e-3 = "at least 0.1% better at some point in the patience window".
    #
    # Assumes a positive metric, which every loss here is.
    min_rel_delta: float = 1e-3

    # Never stop before this step, whatever the metric does. The first few hundred steps
    # are warmup, where the loss moves for reasons that have nothing to do with
    # convergence, and a patience window that opens inside them can trip on the warmup
    # ramp itself. Set this past `warmup_steps + patience_steps`.
    min_steps: int = 0

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("early_stop.metric must be a non-empty info key.")
        if self.patience_steps <= 0:
            raise ValueError(f"early_stop.patience_steps must be positive, got {self.patience_steps}")
        if not 0.0 <= self.min_rel_delta < 1.0:
            raise ValueError(f"early_stop.min_rel_delta must be in [0, 1), got {self.min_rel_delta}")
        if self.min_steps < 0:
            raise ValueError(f"early_stop.min_steps must be >= 0, got {self.min_steps}")


class EarlyStopTracker:
    """Plateau detection over the `log_interval` averages, per `config.early_stop`.

    Deliberately reads the SAME reduced info dict that already goes to stdout and wandb,
    so it costs no extra host sync and cannot disagree with what the logs show. It holds
    no model state: `should_stop` is a pure function of the metric history.

    Read `EarlyStop` before tuning any of it -- in particular why a cosine
    schedule makes a training-loss plateau rare, and why the step this stops at is a
    spending decision rather than a checkpoint-selection one.
    """

    def __init__(self, spec: EarlyStop, *, start_step: int):
        self._spec = spec
        self._best_value = float("inf")
        self._best_step = start_step

    @property
    def best(self) -> tuple[float, int]:
        return self._best_value, self._best_step

    def should_stop(self, reduced_info: dict[str, Any], step: int) -> str | None:
        """Update from one log point; return a reason string when training should end.

        A missing metric key raises rather than silently disabling the stop -- a run
        that quietly ignores its own stopping rule is worse than one that fails at the
        first log point.
        """
        spec = self._spec
        if spec.metric not in reduced_info:
            raise KeyError(
                f"early_stop.metric={spec.metric!r} is not in the training info "
                f"{sorted(reduced_info)}. Note the pi0.5-only keys (flow_loss, "
                "flow_loss_chunk_first/last) are absent on pi0-FAST arms."
            )
        value = float(reduced_info[spec.metric])
        # Relative threshold: `inf * (1 - d)` is `inf`, so the first value always wins
        # and no special case is needed for the initial state.
        if value < self._best_value * (1.0 - spec.min_rel_delta):
            self._best_value, self._best_step = value, step
            return None
        if step < spec.min_steps:
            return None
        stalled = step - self._best_step
        if stalled < spec.patience_steps:
            return None
        return (
            f"{spec.metric} has not improved by {spec.min_rel_delta:.3%} in {stalled} steps "
            f"(best {self._best_value:.6f} at step {self._best_step}, now {value:.6f}); "
            f"patience is {spec.patience_steps}"
        )
