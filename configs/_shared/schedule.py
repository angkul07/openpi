"""Step counts and LR schedules, computed rather than commented.

Two recurring bugs live here, and both are now structural rather than a convention:

  1. `decay_steps` drifting from `num_train_steps`. openpi's `CosineDecaySchedule`
     defaults `decay_steps` to 30k independently of how long you train, so a 7k-step
     run silently never finishes decaying. `Schedule.lr_schedule()` derives
     `decay_steps` from `num_train_steps`; they cannot disagree.

  2. Epoch arithmetic done by hand in a comment. Every arm used to carry a block like
     "at other batch sizes one epoch is: 32 -> 22,828 | 48 -> 15,219 | 96 -> 7,610",
     which is correct only until someone edits `batch_size` and not the table.
     `Schedule.for_epochs()` computes it, and `epochs_at()` reports it back.

Epoch definitions used throughout, and why they differ:

  * SINGLE SOURCE: `epochs = steps * batch_size / frames`. `len(LeRobotDataset)` is
    the frame count -- `delta_timestamps` clamps at episode ends rather than dropping
    samples -- so every step visits `batch_size` frames.

  * MIXTURE: each source has its own epoch count, because the draw is a fixed count
    per batch, not a probability: `epochs_i = steps * samples_per_batch_i / frames_i`.
    A mixture therefore has no single epoch number, and asking for one is usually the
    first sign of a modelling mistake.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

import openpi.training.optimizer as _optimizer

# The peak/floor this project has used on every YAM and Piper arm. Kept together so
# "unchanged from the previous run" is a fact about one constant, not four literals.
DEFAULT_PEAK_LR = 3.5e-5
DEFAULT_DECAY_LR = 3.5e-6
# Warmup has been ~2% of the run throughout (1000/50k, 500/23.6k, 350/17.7k).
DEFAULT_WARMUP_FRACTION = 0.02


def epochs_to_steps(frames: int, batch_size: int, epochs: float) -> int:
    """Steps needed for `epochs` passes over `frames` at `batch_size`."""
    if frames <= 0 or batch_size <= 0:
        raise ValueError(f"frames and batch_size must be positive, got {frames} and {batch_size}")
    return round(epochs * frames / batch_size)


def epochs_at(frames: int, samples_per_batch: int, steps: int) -> float:
    """How many passes over `frames` a source gets at `samples_per_batch` for `steps`."""
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    return steps * samples_per_batch / frames


@dataclasses.dataclass(frozen=True)
class Schedule:
    """A training length and the LR schedule that matches it.

    Construct with `of_steps` when the step count is the thing you are holding fixed
    (e.g. two arms of one experiment that must differ in mixing ratio alone), and with
    `for_epochs` when data exposure is the thing you are holding fixed.
    """

    num_train_steps: int
    warmup_steps: int
    peak_lr: float = DEFAULT_PEAK_LR
    decay_lr: float = DEFAULT_DECAY_LR

    def __post_init__(self) -> None:
        if self.num_train_steps <= 0:
            raise ValueError(f"num_train_steps must be positive, got {self.num_train_steps}")
        if not 0 <= self.warmup_steps < self.num_train_steps:
            raise ValueError(
                f"warmup_steps must be in [0, num_train_steps), got {self.warmup_steps} of {self.num_train_steps}"
            )

    @classmethod
    def of_steps(
        cls,
        steps: int,
        *,
        warmup_steps: int | None = None,
        warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
        peak_lr: float = DEFAULT_PEAK_LR,
        decay_lr: float = DEFAULT_DECAY_LR,
    ) -> Schedule:
        """A schedule of exactly `steps`. Warmup defaults to `warmup_fraction` of it."""
        if warmup_steps is None:
            warmup_steps = max(1, round(steps * warmup_fraction))
        return cls(num_train_steps=steps, warmup_steps=warmup_steps, peak_lr=peak_lr, decay_lr=decay_lr)

    @classmethod
    def for_epochs(
        cls,
        *,
        frames: int,
        batch_size: int,
        epochs: float,
        round_to: int = 1,
        warmup_steps: int | None = None,
        warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
        peak_lr: float = DEFAULT_PEAK_LR,
        decay_lr: float = DEFAULT_DECAY_LR,
    ) -> Schedule:
        """A schedule sized to `epochs` passes over a single source of `frames`.

        `round_to` rounds the step count UP to a multiple (e.g. 100) when you want a
        tidy number; it moves the realised epoch count slightly, so `describe()` reports
        what you actually get rather than what you asked for.
        """
        steps = epochs_to_steps(frames, batch_size, epochs)
        if round_to > 1:
            steps = math.ceil(steps / round_to) * round_to
        return cls.of_steps(
            steps,
            warmup_steps=warmup_steps,
            warmup_fraction=warmup_fraction,
            peak_lr=peak_lr,
            decay_lr=decay_lr,
        )

    def lr_schedule(self) -> _optimizer.CosineDecaySchedule:
        """The cosine schedule, with `decay_steps` pinned to `num_train_steps`."""
        return _optimizer.CosineDecaySchedule(
            warmup_steps=self.warmup_steps,
            peak_lr=self.peak_lr,
            decay_steps=self.num_train_steps,
            decay_lr=self.decay_lr,
        )

    def describe(self, sources: Sequence[tuple[str, int, int]] = ()) -> str:
        """Human-readable summary; this is what the hand-written epoch comments became.

        `sources` is a sequence of `(label, frames, samples_per_batch)`. Reporting only
        -- nothing here feeds control flow.
        """
        pct = 100 * self.warmup_steps / self.num_train_steps
        out = [
            f"{self.num_train_steps:,} steps, warmup {self.warmup_steps:,} ({pct:.1f}%), "
            f"cosine {self.peak_lr:g} -> {self.decay_lr:g}"
        ]
        for label, frames, per_batch in sources:
            epochs = epochs_at(frames, per_batch, self.num_train_steps)
            out.append(f"  {label}: {frames:,} frames, {per_batch}/batch -> {epochs:.2f} epochs")
        return "\n".join(out)
