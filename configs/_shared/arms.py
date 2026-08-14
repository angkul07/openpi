"""Builders for the training arms this project actually runs.

Every arm used to be ~50 lines of `TrainConfig(...)` in which the same fifteen values
were retyped. That is not just noise -- three of the repetitions were live footguns:

  * `model=Pi0Config(...)` and `freeze_filter=Pi0Config(...).get_freeze_filter()` were
    two SEPARATE instances built from hand-copied arguments. Change the variant on one
    and forget the other and you silently train a different parameter set than you
    think. Here there is one object, used for both. (Safe: `get_freeze_filter()` reads
    only `paligemma_variant` and `action_expert_variant`, so nothing else can drift.)

  * `decay_steps` had to be retyped to equal `num_train_steps`. Now it comes from
    `Schedule`, which derives it -- see `configs/_shared/schedule.py`.

  * Augmentation is TWO independent stacks -- the data-side `ImageAugmentConfig` and
    openpi's built-in model-side one -- and disabling one leaves the other running.
    They are now one `augment=` flag. If you genuinely need them asymmetric, build the
    `TrainConfig` by hand and say why in a comment.

What these builders will NOT let you do, because each has cost real time:

  * a mixture whose per-source draws do not sum to `batch_size` (the sampler asserts
    this far later, on the box);
  * a mixture without its own `asset_id` -- it would fall back to `repo_id` and two
    different mixtures over the same primary dataset would share, and silently
    corrupt, one set of norm stats.
"""

from __future__ import annotations

from collections.abc import Sequence

from configs._shared.schedule import Schedule
import openpi.models.gemma as _gemma
import openpi.models.gemma_fast as _gemma_fast
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.training.augment as _augment
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders

PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
PI0_FAST_BASE = "gs://openpi-assets/checkpoints/pi0_fast_base/params"


def _validate_mixture(name: str, sources: Sequence[_config.MixtureSource], batch_size: int, asset_id: str) -> None:
    if not sources:
        raise ValueError(
            f"{name}: `sources` is empty. Even a single-source run goes through the mixture path "
            "(create_torch_dataset() hardcodes root=None off it, and run_yam.sh stage [0] asserts "
            "data.mixture is non-empty)."
        )
    total = sum(s.samples_per_batch for s in sources)
    if total != batch_size:
        breakdown = ", ".join(f"{s.repo_id}={s.samples_per_batch}" for s in sources)
        raise ValueError(
            f"{name}: per-source draws sum to {total}, but batch_size is {batch_size} ({breakdown}). "
            "The draw is a hard per-batch count, so these must be equal."
        )
    if not asset_id:
        raise ValueError(
            f"{name}: a mixture needs its own `asset_id`. Without one it falls back to `repo_id`, "
            "and two mixtures over the same primary dataset would share one set of norm stats."
        )


def pi05_arm(
    name: str,
    *,
    sources: Sequence[_config.MixtureSource],
    asset_id: str,
    schedule: Schedule,
    repo_id: str | None = None,
    batch_size: int = 64,
    num_workers: int = 16,
    save_interval: int = 1_000,
    max_to_keep: int = 4,
    keep_period: int | None = None,
    augment: bool = True,
    action_dim: int = 32,
    action_horizon: int = 50,
    max_token_len: int = 200,
    paligemma_variant: _gemma.Variant = "gemma_2b_lora",
    weight_loader_path: str = PI05_BASE,
    ema_decay: float | None = None,
) -> _config.TrainConfig:
    """A pi0.5 (flow-matching) arm over a fixed-ratio mixture.

    The defaults are this project's standing recipe, and the non-obvious ones are
    non-obvious for a reason:

      action_dim=32 is MANDATORY, not a tuning choice. `pi05_base` ships
        `action_in_proj` as Linear(32, 1024); `weight_loaders._merge_params()` matches
        on key NAMES only and never compares shapes, so 14 does not raise at load --
        it dies later inside jit, pointing nowhere near the cause. Nothing downstream
        cares: `YamOutputs` slices [..., :14] and `PadStatesAndActions` zero-pads.

      max_token_len=200 because pi0.5 has no FAST action tokens -- actions reach the
        flow expert as continuous conditioning and never enter the token stream. The
        prompt runs ~90 tokens in practice. (pi0-FAST needs 300.)

      `discrete_state_input` is deliberately never set. `Pi0Config.__post_init__`
        defaults it to `pi05`, i.e. True. Do NOT copy `discrete_state_input=False`
        from `pi05_libero`: with pi05=True the state token is already absent from the
        suffix, so False means the model gets NO proprioception at all.

      num_workers=16 is measured, not guessed. A 200-step A/B on 2x A100-80GB gave
        16 -> 3.528 s/step at 95.0% GPU util against 32 -> 3.545 s/step at 94.7%:
        doubling was 0.5% SLOWER, below the 0.7-0.9% window spread. Three-camera
        decode keeps both GPUs fed; these runs are compute-bound, not loader-bound.
    """
    _validate_mixture(name, sources, batch_size, asset_id)

    # ONE model config, used for the model and for the freeze filter. See module docstring.
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        paligemma_variant=paligemma_variant,
        image_augmentation=augment,
    )

    return _config.TrainConfig(
        name=name,
        model=model,
        data=_config.LeRobotYamMixtureDataConfig(
            # The "primary" source; norm stats live under assets/<config>/<asset_id>.
            repo_id=repo_id or sources[0].repo_id,
            assets=_config.AssetsConfig(asset_id=asset_id),
            # Required: the repack transform forwards "prompt", and PaligemmaTokenizer
            # raises "Prompt is required" without it.
            base_config=_config.DataConfig(prompt_from_task=True),
            # The data-side stack. Moves together with the model-side one above.
            augment_config=_augment.ImageAugmentConfig() if augment else None,
            sources=tuple(sources),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(weight_loader_path),
        num_train_steps=schedule.num_train_steps,
        lr_schedule=schedule.lr_schedule(),
        batch_size=batch_size,
        num_workers=num_workers,
        save_interval=save_interval,
        max_to_keep=max_to_keep,
        keep_period=keep_period,
        freeze_filter=model.get_freeze_filter(),
        ema_decay=ema_decay,
    )


def pi0_fast_arm(
    name: str,
    *,
    sources: Sequence[_config.MixtureSource],
    asset_id: str,
    schedule: Schedule,
    repo_id: str | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    save_interval: int = 1_000,
    max_to_keep: int = 4,
    keep_period: int | None = None,
    augment: bool = True,
    action_dim: int = 14,
    action_horizon: int = 50,
    max_token_len: int = 300,
    paligemma_variant: _gemma_fast.Variant = "gemma_2b_lora",
    weight_loader_path: str = PI0_FAST_BASE,
    ema_decay: float | None = None,
) -> _config.TrainConfig:
    """A pi0-FAST arm over a fixed-ratio mixture.

    NOTE `augment` here controls the DATA-side stack only. Unlike `Pi0Config`,
    `Pi0FASTConfig` has no `image_augmentation` flag, so openpi's built-in model-side
    augmentation is always on for this family. `augment=False` on a FAST arm therefore
    does NOT give you an augmentation-free run -- add the flag to `Pi0FASTConfig` first
    if you need one.

    Otherwise the same shape as `pi05_arm`: `action_dim` is the real 14 here (FAST
    has no padded action projection to satisfy), `max_token_len` is 300 to hold the FAST
    action tokens, and the loader wants fewer workers because the model step is cheaper.
    """
    _validate_mixture(name, sources, batch_size, asset_id)

    model = pi0_fast.Pi0FASTConfig(
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        paligemma_variant=paligemma_variant,
    )

    return _config.TrainConfig(
        name=name,
        model=model,
        data=_config.LeRobotYamMixtureDataConfig(
            repo_id=repo_id or sources[0].repo_id,
            assets=_config.AssetsConfig(asset_id=asset_id),
            base_config=_config.DataConfig(prompt_from_task=True),
            augment_config=_augment.ImageAugmentConfig() if augment else None,
            sources=tuple(sources),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(weight_loader_path),
        num_train_steps=schedule.num_train_steps,
        lr_schedule=schedule.lr_schedule(),
        batch_size=batch_size,
        num_workers=num_workers,
        save_interval=save_interval,
        max_to_keep=max_to_keep,
        keep_period=keep_period,
        freeze_filter=model.get_freeze_filter(),
        ema_decay=ema_decay,
    )
