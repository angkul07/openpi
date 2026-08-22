import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.early_stop as _early_stop
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        # Carry the un-reduced loss out as aux so the caller can report its structure
        # without a second forward pass.
        return jnp.mean(chunked_loss), chunked_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, chunked_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }

    # Flow-matching models (pi0 / pi0.5) return loss per action-chunk step, shape
    # (*b, action_horizon): the mean squared error between the predicted velocity v_t
    # and the target u_t = noise - actions. That IS the flow loss, so `flow_loss`
    # equals `loss` -- it is logged under its own name so the objective is
    # unambiguous next to the FAST arms, whose `loss` is token cross-entropy.
    # pi0-FAST returns shape (*b,) and is skipped by the ndim guard (shapes are
    # static under jit, so this branch is resolved at trace time).
    if chunked_loss.ndim > 1:
        # Mean over every batch axis, leaving one value per step in the chunk.
        per_chunk_step = jnp.mean(chunked_loss, axis=tuple(range(chunked_loss.ndim - 1)))
        info["flow_loss"] = loss
        # Later actions in a chunk are predicted from the same observation and are
        # consistently harder; a widening gap means chunk quality is degrading.
        info["flow_loss_chunk_first"] = per_chunk_step[0]
        info["flow_loss_chunk_last"] = per_chunk_step[-1]

    return new_state, info


def _write_per_step_metrics(metrics_file, stacked_infos: dict, *, last_step: int, count: int) -> None:
    """Append one line per training step from an already host-resident info batch.

    `stacked_infos` holds arrays of length `count` covering the steps ending at
    `last_step` inclusive.
    """
    first_step = last_step - count + 1
    for i in range(count):
        values = " ".join(f"{k}={float(np.asarray(v)[i]):.6f}" for k, v in stacked_infos.items())
        metrics_file.write(f"step={first_step + i} {values}\n")
    metrics_file.flush()


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        max_to_keep=config.max_to_keep,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # On resume, seek the data stream forward to where the interrupted run stopped
    # instead of replaying it from the start. The checkpoint holds the completed step,
    # and batch index N feeds train step N, so the next batch to draw is latest_step + 1.
    # (Only the mixture sampler can seek; other loaders log that they ignored this.)
    resume_batch = int(checkpoint_manager.latest_step()) + 1 if resuming else 0

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        start_batch=resume_batch,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    if resuming:
        logging.info(f"Resuming at step {start_step} (data stream sought to batch {resume_batch})")
        if start_step != resume_batch:
            # Not fatal -- a mismatch only means the stream is offset by a few batches --
            # but it means the checkpoint's step and the seek disagree, so say so.
            logging.warning(
                f"Resume step {start_step} != data stream seek {resume_batch}; "
                "the data stream is offset by that many batches."
            )
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # Per-step metrics, one line per training step. stdout/wandb still report the
    # `log_interval` average; this file keeps the un-averaged history so loss and
    # grad_norm spikes between log points are not smoothed away. It is written from
    # the same device_get that already happens at each log point, so there is no
    # extra host sync and no per-step pipeline stall.
    metrics_path = config.checkpoint_dir / "train_metrics.log"
    metrics_file = metrics_path.open("a")
    metrics_file.write(f"# {config.name}/{config.exp_name} start_step={start_step} steps={config.num_train_steps}\n")
    logging.info(f"Per-step metrics log: {metrics_path}")

    early_stop = _early_stop.EarlyStopTracker(config.early_stop, start_step=start_step) if config.early_stop else None
    if early_stop is not None:
        logging.info(f"Early stopping enabled: {config.early_stop}")

    infos = []
    # The step the loop actually finished on, which is `num_train_steps - 1` only when no
    # early stop fires. The tail metric write and the final checkpoint both key off it.
    last_step = start_step
    stop_reason: str | None = None
    for step in pbar:
        last_step = step
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = jax.device_get(common_utils.stack_forest(infos))
            reduced_info = jax.tree.map(np.mean, stacked_infos)
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            _write_per_step_metrics(metrics_file, stacked_infos, last_step=step, count=len(infos))
            infos = []
            if early_stop is not None:
                stop_reason = early_stop.should_stop(reduced_info, step)
        batch = next(data_iter)

        # Always checkpoint the step an early stop lands on, whatever the interval says:
        # it is the last state this run will ever have, and losing it would mean
        # re-running to reach it.
        if (
            (step % config.save_interval == 0 and step > start_step)
            or step == config.num_train_steps - 1
            or stop_reason is not None
        ):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

        if stop_reason is not None:
            break

    if stop_reason is not None:
        best_value, best_step = early_stop.best
        # `decay_lr` is a CosineDecaySchedule field, not part of the LRScheduleConfig
        # protocol, so read it defensively -- an error path that itself raises is worse
        # than a slightly vaguer message.
        floor = getattr(config.lr_schedule, "decay_lr", None)
        annealed_to = f"{floor:g}" if floor is not None else "its floor"
        message = (
            f"Early stop at step {last_step} of {config.num_train_steps}: {stop_reason}. "
            f"NOTE this checkpoint was NOT annealed -- the schedule was set to reach "
            f"{annealed_to} at step {config.num_train_steps}, so it stops mid-decay. Pick "
            f"what to ship by holdout score over the kept checkpoints, not by this step "
            f"being the last one."
        )
        pbar.write(message)
        logging.info(message)
        metrics_file.write(
            f"# early_stop step={last_step} metric={config.early_stop.metric} best={best_value:.6f}@{best_step}\n"
        )
        wandb.log({"early_stop_step": last_step, "early_stop_best": best_value}, step=last_step)

    if infos:  # steps since the last log point would otherwise never be written
        _write_per_step_metrics(
            metrics_file,
            jax.device_get(common_utils.stack_forest(infos)),
            last_step=last_step,
            count=len(infos),
        )
    metrics_file.close()
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
