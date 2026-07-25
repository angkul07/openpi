import bisect
from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class MixtureDataset(Dataset):
    """Concatenation of several datasets that keeps track of per-source boundaries.

    Indices are global: source `s` owns `[offsets[s], offsets[s] + sizes[s])`.
    `StratifiedBatchSampler` uses `sizes`/`offsets` to build fixed-ratio batches.
    """

    def __init__(self, datasets: Sequence[Dataset], repo_ids: Sequence[str]):
        if not datasets:
            raise ValueError("MixtureDataset requires at least one source dataset.")
        self._datasets = list(datasets)
        self.repo_ids = list(repo_ids)
        self.sizes = [len(d) for d in self._datasets]
        # Cumulative boundaries; `_cum[s]` is the global start index of source s.
        self._cum = [0]
        for size in self.sizes:
            self._cum.append(self._cum[-1] + size)
        self.offsets = self._cum[:-1]

    def __getitem__(self, index: SupportsIndex):
        idx = int(index.__index__())
        if not 0 <= idx < self._cum[-1]:
            raise IndexError(f"index {idx} out of range for mixture of size {self._cum[-1]}")
        source = bisect.bisect_right(self._cum, idx) - 1
        return self._datasets[source][idx - self._cum[source]]

    def __len__(self) -> int:
        return self._cum[-1]


class _NoVideoLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobotDataset that returns dummy frames instead of decoding video.

    Only used by `compute_norm_stats.py`: normalization statistics are computed over
    `state` / `actions` (parquet columns) and never touch pixels, but the standard
    `__getitem__` decodes every camera anyway, which dominates runtime by ~2 orders
    of magnitude on a 1.6M-frame mixture.
    """

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int):
        return {key: torch.zeros(3, 2, 2, dtype=torch.uint8) for key in query_timestamps}


def select_holdout_episodes(total_episodes: int, fraction: float, seed: int) -> list[int]:
    """Deterministically pick validation episode indices.

    Matches fidelity-sdk's `HoldoutSpec.select` math (n = max(1, round(N * fraction)),
    `default_rng(seed).choice` without replacement) so the episodes withheld here are
    the same ones the offline eval scores on.
    """
    if fraction <= 0.0:
        return []
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {fraction}")
    pool = list(range(total_episodes))
    num = max(1, round(len(pool) * fraction))
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(pool, size=num, replace=False))


def _remap_episode_data_index(dataset: lerobot_dataset.LeRobotDataset, episodes: Sequence[int]) -> None:
    """Re-key `episode_data_index` by original episode index.

    Upstream bug (lerobot @0cf8648): `get_episode_data_index` builds `from`/`to` as
    dense arrays *positional* over the kept episodes, but `__getitem__` looks them up
    as `episode_data_index["from"][ep_idx]` with `ep_idx` read from the frame itself --
    i.e. the *original* episode index. With `episodes=None` the two coincide and all is
    well, which is why stock openpi never trips over it. As soon as episodes are held
    out they diverge: keeping 500..999 makes ep_idx 713 index into a length-500 array
    (IndexError), and keeping e.g. 3..1644 silently returns *another episode's*
    boundaries, so action chunks would run across episode edges with wrong padding.

    Called after `__init__`, which is deliberate: `check_timestamps_sync` consumes the
    positional layout (`to[:-1]` = last frame of each episode in dataset order) and has
    already run by then. Only the read path uses it afterwards.
    """
    kept = list(episodes)
    lengths = [dataset.meta.episodes[ep]["length"] for ep in kept]
    starts = np.cumsum([0, *lengths[:-1]])

    size = max(kept) + 1
    from_ = np.zeros(size, dtype=np.int64)
    to_ = np.zeros(size, dtype=np.int64)
    for pos, ep in enumerate(kept):
        from_[ep] = starts[pos]
        to_[ep] = starts[pos] + lengths[pos]

    dataset.episode_data_index = {"from": torch.from_numpy(from_), "to": torch.from_numpy(to_)}


def _create_lerobot_dataset(
    repo_id: str,
    *,
    root: str | None,
    action_horizon: int,
    data_config: _config.DataConfig,
    exclude_episodes: Sequence[int] = (),
    holdout_fraction: float = 0.0,
    holdout_seed: int = 0,
    skip_videos: bool = False,
) -> Dataset:
    """Create one LeRobot dataset, optionally excluding a validation episode split."""
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    total_episodes = dataset_meta.total_episodes

    held_out = {int(ep) for ep in exclude_episodes}
    out_of_range = sorted(ep for ep in held_out if not 0 <= ep < total_episodes)
    if out_of_range:
        raise ValueError(f"[{repo_id}] exclude_episodes contains indices outside [0, {total_episodes}): {out_of_range}")
    if holdout_fraction > 0.0:
        held_out |= set(select_holdout_episodes(total_episodes, holdout_fraction, holdout_seed))

    episodes = None
    if held_out:
        episodes = [ep for ep in range(total_episodes) if ep not in held_out]
        if not episodes:
            raise ValueError(f"[{repo_id}] every episode is held out; nothing left to train on")
        logging.info(
            f"[{repo_id}] holding out {len(held_out)}/{total_episodes} episodes from training; "
            f"first few: {sorted(held_out)[:10]}"
        )

    dataset_cls = _NoVideoLeRobotDataset if skip_videos else lerobot_dataset.LeRobotDataset
    dataset = dataset_cls(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if episodes is not None:
        _remap_episode_data_index(dataset, episodes)

    if data_config.prompt_from_task:
        # Task strings are per-dataset, so this must be applied per source.
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_videos: bool = False,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if data_config.mixture:
        datasets = [
            _create_lerobot_dataset(
                source.repo_id,
                root=source.root,
                action_horizon=action_horizon,
                data_config=data_config,
                exclude_episodes=source.exclude_episodes,
                holdout_fraction=source.holdout_fraction,
                holdout_seed=source.holdout_seed,
                skip_videos=skip_videos,
            )
            for source in data_config.mixture
        ]
        mixture = MixtureDataset(datasets, [s.repo_id for s in data_config.mixture])
        for source, size in zip(data_config.mixture, mixture.sizes, strict=True):
            logging.info(
                f"mixture source {source.repo_id}: {size} train frames, {source.samples_per_batch} samples/batch"
            )
        return mixture

    return _create_lerobot_dataset(
        repo_id,
        root=None,
        action_horizon=action_horizon,
        data_config=data_config,
        skip_videos=skip_videos,
    )


class StratifiedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yields batches with an exact, fixed number of samples from each source.

    Every batch contains `counts[s]` indices from source `s`, so the gradient share
    of a source is `counts[s] / sum(counts)` exactly -- no loss weighting involved.

    Each source has its own index stream: a random permutation of the source that is
    consumed in order and reshuffled (with a new permutation) when it wraps around.
    Because sources have different sizes and different per-batch counts, they wrap at
    different rates, which is the whole point (teleop is visited ~2x per ego visit at
    32/32 over a 540k/1080k split).

    The stream position is a monotonically increasing counter that survives dataloader
    epoch restarts, so torch re-iterating the sampler does not replay the same batches.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        counts: Sequence[int],
        offsets: Sequence[int],
        *,
        seed: int = 0,
        shuffle: bool = True,
        batches_per_epoch: int | None = None,
        start_batch: int = 0,
    ):
        if len(sizes) != len(counts) or len(sizes) != len(offsets):
            raise ValueError("sizes, counts and offsets must have the same length")
        for size, count, repo_index in zip(sizes, counts, range(len(sizes)), strict=True):
            if count <= 0:
                raise ValueError(f"source {repo_index} has non-positive samples_per_batch={count}")
            if size < count:
                raise ValueError(f"source {repo_index} has {size} frames, fewer than its per-batch count {count}")

        self._sizes = list(sizes)
        self._counts = list(counts)
        self._offsets = list(offsets)
        self._seed = seed
        self._shuffle = shuffle
        # One "epoch" for torch's iterator bookkeeping: enough batches for the source
        # that needs the most of them to be seen once. Purely cosmetic -- the streams
        # wrap independently and carry over across restarts.
        self._batches_per_epoch = batches_per_epoch or max(
            1, *(size // count for size, count in zip(sizes, counts, strict=True))
        )
        # Each source's stream position is a pure function of the batch counter, so a
        # resume can seek straight to where the interrupted run left off instead of
        # replaying the start of the stream. (The ratio is exact either way; this keeps
        # the *frames* from being re-drawn.)
        if start_batch < 0:
            raise ValueError(f"start_batch must be non-negative, got {start_batch}")
        self._positions = [start_batch * count for count in self._counts]
        self._batch_index = start_batch
        # Per source, the (epoch, permutation) currently being consumed. Only one epoch
        # per source is ever live, so this holds at most `len(sizes)` permutations.
        self._perm_cache: list[tuple[int, np.ndarray] | None] = [None] * len(sizes)

    @property
    def batch_size(self) -> int:
        return sum(self._counts)

    def _permutation(self, source: int, epoch: int) -> np.ndarray:
        cached = self._perm_cache[source]
        if cached is None or cached[0] != epoch:
            size = self._sizes[source]
            if self._shuffle:
                perm = np.random.default_rng([self._seed, source, epoch]).permutation(size)
            else:
                perm = np.arange(size)
            self._perm_cache[source] = (epoch, perm)
            return perm
        return cached[1]

    def _next_index(self, source: int) -> int:
        size = self._sizes[source]
        position = self._positions[source]
        self._positions[source] = position + 1
        perm = self._permutation(source, position // size)
        return self._offsets[source] + int(perm[position % size])

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self._batches_per_epoch):
            batch = [self._next_index(s) for s, count in enumerate(self._counts) for _ in range(count)]
            if self._shuffle:
                # Shuffle within the batch so source order inside a batch is not fixed.
                order = np.random.default_rng([self._seed, 0xB47C4, self._batch_index]).permutation(len(batch))
                batch = [batch[i] for i in order]
            self._batch_index += 1
            yield batch

    def __len__(self) -> int:
        return self._batches_per_epoch


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    skip_train_only_transforms: bool = False,
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    train_only = () if skip_train_only_transforms else data_config.train_only_transforms.inputs

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Training-only (e.g. image augmentation): before normalization, and never
            # part of the inference chain.
            *train_only,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    start_batch: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        start_batch: Batch index to start the data stream at, so a resumed run continues
            the stream instead of replaying it from the beginning. Only the mixture
            sampler can seek; other loaders ignore it (and log that they did).
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if start_batch and not data_config.mixture:
        logging.warning(
            f"start_batch={start_batch} ignored: only the mixture sampler can seek. "
            "The data stream restarts from the beginning (stock openpi behaviour)."
        )

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        start_batch=start_batch,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    start_batch: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    mixture_sizes = getattr(dataset, "sizes", None) if isinstance(dataset, MixtureDataset) else None
    mixture_offsets = dataset.offsets if isinstance(dataset, MixtureDataset) else None
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")

    batch_sampler = None
    if mixture_sizes is not None:
        assert mixture_offsets is not None
        if sampler is not None:
            raise NotImplementedError("Mixture sampling is not supported with PyTorch DDP samplers.")
        batch_sampler = make_stratified_batch_sampler(
            data_config,
            mixture_sizes,
            mixture_offsets,
            local_batch_size,
            seed=seed,
            shuffle=shuffle,
            start_batch=start_batch,
        )

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and batch_sampler is None and shuffle),  # Don't shuffle if using a sampler
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def make_stratified_batch_sampler(
    data_config: _config.DataConfig,
    sizes: Sequence[int],
    offsets: Sequence[int],
    batch_size: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    start_batch: int = 0,
) -> StratifiedBatchSampler:
    """Build the fixed-count mixture sampler and check it adds up to the batch size."""
    counts = [source.samples_per_batch for source in data_config.mixture]
    if sum(counts) != batch_size:
        raise ValueError(
            f"Mixture samples_per_batch {counts} sums to {sum(counts)}, which does not match the "
            f"batch size {batch_size}. Fix the config so every batch is exactly full."
        )
    for source, count, size in zip(data_config.mixture, counts, sizes, strict=True):
        share = count / batch_size
        logging.info(
            f"mixture sampler: {source.repo_id} -> {count}/{batch_size} per batch "
            f"({share:.1%} gradient share), {size} frames, "
            f"{count / batch_size / (size / sum(sizes)):.2f}x its storage share"
        )
    if start_batch:
        logging.info(f"mixture sampler: seeking to batch {start_batch} (resume)")
    return StratifiedBatchSampler(sizes, counts, offsets, seed=seed, shuffle=shuffle, start_batch=start_batch)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler[list[int]] | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            batch_sampler: If provided, yields whole batches of indices and takes over
                batching entirely (used for fixed-ratio mixture sampling). Mutually
                exclusive with `shuffle`/`sampler`/`local_batch_size`.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        # A batch_sampler already defines batching, so torch forbids passing
        # batch_size/shuffle/sampler/drop_last alongside it.
        batching_kwargs: dict[str, typing.Any] = (
            {"batch_sampler": batch_sampler}
            if batch_sampler is not None
            else {
                "batch_size": local_batch_size,
                "shuffle": (sampler is None and shuffle),  # Don't shuffle if using sampler
                "sampler": sampler,
                "drop_last": True,
            }
        )
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=generator,
            **batching_kwargs,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
