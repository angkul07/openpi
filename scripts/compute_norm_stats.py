"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.fingerprint as _fingerprint
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    skip_videos: bool = False,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config, skip_videos=skip_videos
    )
    mixture_sizes = dataset.sizes if isinstance(dataset, _data_loader.MixtureDataset) else None
    mixture_offsets = dataset.offsets if isinstance(dataset, _data_loader.MixtureDataset) else None
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # NOTE: train_only_transforms (image augmentation) is deliberately NOT applied
            # here -- norm stats must describe the un-augmented state/action distribution.
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    batch_sampler = None
    if mixture_sizes is not None:
        assert mixture_offsets is not None
        # Draw through the SAME fixed-ratio sampler used for training, so the stats
        # describe the sampled mixture (e.g. 50/50) rather than the raw storage ratio
        # (33/67), which would be ego-dominated and would misnormalize teleop.
        batch_sampler = _data_loader.make_stratified_batch_sampler(
            data_config, mixture_sizes, mixture_offsets, batch_size, shuffle=True
        )
        if max_frames is None:
            raise ValueError(
                "Computing norm stats over a mixture requires --max-frames: a full pass would "
                "oversample the smaller source many times over. 200k frames is plenty for QUANTILES."
            )
        num_batches = max_frames // batch_size

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle and batch_sampler is None,
        batch_sampler=batch_sampler,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None, skip_videos: bool = False):
    """Compute norm stats for a config.

    Args:
        config_name: Name of the train config.
        max_frames: Number of frames to sample. Required for mixture configs.
        skip_videos: Skip video decoding entirely (dummy 2x2 frames). Norm stats only
            use `state`/`actions` from parquet, so this is a pure speedup -- on the
            YAM mixture it is the difference between ~hours and ~minutes.
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
            skip_videos=skip_videos,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # asset_id, not repo_id: a mixture has several repo ids, and different mixture
    # ratios need their own stats even when the source datasets are identical.
    asset_id = data_config.asset_id or data_config.repo_id
    output_path = config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # Record WHICH distribution these describe, right next to them. Stats are computed
    # through the training sampler, so a copied directory or a reused asset_id would
    # otherwise normalise training against the wrong distribution with nothing to
    # notice it -- and run_yam.sh skips this stage whenever the file already exists.
    # `_load_norm_stats` checks this on the way back in.
    _fingerprint.write(
        output_path,
        _fingerprint.from_data_config_factory(config.data),
        config_name=config.name,
        asset_id=asset_id,
    )


if __name__ == "__main__":
    tyro.cli(main)
