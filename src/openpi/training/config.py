"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
from openpi.policies import yam_policy
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.augment as _augment
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class MixtureSource:
    """One dataset in a stratified fixed-count mixture (see `DataConfig.mixture`).

    `samples_per_batch` is a hard per-batch count, not a probability: every batch
    contains exactly this many samples from this dataset, so the gradient share of
    a source equals `samples_per_batch / batch_size` exactly. The counts across all
    sources must sum to the training batch size.
    """

    # LeRobot repo id for this source.
    repo_id: str
    # Exact number of samples this source contributes to every batch.
    samples_per_batch: int
    # Optional local dataset root. If None, LeRobot resolves $HF_LEROBOT_HOME/<repo_id>.
    root: str | None = None
    # Explicit episode indices to withhold from training. Use this when the validation
    # episodes are chosen by hand (inspected on the box) rather than sampled: list them
    # here and they are excluded from the training stream without touching the dataset
    # on disk. Out-of-range indices raise rather than being silently ignored.
    exclude_episodes: Sequence[int] = ()
    # Alternative to `exclude_episodes`: withhold a deterministic random fraction of this
    # source's episodes. Selection matches fidelity-sdk's HoldoutSpec given the same seed;
    # see `select_holdout_episodes`. The two can be combined (the union is withheld).
    holdout_fraction: float = 0.0
    holdout_seed: int = 0


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Transforms applied ONLY by the training data loader -- not by the policy/inference
    # chain and not by compute_norm_stats. This is where image augmentation lives, so it
    # is off at eval/serving by construction. Applied after `data_transforms.inputs` and
    # before normalization.
    train_only_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # If non-empty, the data loader samples from several LeRobot datasets with a fixed
    # per-batch count per source instead of from the single `repo_id`. All sources must
    # share the same feature schema (the repack/data transforms are applied to all of
    # them). `repo_id`/`asset_id` still determine where norm stats are read from.
    mixture: Sequence[MixtureSource] = ()

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYamDataConfig(DataConfigFactory):
    # YAM stores ABSOLUTE actions, so convert the arm-joint dims to deltas
    # (relative to current state); grippers stay absolute. This matches how pi0 is
    # trained and mirrors ALOHA's `use_delta_joint_actions`. Set False only if your
    # dataset already stores delta actions (like LIBERO).
    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs, model_config):
        # Rename raw LeRobot dataset keys -> intermediate keys used by YamInputs.
        # LEFT = target key (what YamInputs reads), RIGHT = your dataset feature name.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_image": "observation.images.top",
                        "observation/left_wrist_image": "observation.images.left_wrist",
                        "observation/right_wrist_image": "observation.images.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        # prompt_from_task adds "prompt" BEFORE this repack; it must
                        # be listed here or RepackTransform drops it -> "Prompt is required".
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[yam_policy.YamInputs(model_type=model_config.model_type)],
            outputs=[yam_policy.YamOutputs()],
        )

        # ABSOLUTE -> DELTA conversion. YAM stores ABSOLUTE actions but pi0 trains
        # on deltas, so subtract the current state from the arm-joint dims. The mask
        # make_bool_mask(6, -1, 6, -1) = [6 joints -> delta, 1 gripper -> absolute]
        # per arm (14 dims total), same as ALOHA. At inference AbsoluteActions adds
        # the state back. ASSUMES the action/state layout is:
        #   [arm0: 6 joints, arm0 gripper, arm1: 6 joints, arm1 gripper]
        # If your ordering differs, change the mask accordingly.
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Our YAM dataset's action column is "action" (singular), like ALOHA.
            # openpi defaults to "actions" (LIBERO's converted name) -> KeyError,
            # so tell LeRobot the real column name for the temporal action chunk.
            action_sequence_keys=("action",),
        )


def _load_teleop_holdout() -> tuple[int, ...]:
    """Teleop episodes withheld from training, copied out for offline eval.

    Generated by `vast_run/make_holdout.py`, which also wrote the standalone holdout
    dataset; the manifest is checked in so the split travels with the code and the eval
    scores exactly the episodes training never saw. Selected by DURATION rather than
    episode count, which is what brings training storage to the 30/70 teleop:ego split
    the spec asks for: 249 episodes / 77,806 frames / 0.720 h held out, leaving 460,647
    teleop frames against ego's 1,074,893 -- teleop is 30.00% of stored training data.

    Note this does NOT change the training mixture. The sampler draws a fixed 32/32 per
    batch, so teleop's gradient share stays exactly 50% however much of it is on disk.
    What changes is that these frames are never trained on, so eval on them is honest.
    """
    manifest = pathlib.Path(__file__).parent / "teleop_holdout.json"
    episodes = json.loads(manifest.read_text())["episodes"]
    return tuple(int(ep) for ep in episodes)


_TELEOP_HOLDOUT_EPISODES = _load_teleop_holdout()

# Roots for the pre-selected 7h mixture, built by vast_run/select_mixture.py +
# vast_run/build_mixture.py. These are standalone LeRobot v2.1 datasets renumbered
# 0..N-1, not views onto the full sources, so no index-based exclusion applies to them:
#   teleop  761 eps / 251,703 frames / 2.331 h   (holdout already physically removed)
#   ego   4,174 eps / 505,837 frames / 4.684 h   (min 30 episodes per object)
# Override with YAM7H_TELEOP_ROOT / YAM7H_EGO_ROOT if the box lays them out elsewhere.
_TELEOP_7H_ROOT = os.environ.get("YAM7H_TELEOP_ROOT", "/workspace/data/yam7h/teleop")
_EGO_7H_ROOT = os.environ.get("YAM7H_EGO_ROOT", "/workspace/data/yam7h/ego")


@dataclasses.dataclass(frozen=True)
class LeRobotYamMixtureDataConfig(LeRobotYamDataConfig):
    """YAM data config that samples from several LeRobot datasets at a fixed ratio.

    Used for the teleop-oversampling experiment: the two source datasets are stored
    at ~33% teleop / 67% ego by frames, but training draws a fixed number of samples
    per source per batch (e.g. 32/32), so teleop's gradient share is set explicitly
    and independently of how much of it there is on disk. There is deliberately no
    per-source loss weighting -- sampling is the single lever.

    All sources share this config's repack/data transforms, so they must share a
    feature schema (same camera keys, same state/action layout, same fps).
    """

    # The mixture. `samples_per_batch` across all sources must equal TrainConfig.batch_size.
    sources: tyro.conf.Suppress[Sequence[MixtureSource]] = ()
    # Training-time image augmentation. Set to None to disable.
    augment_config: tyro.conf.Suppress[_augment.ImageAugmentConfig | None] = dataclasses.field(
        default_factory=_augment.ImageAugmentConfig
    )

    @override
    def create(self, assets_dirs, model_config):
        config = super().create(assets_dirs, model_config)
        train_only = _transforms.Group()
        if self.augment_config is not None:
            train_only = _transforms.Group(inputs=[_augment.ImageAugment(self.augment_config)])
        return dataclasses.replace(config, mixture=tuple(self.sources), train_only_transforms=train_only)


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # How many of the most recent checkpoints to keep. Checkpoints pinned by `keep_period`
    # are kept on top of this rolling window.
    max_to_keep: int = 1

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # ---- full fine-tune (needs 80GB; NOT the recommended first run for YAM) ----
    TrainConfig(
        name="pi0_fast_yam",
        model=pi0_fast.Pi0FASTConfig(action_dim=14, action_horizon=50, max_token_len=300),
        data=LeRobotYamDataConfig(
            repo_id="Kavin60606/yam_pi0fast_train",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    # ---- LoRA (low-memory) fine-tune -- RECOMMENDED for YAM ----
    # Hyperparameters tuned for the YAM dataset (335k frames / 163 tasks, bimanual
    # teleop, ~12 demos/task). GOTCHA: keep lr_schedule.decay_steps == num_train_steps
    # so the cosine LR fully decays by the end -- openpi's decay_steps defaults to
    # 30k independently of num_train_steps.
    TrainConfig(
        name="pi0_fast_yam_low_mem_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=14,
            action_horizon=50,
            max_token_len=300,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotYamDataConfig(
            repo_id="Kavin60606/yam_pi0fast_train",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=7_000,  # ~1.3 epochs at batch 64 over 335k frames (budget-fit; --resume for more)
        # LR scaled UP for the larger batch (sqrt rule: 2.5e-5 * sqrt(64/32) ~= 3.5e-5).
        # decay_steps MUST equal num_train_steps so the cosine LR fully decays.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=3.5e-5, decay_steps=7_000, decay_lr=3.5e-6
        ),
        batch_size=64,  # 2x A100-80GB data-parallel -> 32 samples/GPU
        num_workers=8,  # 3-camera video decode is the loader bottleneck; feed 2 GPUs at batch 64
        save_interval=500,  # ~7 checkpoints over the run
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=14,
            action_horizon=50,
            max_token_len=300,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # ---- Teleop-oversampling co-fine-tune (E-A / E-B / E-C) ----
    #
    # Two source datasets, stored at ~33% teleop / 67% ego by frames:
    #   teleop: angkul07/abc-teleop                              ~540k frames  (~5h, real robot)
    #   ego:    angkul07/EgoDex-PickPlace-YAM-14dof-multiview   ~1080k frames (~10h, retargeted)
    # Training draws a FIXED number of samples per source per batch, so teleop's
    # gradient share is set explicitly (p_teleop) rather than inherited from the
    # storage ratio. No per-source loss weighting -- sampling is the only lever.
    #
    # Epoch math (B=64, S=steps, p=p_teleop):
    #   teleop epochs = p*B*S/540k ; ego epochs = (1-p)*B*S/1080k
    #   E-A: 3.0 / 1.5   E-B: 3.7 / 1.1   E-C: 5.9 / 3.0
    #
    # ---- yam7h_* arms: same experiment on the 7h mixture ----
    #
    # Same two upstreams, pre-selected down to 7.014 h and stored locally:
    #   teleop  251,703 frames (2.331 h)   ego  505,837 frames (4.684 h)
    # Storage is 33/67 here, but the fixed 32/32 draw means the gradient share is 50/50
    # regardless -- the ratio is set by sampling, never by how much is on disk.
    #
    # Step counts come from holding the ORIGINAL epoch budget, not from scaling steps by
    # the hours removed. The 15h E-A ran 3.0 teleop / 1.5 ego epochs; keeping that on 7h:
    #   S = 3.0 * 251,703 / 32 = 23,597 -> 23,600, which also lands ego at exactly 1.5.
    # E-B holds the SAME 23,600 steps rather than recomputing from its 40/24 split: the
    # arms must differ in mixing ratio alone, so giving them unequal optimization budgets
    # would confound the comparison. (E-B therefore runs 3.75 teleop / 0.9 ego epochs,
    # exactly as the 15h E-B ran 3.7 / 1.1.) E-C stays the long arm at 2x E-A.
    # Warmup is ~2% of steps throughout, matching 1000/50k on the 15h arms.
    #
    # Deliberately UNCHANGED from the 15h arms: peak LR 3.5e-5, batch 64, LoRA variant,
    # EMA disabled. The previous run was undertrained, not misoptimized -- grad norms were
    # stable -- so only schedule-length-dependent knobs move. Changing LR or batch size at
    # the same time as dataset scale would make the result unattributable.
    #
    # GOTCHAS carried over from the 7k run:
    #   * decay_steps MUST equal num_train_steps (openpi defaults it to 30k).
    #   * norm stats are per-ratio, computed through the SAME sampler:
    #       uv run scripts/compute_norm_stats.py --config-name <name> \
    #           --max-frames 200000 --skip-videos
    #     Raw-storage stats would be ego-dominated and would misnormalize the teleop
    #     grippers this experiment prioritizes.
    #   * The 7h arms need their OWN norm stats -- hence asset_id yam7h_* rather than
    #     yam_mix_*. The mixture ratio is unchanged but the underlying distribution is
    #     not: ego dropped 160 low-count objects and teleop is a different 761-episode
    #     sample, so the q01/q99 quantiles move. Reusing yam_mix_p50 would silently
    #     normalize against the 15h distribution, and run_yam.sh skips stat computation
    #     whenever the file already exists.
    #   * These are fresh runs from pi0_fast_base, NOT resumes of the 7k checkpoint
    #     (a resume would drag along its exhausted cosine schedule and old norm stats).
    *[
        TrainConfig(
            name=name,
            model=pi0_fast.Pi0FASTConfig(
                action_dim=14,
                action_horizon=50,
                max_token_len=300,
                paligemma_variant="gemma_2b_lora",
            ),
            data=LeRobotYamMixtureDataConfig(
                # repo_id is the "primary" source; norm stats live under assets/<config>/<asset_id>.
                repo_id="angkul07/abc-teleop",
                assets=AssetsConfig(asset_id=asset_id),
                # Required: the repack transform forwards "prompt", and pi0-FAST's
                # tokenizer raises "Prompt is required" without it.
                base_config=DataConfig(prompt_from_task=True),
                sources=(
                    MixtureSource(
                        repo_id="angkul07/abc-teleop",
                        samples_per_batch=teleop_per_batch,
                        # None for the full-dataset arms (resolves $HF_LEROBOT_HOME/<repo_id>);
                        # a local path for the pre-selected 7h arms.
                        root=teleop_root,
                        # Withheld for offline eval; also what makes training storage 30/70.
                        # Nothing moves on disk -- these episodes are simply never sampled.
                        #
                        # EMPTY for the 7h arms, and it must stay empty: those indices are
                        # ORIGINAL abc-teleop indices, but the 7h dataset is a renumbered
                        # 0..760 subset that already physically excludes the holdout. Reusing
                        # them there would be wrong twice over -- 143 of the 249 fall outside
                        # 0..760 (which raises), and the other 106 are in range and would
                        # silently withhold completely unrelated episodes.
                        exclude_episodes=exclude_teleop,
                    ),
                    MixtureSource(
                        repo_id="angkul07/EgoDex-PickPlace-YAM-14dof-multiview",
                        samples_per_batch=64 - teleop_per_batch,
                        root=ego_root,
                        # No ego holdout: headline metrics are teleop-only by design.
                    ),
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=num_steps,
            # Peak LR unchanged from the 7k run: it was stable there (grad_norm ~2.5),
            # and the failure mode was too few steps, not a bad optimizer config.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=warmup_steps, peak_lr=3.5e-5, decay_steps=num_steps, decay_lr=3.5e-6
            ),
            batch_size=64,
            num_workers=8,
            # Save often, keep a short rolling window: 4 x ~10GB instead of 50.
            # keep_period=None means nothing is pinned permanently, so this really is
            # "the most recent 4" -- eval or upload a checkpoint before it rolls off
            # (4 checkpoints = 4k steps of headroom).
            save_interval=1_000,
            max_to_keep=4,
            keep_period=None,
            freeze_filter=pi0_fast.Pi0FASTConfig(
                action_dim=14,
                action_horizon=50,
                max_token_len=300,
                paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        )
        for (
            name,
            asset_id,
            teleop_per_batch,
            num_steps,
            warmup_steps,
            teleop_root,
            ego_root,
            exclude_teleop,
        ) in [
            # ---- 15h arms: full abc-teleop + full ego, holdout excluded by index ----
            # E-A: baseline ratio, matched to the previous 50/50 experiment.
            ("pi0_fast_yam_mix_ea", "yam_mix_p50", 32, 50_000, 1_000, None, None, _TELEOP_HOLDOUT_EPISODES),
            # E-B: teleop-biased arm (62.5% gradient share on the higher-quality source).
            ("pi0_fast_yam_mix_eb", "yam_mix_p625", 40, 50_000, 1_000, None, None, _TELEOP_HOLDOUT_EPISODES),
            # E-C: long run at the safe ratio, aligned with the official recipe length.
            ("pi0_fast_yam_mix_ec", "yam_mix_p50", 32, 100_000, 2_000, None, None, _TELEOP_HOLDOUT_EPISODES),
            # ---- 7h arms: pre-selected local subsets, holdout already physically absent ----
            ("pi0_fast_yam7h_ea", "yam7h_p50", 32, 23_600, 500, _TELEOP_7H_ROOT, _EGO_7H_ROOT, ()),
            ("pi0_fast_yam7h_eb", "yam7h_p625", 40, 23_600, 500, _TELEOP_7H_ROOT, _EGO_7H_ROOT, ()),
            ("pi0_fast_yam7h_ec", "yam7h_p50", 32, 47_200, 950, _TELEOP_7H_ROOT, _EGO_7H_ROOT, ()),
        ]
    ],
    # ---- pi05_yam7h_* arms: the same 7h mixture under pi0.5 (flow matching) ----
    #
    # Deltas from the pi0_fast_yam7h_* block above, and the reason for each:
    #
    #   action_dim 14 -> 32  (NOT OPTIONAL)
    #     pi05_base ships action_in_proj as Linear(32, 1024) and action_out_proj as
    #     Linear(1024, 32). _merge_params() in training/weight_loaders.py matches on
    #     KEY NAMES ONLY and never compares shapes, so action_dim=14 does not raise at
    #     load time -- it fails later inside jit with an error that points nowhere near
    #     the cause. Nothing downstream cares: YamOutputs already slices [..., :14] and
    #     PadStatesAndActions zero-pads 14 -> 32.
    #
    #     Note this affects the ACTIONS only. With pi05=True, embed_suffix never emits
    #     a state token, so the padded 32-dim `state` array is carried for the type
    #     spec and never read by the model.
    #
    #   max_token_len 300 -> 200
    #     300 existed to hold FAST action tokens. pi0.5 has none -- actions go to the
    #     flow expert as continuous conditioning and never enter the token stream.
    #     The prompt is "Task: {task}, State: {ints};\nAction: ", and TokenizePrompt
    #     runs BEFORE PadStatesAndActions in ModelTransformFactory, so it tokenizes the
    #     real 14 state values, not 32 padded ones. ~90 tokens in practice.
    #     Confirm on your own task strings with vast_run/pi05/pi05_preflight.py.
    #
    #   discrete_state_input: deliberately NOT SET.
    #     Pi0Config.__post_init__ defaults it to `pi05`, i.e. True. Do NOT copy
    #     discrete_state_input=False from pi05_libero: with pi05=True the state token
    #     is already absent from the suffix, so False means the model receives NO
    #     proprioception at all.
    #
    #   action expert is full-rank and trainable.
    #     get_freeze_filter() with paligemma_variant="gemma_2b_lora" freezes ".*llm.*"
    #     EXCEPT the "_1"-suffixed action-expert params and EXCEPT lora.
    #     MEASURED on this config via jax.eval_shape (vast_run/pi05/pi05_preflight.py),
    #     not estimated: trainable 872.8M = action expert 427.9M + SigLIP 414.8M +
    #     LoRA 27.9M + action/time projections 2.2M; frozen Gemma 2B trunk 2508.5M.
    #     That is ~10.5 GB/GPU of AdamW moments + grads under --fsdp-devices 1
    #     (12 bytes/param), against ~5.4 GB for the pi0-FAST arms' ~448M.
    #     NOTE the expert is 427.9M, not the 311M quoted for gemma_300m -- the "300m"
    #     name counts the transformer stack only.
    #     SigLIP is trainable in BOTH families -- the freeze regex is ".*llm.*" and
    #     SigLIP lives at PaliGemma.img, not PaliGemma.llm.
    #     If grad_norm runs hot in the first 500 steps, the conservative fallback is
    #     action_expert_variant="gemma_300m_lora" (rank 32), which puts the trainable
    #     set back at ~459M. Set it on BOTH the model and the freeze_filter.
    #
    #   norm stats are REUSED from the pi0-FAST arms -- do not recompute.
    #     compute_norm_stats.py applies repack + data_transforms only (never
    #     model_transforms), and DataConfig.use_quantile_norm is
    #     `model_type != ModelType.PI0`, which is True for PI0_FAST and PI05 alike.
    #     Identical 14-dim quantile stats. Copy the directory per arm:
    #       cp -r assets/pi0_fast_yam7h_ea/yam7h_p50  assets/pi05_yam7h_ea/
    #       cp -r assets/pi0_fast_yam7h_eb/yam7h_p625 assets/pi05_yam7h_eb/
    #       cp -r assets/pi0_fast_yam7h_ec/yam7h_p50  assets/pi05_yam7h_ec/
    #     run_yam.sh stage [1/3] then skips recomputation on its own.
    #
    #   Held fixed on purpose, so the only moving part is the architecture:
    #     batch 64, 23,600 steps (3.00 teleop / 1.49 ego epochs), warmup 500,
    #     cosine 3.5e-5 -> 3.5e-6, ema off, clip_gradient_norm 1.0 (AdamW default,
    #     already matches pi05_libero).
    #     For reference, openpi's own pi0.5 recipes (pi05_libero,
    #     pi05_full_droid_finetune) use 5e-5 HELD CONSTANT -- decay_lr == peak_lr, no
    #     cosine. That is a legitimate alternative, but adopting it here would confound
    #     the architecture comparison. If you do switch, run_yam.sh's
    #     `decay_steps == num_train_steps` assert still holds; just set decay_lr=5e-5.
    #
    #   Checkpoints are ~25-30% larger than pi0-FAST (+311M expert params and their
    #   optimizer state). Drop max_to_keep to 2 if the checkpoint volume is tight.
    *[
        TrainConfig(
            name=name,
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=50,
                max_token_len=200,
                paligemma_variant="gemma_2b_lora",
            ),
            data=LeRobotYamMixtureDataConfig(
                # repo_id is the "primary" source; norm stats live under assets/<config>/<asset_id>.
                repo_id="angkul07/abc-teleop",
                assets=AssetsConfig(asset_id=asset_id),
                # Still required: the repack transform forwards "prompt", and
                # PaligemmaTokenizer needs it. TokenizePrompt raises without it.
                base_config=DataConfig(prompt_from_task=True),
                sources=(
                    MixtureSource(
                        repo_id="angkul07/abc-teleop",
                        samples_per_batch=teleop_per_batch,
                        root=_TELEOP_7H_ROOT,
                        # MUST stay empty, same as the pi0-FAST 7h arms: the 7h teleop
                        # set is a renumbered 0..760 subset that already physically
                        # excludes the holdout. _TELEOP_HOLDOUT_EPISODES holds ORIGINAL
                        # abc-teleop indices and would be wrong twice over here.
                        exclude_episodes=(),
                    ),
                    MixtureSource(
                        repo_id="angkul07/EgoDex-PickPlace-YAM-14dof-multiview",
                        samples_per_batch=64 - teleop_per_batch,
                        root=_EGO_7H_ROOT,
                        # No ego holdout: headline metrics are teleop-only by design.
                    ),
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=num_steps,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=warmup_steps, peak_lr=3.5e-5, decay_steps=num_steps, decay_lr=3.5e-6
            ),
            batch_size=64,
            # 16, not the 8 the pi0-FAST arms use. Three-camera video decode is the
            # loader bottleneck, and pi0.5's per-step GPU cost is ~0.95-1.05x pi0-FAST's
            # -- so if the loader was already the binding constraint, the architecture
            # swap buys nothing until this is raised. The box has 128 cores and 503 GB
            # RAM, so 16 workers is still far from saturating either.
            # This knob does NOT touch the optimization: same batch, same steps, same
            # sample order. It only changes how fast batches are produced, so it cannot
            # confound the pi0-FAST comparison.
            # Watch RAM in the first few hundred steps -- each worker holds its own
            # decode buffers. If GPU util is still under ~85%, raise it again.
            num_workers=16,
            save_interval=1_000,
            max_to_keep=4,
            keep_period=None,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=50,
                max_token_len=200,
                paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        )
        for (name, asset_id, teleop_per_batch, num_steps, warmup_steps) in [
            # E-A: baseline ratio, direct counterpart to pi0_fast_yam7h_ea.
            ("pi05_yam7h_ea", "yam7h_p50", 32, 23_600, 500),
            # E-B: teleop-biased arm (62.5% gradient share on the higher-quality source).
            ("pi05_yam7h_eb", "yam7h_p625", 40, 23_600, 500),
            # E-C: 2x length at the baseline ratio.
            ("pi05_yam7h_ec", "yam7h_p50", 32, 47_200, 950),
        ]
    ],
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
