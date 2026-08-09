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
from openpi.policies import piper_policy
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

# Roots for the 10/90 mixture: the same 7.014 h total re-split so teleop is 10% of the
# frames instead of 33%. Same builders (vast_run/select_mixture.py + build_mixture.py),
# same renumbered-0..N-1 standalone layout, so no index-based exclusion applies here either.
#   teleop   75,754 frames (0.701 h,  ~229 eps)  -- a subset of the 7h teleop selection
#   ego     681,786 frames (6.313 h, ~5,626 eps) -- LARGER than the 7h ego selection
# NOTE the ego side cannot be carved out of _EGO_7H_ROOT: that root holds only 505,837
# frames, so select_mixture.py must re-run against the full upstream (~1,080k frames
# available) with a lower min-episodes-per-object than the 30 used for the 7h build.
_TELEOP_1090_ROOT = os.environ.get("YAM1090_TELEOP_ROOT", "/workspace/data/yam1090/teleop")
_EGO_1090_ROOT = os.environ.get("YAM1090_EGO_ROOT", "/workspace/data/yam1090/ego")

# 100%-teleop single-source run: abc-ego `put_the_screwdriver_in_the_bin`, converted
# from MCAP by vast_run/mcap_to_lerobot.py.
# 2,234 episodes / 730,496 frames / 6.76 h at 30 fps.
_ABCEGO_SD_ROOT = os.environ.get("ABCEGO_SD_ROOT", "/workspace/abc-ego-lerobot")

# 50/50 ego+teleop merge, pre-merged into ONE LeRobot v2.1 dataset rather than sampled
# from two roots. 4,272 episodes / 755,964 frames / 7.00 h at 30 fps, 224x224.
# Built by fd/sdk/lerobot_run/dataset/build_50run.py; the teleop half excludes every
# episode in abc-teleop-holdout (249/249 matched by content hash, 0 leaks).
# Published as angkul07/50_run_v21_fixed (private). Fetch and unpack with:
#   hf download angkul07/50_run_v21_fixed 50_run_v21.tar --repo-type dataset --local-dir /workspace
#   tar xf /workspace/50_run_v21.tar -C /workspace && mv /workspace/50_run_old /workspace/50_run
# The tar unpacks to `50_run_old/` (its name on the build box); rename it or point
# YAM50RUN_ROOT at it, otherwise stage [0] reports the dataset as missing.
_50RUN_ROOT = os.environ.get("YAM50RUN_ROOT", "/workspace/50_run")

# Roots for the 1-hour Piper H mixture. A DIFFERENT EMBODIMENT from every YAM root
# above -- Piper H, not YAM -- and a different frame rate: 20 Hz on both halves,
# against 30 Hz everywhere else in this file. Both halves are LeRobot v2.1, 14-D
# state/action, and carry the same three camera keys (front/right/top).
#   ego     613 eps /  47,953 frames / 39.96 min  -- retargeted EgoDex, 100 tasks
#   teleop  154 eps /  25,075 frames / 20.90 min  -- real Piper H, 1 task
#   total   767 eps /  73,028 frames / 60.86 min  -- ego 65.7% / teleop 34.3% by frames
# Published as angkul07/piper-h-ego-teleop-v21 (private); the README there carries the
# full stats table. Built on the vast box at /workspace/{ego_v21,teleop_v21}, which is
# what these defaults point at.
#
# NOTE both halves are standalone datasets numbered 0..N-1, so `exclude_episodes`
# indices refer to THESE datasets, not to any upstream numbering.
_PIPER1H_TELEOP_ROOT = os.environ.get("PIPER1H_TELEOP_ROOT", "/workspace/teleop_v21")
_PIPER1H_EGO_ROOT = os.environ.get("PIPER1H_EGO_ROOT", "/workspace/ego_v21")

# Ego episodes to withhold for bad retargeting. EMPTY BY DEFAULT -- populating it is a
# decision that has not been made yet, and it changes the step count (see the epoch
# table on pi05_piper1h_ea).
#
# The retarget error is BIMODAL, not uniform: 4.770 cm mean over all 613 clips, but
# 0.701 cm on clips where the wrist orientation constraint is never pinned versus
# 6.944 cm on clips where it is pinned in >50% of frames. The cause is Piper's joint5
# range (+-1.22 rad) against the +-1.571 the source retarget assumed. So the useful
# filter is the WRIST-PINNING FRACTION, not the PASS/WARN/FAIL bucket from QA -- the
# 105/328/180 split cuts across both modes. Fill this with the pinned->50% clip indices
# once that per-episode statistic is exported.
_PIPER1H_EGO_EXCLUDE: tuple[int, ...] = ()

# The 20-minute cut: 10 min teleop + 10 min ego, built on the vast box at
# /workspace/10t_10e/{teleop_v21,ego_v21}. Same schema, same 20 Hz, same three stored
# camera keys, same 14-dim state/action as the 1-hour pair -- so it reuses
# LeRobotPiperMixtureDataConfig and piper_policy unchanged.
#
# It is NOT a proportional scale-down of the 1-hour mixture. That one is 1:2 teleop:ego
# by frames; this one is 1:1 (12,048 vs 12,049 frames). Anything read off this run about
# mixture RATIO does not transfer to the 1-hour run, and vice versa.
#
# Verified on the box before first use, because both defects that broke the 1-hour run
# are properties of the export pipeline, not of the data:
#   * parquet HF metadata: 75/75 teleop files carry it, 0 contain "_type": "List"
#     (the datasets-4.x break). Ego carries no huggingface metadata at all, as before.
#   * video PTS: all 684 videos (225 teleop + 459 ego) start at pts 0.000000, so no
#     setts=PTS-STARTPTS rebase is needed. The 1-hour teleop videos all started at 0.05.
_PIPER20M_TELEOP_ROOT = os.environ.get("PIPER20M_TELEOP_ROOT", "/workspace/10t_10e/teleop_v21")
_PIPER20M_EGO_ROOT = os.environ.get("PIPER20M_EGO_ROOT", "/workspace/10t_10e/ego_v21")


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
class LeRobotPiperDataConfig(DataConfigFactory):
    """Piper H data config. Mirrors LeRobotYamDataConfig; the deltas are the cameras.

    TWO of the three stored camera keys are used. `observation.images.top` is
    deliberately absent from the repack below, which is the whole mechanism for
    dropping it -- RepackTransform discards any key it does not list. See
    `piper_policy` for why: the key names do not describe the content, and `top` is
    the one slot whose content does NOT correspond across the two halves of the
    mixture (teleop `top` is a sideways view of the robot; ego `top` is a wide crop
    on the grasp point).

    DROPPING IT HERE DOES NOT SKIP ITS VIDEO DECODE. LeRobotDataset decodes every
    video feature before the repack runs, so the `top` stream is still paid for on
    the dataloader. Removing that cost means not writing the key during conversion
    (or subsetting the features at the dataset level) -- worth doing, since decode
    already saturates the cgroup CPU quota on the vast boxes.
    """

    # Piper stores ABSOLUTE actions on both halves, so convert the arm-joint dims to
    # deltas relative to current state; grippers stay absolute. Same as YAM/ALOHA.
    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs, model_config):
        # LEFT = target key (what PiperInputs reads), RIGHT = dataset feature name.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/front_image": "observation.images.front",
                        "observation/right_image": "observation.images.right",
                        # "observation.images.top" intentionally omitted -- see docstring.
                        "observation/state": "observation.state",
                        "actions": "action",
                        # prompt_from_task adds "prompt" BEFORE this repack; it must be
                        # listed or RepackTransform drops it -> "Prompt is required".
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[piper_policy.PiperInputs(model_type=model_config.model_type)],
            outputs=[piper_policy.PiperOutputs()],
        )

        # ABSOLUTE -> DELTA. make_bool_mask(6, -1, 6, -1) = [6 joints -> delta,
        # 1 gripper -> absolute] per arm. Piper H is 6-DOF + gripper per arm, so the
        # 14-D layout matches YAM's:
        #   [arm0: 6 joints, arm0 gripper, arm1: 6 joints, arm1 gripper]
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
            # Action column is "action" (singular), like ALOHA and the YAM datasets.
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperMixtureDataConfig(LeRobotPiperDataConfig):
    """Piper H config that samples from several LeRobot datasets at a fixed ratio.

    Identical machinery to LeRobotYamMixtureDataConfig -- fixed samples-per-source
    per batch, so a source's gradient share is set explicitly and independently of
    how much of it is on disk. All sources share this config's repack/data
    transforms, so they must share a feature schema (same camera keys, same
    state/action layout, same fps).
    """

    # `samples_per_batch` across all sources must equal TrainConfig.batch_size.
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
            # 16, not the 8 the pi0-FAST arms use. Do NOT raise this further: a
            # 200-step A/B on 2x A100-80GB (page cache pre-warmed, steady-state
            # windows only) measured 16 -> 3.528 s/step at 95.0% GPU util and
            # 32 -> 3.545 s/step at 94.7%. Doubling workers was 0.5% SLOWER, against
            # a within-config window spread of 0.7-0.9% -- i.e. below the noise
            # floor. Three-camera decode keeps both GPUs fed; this run is
            # compute-bound, not loader-bound.
            # Whether 8 also suffices was not tested; 16 is known-sufficient and
            # costs nothing on a 128-core / 503 GB box, so it stays.
            # The knob does not touch the optimization -- param_norm was bit-identical
            # across both A/B runs at step 175 -- so it cannot confound the pi0-FAST
            # comparison.
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
    # ---- pi05_yam1090_ea: the 10/90 arm -- same 7h, teleop cut to 10% of the frames ----
    #
    # Fourth point on the mixture-ratio sweep. Storage teleop share across the series:
    #   100 % (pi05_abcego_sd)  ->  50 % (pi05_50run_ea)  ->  33 % (pi05_yam7h_ea)  ->  10 % (here)
    # Total hours are held at 7.014 h throughout; only the split moves.
    #
    #   teleop   75,754 frames (0.701 h,  ~229 eps)   10.0 % of stored frames
    #   ego     681,786 frames (6.313 h, ~5,626 eps)  90.0 %
    #   total   757,540 frames (7.014 h @ 30 fps)     same total as the yam7h arms
    #
    # DRAW 24/40 (teleop gradient share 37.5 %), 32,000 steps at batch 64:
    #
    #                              teleop        ego
    #   samples per batch              24         40
    #   gradient share             37.5 %     62.5 %
    #   frames seen over run      768,000  1,280,000
    #   hours-equivalent seen       7.11 h    11.85 h
    #   effective epochs             10.14       1.88
    #   oversample vs storage       3.75x      0.69x
    #   per-frame exposure ratio: each teleop frame is drawn 5.40x as often as each ego frame
    #
    # WHY 24/40 AND NOT 32/32. Three different quantities all get called "oversampling"
    # here and they do NOT move together once the pools are 9.0x apart in size -- the
    # yam7h arms hid this because their pools were only 2.0x apart. Spelled out so the
    # next reader does not re-derive it:
    #   1. gradient share vs storage share -- 37.5 % drawn vs 10.0 % stored = 3.75x.
    #   2. per-frame revisit rate (== the epoch ratio) -- 10.14 / 1.88 = 5.40x.
    #   3. TOTAL presentations over the run = samples_per_batch ratio, 24:40, i.e. ego is
    #      still seen 1.67x more often in absolute terms. Steps cancel out of this one:
    #      it is fixed by the draw alone and no step count can change it.
    # Teleop is oversampled on (1) and (2) and remains the minority on (3). Getting
    # teleop to parity on (3) needs a 32/32 draw, which at this split costs 18.0 teleop
    # epochs to hold ego at 2.0 (42,600 steps). 24/40 was chosen as the point that buys a
    # 3.75x oversample and ~1.9 ego epochs for 10.1 teleop epochs and 32k steps.
    #
    # The break-even draw is 6.4/57.6 -- that is 10 % of 64, i.e. proportional sampling.
    # ANY draw above 6 teleop samples oversamples teleop on measures (1) and (2).
    #
    # Deltas from pi05_yam7h_ea, and the reason for each:
    #
    #   samples_per_batch 32/32 -> 24/40, and steps 23,600 -> 32,000.
    #     Steps are NOT held at 23,600 here, unlike the E-A/E-B pair. That rule existed to
    #     keep two arms of the SAME experiment differing in ratio alone; this is a
    #     different mixture, and at 40 ego samples 23,600 steps would leave ego at 1.39
    #     epochs. 32,000 puts ego at 1.88, close to yam7h_ea's 1.49, so ego exposure stays
    #     roughly comparable across the two mixtures and the teleop volume is the variable.
    #
    #   warmup 500 -> 700 (2.19 % of 32,000, matching the ~2.1 % used throughout).
    #
    #   asset_id yam1090_p375 -- FRESH NORM STATS, do not copy yam7h_* or reuse p50.
    #     Both the ratio and the underlying distribution moved: teleop is a ~229-episode
    #     subsample and ego gained ~176k frames from objects the 30-episode-minimum filter
    #     had excluded, so the q01/q99 quantiles shift on both sides. Stats are computed
    #     through the SAME sampler, so the 24/40 draw is baked into them:
    #       uv run scripts/compute_norm_stats.py --config-name pi05_yam1090_ea \
    #           --max-frames 200000 --skip-videos
    #     run_yam.sh stage [1/3] skips computation whenever the file already exists, so a
    #     stray copied directory would silently normalize against the wrong distribution.
    #
    #   keep_period 5_000 (the yam7h arms use None).
    #     At 10.14 teleop epochs overfitting is the failure mode to watch, and
    #     max_to_keep=4 with save_interval=1_000 retains only a 4,000-step rolling window
    #     -- every early checkpoint would be gone before it could be scored. Pinning
    #     5k/10k/.../30k lets the overfit knee be located after the fact. Set to None if
    #     checkpoint volume is tight (each is ~13 GB).
    #
    # Augmentation: ON, inherited unchanged from LeRobotYamMixtureDataConfig's default
    # (augment_config -> ImageAugmentConfig), same as every yam7h arm. It applies via
    # train_only_transforms, so it is off at eval by construction. Worth knowing before
    # reading the loss curve: it augments IMAGES ONLY. The state and action streams repeat
    # verbatim on all ~10 teleop revisits, so it blunts visual memorization, not action
    # memorization -- which is the other reason keep_period is set above.
    #
    # Held fixed on purpose so the mixture is the only moving part: batch 64, peak LR
    # 3.5e-5 -> 3.5e-6 cosine, EMA off, gemma_2b_lora, action_dim 32, max_token_len 200,
    # num_workers 16. Fresh finetune from pi05_base, not a resume of any yam7h checkpoint.
    #
    # Cost: ~12.3 h on 2x H100 SXM, extrapolated from the 1.388 s/step measured on
    # pi05_50run_ea at the same batch size, architecture and camera count.
    TrainConfig(
        name="pi05_yam1090_ea",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotYamMixtureDataConfig(
            repo_id="angkul07/abc-teleop",
            assets=AssetsConfig(asset_id="yam1090_p375"),
            base_config=DataConfig(prompt_from_task=True),
            sources=(
                MixtureSource(
                    repo_id="angkul07/abc-teleop",
                    samples_per_batch=24,
                    root=_TELEOP_1090_ROOT,
                    # Empty for the same reason as the 7h arms: this root is a renumbered
                    # 0..228 subset that already physically excludes the holdout, while
                    # _TELEOP_HOLDOUT_EPISODES holds ORIGINAL abc-teleop indices.
                    exclude_episodes=(),
                ),
                MixtureSource(
                    repo_id="angkul07/EgoDex-PickPlace-YAM-14dof-multiview",
                    samples_per_batch=40,
                    root=_EGO_1090_ROOT,
                    # No ego holdout: headline metrics are teleop-only by design.
                ),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=32_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=700, peak_lr=3.5e-5, decay_steps=32_000, decay_lr=3.5e-6
        ),
        batch_size=64,
        num_workers=16,
        save_interval=1_000,
        max_to_keep=4,
        keep_period=5_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # ---- pi05_piper1h_ea: the 1-hour Piper H mixture, ~2 epochs per source ----
    #
    # FIRST NON-YAM ARM IN THIS FILE. Different embodiment (Piper H), different rate
    # (20 Hz, not 30), and 10x less data than any yam7h arm.
    #
    #   ego     613 eps /  47,953 frames / 39.96 min   65.7% of stored frames, 100 tasks
    #   teleop  154 eps /  25,075 frames / 20.90 min   34.3% of stored frames, 1 task
    #   total   767 eps /  73,028 frames / 60.86 min
    #
    # DRAW 24/40 at batch 64, 2,400 steps:
    #
    #                              teleop        ego
    #   samples per batch              24         40
    #   gradient share             37.5 %     62.5 %
    #   frames seen over run       57,600     96,000
    #   effective epochs             2.55       2.00
    #   oversample vs storage       1.17x      0.92x
    #   per-frame exposure ratio: each teleop frame is drawn 1.27x as often as each ego frame
    #
    # Teleop's epoch count is against the POST-HOLDOUT pool (~22,600 frames, see
    # holdout_fraction below), not the 25,075 stored -- so teleop is 32.1% of the pool
    # actually trained on, not 34.3%. Combined that is 153,600 presentations over
    # ~70,600 trained frames = 2.18 epochs.
    #
    # THE POST-HOLDOUT FIGURE IS AN ESTIMATE. It assumes the 10% holdout removes an
    # average slice, but teleop episode lengths run 4..352 frames, so the real number
    # will differ. create_torch_dataset logs it at startup ("mixture source ...: N train
    # frames"); read it off the first run and re-derive the epochs if it is far off.
    #
    # WHY ~2 EPOCHS PER SOURCE, AND WHY THAT FORCES 24/40 RATHER THAN AN EVEN DRAW.
    # A fixed-count sampler gives every source the SAME number of frames per step, so
    # epoch counts land in inverse proportion to pool size and one draw fixes one ratio.
    # The pools are 2.12x apart post-holdout, so 32/32 cannot put both sources near 2:
    # it reaches ego 2.0 only at 3,000 steps, by which point teleop is at 4.24. Matching
    # epochs across sources means matching the draw to storage share.
    #
    # The proportional (zero-oversample) draw is 20.5/43.5, i.e. 32.1% of 64. Exactly
    # 2.0/2.0 would be 21/43 at 2,200 steps -- but that is proportional sampling, and it
    # gives up teleop oversampling altogether even though teleop is the real embodiment
    # and the only half the eval scores. 24/40 is the deliberate middle: ego at exactly
    # 2.00, teleop at 2.55, and a residual 1.17x tilt toward the eval domain. Any draw
    # above 21 teleop samples oversamples teleop on gradient share and revisit rate.
    #
    # Note teleop remains the MINORITY on total presentations (24:40) -- steps cancel out
    # of that measure, it is fixed by the draw alone. Same three-quantities point as
    # pi05_yam1090_ea, just at a much smaller pool ratio.
    #
    # Deltas from pi05_yam1090_ea, and the reason for each:
    #
    #   action_horizon 50 -> 30. TWO independent reasons, either sufficient.
    #     (a) openpi's LeRobot path CLAMPS delta_timestamps at episode ends (it repeats
    #         the final action) rather than dropping the sample, and NOTHING in openpi
    #         consumes `is_pad` -- grep it. So clamped steps enter the flow-matching loss
    #         as genuine regression targets. An 11-frame ego episode at H=50 teaches
    #         "emit this pose 39 more times". The clamped share of supervised action
    #         steps is (H-1)/2L; at ego's mean length of 78 frames that is 31.3% at H=50
    #         against 18.5% at H=30, and at ego's MEDIAN length of 60 it is 40.8% against
    #         24.2%. This is signal corruption, not merely reweighting.
    #     (b) 20 Hz. H=30 at 20 Hz is 1.5 s, which is the same physical horizon that H=50
    #         gave the 30 Hz YAM arms (1.67 s). Inheriting 50 here would silently ask for
    #         a 2.5 s chunk.
    #     Must match in `freeze_filter` too, or the LoRA filter is built for a different
    #     model shape than the one being trained.
    #
    #   batch 64 held, steps 32,000 -> 2,400. Set by the ~2-epochs-per-source target
    #     above, not by scaling the yam1090 step count: 40 ego samples x 2,400 steps is
    #     2.00 ego epochs on the nose.
    #
    #     THIS IS A DELIBERATELY LIGHT TOUCH. At ~2 epochs on one hour, from pi05_base
    #     with LoRA, the run will largely preserve base behaviour rather than specialise
    #     to the task -- a conservative first arm, not a converged one. The failure mode
    #     is UNDER-training, which is the opposite of every arm above; do not read a
    #     flat-ish loss curve here as the overfit knee.
    #
    #   warmup 700 -> 60 (2.5% of 2,400, matching the ~2.1-2.5% used throughout).
    #
    #   keep_period 5_000 -> 1_000, save_interval 1_000 -> 250, max_to_keep 4 -> 2.
    #     Not for overfit-knee hunting -- at 2.55 teleop epochs there is unlikely to be
    #     one. save_interval is about resolution: a 2,400-step run needs checkpoints
    #     close enough together to compare, and 1_000 would yield two.
    #
    #     max_to_keep and keep_period are set by DISK, measured on the target box: the
    #     A100 container has a 150 GB overlay and nothing else usable (the 16 T
    #     vg0-lv_storage is a host bind-mount on /etc/hosts, not writable space).
    #     pi0.5 checkpoints run ~16-17 GB -- the ~13 GB of the pi0-FAST yam arms plus
    #     the 25-30% from training the action expert full-rank. Budget:
    #       uv env 11 GB (MEASURED after uv sync) + pi05_base ~14 GB + datasets ~5 GB
    #       = ~30 GB before training, leaving ~120 GB.
    #     At max_to_keep=4 / keep_period=500 the retained set is pins {500,1000,1500,
    #     2000} union the rolling last four {1750,2250,2400} = 7 checkpoints ~115 GB.
    #     That fits only if every checkpoint lands at the bottom of the 16-17 GB range
    #     and nothing else is written -- ~5 GB of slack on a 2.4 h run, i.e. an ENOSPC
    #     coin flip late in training.
    #     At max_to_keep=2 / keep_period=1_000 it is {1000,2000} union {2250,2400} = 4
    #     checkpoints ~68 GB, ~98 GB total, ~52 GB of headroom.
    #     vast_run/README.md gives the same advice independently ("drop max_to_keep to
    #     2 if the checkpoint volume is tight") for exactly this full-rank reason.
    #     If you free disk and want finer granularity back, keep_period=500 is the knob.
    #
    #   holdout_fraction=0.1 on teleop instead of an explicit index manifest.
    #     There is no vast_run/make_holdout.py run for this dataset yet, and
    #     _TELEOP_HOLDOUT_EPISODES holds abc-teleop indices that mean nothing here.
    #     select_holdout_episodes is deterministic given (total_episodes, fraction,
    #     seed), so the split is reproducible and travels with the config -- but it is
    #     chosen at RANDOM, not inspected. Replace with an explicit list once someone has
    #     looked at the episodes. ~15 of 154 episodes withheld.
    #     Without this there is no honest offline eval at all.
    #
    #   asset_id piper1h_p50 -- FRESH NORM STATS. Nothing above can be copied: different
    #     embodiment, different joint ranges, different camera set, different fps. Stats
    #     are computed through the SAME sampler, so the 24/40 draw is baked into them:
    #       uv run scripts/compute_norm_stats.py --config-name pi05_piper1h_ea \
    #           --max-frames 200000 --skip-videos
    #     run_yam.sh stage [1/3] skips computation whenever the file already exists, so a
    #     stray copied assets directory would silently normalize the wrong distribution.
    #
    # CAMERAS: 2 real + 1 masked padding slot, against 3 real on every YAM arm. The
    # `top` key is dropped in LeRobotPiperDataConfig's repack because its content does
    # not correspond across the two halves -- see piper_policy's module docstring for
    # the mapping table and for why the wrist view goes in slot 1 rather than slot 2.
    #
    # KNOWN GAPS, deliberately not addressed here:
    #   * Ego gripper scale is NOT rescaled onto the teleop range. The measured problem
    #     from the 7h mixture (ego "open" normalizing closer to teleop's CLOSED, which
    #     pi0.5's few-step flow matching then mode-averages into a half-open hand) has
    #     not been re-measured on Piper. vast_run/pi05/rescale_ego_gripper.py is the
    #     tool; run it against this mixture before trusting grasp behaviour.
    #   * No rotation matching between halves: teleop is 480x640 portrait and stored
    #     rotated, ego is 224x224 square, so the resize to 224 squashes teleop ~1.33x
    #     vertically. Squash-vs-crop is an open preprocessing choice (ROT90_K = 0).
    #   * _PIPER1H_EGO_EXCLUDE is empty. Filling it shrinks the ego pool by ~29% (to
    #     ~33,900 frames) and pushes ego to 2.83 epochs at 2,400 steps; hold ego at 2.00
    #     by dropping to ~1,700 steps, which also takes teleop to ~1.80.
    #   * Ego has 100 task strings and teleop has 1, so `prompt_from_task` conditions
    #     the ego half and does nothing for the half the eval scores.
    #
    # Held fixed on purpose: peak LR 3.5e-5 -> 3.5e-6 cosine, EMA off, gemma_2b_lora,
    # action_dim 32, max_token_len 200. Fresh finetune from pi05_base.
    #
    # TARGET BOX: 2x A100-SXM4-80GB (NVLink), data parallel, 32 samples/GPU.
    #
    # Cost: ~2.4 h. That is 2,400 steps at the 3.528 s/step MEASURED on exactly this
    # hardware at batch 64 -- see the benchmark table in vast_run/README.md. It is not
    # extrapolated across GPU types: the A100-SXM row matches this config's batch size,
    # architecture and camera count directly. H=30 trims the suffix by 20 tokens out of
    # ~1,000, so ~3.4-3.5 s/step if anything; the masked camera saves NOTHING, since
    # Pi0.embed_prefix runs SigLIP on the zeros image regardless.
    #
    # DO NOT re-derive this from the 1.388 s/step quoted on pi05_yam1090_ea. That is an
    # H100 SXM number and is ~2.5x optimistic for an A100. Measured reference points at
    # batch 64: A100 SXM 3.528 | H100 PCIe 2.178 (H100 PCIe is only 1.62x the A100, not
    # the 2.65x the FP32 TFLOPS ratio suggests -- no NVLink on that PCIe box).
    #
    # Memory: fits, with the same headroom as the measured run. pi0.5 trains the action
    # expert FULL-RANK, so the trainable set is 872.8M (expert 427.9M + SigLIP 414.8M +
    # LoRA 27.9M + projections 2.2M) against the 2508.5M frozen Gemma trunk, and AdamW
    # moments and grads come to ~10.5 GB/GPU under --fsdp-devices 1. Checkpoints are
    # ~25-30% larger than a pure-LoRA arm for the same reason.
    #
    # num_workers stays 16: measured non-lever on this exact box (3.528 s/step at 16 vs
    # 3.545 at 32, below the noise floor). The loader uses ~3 cores; it is compute-bound.
    # Worth one preflight though -- check per-GPU clocks and temps before committing the
    # run; the benchmarked A100 pair was power-capped at 275 of 400 W.
    TrainConfig(
        name="pi05_piper1h_ea",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotPiperMixtureDataConfig(
            repo_id="angkul07/piper-h-teleop-v21",
            assets=AssetsConfig(asset_id="piper1h_p50"),
            base_config=DataConfig(prompt_from_task=True),
            sources=(
                MixtureSource(
                    repo_id="angkul07/piper-h-teleop-v21",
                    samples_per_batch=24,
                    root=_PIPER1H_TELEOP_ROOT,
                    # Deterministic random split; see the holdout note above.
                    holdout_fraction=0.1,
                    holdout_seed=0,
                ),
                MixtureSource(
                    repo_id="angkul07/piper-h-ego-v21",
                    samples_per_batch=40,
                    root=_PIPER1H_EGO_ROOT,
                    # No ego holdout: headline metrics are teleop-only by design.
                    exclude_episodes=_PIPER1H_EGO_EXCLUDE,
                ),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=2_400,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=60, peak_lr=3.5e-5, decay_steps=2_400, decay_lr=3.5e-6
        ),
        batch_size=64,
        num_workers=16,
        save_interval=250,
        max_to_keep=2,
        keep_period=1_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # ---- pi05_piper1h_teleop: the teleop half alone, 600 steps ----
    #
    # The single-source control for pi05_piper1h_ea. Same embodiment, same cameras, same
    # horizon, same LR -- the ONLY variable removed is the ego half. Without this arm
    # there is nothing to attribute the mixture's behaviour to: pi05_piper1h_ea cannot
    # tell you whether the 613 retargeted ego clips helped, hurt, or did nothing.
    #
    #   teleop  154 eps / 25,075 frames / 20.90 min / 1 task / 20 Hz
    #   minus the same 15-episode holdout      ->  23,347 train frames
    #
    # 600 steps at batch 64 = 38,400 frames seen = 1.64 epochs.
    #
    # NOTE THIS IS LESS TELEOP EXPOSURE THAN THE MIXTURE ARM GOT. pi05_piper1h_ea drew 24
    # teleop samples for 2,400 steps = 57,600 teleop frames = 2.47 teleop epochs. At 600
    # steps this arm sees 38,400, i.e. 0.67x as much teleop. So a head-to-head is NOT a
    # clean ablation of "ego added or not" -- it also halves teleop exposure and cuts
    # total optimiser steps 4x. To isolate the ego contribution properly the teleop-only
    # arm needs 973 steps (57,600 / 64) to match teleop frames seen, or 2,400 steps to
    # match optimiser steps. 600 was chosen for cost; read the comparison accordingly.
    #
    # Deltas from pi05_piper1h_ea, and the reason for each:
    #
    #   A MIXTURE OF ONE, deliberately -- not a plain single-source data config.
    #     Same two reasons as pi05_abcego_sd below: create_torch_dataset() hardcodes
    #     root=None on the non-mixture path (so the dataset would have to sit at
    #     $HF_LEROBOT_HOME/<repo_id> rather than /workspace/teleop_v21), and run_yam.sh
    #     stage [0] asserts data.mixture is non-empty. StratifiedBatchSampler with one
    #     source just draws all 64 indices from a reshuffled permutation of it, i.e.
    #     ordinary shuffled training.
    #
    #   asset_id piper1h_teleop_only -- FRESH NORM STATS, and this one is not optional.
    #     Reusing piper1h_p50 would be a silent, material bug: those quantiles were
    #     computed over the 24/40 mixture, whose distribution is dominated by ego's much
    #     wider joint ranges (ego action dim0 spans [0.24, 2.56] against teleop's
    #     [0.15, 0.91]) and whose gripper duty cycle is different (teleop p50 = 1.0, ego
    #     p50 = 0.0). Normalizing teleop against that mixture compresses teleop into a
    #     fraction of the [-1, 1] axis and shifts the gripper midpoint. Recompute:
    #       uv run scripts/compute_norm_stats.py --config-name pi05_piper1h_teleop \
    #           --max-frames 200000 --skip-videos
    #
    #   save_interval 250 -> 600, max_to_keep 2 -> 1, keep_period 1_000 -> None.
    #     Last checkpoint only, by request. num_train_steps=600 runs steps 0..599, so an
    #     interval of 600 never fires mid-run and the only write is the final one at 599.
    #     ~13 GB total instead of the mixture arm's ~52 GB.
    #
    #   warmup 60 -> 30 (5% of 600, not the 2.5% used elsewhere).
    #     2.5% of 600 is 15 steps, which is a very fast ramp to a 3.5e-5 peak.
    #     pi05_piper1h_ea's gradient clipping bound on steps 0-2 even with a 60-step
    #     warmup, so the opening is the one place this recipe is near its limit. 30 keeps
    #     the ramp gentle without materially changing the schedule.
    #
    # Held identical to pi05_piper1h_ea on purpose, so the ego half is the only variable:
    # batch 64, action_horizon 30, action_dim 32, max_token_len 200, gemma_2b_lora, peak
    # LR 3.5e-5 -> 3.5e-6 cosine, EMA off, augmentation on, the same 15-episode holdout
    # (holdout_fraction=0.1, seed 0 -> the SAME episodes, since selection is deterministic
    # in total_episodes/fraction/seed and the source dataset is unchanged), and the same
    # two-camera repack that drops `top`.
    #
    # Cost: ~31 min on 2x A100-SXM4-80GB at the 3.07 s/step measured on pi05_piper1h_ea.
    TrainConfig(
        name="pi05_piper1h_teleop",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotPiperMixtureDataConfig(
            repo_id="angkul07/piper-h-teleop-v21",
            assets=AssetsConfig(asset_id="piper1h_teleop_only"),
            base_config=DataConfig(prompt_from_task=True),
            sources=(
                MixtureSource(
                    repo_id="angkul07/piper-h-teleop-v21",
                    samples_per_batch=64,
                    root=_PIPER1H_TELEOP_ROOT,
                    # Same split as the mixture arm -- deterministic in (total_episodes,
                    # fraction, seed), so these are the same 15 episodes.
                    holdout_fraction=0.1,
                    holdout_seed=0,
                ),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=600,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=30, peak_lr=3.5e-5, decay_steps=600, decay_lr=3.5e-6
        ),
        batch_size=64,
        num_workers=16,
        save_interval=600,
        max_to_keep=1,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # ---- pi05_piper20m_ea: the 20-minute cut, 10 min teleop + 10 min ego, 800 steps ----
    #
    # A small, cheap mixture on the same embodiment and the same recipe as
    # pi05_piper1h_ea. Roughly 1/6 the data and 1/3 the steps, so ~40 min instead of
    # ~2 h. Useful as a fast turn of the crank; see the WHAT THIS CANNOT TELL YOU note
    # at the bottom before comparing it to anything.
    #
    #   teleop   75 eps / 12,048 frames / 10.04 min / 20 Hz / mean ep 164.7 frames
    #     minus an 8-episode holdout (fraction 0.1, seed 0)  ->  11,035 train frames
    #     held out: [1, 2, 5, 19, 22, 35, 43, 57]
    #   ego     153 eps / 12,049 frames / 10.04 min / 20 Hz / mean ep  78.8 frames
    #     no holdout, matching pi05_piper1h_ea
    #
    # THE DRAW IS 32/32, NOT THE 1-HOUR RUN'S 24/40. The 24/40 split was not a preference
    # for ego, it was derived: the pools there are 1:2, so an even gradient share would
    # have given the halves different revisit rates, and 24/40 was what put both at ~2
    # epochs. Here the pools are already 1:1, so the same rule gives 32/32:
    #
    #   teleop  32 x 800 = 25,600 presentations / 11,035 =  2.32 epochs
    #   ego     32 x 800 = 25,600 presentations / 12,049 =  2.12 epochs
    #
    # against the 1-hour run's 2.47 / 2.00. So the design principle carries over even
    # though the numbers do not. Note the two knobs move together and only their product
    # matters for exposure -- 32/32 at 800 and 16/16 at 1,600 present identical frame
    # counts, and differ only in optimiser steps and batch composition.
    #
    # Deltas from pi05_piper1h_ea, and the reason for each:
    #
    #   asset_id piper20m_p50 -- FRESH NORM STATS, not optional. Same argument as
    #     pi05_piper1h_teleop: piper1h_p50's quantiles were computed over a 1:2 mixture
    #     of different episodes. Different pool, different q01/q99, and reusing them
    #     silently mis-scales the inputs. Recompute BEFORE training:
    #       uv run scripts/compute_norm_stats.py --config-name pi05_piper20m_ea \
    #           --max-frames 200000 --skip-videos
    #
    #   num_train_steps 2,400 -> 800, decay_steps tracks it so the cosine still lands on
    #     3.5e-6 at the end rather than being truncated mid-decay.
    #
    #   warmup 60 -> 40 (5% of 800, the same fraction pi05_piper1h_teleop used at 600).
    #     2.5% of 800 is 20 steps, a sharp ramp to a 3.5e-5 peak, and the opening is where
    #     this recipe is closest to its limit -- pi05_piper1h_ea still hit the gradient
    #     clip on steps 0-2 with a 60-step warmup.
    #
    #   save_interval 250 -> 800, max_to_keep 2 -> 1, keep_period 1_000 -> None.
    #     Last checkpoint only, by request. 800 steps runs indices 0..799, so an interval
    #     of 800 never fires mid-run and the only write is the final one at 799. ~13 GB.
    #
    # Held identical to pi05_piper1h_ea so the recipe is the constant: batch 64,
    # action_horizon 30, action_dim 32, max_token_len 200, gemma_2b_lora, peak LR
    # 3.5e-5 -> 3.5e-6 cosine, EMA off, augmentation on, num_workers 16, and the same
    # two-camera repack that drops `top`.
    #
    # H=30 is still the right horizon here, and by a wider margin on teleop: the clamped
    # share -- the fraction of training targets that are the episode's final action
    # repeated, which openpi feeds to the flow loss as a real regression target because
    # nothing consumes is_pad -- is (H-1)/2L, i.e. 8.8% on teleop (mean ep 164.7) and
    # 18.4% on ego (mean ep 78.8). At H=50 those become 14.9% and 31.1%.
    #
    # WHAT THIS CANNOT TELL YOU. Three things break a head-to-head with pi05_piper1h_ea:
    # the mixture ratio differs (1:1 vs 1:2 on disk, 32/32 vs 24/40 drawn), the episodes
    # are a different and possibly overlapping subset, and the holdout is a DIFFERENT
    # 8 episodes than the 1-hour run's 15 -- selection is deterministic in
    # (total_episodes, fraction, seed) and total_episodes is 75 here versus 154 there.
    # So a checkpoint from this config must not be scored against the 1-hour holdout, and
    # a checkpoint from that one must not be scored against this holdout: in both
    # directions some of the "held out" episodes are very likely in the other's training
    # set. Whether these 75 teleop episodes are a subset of the 1-hour 154 has not been
    # checked; until it is, treat any cross-run eval number as contaminated.
    #
    # Cost: ~41 min on 2x A100-SXM4-80GB at the 3.07 s/step measured on pi05_piper1h_ea.
    TrainConfig(
        name="pi05_piper20m_ea",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotPiperMixtureDataConfig(
            repo_id="angkul07/piper-h-teleop-v21",
            assets=AssetsConfig(asset_id="piper20m_p50"),
            base_config=DataConfig(prompt_from_task=True),
            sources=(
                MixtureSource(
                    repo_id="angkul07/piper-h-teleop-v21",
                    samples_per_batch=32,
                    root=_PIPER20M_TELEOP_ROOT,
                    # 8 of 75 episodes; NOT the 1-hour run's 15. See the note above.
                    holdout_fraction=0.1,
                    holdout_seed=0,
                ),
                MixtureSource(
                    repo_id="angkul07/piper-h-ego-v21",
                    samples_per_batch=32,
                    root=_PIPER20M_EGO_ROOT,
                    # No ego holdout: headline metrics are teleop-only by design.
                ),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=800,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=40, peak_lr=3.5e-5, decay_steps=800, decay_lr=3.5e-6
        ),
        batch_size=64,
        num_workers=16,
        save_interval=800,
        max_to_keep=1,
        keep_period=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # ---- pi05_abcego_sd: 100% teleop, single source, exactly one epoch ----
    #
    # Source: angkul07/abc-ego `put_the_screwdriver_in_the_bin`, converted from MCAP by
    # vast_run/mcap_to_lerobot.py. 2,234 episodes / 730,496 frames / 6.76 h at 30 fps.
    # Single task string, so `prompt` carries no discriminative signal -- expected for a
    # single-task finetune, but it does mean language is doing nothing here.
    #
    # Differences from the pi05_yam7h_* arms above, and why:
    #
    #   A MIXTURE OF ONE, deliberately -- not LeRobotYamDataConfig.
    #     100% teleop is one source, so the plain single-source config would be the
    #     obvious choice. It is the wrong one here for two concrete reasons:
    #       1. create_torch_dataset() hardcodes root=None on the non-mixture path, so
    #          the dataset would have to live at $HF_LEROBOT_HOME/<repo_id> and be
    #          symlinked into place. MixtureSource carries `root`, so the dataset stays
    #          wherever the converter wrote it.
    #       2. run_yam.sh stage [0] asserts `data.mixture` is non-empty and that
    #          sum(samples_per_batch) == batch_size. A single-source config fails the
    #          launcher outright.
    #     Sampling is unaffected: StratifiedBatchSampler with one source draws all 64
    #     indices from a random permutation of that source, reshuffled on wrap -- i.e.
    #     ordinary shuffled training. Its batches_per_epoch is size // 64 = 11,414,
    #     which is the same epoch this config's num_train_steps encodes.
    #
    #   The inherited repack already maps exactly the keys the converter emits
    #   (observation.images.{top,left_wrist,right_wrist} / observation.state / action),
    #   and the delta mask make_bool_mask(6, -1, 6, -1) matches the stored 14-D layout
    #   [L j0-5, L grip, R j0-5, R grip]. Nothing to override.
    #
    #   use_delta_joint_actions stays True (the default).
    #     The converter writes the raw teleop leader positions, i.e. ABSOLUTE actions.
    #
    #   exclude_episodes=() -- trains on 100% of the data, by request.
    #     So there is NO held-out split and no honest offline eval. If you want a
    #     number later, set holdout_fraction on the source (it is supported on this
    #     path) or score against a separately converted set.
    #
    #   norm stats MUST be recomputed -- do NOT copy yam7h_*.
    #     Different robot campaign, different joint distribution (rig A parks the left
    #     arm entirely, rigs B/C do not), so the q01/q99 quantiles move. Reusing the 7h
    #     stats would silently normalize against the wrong distribution:
    #       uv run scripts/compute_norm_stats.py --config-name pi05_abcego_sd \
    #              --max-frames 200000 --skip-videos
    #     --skip-videos is safe and ~100x faster: the script reads only state/actions.
    #     --max-frames is REQUIRED on the mixture path and is what run_yam.sh already
    #     passes; 200k of 730k frames is ample for stable q01/q99.
    #
    #   num_train_steps = ONE epoch, and it is tied to batch_size.
    #     730,496 frames / 64 = 11,414 exactly. len(LeRobotDataset) is the
    #     frame count (delta_timestamps clamps at episode ends rather than dropping
    #     samples), so epochs = steps * batch_size / total_frames. If you change
    #     batch_size, recompute BOTH num_train_steps and decay_steps or you silently
    #     change the epoch count. Verify against the real dataset after conversion:
    #       python -c "import json;i=json.load(open('/workspace/abc-ego-lerobot/meta/info.json'));\
    #                  print(i['total_frames'], -(-i['total_frames']//64))"
    #     At other batch sizes one epoch is: 32 -> 22,828 | 48 -> 15,219 | 96 -> 7,610.
    #
    #   warmup 500 is 4.4% of this run, against 2.1% of the 23.6k-step arms.
    #     Fine for a cosine schedule; drop to 250 if the first 500 steps look wasted.
    #
    # batch_size 64 assumes 80GB-class hardware, as measured for the pi05_yam7h arms
    # (~10.5 GB/GPU of AdamW moments + grads for the 872.8M trainable set under
    # --fsdp-devices 1). H100 SXM 80GB -- the intended target -- clears this with room
    # to spare. Halve the batch and num_train_steps doubles to stay at one epoch.
    TrainConfig(
        name="pi05_abcego_sd",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            image_augmentation=False,
        ),
        data=LeRobotYamMixtureDataConfig(
            repo_id="angkul07/abc-ego-screwdriver",
            assets=AssetsConfig(asset_id="abcego_sd"),
            base_config=DataConfig(prompt_from_task=True),
            # No augmentation on this run. This kills the ImageAugmentConfig stack
            # (ColorJitter 0.15/0.15/0.10/0.03 on every camera + 0.95-1.0 area crop on
            # the top camera) added for the 70/30 experiment. The model-side stack is
            # killed separately by image_augmentation=False above -- BOTH are needed.
            augment_config=None,
            sources=(
                MixtureSource(
                    repo_id="angkul07/abc-ego-screwdriver",
                    samples_per_batch=64,  # == batch_size: the only source
                    root=_ABCEGO_SD_ROOT,
                    exclude_episodes=(),
                ),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=11_414,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=3.5e-5, decay_steps=11_414, decay_lr=3.5e-6
        ),
        batch_size=64,
        num_workers=16,
        save_interval=1_000,
        max_to_keep=4,
        # Pin every 5,000th checkpoint permanently, so steps 5,000 and 10,000 survive
        # the rolling max_to_keep=4 window instead of being deleted by later saves.
        # Final checkpoints kept: the last 4 (8k/9k/10k/11k + the 11,413 end-of-run
        # save) PLUS pinned 5,000. The pi05_yam7h_* arms use keep_period=None.
        keep_period=5_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # ---- pi05_50run_ea: the 50/50 ego+teleop merge, single PRE-MERGED source ----
    #
    # This is the openpi counterpart of the LeRobot run in fd/sdk/lerobot_run
    # (`run_yam_lerobot.sh pi05_50_ea`). Same data, same schedule, same trainable set --
    # so the two are directly comparable, which is the point of having both.
    #
    # Source: angkul07/50_run_v21_fixed, LeRobot v2.1.
    #   4,272 episodes / 755,964 frames / 7.00 h at 30 fps, 3 cameras at 224x224.
    #   3.5 h of ego + 3.5 h of teleop, merged into ONE dataset at 50.00%/50.00% by
    #   frames. The teleop half excludes every episode in abc-teleop-holdout
    #   (249/249 matched by content hash, 0 leaks), so offline eval on that holdout
    #   is honest.
    #
    # Differences from the pi05_yam7h_* arms above, and why:
    #
    #   THE RATIO LIVES IN STORAGE, not in the sampler.
    #     The yam7h arms hold two roots and let MixtureSource draw a hard 32/32 every
    #     batch. Here the 50/50 is already baked into one dataset, so a single source
    #     drawing all 64 gives the same expected ratio. The difference is that it is
    #     50/50 IN EXPECTATION -- Binomial(64, 0.5), sd ~4 samples/batch -- rather than
    #     exact per batch. Fine in aggregate over 17.7k steps; it is also precisely the
    #     constraint that forced the LeRobot port to merge in the first place, since
    #     LeRobot has no per-source count. Keeping openpi on the merged dataset is what
    #     makes the two runs comparable; use the yam7h arms if you want exact batches.
    #
    #   STILL a mixture-of-one, for the same two mechanical reasons as pi05_abcego_sd:
    #     create_torch_dataset() hardcodes root=None off the mixture path, and
    #     run_yam.sh stage [0] asserts data.mixture is non-empty.
    #
    #   NO AUGMENTATION, both stacks off -- matches the LeRobot run, which applied none.
    #     augment_config=None kills the data-side ImageAugmentConfig; image_augmentation
    #     =False kills the model-side stack. BOTH are needed, and leaving either on
    #     would break comparability with the LeRobot numbers.
    #
    #   use_delta_joint_actions stays True (inherited default).
    #     make_bool_mask(6, -1, 6, -1) = 6 joints delta + 1 gripper absolute, per arm.
    #     This dataset is ordered [R j1-6, R grip, L j1-6, L grip] -- right arm FIRST,
    #     unlike the abc-ego set's left-first layout. The mask is symmetric across the
    #     two arms, so it is correct either way; only a mask with different per-arm
    #     structure would care. This is openpi's positional equivalent of the LeRobot
    #     side's `relative_exclude_joints=['gripper']`, which resolves BY NAME.
    #
    #   norm stats MUST be recomputed -- do NOT copy yam7h_* or abcego_*.
    #     Different mixture, different distribution, so q01/q99 move:
    #       uv run scripts/compute_norm_stats.py --config-name pi05_50run_ea \
    #              --max-frames 200000 --skip-videos
    #     run_yam.sh stage [1/3] does this for you. Quantile stats are computed AFTER
    #     data_transforms, i.e. in DELTA space -- the same space the LeRobot run
    #     normalised in, so the two are on equal footing.
    #
    #   num_train_steps = 17,700, tied to batch_size, and it is NOT a round epoch.
    #     One epoch = 755,964 / 64 = 11,812 steps. 17,700 x 64 = 1,132,800 frame-visits
    #     = 1.4985 epochs. Carried over verbatim from the LeRobot run so the two match;
    #     use 17,718 if you want exactly 1.5. Change batch_size and you MUST recompute
    #     num_train_steps AND decay_steps together (stage [0] asserts they are equal).
    #     At other batch sizes 1.5 epochs is: 32 -> 35,436 | 96 -> 11,812 | 128 -> 8,859.
    #
    #   warmup 350 is 2.0% of the run, matching the yam7h arms' 500/23,600.
    #
    # batch_size 64 with --fsdp-devices 1 = pure data parallel, 32/GPU. Measured on
    # 2x RTX PRO 6000 Blackwell (96 GB) under the PyTorch LeRobot port: 32.6 GB/GPU at
    # 32/GPU, so 80 GB-class hardware clears this comfortably. Note openpi is JAX/XLA
    # here versus eager PyTorch there, so step time will NOT match the 4.49 s/step
    # measured on the LeRobot side; only the recipe is shared.
    TrainConfig(
        name="pi05_50run_ea",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            image_augmentation=False,
        ),
        data=LeRobotYamMixtureDataConfig(
            repo_id="angkul07/50_run_v21_fixed",
            assets=AssetsConfig(asset_id="yam50run"),
            base_config=DataConfig(prompt_from_task=True),
            augment_config=None,
            sources=(
                MixtureSource(
                    repo_id="angkul07/50_run_v21_fixed",
                    samples_per_batch=64,  # == batch_size: the only source
                    root=_50RUN_ROOT,
                    # Holdout is already physically absent from this dataset (it was
                    # excluded at build time), so nothing to withhold here.
                    exclude_episodes=(),
                ),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=17_700,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=350, peak_lr=3.5e-5, decay_steps=17_700, decay_lr=3.5e-6
        ),
        batch_size=64,
        num_workers=16,
        # 1,500 matches the LeRobot run's save_freq. 11 saves over 17.7k steps; at
        # ~15-25 GB each that is 165-275 GB if you keep them all, which is why
        # max_to_keep is set. Bump max_to_keep only if the box has the disk.
        save_interval=1_500,
        max_to_keep=4,
        keep_period=5_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
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
