"""The two original single-source YAM pi0-FAST arms.

These predate the mixture work: one dataset, no `MixtureSource`, so they use
`LeRobotRobotDataConfig` directly and do not go through `configs/_shared/arms.py`.
They are kept because the LoRA arm is the baseline every later number is quoted
against, not because either is a recommended starting point today.

Dataset: `Kavin60606/yam_pi0fast_train` -- 335k frames / 163 tasks, bimanual teleop,
~12 demos per task. Resolved from `$HF_LEROBOT_HOME/<repo_id>`; there is no `root`
knob on the non-mixture path.
"""

from configs._shared.arms import PI0_FAST_BASE
from configs._shared.robots import YAM
from configs._shared.schedule import Schedule
import openpi.models.pi0_fast as pi0_fast
from openpi.training import registry
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders

_REPO = "Kavin60606/yam_pi0fast_train"


def _data() -> _config.LeRobotRobotDataConfig:
    return _config.LeRobotRobotDataConfig(
        robot=YAM,
        repo_id=_REPO,
        base_config=_config.DataConfig(prompt_from_task=True),
    )


# ---- full fine-tune (needs 80GB; NOT the recommended first run for YAM) ----
_full = _config.TrainConfig(
    name="pi0_fast_yam",
    model=pi0_fast.Pi0FASTConfig(action_dim=14, action_horizon=50, max_token_len=300),
    data=_data(),
    weight_loader=weight_loaders.CheckpointWeightLoader(PI0_FAST_BASE),
    num_train_steps=30_000,
)


# ---- LoRA (low-memory) fine-tune ----
# The budget-fit baseline: ~1.3 epochs at batch 64 over 335k frames. It was
# UNDERTRAINED, not misoptimised -- grad norms were stable throughout at ~2.5 -- which
# is why every later arm changes schedule length and data, and leaves peak LR, batch
# size and the LoRA variant alone.
#
# LR was scaled up for the larger batch by the sqrt rule: 2.5e-5 * sqrt(64/32) ~= 3.5e-5.
_lora_model = pi0_fast.Pi0FASTConfig(
    action_dim=14,
    action_horizon=50,
    max_token_len=300,
    paligemma_variant="gemma_2b_lora",
)
_lora_schedule = Schedule.of_steps(7_000, warmup_steps=1_000)

_lora = _config.TrainConfig(
    name="pi0_fast_yam_low_mem_finetune",
    model=_lora_model,
    data=_data(),
    weight_loader=weight_loaders.CheckpointWeightLoader(PI0_FAST_BASE),
    num_train_steps=_lora_schedule.num_train_steps,
    lr_schedule=_lora_schedule.lr_schedule(),
    batch_size=64,  # 2x A100-80GB data-parallel -> 32 samples/GPU
    num_workers=8,  # 3-camera video decode is the loader bottleneck
    save_interval=500,  # ~7 checkpoints over the run
    freeze_filter=_lora_model.get_freeze_filter(),
    ema_decay=None,
)

registry.register(_full, _lora)
