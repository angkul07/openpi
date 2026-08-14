# `configs/` — client experiment configs

Training arms live here, **not** in `src/openpi/training/config.py`.

That file is upstream openpi's. Every arm we added to it was a line we had to
reconcile on every rebase, and by the time there were fourteen of them the reasoning
that made each one correct — epoch math, why `action_dim=32`, which norm stats may be
copied — existed only as comments wedged between the literals, invisible to `git log`
and impossible to test.

## Layout

```
configs/
  _shared/          helpers, not arms. Leading underscore = never scanned.
    arms.py         pi05_arm() / pi0_fast_arm() builders + their validation
    schedule.py     Schedule: step counts and LR schedules, computed not commented
    robots.py       RobotSpec per embodiment (YAM, PIPER_H)
  _template/        skeleton to copy for a new client
  fd/               one directory per client (fd = our own internal R&D)
    datasets.py     that client's roots, repo ids, frame counts, holdout indices
    teleop_holdout.json
    yam/            one directory per robot, when a client has more than one
      mix_7h.py     an experiment: its arms, and why they are what they are
```

Anything whose path contains a component starting with `_` is skipped by the scanner
but stays importable. That is how `_shared` and `_template` stay out of the registry
while still being usable.

## How it loads

`openpi.training.registry` walks this tree, imports every module, and each module
registers its arms:

```python
from openpi.training import registry
registry.register(my_train_config)
```

Discovery is lazy — it runs on the first `get_config()` / CLI lookup, not at import
time, because these modules import `openpi.training.config` to get `TrainConfig`.

Nothing else changes: `uv run scripts/train.py <name>`, `--config-name <name>` and
`vast_run/run_yam.sh <name>` all work exactly as before.

A module that fails to import is a **hard error**, deliberately. A config that quietly
vanishes from the CLI is how you launch the wrong arm.

The configs root is found via `$OPENPI_CONFIGS_DIR`, else `<repo>/configs`, else
`$CWD/configs`. A checkout with no `configs/` directory just works — discovery is a
no-op, which keeps a plain upstream openpi unaffected.

## Adding a client

```bash
cp -r configs/_template configs/mm
```

Then edit `configs/mm/datasets.py` with that client's roots and frame counts, and add
one module per experiment.

**Config names are global and must be unique.** Registering a name twice raises and
names both owners; shadowing a built-in openpi config raises too. Prefix client arms
with the client slug — `mm_pi05_ea`, not `pi05_ea`. (The `fd/` arms predate this rule
and keep their historical names so existing checkpoint paths, W&B history and
`assets/<config>/` directories stay valid. Do not rename them.)

## Adding an experiment

Use the builders. They exist because the same fifteen values were being retyped per
arm, and three of those repetitions were live footguns — see the docstring in
`_shared/arms.py`.

```python
from openpi.training import registry
import openpi.training.config as _config

from configs._shared.arms import pi05_arm
from configs._shared.robots import YAM
from configs._shared.schedule import Schedule
from configs.mm import datasets as ds

SCHEDULE = Schedule.for_epochs(frames=ds.TELEOP_FRAMES, batch_size=64, epochs=2.0, warmup_steps=500)

registry.register(
    pi05_arm(
        "mm_pi05_ea",
        robot=YAM,
        sources=(
            _config.MixtureSource(repo_id=ds.TELEOP_REPO, samples_per_batch=32, root=ds.TELEOP_ROOT),
            _config.MixtureSource(repo_id=ds.EGO_REPO, samples_per_batch=32, root=ds.EGO_ROOT),
        ),
        asset_id="mm_p50",
        schedule=SCHEDULE,
    )
)
```

## Adding a robot

`robot=` is a `RobotSpec`: the embodiment stated once. The repack map, the delta
action mask, the policy input/output transforms and the action width are all derived
from it, so a new robot is a data change, not a new `<name>_policy.py` plus a new
`LeRobot<Name>DataConfig` (which is what it used to be — the YAM and Piper versions of
those were ~95% identical and drifted independently).

Check `configs/_shared/robots.py` first; a spec is reusable across clients. For a
genuinely bespoke rig, copy `configs/_template/robots.py`.

```python
PIPER_H = RobotSpec(
    name="piper_h",
    cameras=(
        "observation.images.front",  # -> base_0_rgb
        "observation.images.right",  # -> left_wrist_0_rgb
        None,                        # padding slot
    ),
    arms=2, joints_per_arm=6, gripper_per_arm=True,   # -> action_dim 14, mask (6,-1,6,-1)
)
```

Two rules the spec enforces, both learned the hard way:

- **Real cameras fill the leading slots; padding trails.** `_model.IMAGE_KEYS` is
  ordered and `embed_prefix` concatenates in that order, so slot position is part of
  what pretraining learned. openpi's own two-camera policies mask only the *trailing*
  slot. `("scene", None, "wrist")` is rejected.
- **Omitting a camera drops it; masking does not.** A feature absent from `cameras`
  never enters the repack. A *masked* slot still pays full SigLIP cost — the model
  embeds every image and applies masks only to the attention mask. Neither skips the
  video decode; only not writing the key at conversion time does that.

And one thing to check before trusting any of it: **verify what the cameras actually
see.** On the Piper H rig the stored key names did not describe the content (`front`
was top-down, `top` was sideways), which was only found by extracting frames and
looking.

What the builders refuse, because each has cost real time:

- per-source draws that do not sum to `batch_size` — the sampler would only assert
  this much later, on the box;
- a mixture with no `asset_id` of its own — it falls back to `repo_id`, and two
  mixtures over the same primary dataset would then share, and silently corrupt, one
  set of norm stats;
- `decay_steps` disagreeing with `num_train_steps` — not possible, `Schedule` derives
  it. openpi defaults `decay_steps` to 30k independently of run length, so a short run
  otherwise never finishes decaying.

Put the *reasoning* in the module docstring, next to the arm. That is the whole point
of the file existing.

## Norm stats are fingerprinted

`compute_norm_stats.py` writes `norm_stats_fingerprint.json` beside `norm_stats.json`,
recording the sources, roots, per-batch draws and holdout the stats were computed
through. `_load_norm_stats` checks it and **raises** on a mismatch.

This closes the worst silent failure in the pipeline: mixture stats describe the
*sampled* distribution, `run_yam.sh` skips recomputation whenever the file already
exists, and a stats directory is just a directory — so a `cp -r` from a neighbouring
arm used to train quietly against the wrong distribution.

Stats written before fingerprinting have no such file and load with a warning rather
than an error, so existing boxes keep working. Recompute to silence it.

Copying stats between arms is still legitimate when the mixture is genuinely identical
— the `pi05_yam7h_*` arms reuse their `pi0_fast_yam7h_*` twins' stats, and the
fingerprint matches because the distribution really is the same.

## Verifying a change

`scripts/dump_configs.py` dumps a canonical, address-scrubbed form of every config,
including what `data.create()` actually produces. Dump before, dump after, diff:

```bash
uv run python scripts/dump_configs.py --out /tmp/before.json
# ... change something ...
uv run python scripts/dump_configs.py --out /tmp/after.json
diff /tmp/before.json /tmp/after.json
```

An empty diff means you changed nothing observable. This is how the original migration
out of `config.py` was proved behaviour-preserving across all 45 configs.

## Note on `run_yam.sh`

The launcher stays single and shared — it is pure logic parameterised by config name,
and forking it per experiment would be a regression. It derives the checkpoint
subdirectory by stripping a known family prefix (`pi05_yam7h_`, …) from the config
name; a new client's arms will not match, so **pass the experiment name explicitly**
as the second argument:

```bash
./vast_run/run_yam.sh mm_pi05_ea ea
```

Per-experiment *data preparation* (conversion, mixture building) does belong in the
client directory.
