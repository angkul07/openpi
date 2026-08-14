"""Invariants of the registered client arms that no single file can enforce.

The builders in `configs/_shared/arms.py` already refuse a malformed arm. What they
cannot see is a property that only holds ACROSS arms -- that three arms of one study run
the same number of steps, that they do not share norm stats, that every one of them
withholds the same eval episodes. Those are exactly the ones that break silently, months
later, when somebody edits one arm of three.

Tests are here rather than under `configs/` because `testpaths` is `src`/`scripts`/
`packages`; a test module inside the config tree would also be imported by the registry
scanner on every CLI invocation.
"""

import pytest

from openpi.training import config as _config
from openpi.training import registry


@pytest.fixture(scope="module", autouse=True)
def _discovered() -> None:
    """`configs` is only importable once discovery has put its parent on `sys.path`.

    Without this the module's outcome depends on whether some earlier test happened to
    call `get_config()` first, which is how these tests passed in a full run and failed
    on their own.
    """
    registry.discover()


# ---------------------------------------------------------------------------
# SO-101 -- the first single-arm spec, and the first with fewer than 6 joints
# ---------------------------------------------------------------------------


def test_so101_layout_is_derived_not_asserted():
    from configs._shared.robots import SO101

    # 5 arm joints + 1 gripper on ONE arm. Not 6+1, and not doubled.
    assert SO101.action_dim == 6
    # Joints become deltas, the gripper stays absolute -- the channel where cross-source
    # unit disagreement survives normalisation, hence the preflight in datasets.py.
    assert SO101.delta_action_mask() == (True, True, True, True, True, False)


def test_so101_fills_the_leading_camera_slots():
    from configs._shared.robots import SO101

    assert [feature for _, feature in SO101.real_cameras] == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    # Two real cameras, trailing slot padded -- the DROID/libero occupancy pattern.
    assert SO101.cameras[2] is None


def test_so101_chunk_matches_yam():
    from configs._shared.robots import SO101
    from configs._shared.robots import YAM

    # Both 30 Hz, so action_horizon means the same amount of future on each and the two
    # ARE comparable at equal horizon -- unlike Piper H at 20 Hz.
    assert SO101.control_hz == 30.0
    assert SO101.chunk_duration_s(50) == pytest.approx(YAM.chunk_duration_s(50))


# ---------------------------------------------------------------------------
# mm / SO-101 sim-vs-ego: three arms that only mean something together
# ---------------------------------------------------------------------------

_MM_ARMS = ("mm_pi05_sim10", "mm_pi05_mix10", "mm_pi05_mix20")


@pytest.fixture
def mm_configs() -> dict[str, _config.TrainConfig]:
    return {name: _config.get_config(name) for name in _MM_ARMS}


def test_the_three_arms_are_matched_on_everything_but_the_data(mm_configs):
    """Unequal step counts would confound exposure with schedule position.

    `decay_steps` tracks `num_train_steps`, so a longer arm also spends more of its run
    at floor LR and reports a flattered final loss. The pi0.5 mixture study could not
    rank its arms on training loss for exactly this reason.
    """
    optimisation = {
        name: (
            config.num_train_steps,
            config.batch_size,
            config.lr_schedule,
            config.model.action_horizon,
            config.model.action_dim,
        )
        for name, config in mm_configs.items()
    }
    assert len(set(map(str, optimisation.values()))) == 1, optimisation


def test_each_arm_has_its_own_norm_stats(mm_configs):
    """Three different sampled distributions cannot share one set of quantiles."""
    asset_ids = [config.data.assets.asset_id for config in mm_configs.values()]
    assert len(set(asset_ids)) == len(asset_ids), asset_ids


def test_the_mixture_arms_draw_a_hard_half_and_half(mm_configs):
    for name in ("mm_pi05_mix10", "mm_pi05_mix20"):
        config = mm_configs[name]
        draws = [source.samples_per_batch for source in config.data.sources]
        # Gradient share is the draw over the batch size, whatever is on disk.
        assert draws == [32, 32], name
        assert sum(draws) == config.batch_size

    baseline = mm_configs["mm_pi05_sim10"]
    assert [source.samples_per_batch for source in baseline.data.sources] == [64]


def test_every_arm_withholds_the_same_eval_episodes(mm_configs):
    """The half-size arm's exclusion list is holdout + dropped, merged in the manifest.

    If that merge is ever undone, the half arm trains on the episodes the other two are
    scored against and the comparison is silently contaminated -- while everything still
    runs.
    """
    from configs.mm import datasets as ds

    for name, config in mm_configs.items():
        sim = next(source for source in config.data.sources if source.repo_id == ds.SIM_REPO)
        assert set(ds.SIM_HOLDOUT_EPISODES) <= set(sim.exclude_episodes), name


def test_the_half_split_is_half_of_what_is_left_after_the_holdout():
    from configs.mm import datasets as ds

    assert set(ds.SIM_HOLDOUT_EPISODES) < set(ds.SIM_HALF_EXCLUDE_EPISODES)
    kept = ds.SIM_TOTAL_EPISODES - len(ds.SIM_HALF_EXCLUDE_EPISODES)
    assert kept == 23
    # Frames, not episodes, are what the epoch arithmetic uses; episode lengths vary by
    # ~1.5x here, so "half the episodes" is only worth having if it is also ~half the
    # frames. The stride selection lands at 51.75%.
    assert 0.45 < ds.SIM_HALF_TRAIN_FRAMES / ds.SIM_TRAIN_FRAMES < 0.55


def test_ego_is_absent_from_the_baseline_and_present_in_both_mixtures(mm_configs):
    from configs.mm import datasets as ds

    def repos(name: str) -> set[str]:
        return {source.repo_id for source in mm_configs[name].data.sources}

    assert repos("mm_pi05_sim10") == {ds.SIM_REPO}
    assert repos("mm_pi05_mix10") == {ds.SIM_REPO, ds.EGO_REPO}
    assert repos("mm_pi05_mix20") == {ds.SIM_REPO, ds.EGO_REPO}


def test_the_mixture_arms_read_norm_stats_off_the_sim_source(mm_configs):
    """`repo_id` names the primary source and must not drift to `sources[0]` by accident."""
    from configs.mm import datasets as ds

    for config in mm_configs.values():
        assert config.data.repo_id == ds.SIM_REPO
