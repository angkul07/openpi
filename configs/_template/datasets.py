"""Datasets, roots and splits for <CLIENT>.

Copy this directory to `configs/<client>/` and fill it in. Everything here is a fact
about *this client's data on our boxes* -- where it lives, how many frames it has,
which episodes are withheld. Nothing here is a fact about openpi.

Keep every root overridable by environment variable so a box that lays data out
differently needs no code change.

RECORD FRAME COUNTS. Step counts are derived from them (see
`configs/_shared/schedule.py`), so a measured number here means the schedule
recomputes itself instead of a stale comment going quietly wrong. Read them from
`meta/info.json` after the dataset is built:

    python -c "import json;print(json.load(open('<root>/meta/info.json'))['total_frames'])"
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

TELEOP_REPO = "<org>/<teleop-repo>"
EGO_REPO = "<org>/<ego-repo>"

# Local roots. None -> LeRobot resolves $HF_LEROBOT_HOME/<repo_id>.
TELEOP_ROOT = os.environ.get("CLIENT_TELEOP_ROOT", "/workspace/<client>/teleop")
EGO_ROOT = os.environ.get("CLIENT_EGO_ROOT", "/workspace/<client>/ego")

# Measured, not estimated.
TELEOP_FRAMES = 0
EGO_FRAMES = 0


# ---------------------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------------------
# Episodes withheld from training for offline eval. Nothing moves on disk -- these are
# simply never sampled.
#
# THE INDEX SPACE MATTERS. These must be indices into the dataset the arm actually
# reads. If an arm reads a pre-selected subset that was renumbered 0..N-1, indices from
# the original dataset are wrong twice over: out-of-range ones raise, and in-range ones
# silently withhold completely unrelated episodes. Leave this empty for arms whose
# holdout was already removed physically at build time.
HOLDOUT_EPISODES: tuple[int, ...] = ()
