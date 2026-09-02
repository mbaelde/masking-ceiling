#!/bin/sh
# The PC perso counterpart of daneel.sh: the same plans, outside any container,
# on the full MUSDB18 rather than the seven-second preview edition.
#
#   MAX_TRAIN_FRAMES=20000 sh experiments/local.sh pilot
#   MAX_TRAIN_FRAMES=150000 sh experiments/local.sh pilot
#   MAX_TRAIN_FRAMES=150000 sh experiments/local.sh calib
#
# Two volume arms on one corpus, because what the pilot left open is whether the
# flatness in K survives an order of magnitude more data: one point moves the
# level without controlling the slope. 20000 keeps the published cap and lands
# near 19.5 frames per dimension, 150000 lands near 146, against 14.3 today.
#
# LOG is per arm and derived from the frame cap. Cells are named fixed and
# skipped on their .done marker, so a second arm run into the first one's
# directory would be skipped whole and hand back a journal that is not the arm
# asked for. CHECKPOINT_DIR is shared: MAX_TRAIN_FRAMES is part of the
# checkpoint key, so the arms cannot collide there. What is *not* part of that
# key is the MUSDB edition, which is why these paths stay clear of the
# seven-second run's on Daneel: same key, different data, silently wrong fits.
set -eu
ROOT=${ROOT:-/d/data/gasm-demos}
REPO=${REPO:-/d/repos/masking-ceiling}
FRAMES=${MAX_TRAIN_FRAMES:-20000}
PLAN=${1:-pilot}

# N_TRAIN is the held-out cell's, the whole train split rather than the
# published 25: at a fixed frame budget the per-track quota buys timbral
# diversity for free, and the objection being answered is about coverage.
# TEST_SECONDS is six times the published five, so the paired deficit rests on
# whole sections rather than on one intro each.
WORKDIR="$REPO" \
PYTHON=${PYTHON:-$REPO/.venv/Scripts/python.exe} \
MUSDB_ROOT=${MUSDB_ROOT:-D:/data/gasm-demos/musdb18} \
LOG=${LOG:-$ROOT/scratch_local/logs/full_m$FRAMES} \
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$ROOT/scratch_local/ckpt_full} \
MAX_TRAIN_FRAMES="$FRAMES" \
TEST_SECONDS=${TEST_SECONDS:-30} \
N_TRAIN=${N_TRAIN:-100} \
COV_TYPE=${COV_TYPE:-diag} \
REG_COVAR=${REG_COVAR:-1e-6} \
ENERGY_PCT=${ENERGY_PCT:-0} \
PYTHONUNBUFFERED=1 \
    sh "$REPO/experiments/run.sh" "$PLAN"
