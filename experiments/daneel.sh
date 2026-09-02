#!/bin/sh
# Launch a plan of experiments/run.sh on Daneel, detached. Run from the host.
#
#   sh experiments/daneel.sh cond          start or resume the conditioning probes
#   sh experiments/daneel.sh pilot         start or resume the pilot
#   REG_COVAR=1e-3 sh experiments/daneel.sh pilot     pilot at a tuned reg_covar
#   COV_TYPE=diag LOG=/scratch/logs/pilot_diag sh experiments/daneel.sh pilot
#   SLICES="0:20 20:30 30:40 40:50" LOG=/scratch/logs/pilot_shard sh experiments/daneel.sh pilot
#   MUSDB=$HOME/data/gasm-demos/musdb18 MAX_TRAIN_FRAMES=150000 TEST_SECONDS=30 \
#   LOG=/scratch/logs/full256 sh experiments/daneel.sh full256
#     the full covariance arm, on the corpus and the excerpt length of the
#     diagonal arms the article reports. MAX_TRAIN_FRAMES and TEST_SECONDS are
#     forwarded rather than left at the ceiling_sweep defaults of 20000 and five
#     seconds, those defaults being the seven-second edition's and not the
#     published arm's, and MUSDB has to name the full database for the same
#     reason. A cell run at the defaults is silently a different experiment.
#   MUSDB_TEST=$HOME/data/gasm-demos/musdb18 TEST_OFFSETS=0,30,60 ... held256
#     three excerpts per track. Offsets past seven seconds need the full database,
#     hence MUSDB_TEST: the default musdb18-7s holds seven-second tracks and every
#     excerpt past the first would be skipped as too short. MUSDB, the training
#     root, is deliberately left alone: it is what fixes the dictionary, and a
#     robustness check on the excerpt has to move the excerpt and nothing else.
#   MUSDB_TEST=$HOME/data/gasm-demos/musdb18 TEST_ANCHORS=/scratch/locate_offsets.tsv \
#   TEST_OFFSETS=-30,0,30 ... held256
#     the same three excerpts per track, but counted from where that track's
#     musdb18-7s clip actually sits rather than from the head of the track. The
#     clips are not the heads: they start between 22 s and 298 s in, median 135 s,
#     so absolute offsets 0,30,60 measure intros and answer a different question.
#     Anchored, the @0 column reproduces the excerpt the paper measured and the
#     neighbours say whether the numbers survive sliding it by half a minute. The
#     TSV is produced by scratch/locate_all.sh, whose columns are the track name,
#     the anchor in seconds, then the correlation and residual that certify it.
#
# Relaunching after a stop is the same command: the plan skips what is complete
# and the EM checkpoints under scratch/checkpoints survive the container.
#
# LOG has to be given a directory of its own per arm. Cells are named by plan and
# skipped on their .done marker, so a second arm run into the same directory is
# skipped in full and hands back a complete journal that is not the arm asked for.
#
# SLICES shards a plan by test track, one container and one LOG subdirectory per
# slice, which is what makes a plan use more than one core: a cell is a single
# Python process and measured at 100 % of one. Four slices is the ceiling here,
# the box having four physical cores, and each shard then gets --cpus=1: the July
# runs put the package near 86 C at full width and near 80 C at --cpus=3.
# The slices are read inside the N_TEST selection, so shards stay comparable (the
# training pool is fixed by N_TRAIN and SEED) and their journals merge by track.
set -eu
PLAN=${1:-cond}
ROOT=${ROOT:-$HOME/data/gasm-demos}
LOG=${LOG:-/scratch/logs/ceiling}
SLICES=${SLICES:-}
MUSDB=${MUSDB:-$ROOT/musdb18-7s}
# mounted and passed only when asked for, so a plain run keeps one corpus
MUSDB_TEST=${MUSDB_TEST:-}
if [ -n "$MUSDB_TEST" ]; then
    TEST_CORPUS="-v $MUSDB_TEST:/data/musdb_test:ro -e MUSDB_TEST_ROOT=/data/musdb_test"
else
    TEST_CORPUS=""
fi
if [ -n "$SLICES" ]; then
    CPUS=${CPUS:-1}
    # A shard is one worker, so it is told so. Left alone, the BLAS pool sizes
    # itself on the eight CPUs the container sees and spends a one-CPU quota on
    # its own contention: fifteen threads were measured on a shard, and its d100
    # cell took 188 s against 64 s for the same cell in a three-CPU container.
    # Not set in single-container mode: the thread count changes the summation
    # order, and the arms already measured there are not to be perturbed.
    THREADS="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1"
    THREADS="$THREADS -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1"
else
    CPUS=${CPUS:-3}
    THREADS=""
fi

launch() {
    name=$1
    log=$2
    slice=$3
    # a stopped container still owns its name, and its logs are the record of what
    # the previous attempt did, so it is removed rather than reused
    docker rm -f "$name" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "$name" --memory=12g --cpus="$CPUS" $THREADS $TEST_CORPUS \
        -e PYTHONUNBUFFERED=1 -e MUSDB_ROOT=/data/musdb \
        -e REG_COVAR="${REG_COVAR:-1e-6}" -e COV_TYPE="${COV_TYPE:-full}" \
        -e ENERGY_PCT="${ENERGY_PCT:-0}" -e N_TRAIN="${N_TRAIN:-25}" \
        -e MAX_TRAIN_FRAMES="${MAX_TRAIN_FRAMES:-20000}" \
        -e TEST_SECONDS="${TEST_SECONDS:-5}" \
        -e TEST_SLICE="$slice" -e TEST_OFFSETS="${TEST_OFFSETS:-0}" \
        -e TEST_ANCHORS="${TEST_ANCHORS:-}" \
        -e LOG="$log" \
        -v "$ROOT/src/masking-ceiling:/w" \
        -v "$ROOT/scratch:/scratch" \
        -v "$MUSDB:/data/musdb:ro" \
        -w /w ghcr.io/astral-sh/uv:python3.13-bookworm \
        sh -c "apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1 && \
               sh experiments/run.sh $PLAN"
    echo "$name started on ${slice:-the whole selection}; follow with: docker logs -f $name"
}

if [ -z "$SLICES" ]; then
    launch "ceiling-$PLAN" "$LOG" ""
else
    i=0
    for slice in $SLICES; do
        launch "ceiling-$PLAN-s$i" "$LOG/s$i" "$slice"
        i=$((i + 1))
    done
fi
