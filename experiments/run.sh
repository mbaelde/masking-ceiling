#!/bin/sh
# Every cell of plan_exp_article1_plafond.md, resumable, one plan per invocation.
#
#   sh experiments/run.sh cond     conditioning probes, ~1 h, configures the pilot
#   sh experiments/run.sh pilot    the GO/NO-GO pilot, ~7 h
#   sh experiments/run.sh full256  the fourth rung, full covariance at nfft = 256
#   sh experiments/run.sh held256  the published diagonal nfft = 256 row, re-measured
#   sh experiments/run.sh classes  the three ceilings of the real-mask class, no fit
#   sh experiments/run.sh calib    the prior calibration gate, ~1 h, no fit
#   sh experiments/run.sh offsupport  the off-support prediction, no fit, minutes
#
# The Def-MAP probes of the companion article live in their own repository,
# https://github.com/mbaelde/defmap-repair, and are driven by the same runner
# there.
#
# Resumption is the point. A killed run is relaunched with the same command: a
# probe that finished is skipped on its .done marker, a probe that was cut open
# reads its own log back and recomputes only the (K, track) cells missing from
# it, and the EM fit behind those cells restarts from its last checkpoint rather
# than from scratch. Nothing is recomputed except what was actually lost.
#
# The pilot's REG_COVAR and COV_TYPE are meant to be set from what `cond`
# measured, hence the environment overrides rather than hardcoded values.
set -u
# /w is where daneel.sh mounts the repo; a run outside the container gives its own
# WORKDIR, its own PYTHON, and its own LOG and CHECKPOINT_DIR
cd "${WORKDIR:-/w}"
PYTHON=${PYTHON:-uv run python}
PLAN=${1:-cond}
LOG=${LOG:-/scratch/logs/ceiling}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/scratch/checkpoints}
export CHECKPOINT_DIR
# Which script the plan drives. A probe that only re-reads the fits shares the
# whole resume protocol below and differs from the sweep in its entry point
# alone, so the entry point is the variable rather than the function.
SCRIPT=${SCRIPT:-experiments/ceiling_sweep.py}
mkdir -p "$LOG" "$CHECKPOINT_DIR"

run() {
    name=$1
    shift
    if [ -f "$LOG/$name.done" ]; then
        echo "=== $name already complete, skipped ===" >&2
        return
    fi
    echo "=== $name : $* ===" >&2
    date >&2
    # append, and hand the log back as RESUME_FROM so finished cells are not redone
    # shellcheck disable=SC2086
    RESUME_FROM="$LOG/$name.jsonl" \
        env "$@" $PYTHON "$SCRIPT" \
        >>"$LOG/$name.jsonl" 2>>"$LOG/$name.err" || {
        echo "=== $name interrupted, relaunch the same command to resume ===" >&2
        exit 1
    }
    touch "$LOG/$name.done"
    echo "=== $name complete, $(wc -l <"$LOG/$name.jsonl") rows ===" >&2
}

# Which knob governs the covariance degeneracy at nfft = 1024? The stacked
# dimension is 1026 and musdb18-7s gives ~429 frames per track, so the frames
# per component barely exceed the dimension and a full covariance is close to
# singular. Five probes at K = 8, everything else equal.
plan_cond() {
    base="CORPUS=musdb REGIME=unseen NFFT=1024 K=8 N_TRAIN=25 N_TEST=3 MAX_ITER=10"
    # shellcheck disable=SC2086
    run cond_reg_1e-6 $base REG_COVAR=1e-6
    # shellcheck disable=SC2086
    run cond_reg_1e-3 $base REG_COVAR=1e-3
    # shellcheck disable=SC2086
    run cond_reg_1e-1 $base REG_COVAR=1e-1
    # shellcheck disable=SC2086
    run cond_diag $base COV_TYPE=diag
    # shellcheck disable=SC2086
    run cond_energy50 $base ENERGY_PCT=50
    # shellcheck disable=SC2086
    run cond_ntrain50 $base N_TRAIN=50
}

# The pilot. Only this decides GO / NO-GO, and only the K = 32 row can: the
# preliminary measurement puts the crossing there and nowhere below.
plan_pilot() {
    # no NFFT in the shared part: env applies assignments in order and relying on
    # a later one to win is the kind of thing that silently runs the wrong cell
    base="CORPUS=musdb K=8,16,32 MAX_ITER=30 REG_COVAR=${REG_COVAR:-1e-6}"
    base="$base COV_TYPE=${COV_TYPE:-full} ENERGY_PCT=${ENERGY_PCT:-0}"
    # in support first: the decision table makes this the only cell that can
    # produce a NO-GO, and a NO-GO makes the unseen cell pointless. Ten tracks
    # rather than five because the two sources carry one measurement and not two,
    # their estimates summing to the mixture, so the paired bootstrap has as many
    # samples as there are test tracks and five of them exclude nothing.
    # shellcheck disable=SC2086
    run pilot_support $base NFFT=1024 REGIME=in_support N_TRAIN=10 N_TEST=10
    # shellcheck disable=SC2086
    run pilot_unseen $base NFFT=1024 REGIME=unseen N_TRAIN="${N_TRAIN:-25}" N_TEST=10
    # continuity control: the same at nfft = 256, to show that the STFT size is
    # what moved the result and not the change of corpus
    # shellcheck disable=SC2086
    run pilot_nfft256 $base NFFT=256 REGIME=unseen N_TRAIN=5 N_TEST=5
}

# The fourth rung of the ladder, which no arm of the article exercises. Every
# published measurement is COV_TYPE=diag, hence third rung, while the only
# crossing ever observed in this project was a full covariance at nfft = 256 on
# two stems in support. That crossing is either reproducible at scale or it was
# memorisation, and nothing short of this plan says which.
#
# nfft = 256 and nothing else: the stacked dimension is 258 there against 1026 at
# nfft = 1024, so a full covariance costs 33 411 free parameters per component
# instead of 527 001, and the large training pool affords ~580 frames per
# dimension rather than ~145. At nfft = 1024 the full covariance is out of
# arithmetic reach and the plan does not pretend otherwise.
#
# MAX_TRAIN_FRAMES, TEST_SECONDS and N_TRAIN are left to the caller exactly as
# the pilot leaves them, so that the wrapper which fixed the diagonal arm fixes
# this one identically: a row here has to read against its diagonal counterpart,
# and a knob set twice in two places is how two arms stop being comparable.
plan_full256() {
    base="CORPUS=musdb K=8,16,32 MAX_ITER=30 REG_COVAR=${REG_COVAR:-1e-6}"
    base="$base COV_TYPE=full ENERGY_PCT=${ENERGY_PCT:-0} NFFT=256"
    # Five training and five test tracks in both cells, which is pilot_nfft256's
    # selection to the letter: that cell is the published nfft = 256 row and the
    # only thing this plan is allowed to change is the covariance structure.
    # In support first, as in the pilot, it being the cell that reproduces or
    # refutes the historical crossing.
    # shellcheck disable=SC2086
    run full256_support $base REGIME=in_support N_TRAIN=5 N_TEST=5
    # shellcheck disable=SC2086
    run full256_unseen $base REGIME=unseen N_TRAIN=5 N_TEST=5
}

# The published diagonal nfft = 256 row, re-measured. Its journal is gone: the
# only surviving nfft = 256 diagonal log is pilot_diag/pilot_nfft256.jsonl, run on
# the seven-second preview corpus at the default caps, and its deficits read
# -7.4 dB against the -9.8 dB the article's table carries. The two are therefore
# not the same measurement, and the table's is the one without a log.
#
# The cell is pilot_nfft256's to the letter but for the corpus and the caps,
# which are the caller's exactly as in plan_full256, so that the same wrapper
# fixes this arm and the full covariance arm identically. Run it twice, once per
# training cap, each into a LOG of its own:
#
#   MUSDB=$HOME/data/gasm-demos/musdb18 MAX_TRAIN_FRAMES=20000 TEST_SECONDS=30 \
#   COV_TYPE=diag LOG=/scratch/logs/held256_m20k sh experiments/daneel.sh held256
#   ... MAX_TRAIN_FRAMES=150000 LOG=/scratch/logs/held256_m150k ...
plan_held256() {
    base="CORPUS=musdb K=8,16,32 MAX_ITER=30 REG_COVAR=${REG_COVAR:-1e-6}"
    base="$base COV_TYPE=${COV_TYPE:-diag} ENERGY_PCT=${ENERGY_PCT:-0} NFFT=256"
    # shellcheck disable=SC2086
    run held256_unseen $base REGIME=unseen N_TRAIN=5 N_TEST=5
    # The diagonal in-support cell, which no plan produced until now. Without it
    # the covariance structure and the evaluation regime are each measured at the
    # other held fixed, and their crossing is not measured at all.
    # shellcheck disable=SC2086
    run held256_support $base REGIME=in_support N_TRAIN=5 N_TEST=5
}

# The three ceilings of the real-mask class, side by side. Corollary 2 gives them
# in closed form, so nothing here is fitted and ORACLES_ONLY stops each cell after
# the oracle pass: the whole plan is minutes rather than hours.
#
# The same oracle rows now carry the operator cascade M1 c M2 c M3 c M4, read with
# `report_ceiling.py --cascade`, so this plan feeds two tables and no second plan
# exists. One caveat belongs to the reader rather than to the runner: M4 spends 4F
# real parameters per bin against 2T real observations, so at NFFT=1024 and 30 s
# the corrected column is a 1.7 dB correction and the raw one is not usable alone,
# while at NFFT=256 the ratio is comfortable. Both columns are printed for that
# reason.
#
# The two cells reproduce the excerpt selection of the two rows the article
# reports, pilot_unseen at nfft = 1024 and held256_unseen at nfft = 256, because a
# ceiling is a statistic of the test excerpts alone and a table that mixes
# selections compares nothing. N_TRAIN is carried along untouched for the same
# reason: under REGIME=unseen it is what the test selection starts after.
# MUSDB and TEST_SECONDS are the caller's, exactly as in plan_full256:
#
#   MUSDB=$HOME/data/gasm-demos/musdb18 TEST_SECONDS=30 \
#   LOG=/scratch/logs/classes sh experiments/daneel.sh classes
plan_classes() {
    base="CORPUS=musdb REGIME=unseen K=8 MAX_ITER=30 ORACLES_ONLY=1"
    # shellcheck disable=SC2086
    run classes_1024 $base NFFT=1024 N_TRAIN="${N_TRAIN:-25}" N_TEST=10
    # shellcheck disable=SC2086
    run classes_256 $base NFFT=256 N_TRAIN=5 N_TEST=5
    # the nfft = 1024 counterpart of the cell above, on its five-excerpt selection
    # rather than the ten of classes_1024: the article's frame-length comparison is
    # made on that subset, and the subclass ceilings have to move with the frame
    # length on the same excerpts or the displacement is a change of corpus
    # shellcheck disable=SC2086
    run classes_1024_five $base NFFT=1024 N_TRAIN=5 N_TEST=5
}

# The off-support prediction, on the pilot's own fits. The environment is
# pilot_support's to the letter for the reason plan_calib gives, the checkpoint key
# being built from it, and K is 32 alone: the article's off-support reading is made
# at the pilot's operating point and the two smaller values would only add rows
# that no section quotes.
#
# Not resumable, unlike every other cell here: a path is cheap enough that the
# whole cell is minutes, and the script appends rather than reading its own
# journal back, so a killed run leaves rows that a relaunch would duplicate.
# Delete the .jsonl before relaunching.
plan_offsupport() {
    SCRIPT=experiments/offsupport_path.py
    base="CORPUS=musdb K=32 REG_COVAR=${REG_COVAR:-1e-6}"
    base="$base COV_TYPE=diag ENERGY_PCT=${ENERGY_PCT:-0}"
    # shellcheck disable=SC2086
    run offsupport_support $base NFFT=1024 REGIME=in_support N_TRAIN=10 N_TEST=3
}

# The gate on a Student prior. Proposition 6 already caps that route, since a
# random scale per component leaves Propositions 2, 3, 5 and Lemma 4 word for
# word, so B.1 can only narrow the deficit and never cross the ceiling. What
# decides whether it can narrow it at all is whether the deficit is variance the
# model predicts, and whether that variance sits inside a component pair, which
# a scale mixture replaces, or across pairs, which it leaves untouched. The
# environment is pilot_support's to the letter, because the checkpoint key is
# built from it and one wrong value silently fits nothing and reads no file.
plan_calib() {
    SCRIPT=experiments/prior_calibration.py
    # COV_TYPE is fixed rather than defaulted: daneel.sh passes full into the
    # container, so an override here can only ever be the wrong value, and the
    # probe exists solely for the diagonal arm.
    base="CORPUS=musdb K=8,16,32 REG_COVAR=${REG_COVAR:-1e-6}"
    base="$base COV_TYPE=diag ENERGY_PCT=${ENERGY_PCT:-0}"
    # shellcheck disable=SC2086
    run calib_support $base NFFT=1024 REGIME=in_support N_TRAIN=10 N_TEST=10
}

case "$PLAN" in
    cond) plan_cond ;;
    pilot) plan_pilot ;;
    full256) plan_full256 ;;
    held256) plan_held256 ;;
    classes) plan_classes ;;
    calib) plan_calib ;;
    offsupport) plan_offsupport ;;
    *) echo "unknown plan: $PLAN (cond | pilot | full256 | held256 | classes | calib | offsupport)" >&2; exit 2 ;;
esac

echo "=== plan $PLAN done ===" >&2
date >&2
