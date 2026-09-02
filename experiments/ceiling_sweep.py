"""Does the DM-GMM prior cross the ceiling of the real-mask class, and where?

The article's one surviving claim (plan_article1_dmgmm.md, C3') is that a
non-circular Gaussian mixture prior on [Re ; Im] produces an estimator that is
not a mask, and that it beats the exact mask-class ceiling m* rather than the
IRM. The preliminary measurement of 2026-08-19 (nfft = 256, two stems, in
support) gave 1.5 dB of margin over m*, down from the 7.7 dB that comparing
against the IRM alone had suggested. This script is what turns that measurement
into a result: same metrics, at the thesis STFT size, on a corpus, and out of
support, where the margin is expected to collapse.

Everything is set by environment variables so one image runs every cell of the
plan, and every row lands as one JSON object per line on stdout, so a log is
directly a dataframe.

    CORPUS      stems | musdb    stems replays 2026-08-19 and is the control point
    REGIME      in_support | unseen
    NFFT, HOP   1024 / 512 by default, the thesis values
    K           comma-separated component counts, one fit each
    COV_TYPE    full | diag      diag still escapes the mask class, it only drops
                                 the cross-frequency coupling
    N_TRAIN, N_TEST, SECONDS, TEST_SECONDS, MAX_ITER, REG_COVAR
    MAX_TRAIN_FRAMES  total training frames, spent as an equal quota per track so
                      that peak memory is one track's spectra and not the corpus'
    ENERGY_PCT  drop training frames below this energy percentile (0 = keep all)
    CHECKPOINT_DIR    where to persist the EM state, "" disables it
    CHECKPOINT_EVERY  EM iterations between two saves
    RESUME_FROM       a previous log of this same cell, whose rows are not recomputed

A fit is the expensive state: 832 s for ten iterations at K = 8 and 14 625
frames, so the pilot's K = 32 cell runs for hours. With CHECKPOINT_DIR set, the
fit is run in chunks of CHECKPOINT_EVERY iterations through scikit-learn's
`warm_start` and saved after each, so a killed run resumes at the last chunk
instead of the start. The iteration count lives in the file rather than the key,
so a fit stopped at ten iterations is *extended* to thirty by a later run at a
higher MAX_ITER instead of being thrown away.

The control point first, on the machine that produced the preliminary numbers:

    CORPUS=stems REGIME=in_support NFFT=256 HOP=128 K=8 python experiments/ceiling_sweep.py

It must return drums/bass within 0.2 dB of revue_litterature.md section 10.4.
Nothing measured at nfft = 1024 means anything until it does.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from evaluation import (
    CLASS_BOUNDS,
    ORACLES,
    analyze,
    cascade_residuals,
    class_witnesses,
    oracle_spectra,
    sdr,
    si_sdr,
    spectral_ceiling,
    synthesize,
)
from gasm.rase.dmgmm import DMGMMSeparator, SourceGMM, _stack_real_imag

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
# One training track, decoded on call. Training audio is handed over as closures
# rather than arrays because a full-length MUSDB18 track is 40 MB of float64 per
# source and `_training_frames` keeps a few hundred of its frames.
Loader = Callable[[], FloatArray]

CORPUS = os.environ.get("CORPUS", "musdb")
REGIME = os.environ.get("REGIME", "unseen")
NFFT = int(os.environ.get("NFFT", "1024"))
HOP = int(os.environ.get("HOP", str(NFFT // 2)))
K_VALUES = tuple(int(k) for k in os.environ.get("K", "8,16,32").split(","))
COV_TYPE = os.environ.get("COV_TYPE", "full")
N_TRAIN = int(os.environ.get("N_TRAIN", "25"))
N_TEST = int(os.environ.get("N_TEST", "10"))
TEST_SLICE = os.environ.get("TEST_SLICE", "")
SECONDS = float(os.environ.get("SECONDS", "20"))
TEST_SECONDS = float(os.environ.get("TEST_SECONDS", "5"))
# Where in the track the test excerpt starts, in seconds, one excerpt per value.
# The default reproduces the single head excerpt the July and August runs measured.
TEST_OFFSETS = tuple(float(o) for o in os.environ.get("TEST_OFFSETS", "0").split(","))
# Optional TSV, one row per track, whose first two columns are a track name and a
# per-track anchor in seconds. An offset is then counted from that anchor instead
# of from the head of the track, so offset 0 is the anchor itself.
TEST_ANCHORS = os.environ.get("TEST_ANCHORS", "")
MAX_TRAIN_FRAMES = int(os.environ.get("MAX_TRAIN_FRAMES", "20000"))
MAX_ITER = int(os.environ.get("MAX_ITER", "30"))
REG_COVAR = float(os.environ.get("REG_COVAR", "1e-6"))
ENERGY_PCT = float(os.environ.get("ENERGY_PCT", "0"))
SEED = int(os.environ.get("SEED", "0"))
ORACLES_ONLY = bool(os.environ.get("ORACLES_ONLY", ""))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "")
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "5"))
RESUME_FROM = os.environ.get("RESUME_FROM", "")

# The `stems` corpus is the two-source control of 2026-08-19, run on a private
# multitrack that is not redistributable. Nothing in the paper rests on it, every
# published number coming from MUSDB18, so both variables are left unset here:
# point STEM_DIR at a directory holding two mono or stereo files and name them in
# STEM_FILES to reproduce the control on other material.
STEM_DIR = os.environ.get("STEM_DIR", "")
STEM_FILES = tuple(f for f in os.environ.get("STEM_FILES", "").split(",") if f)
STEM_NAMES = ("drums", "bass")
_SILENCE_FLOOR = 1e-3  # of the mixture's peak magnitude, below which a gain is noise


def _mono(audio: NDArray[np.float64]) -> FloatArray:
    return np.asarray(audio.mean(axis=-1) if audio.ndim > 1 else audio, dtype=np.float64)


def _ready(signal: FloatArray) -> Loader:
    """A signal already in memory, wrapped in the loader interface."""
    return lambda: signal


def _load_stems(rate: int = 44100) -> tuple[list[list[Loader]], list[tuple[str, list[FloatArray]]]]:
    """The two stems of 2026-08-19. `in_support` trains on `SECONDS` and tests on
    the first `TEST_SECONDS` of that same segment, which is what the preliminary
    sweep did and the whole point of the control: it reproduces the optimistic
    number rather than a fair one."""
    import soundfile as sf

    if not STEM_DIR or not STEM_FILES:
        raise SystemExit("CORPUS=stems needs STEM_DIR and a comma-separated STEM_FILES; the original two stems are private, see the note next to STEM_DIR")

    train: list[FloatArray] = []
    test: list[FloatArray] = []
    for name in STEM_FILES:
        audio, sample_rate = sf.read(os.path.join(STEM_DIR, name))
        signal = _mono(audio)
        start = int(SECONDS * sample_rate)
        length = int(TEST_SECONDS * sample_rate)
        train.append(signal[:start])
        test.append(signal[:length] if REGIME == "in_support" else signal[start : start + length])
    return [[_ready(s)] for s in train], [("stems", test)]


def _load_musdb() -> tuple[list[list[Loader]], list[tuple[str, list[FloatArray]]]]:
    """Vocals versus accompaniment, the split the July 2026 sweep used, so the
    two are comparable. `in_support` evaluates on the training tracks themselves
    and is only there to bracket the generalization gap, not to be reported alone."""
    import musdb

    root = os.environ.get("MUSDB_ROOT")
    train_db = musdb.DB(root=root, subsets="train", download=root is None)
    train_tracks = train_db.tracks[:N_TRAIN]
    train: list[list[Loader]] = [[], []]
    for track in train_tracks:
        # one closure per track, not its decoded audio: on the full database this
        # list would otherwise hold 25 four-minute tracks twice over, several GB
        # of samples of which `_training_frames` keeps a few hundred frames
        train[0].append(lambda t=track: _mono(t.targets["vocals"].audio))
        train[1].append(lambda t=track: _mono(t.audio) - _mono(t.targets["vocals"].audio))

    if REGIME == "in_support":
        test_tracks = train_tracks[:N_TEST]
    else:
        # a root of its own for the test subset, because excerpts past seven seconds
        # need the full database while the training pool has to stay the one already
        # measured: on musdb18-7s the per-track quota of `MAX_TRAIN_FRAMES // N_TRAIN`
        # is never reached, on full tracks it would be, and the dictionary would then
        # change along with the excerpt. Defaults to `root`, so nothing moves unasked.
        test_root = os.environ.get("MUSDB_TEST_ROOT") or root
        test_tracks = musdb.DB(
            root=test_root, subsets="test", download=test_root is None
        ).tracks[:N_TEST]
    if TEST_SLICE:
        # `a:b` inside the N_TEST selection, so one plan can be sharded across
        # containers without changing what any of them measures: the training pool
        # is fixed by N_TRAIN and SEED, and a journal row names its own track.
        first, _, last = TEST_SLICE.partition(":")
        test_tracks = test_tracks[int(first or 0) : int(last) if last else None]
    test: list[tuple[str, list[FloatArray]]] = []
    for track in test_tracks:
        voice = _mono(track.targets["vocals"].audio)
        test.extend(_excerpts(track.name, voice, _mono(track.audio) - voice, track.rate))
    return train, test


@functools.lru_cache(maxsize=1)
def _anchors() -> dict[str, float]:
    """Track name to anchor in seconds, from `TEST_ANCHORS`, empty when unset.

    A track missing from the file anchors at 0, which is the head of the track:
    the file decides which tracks move, so a partial file is a partial control
    and not an error. Extra columns are ignored, so the localisation report can
    keep its correlation and residual beside the offset it certifies.
    """
    if not TEST_ANCHORS:
        return {}
    out: dict[str, float] = {}
    with open(TEST_ANCHORS, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for line in lines:
        cols = line.split("\t")
        if len(cols) >= 2 and cols[0]:
            out[cols[0]] = float(cols[1])
    assert out, f"TEST_ANCHORS={TEST_ANCHORS} named no track"
    return out


def _excerpts(
    name: str, voice: FloatArray, accompaniment: FloatArray, rate: int
) -> list[tuple[str, list[FloatArray]]]:
    """One test item per entry of `TEST_OFFSETS`, cut out of one track.

    The offset goes in the item's name, not beside it: both reporters key an
    oracle by (probe, track, source) and screen silence by (probe, track), so two
    excerpts of one track have to be two names or the second pairs against the
    first's oracle. A single offset keeps the bare name, which is what makes the
    journals already measured comparable and resumable against a new run.

    With `TEST_ANCHORS`, offsets count from that track's anchor rather than from
    its head, so a run can slide around a chosen excerpt instead of around the
    intro. Offset 0 then IS the anchored excerpt, which is what lets a new run
    reproduce an old one on its `@0` column while the neighbours say whether the
    numbers hold when the excerpt moves.
    """
    length = int(TEST_SECONDS * rate)
    anchor = _anchors().get(name, 0.0)
    items = []
    for offset in TEST_OFFSETS:
        start = int((anchor + offset) * rate)
        if start < 0 or start + length > len(voice):
            continue  # the excerpt falls outside the track
        label = name if len(TEST_OFFSETS) == 1 else f"{name}@{offset:g}"
        cut = slice(start, start + length)
        items.append((label, [voice[cut], accompaniment[cut]]))
    return items


def _training_frames(
    loaders: list[Loader], rng: np.random.Generator, quota: int | None = None
) -> ComplexArray:
    """Frames of one source, pooled over its training tracks, filtered and capped.

    The cap is spent as a quota per track rather than over the pool, so a track
    is decoded, framed, sampled from and dropped before the next one is touched.
    Pooling first is what the seven-second edition allowed and the full database
    does not: 25 four-minute tracks are half a million frames, four GB of spectra
    per source to draw twenty thousand rows from, and the container has twelve.

    The quota also changes the sample rather than only the peak memory. Over
    tracks of equal length it draws the same number of frames from the same
    distribution, which is the case on the seven-second edition; over unequal
    lengths it is a stratified draw where the pooled version was proportional to
    duration, so a long track no longer dominates the fit. Neither draw is
    reproducible frame for frame from the other, only distributionally.

    The energy filter is off by default. It is exposed because a GMM spends
    components on whatever is most frequent, and MUSDB vocal stems are silent for
    much of a track: if a K = 8 fit reads as component collapse in the pilot,
    this is the first knob, not reg_covar. It is applied per track for the same
    reason as the quota, so its percentile is now a within-track one.
    """
    if quota is None:
        quota = max(1, MAX_TRAIN_FRAMES // len(loaders))
    kept: list[ComplexArray] = []
    for load in loaders:
        spectra = analyze(load(), NFFT, HOP)
        if ENERGY_PCT > 0:
            energy = (np.abs(spectra) ** 2).sum(axis=-1)
            spectra = spectra[energy >= np.percentile(energy, ENERGY_PCT)]
        if len(spectra) > quota:
            spectra = spectra[rng.permutation(len(spectra))[:quota]]
        kept.append(spectra)
    return np.concatenate(kept)


def _checkpoint_path(index: int, n_components: int) -> str:
    """One file per fit, keyed by everything that changes what is being fitted.

    MAX_ITER is deliberately absent from the key: how far EM has gone is state,
    not identity, so raising MAX_ITER continues an existing fit rather than
    starting a competing one. Changing the corpus, the STFT, the covariance
    structure or the seed does change identity and does take a new file.
    """
    if not CHECKPOINT_DIR:
        return ""
    key = "_".join([
        CORPUS, REGIME, f"n{NFFT}", f"t{N_TRAIN}", COV_TYPE, f"r{REG_COVAR:g}",
        f"e{ENERGY_PCT:g}", f"m{MAX_TRAIN_FRAMES}", f"s{SEED}",
        f"k{n_components}", f"src{index}",
    ])
    return os.path.join(CHECKPOINT_DIR, key + ".npz")


def _restore(gmm: GaussianMixture, path: str) -> int:
    """Load an interrupted fit back into `gmm`, returning the iterations already done.

    Setting `converged_` is what makes `warm_start` skip initialization, so it has
    to be restored even when it is False, and `precisions_cholesky_` has to come
    from the file rather than be recomputed: it is the parameterization scikit-learn
    actually runs EM on.
    """
    if not path or not os.path.exists(path):
        return 0
    state = np.load(path)
    gmm.weights_ = state["weights"]
    gmm.means_ = state["means"]
    gmm.covariances_ = state["covariances"]
    gmm.precisions_cholesky_ = state["precisions_cholesky"]
    gmm.lower_bound_ = float(state["lower_bound"])
    gmm.converged_ = bool(state["converged"])
    return int(state["n_iter"])


def _fit_source(spectra: ComplexArray, n_components: int, max_iter: int, path: str = "") -> SourceGMM:
    """`SourceGMM.fit` hardcodes full covariances and the default reg_covar, so
    the ablation refits here. A diagonal fit is widened back into full matrices
    rather than forked into the estimator: the point of the ablation is which
    covariance *structure* is needed, not which code path is faster.

    With `path` set the run is chunked and saved, so it survives being killed.
    `warm_start` carries `lower_bound_` across chunks, so the convergence test is
    the same one an uninterrupted fit would apply and the result is identical,
    which `_check_resume` asserts rather than assumes.
    """
    data = _stack_real_imag(spectra)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=COV_TYPE,
        random_state=SEED,
        max_iter=max_iter,
        reg_covar=REG_COVAR,
        warm_start=bool(path),
    )
    done = _restore(gmm, path)
    while done < max_iter and not getattr(gmm, "converged_", False):
        gmm.max_iter = min(CHECKPOINT_EVERY, max_iter - done) if path else max_iter
        gmm.fit(data)
        done += gmm.n_iter_
        if path:
            # written aside and renamed: a kill lands on the run far more often than
            # on any other moment, and a checkpoint truncated mid-write is worse than
            # no checkpoint at all, since the next run reads it back as a valid fit.
            # np.savez appends .npz to a name that lacks it, hence the suffix order.
            tmp = path + ".tmp.npz"
            np.savez(
                tmp, weights=gmm.weights_, means=gmm.means_, covariances=gmm.covariances_,
                precisions_cholesky=gmm.precisions_cholesky_, lower_bound=gmm.lower_bound_,
                converged=gmm.converged_, n_iter=done,
            )
            os.replace(tmp, path)
    covariances = gmm.covariances_
    if COV_TYPE == "diag":
        covariances = np.stack([np.diag(v) for v in covariances])
    return SourceGMM(gmm.weights_, gmm.means_, covariances)


def _score(
    references: list[FloatArray], estimates: list[ComplexArray], mixture: ComplexArray
) -> list[dict[str, float]]:
    """Per source: SDR, SI-SDR, and the three witnesses that it left the mask class."""
    floor = _SILENCE_FLOOR * float(np.abs(mixture).max())
    rows: list[dict[str, float]] = []
    for reference, estimate in zip(references, estimates):
        signal = synthesize(estimate, NFFT, HOP)
        row = {"sdr": sdr(reference, signal, NFFT), "si_sdr": si_sdr(reference, signal, NFFT)}
        row.update(class_witnesses(estimate, mixture, floor))
        rows.append(row)
    return rows


def _completed(path: str) -> set[tuple[object, ...]]:
    """Which (method, K, track) cells an earlier run of this same probe already wrote.

    Rows are appended, so a relaunch reads its own output back and skips what is
    there. The last line of a killed run is often half-written, hence the
    tolerance: a row that does not parse never counts as done and is simply
    recomputed.
    """
    if not path or not os.path.exists(path):
        return set()
    done: set[tuple[object, ...]] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(("oracle", row["track"]) if row["method"] == "oracle"
                     else ("dmgmm", row["k"], row["track"]))
    return done


def main() -> None:
    rng = np.random.default_rng(SEED)
    train_loaders, test_items = _load_stems() if CORPUS == "stems" else _load_musdb()
    # decoding the training pool is the bulk of a run and an oracles-only pass has
    # no use for it, the test selection being fixed by N_TRAIN and not by the audio
    train_spectra = (
        [] if ORACLES_ONLY
        else [_training_frames(loaders, rng) for loaders in train_loaders]
    )
    config = {
        "corpus": CORPUS, "regime": REGIME, "nfft": NFFT, "hop": HOP, "cov_type": COV_TYPE,
        "n_train": N_TRAIN, "max_iter": MAX_ITER, "reg_covar": REG_COVAR,
        "energy_pct": ENERGY_PCT, "seed": SEED,
        "train_frames": [int(len(s)) for s in train_spectra],
    }

    done = _completed(RESUME_FROM)

    # oracles are independent of K, so they are emitted once per test item
    for name, references in test_items:
        if ("oracle", name) in done:
            continue
        spectra = [analyze(reference, NFFT, HOP) for reference in references]
        mixture = np.sum(spectra, axis=0)
        for index, reference in enumerate(references):
            estimates = oracle_spectra(spectra, index)
            row: dict[str, object] = {
                **config, "track": name, "source": index, "method": "oracle",
                # one analytic ceiling per subclass of the real-mask class; the
                # unconstrained one keeps its historical key, the readers use it
                "spectral_ceiling": spectral_ceiling(spectra, index),
                **{
                    f"ceiling_{oracle}": spectral_ceiling(spectra, index, bounds)
                    for oracle, bounds in CLASS_BOUNDS.items()
                    if bounds is not None
                },
                # the other reading of the ceiling: one operator fixed across the
                # frames of this excerpt, projected on each rung of the chain
                **cascade_residuals(spectra, index),
            }
            for oracle in ORACLES:
                signal = synthesize(estimates[oracle], NFFT, HOP)
                row[f"sdr_{oracle}"] = sdr(reference, signal, NFFT)
                row[f"si_sdr_{oracle}"] = si_sdr(reference, signal, NFFT)
            print(json.dumps(row), flush=True)

    # the ceilings and the oracles do not depend on the model, so a run that only
    # needs them can stop here instead of paying for a fit at every K
    if ORACLES_ONLY:
        return

    for n_components in K_VALUES:
        pending = [item for item in test_items if ("dmgmm", n_components, item[0]) not in done]
        if not pending:
            continue
        start = time.time()
        separator = DMGMMSeparator([n_components] * len(train_spectra))
        separator.sources_ = [
            _fit_source(s, n_components, MAX_ITER, _checkpoint_path(index, n_components))
            for index, s in enumerate(train_spectra)
        ]
        # on a resumed run this counts only the chunks actually run now, so it
        # reads low; the honest total is the sum over the runs that produced the fit
        fit_seconds = time.time() - start

        for name, references in pending:
            spectra = [analyze(reference, NFFT, HOP) for reference in references]
            mixture = np.sum(spectra, axis=0)
            start = time.time()
            estimates = separator.predict(mixture)
            predict_seconds = time.time() - start
            for index, scores in enumerate(_score(references, estimates, mixture)):
                print(json.dumps({
                    **config, "track": name, "source": index, "method": "dmgmm",
                    "k": n_components, "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "frames": int(len(mixture)), **scores,
                }), flush=True)


def _check_resume() -> None:
    """An interrupted-then-resumed fit must equal the uninterrupted one.

    This is the whole claim of the checkpointing, and it is not obvious: it holds
    only because `warm_start` carries the parameters *and* the lower bound, so the
    chunk boundary is invisible to EM. Run it after any scikit-learn upgrade, the
    restored attributes are that library's internals and nothing guarantees them
    across versions.
    """
    rng = np.random.default_rng(0)
    spectra = np.asarray(
        rng.normal(size=(400, 17)) + 1j * rng.normal(size=(400, 17)), dtype=np.complex128
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "fit.npz")
        _fit_source(spectra, 3, 4, path)  # killed here
        resumed = _fit_source(spectra, 3, 10, path)  # relaunched, same command
        assert _restore(GaussianMixture(n_components=3), path) == 10
    straight = _fit_source(spectra, 3, 10)
    assert np.allclose(resumed.means, straight.means), "resumed fit drifted from the plain one"
    assert np.allclose(resumed.weights, straight.weights)
    assert np.allclose(resumed.covariances, straight.covariances)

    with tempfile.TemporaryDirectory() as directory:
        log = os.path.join(directory, "probe.jsonl")
        with open(log, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"method": "oracle", "track": "a"}) + "\n")
            handle.write(json.dumps({"method": "dmgmm", "k": 8, "track": "a"}) + "\n")
            handle.write('{"method": "dmgmm", "k": 8, "tra')  # killed mid-write
        assert _completed(log) == {("oracle", "a"), ("dmgmm", 8, "a")}
    assert _completed("") == set()

    # the quota is what bounds the memory on the full database, so it is asserted
    # rather than trusted: three tracks of twenty-odd frames, five frames each
    tracks = [_ready(rng.normal(size=20 * HOP + NFFT)) for _ in range(3)]
    assert len(analyze(tracks[0](), NFFT, HOP)) > 5, "the probe tracks are too short to be capped"
    assert len(_training_frames(tracks, np.random.default_rng(0), quota=5)) == 15
    under = _training_frames(tracks, np.random.default_rng(0), quota=10**6)
    assert len(under) == sum(len(analyze(load(), NFFT, HOP)) for load in tracks), "frames lost"

    print("self-check: a fit resumed across a kill equals an uninterrupted one,")
    print("            a half-written row never counts as done, and the training")
    print("            cap is spent per track so peak memory is one track's frames")


def _check_excerpts() -> None:
    """Three excerpts of one track must be three names, and a short track fewer."""
    global TEST_OFFSETS
    rate = 100
    voice = np.arange(1000, dtype=np.float64)
    accompaniment = -voice
    keep = TEST_OFFSETS
    try:
        TEST_OFFSETS = (0.0,)
        items = _excerpts("t", voice, accompaniment, rate)
        assert [name for name, _ in items] == ["t"], "a single offset renamed the track"
        assert len(items[0][1][0]) == int(TEST_SECONDS * rate)

        TEST_OFFSETS = (0.0, 3.0, 4.5)
        items = _excerpts("t", voice, accompaniment, rate)
        assert [name for name, _ in items] == ["t@0", "t@3", "t@4.5"]
        assert np.array_equal(items[1][1][0], voice[300 : 300 + int(TEST_SECONDS * rate)])
        assert all(np.array_equal(s[0], -s[1]) for _, s in items), "sources came apart"

        # 5 s at offset 4.5 needs 9.5 s of track, and there are 10 s
        assert len(_excerpts("t", voice[:900], accompaniment[:900], rate)) == 2, "no guard"
    finally:
        TEST_OFFSETS = keep
    print("self-check: one offset keeps the bare track name, several give one item")
    print("            per offset, and an excerpt past the end of a track is dropped")


def _check_anchors() -> None:
    """An anchored offset counts from the anchor, and 0 is the anchored excerpt."""
    global TEST_OFFSETS, TEST_ANCHORS
    rate = 100
    voice = np.arange(1000, dtype=np.float64)
    accompaniment = -voice
    keep_off, keep_anchor = TEST_OFFSETS, TEST_ANCHORS
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anchors.tsv")
        with open(path, "w", encoding="utf-8") as handle:
            # trailing columns are the localisation evidence, and must be ignored
            handle.write("t\t2.5\t0.9998\t0.0216\n")
        try:
            TEST_ANCHORS = path
            _anchors.cache_clear()
            length = int(TEST_SECONDS * rate)

            TEST_OFFSETS = (0.0,)
            items = _excerpts("t", voice, accompaniment, rate)
            assert [name for name, _ in items] == ["t"], "one offset renamed the track"
            assert np.array_equal(items[0][1][0], voice[250 : 250 + length]), "not anchored"

            # the label stays the offset, so @0 names the anchored excerpt itself
            TEST_OFFSETS = (-1.0, 0.0, 1.0)
            items = _excerpts("t", voice, accompaniment, rate)
            assert [name for name, _ in items] == ["t@-1", "t@0", "t@1"]
            assert np.array_equal(items[0][1][0], voice[150 : 150 + length])

            # an anchor of 2.5 s with offset -3 s falls before the track: dropped
            TEST_OFFSETS = (-3.0, 0.0)
            assert [n for n, _ in _excerpts("t", voice, accompaniment, rate)] == ["t@0"]

            # a track absent from the file keeps the head of the track
            TEST_OFFSETS = (0.0,)
            items = _excerpts("other", voice, accompaniment, rate)
            assert np.array_equal(items[0][1][0], voice[:length]), "absent track moved"
        finally:
            TEST_OFFSETS, TEST_ANCHORS = keep_off, keep_anchor
            _anchors.cache_clear()
    print("self-check: an anchored offset counts from its track's anchor, offset 0")
    print("            is the anchored excerpt, and a track with no anchor is unmoved")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check_resume()
        _check_excerpts()
        _check_anchors()
    else:
        main()
