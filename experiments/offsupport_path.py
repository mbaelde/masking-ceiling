"""Does the error break where the arg max of the responsibilities switches?

    uv run python experiments/offsupport_path.py            # the journal
    uv run python experiments/offsupport_path.py --check     # self-check only

Article 1's off-support section splits the sensitivity of the estimator into a
regression term, non-expansive under the lemma, and a responsibility term, and
claims the second one is the mechanism: `phi` is a softmax of quadratic forms, so
it partitions observation space into cells bounded by quadrics, one per dominant
component pair, and crossing a wall swaps one affine map for another whose offset
may be far away. The prediction is signed and falsifiable. Under a continuously
increasing deformation of the observed spectrum, the error should grow smoothly
while the arg max of `phi` is stable, and break where it switches. A smooth
monotone curve with no break would move the blame to the regression term and
refute the analysis.

The deformation is a spectral tilt, `s(f) -> s(f) exp(t (f/F - 1/2))` applied to
both sources at once and then renormalized so that the observed frame keeps its
energy. That is a plain filter, it leaves the mixing identity intact so the truth
moves with the observation and the error stays defined, and it changes the shape
alone: a level change would leave the support for a reason the article does not
claim to be about the partition.

The statistic is a difference of two conditional distributions of the same
quantity: the step-to-step jump in error at steps where the arg max switches,
against the jump at steps where it does not. A ratio near one is the refutation.
Nothing is fitted here, the models being the pilot's own checkpoints, so this is
a second read of state that already exists.

Environment: ceiling_sweep's, since its loaders, its config and its checkpoint
key are reused, plus TILT_MAX, PATH_STEPS and PATH_FRAMES. Defaults reproduce the
pilot's in-support arm.

    CORPUS=musdb REGIME=in_support NFFT=1024 K=32 COV_TYPE=diag N_TRAIN=10 \
    N_TEST=3 CHECKPOINT_DIR=/scratch/checkpoints python experiments/offsupport_path.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

import ceiling_sweep as cs
from estimator_readout import _load_source
from evaluation import analyze
from gasm.rase.dmgmm import SourceGMM, _pair_terms, _regress, _stack_real_imag

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

# the tilt reaches +-TILT_MAX/2 in nats of gain at the two ends of the band, so
# 3.0 is a 13 dB slope across the spectrum: enough to leave the fit, small enough
# to stay a filter one could plausibly apply to music
TILT_MAX = float(os.environ.get("TILT_MAX", "3.0"))
PATH_STEPS = int(os.environ.get("PATH_STEPS", "41"))
PATH_FRAMES = int(os.environ.get("PATH_FRAMES", "12"))


def tilt_path(
    source1: ComplexArray, source2: ComplexArray, tilts: FloatArray
) -> tuple[ComplexArray, ComplexArray]:
    """The two deformed sources along the path, shape (frames, steps, bins).

    The gain is the same on both sources, so the sum of the two returned arrays is
    the deformed observation and the errors of the two sources stay opposite, as
    everywhere else in this repo. The renormalization is per (frame, step) and
    shared by the pair, for the same reason.
    """
    bins = source1.shape[-1]
    ramp = np.linspace(-0.5, 0.5, bins)
    gain = np.exp(tilts[:, None] * ramp)
    first = source1[:, None, :] * gain
    second = source2[:, None, :] * gain
    keep = np.linalg.norm(source1 + source2, axis=-1)[:, None] / np.linalg.norm(
        first + second, axis=-1
    )
    return first * keep[..., None], second * keep[..., None]


def mean_and_phi(
    source1: SourceGMM, source2: SourceGMM, x: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The conditional mean and the pair responsibilities on a block of frames.

    Two passes over the component pairs, as in `_regress` and for its reason:
    caching mu_tilde would need n1*n2*frames*dim floats. `demo` checks the mean
    against `_regress` rather than assuming the second pass enumerates in the
    order of the first.
    """
    n_pairs = len(source1.weights) * len(source2.weights)
    log_phi = np.empty((n_pairs, len(x)))
    for i, (_, row, _) in enumerate(_pair_terms(source1, source2, x)):
        log_phi[i] = row
    phi = np.exp(log_phi - logsumexp(log_phi, axis=0))

    mean = np.zeros_like(x)
    for i, (k1, _, solved) in enumerate(_pair_terms(source1, source2, x)):
        cov1 = source1.covariances[k1]
        mean += phi[i][:, None] * (source1.means[k1] + solved @ cov1.T)
    return mean, phi


def path_curves(
    sources: list[SourceGMM],
    spectra: list[ComplexArray],
    frames: NDArray[np.int64],
    tilts: FloatArray,
) -> dict[str, FloatArray]:
    """Error, arg max pair and its weight along the path, shape (frames, steps).

    The whole (frames x steps) block goes through `mean_and_phi` in one call: the
    Cholesky factor of a pair does not depend on the frame, so one call over the
    block costs what one call over a single frame costs plus the solves.
    """
    first, second = tilt_path(spectra[0][frames], spectra[1][frames], tilts)
    shape = first.shape[:2]
    stacked = _stack_real_imag(np.reshape(first + second, (-1, first.shape[-1])))
    truth = _stack_real_imag(np.reshape(first, (-1, first.shape[-1])))

    estimate, phi = mean_and_phi(sources[0], sources[1], stacked)
    error = np.sum((truth - estimate) ** 2, axis=1) / np.sum(truth**2, axis=1)
    return {
        "error_db": np.reshape(10.0 * np.log10(np.maximum(error, 1e-30)), shape),
        "argmax": np.reshape(phi.argmax(axis=0), shape),
        "phi_max": np.reshape(phi.max(axis=0), shape),
    }


def switch_statistics(error_db: FloatArray, argmax: FloatArray) -> dict[str, float]:
    """The jump in error at a switch of the arg max, against the jump elsewhere.

    Both are absolute step-to-step differences of the same curve, so the ratio is
    scale-free and the refutation is a ratio near one. `monotone` is the fraction
    of steps on which the error rises at all, which is the other half of the
    prediction: growth away from the walls is supposed to be smooth, not erratic.
    """
    jumps = np.abs(np.diff(error_db, axis=1))
    switched = np.diff(argmax, axis=1) != 0
    rising = np.diff(error_db, axis=1) > 0
    return {
        "n_paths": float(error_db.shape[0]),
        "n_steps": float(jumps.size),
        "n_switches": float(switched.sum()),
        "paths_with_switch": float((switched.any(axis=1)).sum()),
        "jump_at_switch": float(np.median(jumps[switched])) if switched.any() else float("nan"),
        "jump_elsewhere": float(np.median(jumps[~switched])) if (~switched).any() else float("nan"),
        "jump_at_switch_max": float(jumps[switched].max()) if switched.any() else float("nan"),
        "jump_elsewhere_p95": float(np.percentile(jumps[~switched], 95))
        if (~switched).any()
        else float("nan"),
        "monotone": float(rising.mean()),
        "error_start": float(np.median(error_db[:, 0])),
        "error_end": float(np.median(error_db[:, -1])),
    }


def _loud_frames(mixture: ComplexArray, count: int) -> NDArray[np.int64]:
    """`count` frames spread over the loudest half of the excerpt.

    Silent frames have no error to speak of and no support to leave, and taking
    the single loudest frames instead would sample one event of the track. Evenly
    spaced inside the loud half is the cheapest thing that is neither.
    """
    energy = np.sum(np.abs(mixture) ** 2, axis=1)
    loud = np.argsort(energy)[len(energy) // 2 :]
    return np.sort(loud[np.linspace(0, len(loud) - 1, count).astype(int)])


def render(records: list[dict]) -> str:
    header = f"{'track':<22} {'k':>3} {'sw':>4} {'at sw':>7} {'else':>7} {'ratio':>6} {'mono':>5}"
    lines = [header, "-" * len(header)]
    for r in records:
        ratio = r["jump_at_switch"] / r["jump_elsewhere"] if r["jump_elsewhere"] else float("nan")
        lines.append(
            f"{r['track'][:22]:<22} {r['k']:>3} {r['n_switches']:>4.0f} "
            f"{r['jump_at_switch']:>7.3f} {r['jump_elsewhere']:>7.3f} {ratio:>6.1f} "
            f"{r['monotone']:>5.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    _, test_items = cs._load_stems() if cs.CORPUS == "stems" else cs._load_musdb()
    tilts = np.linspace(0.0, TILT_MAX, PATH_STEPS)
    config = {
        "corpus": cs.CORPUS, "regime": cs.REGIME, "nfft": cs.NFFT, "hop": cs.HOP,
        "cov_type": cs.COV_TYPE, "n_train": cs.N_TRAIN, "reg_covar": cs.REG_COVAR,
        "energy_pct": cs.ENERGY_PCT, "seed": cs.SEED,
        "tilt_max": TILT_MAX, "path_steps": PATH_STEPS, "path_frames": PATH_FRAMES,
    }
    summaries: list[dict] = []

    for n_components in cs.K_VALUES:
        try:
            sources = [_load_source(i, n_components) for i in range(2)]
        except FileNotFoundError as err:
            print(f"# {err}", file=sys.stderr)
            continue

        for name, references in test_items:
            spectra = [analyze(reference, cs.NFFT, cs.HOP) for reference in references]
            frames = _loud_frames(np.sum(spectra, axis=0), PATH_FRAMES)
            curves = path_curves(sources, spectra, frames, tilts)

            # one row per (frame, step): the figure of the article is drawn from
            # these, and the summary below is recomputable from them
            for row, frame in enumerate(frames):
                for step, tilt in enumerate(tilts):
                    print(json.dumps({
                        **config, "kind": "point", "track": name, "k": n_components,
                        "frame": int(frame), "tilt": float(tilt),
                        "error_db": float(curves["error_db"][row, step]),
                        "argmax": int(curves["argmax"][row, step]),
                        "phi_max": float(curves["phi_max"][row, step]),
                    }), flush=True)

            summaries.append({
                **config, "kind": "summary", "track": name, "k": n_components,
                **switch_statistics(curves["error_db"], curves["argmax"]),
            })
            print(json.dumps(summaries[-1]), flush=True)

    print(render(summaries), file=sys.stderr)


def demo() -> None:
    """The machinery on a synthetic pair built so that a wall is crossed.

    Source 1 gets two components far apart in the tilt direction and tight around
    their means, source 2 one broad component, so the path starts inside the first
    cell and ends inside the second, and the estimate has to jump when it crosses.
    Two things are checked that nothing else would catch: the mean of
    `mean_and_phi` against the reference regression, which pins the pair ordering
    of the second pass, and the sign of the statistic itself on a case where the
    answer is known by construction.
    """
    rng = np.random.default_rng(0)
    bins, frames = 8, 5
    dim = 2 * bins
    ramp = np.linspace(-0.5, 0.5, bins)
    tilts = np.linspace(0.0, 3.0, 41)

    # real spectra, so a stacked frame is its magnitude followed by zeros and the
    # geometry of the check is readable
    spectra = [
        1.0 + 0.05 * rng.normal(size=(frames, bins)),
        0.3 + 0.02 * rng.normal(size=(frames, bins)),
    ]
    spectra = [s.astype(np.complex128) for s in spectra]
    first, second = tilt_path(spectra[0], spectra[1], tilts)

    # the two components of source 1 sit at the two ends of the path, so the walk
    # has to cross the wall between their cells, and the far one carries a large
    # offset in the imaginary block, which is the article's "offset that may be
    # far away": crossing swaps an affine map for one centred elsewhere
    source1 = SourceGMM(
        np.array([0.5, 0.5]),
        _stack_real_imag(np.stack([first[:, 0].mean(0), first[:, -1].mean(0) + 0.4j])),
        np.stack([0.1 * np.eye(dim), 0.1 * np.eye(dim)]),
    )
    source2 = SourceGMM(np.ones(1), np.zeros((1, dim)), np.stack([0.1 * np.eye(dim)]))

    x = _stack_real_imag(first[:, 0] + second[:, 0])
    mean, phi = mean_and_phi(source1, source2, x)
    assert np.allclose(mean, _regress(source1, source2, x)), "pair ordering broke"
    assert np.allclose(phi.sum(axis=0), 1.0), "responsibilities do not normalize"
    assert np.allclose(
        np.linalg.norm(first + second, axis=-1),
        np.linalg.norm(spectra[0] + spectra[1], axis=-1)[:, None],
    ), "the path does not preserve the energy of the observation"
    assert np.allclose(first[:, 0], spectra[0]), "the path does not start at the frame"
    tilted = np.abs(first[:, -1, -1] / first[:, -1, 0])
    flat = np.abs(spectra[0][:, -1] / spectra[0][:, 0])
    assert np.all(tilted > 5.0 * flat), "the deformation does not tilt the spectrum"

    curves = path_curves(
        [source1, source2], spectra, np.arange(frames), tilts
    )
    stats = switch_statistics(curves["error_db"], curves["argmax"])
    assert stats["n_switches"] > 0, "the synthetic path crosses no wall"
    assert stats["jump_at_switch"] > 3.0 * stats["jump_elsewhere"], (
        "no break at the wall on a case built to have one: "
        f"{stats['jump_at_switch']:.3f} against {stats['jump_elsewhere']:.3f}"
    )

    print("self-check: the mean equals the reference regression, the path is a", file=sys.stderr)
    print("            shape-only tilt, and the error breaks at the wall", file=sys.stderr)


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        demo()
    else:
        demo()
        main()
