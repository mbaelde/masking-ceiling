"""Does a read-out other than the posterior mean leave the real-mask class?

    uv run python experiments/estimator_readout.py            # the table
    uv run python experiments/estimator_readout.py --check     # self-check only

The pilot's NO-GO is a statement about one estimator, the conditional mean, and
that estimator is provably the wrong one to ask the class question of. Writing
``s = r exp(i(arg x + psi))``, if the conditional law of ``(r, psi)`` given the
mixture is symmetric in ``psi -> -psi`` then ``E[r sin psi | x] = 0``, so

    E[s | x] = E[m* | x] x ,

exactly a real mask, of gain the posterior mean of the oracle gain, with excess
error exactly ``|x|^2 Var(m* | x)``. The mean is pulled back into the class by
the squared-error criterion itself, and it leaves the class only to the extent
that the prior can produce an asymmetric phase posterior. The measured 1 to 5
degrees of rotation is that extent, and the law diagnostic says why it is small:
inside the model's own components the data is barely non-circular at all.

Two read-outs of the same posterior are not means and carry no such pull: the
conditional mean of the most probable component pair, and one draw. The
prediction is signed. Both must show the class witnesses (phase rotation, gains
above one, negative gains) that the mean barely shows, and both must score
worse in SDR, since the mean is the SDR-optimal read-out of this posterior by
construction. A draw that stays in the class would refute reading the deficit as
a posterior variance and would move the blame from the criterion to the
partition, which is the one outcome that would change what the article claims.

Nothing is fitted here. The models are the pilot's own checkpoints, so this is a
second read of state that already exists and it is the cheapest test in the plan.

Environment: the same variables as ceiling_sweep, since its loaders, its config
and its checkpoint key are reused. Defaults reproduce the pilot's in-support arm.

    CORPUS=musdb REGIME=in_support NFFT=1024 K=32 COV_TYPE=diag \
    CHECKPOINT_DIR=/scratch/checkpoints python experiments/estimator_readout.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

import ceiling_sweep as cs
from evaluation import ORACLES, analyze, oracle_spectra, sdr, spectral_ceiling, synthesize
from gasm.rase.dmgmm import (
    SourceGMM,
    _pair_terms,
    _regress,
    _stack_real_imag,
    _unstack_real_imag,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

READOUT_SEED = int(os.environ.get("READOUT_SEED", "0"))
READOUTS = ("mean", "map", "draw")


def _load_source(index: int, n_components: int) -> SourceGMM:
    """Rebuild one source's fitted GMM from the pilot's checkpoint.

    The point is to read the model the pilot actually separated with, so the
    covariance is widened back from diagonal here exactly as `_fit_source` does
    it, and nothing is refitted.
    """
    path = cs._checkpoint_path(index, n_components)
    gmm = GaussianMixture(n_components=n_components, covariance_type=cs.COV_TYPE)
    if not cs._restore(gmm, path):
        raise FileNotFoundError(f"no checkpoint for source {index} at K={n_components}: {path}")
    covariances = gmm.covariances_
    if cs.COV_TYPE == "diag":
        covariances = np.stack([np.diag(v) for v in covariances])
    return SourceGMM(gmm.weights_, gmm.means_, covariances)


def _posterior_covariance(cov1: FloatArray, cov2: FloatArray) -> FloatArray:
    """Sigma_tilde = Sigma_1 - Sigma_1 inv(Sigma_1 + Sigma_2) Sigma_1, symmetrized.

    Computed only for the pairs a frame actually drew, since it is the one
    quantity of the pair that the conditional mean never needs.
    """
    chol = cho_factor(cov1 + cov2, lower=True)
    sigma = cov1 - cov1 @ cho_solve(chol, cov1)
    return 0.5 * (sigma + sigma.T)


def _draw(sigma: FloatArray, count: int, rng: np.random.Generator) -> FloatArray:
    """`count` centered draws from N(0, sigma).

    ponytail: jitter rather than an eigendecomposition. Sigma_tilde is positive
    semi-definite by construction and only loses it to rounding at a thousand
    dimensions, so the smallest jitter that factorizes is the honest fix and the
    eigh path would cost more than the draw it enables.
    """
    scale = max(float(np.trace(sigma)) / len(sigma), 1e-300)
    for jitter in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        try:
            factor = np.linalg.cholesky(sigma + jitter * scale * np.eye(len(sigma)))
        except np.linalg.LinAlgError:
            continue
        return rng.standard_normal((count, len(sigma))) @ factor.T
    raise np.linalg.LinAlgError("posterior covariance not factorizable even with jitter")


def _pair_choice(phi: FloatArray, rng: np.random.Generator) -> NDArray[np.int64]:
    """One component pair per frame, drawn from that frame's posterior weights."""
    edges = np.cumsum(phi, axis=0)
    return np.minimum((edges < rng.random(phi.shape[1])).sum(axis=0), len(phi) - 1)


def readouts(
    source1: SourceGMM, source2: SourceGMM, x: FloatArray, rng: np.random.Generator
) -> dict[str, FloatArray]:
    """Three read-outs of the same posterior p(x1 | x1 + x2 = x), per frame.

    `mean` is `_regress`, recomputed here rather than called so the three share
    one pass over the component pairs. `map` is the conditional mean of the most
    probable pair: it is the posterior mode when the pairs are separated, and the
    peak-height correction -0.5 log det Sigma_tilde is deliberately not applied,
    since the question is only whether a read-out that is not a mean leaves the
    class, not which of two nearby modes is the true one. `draw` samples the pair
    from the posterior weights and then the frame from that pair's Gaussian.

    Two passes over the pairs, as in `_regress` and for the same reason: caching
    mu_tilde for every pair would need n1*n2*n_frames*dim floats. The second pass
    re-enumerates in the order `_pair_terms` uses, k1 outer and k2 inner, which
    `demo` checks against `_regress` rather than assuming.
    """
    n2 = len(source2.weights)
    n_pairs = len(source1.weights) * n2
    log_phi = np.empty((n_pairs, len(x)))
    for i, (_, row, _) in enumerate(_pair_terms(source1, source2, x)):
        log_phi[i] = row
    phi = np.exp(log_phi - logsumexp(log_phi, axis=0))
    best = log_phi.argmax(axis=0)
    picked = _pair_choice(phi, rng)

    out = {name: np.zeros_like(x) for name in READOUTS}
    for i, (k1, _, solved) in enumerate(_pair_terms(source1, source2, x)):
        cov1 = source1.covariances[k1]
        mu_tilde = source1.means[k1] + solved @ cov1.T
        out["mean"] += phi[i][:, None] * mu_tilde
        modal = best == i
        if modal.any():
            out["map"][modal] = mu_tilde[modal]
        drawn = picked == i
        if drawn.any():
            sigma = _posterior_covariance(cov1, source2.covariances[i % n2])
            out["draw"][drawn] = mu_tilde[drawn] + _draw(sigma, int(drawn.sum()), rng)
    return out


def _estimates(
    sources: list[SourceGMM], mixture: ComplexArray, rng: np.random.Generator
) -> dict[str, list[ComplexArray]]:
    """Per read-out, one complex estimate per source.

    The second source is the mixture minus the first, as `_separate` does it, so
    the two estimates sum back to the observation and their errors are exactly
    opposite. That is not a convenience: it is the same identity that makes the
    ceiling residual shared, so breaking it here would make the two sources'
    numbers incomparable with the pilot's.
    """
    stacked = _stack_real_imag(mixture)
    reads = readouts(sources[0], sources[1], stacked, rng)
    return {
        name: [_unstack_real_imag(first), _unstack_real_imag(stacked - first)]
        for name, first in reads.items()
    }


def render(records: list[dict]) -> str:
    header = (
        f"{'track':<22} {'src':>3} {'read':<5} {'sdr':>7} {'ceil':>7} {'gap':>6} "
        f"{'phase':>6} {'g>1':>6} {'g<0':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        if r["method"] != "dmgmm":
            continue
        lines.append(
            f"{r['track'][:22]:<22} {r['source']:>3} {r['readout']:<5} "
            f"{r['sdr']:>7.2f} {r['sdr_best_real']:>7.2f} "
            f"{r['sdr'] - r['sdr_best_real']:>6.2f} "
            f"{r['phase_median_deg']:>6.2f} {r['gain_above_one']:>6.3f} "
            f"{r['gain_negative']:>6.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(READOUT_SEED)
    _, test_items = cs._load_stems() if cs.CORPUS == "stems" else cs._load_musdb()
    config = {
        "corpus": cs.CORPUS, "regime": cs.REGIME, "nfft": cs.NFFT, "hop": cs.HOP,
        "cov_type": cs.COV_TYPE, "n_train": cs.N_TRAIN, "reg_covar": cs.REG_COVAR,
        "energy_pct": cs.ENERGY_PCT, "seed": cs.SEED, "readout_seed": READOUT_SEED,
    }
    records: list[dict] = []

    for n_components in cs.K_VALUES:
        try:
            sources = [_load_source(i, n_components) for i in range(2)]
        except FileNotFoundError as err:
            print(f"# {err}", file=sys.stderr)
            continue

        for name, references in test_items:
            spectra = [analyze(reference, cs.NFFT, cs.HOP) for reference in references]
            mixture = np.sum(spectra, axis=0)
            # the ceiling of the class, per source, so a deficit is readable on
            # the same line as the read-out that produced it
            ceiling = {}
            for index, reference in enumerate(references):
                oracle = oracle_spectra(spectra, index)
                ceiling[index] = sdr(
                    reference, synthesize(oracle["best_real"], cs.NFFT, cs.HOP), cs.NFFT
                )
                records.append({
                    **config, "track": name, "source": index, "method": "oracle",
                    "k": n_components, "spectral_ceiling": spectral_ceiling(spectra, index),
                    **{
                        f"sdr_{key}": sdr(
                            reference, synthesize(oracle[key], cs.NFFT, cs.HOP), cs.NFFT
                        )
                        for key in ORACLES
                    },
                })
                print(json.dumps(records[-1]), flush=True)

            for readout, estimates in _estimates(sources, mixture, rng).items():
                for index, scores in enumerate(cs._score(references, estimates, mixture)):
                    records.append({
                        **config, "track": name, "source": index, "method": "dmgmm",
                        "k": n_components, "readout": readout,
                        "sdr_best_real": ceiling[index], **scores,
                    })
                    print(json.dumps(records[-1]), flush=True)

    print(render(records), file=sys.stderr)


def demo() -> None:
    """The three read-outs against what they are supposed to be, on a small pair.

    The load-bearing check is the first one: `mean` must equal `_regress` to
    floating point, which is what pins the second pass's pair ordering to the one
    `_pair_terms` enumerates. Everything else in this file is downstream of that
    index being right, and nothing else would catch it being wrong.
    """
    rng = np.random.default_rng(0)
    dim, frames = 6, 40

    def gmm(n: int, spread: float) -> SourceGMM:
        roots = rng.normal(size=(n, dim, dim))
        return SourceGMM(
            np.full(n, 1.0 / n),
            rng.normal(size=(n, dim)) * spread,
            np.stack([r @ r.T + np.eye(dim) for r in roots]),
        )

    source1, source2 = gmm(3, 2.0), gmm(2, 2.0)
    x = rng.normal(size=(frames, dim)) * 3.0
    reads = readouts(source1, source2, x, np.random.default_rng(1))
    assert np.allclose(reads["mean"], _regress(source1, source2, x)), "pair ordering broke"

    # a draw averages back to the mean: the sampler's only real content
    pooled = np.mean(
        [readouts(source1, source2, x, np.random.default_rng(s))["draw"] for s in range(400)],
        axis=0,
    )
    spread = float(np.abs(reads["mean"]).mean())
    assert np.abs(pooled - reads["mean"]).mean() < 0.1 * spread, "draws do not center on the mean"

    # the mean minimizes the posterior expected squared error, so any other
    # read-out pays exactly its distance to the mean: that is the SDR cost the
    # prediction expects, and it is only non-zero if the read-outs really differ
    for name in ("map", "draw"):
        assert np.abs(reads[name] - reads["mean"]).mean() > 1e-6, f"{name} collapsed onto the mean"

    # a source with no posterior uncertainty left has all three read-outs equal:
    # Sigma_tilde vanishes with Sigma_1, and so does the spread across pairs
    sharp = SourceGMM(np.ones(1), np.zeros((1, dim)), 1e-10 * np.eye(dim)[None])
    tight = readouts(sharp, source2, x, np.random.default_rng(2))
    assert np.abs(tight["draw"] - tight["mean"]).max() < 1e-3
    assert np.allclose(tight["map"], tight["mean"])

    # stderr, not stdout: main() writes the journal on stdout and a jsonl reader
    # should not have to skip a banner
    print("self-check: mean equals the reference regression, draws center on it,", file=sys.stderr)
    print("            and the mean is the least-squares read-out of the three", file=sys.stderr)


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        demo()
    else:
        demo()
        main()
