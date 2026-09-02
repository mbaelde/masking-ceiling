"""Is the deficit the model's own posterior variance, and which term carries it?

    uv run python experiments/prior_calibration.py           # the table
    uv run python experiments/prior_calibration.py --check    # self-check only

Proposition 5 says the posterior mean's excess error over the best real mask is
``E[|x|^2 Var(m* | x)]``, and the read-out probe measured that excess: 6.57 dB of
deficit under a 14.35 dB ceiling at K = 32. What it did not measure is whether
the model *believes* that excess. Both halves of the identity are computable here
on the pilot's own checkpoints, per bin and in closed form:

    predicted   sum |x|^2 Var_model(m* | x)     what the prior expects to lose
    realized    sum |x|^2 (m_hat - m*)^2        what it actually loses

since ``m* = Re(s conj(x)) / |x|^2`` is a linear functional of the stacked source
at fixed ``x``, so the posterior of ``m*`` is the same Gaussian mixture over
component pairs that the estimator already sums over, read through one vector.

The ratio decides whether heavier tails are even the right lever, and the
decomposition decides where they would act:

    ratio ~ 1     the prior is calibrated. The deficit is honest posterior
                  variance, and narrowing it needs a genuinely sharper posterior.
    ratio >> 1    the prior is overconfident. The error lives on frames it gives
                  almost no mass, which is what a heavy tail is for.
    ratio << 1    the excess reading itself is wrong, and Proposition 5 is not
                  what the pilot measured. That outcome stops phase B.

    within        variance inside a component pair, which a Gaussian scale
                  mixture replaces wholesale.
    between       variance across pairs, which it leaves alone, since giving a
                  component a random scale does not move its mean.

So a deficit that is `between` says the lever is more components or conditioning,
not Student, and that is the case the read-out probe already hinted at: the MAP
sat 0.04 dB from the mean at all three K, which can only happen if the per-frame
responsibilities are near degenerate. This file is the quantitative version of
that hint, and it is the gate on writing a Student EM.

Nothing is fitted. Same checkpoints, same test split, same STFT as the read-out
probe, so the numbers are comparable line by line, and the spectral-domain
deficit printed here is the cross-check: it must land near the time-domain
deficit the pilot measured, or the closed forms are wrong rather than the prior.

Environment: the same variables as ceiling_sweep. Defaults reproduce the pilot's
in-support arm.

    CORPUS=musdb REGIME=in_support NFFT=1024 K=8,16,32 COV_TYPE=diag \
    CHECKPOINT_DIR=/scratch/checkpoints python experiments/prior_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

import ceiling_sweep as cs
from evaluation import _EPS, analyze
from gasm.rase.dmgmm import _LOG_2PI, _stack_real_imag

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DiagGMM = tuple[FloatArray, FloatArray, FloatArray]  # weights, means, diagonal variances
QUANTILES = (0.25, 0.5, 0.75)


def _require_diag() -> None:
    """Checked before the corpus is loaded, since loading musdb costs minutes."""
    if cs.COV_TYPE != "diag":
        raise NotImplementedError(
            f"COV_TYPE={cs.COV_TYPE}: Sigma_tilde is dense, so the per-pair variance would "
            "need one solve per pair. The pilot's arm is diagonal."
        )


def _load_diag(index: int, n_components: int) -> DiagGMM:
    """One source's fitted GMM from the pilot's checkpoint, diagonal kept diagonal.

    `estimator_readout` widens the same file into full matrices because the
    separator wants them. Here the diagonal is load bearing: it is what makes the
    posterior covariance of a pair available without a dim-by-dim solve, so the
    variance of `m*` costs one pass over the pairs instead of a thousand
    factorizations of a 1026-square matrix.
    """
    _require_diag()
    path = cs._checkpoint_path(index, n_components)
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
    if not cs._restore(gmm, path):
        raise FileNotFoundError(f"no checkpoint for source {index} at K={n_components}: {path}")
    return gmm.weights_, gmm.means_, gmm.covariances_


def _pair_terms(
    model1: DiagGMM, model2: DiagGMM, x: FloatArray
) -> Iterator[tuple[int, int, FloatArray, FloatArray]]:
    """`gasm.rase.dmgmm._pair_terms` with the diagonal exploited, plus k2.

    Same enumeration order, k1 outer and k2 inner, and the same two quantities,
    but the Cholesky collapses to an elementwise reciprocal. `demo` asserts the
    equality against the full-matrix original rather than trusting it.
    """
    weights1, means1, vars1 = model1
    weights2, means2, vars2 = model2
    dim = x.shape[-1]
    for k1 in range(len(weights1)):
        for k2 in range(len(weights2)):
            total = vars1[k1] + vars2[k2]
            delta = x - (means1[k1] + means2[k2])
            solved = delta / total
            log_phi = np.log(weights1[k1] * weights2[k2]) - 0.5 * (
                np.einsum("td,td->t", delta, solved)
                + np.log(total).sum()
                + dim * _LOG_2PI
            )
            yield k1, k2, log_phi, solved


def gain_posterior(model1: DiagGMM, model2: DiagGMM, x: FloatArray) -> dict[str, FloatArray]:
    """Posterior mean and variance of the oracle real gain of source 1, per bin.

    Writing ``a`` for the row vector that reads a gain off a stacked spectrum,
    ``a = (Re x, Im x) / |x|^2``, the oracle gain is ``m* = a s1`` exactly, so
    under the pair mixture ``p(s1 | x) = sum_i phi_i N(mu_i, Sigma_i)``

        E[m* | x]   = sum_i phi_i a mu_i
        Var(m* | x) = sum_i phi_i a Sigma_i a'  +  sum_i phi_i (a mu_i - E[m*|x])^2

    which is the law of total variance over the pair label, `within` then
    `between`. The diagonal makes ``a Sigma_i a'`` two terms rather than a
    quadratic form, and drops the real/imaginary cross term with it: that is a
    property of the pilot's fit, not an approximation introduced here.

    Two passes over the pairs, as in `_regress` and for its reason: caching
    ``mu_i`` for every pair would cost n1*n2*n_frames*dim floats.
    """
    weights1, means1, vars1 = model1
    _, _, vars2 = model2
    bins = x.shape[-1] // 2
    power = np.maximum(x[:, :bins] ** 2 + x[:, bins:] ** 2, _EPS)
    a_re, a_im = x[:, :bins] / power, x[:, bins:] / power

    n_pairs = len(weights1) * len(model2[0])
    log_phi = np.empty((n_pairs, len(x)))
    for i, (_, _, row, _) in enumerate(_pair_terms(model1, model2, x)):
        log_phi[i] = row
    phi = np.exp(log_phi - logsumexp(log_phi, axis=0))

    mean = np.zeros_like(power)
    second = np.zeros_like(power)
    within = np.zeros_like(power)
    for i, (k1, k2, _, solved) in enumerate(_pair_terms(model1, model2, x)):
        mu = means1[k1] + vars1[k1] * solved
        gain = a_re * mu[:, :bins] + a_im * mu[:, bins:]
        sigma = vars1[k1] * vars2[k2] / (vars1[k1] + vars2[k2])
        weight = phi[i][:, None]
        mean += weight * gain
        second += weight * gain**2
        within += weight * (a_re**2 * sigma[:bins] + a_im**2 * sigma[bins:])
    return {
        "mean": mean,
        "within": within,
        "between": np.maximum(second - mean**2, 0.0),
        "var": within + np.maximum(second - mean**2, 0.0),
        "power": power,
    }


def calibration(
    post: dict[str, FloatArray], truth: ComplexArray, mixture: ComplexArray
) -> dict[str, float]:
    """Predicted against realized excess, in total and by quartile of prediction.

    The excess is the whole gap between a real mask and the best one: the residual
    of gain ``m`` is ``R* + sum |x|^2 (m - m*)^2`` with ``R*`` the residual of
    ``m*`` itself, an identity in the bins and not an approximation, which is why
    a spectral deficit computed from these two numbers is comparable with the
    pilot's time-domain deficit at all.

    The two sources of a pair share every number this returns, ``R*`` included,
    so one of them is the whole measurement. Write ``r = s - m* x`` for the part
    of the source no real mask can reach: the second source has ``s2 = x - s``
    and ``m*2 = 1 - m*``, hence ``r2 = -r`` exactly, and ``R*`` is the same sum.
    The posterior mean complements the same way, so the error ``(m_hat - m*)^2``
    and the variance are identical bin by bin. Only ``sum |s|^2`` differs, and it
    cancels in a deficit, which is a ratio of residuals. Hence one record per
    track: a second one would be a copy, and computing it from this ``post``
    without complementing the mean is how it silently stops being one.
    """
    power = post["power"]
    star = (truth.real * mixture.real + truth.imag * mixture.imag) / power
    error = (post["mean"] - star) ** 2
    realized = float(np.sum(power * error))
    predicted = float(np.sum(power * post["var"]))
    residual = float(np.sum(np.abs(truth) ** 2) - np.sum(power * star**2))

    # the calibration curve, on bins the mixture actually occupies: a single
    # ratio cannot tell an overconfident prior from one that is wrong on a few
    # loud bins, and the quartiles can
    floor = cs._SILENCE_FLOOR * float(np.abs(mixture).max())
    live = power > floor**2
    edges = np.quantile(post["var"][live], QUANTILES)
    group = np.digitize(post["var"], edges)
    ratios: list[float] = []
    for q in range(len(QUANTILES) + 1):
        cell = live & (group == q)
        den = float(np.sum(power[cell] * post["var"][cell]))
        ratios.append(float(np.sum(power[cell] * error[cell]) / den) if den > 0 else float("nan"))

    return {
        "realized_excess": realized,
        "predicted_excess": predicted,
        "ratio": realized / predicted if predicted > 0 else float("nan"),
        "within_fraction": float(np.sum(power * post["within"])) / predicted if predicted > 0 else float("nan"),
        "residual_star": residual,
        "deficit_realized_db": 10.0 * np.log10((residual + realized) / residual),
        "deficit_predicted_db": 10.0 * np.log10((residual + predicted) / residual),
        "ratio_by_quartile": ratios,
    }


def render(records: list[dict]) -> str:
    header = (
        f"{'track':<22} {'K':>3} {'def_real':>9} {'def_pred':>9} "
        f"{'ratio':>7} {'within':>7} {'quartile ratios':>28}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        quartiles = "  ".join(f"{q:5.2f}" for q in r["ratio_by_quartile"])
        lines.append(
            f"{r['track'][:22]:<22} {r['k']:>3} "
            f"{r['deficit_realized_db']:>9.2f} {r['deficit_predicted_db']:>9.2f} "
            f"{r['ratio']:>7.2f} {r['within_fraction']:>7.3f} {quartiles:>28}"
        )
    return "\n".join(lines)


def main() -> None:
    _require_diag()
    _, test_items = cs._load_stems() if cs.CORPUS == "stems" else cs._load_musdb()
    config = {
        "corpus": cs.CORPUS, "regime": cs.REGIME, "nfft": cs.NFFT, "hop": cs.HOP,
        "cov_type": cs.COV_TYPE, "n_train": cs.N_TRAIN, "reg_covar": cs.REG_COVAR,
        "energy_pct": cs.ENERGY_PCT, "seed": cs.SEED,
    }
    # ponytail: reuse ceiling_sweep's resume key. Its `else` branch keys on
    # (k, track), which is exactly this probe's cell, so any method name but
    # "oracle" works and no second parser is needed.
    done = cs._completed(cs.RESUME_FROM)
    records: list[dict] = []

    for n_components in cs.K_VALUES:
        try:
            models = [_load_diag(i, n_components) for i in range(2)]
        except FileNotFoundError as err:
            print(f"# {err}", file=sys.stderr)
            continue

        for name, references in test_items:
            if ("dmgmm", n_components, name) in done:
                continue
            spectra = [analyze(reference, cs.NFFT, cs.HOP) for reference in references]
            mixture = np.sum(spectra, axis=0)
            post = gain_posterior(models[0], models[1], _stack_real_imag(mixture))
            # one record per track, not per source: `calibration` explains why the
            # second source is a copy of the first
            records.append({
                **config, "track": name, "method": "calibration",
                "k": n_components, "frames": int(len(mixture)),
                **calibration(post, spectra[0], mixture),
            })
            print(json.dumps(records[-1]), flush=True)

    print(render(records), file=sys.stderr)


def demo() -> None:
    """The closed forms against the machinery they are supposed to shortcut.

    Three independent checks, in order of what they would catch. The first pins
    the diagonal pair pass to `gasm`'s full-matrix one, so a wrong reciprocal or a
    swapped index cannot survive. The second pins the variance to the sampler the
    read-out probe already uses, so a missing `between` term cannot survive. The
    third is the one that makes the statistic mean anything: when the prior is
    true by construction, the ratio must read one, and if it does not then a ratio
    away from one on real audio says nothing about the prior.
    """
    import estimator_readout as er
    from gasm.rase.dmgmm import SourceGMM, _pair_terms as full_pair_terms, _regress

    rng = np.random.default_rng(0)
    dim, bins, frames = 8, 4, 60

    def diag_gmm(n: int) -> DiagGMM:
        return (
            np.full(n, 1.0 / n),
            rng.normal(size=(n, dim)) * 2.0,
            rng.gamma(4.0, 0.5, size=(n, dim)) + 0.2,
        )

    def widen(model: DiagGMM) -> SourceGMM:
        weights, means, variances = model
        return SourceGMM(weights, means, np.stack([np.diag(v) for v in variances]))

    model1, model2 = diag_gmm(3), diag_gmm(2)
    x = rng.normal(size=(frames, dim)) * 3.0

    # 1. the diagonal shortcut is the same operator, term by term and in order
    mine = list(_pair_terms(model1, model2, x))
    theirs = list(full_pair_terms(widen(model1), widen(model2), x))
    assert len(mine) == len(theirs)
    for (k1, _, log_phi, solved), (their_k1, their_log_phi, their_solved) in zip(mine, theirs):
        assert k1 == their_k1, "pair enumeration diverged"
        assert np.allclose(log_phi, their_log_phi), "diagonal log-density is not the full one"
        assert np.allclose(solved, their_solved), "diagonal solve is not the full one"

    post = gain_posterior(model1, model2, x)
    power = post["power"]
    reference = _regress(widen(model1), widen(model2), x)
    gain = (x[:, :bins] * reference[:, :bins] + x[:, bins:] * reference[:, bins:]) / power
    assert np.allclose(post["mean"], gain), "posterior mean gain is not the regression read through a"

    # 2. the variance against draws from the same posterior, weighted the way the
    #    excess weights it, so the check is on the number the probe reports
    draws = np.stack([
        er.readouts(widen(model1), widen(model2), x, np.random.default_rng(seed))["draw"]
        for seed in range(600)
    ])
    sampled = (x[:, :bins] * draws[:, :, :bins] + x[:, bins:] * draws[:, :, bins:]) / power
    empirical = float(np.sum(power * sampled.var(axis=0)))
    analytic = float(np.sum(power * post["var"]))
    assert abs(empirical / analytic - 1.0) < 0.1, f"variance off: {empirical:.4g} vs {analytic:.4g}"
    assert post["between"].sum() > 0.0, "no across-pair variance on a three-by-two mixture"

    # 3. calibrated by construction: draw the truth from the prior itself and the
    #    realized excess must match the predicted one
    big = 6000
    truth = np.empty((big, dim))
    other = np.empty((big, dim))
    for target, model in ((truth, model1), (other, model2)):
        weights, means, variances = model
        label = rng.choice(len(weights), size=big, p=weights)
        target[:] = means[label] + np.sqrt(variances[label]) * rng.normal(size=(big, dim))
    observed = truth + other
    complex_x = observed[:, :bins] + 1j * observed[:, bins:]
    complex_truth = truth[:, :bins] + 1j * truth[:, bins:]
    calibrated = gain_posterior(model1, model2, observed)
    stats = calibration(calibrated, complex_truth, complex_x)
    assert abs(stats["ratio"] - 1.0) < 0.1, f"not calibrated under its own prior: {stats['ratio']:.3f}"

    # 4. the second source is the same measurement, and stays so only if the mean
    #    is complemented with it. This is the identity behind one record per track,
    #    and reading source two off source one's mean is how it breaks in silence.
    mirrored = calibration(
        {**calibrated, "mean": 1.0 - calibrated["mean"]}, complex_x - complex_truth, complex_x
    )
    for key in ("realized_excess", "predicted_excess", "residual_star", "deficit_realized_db"):
        assert abs(mirrored[key] - stats[key]) <= 1e-9 * abs(stats[key]), f"{key} differs across sources"

    # a source with no posterior spread left has neither term
    sharp = (np.ones(1), np.zeros((1, dim)), np.full((1, dim), 1e-12))
    assert gain_posterior(sharp, model2, x)["var"].max() < 1e-9

    print(
        "self-check: the diagonal pass equals gasm's full one, the closed-form variance "
        f"equals the sampler to {abs(empirical / analytic - 1.0):.1%},",
        file=sys.stderr,
    )
    print(
        f"            and the excess reads {stats['ratio']:.3f} of prediction when the prior is true",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        demo()
    else:
        demo()
        main()
