"""Which distributional assumption do the stems actually violate, and by how much.

    uv run python experiments/law_diagnostic.py            # the table
    uv run python experiments/law_diagnostic.py --check     # self-check only

The pilot says the DM-GMM sits 6.6 dB under the best real mask and rotates the
phase by 5 degrees at K = 32. Two families of explanation: the prior's law is
wrong, or the prior's law is right and there are not enough frames per dimension
to fit it. Nothing in the sweep separates them, because the sweep only ever
looks at the fitted model.

This looks at the data instead, with no separation and no fit of our own. Four
statistics on the training spectra, each the direct test of one assumption the
model makes, each reported against the value the same statistic takes on
synthetic data that satisfies the assumption, at the same sample size. That null
column is the point: at 300 frames per cell, |E[s^2]| / E[|s|^2] is about 0.05
under perfect circularity, so a measured 0.05 is evidence of nothing.

  rho    = |E[s^2]| / E[|s|^2], zero for a circular complex Gaussian. This is
           the premise of the whole article, that there is phase structure in
           the prior to exploit.
  ckurt  = E[|s|^4] / E[|s|^2]^2, two for a circular complex Gaussian, larger
           for heavy tails. A GMM can only imitate tails by spending components
           on them, so this is where the parameter budget leaks.
  alpha  = Hill estimator of the tail index on the upper 5 % of |s|. Finite for
           an alpha-stable law, unbounded (grows with n) for a Gaussian.
  gap    = corr(z_i^2, z_j^2) - corr(z_i, z_j)^2 on adjacent bins, after
           marginal rank-gaussianization. Exactly zero for a Gaussian copula,
           since a bivariate normal of correlation r has corr(z1^2, z2^2) = r^2.
           Positive under a shared latent scale, which is the signature of a
           Gaussian scale mixture, hence of a Student-t or alpha-stable prior.

Marginal and conditional rows both matter and they answer different questions.
Marginally the phase of a music partial drifts across frames, so circularity is
close to satisfied whatever the prior needs; the model never sees that marginal,
it sees each component. The conditioning here is therefore the fitted model's own
hard assignment, read off the K = 32 checkpoint of the pilot: if rho is at its
null floor inside the model's own components, then no amount of fitting could
have carried phase information and the 5 degrees are explained at the root.

Environment: the same variables as ceiling_sweep, since the module is imported
for its loaders and its checkpoint key. Defaults here reproduce the pilot's
in-support arm, which is the arm whose deficit is being explained.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

import ceiling_sweep as cs
from gasm.rase.dmgmm import _stack_real_imag

K_DIAG = int(os.environ.get("K_DIAG", "32"))
MIN_CELL = int(os.environ.get("MIN_CELL", "30"))  # frames below which a cell is dropped
TAIL_PCT = float(os.environ.get("TAIL_PCT", "5"))
SEED = int(os.environ.get("DIAG_SEED", "0"))

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]


def circularity(s: Complex) -> Float:
    """|E[s^2]| / E[|s|^2] per bin. Zero for a circular law, one for a real one."""
    return np.abs((s**2).mean(axis=0)) / np.maximum((np.abs(s) ** 2).mean(axis=0), 1e-30)


def complex_kurtosis(s: Complex) -> Float:
    """E[|s|^4] / E[|s|^2]^2 per bin. Two for a circular complex Gaussian."""
    power = np.abs(s) ** 2
    return (power**2).mean(axis=0) / np.maximum(power.mean(axis=0) ** 2, 1e-30)


def hill(s: Complex) -> Float:
    """Hill estimator of the tail index of |s|, per bin, on the upper TAIL_PCT.

    Capped at 99: for a Gaussian the estimate diverges with the sample size and
    the interesting question is only whether it is small, not how large it is.
    """
    mag = np.sort(np.abs(s), axis=0)[::-1]
    m = max(20, int(len(mag) * TAIL_PCT / 100))
    if m + 1 >= len(mag):
        return np.full(mag.shape[1], np.nan)
    logs = np.log(np.maximum(mag[: m + 1], 1e-30))
    xi = (logs[:m] - logs[m]).mean(axis=0)
    return np.minimum(1.0 / np.maximum(xi, 1e-6), 99.0)


def _normal_scores(x: Float) -> Float:
    """Rank-gaussianize each column, so any Gaussian-copula question is about
    dependence alone and not about the marginals."""
    n = len(x)
    ranks = np.argsort(np.argsort(x, axis=0), axis=0)
    return norm.ppf((ranks + 0.5) / n)


def copula_gap(x: Float) -> Float:
    """corr(z_i^2, z_j^2) - corr(z_i, z_j)^2 on adjacent columns of x.

    Zero for any Gaussian copula, whatever the correlation. Positive when the
    columns share a latent scale, which is what a Student-t or an alpha-stable
    prior is and what a Gaussian mixture has to fake with extra components.
    """
    z = _normal_scores(x)
    a, b = z[:, :-1], z[:, 1:]
    r = (a * b).mean(axis=0) / np.sqrt((a**2).mean(axis=0) * (b**2).mean(axis=0))
    p, q = a**2 - 1.0, b**2 - 1.0
    r2 = (p * q).mean(axis=0) / np.sqrt(
        np.maximum((p**2).mean(axis=0) * (q**2).mean(axis=0), 1e-30)
    )
    return r2 - r**2


def _null(n: int, f: int, rng: np.random.Generator) -> dict[str, float]:
    """The same four statistics on data that satisfies every assumption, at the
    same sample size. Without this column none of the measured numbers has a
    scale: rho alone falls off like 1/sqrt(n) under perfect circularity."""
    s = (rng.standard_normal((n, f)) + 1j * rng.standard_normal((n, f))) / np.sqrt(2)
    return {
        "rho": float(np.mean(circularity(s))),
        "ckurt": float(np.mean(complex_kurtosis(s))),
        "alpha": float(np.nanmean(hill(s))),
        "gap": float(np.mean(copula_gap(s.real))),
    }


def _weighted(values: Float, weights: Float) -> float:
    good = np.isfinite(values)
    if not good.any():
        return float("nan")
    return float(np.average(values[good], weights=weights[good]))


def _energy(s: Complex) -> Float:
    return (np.abs(s) ** 2).mean(axis=0)


def marginal(s: Complex, rng: np.random.Generator) -> dict[str, float]:
    w = _energy(s)
    return {
        "cells": 1,
        "frames": len(s),
        "rho": _weighted(circularity(s), w),
        "ckurt": _weighted(complex_kurtosis(s), w),
        "alpha": _weighted(hill(s), w),
        "gap": float(np.mean(copula_gap(s.real))),
        "null": _null(len(s), s.shape[1], rng),
    }


def assignment(spectra: Complex, index: int, k: int) -> NDArray[np.int64]:
    """Hard assignment under the pilot's own fitted model, from its checkpoint.

    The checkpoint is the parameterization scikit-learn runs EM on, so the model
    is rebuilt from it rather than refitted: this has to be the same partition
    the estimator used, not a similar one.
    """
    path = cs._checkpoint_path(index, k)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint for source {index} at K={k}: {path}")
    state = np.load(path)
    gmm = GaussianMixture(n_components=k, covariance_type=cs.COV_TYPE)
    gmm.weights_ = state["weights"]
    gmm.means_ = state["means"]
    gmm.covariances_ = state["covariances"]
    gmm.precisions_cholesky_ = state["precisions_cholesky"]
    gmm.n_features_in_ = gmm.means_.shape[1]
    return gmm.predict(_stack_real_imag(spectra))


def conditional(s: Complex, labels: NDArray[np.int64], rng: np.random.Generator) -> dict[str, float]:
    """The same statistics inside each component, pooled with weight equal to the
    energy the component actually carries. Cells under MIN_CELL frames are
    dropped and counted, never silently folded into a neighbour."""
    rows, weights, nulls, dropped = [], [], [], 0
    for label in np.unique(labels):
        cell = s[labels == label]
        if len(cell) < MIN_CELL:
            dropped += len(cell)
            continue
        w = _energy(cell)
        rows.append(
            (
                _weighted(circularity(cell), w),
                _weighted(complex_kurtosis(cell), w),
                _weighted(hill(cell), w),
                float(np.mean(copula_gap(cell.real))),
            )
        )
        weights.append(w.sum())
        nulls.append(_null(len(cell), cell.shape[1], rng))
    table, weight = np.array(rows), np.array(weights)
    return {
        "cells": len(rows),
        "dropped_frames": dropped,
        "frames": len(labels) - dropped,
        "rho": _weighted(table[:, 0], weight),
        "ckurt": _weighted(table[:, 1], weight),
        "alpha": _weighted(table[:, 2], weight),
        "gap": _weighted(table[:, 3], weight),
        "null": {
            key: _weighted(np.array([n[key] for n in nulls]), weight)
            for key in ("rho", "ckurt", "alpha", "gap")
        },
    }


def render(records: list[dict]) -> str:
    header = (
        f"{'src':>3} {'scope':<12} {'cells':>5} {'frames':>7} "
        f"{'rho':>6} {'rho0':>6} {'ckurt':>6} {'ckurt0':>6} "
        f"{'alpha':>6} {'alpha0':>6} {'gap':>7} {'gap0':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        n = r["null"]
        lines.append(
            f"{r['source']:>3} {r['scope']:<12} {r['cells']:>5} {r['frames']:>7} "
            f"{r['rho']:>6.3f} {n['rho']:>6.3f} {r['ckurt']:>6.2f} {n['ckurt']:>6.2f} "
            f"{r['alpha']:>6.1f} {n['alpha']:>6.1f} {r['gap']:>+7.3f} {n['gap']:>+7.3f}"
        )
    return "\n".join(lines)


def demo() -> None:
    """Each statistic is checked on a law that violates exactly one assumption."""
    rng = np.random.default_rng(0)
    n, f = 4000, 40
    circular = (rng.standard_normal((n, f)) + 1j * rng.standard_normal((n, f))) / np.sqrt(2)
    assert np.mean(circularity(circular)) < 0.05
    assert abs(np.mean(complex_kurtosis(circular)) - 2.0) < 0.15
    assert abs(np.mean(copula_gap(circular.real))) < 0.02
    assert np.mean(hill(circular)) > 4.0

    # non-circular: the imaginary part squeezed, rho has to come up
    squeezed = circular.real + 0.2j * circular.imag
    assert np.mean(circularity(squeezed)) > 0.5

    # Gaussian scale mixture: heavy tails and a positive copula gap, the two
    # signatures a Gaussian mixture can only imitate by spending components
    scale = np.sqrt(rng.gamma(1.5, 1 / 1.5, size=(n, 1)))
    heavy = circular * scale
    assert np.mean(complex_kurtosis(heavy)) > 3.0
    assert np.mean(copula_gap(heavy.real)) > 0.05
    assert np.mean(hill(heavy)) < np.mean(hill(circular))
    # ... while staying circular, so rho does not confuse tails with phase
    assert np.mean(circularity(heavy)) < 0.05

    # alpha-stable-like: a heavier scale still, tail index must drop further
    heavier = circular * np.sqrt(1.0 / rng.gamma(1.0, 1.0, size=(n, 1)))
    assert np.mean(hill(heavier)) < np.mean(hill(heavy))

    # the copula gap is invariant to the marginals, since it is on normal scores
    assert abs(np.mean(copula_gap(np.exp(circular.real))) - np.mean(copula_gap(circular.real))) < 0.02

    # the null column tracks the sample size, which is the whole reason it exists
    assert _null(200, f, rng)["rho"] > 3 * _null(4000, f, rng)["rho"]
    print("demo ok")


def main() -> None:
    rng = np.random.default_rng(SEED)
    train, _ = cs._load_musdb() if cs.CORPUS == "musdb" else cs._load_stems()
    records = []
    for index, loaders in enumerate(train):
        spectra = cs._training_frames(loaders, np.random.default_rng(cs.SEED))
        row = {"source": index, "scope": "marginal", **marginal(spectra, rng)}
        records.append(row)
        print(json.dumps(row), flush=True)
        try:
            labels = assignment(spectra, index, K_DIAG)
        except FileNotFoundError as err:
            print(f"# {err}", file=sys.stderr)
            continue
        row = {
            "source": index,
            "scope": f"gmm K={K_DIAG}",
            **conditional(spectra, labels, rng),
        }
        records.append(row)
        print(json.dumps(row), flush=True)
    print(render(records), file=sys.stderr)


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        demo()
    else:
        demo()
        main()
