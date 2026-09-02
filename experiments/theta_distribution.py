"""What the ceiling of the real-gain class is made of, bin by bin.

Proposition "ceiling" says the residual of the best real gain is
``|s|^2 sin^2(theta)`` summed over bins, theta being the angle between the
source and the mixture in that bin. The ceiling is therefore an energy-weighted
average of ``sin^2(theta)`` and nothing else, and a reader is entitled to ask
what that distribution looks like on real music rather than to take the summary
number on trust. This script measures it.

Oracle-only: no prior, no fit, no estimator. It reads the same excerpts as
`ceiling_sweep.py` through the same loaders, so a row here and a row there
describe the same audio, and it is cheap enough to run on the laptop.

    MUSDB_ROOT=D:/data/gasm-demos/musdb18 NFFT=256 HOP=128 REGIME=unseen \
    N_TRAIN=5 N_TEST=5 TEST_SECONDS=30 python experiments/theta_distribution.py

Every environment variable is `ceiling_sweep`'s, read from that module rather
than re-declared, so the two cannot drift apart. One JSON object per line:
energy-weighted quantiles of |theta| in degrees, the share of source energy
sitting past a few angles, and the ceiling recomputed from the distribution as
a cross-check against `spectral_ceiling`.
"""

from __future__ import annotations

import json

import numpy as np

import ceiling_sweep as cs
from evaluation import analyze, spectral_ceiling

# The energy weights make the median a weighted one, which numpy does not have.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
DEGREES = (5.0, 15.0, 30.0, 45.0, 60.0)


def _weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return [float(np.interp(q, cumulative, values)) for q in quantiles]


def main() -> None:
    _, test_items = cs._load_stems() if cs.CORPUS == "stems" else cs._load_musdb()
    config = {
        "corpus": cs.CORPUS, "regime": cs.REGIME, "nfft": cs.NFFT, "hop": cs.HOP,
        "n_test": cs.N_TEST, "test_seconds": cs.TEST_SECONDS,
    }
    for name, references in test_items:
        spectra = [analyze(reference, cs.NFFT, cs.HOP) for reference in references]
        mixture = np.sum(spectra, axis=0)
        for index, target in enumerate(spectra):
            energy = np.abs(target) ** 2
            # cos^2 of the angle between s and x, clipped because a bin where both
            # are numerically zero is 0/0 and carries no weight anyway
            denominator = np.maximum(energy * np.abs(mixture) ** 2, 1e-30)
            cosine2 = np.clip(np.real(target * np.conj(mixture)) ** 2 / denominator, 0.0, 1.0)
            sine2 = 1.0 - cosine2
            theta = np.degrees(np.arcsin(np.sqrt(sine2)))

            weights = energy.ravel()
            active = weights > 0.0
            weights, flat_theta, flat_sine2 = weights[active], theta.ravel()[active], sine2.ravel()[active]
            share = float(np.sum(weights * flat_sine2) / np.sum(weights))
            row = {
                **config, "track": name, "source": index,
                "bins": int(active.sum()),
                "sin2_energy_weighted": share,
                # the same number `spectral_ceiling` returns, recomputed from the
                # angles alone: they must agree, or the identity is misread
                "ceiling_from_theta": float(-10.0 * np.log10(share)),
                "spectral_ceiling": spectral_ceiling(spectra, index),
                "theta_deg_quantiles": dict(
                    zip((str(q) for q in QUANTILES),
                        _weighted_quantiles(flat_theta, weights, QUANTILES))
                ),
                "energy_share_above_deg": {
                    str(d): float(np.sum(weights[flat_theta > d]) / np.sum(weights))
                    for d in DEGREES
                },
            }
            print(json.dumps(row), flush=True)


def demo() -> None:
    """Two synthetic bins of known angle, so the weighting is checked, not assumed."""
    values = np.array([0.0, 60.0])
    weights = np.array([3.0, 1.0])
    assert _weighted_quantiles(values, weights, (0.5,))[0] == 0.0
    assert abs(_weighted_quantiles(values, weights, (0.9,))[0] - 36.0) < 1e-9
    print("theta_distribution demo ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
