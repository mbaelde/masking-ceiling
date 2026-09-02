"""Why is the spectral deficit 11 dB where the pilot measured 6.6 dB in time?

    uv run python experiments/calibration_cross_check.py

`prior_calibration` reports a deficit built from the closed-form excess of the
posterior mean, and its docstring makes the pilot's time-domain deficit the
validation: the two should land near each other. They do not, by four and a half
dB, and only two things can explain that. Either the closed form is wrong, or the
overlap-add inverse discards a large part of the estimator's coefficient error,
which section 2.4 says it is entitled to do since ``S (I - A S) = 0``.

The two are separable, on one track and one K, by an identity rather than an
argument. Per bin, split both the estimate and the truth on the mixture:

    s_hat = m_hat x + q,    s = m* x + r,    q, r orthogonal to x

so the whole spectral error decomposes with no cross term,

    sum |s_hat - s|^2 = sum |x|^2 (m_hat - m*)^2  +  sum |q - r|^2

whose first term is exactly what the closed form computes. Reproducing the left
side from the closed form plus a directly measured second term validates the
closed form to floating point. Then whatever remains between that spectral total
and the measured time-domain error is the projection, and nothing else.

Environment: pilot_support's, since the checkpoints are keyed by it. One track
and one K by default, because this answers a yes-or-no question and the estimate
costs a full pass over the component pairs.
"""

from __future__ import annotations

import os

import numpy as np

import ceiling_sweep as cs
import prior_calibration as pcal
from estimator_readout import _load_source
from evaluation import _EPS, analyze, oracle_spectra, sdr, synthesize
from gasm.rase.dmgmm import _regress, _stack_real_imag, _unstack_real_imag

TRACKS = int(os.environ.get("CROSS_TRACKS", "2"))


def main() -> None:
    n_components = cs.K_VALUES[-1]
    _, test_items = cs._load_stems() if cs.CORPUS == "stems" else cs._load_musdb()
    models = [_load_source(i, n_components) for i in range(2)]
    diag = [pcal._load_diag(i, n_components) for i in range(2)]

    for name, references in test_items[:TRACKS]:
        spectra = [analyze(reference, cs.NFFT, cs.HOP) for reference in references]
        mixture = np.sum(spectra, axis=0)
        stacked = _stack_real_imag(mixture)
        estimate = _unstack_real_imag(_regress(models[0], models[1], stacked))
        post = pcal.gain_posterior(diag[0], diag[1], stacked)
        power = np.maximum(np.abs(mixture) ** 2, _EPS)

        for index, truth in enumerate(spectra):
            estimated = estimate if index == 0 else mixture - estimate
            gain_hat = np.real(estimated * np.conj(mixture)) / power
            gain_star = np.real(truth * np.conj(mixture)) / power
            # the closed form never sees the estimate, only the models, so this
            # equality is the whole point of the file
            closed = post["mean"] if index == 0 else 1.0 - post["mean"]
            # the gain is unbounded where the mixture vanishes, so relative
            assert np.allclose(gain_hat, closed, rtol=1e-4, atol=1e-6), "closed form is not the estimate"

            in_class = float(np.sum(power * (gain_hat - gain_star) ** 2))
            ortho = float(np.sum(np.abs((estimated - gain_hat * mixture) - (truth - gain_star * mixture)) ** 2))
            total = float(np.sum(np.abs(estimated - truth) ** 2))
            residual = float(np.sum(np.abs(truth) ** 2) - np.sum(power * gain_star**2))
            assert abs(in_class + ortho - total) < 1e-6 * total, "the split is not orthogonal"

            ceiling = sdr(
                references[index],
                synthesize(oracle_spectra(spectra, index)["best_real"], cs.NFFT, cs.HOP),
                cs.NFFT,
            )
            measured = sdr(
                references[index], synthesize(estimated, cs.NFFT, cs.HOP), cs.NFFT
            )
            print(
                f"{name[:26]:<26} src {index} K={n_components}  "
                f"in_class {10 * np.log10(in_class / residual):+6.2f} dB of R*  "
                f"ortho {10 * np.log10(ortho / residual):+6.2f}  "
                f"deficit spectral {10 * np.log10(total / residual):6.2f}  "
                f"time {ceiling - measured:6.2f}  "
                f"projection removes {10 * np.log10(total / residual) - (ceiling - measured):5.2f} dB",
                flush=True,
            )


if __name__ == "__main__":
    main()
