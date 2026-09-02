"""Is the WOLA round trip of `evaluation.py` an orthogonal projection?

Section 2.4 of the article claims the measured ceiling sits above the analytic
one because overlap-add projects a masked spectrogram back onto the consistent
spectrograms, and the reference being consistent, a projection towards it can
only remove residual. The draft hedged that claim, saying the projection is
oblique for periodic Hann at hop L/2 and the argument therefore a heuristic.
This file settles it, and the hedge was wrong in cause though right to exist:

  * the projection is *orthogonal*. `synthesize` normalizes by the sum of
    squared analysis windows, which is the canonical dual window of a painless
    Gabor frame (window support <= number of channels), hence the least-squares
    inverse; A S is then self-adjoint. It looks oblique only in the unweighted
    rfft convention `spectral_ceiling` counts in, where an interior bin stands
    for a conjugate pair and carries twice the energy of DC or Nyquist.
  * S (I - A S) = 0 exactly, which needs nothing but S A = I: resynthesis
    annihilates the inconsistent part of any coefficient error.
  * the frame is *not* tight at hop L/2, sum_k w^2 oscillating in [0.5, 1], so
    transporting the coefficient inequality to a ratio of dB is not automatic.
    Measured on random masks it never fails once the signal is long enough that
    the trimmed edges do not dominate. On ten-frame signals it fails in about a
    tenth of draws by up to 0.6 dB, and that is a trim artifact, not geometry:
    the analytic sum runs over every frame while the measurement drops TRIM
    samples at each end.

Run: uv run python experiments/ola_projection_check.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from experiments.evaluation import analyze, synthesize, window

FloatArray = NDArray[np.float64]

LENGTH = 64
HOP = LENGTH // 2
BINS = LENGTH // 2 + 1
TRIM = LENGTH  # what every caller of `sdr` passes: one full frame at each end


def _pack(spectra: NDArray[np.complex128]) -> FloatArray:
    return np.concatenate([spectra.real.ravel(), spectra.imag.ravel()])


def _unpack(vector: FloatArray, frames: int) -> NDArray[np.complex128]:
    half = vector.size // 2
    return vector[:half].reshape(frames, BINS) + 1j * vector[half:].reshape(frames, BINS)


def _parseval(frames: int) -> FloatArray:
    """Energy weight of each rfft bin, packed as `_pack` packs.

    An interior bin stands for a conjugate pair. Without these weights the
    coefficient inner product is not the signal one and the round trip only
    looks oblique.
    """
    bins = np.full(BINS, 2.0)
    bins[0] = 1.0
    if LENGTH % 2 == 0:
        bins[-1] = 1.0
    tiled = np.tile(bins, (frames, 1)).ravel() / LENGTH
    return np.concatenate([tiled, tiled])


def _samples(frames: int) -> int:
    """Signal length that `analyze` turns into exactly `frames` frames."""
    length = (frames - 1) * HOP + LENGTH
    got = analyze(np.zeros(length), LENGTH, HOP).shape[0]
    assert got == frames, f"analyze framed {length} samples into {got} frames, not {frames}"
    return length


def _operators(frames: int) -> tuple[FloatArray, FloatArray]:
    samples = _samples(frames)
    identity = np.eye(samples)
    analysis = np.stack(
        [_pack(analyze(identity[n], LENGTH, HOP)) for n in range(samples)], axis=1
    )
    coefficients = np.eye(2 * frames * BINS)
    synthesis = np.stack(
        [synthesize(_unpack(coefficients[j], frames), LENGTH, HOP) for j in range(coefficients.shape[0])],
        axis=1,
    )
    return analysis, synthesis


def _sdr_gap(frames: int, trials: int, seed: int) -> tuple[int, float]:
    """Draws masks and returns how often the measured SDR falls below the
    analytic one, and the worst gap in dB. Masks span the class: the optimum,
    the IRM, and arbitrary real gains."""
    samples = _samples(frames)
    rng = np.random.default_rng(seed)
    weight = _parseval(frames)[: frames * BINS].reshape(frames, BINS)
    flips, worst = 0, np.inf
    for _ in range(trials):
        source = analyze(rng.normal(size=samples), LENGTH, HOP)
        other = analyze(rng.normal(size=samples) * rng.uniform(0.2, 2.0), LENGTH, HOP)
        mixture = source + other
        kind = rng.integers(0, 3)
        if kind == 0:
            mask = np.real(source * np.conj(mixture)) / np.maximum(np.abs(mixture) ** 2, 1e-24)
        elif kind == 1:
            mask = np.abs(source) / np.maximum(np.abs(source) + np.abs(other), 1e-24)
        else:
            mask = rng.normal(size=mixture.shape)
        estimate = mask * mixture
        analytic = 10.0 * np.log10(
            float(np.sum(weight * np.abs(source) ** 2))
            / float(np.sum(weight * np.abs(source - estimate) ** 2))
        )
        reference = synthesize(source, LENGTH, HOP)
        error = synthesize(estimate, LENGTH, HOP) - reference
        inner = slice(TRIM, samples - TRIM)
        measured = 10.0 * np.log10(
            float(np.sum(reference[inner] ** 2)) / float(np.sum(error[inner] ** 2))
        )
        gap = measured - analytic
        flips += gap < 0
        worst = min(worst, gap)
    return flips, worst


def demo() -> None:
    frames = 10
    samples = _samples(frames)
    taper = window(LENGTH) ** 2
    squares = np.zeros(samples)
    for i in range(frames):
        squares[i * HOP : i * HOP + LENGTH] += taper
    inner = squares[TRIM : samples - TRIM]
    assert np.ptp(inner) > 0.1, "periodic Hann at hop L/2 is not a tight frame"

    analysis, synthesis = _operators(frames)
    projection = analysis @ synthesis
    assert np.abs(projection @ projection - projection).max() < 1e-10, "A S is not idempotent"

    scaling = np.diag(_parseval(frames))
    adjoint = np.abs(scaling @ projection - projection.T @ scaling).max()
    assert adjoint < 1e-12, f"A S is not self-adjoint under Parseval: {adjoint:.2e}"
    naive = np.abs(projection - projection.T).max()
    assert naive > 0.1, "the unweighted rfft convention was expected to hide it"

    annihilated = synthesis @ (np.eye(projection.shape[0]) - projection)
    assert np.abs(annihilated).max() < 1e-10, "synthesis does not annihilate inconsistency"

    short_flips, short_worst = _sdr_gap(frames=10, trials=400, seed=0)
    long_flips, long_worst = _sdr_gap(frames=200, trials=60, seed=0)
    assert short_flips > 0, "the trim artifact was expected on ten-frame signals"
    assert long_flips == 0, f"measured fell below analytic {long_flips} times on long signals"

    print(
        "self-check: the round trip is an orthogonal projection under Parseval "
        f"(adjoint {adjoint:.1e}, naive {naive:.2f}), synthesis kills the "
        f"inconsistent error, and measured >= analytic on long signals "
        f"(worst {long_worst:+.2f} dB) while ten-frame signals flip "
        f"{short_flips}/400 times (worst {short_worst:+.2f} dB)"
    )


if __name__ == "__main__":
    demo()
