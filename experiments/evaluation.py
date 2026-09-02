"""Time-domain reconstruction and the exact ceiling of the real-mask class.

Every mask-based separator writes its estimate as ``s_hat = m * x`` with ``m``
real and non-negative per bin, which forces the mixture's phase onto the
estimate. Minimizing ``|s - m x|^2`` over real ``m`` has a closed form,

    m*[f] = Re(s[f] conj(x[f])) / |x[f]|^2 ,

with residual ``|s|^2 sin^2(angle(s, x))``. No mask, and no method that
estimates one, goes below that residual: it is the class ceiling. The IRM and
the oracle Wiener filter are two particular masks, and on the material measured
on 2026-08-19 they sit ~6 dB below ``m*``. Comparing a phase-aware estimator to
the IRM alone therefore overstates its margin by that much, which is why every
number this module produces is reported against ``m*``.

``m*`` is signed and unbounded, so the two restrictions of the class are
reported next to it, ``positive`` (its projection on [0, +inf)) and ``clipped``
(its projection on [0, 1]): bounded positive masks are a strictly smaller class,
and the gap between the rows says how much of the ceiling is out of reach for a
sigmoid-headed network before any phase argument is made. Each of the three has
its own ceiling in closed form, and `spectral_ceiling` takes the interval as an
argument to return it: clipping a scalar convex quadratic costs one penalty term
per clipped bin, and the article quotes the three ceilings rather than the
unconstrained one alone, since a comparison against one of them bounds no other.

The reconstruction path is here rather than in ``gasm`` because ``gasm`` frames
signals but never inverts them. Two facts about the inverse make or break every
SDR below. The normalization is the sum of squared analysis windows, which
decays to zero over the first and last ``frame_length`` samples; a *modified*
spectrogram divided by it explodes there, while an unmodified one cancels
exactly. A perfect-reconstruction test therefore passes at 100+ dB while every
oracle reads negative, which is exactly the failure diagnosed on 2026-08-19
(oracle IRM at -2.81 dB, below the untreated mixture). Hence ``TRIM``: one full
frame is discarded at each end, inside the metrics, for every signal alike.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gasm.common.power_spectrum import complex_spectrum, frame_signal

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_EPS = 1e-24
ORACLES = ("mixture", "irm", "wiener", "best_real", "positive", "clipped")
# The admissible interval of each subclass of the real-mask class, in the notation
# of the article: M_1 unconstrained, M_+ non-negative, M_[0,1] bounded. The key is
# the oracle that realises the class, so one dict serves the masks and the ceilings.
CLASS_BOUNDS: dict[str, tuple[float, float] | None] = {
    "best_real": None,
    "positive": (0.0, np.inf),
    "clipped": (0.0, 1.0),
}


def window(frame_length: int) -> FloatArray:
    """Periodic Hann, not the symmetric one `gasm.frame_signal` defaults to.

    At hop = L/2 the periodic window satisfies COLA exactly while the symmetric
    one does not, and the difference is not cosmetic here: the GMM is fitted on
    the windowed frames, so the window is part of the feature. Refitting the
    2026-08-19 control with the symmetric window moved DM-GMM by 1.8 dB at
    K = 32 while leaving every oracle within 0.03 dB, which says more about how
    sensitive the prior is than about the window.
    """
    return np.asarray(np.hanning(frame_length + 1)[:frame_length], dtype=np.float64)


def analyze(signal: FloatArray, frame_length: int, shift_length: int) -> ComplexArray:
    """STFT via `gasm.frame_signal`, with no bin truncation so it can be inverted."""
    frames = frame_signal(signal, frame_length, shift_length, window(frame_length))
    return np.asarray(complex_spectrum(frames, frame_length // 2 + 1), dtype=np.complex128)


def synthesize(spectra: ComplexArray, frame_length: int, shift_length: int) -> FloatArray:
    """Weighted overlap-add inverse of `analyze`, normalized by the squared windows.

    The synthesis window is the analysis window again (WOLA), so the estimate is
    tapered before it is added back; the sum-of-squares normalization then makes
    the unmodified round trip exact wherever that sum is non-zero. Near the edges
    it is not, and the caller must drop `TRIM` samples there.
    """
    taper = window(frame_length)
    frames = np.fft.irfft(spectra, n=frame_length, axis=-1) * taper
    length = (len(frames) - 1) * shift_length + frame_length
    signal = np.zeros(length)
    weight = np.zeros(length)
    squared = taper**2
    for i, frame in enumerate(frames):
        start = i * shift_length
        signal[start : start + frame_length] += frame
        weight[start : start + frame_length] += squared
    return signal / np.maximum(weight, _EPS)


def _align(reference: FloatArray, estimate: FloatArray, trim: int) -> tuple[FloatArray, FloatArray]:
    n = min(len(reference), len(estimate))
    return reference[trim : n - trim], estimate[trim : n - trim]


def sdr(reference: FloatArray, estimate: FloatArray, trim: int) -> float:
    """Time-domain SDR in dB, edges dropped. No scale invariance: an estimator
    that gets the gain wrong is penalized here, which is what we want when the
    claim is about phase and amplitude jointly."""
    ref, est = _align(reference, estimate, trim)
    error = float(np.sum((ref - est) ** 2))
    return float("inf") if error == 0.0 else 10.0 * np.log10(float(np.sum(ref**2)) / error)


def si_sdr(reference: FloatArray, estimate: FloatArray, trim: int) -> float:
    """Scale-invariant SDR (Le Roux 2019), reported alongside `sdr` so a reviewer
    can tell a genuine improvement from a gain calibration difference."""
    ref, est = _align(reference, estimate, trim)
    target = ref * (float(np.dot(est, ref)) / max(float(np.dot(ref, ref)), _EPS))
    noise = float(np.sum((est - target) ** 2))
    return float("inf") if noise == 0.0 else 10.0 * np.log10(float(np.sum(target**2)) / noise)


def oracle_spectra(sources: list[ComplexArray], index: int) -> dict[str, ComplexArray]:
    """The five reference estimates of source `index`, all masks on the mixture.

    `irm` is the magnitude ratio, `wiener` the power ratio, `best_real` the exact
    minimizer m*, `clipped` its projection on [0, 1], `mixture` the do-nothing
    floor. The masks are built from the true sources: these are oracles, not
    methods, and they bound what any method of the class can reach.
    """
    mixture = sum(sources)
    target = sources[index]
    denominator = np.maximum(np.abs(mixture) ** 2, _EPS)
    best = np.real(target * np.conj(mixture)) / denominator
    masks = {
        "mixture": np.ones_like(best),
        "irm": np.abs(target) / np.maximum(sum(np.abs(s) for s in sources), _EPS),
        "wiener": np.abs(target) ** 2 / np.maximum(sum(np.abs(s) ** 2 for s in sources), _EPS),
        "best_real": best,
        "positive": np.clip(best, 0.0, np.inf),
        "clipped": np.clip(best, 0.0, 1.0),
    }
    return {name: mask * mixture for name, mask in masks.items()}


def spectral_ceiling(
    sources: list[ComplexArray], index: int, bounds: tuple[float, float] | None = None
) -> float:
    """Analytic ceiling of the mask class, in dB, from the residual
    ``|s|^2 sin^2(angle(s, x))`` summed over bins.

    With `bounds` set to the admissible interval of a subclass (see CLASS_BOUNDS)
    the ceiling is that subclass'. The criterion is a scalar convex quadratic in
    ``m`` at fixed bin, ``|s - m x|^2 = |s|^2 sin^2(theta) + |x|^2 (m - m*)^2``,
    so the constrained optimum is the projection of ``m*`` on the interval and the
    residual picks up one penalty term per clipped bin. Two of those terms have a
    reading of their own. Where ``m* < 0``, that is where the phase disagreement
    exceeds pi/2, the best non-negative mask is zero and the bin is a total loss,
    the residual becoming ``|s|^2``. Where ``m* > 1``, that is where the two
    sources partially cancel, the best bounded mask is one and the residual
    becomes ``|x - s|^2``, the energy of the *other* source, exactly.

    Computed without ever forming a mask, so it cross-checks `oracle_spectra`
    and the whole reconstruction path at once: if the measured SDR of
    `best_real` falls *below* this, the STFT round trip is what is broken, not
    the estimator under test.

    It is a spectral-domain bound and the time-domain SDR of `best_real` sits
    slightly above it, by 0.2 dB on the 2026-08-19 material and 1.3 dB on the
    tones of `demo`. That gap is structural, not noise: a per-frame masked
    spectrogram is not consistent, overlap-add projects it back onto the
    spectrograms that are, and the reference being consistent the projection can
    only remove residual. The article must therefore quote the *measured*
    `best_real` as the ceiling and keep this number as the cross-check it is,
    because quoting the lower analytic value understates what a mask can do.
    """
    mixture = sum(sources)
    target = sources[index]
    power = float(np.sum(np.abs(target) ** 2))
    energy = np.maximum(np.abs(mixture) ** 2, _EPS)
    aligned = np.real(target * np.conj(mixture)) ** 2 / energy
    residual = power - float(np.sum(aligned))
    if bounds is not None:
        best = np.real(target * np.conj(mixture)) / energy
        residual += float(np.sum(energy * (best - np.clip(best, *bounds)) ** 2))
    return float("inf") if residual <= 0.0 else 10.0 * np.log10(power / residual)


def cascade_residuals(sources: list[ComplexArray], index: int) -> dict[str, float]:
    """The four residuals of the operator chain M1 c M2 c M3 c M4, in dB.

    Two different questions are called "the ceiling of a class" and they have
    different answers. Per frame, with ``(s, x)`` both known, the real mask leaves
    ``|s|^2 sin^2(theta)`` while a *complex* gain leaves exactly zero, ``g = s/x``
    being admissible; the chain is degenerate from its second rung and the whole
    mask-format residual is charged to one missing real parameter per bin, the
    phase. What a *method* faces is the other question: one operator fixed across
    frames, ``min E|s - A x|^2``. There the four classes are four nested real
    subspaces of one Hilbert space, of real dimension 1, 2, 4 and 4F per bin (the
    "parameters per bin" column of the article's table), the four minima are four
    orthogonal projections of the same target, and Pythagoras makes the residuals
    telescope: each gap is the energy of one structure the smaller class cannot
    represent.

    Everything below is in closed form from four second-order statistics per bin,
    ``p = E|x|^2``, ``c = E[s conj(x)]``, ``ctilde = E[s x]`` and ``q = E[x^2]``.
    M1 (real gain) leaves ``E|s|^2 - Re(c)^2/p``, M2 (complex gain) leaves
    ``E|s|^2 - |c|^2/p``, so the first gap is ``Im(c)^2/p``: quadrature
    correlation, that is *statistical* phase information. M3 (widely linear,
    ``g x + h conj(x)``) solves a 2x2 Hermitian system and its gap to M2 is
    ``|ctilde - rho c|^2 / (p (1 - |rho|^2))`` with ``rho = q/p``, non-circularity.
    M4 projects on all bins at once through the augmented covariance, and its gap
    to M3 is inter-frequency coupling.

    The prediction worth testing is that the first gap vanishes on real material:
    for zero-mean uncorrelated sources ``c = E|s|^2`` is real, so M1 and M2 leave
    the same residual. The phase would then buy *everything* per instance and
    *nothing* statistically, which is precisely why an estimator of this kind needs
    a prior rather than a linear fit, and why the ceiling of M1 is the landmark to
    quote.

    Fitted in sample, so the higher rungs are optimistic: ``2T`` real observations
    per bin against 1, 2, 4 and ``2 * rank`` real parameters. The first three are
    negligible at any usable frame count; M4 is not, and the corrected value
    ``r4_corrected`` (residual scaled by ``2T / (2T - 2 rank)``) is reported beside
    the raw one so a reader can see the size of the correction rather than trust it.
    """
    mixture = sum(sources)
    target = sources[index]
    frames = int(target.shape[0])
    power = float(np.sum(np.abs(target) ** 2))
    if power <= 0.0:
        return {}
    energy = np.sum(np.abs(target) ** 2, axis=0)
    p = np.maximum(np.sum(np.abs(mixture) ** 2, axis=0), _EPS)
    c = np.sum(target * np.conj(mixture), axis=0)
    ctilde = np.sum(target * mixture, axis=0)
    q = np.sum(mixture * mixture, axis=0)

    r1 = float(np.sum(energy - np.real(c) ** 2 / p))
    r2 = float(np.sum(energy - np.abs(c) ** 2 / p))

    # At the DC and Nyquist bins the frames are real, conj(x) = x, and M3 collapses
    # onto M2: the 2x2 system is singular there and its solution meaningless.
    determinant = p**2 - np.abs(q) ** 2
    degenerate = determinant <= 1e-12 * p**2
    safe = np.where(degenerate, 1.0, determinant)
    gain = (p * c - np.conj(q) * ctilde) / safe
    conjugate_gain = (p * ctilde - q * c) / safe
    explained = np.real(np.conj(gain) * c + np.conj(conjugate_gain) * ctilde)
    r3 = float(np.sum(energy - np.where(degenerate, np.abs(c) ** 2 / p, explained)))

    # One Gram for every output bin: the augmented covariance does not depend on f,
    # only the cross-term does, so this is a single factorization. lstsq rather than
    # solve because the duplicated real bins make the Gram rank deficient by design.
    augmented = np.concatenate([mixture, np.conj(mixture)], axis=1).T
    gram = augmented @ augmented.conj().T
    cross = augmented @ np.conj(target)
    weights, _, rank, _ = np.linalg.lstsq(gram, cross, rcond=None)
    r4 = power - float(np.real(np.sum(np.conj(cross) * weights)))

    observations = 2 * frames
    corrected = r4 * observations / max(observations - 2 * int(rank), 1)
    return {
        "cascade_frames": float(frames),
        "cascade_rank": float(rank),
        **{
            f"cascade_{name}": (
                float("inf") if residual <= 0.0 else 10.0 * np.log10(power / residual)
            )
            for name, residual in (
                ("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4), ("r4_corrected", corrected)
            )
        },
    }


def class_witnesses(estimate: ComplexArray, mixture: ComplexArray, floor: float) -> dict[str, float]:
    """Evidence that an estimate left the real-mask class, bin by bin.

    Writing the estimate as ``g = s_hat / x``, a real mask means ``g`` real and
    in [0, 1]. Three departures are counted on the bins carrying real energy
    (mixture magnitude above `floor`, otherwise the ratio is numerical noise):
    the median absolute phase rotation, the share of bins whose gain exceeds one,
    and the share whose gain is negative. A separator that reads zero on all
    three *is* a mask, whatever its derivation, and cannot beat the ceiling.
    """
    active = np.abs(mixture) > floor
    if not active.any():
        return {"phase_median_deg": 0.0, "gain_above_one": 0.0, "gain_negative": 0.0}
    gain = estimate[active] / mixture[active]
    return {
        "phase_median_deg": float(np.degrees(np.median(np.abs(np.angle(gain))))),
        "gain_above_one": float(np.mean(np.abs(gain) > 1.0)),
        "gain_negative": float(np.mean(np.real(gain) < 0.0)),
    }


def _check_cascade() -> None:
    """Each gap of `cascade_residuals` opened by the one structure it names.

    Synthetic spectra rather than audio, because the point is that the three gaps
    are not interchangeable: switching on quadrature correlation, non-circularity
    or inter-frequency coupling must move one gap and leave the others shut. The
    first case doubles as the article's claim, uncorrelated circular sources
    leaving M1 and M2 at the same residual, and the third has a closed form worth
    checking by hand: with an interferer shared by two bins, the projection on both
    bins leaves 1/3 of the target energy against 1/2 for the per-bin complex gain,
    that is 1.76 dB.
    """
    rng = np.random.default_rng(1)
    frames, bins = 20000, 4

    def circular(scale: float, shape: tuple[int, int]) -> ComplexArray:
        draw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return np.asarray(scale * draw / np.sqrt(2), dtype=np.complex128)

    shape = (frames, bins)
    plain = cascade_residuals([circular(1.0, shape), circular(1.0, shape)], 0)
    steps = [plain[f"cascade_r{i}"] for i in (1, 2, 3, 4)]
    assert max(steps) - min(steps) < 0.05, f"circular independent sources must not separate: {steps}"

    # non-circular: variance split unevenly between the two quadratures, and split
    # the other way on the interferer so that E[x^2] stays zero and only the 2->3
    # gap can open
    def elliptic(real: float, imaginary: float) -> ComplexArray:
        return real * rng.normal(size=shape) + 1j * imaginary * rng.normal(size=shape)

    skewed = cascade_residuals([elliptic(1.0, 0.5), elliptic(0.5, 1.0)], 0)
    assert skewed["cascade_r2"] - skewed["cascade_r1"] < 0.05, skewed
    assert skewed["cascade_r3"] - skewed["cascade_r2"] > 1.0, skewed

    # inter-frequency coupling: one interferer realization shared by two bins, so
    # the second bin says something about the first that no per-bin operator sees
    target = circular(1.0, (frames, 2))
    shared = circular(1.0, (frames, 1))
    coupled = cascade_residuals([target, np.repeat(shared, 2, axis=1)], 0)
    assert coupled["cascade_r3"] - coupled["cascade_r2"] < 0.05, coupled
    assert abs(coupled["cascade_r4"] - coupled["cascade_r3"] - 1.76) < 0.1, coupled
    print(
        "self-check cascade: circular flat, non-circularity opens 2->3 by "
        f"{skewed['cascade_r3'] - skewed['cascade_r2']:.2f} dB, coupling opens 3->4 by "
        f"{coupled['cascade_r4'] - coupled['cascade_r3']:.2f} dB"
    )


def demo() -> None:
    """The reconstruction and the ceiling check each other; run before any sweep."""
    rng = np.random.default_rng(0)
    frame_length, shift_length = 256, 128
    trim = frame_length
    time = np.arange(20000) / 16000.0
    # two tonal sources, so their bins genuinely overlap and the phase term bites
    signals = [
        np.sin(2 * np.pi * 440 * time) * (1 + 0.5 * np.sin(2 * np.pi * 3 * time)),
        np.sin(2 * np.pi * 660 * time + 1.1) + 0.3 * rng.normal(size=len(time)),
    ]
    spectra = [analyze(s, frame_length, shift_length) for s in signals]
    mixture = synthesize(spectra[0] + spectra[1], frame_length, shift_length)

    round_trip = sdr(signals[0], synthesize(spectra[0], frame_length, shift_length), trim)
    assert round_trip > 100.0, f"WOLA round trip only {round_trip:.1f} dB"
    # the edges are the trap: same round trip, no trim, must be visibly worse
    assert sdr(signals[0], synthesize(spectra[0], frame_length, shift_length), 0) < round_trip

    for index in (0, 1):
        estimates = oracle_spectra(spectra, index)
        scores = {
            name: sdr(signals[index], synthesize(spec, frame_length, shift_length), trim)
            for name, spec in estimates.items()
        }
        ceilings = {
            name: spectral_ceiling(spectra, index, bounds)
            for name, bounds in CLASS_BOUNDS.items()
        }
        ceiling = ceilings["best_real"]
        assert scores["best_real"] > scores["wiener"] > scores["mixture"]
        assert scores["best_real"] > scores["irm"]
        assert scores["best_real"] >= scores["positive"] >= scores["clipped"]
        # the nesting M_[0,1] c M_+ c M_1 must show up on both sides, analytic
        # ceilings and measured scores, or the clipping is wired to the wrong class
        assert ceiling >= ceilings["positive"] >= ceilings["clipped"], ceilings
        # one-sided: consistency projection can only help, so the analytic value
        # is a floor for the measured one, and a broken round trip breaks that
        for name, value in ceilings.items():
            assert value - 0.2 < scores[name] < value + 3.0, (
                f"{name}: measured {scores[name]:.2f} dB vs analytic {value:.2f} dB"
            )
        # the penalty term is not a bound but an identity: where m* > 1 the best
        # bounded mask is 1 and the residual is exactly the other source's energy
        mixture_spec = spectra[0] + spectra[1]
        target_spec = spectra[index]
        energy = np.maximum(np.abs(mixture_spec) ** 2, _EPS)
        best = np.real(target_spec * np.conj(mixture_spec)) / energy
        above = best > 1.0
        assert above.any(), "no bin cancels, the identity below would be vacuous"
        aligned = np.real(target_spec * np.conj(mixture_spec)) ** 2 / energy
        decomposed = (np.abs(target_spec) ** 2 - aligned + energy * (best - 1.0) ** 2)[above]
        other = np.abs(mixture_spec - target_spec)[above] ** 2
        assert np.allclose(decomposed, other, rtol=1e-8, atol=1e-12 * other.max())
        floor = 1e-3 * float(np.abs(spectra[0] + spectra[1]).max())
        witnesses = class_witnesses(estimates["clipped"], spectra[0] + spectra[1], floor)
        assert witnesses["phase_median_deg"] < 1e-9, "a mask cannot rotate phase"
        assert witnesses["gain_above_one"] == 0.0
        print(
            f"source {index}: "
            + "  ".join(f"{name} {scores[name]:6.2f}" for name in ORACLES)
            + "  | analytic "
            + "  ".join(f"{name} {value:6.2f}" for name, value in ceilings.items())
        )
        cascade = cascade_residuals(spectra, index)
        rungs = [cascade[f"cascade_r{i}"] for i in (1, 2, 3, 4)]
        # nested subspaces, so the projections are ordered whatever the material
        assert all(b >= a - 1e-9 for a, b in zip(rungs, rungs[1:])), cascade
        print("  cascade " + "  ".join(f"{value:6.2f}" for value in rungs))
    print("self-check: reconstruction exact, m* above IRM and Wiener, masks phase-blind")
    _check_cascade()


if __name__ == "__main__":
    demo()
